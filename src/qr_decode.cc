#include "qr_decode.h"

// global variables
std::atomic<bool> image_save_flag(false);
bool use_csi_flag = false;
std::string image_save_path = "";

// private variables
cv::VideoCapture *cap;

// Constants
const cv::Size IMG_SIZE(640, 640);
const float CONF_THRESH = 0.5f;
const float IOU_THRESH = 0.45f;
const int MAX_DET = 100;
const float EXPAND_RATIO = 0.3f;
const float SR_SCALE = 2.0f;

SuperResolution *sr;

// Helper functions
cv::Mat applyCLAHE(const cv::Mat &image) {
  cv::Mat lab, l_clahe, lab_clahe;
  cv::cvtColor(image, lab,
               cv::COLOR_BGR2Lab); // Note: Changed from COLOR_BGR2LAB

  std::vector<cv::Mat> lab_channels;
  cv::split(lab, lab_channels);

  auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
  clahe->apply(lab_channels[0], l_clahe);

  std::vector<cv::Mat> merged_channels = {l_clahe, lab_channels[1],
                                          lab_channels[2]};
  cv::merge(merged_channels, lab_clahe);

  cv::Mat result;
  cv::cvtColor(lab_clahe, result,
               cv::COLOR_Lab2BGR); // Note: Changed from COLOR_LAB2BGR
  return result;
}

cv::Mat applySuperResolution(const cv::Mat &image, float scale = 2.0f) {
  if (image.rows < 100) {
    cv::Mat enlarged;
    cv::resize(image, enlarged, cv::Size(), scale, scale, cv::INTER_CUBIC);

    float kernel_data[] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
    cv::Mat kernel(3, 3, CV_32F, kernel_data);

    cv::Mat sharpened;
    cv::filter2D(enlarged, sharpened, -1, kernel);
    return sharpened;
  }
  return image;
}

std::vector<float> preprocess(const cv::Mat &image, bool enhance = true) {
  cv::Mat processed;
  if (enhance) {
    processed = applyCLAHE(image);
  } else {
    processed = image.clone();
  }

  cv::resize(processed, processed, IMG_SIZE);
  cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

  // Convert to CHW format
  std::vector<cv::Mat> channels;
  cv::split(processed, channels);

  std::vector<float> result;
  result.reserve(3 * IMG_SIZE.area());

  for (const auto &channel : channels) {
    cv::Mat float_channel;
    channel.convertTo(float_channel, CV_32F, 1.0 / 255.0);

    // Flatten the channel
    result.insert(result.end(), float_channel.ptr<float>(),
                  float_channel.ptr<float>() + float_channel.total());
  }

  return result;
}

struct Detection {
  float x1, y1, x2, y2;
  float confidence;
  int class_id;
};

// Simple NMS implementation since cv::dnn::NMSBoxes might not be available
std::vector<Detection> nonMaximumSuppression(std::vector<Detection> &detections,
                                             float iou_threshold) {
  std::vector<Detection> result;

  // Sort by confidence (descending)
  std::sort(detections.begin(), detections.end(),
            [](const Detection &a, const Detection &b) {
              return a.confidence > b.confidence;
            });

  while (!detections.empty()) {
    // Add the detection with highest confidence
    result.push_back(detections[0]);

    // Compute IoU with remaining detections
    std::vector<Detection> remaining;
    for (size_t i = 1; i < detections.size(); ++i) {
      const auto &a = detections[0];
      const auto &b = detections[i];

      // Compute intersection
      float x1 = std::max(a.x1, b.x1);
      float y1 = std::max(a.y1, b.y1);
      float x2 = std::min(a.x2, b.x2);
      float y2 = std::min(a.y2, b.y2);

      float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

      // Compute union
      float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
      float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
      float union_area = area_a + area_b - intersection;

      float iou = intersection / union_area;

      if (iou < iou_threshold) {
        remaining.push_back(b);
      }
    }

    detections = std::move(remaining);
  }

  return result;
}

std::vector<Detection> postprocess(const std::vector<float> &output,
                                   float conf_thresh = CONF_THRESH) {
  std::vector<Detection> detections;
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;

  // Assuming output is shaped [1, 25200, 6]
  for (size_t i = 0; i < 25200; ++i) {
    size_t base_idx = i * 6;
    float x = output[base_idx];
    float y = output[base_idx + 1];
    float w = output[base_idx + 2];
    float h = output[base_idx + 3];
    float obj_conf = output[base_idx + 4];
    float cls_conf = output[base_idx + 5];

    float confidence = obj_conf * cls_conf;
    if (confidence < conf_thresh)
      continue;

    // Convert to OpenCV Rect format
    boxes.emplace_back(cv::Rect(x - w / 2, y - h / 2, w, h));
    scores.push_back(confidence);

    // Store original detection info
    Detection det;
    det.x1 = x - w / 2;
    det.y1 = y - h / 2;
    det.x2 = x + w / 2;
    det.y2 = y + h / 2;
    det.confidence = confidence;
    det.class_id = 0;
    detections.push_back(det);
  }

  // Apply OpenCV's NMS
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, scores, conf_thresh, IOU_THRESH, indices);

  // Create filtered detections
  std::vector<Detection> filtered_detections;
  for (int idx : indices) {
    filtered_detections.push_back(detections[idx]);
  }

  // Print debug info
  // printf("Found %zu detections, after NMS: %zu\n", detections.size(),
  //        filtered_detections.size());

  return filtered_detections;
}

std::pair<cv::Mat, cv::Rect> expandAndCrop(const cv::Mat &image,
                                           const Detection &det,
                                           float expand_ratio = EXPAND_RATIO) {
  // 捕获错误

  try {
    int h = image.rows;
    int w = image.cols;

    // Convert normalized coords to absolute
    float scale_x = (float)w / (float)IMG_SIZE.width;
    float scale_y = (float)h / (float)IMG_SIZE.height;
    int x1_abs = static_cast<int>(det.x1 * scale_x);
    int y1_abs = static_cast<int>(det.y1 * scale_y);
    int x2_abs = static_cast<int>(det.x2 * scale_x);
    int y2_abs = static_cast<int>(det.y2 * scale_y);

    // Dynamic expansion
    float qr_size = std::max(x2_abs - x1_abs, y2_abs - y1_abs);
    float actual_expand = (qr_size < 0.1f * std::min(h, w)) ? 0.4f : 0.2f;

    float dw = (x2_abs - x1_abs) * actual_expand / 2;
    float dh = (y2_abs - y1_abs) * actual_expand / 2;

    x1_abs = std::max(0, static_cast<int>(x1_abs - dw));
    y1_abs = std::max(0, static_cast<int>(y1_abs - dh));
    x2_abs = std::min(w, static_cast<int>(x2_abs + dw));
    y2_abs = std::min(h, static_cast<int>(y2_abs + dh));

    // printf("Expanded ROI: x1=%d, y1=%d, x2=%d, y2=%d\n", x1_abs, y1_abs,
    // x2_abs,
    //        y2_abs);

    cv::Rect roi(x1_abs, y1_abs, x2_abs - x1_abs, y2_abs - y1_abs);

    // 打印 roi 的参数
    // printf("ROI: x=%d, y=%d, width=%d, height=%d\n", roi.x, roi.y, roi.width,
    //        roi.height);

    return {image(roi), roi};
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }

  return {cv::Mat(), cv::Rect()};
}

std::string decodeWithZbar(const cv::Mat &image) {
  cv::Mat sr_image;

  // Apply super resolution
  if (!sr->processImage(image, sr_image)) {
    std::cerr << "Super resolution failed\n";
    return "";
  }

  // Convert to grayscale
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  // // Auto thresholding
  // cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  // Debug print
  cv::imshow("sr_image", gray);

  // Create zbar scanner
  zbar::ImageScanner scanner;
  scanner.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);

  // Wrap image data
  zbar::Image zbar_image(gray.cols, gray.rows, "Y800", gray.data,
                         gray.cols * gray.rows);

  // Scan the image for barcodes
  int n = scanner.scan(zbar_image);

  // Extract results
  if (n > 0) {
    for (zbar::Image::SymbolIterator symbol = zbar_image.symbol_begin();
         symbol != zbar_image.symbol_end(); ++symbol) {
      return symbol->get_data();
    }
  }

  return "";
}

cv::Mat processFrame(cv::Mat &frame, Ort::Session &session,
                     const char *input_name, const char *output_name,
                     const std::vector<int64_t> &input_shape,
                     const std::vector<int64_t> &output_shape) {

  // cv::Mat frame_disp = frame;
  cv::Mat frame_disp = frame.clone();

  int orig_h = frame.rows;
  int orig_w = frame.cols;

  // 开始计时
  auto start_time = std::chrono::high_resolution_clock::now();

  // Preprocess
  auto input_data = preprocess(frame);

  // Create input tensor
  std::vector<Ort::Value> input_tensors;
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, input_data.data(), input_data.size(), input_shape.data(),
      input_shape.size()));

  // Run inference
  const char *input_names[] = {input_name};
  const char *output_names[] = {output_name};

  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names,
                                    input_tensors.data(), 1, output_names, 1);

  // 结束计时
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  // 输出推理时间
  // std::cout << "Inference time: " << duration.count() << "ms" << std::endl;

  // Get output data
  float *output_data = output_tensors.front().GetTensorMutableData<float>();
  size_t output_size = std::accumulate(output_shape.begin(), output_shape.end(),
                                       1, std::multiplies<size_t>());
  std::vector<float> output_vector(output_data, output_data + output_size);

  // printf("Output shape: [%ld, %ld, %ld]\n", output_shape[0], output_shape[1],
  // output_shape[2]);

  // Postprocess
  auto detections = postprocess(output_vector);

  // printf("[%s:%d] Found %zu detections\n", __FILE__, __LINE__,
  //        detections.size());

  // Process detections
  for (const auto &det : detections) {
    // Convert normalized coords to absolute
    float scale_x = static_cast<float>(orig_w) / (float)IMG_SIZE.width;
    float scale_y = static_cast<float>(orig_h) / (float)IMG_SIZE.height;

    int x1_abs = static_cast<int>(det.x1 * scale_x);
    int y1_abs = static_cast<int>(det.y1 * scale_y);
    int x2_abs = static_cast<int>(det.x2 * scale_x);
    int y2_abs = static_cast<int>(det.y2 * scale_y);

    // printf("[%s:%d]Detection: x1=%.2f, y1=%.2f, x2=%.2f, y2=%.2f,
    // confidence=%.2f\n", __FILE__, __LINE__,
    //        det.x1, det.y1, det.x2, det.y2, det.confidence);

    // printf("[%s:%d]Absolute coords: x1=%d, y1=%d, x2=%d, y2=%d\n", __FILE__,
    // __LINE__, x1_abs, y1_abs, x2_abs, y2_abs);

    // Expand and crop
    auto [cropped, roi] = expandAndCrop(frame, det);

    if (!cropped.empty()) {
      // printf("Cropped ROI: x=%d, y=%d, w=%d, h=%d\n", roi.x, roi.y,
      // roi.width, roi.height);

      // Decode QR
      std::string qr_data = decodeWithZbar(cropped);

      // Draw Decode result
      std::string label = qr_data.empty() ? "No QR" : "QR: " + qr_data;
      cv::putText(frame_disp, label, cv::Point(x1_abs, y1_abs - 30),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

      if (!qr_data.empty()) {
        std::cout << "Decoded: " << qr_data
                  << " | Confidence: " << det.confidence << std::endl;
      }
    }

    // 在 frame 上绘制检测框
    cv::rectangle(frame_disp, cv::Point(x1_abs, y1_abs),
                  cv::Point(x2_abs, y2_abs), cv::Scalar(0, 255, 0), 2);
    std::string label = "QR: " + std::to_string(det.confidence);
    cv::putText(frame_disp, label, cv::Point(x1_abs, y1_abs - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
  }

  // Display
  cv::resize(frame_disp, frame_disp, cv::Size(640, 480));
  cv::imshow("QR Detection", frame_disp);

  // Save image if flag is set
  if (image_save_flag) {
    // gen filename <image_save_path> + <timestamp> + <.jpg>
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();
    std::string filename = image_save_path + "/" + timestamp + ".jpg";

    cv::imwrite(filename, frame);
    std::cout << "Image saved as " << filename << std::endl;
    image_save_flag = false; // Reset
  }

  return frame;
}

void runInference(const std::string &weightsPath, const std::string &source) {
  // Initialize ONNX Runtime
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "QRDetector");
  Ort::SessionOptions session_options;

  // Enable CUDA execution provider
  OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

  // Set graph optimization level
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  // You may want to adjust these based on your GPU
  session_options.SetIntraOpNumThreads(1);
  session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

  Ort::Session session(env, weightsPath.c_str(), session_options);

  Ort::AllocatedStringPtr input_name =
      session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  Ort::AllocatedStringPtr output_name =
      session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());

  // Input/output shapes
  auto input_shape =
      session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  auto output_shape =
      session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

  // Handle input source
  cv::Mat frame;

  const std::string pipeline =
      "nvarguscamerasrc sensor-id=0 ! "
      "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
      "nvvidconv flip-method=2 ! "
      "video/x-raw,width=1280,height=720,format=BGRx ! "
      "videoconvert ! "
      "appsink";

  // FIXME 增加对usb csi 的区分
  if (use_csi_flag) {
    cap = new cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
    if (!cap->isOpened()) {
      std::cerr << "Error: Could not open camera" << std::endl;
      return;
    }
  } else{
    // Check if source is a camera index (single digit)
    int camera_index = source[0] - '0';
    cap = new cv::VideoCapture(camera_index, cv::CAP_V4L2);
    if (!cap->isOpened()) {
      std::cerr << "Error: Could not open camera " << camera_index << std::endl;
      return;
    }
  }

  while (true) {
    *cap >> frame;
    if (frame.empty())
      break;
    processFrame(frame, session, input_name.get(), output_name.get(),
                 input_shape, output_shape);

    if (cv::waitKey(1) == 'q')
      break;
  }
}

void thread_qr_decode(std::string yolo_weights_path,
                      std::string sr_weights_path) {
  sr = new SuperResolution();
  sr->loadModel(sr_weights_path);

  runInference(yolo_weights_path, "1");
}
