#ifndef QR_DECODE_H__
#define QR_DECODE_H__

#include <chrono>
#include <iostream>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <vector>
#include <zbar.h>

#include "fsrcnn_module.h"

extern bool use_csi_flag;
extern bool debug_view_full;
extern bool debug_view_det;
extern int process_delay_ms;
extern int usb_cam_index;
extern std::atomic<bool> image_save_flag;
extern std::string image_save_path;

extern void thread_qr_decode(std::string yolo_weights_path,
    std::string sr_weights_path);

#endif /* QR_DECODE_H__ */
