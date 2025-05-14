#include <ros/ros.h>
#include <std_msgs/Bool.h>


#include <thread>
#include <mutex>

#include "qr_decode.h"

class QRcodeControlNode {
public:
    QRcodeControlNode() {
        sub_ = nh_.subscribe("qrcode_img_tigger", 1, &QRcodeControlNode::callback, this);
    }

    void callback(const std_msgs::Bool::ConstPtr& msg) {
        ROS_INFO("Received: %d", msg->data);

        // 设置保存标志位
        if (msg->data) {
            image_save_flag = true;
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "qr_decode_node");

    // 获取参数
    std::string _yolov5s_model_path, _fsrcnn_model_path, _image_save_path;
    if (!ros::param::get("~yolov5s_model_path", _yolov5s_model_path)) {
        ROS_ERROR("Failed to get yolov5s_model_path parameter");
        return 1;
    }
    if (!ros::param::get("~fsrcnn_model_path", _fsrcnn_model_path)) {
        ROS_ERROR("Failed to get fsrcnn_model_path parameter");
    }
    if (!ros::param::get("~image_save_path", _image_save_path)) {
        ROS_ERROR("Failed to get image_save_path parameter");
    }
    if (!ros::param::get("~use_csi_flag", use_csi_flag)) {
        
    }

    // 初始化全局变量
    image_save_flag = false;
    image_save_path = _image_save_path;

    // 启动本地库的线程
    std::thread qr_decode_thread(thread_qr_decode, _yolov5s_model_path, _fsrcnn_model_path);
    qr_decode_thread.detach();

    QRcodeControlNode node;
    ros::spin();
    return 0;
}
