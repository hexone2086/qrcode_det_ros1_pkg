# qrcode detect and decode ROS1 package

## Installation

```bash
# To workspace
cd ~/catkin_ws/src

# Clone repository
git clone https://github.com/hexone2086/qrcode_det_ros1_pkg.git

# To package
cd qrcode_det

# Download and install onnxruntime library(precompiled, for Jetson Jetpack. Thanks to csukuangfj)
wget https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.16.0/onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2

# Extract the library to onnxruntime
tar -xjf onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2
mv onnxruntime-linux-aarch64-gpu-1.16.0 onnxruntime

# install libzbar
sudo apt-get install libzbar-dev

# Build
cd ..
catkin_make
```

## Usage

launch
```bash
# launch using usb camera (TODO!  The current code hardcodes the USB camera serial number as 1 (/dev/video1); this should be modified in the future.)
roslaunch qrcode_det qrdecode_usb.launch

# launch using csi camera
roslaunch qrcode_det qrdecode_csi.launch
```

tigger to save image
```bash
# save image using CLI
rostopic pub /qrcode_img_tigger std_msgs/Bool -- 1
```

## TODO
- [ ] Modify the code to use the camera serial number as a parameter.