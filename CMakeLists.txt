cmake_minimum_required(VERSION 3.0.2)
project(qrcode_det)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")

# Set OpenCV path (using opencv-mobile)
# set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please check OpenCV_DIR or install OpenCV.")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

# 设置 ONNX Runtime 的根目录
set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/onnxruntime)
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

message(STATUS "onnxruntime path: ${ONNXRUNTIME_ROOT}")

# Find ZBar
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZBAR REQUIRED zbar)

find_package(Threads REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)


catkin_package(
 CATKIN_DEPENDS roscpp std_msgs
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${ZBAR_INCLUDE_DIRS}
)


add_executable(${PROJECT_NAME}_node src/qr_decode.cc src/fsrcnn_module.cc
src/control_node.cc)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  Threads::Threads
  ${OpenCV_LIBS}
  ${ZBAR_LIBRARIES}
  onnxruntime
)

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
