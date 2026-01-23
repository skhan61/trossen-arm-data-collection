# System Setup Information

This file contains key system configuration information for the trossen-arm-data-collection project.

## ROS 2 Installation

- **Distribution**: ROS 2 Jazzy
- **Installation Path**: `/opt/ros/jazzy/`
- **Setup Command**: `source /opt/ros/jazzy/setup.bash`
- **ROS 2 Workspace**: `~/ros2_ws/`
- **Status**: ✓ Installed and working

## Project Structure

**Project Root**: `/home/skhan61/Desktop/trossen-arm-data-collection/`

### Key Directories

```
trossen-arm-data-collection/
├── .claude/                           # Documentation and guides
│   ├── hand_eye_calibration_explanation.md
│   ├── realsense_calibration_guide.md
│   └── system_setup.md               # This file
├── src/                              # Python source code
│   ├── extract_realsense_calibration.py  # Extract factory calibration
│   ├── calibration_utils.py          # Calibration loading utilities
│   ├── calibrate_camera.py           # Custom chessboard calibration
│   ├── hand_eye_calibration.py       # Hand-eye calibration
│   └── collect_data.py               # Data collection script
├── data/                             # Calibration data and datasets
│   ├── sensor_msgs/
│   │   └── CameraInfo.yaml           # ROS 2 camera calibration
│   ├── realsense_calibration/
│   │   ├── realsense_color.yaml
│   │   ├── realsense_depth.yaml
│   │   └── device_info.yaml
│   └── hand_eye_calibration_data/
├── ws_moveit/                        # ROS 2 MoveIt workspace
│   └── src/
│       └── moveit_calibration/       # MoveIt calibration tools (cloned)
└── .venv/                            # Python virtual environment
```

## Hardware

### Robot Arm
- **Model**: Interbotix WidowX
- **IP Address**: 192.168.1.99
- **Connection**: Ethernet/WiFi

### RealSense Camera
- **Model**: Intel RealSense D405
- **Serial Number**: 130322272684
- **Firmware Version**: 5.17.0.9

### Color Camera Specifications (640x480)
- fx = 385.44 pixels
- fy = 384.88 pixels
- cx = 316.22 pixels
- cy = 240.34 pixels
- Distortion model: Brown-Conrady (inverse)

### Depth Camera Specifications (640x480)
- fx = 381.00 pixels
- fy = 381.00 pixels
- cx = 322.96 pixels
- cy = 239.83 pixels
- Distortion: None (factory pre-rectified)

## Python Environment

**Virtual Environment**: `.venv/`
**Activation**: `source .venv/bin/activate`

### Key Python Packages
- pyrealsense2
- opencv-python (cv2)
- numpy
- yaml
- interbotix_xs_modules

## Calibration Files

### Factory Calibration (sensor_msgs/CameraInfo format)
**Location**: `data/sensor_msgs/CameraInfo.yaml`
**Format**: ROS 2 compatible YAML
**Generated**: 2026-01-19

### Usage in Python
```python
from src.calibration_utils import load_camera_calibration

# Auto-detect and load
camera_matrix, dist_coeffs, metadata = load_camera_calibration()

# Or specify type
camera_matrix, dist_coeffs, metadata = load_camera_calibration('factory')
```

### Usage in ROS 2
```bash
# Camera info file is ready for use with camera_info_manager
# Path: data/sensor_msgs/CameraInfo.yaml
```

## Common Commands

### Launch Robot with MoveIt + RViz
```bash
# Terminal 1: Launch robot arm with MoveIt and RViz
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash && ros2 launch trossen_arm_moveit moveit.launch.py robot_model:=wxai hardware_type:=real ip_address:=192.168.1.99 arm_variant:=follower
```

### Launch RealSense Camera Node
```bash
# Terminal 2: Launch RealSense camera
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash && ros2 launch realsense2_camera rs_launch.py
```

### ROS 2 Setup
```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Build workspace
cd ~/ros2_ws
colcon build

# Source workspace
source install/setup.bash
```

### Python Scripts
```bash
# Activate Python environment
source .venv/bin/activate

# Extract RealSense calibration
python src/extract_realsense_calibration.py

# Custom chessboard calibration
python src/calibrate_camera.py

# View calibration info
python src/calibration_utils.py

# Hand-eye calibration
python src/automated_hand_eye_calibration.py

# Data collection
python src/collect_data.py
```

## MoveIt Workspace

**Location**: `ws_moveit/`
**Repository Cloned**: moveit_calibration (ROS 2 branch)

### Build Instructions
```bash
cd ws_moveit
source /opt/ros/rolling/setup.bash
rosdep install --from-paths src --ignore-src -r -y --rosdistro rolling
colcon build
source install/setup.bash
```

## Network/System Info

- **OS**: Linux 6.14.0-37-generic
- **Date**: 2026-01-19
- **User**: skhan61
- **Working Directory**: /home/skhan61/Desktop/trossen-arm-data-collection

## Notes

- ROS 2 Rolling is the development version and receives continuous updates
- Factory calibration has been extracted and saved in ROS 2 format
- Custom calibration can be performed for higher accuracy if needed
- All calibration files are compatible with both Python (OpenCV) and ROS 2

---

**Last Updated**: 2026-01-19
**For Project**: GelSight Tactile Data Collection with WidowX Robot
