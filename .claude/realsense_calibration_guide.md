# RealSense Camera Calibration Guide

This document explains how to extract and use the RealSense camera calibration for your GelSight data collection project.

## Overview

Your RealSense D405 comes with **factory calibration** from Intel. This includes:
- Camera intrinsic parameters (focal length, principal point)
- Distortion coefficients
- Both color and depth camera calibrations

The script `extract_realsense_calibration.py` extracts these factory parameters and saves them in **ROS-compatible YAML format**.

## Quick Start

### Extract Factory Calibration

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the extraction script
python src/extract_realsense_calibration.py
```

This will create:
- `data/realsense_calibration/realsense_color.yaml` - Color camera calibration
- `data/realsense_calibration/realsense_depth.yaml` - Depth camera calibration
- `data/realsense_calibration/device_info.yaml` - Camera metadata

### View Your Camera Info

Your **Intel RealSense D405** has the following specs:

**Color Camera (640x480):**
- fx = 385.44 pixels
- fy = 384.88 pixels
- cx = 316.22 pixels (principal point x)
- cy = 240.34 pixels (principal point y)
- Distortion model: Brown-Conrady (inverse)
- Serial: 130322272684
- Firmware: 5.17.0.9

**Depth Camera (640x480):**
- fx = 381.00 pixels
- fy = 381.00 pixels
- cx = 322.96 pixels
- cy = 239.83 pixels
- No distortion (factory pre-rectified)

## Calibration Options

You have **two options** for camera calibration:

### Option 1: Use Factory Calibration (Recommended for most users)

**Advantages:**
- No calibration procedure needed
- Already optimized by Intel
- Good for most applications

**Use when:**
- You trust the factory calibration
- You need a quick start
- Your application doesn't require sub-millimeter precision

**How to use:**
```bash
python src/extract_realsense_calibration.py
# Use the generated YAML files
```

### Option 2: Custom Chessboard Calibration

**Advantages:**
- Can potentially improve accuracy
- Accounts for any lens changes or damage
- Recommended for scientific applications

**Use when:**
- You need maximum accuracy
- The factory calibration seems off
- You're doing precision measurements

**How to use:**
```bash
python src/calibrate_camera.py
# Follow the on-screen instructions
# Capture 15-20 images of the chessboard
```

## Integration with Hand-Eye Calibration

Both calibration methods produce camera intrinsics that can be used in your hand-eye calibration workflow:

```python
import json
import yaml
import numpy as np

# Option 1: Load factory calibration (ROS format)
with open('data/realsense_calibration/realsense_color.yaml', 'r') as f:
    ros_calib = yaml.safe_load(f)
    # Convert flat list to 3x3 matrix
    K_data = ros_calib['camera_matrix']['data']
    camera_matrix = np.array(K_data).reshape(3, 3)
    dist_coeffs = np.array(ros_calib['distortion_coefficients']['data'])

# OR Option 2: Load custom calibration (JSON format)
with open('data/camera_calibration_data/camera_intrinsics.json', 'r') as f:
    custom_calib = json.load(f)
    camera_matrix = np.array(custom_calib['camera_matrix'])
    dist_coeffs = np.array(custom_calib['dist_coeffs'])

# Use in hand-eye calibration
# ... (existing hand-eye calibration code)
```

## File Formats

### ROS YAML Format (factory calibration)
```yaml
image_width: 640
image_height: 480
camera_name: realsense_color
camera_matrix:
  rows: 3
  cols: 3
  data: [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [k1, k2, p1, p2, k3]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
```

### JSON Format (custom calibration)
```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs": [[k1, k2, p1, p2, k3]],
  "reprojection_error": 0.45,
  "num_images": 20
}
```

## Utility Module for Loading Calibration

A utility module has been created to easily load both calibration formats:

```python
from src.calibration_utils import load_camera_calibration

# Auto-detect and load calibration
camera_matrix, dist_coeffs, metadata = load_camera_calibration()

# Or specify which one to load
camera_matrix, dist_coeffs, metadata = load_camera_calibration('factory')
camera_matrix, dist_coeffs, metadata = load_camera_calibration('custom')
```

## Comparison

| Aspect | Factory Calibration | Custom Calibration |
|--------|--------------------|--------------------|
| **Time required** | 1 second | 10-15 minutes |
| **Accuracy** | Good (±0.5mm) | Better (±0.3mm) if done well |
| **Effort** | None | Moderate (need chessboard) |
| **Reproducibility** | Perfect | Depends on your procedure |
| **Resolution** | 640x480 | 640x480 (or custom) |

## Troubleshooting

### Camera not detected
```bash
# Check if camera is connected
rs-enumerate-devices

# Make sure pyrealsense2 is installed
pip install pyrealsense2
```

### Wrong resolution
Edit the script to change resolution:
```python
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
```

Then re-run the extraction script.

### Distortion model differences
The RealSense D405 uses **inverse Brown-Conrady** model for color, while OpenCV uses standard Brown-Conrady. The script handles this conversion automatically.

## Next Steps

After extracting calibration:

1. **Run hand-eye calibration** to get the camera-to-gripper transform:
   ```bash
   python src/automated_hand_eye_calibration.py
   ```

2. **Run camera-to-GelSight calibration** to get the camera-to-sensor transform:
   ```bash
   # (You'll need to implement this based on your setup)
   ```

3. **Use in data collection**:
   ```bash
   python src/collect_data.py
   ```

## References

- [RealSense D405 Datasheet](https://www.intelrealsense.com/depth-camera-d405/)
- [ROS camera_info format](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html)
- [Brown-Conrady distortion model](https://en.wikipedia.org/wiki/Distortion_(optics))

## Related Files

- [src/extract_realsense_calibration.py](../src/extract_realsense_calibration.py) - Extract factory calibration
- [src/calibrate_camera.py](../src/calibrate_camera.py) - Custom chessboard calibration
- [src/calibration_utils.py](../src/calibration_utils.py) - Utility functions for loading calibration
- [src/hand_eye_calibration.py](../src/hand_eye_calibration.py) - Hand-eye calibration
- [hand_eye_calibration_explanation.md](hand_eye_calibration_explanation.md) - Complete mathematical explanation

---

**Created:** 2026-01-19
**For project:** GelSight Tactile Data Collection with WidowX Robot
