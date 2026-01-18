# Trossen Arm Data Collection - Project Instructions

## Hardware Setup

### Robot Arm
- **Model**: Trossen WXAI V0
- **End Effector**: wxai_v0_leader
- **IP Address**: 192.168.1.99
- **Joints**: 7 (Joint 6 is gripper: 0.0 - 0.04m range)
- **Controller version**: 1.9.1
- **Driver version**: 1.9.0

### Camera
- **Model**: Intel RealSense Depth Camera D405
- **Device**: /dev/video2 or /dev/video4

### Network Setup
- PC connected via WiFi for internet
- Arm connected via Ethernet directly to PC
- PC Ethernet IP: 192.168.1.10 (manual)
- Arm IP: 192.168.1.99

## Python Environment
- Python 3.12
- Virtual environment: `.venv`
- Package manager: `uv`

## Key Dependencies
- trossen-arm
- opencv-python
- numpy, pandas, scipy, scikit-learn
- matplotlib
- jupyter

## Robot Control API

### Initialize Driver
```python
import trossen_arm
from trossen_arm import TrossenArmDriver, Mode

driver = TrossenArmDriver()
driver.configure(
    trossen_arm.Model.wxai_v0,
    trossen_arm.StandardEndEffector.wxai_v0_leader,
    "192.168.1.99",
    True  # clear_error
)
```

### Modes
- `Mode.idle` - Arm relaxed
- `Mode.position` - Position control
- `Mode.external_effort` - Gravity compensation (move by hand)

### Position Control
```python
# Joint space
driver.set_all_modes(Mode.position)
driver.set_all_positions([0.0] * 7, goal_time=2.0)

# Cartesian space
cart_pos = driver.get_cartesian_positions()  # [x, y, z, rx, ry, rz]
driver.set_cartesian_positions(cart_pos, trossen_arm.InterpolationSpace.cartesian)
```

### Manual Teaching (Move by Hand)
```python
driver.set_all_modes(Mode.idle)
time.sleep(0.5)
driver.set_arm_modes(Mode.external_effort)
# Now arm can be moved by hand
```

### Joint Limits
- Joint 0: [-3.14, 3.14] rad
- Joint 1: [0.00, 3.14] rad
- Joint 2: [0.00, 2.36] rad
- Joint 3: [-1.57, 1.57] rad
- Joint 4: [-1.57, 1.57] rad
- Joint 5: [-3.14, 3.14] rad
- Joint 6 (gripper): [0.00, 0.04] m

## Calibration Needed
1. **Camera Intrinsic Calibration** - Lens distortion correction
2. **Hand-Eye Calibration** - Transform between camera and robot frames

## Project Structure
```
trossen-arm-data-collection/
├── .claude/
│   └── instructions.md
├── src/
│   └── manual_teach.py
├── .venv/
├── pyproject.toml
└── README.md
```
