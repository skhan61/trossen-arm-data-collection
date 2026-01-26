# Visual-Haptic Deformation Dataset Schema

## Folder Structure

```
dataset/
├── calibration/
│   ├── X.npy
│   ├── T_u_left_params.npy
│   └── T_u_right_params.npy
│
├── objects/
│   ├── object_001.json
│   └── object_002.json
│
├── samples/
│   ├── 000001/
│   │   ├── sample.json
│   │   ├── rgb.mp4
│   │   ├── depth.npy
│   │   ├── gelsight_left.mp4
│   │   ├── gelsight_right.mp4
│   │   └── poses.npy
│   │
│   └── 000002/
│       └── ...
│
└── metadata.json
```

---

## metadata.json

```json
{
    "name": "visual_haptic_deformation",
    "date": "2026-01-25",
    "num_objects": 10,
    "num_samples": 500
}
```

---

## objects/object_001.json

```json
{
    "object_id": "object_001",
    "description": "soft foam cube"
}
```

---

## samples/000001/sample.json

```json
{
    "sample_id": "000001",
    "object_id": "object_001",
    "contact_frame": 15,
    "max_press_frame": 45,
    "fps": 30,
    "num_frames": 90
}
```

---

## Sample Files

| File | Format | Description |
|------|--------|-------------|
| rgb.mp4 | Video (N, H, W, 3) | RealSense RGB video |
| depth.npy | (N, H, W) float32 | Depth frames in meters |
| gelsight_left.mp4 | Video (N, H, W, 3) | Left GelSight video |
| gelsight_right.mp4 | Video (N, H, W, 3) | Right GelSight video |
| poses.npy | (N, 2, 4, 4) float32 | T_base_to_gelsight [left, right] per frame |

---

## Computed At Load Time

| Variable | Source |
|----------|--------|
| V_before | rgb.mp4 frame 0 |
| D_before | depth.npy[0] |
| V_after | rgb.mp4 frame max_press_frame |
| D_after | depth.npy[max_press_frame] |
| H_left | gelsight_left.mp4 frame max_press_frame |
| H_right | gelsight_right.mp4 frame max_press_frame |
| Pose | poses.npy[max_press_frame] |
| F | Z difference: poses[contact_frame] - poses[max_press_frame] |
