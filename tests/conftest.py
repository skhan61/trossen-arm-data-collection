"""Pytest configuration and fixtures."""

import glob

import cv2


def find_gelsight_devices() -> list[int]:
    """Find all GelSight devices by scanning video devices.

    Scans /dev/video* and checks for GelSight resolution (3280x2464).

    Returns:
        List of device IDs that are GelSight sensors
    """
    gelsight_devices = []
    video_devices = sorted(glob.glob("/dev/video*"))

    for device_path in video_devices:
        device_id = int(device_path.replace("/dev/video", ""))
        try:
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                # Check if this is a GelSight by resolution
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # GelSight Mini has 3280x2464 resolution
                if width == 3280 and height == 2464:
                    gelsight_devices.append(device_id)
                cap.release()
        except Exception:
            pass

    return gelsight_devices


# Cache detected devices at module load
_GELSIGHT_DEVICES = None


def get_gelsight_devices() -> list[int]:
    """Get cached GelSight device list."""
    global _GELSIGHT_DEVICES
    if _GELSIGHT_DEVICES is None:
        _GELSIGHT_DEVICES = find_gelsight_devices()
    return _GELSIGHT_DEVICES


def pytest_generate_tests(metafunc):
    """Parametrize tests with all detected GelSight devices."""
    if "device_id" in metafunc.fixturenames:
        devices = get_gelsight_devices()
        if devices:
            metafunc.parametrize("device_id", devices, ids=[f"device{d}" for d in devices])
