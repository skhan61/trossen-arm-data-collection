"""Sensor interfaces."""

__all__ = ["GelSightSensor", "RealSenseCamera"]


def __getattr__(name):
    """Lazy import to avoid import errors when hardware libs missing."""
    if name == "GelSightSensor":
        from src.sensors.gelsight import GelSightSensor
        return GelSightSensor
    elif name == "RealSenseCamera":
        from src.sensors.realsense import RealSenseCamera
        return RealSenseCamera
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
