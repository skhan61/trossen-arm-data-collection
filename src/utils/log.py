"""
Centralized Logging System for Trossen Arm Data Collection

Usage:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # Add project root
    from src.utils.log import get_logger

    logger = get_logger(__name__)
    logger.info("Message")

Log files are saved following the folder tree of the source file:
    src/calibration/eye_in_hand/script.py -> logs/calibration/eye_in_hand/script_TIMESTAMP.log
"""

import inspect
import logging
import sys
from pathlib import Path
from datetime import datetime

# Project root and log directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

# Add project root to sys.path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Track initialized loggers to avoid duplicate handlers
_initialized_loggers = set()


def _get_caller_path() -> Path | None:
    """Get the file path of the caller that invoked get_logger()."""
    for frame_info in inspect.stack():
        frame_path = Path(frame_info.filename)
        # Skip this file and standard library
        if frame_path == Path(__file__):
            continue
        if "site-packages" in str(frame_path):
            continue
        # Return the first caller outside this module
        if frame_path.is_file():
            return frame_path
    return None


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance

    Log file location follows the source file's folder structure:
        src/calibration/eye_in_hand/script.py -> logs/calibration/eye_in_hand/script_TIMESTAMP.log

    Example:
        logger = get_logger(__name__)
        logger.info("Starting process...")
    """
    # If __main__, resolve actual file path
    if name == "__main__":
        caller_path = _get_caller_path()
        if caller_path:
            try:
                rel_path = caller_path.relative_to(PROJECT_ROOT)
                # Convert path to module-like name: src/calibration/foo.py -> src.calibration.foo
                name = (
                    str(rel_path.with_suffix("")).replace("/", ".").replace("\\", ".")
                )
            except ValueError:
                pass  # Not under PROJECT_ROOT, use __main__

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if name in _initialized_loggers:
        return logger

    logger.setLevel(level)

    # Parse module name to get folder structure
    # e.g., "src.calibration.eye_in_hand.export_X_to_npy" -> ["src", "calibration", "eye_in_hand", "export_X_to_npy"]
    parts = name.split(".")

    # Remove "src" prefix if present
    if parts and parts[0] == "src":
        parts = parts[1:]

    # Script name is the last part
    script_name = parts[-1] if parts else "unknown"

    # Folder path is everything except the last part
    folder_parts = parts[:-1] if len(parts) > 1 else []

    # Build log directory path
    log_subdir = LOG_DIR
    for folder in folder_parts:
        log_subdir = log_subdir / folder
    log_subdir.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{script_name}_{timestamp}.log"
    log_path = log_subdir / log_file

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Mark as initialized
    _initialized_loggers.add(name)

    # Log the log file path
    logger.info(f"Log file: {log_path}")

    return logger
