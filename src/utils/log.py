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
    script_name = parts[-1] if parts else "main"
    if script_name == "__main__":
        script_name = "main"

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
