"""
Configuration loader for Trossen Arm Data Collection

Loads configuration from .env file at project root.

Usage:
    from src.utils.config import get_arm_ip

    ip = get_arm_ip()
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _load_env() -> dict:
    """Load .env file and return as dictionary."""
    env_vars = {}
    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


def get_arm_ip() -> str:
    """Get robot ARM_IP from .env file."""
    env = _load_env()
    ip = env.get("ARM_IP")
    if ip is None:
        raise ValueError(
            f"ARM_IP not found in {ENV_FILE}. Please add 'ARM_IP=192.168.1.99' to .env"
        )
    return ip
