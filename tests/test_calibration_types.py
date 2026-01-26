"""
Unit tests for calibration data types.

Verifies calibration files match schema in src/utils/types.py.
"""

import numpy as np
import pytest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.types import T


CALIBRATION_DIR = Path(__file__).parent.parent / "dataset" / "calibration"


class TestX:
    """Test X (eye-in-hand) calibration."""

    def test_exists(self):
        path = CALIBRATION_DIR / "X.npy"
        assert path.exists(), f"X.npy not found at {path}"

    @pytest.fixture
    def x(self):
        return np.load(CALIBRATION_DIR / "X.npy")

    def test_shape(self, x):
        assert x.shape == (4, 4)

    def test_dtype(self, x):
        assert x.dtype == np.float64

    def test_homogeneous_row(self, x):
        np.testing.assert_array_almost_equal(x[3, :], [0, 0, 0, 1])

    def test_rotation_orthonormal(self, x):
        det = np.linalg.det(x[:3, :3])
        assert np.isclose(abs(det), 1.0, atol=1e-6)


class TestT:
    """Test T (GelSight) calibration."""

    def test_left_exists(self):
        path = CALIBRATION_DIR / "T_u_left_params.npy"
        assert path.exists(), f"T_u_left_params.npy not found at {path}"

    def test_right_exists(self):
        path = CALIBRATION_DIR / "T_u_right_params.npy"
        assert path.exists(), f"T_u_right_params.npy not found at {path}"

    @pytest.fixture
    def t_left(self):
        return np.load(CALIBRATION_DIR / "T_u_left_params.npy")

    @pytest.fixture
    def t_right(self):
        return np.load(CALIBRATION_DIR / "T_u_right_params.npy")

    def test_left_shape(self, t_left):
        assert t_left.shape == (6,)

    def test_left_dtype(self, t_left):
        assert t_left.dtype == np.float64

    def test_right_shape(self, t_right):
        assert t_right.shape == (6,)

    def test_right_dtype(self, t_right):
        assert t_right.dtype == np.float64

    def test_enum_access(self, t_left):
        assert np.isfinite(t_left[T.T0_X.value])
        assert np.isfinite(t_left[T.K_Z.value])

    def test_model_computation(self, t_left):
        """T(u) = t0 + k*u"""
        u = 0.030  # 30mm
        position = t_left[:3] + t_left[3:] * u
        assert position.shape == (3,)
        assert np.all(np.isfinite(position))
