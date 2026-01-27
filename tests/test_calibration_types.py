"""
Unit tests for calibration data types.

Verifies calibration files match schema in src/utils/types.py.
"""

from pathlib import Path

import numpy as np
import pytest

from src.utils.transforms import is_valid_transform, load_calibration
from src.utils.types import T, TuParams, X, XMatrix


CALIBRATION_DIR = Path(__file__).parent.parent / "dataset" / "calibration"


class TestX:
    """Test X (eye-in-hand) calibration using X slice accessors."""

    def test_exists(self):
        path = CALIBRATION_DIR / "X.npy"
        assert path.exists(), f"X.npy not found at {path}"

    @pytest.fixture
    def x(self) -> XMatrix:
        return np.load(CALIBRATION_DIR / "X.npy")

    def test_shape(self, x: XMatrix):
        assert x.shape == (4, 4)

    def test_dtype(self, x: XMatrix):
        assert x.dtype == np.float64

    def test_rotation_shape(self, x: XMatrix):
        """X.ROTATION accessor returns (3, 3) rotation matrix."""
        rotation = x[X.ROTATION]
        assert rotation.shape == (3, 3)

    def test_rotation_orthonormal(self, x: XMatrix):
        """Rotation matrix is orthonormal (det = +1)."""
        rotation = x[X.ROTATION]
        det = np.linalg.det(rotation)
        assert np.isclose(det, 1.0, atol=1e-6)

    def test_translation_shape(self, x: XMatrix):
        """X.TRANSLATION accessor returns (3,) translation vector."""
        translation = x[X.TRANSLATION]
        assert translation.shape == (3,)

    def test_homogeneous_row(self, x: XMatrix):
        """X.HOMOGENEOUS_ROW accessor returns [0, 0, 0, 1]."""
        homogeneous = x[X.HOMOGENEOUS_ROW]
        np.testing.assert_array_almost_equal(homogeneous, [0, 0, 0, 1])

    def test_is_valid_transform(self, x: XMatrix):
        assert is_valid_transform(x)


class TestT:
    """Test T (GelSight) calibration - TuParams type."""

    def test_left_exists(self):
        path = CALIBRATION_DIR / "T_u_left_params.npy"
        assert path.exists(), f"T_u_left_params.npy not found at {path}"

    def test_right_exists(self):
        path = CALIBRATION_DIR / "T_u_right_params.npy"
        assert path.exists(), f"T_u_right_params.npy not found at {path}"

    @pytest.fixture
    def t_left(self) -> TuParams:
        return np.load(CALIBRATION_DIR / "T_u_left_params.npy")

    @pytest.fixture
    def t_right(self) -> TuParams:
        return np.load(CALIBRATION_DIR / "T_u_right_params.npy")

    def test_left_shape(self, t_left: TuParams):
        assert t_left.shape == (6,)

    def test_left_dtype(self, t_left: TuParams):
        assert t_left.dtype == np.float64

    def test_right_shape(self, t_right: TuParams):
        assert t_right.shape == (6,)

    def test_right_dtype(self, t_right: TuParams):
        assert t_right.dtype == np.float64

    def test_enum_access(self, t_left: TuParams):
        """T enum provides correct indices."""
        assert np.isfinite(t_left[T.T0_X.value])
        assert np.isfinite(t_left[T.T0_Y.value])
        assert np.isfinite(t_left[T.T0_Z.value])
        assert np.isfinite(t_left[T.K_X.value])
        assert np.isfinite(t_left[T.K_Y.value])
        assert np.isfinite(t_left[T.K_Z.value])

    def test_model_computation(self, t_left: TuParams):
        """T(u) = t0 + k*u"""
        u = 0.030  # 30mm gripper opening
        t0 = np.array(
            [
                t_left[T.T0_X.value],
                t_left[T.T0_Y.value],
                t_left[T.T0_Z.value],
            ]
        )
        k = np.array(
            [
                t_left[T.K_X.value],
                t_left[T.K_Y.value],
                t_left[T.K_Z.value],
            ]
        )
        position = t0 + k * u
        assert position.shape == (3,)
        assert np.all(np.isfinite(position))


class TestLoadCalibration:
    """Test load_calibration utility."""

    def test_load_returns_correct_types(self):
        X, T_u_left, T_u_right = load_calibration(CALIBRATION_DIR)

        assert X.shape == (4, 4)
        assert T_u_left.shape == (6,)
        assert T_u_right.shape == (6,)

    def test_X_is_valid(self):
        X, _, _ = load_calibration(CALIBRATION_DIR)
        assert is_valid_transform(X)
