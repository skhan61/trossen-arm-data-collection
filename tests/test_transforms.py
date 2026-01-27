"""Tests for transform utilities."""

import numpy as np
import pytest

from src.utils.transforms import (
    compute_T_u,
    compute_T_base_to_gelsight,
    compute_both_gelsight_poses,
    is_valid_transform,
    load_calibration,
)


class TestComputeTu:
    """Test T(u) computation."""

    def test_zero_opening(self):
        """T(u=0) should return t0."""
        T_u_params = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
        T = compute_T_u(T_u_params, gripper_opening=0.0)

        assert T.shape == (4, 4)
        assert np.allclose(T[:3, 3], [0.1, 0.2, 0.3])
        assert np.allclose(T[:3, :3], np.eye(3))

    def test_nonzero_opening(self):
        """T(u) = t0 + k * u."""
        T_u_params = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0])
        T = compute_T_u(T_u_params, gripper_opening=0.5)

        # t = t0 + k * u = [0.1, 0, 0] + [1, 0, 0] * 0.5 = [0.6, 0, 0]
        assert np.allclose(T[:3, 3], [0.6, 0.0, 0.0])

    def test_output_is_valid_transform(self):
        """Output should be a valid homogeneous transform."""
        T_u_params = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
        T = compute_T_u(T_u_params, gripper_opening=0.04)

        assert is_valid_transform(T)


class TestComputeBaseTоGelsight:
    """Test full transform chain."""

    def test_identity_chain(self):
        """Identity transforms should give identity."""
        T_base_to_ee = np.eye(4)
        X = np.eye(4)
        T_u_params = np.zeros(6)

        T = compute_T_base_to_gelsight(T_base_to_ee, X, T_u_params, 0.0)

        assert np.allclose(T, np.eye(4))

    def test_translation_only(self):
        """Pure translations should add up."""
        T_base_to_ee = np.eye(4)
        T_base_to_ee[:3, 3] = [1.0, 0.0, 0.0]

        X = np.eye(4)
        X[:3, 3] = [0.0, 1.0, 0.0]

        T_u_params = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        T = compute_T_base_to_gelsight(T_base_to_ee, X, T_u_params, 0.0)

        # Total translation: [1, 0, 0] + [0, 1, 0] + [0, 0, 1] = [1, 1, 1]
        assert np.allclose(T[:3, 3], [1.0, 1.0, 1.0])

    def test_output_shape(self):
        """Output should be (4, 4)."""
        T_base_to_ee = np.eye(4)
        X = np.eye(4)
        T_u_params = np.zeros(6)

        T = compute_T_base_to_gelsight(T_base_to_ee, X, T_u_params, 0.04)

        assert T.shape == (4, 4)


class TestComputeBothPoses:
    """Test computing both left and right GelSight poses."""

    def test_output_shape(self):
        """Output should be (2, 4, 4)."""
        T_base_to_ee = np.eye(4)
        X = np.eye(4)
        T_u_left = np.zeros(6)
        T_u_right = np.zeros(6)

        poses = compute_both_gelsight_poses(
            T_base_to_ee, X, T_u_left, T_u_right, 0.04
        )

        assert poses.shape == (2, 4, 4)

    def test_output_dtype(self):
        """Output should be float32 for storage."""
        T_base_to_ee = np.eye(4)
        X = np.eye(4)
        T_u_left = np.zeros(6)
        T_u_right = np.zeros(6)

        poses = compute_both_gelsight_poses(
            T_base_to_ee, X, T_u_left, T_u_right, 0.04
        )

        assert poses.dtype == np.float32

    def test_left_right_different(self):
        """Left and right should differ if params differ."""
        T_base_to_ee = np.eye(4)
        X = np.eye(4)
        T_u_left = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        T_u_right = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

        poses = compute_both_gelsight_poses(
            T_base_to_ee, X, T_u_left, T_u_right, 0.0
        )

        # Left should have +0.1 x translation, right should have -0.1
        assert poses[0, 0, 3] == pytest.approx(0.1, abs=1e-6)
        assert poses[1, 0, 3] == pytest.approx(-0.1, abs=1e-6)


class TestIsValidTransform:
    """Test transform validation."""

    def test_identity_valid(self):
        """Identity matrix is valid."""
        assert is_valid_transform(np.eye(4))

    def test_translation_valid(self):
        """Pure translation is valid."""
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        assert is_valid_transform(T)

    def test_rotation_valid(self):
        """90 degree rotation is valid."""
        T = np.eye(4)
        # 90 degree rotation around Z
        T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        assert is_valid_transform(T)

    def test_wrong_shape_invalid(self):
        """Wrong shape is invalid."""
        assert not is_valid_transform(np.eye(3))
        assert not is_valid_transform(np.eye(5))

    def test_wrong_bottom_row_invalid(self):
        """Wrong bottom row is invalid."""
        T = np.eye(4)
        T[3, 0] = 1.0
        assert not is_valid_transform(T)

    def test_non_orthonormal_invalid(self):
        """Non-orthonormal rotation is invalid."""
        T = np.eye(4)
        T[:3, :3] = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]  # Scaled, not orthonormal
        assert not is_valid_transform(T)


class TestLoadCalibration:
    """Test loading calibration files."""

    def test_load_existing_calibration(self):
        """Test loading real calibration files."""
        calibration_dir = "dataset/calibration"

        X, T_u_left, T_u_right = load_calibration(calibration_dir)

        assert X.shape == (4, 4)
        assert T_u_left.shape == (6,)
        assert T_u_right.shape == (6,)

    def test_X_is_valid_transform(self):
        """X should be a valid transformation matrix."""
        calibration_dir = "dataset/calibration"

        X, _, _ = load_calibration(calibration_dir)

        assert is_valid_transform(X)
