"""Tests for PointCloudBuilder."""

import numpy as np
import pytest
from PIL import Image


def _make_rgba_frame(tmp_path, idx, h=16, w=16, fg_alpha=200):
    """Create a synthetic RGBA PNG with a foreground region."""
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = [200, 100, 50]  # uniform colour
    rgba[h // 4: 3 * h // 4, w // 4: 3 * w // 4, 3] = fg_alpha  # central square
    p = str(tmp_path / f"frame_{idx:05d}_rgba.png")
    Image.fromarray(rgba, mode="RGBA").save(p)
    return p


def _make_depth_map(h=16, w=16, value=0.5):
    """Synthetic depth map with uniform non-zero depth."""
    return np.full((h, w), value, dtype=np.float32)


class TestPointCloudBuilder:
    def test_build_returns_arrays(self, tmp_path):
        """build() must return (points, colors) arrays."""
        from pipeline.point_cloud_builder import PointCloudBuilder

        frames = [_make_rgba_frame(tmp_path, i) for i in range(3)]
        depths = [_make_depth_map() for _ in range(3)]

        builder = PointCloudBuilder(min_confidence=0.1, voxel_size=None)
        points, colors = builder.build(frames, depths)

        assert isinstance(points, np.ndarray)
        assert isinstance(colors, np.ndarray)
        assert points.ndim == 2 and points.shape[1] == 3
        assert colors.ndim == 2 and colors.shape[1] == 3

    def test_build_non_empty(self, tmp_path):
        """build() must return at least one point when foreground is visible."""
        from pipeline.point_cloud_builder import PointCloudBuilder

        frames = [_make_rgba_frame(tmp_path, i) for i in range(4)]
        depths = [_make_depth_map() for _ in range(4)]

        builder = PointCloudBuilder(min_confidence=0.1, voxel_size=None)
        points, colors = builder.build(frames, depths)

        assert len(points) > 0

    def test_build_point_color_count_matches(self, tmp_path):
        """points and colors must have the same number of rows."""
        from pipeline.point_cloud_builder import PointCloudBuilder

        frames = [_make_rgba_frame(tmp_path, i) for i in range(3)]
        depths = [_make_depth_map() for _ in range(3)]

        builder = PointCloudBuilder(min_confidence=0.1, voxel_size=None)
        points, colors = builder.build(frames, depths)

        assert len(points) == len(colors)

    def test_build_raises_when_all_frames_empty(self, tmp_path):
        """build() should raise RuntimeError if no foreground pixels survive."""
        from pipeline.point_cloud_builder import PointCloudBuilder

        # Fully transparent frames (alpha == 0 everywhere).
        frames = []
        for i in range(2):
            rgba = np.zeros((16, 16, 4), dtype=np.uint8)
            p = str(tmp_path / f"empty_{i}.png")
            Image.fromarray(rgba, mode="RGBA").save(p)
            frames.append(p)

        # Depth below min_confidence so pixels are excluded.
        depths = [np.zeros((16, 16), dtype=np.float32) for _ in range(2)]

        builder = PointCloudBuilder(min_confidence=0.1, voxel_size=None)
        with pytest.raises(RuntimeError):
            builder.build(frames, depths)

    def test_build_dtype(self, tmp_path):
        """points must be float32 and colors must be uint8."""
        from pipeline.point_cloud_builder import PointCloudBuilder

        frames = [_make_rgba_frame(tmp_path, i) for i in range(2)]
        depths = [_make_depth_map() for _ in range(2)]

        builder = PointCloudBuilder(min_confidence=0.1, voxel_size=None)
        points, colors = builder.build(frames, depths)

        assert points.dtype == np.float32
        assert colors.dtype == np.uint8

    def test_rotate_y_identity_at_zero(self):
        """Rotating by 0° should return the original points unchanged."""
        from pipeline.point_cloud_builder import PointCloudBuilder

        pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        result = PointCloudBuilder._rotate_y(pts, 0.0)
        np.testing.assert_allclose(result, pts, atol=1e-6)

    def test_rotate_y_90_degrees(self):
        """Rotating (1,0,0) by 90° around Y should give approximately (0,0,-1)."""
        from pipeline.point_cloud_builder import PointCloudBuilder

        pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        result = PointCloudBuilder._rotate_y(pts, 90.0)
        np.testing.assert_allclose(result[0], [0.0, 0.0, -1.0], atol=1e-5)

    def test_build_min_confidence_filters_low_depth(self, tmp_path):
        """Pixels with depth below min_confidence must be excluded."""
        from pipeline.point_cloud_builder import PointCloudBuilder

        frames = [_make_rgba_frame(tmp_path, 0)]
        # All depth values below threshold.
        depths = [np.full((16, 16), 0.05, dtype=np.float32)]

        builder = PointCloudBuilder(min_confidence=0.1, voxel_size=None)
        with pytest.raises(RuntimeError):
            builder.build(frames, depths)

    def test_build_filters_inf_depth_values(self, tmp_path):
        """Depth pixels with Inf values must not produce NaN/Inf point coordinates."""
        from pipeline.point_cloud_builder import PointCloudBuilder

        # Frame with foreground everywhere.
        frames = [_make_rgba_frame(tmp_path, 0, fg_alpha=200)]

        # Depth map: mix of valid values and Inf.
        depth = _make_depth_map(value=0.5)
        depth[0, 0] = float("inf")  # inject a non-finite value
        depths = [depth]

        builder = PointCloudBuilder(min_confidence=0.1, voxel_size=None)
        points, _ = builder.build(frames, depths)

        assert np.isfinite(points).all(), "Point cloud must contain only finite coordinates"
