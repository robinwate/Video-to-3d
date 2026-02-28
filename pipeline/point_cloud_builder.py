"""Build a colored 3D point cloud from depth maps and masked RGB frames."""

import logging
import math
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PointCloudBuilder:
    """Build a colored 3D point cloud from depth maps and masked RGB frames.

    The approach assumes the object rotates in the video.  Each frame's
    foreground pixels are unprojected into 3-D using the depth value as the
    Z coordinate and a pinhole camera model.  The per-frame clouds are then
    rotated around the Y-axis assuming the object completes a full 360°
    rotation over all frames.

    Args:
        min_confidence: Minimum normalised depth value for a pixel to be
            included (filters near-zero / background depth pixels).
        voxel_size: Voxel grid leaf size for optional downsampling.  Set to
            ``None`` to skip downsampling.
    """

    def __init__(
        self,
        min_confidence: float = 0.1,
        voxel_size: float = 0.01,
    ) -> None:
        self.min_confidence = min_confidence
        self.voxel_size = voxel_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        frame_paths: List[str],
        depth_maps: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fuse depth maps and RGBA frames into a single colored point cloud.

        Args:
            frame_paths: Paths to RGBA PNG frames (output of
                :class:`~pipeline.background_remover.BackgroundRemover`).
            depth_maps: Per-frame depth arrays of shape ``(H, W)`` with values
                in ``[0, 1]``, as returned by
                :class:`~pipeline.depth_estimator.DepthEstimator`.

        Returns:
            A tuple ``(points, colors)`` where:

            - *points* is an ``(N, 3)`` float32 array of XYZ coordinates.
            - *colors* is an ``(N, 3)`` uint8 array of RGB values.

        Raises:
            RuntimeError: If the resulting point cloud is empty.
        """
        from PIL import Image

        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []

        n_frames = len(frame_paths)
        for idx, (frame_path, depth_map) in enumerate(zip(frame_paths, depth_maps)):
            img = Image.open(frame_path)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            rgba = np.array(img, dtype=np.uint8)  # (H, W, 4)

            H, W = depth_map.shape

            # Foreground mask: alpha > 0 AND depth above min confidence AND finite.
            # np.isfinite guards against Inf values in the depth map (NaN is already
            # excluded by the > comparison, but Inf passes it and would propagate
            # to NaN/Inf point coordinates after the pinhole un-projection).
            alpha_mask = rgba[:, :, 3] > 0
            depth_mask = np.isfinite(depth_map) & (depth_map > self.min_confidence)
            mask = alpha_mask & depth_mask

            if not mask.any():
                logger.debug("Frame %d: no foreground pixels; skipping.", idx)
                continue

            # Pixel coordinates of foreground pixels.
            ys, xs = np.where(mask)  # row, col

            # Pinhole un-projection:  use a default focal length = max(W, H).
            fx = fy = float(max(W, H))
            cx, cy = W / 2.0, H / 2.0

            z = depth_map[ys, xs].astype(np.float32)
            x = (xs - cx) / fx * z
            y = -(ys - cy) / fy * z  # flip Y so +Y is up

            pts_local = np.stack([x, y, z], axis=1)  # (M, 3)

            # Rotate this frame's cloud around Y-axis.
            angle_deg = idx * 360.0 / n_frames
            pts_rotated = self._rotate_y(pts_local, angle_deg)

            rgb = rgba[ys, xs, :3]  # (M, 3) uint8

            all_points.append(pts_rotated)
            all_colors.append(rgb)

        if not all_points:
            raise RuntimeError(
                "Point cloud is empty after fusing all frames.  "
                "Check that background removal produced valid foreground masks."
            )

        points = np.concatenate(all_points, axis=0).astype(np.float32)
        colors = np.concatenate(all_colors, axis=0).astype(np.uint8)

        logger.info(
            "Fused %d frames → %d raw points.", len(all_points), len(points)
        )

        # Optional voxel-grid downsampling via Open3D.
        if self.voxel_size is not None and self.voxel_size > 0:
            points, colors = self._voxel_downsample(points, colors)

        logger.info("Point cloud: %d points after downsampling.", len(points))
        return points, colors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_y(points: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate *points* around the Y-axis by *angle_deg* degrees."""
        theta = math.radians(angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # Rotation matrix (Y-axis):
        # [ cos  0  sin ]
        # [  0   1   0  ]
        # [-sin  0  cos ]
        R = np.array(
            [
                [cos_t, 0.0, sin_t],
                [0.0, 1.0, 0.0],
                [-sin_t, 0.0, cos_t],
            ],
            dtype=np.float32,
        )
        return (R @ points.T).T

    def _voxel_downsample(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Downsample the point cloud using an Open3D voxel grid."""
        try:
            import open3d as o3d
        except ImportError:
            logger.warning("open3d not available; skipping voxel downsampling.")
            return points, colors

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        pts_out = np.asarray(pcd_down.points, dtype=np.float32)
        clr_out = (np.asarray(pcd_down.colors) * 255).astype(np.uint8)
        return pts_out, clr_out
