"""Multi-view 3D reconstruction using COLMAP (via pycolmap)."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Reconstructor:
    """Run Structure-from-Motion and Multi-View Stereo reconstruction.

    This class wraps ``pycolmap`` to perform:

    1. SIFT feature extraction on each input image.
    2. Exhaustive (or sequential) feature matching.
    3. Incremental SfM to recover camera poses and a sparse 3-D point cloud.
    4. Patch-Match Stereo (PatchMatchStereo) for dense depth estimation.
    5. Stereo fusion to produce a dense, colored point cloud.

    Args:
        use_gpu: Whether to request GPU acceleration in COLMAP.  Falls back to
            CPU if a GPU is not available.
        max_image_size: Downscale images whose longest side exceeds this pixel
            count before feature extraction.  Reduces memory usage and speeds
            up matching for high-resolution inputs.
        dense: If ``True`` run PatchMatch dense reconstruction to obtain a
            coloured dense point cloud.  If ``False`` only the sparse SfM
            point cloud is returned (faster, useful for testing).
    """

    def __init__(
        self,
        use_gpu: bool = False,
        max_image_size: int = 2000,
        dense: bool = True,
    ) -> None:
        self.use_gpu = use_gpu
        self.max_image_size = max_image_size
        self.dense = dense

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        image_paths: List[str],
        output_dir: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct a coloured 3-D point cloud from *image_paths*.

        Args:
            image_paths: Paths to the input images (already filtered).
            output_dir: Root directory for COLMAP working files and outputs.

        Returns:
            A tuple ``(points, colors)`` where:
            - ``points`` is an ``(N, 3)`` float32 array of XYZ coordinates.
            - ``colors`` is an ``(N, 3)`` uint8 array of RGB values.

        Raises:
            RuntimeError: If reconstruction fails or yields too few points.
        """
        try:
            import pycolmap
        except ImportError as exc:
            raise RuntimeError(
                "pycolmap is required for reconstruction.  "
                "Install it with: pip install pycolmap"
            ) from exc

        os.makedirs(output_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # 1. Prepare image directory (symlinks to avoid copying large files)
        # ------------------------------------------------------------------
        image_dir = os.path.join(output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        for src in image_paths:
            dst = os.path.join(image_dir, os.path.basename(src))
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

        database_path = os.path.join(output_dir, "database.db")
        sparse_dir = os.path.join(output_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # 2. Feature extraction
        # ------------------------------------------------------------------
        logger.info("Running COLMAP feature extraction …")
        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.max_image_size = self.max_image_size
        sift_options.use_gpu = self.use_gpu

        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            sift_options=sift_options,
        )

        # ------------------------------------------------------------------
        # 3. Feature matching
        # ------------------------------------------------------------------
        logger.info("Running COLMAP feature matching …")
        matching_options = pycolmap.SequentialMatchingOptions()
        pycolmap.match_sequential(
            database_path=database_path,
            matching_options=matching_options,
        )

        # ------------------------------------------------------------------
        # 4. Incremental SfM (sparse reconstruction)
        # ------------------------------------------------------------------
        logger.info("Running incremental SfM …")
        maps = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_dir,
            output_path=sparse_dir,
        )

        if not maps:
            raise RuntimeError(
                "SfM failed: no reconstructions produced.  "
                "Ensure input images have sufficient overlap and texture."
            )

        # Pick the largest reconstruction.
        reconstruction = max(maps.values(), key=lambda r: r.num_reg_images())
        logger.info(
            "SfM: %d registered images, %d 3-D points",
            reconstruction.num_reg_images(),
            reconstruction.num_points3D(),
        )

        # ------------------------------------------------------------------
        # 5. Dense reconstruction (optional)
        # ------------------------------------------------------------------
        if self.dense:
            points, colors = self._dense_reconstruct(
                reconstruction, image_dir, output_dir, pycolmap
            )
        else:
            points, colors = self._sparse_points(reconstruction)

        if len(points) < 100:
            raise RuntimeError(
                f"Reconstruction produced only {len(points)} points; "
                "the input video may not provide enough multi-view coverage."
            )

        logger.info(
            "Reconstruction complete: %d points", len(points)
        )
        return points, colors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sparse_points(
        self, reconstruction
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract XYZ and RGB arrays from a sparse SfM reconstruction."""
        pts = reconstruction.points3D
        if not pts:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        xyz = np.array([p.xyz for p in pts.values()], dtype=np.float32)
        rgb = np.array([p.color for p in pts.values()], dtype=np.uint8)
        return xyz, rgb

    def _dense_reconstruct(
        self,
        reconstruction,
        image_dir: str,
        output_dir: str,
        pycolmap,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run PatchMatch stereo and fusion; return dense point cloud."""
        dense_dir = os.path.join(output_dir, "dense")
        os.makedirs(dense_dir, exist_ok=True)

        try:
            # Undistort images for dense MVS.
            logger.info("Undistorting images for dense reconstruction …")
            pycolmap.undistort_images(
                output_path=dense_dir,
                input_path=os.path.join(output_dir, "sparse"),
                image_path=image_dir,
            )

            # PatchMatch stereo.
            logger.info("Running PatchMatch stereo …")
            pycolmap.patch_match_stereo(workspace_path=dense_dir)

            # Stereo fusion.
            logger.info("Running stereo fusion …")
            fused_path = os.path.join(dense_dir, "fused.ply")
            pycolmap.stereo_fusion(
                output_path=fused_path,
                workspace_path=dense_dir,
            )

            return self._load_ply(fused_path)

        except Exception as exc:
            logger.warning(
                "Dense reconstruction failed (%s); falling back to sparse.", exc
            )
            return self._sparse_points(reconstruction)

    @staticmethod
    def _load_ply(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load XYZ and RGB data from a PLY file using open3d."""
        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points, dtype=np.float32)
            if pcd.has_colors():
                colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
            else:
                colors = np.full((len(points), 3), 128, dtype=np.uint8)
            return points, colors
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load PLY point cloud from {ply_path}: {exc}"
            ) from exc
