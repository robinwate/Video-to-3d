"""End-to-end Video-to-3D pipeline orchestrator."""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .background_remover import BackgroundRemover
from .depth_estimator import DepthEstimator
from .frame_extractor import FrameExtractor
from .frame_filter import FrameFilter
from .glb_exporter import GLBExporter
from .mesh_processor import MeshProcessor
from .point_cloud_builder import PointCloudBuilder
from .texture_baker import TextureBaker

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full Video-to-3D pipeline.

    All parameters have sensible defaults that balance quality and performance
    for typical product-video inputs captured on a mid-range mobile device.
    """

    # ----- Frame extraction -----
    target_fps: float = 2.0
    """Frame rate at which to sample the input video."""

    max_frames: int = 150
    """Hard cap on the number of extracted frames."""

    # ----- Frame filtering -----
    blur_threshold: float = 15.0
    """Laplacian-variance threshold below which a frame is marked blurry."""

    similarity_threshold: float = 8.0
    """Mean-pixel-difference threshold for duplicate detection."""

    # ----- AI background removal -----
    bg_removal_model: str = "u2net"
    """rembg model used for AI background removal."""

    # ----- AI depth estimation -----
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    """HuggingFace model identifier for per-frame depth estimation."""

    device: str = "cpu"
    """Compute device for AI models: ``"cpu"`` or ``"cuda"``."""

    # ----- Point cloud -----
    min_depth_confidence: float = 0.1
    """Minimum normalised depth value to include a pixel in the point cloud."""

    # ----- Mesh processing -----
    poisson_depth: int = 9
    """Octree depth for Poisson surface reconstruction."""

    max_triangles: int = 50_000
    """Target triangle count after mesh simplification."""

    # ----- Texture baking -----
    texture_size: int = 1024
    """Side length of the square texture atlas in pixels."""

    # ----- GLB export -----
    texture_format: str = "jpeg"
    """Embedded texture format: ``"jpeg"`` or ``"png"``."""

    jpeg_quality: int = 85
    """JPEG quality for texture compression (1–100)."""


@dataclass
class PipelineResult:
    """Results returned by a successful pipeline run."""

    glb_path: str
    """Absolute path to the output ``.glb`` file."""

    num_frames_extracted: int = 0
    num_frames_used: int = 0
    num_points: int = 0
    num_triangles: int = 0
    elapsed_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)


class Pipeline:
    """Orchestrate the full video → 3D → GLB pipeline.

    Args:
        config: Pipeline configuration.  Defaults are used if not supplied.
        work_dir: Directory used for intermediate files.  A sub-directory
            unique to each run is created automatically.  Defaults to a
            system temp directory.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        work_dir: Optional[str] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.work_dir = work_dir or os.path.join(
            os.path.expanduser("~"), ".video_to_3d_cache"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, video_path: str, output_path: str) -> PipelineResult:
        """Run the complete pipeline.

        Args:
            video_path: Path to the input video file.
            output_path: Path where the final ``.glb`` file will be written.

        Returns:
            A :class:`PipelineResult` with statistics and the output path.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
            RuntimeError: If any stage of the pipeline fails critically.
        """
        start = time.monotonic()
        result = PipelineResult(glb_path=os.path.abspath(output_path))

        run_dir = self._make_run_dir(video_path)
        logger.info("Pipeline work directory: %s", run_dir)

        # ----------------------------------------------------------------
        # Stage 1 – Frame extraction
        # ----------------------------------------------------------------
        logger.info("=== Stage 1: Frame extraction ===")
        extractor = FrameExtractor(
            target_fps=self.config.target_fps,
            max_frames=self.config.max_frames,
        )
        frames_dir = os.path.join(run_dir, "frames_raw")
        all_frames = extractor.extract(video_path, frames_dir)
        result.num_frames_extracted = len(all_frames)
        logger.info("Extracted %d frames.", result.num_frames_extracted)

        # ----------------------------------------------------------------
        # Stage 2 – Frame filtering
        # ----------------------------------------------------------------
        logger.info("=== Stage 2: Frame filtering ===")
        filt = FrameFilter(
            blur_threshold=self.config.blur_threshold,
            similarity_threshold=self.config.similarity_threshold,
        )
        filtered_frames = filt.filter(all_frames)
        result.num_frames_used = len(filtered_frames)
        logger.info("Using %d frames after filtering.", result.num_frames_used)

        # ----------------------------------------------------------------
        # Stage 3 – AI background removal
        # ----------------------------------------------------------------
        logger.info("=== Stage 3: AI background removal ===")
        bg_remover = BackgroundRemover(model_name=self.config.bg_removal_model)
        bg_dir = os.path.join(run_dir, "frames_bg_removed")
        rgba_frames = bg_remover.remove(filtered_frames, bg_dir)
        logger.info("Background removed for %d frames.", len(rgba_frames))

        # ----------------------------------------------------------------
        # Stage 4 – AI depth estimation
        # ----------------------------------------------------------------
        logger.info("=== Stage 4: AI depth estimation ===")
        depth_estimator = DepthEstimator(
            model_name=self.config.depth_model,
            device=self.config.device,
        )
        depth_maps = depth_estimator.estimate(rgba_frames)
        logger.info("Depth maps estimated for %d frames.", len(depth_maps))

        # ----------------------------------------------------------------
        # Stage 5 – Neural point cloud fusion
        # ----------------------------------------------------------------
        logger.info("=== Stage 5: Depth-based point cloud fusion ===")
        pc_builder = PointCloudBuilder(min_confidence=self.config.min_depth_confidence)
        points, colors = pc_builder.build(rgba_frames, depth_maps)
        result.num_points = len(points)
        logger.info("Point cloud: %d points.", result.num_points)

        # ----------------------------------------------------------------
        # Stage 6 – Mesh processing
        # ----------------------------------------------------------------
        logger.info("=== Stage 6: Mesh processing ===")
        processor = MeshProcessor(
            poisson_depth=self.config.poisson_depth,
            max_triangles=self.config.max_triangles,
        )
        mesh = processor.process(points, colors)

        result.num_triangles = len(np.asarray(mesh.triangles))
        logger.info("Mesh: %d triangles.", result.num_triangles)

        # ----------------------------------------------------------------
        # Stage 7 – Texture baking
        # ----------------------------------------------------------------
        logger.info("=== Stage 7: Texture baking ===")
        baker = TextureBaker(texture_size=self.config.texture_size)
        uv_mesh, texture = baker.bake(mesh)

        # ----------------------------------------------------------------
        # Stage 8 – GLB export
        # ----------------------------------------------------------------
        logger.info("=== Stage 8: GLB export ===")
        exporter = GLBExporter(
            texture_format=self.config.texture_format,
            jpeg_quality=self.config.jpeg_quality,
        )
        exporter.export(uv_mesh, texture, output_path)

        result.elapsed_seconds = time.monotonic() - start
        logger.info(
            "Pipeline complete in %.1f s → %s",
            result.elapsed_seconds,
            output_path,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_run_dir(self, video_path: str) -> str:
        """Create a unique working directory for this run."""
        ts = int(time.time())
        name = os.path.splitext(os.path.basename(video_path))[0]
        run_dir = os.path.join(self.work_dir, f"{name}_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
