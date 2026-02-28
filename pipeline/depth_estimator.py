"""Per-frame depth estimation using transformer-based models (Depth Anything V2)."""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Estimate per-frame depth maps using a transformer-based model.

    Uses the ``transformers`` library with Depth Anything V2 (or DPT as a
    fallback).  Heavy model weights are loaded lazily on the first call to
    :meth:`estimate` so that importing this module has no GPU/memory cost.

    Args:
        model_name: HuggingFace model identifier.
        device: Compute device – ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._processor = None
        self._model = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the depth estimation model and processor on first use."""
        if self._model is not None:
            return

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for depth estimation.  "
                "Install it with: pip install transformers"
            ) from exc

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch is required for depth estimation.  "
                "Install it with: pip install torch"
            ) from exc

        logger.info("Loading depth model: %s on %s …", self.model_name, self.device)
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, frame_paths: List[str]) -> List[np.ndarray]:
        """Return a depth map for each frame.

        For RGBA inputs, depth is estimated only over foreground pixels
        (alpha > 0); background pixels are set to 0.

        Args:
            frame_paths: Paths to input frames.  RGBA PNGs from
                :class:`~pipeline.background_remover.BackgroundRemover` are
                accepted; standard RGB images also work.

        Returns:
            A list of ``(H, W)`` float32 arrays with values normalised to
            ``[0, 1]``, one per input frame.

        Raises:
            RuntimeError: If ``transformers`` or ``torch`` are not installed.
        """
        import torch
        from PIL import Image

        self._load_model()

        depth_maps: List[np.ndarray] = []

        for idx, frame_path in enumerate(frame_paths):
            logger.debug("Estimating depth for frame %d/%d …", idx + 1, len(frame_paths))
            img = Image.open(frame_path)

            # Extract alpha mask before converting to RGB.
            if img.mode == "RGBA":
                alpha = np.array(img.split()[-1])  # (H, W) uint8
                foreground_mask = alpha > 0
                img_rgb = img.convert("RGB")
            else:
                img_rgb = img.convert("RGB")
                foreground_mask = np.ones(
                    (img_rgb.height, img_rgb.width), dtype=bool
                )

            inputs = self._processor(images=img_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                predicted_depth = outputs.predicted_depth  # (1, H', W')

            # Interpolate back to original resolution.
            import torch.nn.functional as F

            predicted_depth = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(img_rgb.height, img_rgb.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            depth_np = predicted_depth.cpu().numpy().astype(np.float32)

            # Normalise to [0, 1].
            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max > d_min:
                depth_np = (depth_np - d_min) / (d_max - d_min)
            else:
                depth_np = np.zeros_like(depth_np)

            # Zero-out background pixels.
            depth_np[~foreground_mask] = 0.0

            depth_maps.append(depth_np)

        logger.info("Depth estimation complete for %d frames.", len(depth_maps))
        return depth_maps
