"""Remove backgrounds from extracted frames using GrabCut segmentation."""

import logging
import os
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """Remove the background from a set of frames using GrabCut segmentation.

    The algorithm assumes the object of interest is centred in each frame.
    A rectangular region around the centre is used to seed GrabCut, and
    the resulting foreground mask is applied to each image.  Masked-out
    pixels are replaced with a neutral mid-grey so they do not bias the
    colour statistics in the texture-baking stage.

    Args:
        fg_scale: Fraction of the frame dimensions used for the initial
            foreground rectangle that seeds the GrabCut algorithm (0 < fg_scale < 1).
            The rectangle is centred in the frame with width ``w * fg_scale`` and
            height ``h * fg_scale``, so pixels near the edges are treated as
            background while the central region is treated as potential foreground.
            Larger values include more of the frame as potential foreground.
        iterations: Number of GrabCut iterations.  More iterations improve
            segmentation quality at the cost of speed.
        bg_color: BGR tuple used to fill background pixels.  Defaults to
            mid-grey ``(127, 127, 127)`` which is neutral for reconstruction.
        enabled: When ``False`` the class acts as a passthrough and returns
            frames unchanged.  Useful for disabling the stage via config.
    """

    def __init__(
        self,
        fg_scale: float = 0.6,
        iterations: int = 5,
        bg_color: Tuple[int, int, int] = (127, 127, 127),
        enabled: bool = True,
    ) -> None:
        if not 0 < fg_scale < 1:
            raise ValueError("fg_scale must be between 0 and 1 (exclusive)")
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        self.fg_scale = fg_scale
        self.iterations = iterations
        self.bg_color = bg_color
        self.enabled = enabled

    def remove_backgrounds(
        self, frame_paths: List[str], output_dir: str
    ) -> List[str]:
        """Apply background removal to each frame and save the results.

        Args:
            frame_paths: Ordered list of input image paths.
            output_dir: Directory where processed frames will be written.

        Returns:
            Ordered list of output image paths (same order as *frame_paths*).
        """
        if not self.enabled:
            logger.info("Background removal disabled – passing frames through.")
            return frame_paths

        os.makedirs(output_dir, exist_ok=True)
        output_paths: List[str] = []

        for i, path in enumerate(frame_paths):
            img = cv2.imread(path)
            if img is None:
                logger.warning("Cannot read image: %s – skipping", path)
                continue
            masked = self._apply_grabcut(img)
            out_path = os.path.join(output_dir, f"masked_{i:05d}.jpg")
            cv2.imwrite(out_path, masked, [cv2.IMWRITE_JPEG_QUALITY, 95])
            output_paths.append(os.path.abspath(out_path))
            logger.debug("Background removed: %s → %s", path, out_path)

        logger.info(
            "Background removal complete: %d/%d frames processed.",
            len(output_paths),
            len(frame_paths),
        )
        return output_paths

    def _apply_grabcut(self, image: np.ndarray) -> np.ndarray:
        """Return a copy of *image* with the background replaced.

        Uses GrabCut initialised with a centre-biased rectangle so that the
        object (assumed to be centred) is treated as foreground.

        Args:
            image: BGR image as a NumPy array.

        Returns:
            BGR image with background pixels set to :attr:`bg_color`.
        """
        h, w = image.shape[:2]
        margin_x = int(w * (1 - self.fg_scale) / 2)
        margin_y = int(h * (1 - self.fg_scale) / 2)
        # Ensure the rectangle has positive width and height.
        rect_w = max(1, w - 2 * margin_x)
        rect_h = max(1, h - 2 * margin_y)
        rect = (margin_x, margin_y, rect_w, rect_h)

        mask = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(
                image,
                mask,
                rect,
                bgd_model,
                fgd_model,
                self.iterations,
                cv2.GC_INIT_WITH_RECT,
            )
        except cv2.error as exc:
            logger.warning(
                "GrabCut failed on frame (%s); returning original.", exc
            )
            return image.copy()

        # Pixels marked as definite or probable foreground.
        fg_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

        # Morphological cleanup: close small holes and remove speckles.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        result = image.copy()
        result[fg_mask == 0] = self.bg_color
        return result
