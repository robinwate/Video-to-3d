"""AI-based background removal for video frames using rembg / U2-Net."""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """Remove background from frames using AI (rembg / U2-Net).

    Args:
        model_name: Name of the rembg model to use.  ``"u2net"`` is the
            default general-purpose model; ``"u2netp"`` is a lighter variant.
    """

    def __init__(self, model_name: str = "u2net") -> None:
        self.model_name = model_name
        self._session = None

    def _get_session(self):
        """Lazily initialise the rembg session."""
        if self._session is None:
            try:
                from rembg import new_session
            except ImportError as exc:
                raise RuntimeError(
                    "rembg is required for background removal.  "
                    "Install it with: pip install rembg"
                ) from exc
            self._session = new_session(self.model_name)
        return self._session

    def remove(self, frame_paths: List[str], output_dir: str) -> List[str]:
        """Process each frame, remove background, and return RGBA PNG paths.

        Each output image is an RGBA PNG where the alpha channel encodes the
        foreground mask produced by the AI model.

        Args:
            frame_paths: Paths to the input frames (JPEG or PNG).
            output_dir: Directory where the RGBA output PNGs are written.

        Returns:
            List of absolute paths to the processed RGBA PNG images in the
            same order as *frame_paths*.

        Raises:
            RuntimeError: If ``rembg`` is not installed.
        """
        try:
            from rembg import remove as rembg_remove
        except ImportError as exc:
            raise RuntimeError(
                "rembg is required for background removal.  "
                "Install it with: pip install rembg"
            ) from exc

        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)
        session = self._get_session()
        output_paths: List[str] = []

        for idx, frame_path in enumerate(frame_paths):
            logger.debug("Removing background from frame %d/%d …", idx + 1, len(frame_paths))
            img = Image.open(frame_path).convert("RGB")
            rgba: Image.Image = rembg_remove(img, session=session)

            out_name = f"frame_{idx:05d}_rgba.png"
            out_path = os.path.abspath(os.path.join(output_dir, out_name))
            rgba.save(out_path, format="PNG")
            output_paths.append(out_path)

        logger.info(
            "Background removal complete: %d frames written to %s",
            len(output_paths),
            output_dir,
        )
        return output_paths
