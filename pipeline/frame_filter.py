"""Filter extracted frames to remove blurry and near-duplicate images."""

import logging
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameFilter:
    """Remove blurry and near-duplicate frames from a list of image paths.

    Blur is detected via the variance of the Laplacian operator: a low
    variance means the image lacks high-frequency detail and is likely blurry.

    Near-duplicates are detected by computing a perceptual hash (average hash)
    over a small thumbnail and comparing consecutive frames.  Frames that are
    too similar to the previous *kept* frame are discarded so that the
    remaining set has sufficient parallax for reconstruction.

    Args:
        blur_threshold: Frames whose Laplacian variance is below this value
            are considered blurry and discarded.  Typical indoor product
            videos work well with values in the range 50–150.
        similarity_threshold: Frames whose mean absolute difference of their
            perceptual hash thumbnail vs the previous kept frame is below this
            value are considered duplicates and discarded (0–255 scale,
            lower = more strict).
        min_frames: If filtering would leave fewer than this many frames, the
            filter is relaxed and the least-blurry frames are returned.
    """

    _HASH_SIZE = 16  # thumbnail dimension for perceptual hash comparison

    def __init__(
        self,
        blur_threshold: float = 80.0,
        similarity_threshold: float = 8.0,
        min_frames: int = 10,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.similarity_threshold = similarity_threshold
        self.min_frames = min_frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, frame_paths: List[str]) -> List[str]:
        """Return the subset of *frame_paths* that pass quality checks.

        Args:
            frame_paths: Ordered list of image file paths.

        Returns:
            Filtered list of image paths (same order, subset of input).
        """
        if not frame_paths:
            return []

        scored = self._score_frames(frame_paths)

        # Step 1 – remove blurry frames.
        sharp = [(p, score) for p, score in scored if score >= self.blur_threshold]
        logger.info(
            "Blur filter: %d/%d frames kept (threshold=%.1f)",
            len(sharp),
            len(scored),
            self.blur_threshold,
        )

        # If too few survived blur filtering, relax and use the top-N by score.
        if len(sharp) < self.min_frames:
            logger.warning(
                "Too few sharp frames (%d); relaxing blur threshold.",
                len(sharp),
            )
            sharp = sorted(scored, key=lambda x: x[1], reverse=True)[
                : max(self.min_frames, len(scored) // 2)
            ]
            # Restore original order.
            path_set = {p for p, _ in sharp}
            sharp = [(p, s) for p, s in scored if p in path_set]

        # Step 2 – remove near-duplicates.
        deduped = self._remove_duplicates([p for p, _ in sharp])
        logger.info(
            "Dedup filter: %d/%d frames kept (threshold=%.1f)",
            len(deduped),
            len(sharp),
            self.similarity_threshold,
        )

        # Final safety net.
        if len(deduped) < self.min_frames and len(sharp) >= self.min_frames:
            logger.warning(
                "Dedup left too few frames; returning all sharp frames."
            )
            return [p for p, _ in sharp]

        return deduped

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _laplacian_variance(self, image: np.ndarray) -> float:
        """Compute the variance of the Laplacian (sharpness measure)."""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image
        )
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _thumbnail(self, image: np.ndarray) -> np.ndarray:
        """Resize image to a small grayscale thumbnail for hash comparison."""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        return cv2.resize(
            gray, (self._HASH_SIZE, self._HASH_SIZE), interpolation=cv2.INTER_AREA
        ).astype(np.float32)

    def _score_frames(
        self, frame_paths: List[str]
    ) -> List[Tuple[str, float]]:
        """Return (path, blur_score) pairs for each frame path."""
        results: List[Tuple[str, float]] = []
        for path in frame_paths:
            img = cv2.imread(path)
            if img is None:
                logger.warning("Cannot read image: %s – skipping", path)
                continue
            score = self._laplacian_variance(img)
            results.append((path, score))
        return results

    def _remove_duplicates(self, frame_paths: List[str]) -> List[str]:
        """Remove consecutive near-duplicate frames."""
        if not frame_paths:
            return []

        kept: List[str] = []
        prev_thumb: np.ndarray | None = None

        for path in frame_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            thumb = self._thumbnail(img)

            if prev_thumb is None:
                kept.append(path)
                prev_thumb = thumb
                continue

            diff = float(np.mean(np.abs(thumb - prev_thumb)))
            if diff >= self.similarity_threshold:
                kept.append(path)
                prev_thumb = thumb
            else:
                logger.debug(
                    "Duplicate removed (diff=%.2f): %s", diff, path
                )

        return kept
