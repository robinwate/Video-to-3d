"""Extract frames from a video file for 3D reconstruction."""

import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from a video at a controlled frame rate.

    Args:
        target_fps: Desired output frames per second.  For a typical
            product-video of 30–60 fps this defaults to 2 fps which
            gives enough overlap for photogrammetric reconstruction
            without creating thousands of redundant images.
        max_frames: Hard upper limit on the number of frames kept.
            Prevents accidental extraction of tens-of-thousands of
            frames from long videos.
        min_frames: Minimum frames required for a valid reconstruction.
    """

    def __init__(
        self,
        target_fps: float = 2.0,
        max_frames: int = 200,
        min_frames: int = 10,
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if max_frames < min_frames:
            raise ValueError("max_frames must be >= min_frames")
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.min_frames = min_frames

    def extract(self, video_path: str, output_dir: str) -> List[str]:
        """Extract frames from *video_path* and write them to *output_dir*.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory where JPEG frames will be written.

        Returns:
            Sorted list of absolute paths to the extracted frame images.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
            RuntimeError: If the video cannot be opened or has no frames.
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        output_dir = str(output_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(
                "Video: %.1f fps, %d total frames", source_fps, total_frames
            )

            # Compute how many source frames to skip between extractions.
            frame_step = max(1, int(round(source_fps / self.target_fps)))

            frame_paths: List[str] = []
            frame_index = 0
            extracted = 0

            while extracted < self.max_frames:
                # Seek to the desired frame position for efficiency.
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    break

                filename = os.path.join(output_dir, f"frame_{extracted:05d}.jpg")
                cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_paths.append(os.path.abspath(filename))
                logger.debug("Saved frame %d → %s", frame_index, filename)

                extracted += 1
                frame_index += frame_step

        finally:
            cap.release()

        if len(frame_paths) < self.min_frames:
            raise RuntimeError(
                f"Only {len(frame_paths)} frames extracted; "
                f"need at least {self.min_frames} for reconstruction."
            )

        logger.info("Extracted %d frames to %s", len(frame_paths), output_dir)
        return sorted(frame_paths)

    def get_video_metadata(self, video_path: str) -> dict:
        """Return basic metadata for a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary with keys: fps, frame_count, width, height, duration_s.
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_s = frame_count / fps if fps > 0 else 0.0
        finally:
            cap.release()

        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_s": duration_s,
        }
