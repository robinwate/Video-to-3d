"""Tests for FrameFilter."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from pipeline.frame_filter import FrameFilter


def _write_image(path: str, image: np.ndarray) -> str:
    cv2.imwrite(path, image)
    return path


def _sharp_image(size: int = 120) -> np.ndarray:
    """Create a high-contrast checkerboard – high Laplacian variance."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(0, size, 10):
        for j in range(0, size, 10):
            if (i // 10 + j // 10) % 2 == 0:
                img[i : i + 10, j : j + 10] = 255
    return img


def _blurry_image(size: int = 120) -> np.ndarray:
    """Create a heavily blurred (uniform) image – low Laplacian variance."""
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    return cv2.GaussianBlur(img, (21, 21), 10)


class TestFrameFilter:
    def setup_method(self):
        self.filt = FrameFilter(
            blur_threshold=50.0, similarity_threshold=5.0, min_frames=2
        )

    def test_empty_input(self):
        assert self.filt.filter([]) == []

    def test_keeps_sharp_frames(self, tmp_path):
        paths = [
            _write_image(str(tmp_path / f"frame_{i}.jpg"), _sharp_image())
            for i in range(5)
        ]
        kept = self.filt.filter(paths)
        # Should keep at least one (sharp frames should pass blur filter).
        assert len(kept) > 0

    def test_removes_blurry_frames(self, tmp_path):
        sharp_path = _write_image(str(tmp_path / "sharp.jpg"), _sharp_image())
        blurry_paths = [
            _write_image(str(tmp_path / f"blur_{i}.jpg"), _blurry_image())
            for i in range(8)
        ]
        all_paths = [sharp_path] + blurry_paths
        # Use a high blur threshold so blurry ones get filtered.
        filt = FrameFilter(blur_threshold=100.0, similarity_threshold=1.0, min_frames=1)
        kept = filt.filter(all_paths)
        assert sharp_path in kept
        # At least some blurry frames should have been removed.
        assert len(kept) < len(all_paths)

    def test_removes_duplicates(self, tmp_path):
        img = _sharp_image()
        paths = [
            _write_image(str(tmp_path / f"dup_{i}.jpg"), img.copy())
            for i in range(10)
        ]
        filt = FrameFilter(
            blur_threshold=0.0, similarity_threshold=5.0, min_frames=1
        )
        kept = filt.filter(paths)
        # Near-identical frames should be deduplicated.
        assert len(kept) < len(paths)

    def test_min_frames_safety_net(self, tmp_path):
        """When filtering is too aggressive, min_frames saves some frames."""
        # All blurry frames.
        paths = [
            _write_image(str(tmp_path / f"b_{i}.jpg"), _blurry_image())
            for i in range(8)
        ]
        filt = FrameFilter(blur_threshold=1000.0, similarity_threshold=0.1, min_frames=3)
        kept = filt.filter(paths)
        assert len(kept) >= 1  # Safety net should kick in.

    def test_skips_unreadable_images(self, tmp_path):
        good = _write_image(str(tmp_path / "good.jpg"), _sharp_image())
        bad = str(tmp_path / "nonexistent.jpg")
        kept = self.filt.filter([good, bad])
        assert good in kept
        assert bad not in kept

    def test_laplacian_variance_sharp_greater_than_blurry(self):
        sharp = _sharp_image()
        blurry = _blurry_image()
        filt = FrameFilter()
        sharp_score = filt._laplacian_variance(sharp)
        blurry_score = filt._laplacian_variance(blurry)
        assert sharp_score > blurry_score
