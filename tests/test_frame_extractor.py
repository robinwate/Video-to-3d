"""Tests for FrameExtractor."""

import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from pipeline.frame_extractor import FrameExtractor


def _make_test_video(path: str, n_frames: int = 30, fps: int = 30) -> None:
    """Write a minimal synthetic video to *path*."""
    height, width = 120, 160
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(n_frames):
        # Alternating grey frames so blur filter won't reject all of them.
        color = int(i * 255 / n_frames)
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        # Add some texture so blur detection passes.
        cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (255, 0, 0), 2)
        writer.write(frame)
    writer.release()


class TestFrameExtractor:
    def setup_method(self):
        self.extractor = FrameExtractor(target_fps=5.0, max_frames=20, min_frames=2)

    def test_raises_if_video_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self.extractor.extract(str(tmp_path / "missing.mp4"), str(tmp_path))

    def test_raises_if_output_dir_created(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_test_video(video_path)
        out_dir = str(tmp_path / "new_dir" / "frames")
        frames = self.extractor.extract(video_path, out_dir)
        assert os.path.isdir(out_dir)
        assert len(frames) > 0

    def test_extracts_jpeg_files(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_test_video(video_path, n_frames=60)
        out_dir = str(tmp_path / "frames")
        frames = self.extractor.extract(video_path, out_dir)
        assert all(f.endswith(".jpg") for f in frames)
        assert all(os.path.exists(f) for f in frames)

    def test_respects_max_frames(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_test_video(video_path, n_frames=300, fps=30)
        extractor = FrameExtractor(target_fps=30.0, max_frames=10, min_frames=2)
        out_dir = str(tmp_path / "frames")
        frames = extractor.extract(video_path, out_dir)
        assert len(frames) <= 10

    def test_raises_if_too_few_frames(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_test_video(video_path, n_frames=3, fps=30)
        extractor = FrameExtractor(target_fps=0.1, max_frames=200, min_frames=50)
        out_dir = str(tmp_path / "frames")
        with pytest.raises(RuntimeError, match="at least 50"):
            extractor.extract(video_path, out_dir)

    def test_frames_are_sorted(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_test_video(video_path, n_frames=60)
        out_dir = str(tmp_path / "frames")
        frames = self.extractor.extract(video_path, out_dir)
        assert frames == sorted(frames)

    def test_get_video_metadata(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_test_video(video_path, n_frames=30, fps=30)
        meta = self.extractor.get_video_metadata(video_path)
        assert meta["width"] == 160
        assert meta["height"] == 120
        assert meta["frame_count"] == 30
        assert meta["fps"] > 0

    def test_get_metadata_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self.extractor.get_video_metadata(str(tmp_path / "missing.mp4"))

    def test_invalid_target_fps(self):
        with pytest.raises(ValueError, match="target_fps"):
            FrameExtractor(target_fps=0)

    def test_invalid_max_frames(self):
        with pytest.raises(ValueError):
            FrameExtractor(target_fps=1.0, max_frames=5, min_frames=10)
