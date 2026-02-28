"""Tests for BackgroundRemover."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


class TestBackgroundRemover:
    def test_remove_calls_rembg_for_each_frame(self, tmp_path):
        """remove() should call rembg.remove once per input frame."""
        from pipeline.background_remover import BackgroundRemover

        # Create tiny dummy input frames.
        frame_paths = []
        for i in range(3):
            p = str(tmp_path / f"frame_{i:05d}.jpg")
            Image.new("RGB", (8, 8), color=(i * 80, 100, 150)).save(p)
            frame_paths.append(p)

        out_dir = str(tmp_path / "bg_removed")

        # Build a fake RGBA image that rembg would return.
        fake_rgba = Image.new("RGBA", (8, 8), (255, 0, 0, 200))
        mock_remove = MagicMock(return_value=fake_rgba)
        mock_session = MagicMock()

        mock_rembg = MagicMock()
        mock_rembg.remove = mock_remove
        mock_rembg.new_session = MagicMock(return_value=mock_session)

        with patch.dict("sys.modules", {"rembg": mock_rembg}):
            remover = BackgroundRemover(model_name="u2net")
            remover._session = mock_session
            result = remover.remove(frame_paths, out_dir)

        assert len(result) == 3
        assert mock_remove.call_count == 3
        for p in result:
            assert p.endswith(".png")
            assert os.path.isfile(p)

    def test_remove_output_are_rgba_pngs(self, tmp_path):
        """Each output file must be a valid RGBA PNG."""
        from pipeline.background_remover import BackgroundRemover

        frame_paths = []
        for i in range(2):
            p = str(tmp_path / f"frame_{i:05d}.jpg")
            Image.new("RGB", (16, 16), (200, 100, 50)).save(p)
            frame_paths.append(p)

        out_dir = str(tmp_path / "bg_removed")
        # Return a real RGBA image from the mock.
        fake_rgba = Image.new("RGBA", (16, 16), (255, 128, 0, 180))

        with patch.dict("sys.modules", {"rembg": MagicMock()}):
            import sys
            sys.modules["rembg"].remove = MagicMock(return_value=fake_rgba)
            sys.modules["rembg"].new_session = MagicMock(return_value=MagicMock())

            remover = BackgroundRemover()
            remover._session = MagicMock()
            result = remover.remove(frame_paths, out_dir)

        for p in result:
            img = Image.open(p)
            assert img.mode == "RGBA", f"Expected RGBA, got {img.mode}"

    def test_remove_raises_without_rembg(self, tmp_path):
        """remove() should raise RuntimeError when rembg is not installed."""
        import sys
        from unittest.mock import patch

        frame_paths = [str(tmp_path / "f.jpg")]
        Image.new("RGB", (8, 8)).save(frame_paths[0])

        with patch.dict("sys.modules", {"rembg": None}):
            from pipeline.background_remover import BackgroundRemover
            remover = BackgroundRemover()
            with pytest.raises((RuntimeError, ImportError)):
                remover.remove(frame_paths, str(tmp_path / "out"))

    def test_output_directory_created(self, tmp_path):
        """remove() should create the output directory if it does not exist."""
        from pipeline.background_remover import BackgroundRemover

        p = str(tmp_path / "frame_00000.jpg")
        Image.new("RGB", (8, 8)).save(p)

        out_dir = str(tmp_path / "new_dir" / "bg_removed")
        fake_rgba = Image.new("RGBA", (8, 8), (255, 0, 0, 200))

        with patch.dict("sys.modules", {"rembg": MagicMock()}):
            import sys
            sys.modules["rembg"].remove = MagicMock(return_value=fake_rgba)
            sys.modules["rembg"].new_session = MagicMock(return_value=MagicMock())

            remover = BackgroundRemover()
            remover._session = MagicMock()
            remover.remove([p], out_dir)

        assert os.path.isdir(out_dir)
