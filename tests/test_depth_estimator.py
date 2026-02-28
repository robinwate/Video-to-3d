"""Tests for DepthEstimator."""

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


def _make_fake_torch():
    """Return a minimal mock of the torch module sufficient for the tests."""
    torch_mock = MagicMock(name="torch")

    # torch.no_grad() must work as a context manager.
    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    torch_mock.no_grad.return_value = _NoGrad()

    # torch.nn.functional.interpolate must return a tensor-like object.
    def _fake_interpolate(tensor, size, mode, align_corners):
        """Return a float tensor of the requested spatial size."""
        import numpy as _np
        h, w = size
        arr = _np.ones((1, 1, h, w), dtype=_np.float32) * 0.5
        t = MagicMock()
        t.squeeze.return_value = _FakeTensor(arr.reshape(h, w))
        return t

    fn_mock = MagicMock()
    fn_mock.interpolate = _fake_interpolate
    torch_mock.nn.functional = fn_mock

    return torch_mock


class _FakeTensor:
    """Minimal fake torch Tensor that supports .cpu().numpy()."""

    def __init__(self, data: np.ndarray):
        self._data = data

    def cpu(self):
        return self

    def numpy(self):
        return self._data

    def unsqueeze(self, dim):
        return _FakeTensor(np.expand_dims(self._data, axis=dim))

    def min(self):
        return float(self._data.min())

    def max(self):
        return float(self._data.max())


class TestDepthEstimator:
    def _make_frames(self, tmp_path, n=2, mode="RGB", h=16, w=16):
        paths = []
        for i in range(n):
            p = str(tmp_path / f"frame_{i:05d}.png")
            if mode == "RGBA":
                Image.new("RGBA", (w, h), (100, 150, 200, 255)).save(p)
            else:
                Image.new("RGB", (w, h), (100, 150, 200)).save(p)
            paths.append(p)
        return paths

    def _make_estimator_with_mocks(self, h=16, w=16):
        """Return a DepthEstimator with pre-loaded mock model/processor."""
        from pipeline.depth_estimator import DepthEstimator

        estimator = DepthEstimator(device="cpu")

        fake_depth_tensor = _FakeTensor(np.ones((h, w), dtype=np.float32) * 0.5)
        # predicted_depth shape is (1, H, W) before interpolation.
        fake_depth_1hw = _FakeTensor(np.ones((1, h, w), dtype=np.float32) * 0.5)

        mock_outputs = MagicMock()
        mock_outputs.predicted_depth = fake_depth_1hw

        mock_model = MagicMock()
        mock_model.return_value = mock_outputs

        mock_processor = MagicMock()
        # Return a dict-like object that maps to a fake tensor.
        mock_processor.return_value = {"pixel_values": MagicMock()}

        estimator._processor = mock_processor
        estimator._model = mock_model

        return estimator, mock_outputs

    def test_estimate_returns_one_map_per_frame(self, tmp_path):
        """estimate() must return exactly one depth array per input frame."""
        fake_torch = _make_fake_torch()
        frames = self._make_frames(tmp_path, n=3)
        estimator, mock_outputs = self._make_estimator_with_mocks()

        with patch.dict("sys.modules", {"torch": fake_torch, "torch.nn": fake_torch.nn,
                                         "torch.nn.functional": fake_torch.nn.functional}):
            result = estimator.estimate(frames)

        assert len(result) == 3

    def test_estimate_output_shape_matches_frame(self, tmp_path):
        """Each depth map must match the spatial size of the input frame."""
        h, w = 24, 32
        fake_torch = _make_fake_torch()
        frames = self._make_frames(tmp_path, n=2, h=h, w=w)
        estimator, _ = self._make_estimator_with_mocks(h=h, w=w)

        with patch.dict("sys.modules", {"torch": fake_torch, "torch.nn": fake_torch.nn,
                                         "torch.nn.functional": fake_torch.nn.functional}):
            result = estimator.estimate(frames)

        for dm in result:
            assert dm.shape == (h, w), f"Expected ({h}, {w}), got {dm.shape}"

    def test_estimate_values_normalised_zero_to_one(self, tmp_path):
        """Depth values must be in [0, 1]."""
        fake_torch = _make_fake_torch()
        frames = self._make_frames(tmp_path, n=1)
        estimator, mock_outputs = self._make_estimator_with_mocks()
        # Override with large raw values that need normalisation.
        large = np.random.rand(16, 16).astype(np.float32) * 100
        mock_outputs.predicted_depth = _FakeTensor(large[np.newaxis])

        with patch.dict("sys.modules", {"torch": fake_torch, "torch.nn": fake_torch.nn,
                                         "torch.nn.functional": fake_torch.nn.functional}):
            result = estimator.estimate(frames)

        dm = result[0]
        assert dm.min() >= 0.0
        assert dm.max() <= 1.0 + 1e-6

    def test_estimate_rgba_zeros_background(self, tmp_path):
        """For RGBA inputs, pixels with alpha==0 must have depth==0."""
        h, w = 16, 16
        fake_torch = _make_fake_torch()

        # RGBA frame: only bottom-right quarter is opaque.
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = 128
        rgba[h // 2:, w // 2:, 3] = 255
        p = str(tmp_path / "frame.png")
        Image.fromarray(rgba, mode="RGBA").save(p)

        estimator, mock_outputs = self._make_estimator_with_mocks(h, w)
        mock_outputs.predicted_depth = _FakeTensor(
            np.ones((1, h, w), dtype=np.float32) * 0.7
        )

        with patch.dict("sys.modules", {"torch": fake_torch, "torch.nn": fake_torch.nn,
                                         "torch.nn.functional": fake_torch.nn.functional}):
            result = estimator.estimate([p])

        dm = result[0]
        # Background region (alpha==0) should be 0.
        assert dm[:h // 2, :w // 2].max() == 0.0

    def test_estimate_dtype_is_float32(self, tmp_path):
        """Depth arrays must be float32."""
        fake_torch = _make_fake_torch()
        frames = self._make_frames(tmp_path, n=1)
        estimator, _ = self._make_estimator_with_mocks()

        with patch.dict("sys.modules", {"torch": fake_torch, "torch.nn": fake_torch.nn,
                                         "torch.nn.functional": fake_torch.nn.functional}):
            result = estimator.estimate(frames)

        assert result[0].dtype == np.float32

