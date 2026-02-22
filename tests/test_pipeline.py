"""Tests for the Pipeline orchestrator (unit-level with mocks)."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestPipelineConfig:
    def test_default_values(self):
        from pipeline.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.target_fps == 2.0
        assert cfg.max_frames == 150
        assert cfg.texture_size == 1024
        assert cfg.texture_format == "jpeg"

    def test_custom_values(self):
        from pipeline.pipeline import PipelineConfig

        cfg = PipelineConfig(target_fps=5.0, max_triangles=10_000)
        assert cfg.target_fps == 5.0
        assert cfg.max_triangles == 10_000


class TestPipeline:
    def test_init_default_config(self):
        from pipeline import Pipeline

        p = Pipeline()
        assert p.config is not None

    def test_init_custom_config(self):
        from pipeline import Pipeline
        from pipeline.pipeline import PipelineConfig

        cfg = PipelineConfig(target_fps=5.0)
        p = Pipeline(config=cfg)
        assert p.config.target_fps == 5.0

    def test_run_raises_file_not_found(self, tmp_path):
        from pipeline import Pipeline

        p = Pipeline()
        with pytest.raises(FileNotFoundError):
            p.run(str(tmp_path / "missing.mp4"), str(tmp_path / "out.glb"))

    @patch("pipeline.pipeline.GLBExporter")
    @patch("pipeline.pipeline.TextureBaker")
    @patch("pipeline.pipeline.MeshProcessor")
    @patch("pipeline.pipeline.Reconstructor")
    @patch("pipeline.pipeline.FrameFilter")
    @patch("pipeline.pipeline.FrameExtractor")
    def test_run_calls_all_stages(
        self,
        MockExtractor,
        MockFilter,
        MockReconstructor,
        MockMeshProcessor,
        MockTextureBaker,
        MockGLBExporter,
        tmp_path,
    ):
        import open3d as o3d

        from pipeline import Pipeline
        from pipeline.pipeline import PipelineConfig

        # Create a dummy video file so the existence check passes.
        video = str(tmp_path / "test.mp4")
        open(video, "wb").close()

        # Set up mocks.
        mock_frames = [str(tmp_path / f"f{i}.jpg") for i in range(5)]
        MockExtractor.return_value.extract.return_value = mock_frames
        MockFilter.return_value.filter.return_value = mock_frames

        points = np.random.rand(200, 3).astype(np.float32)
        colors = np.random.randint(0, 255, (200, 3), dtype=np.uint8)
        MockReconstructor.return_value.reconstruct.return_value = (points, colors)

        fake_mesh = o3d.geometry.TriangleMesh()
        # Give it some triangles so the count is > 0.
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        tris = np.array([[0, 1, 2]], dtype=np.int32)
        fake_mesh.vertices = o3d.utility.Vector3dVector(verts)
        fake_mesh.triangles = o3d.utility.Vector3iVector(tris)
        MockMeshProcessor.return_value.process.return_value = fake_mesh

        texture = np.zeros((256, 256, 3), dtype=np.uint8)
        MockTextureBaker.return_value.bake.return_value = (fake_mesh, texture)
        MockGLBExporter.return_value.export.return_value = str(tmp_path / "out.glb")

        config = PipelineConfig(dense_reconstruction=False)
        pipeline = Pipeline(config=config, work_dir=str(tmp_path))
        result = pipeline.run(video, str(tmp_path / "out.glb"))

        MockExtractor.return_value.extract.assert_called_once()
        MockFilter.return_value.filter.assert_called_once()
        MockReconstructor.return_value.reconstruct.assert_called_once()
        MockMeshProcessor.return_value.process.assert_called_once()
        MockTextureBaker.return_value.bake.assert_called_once()
        MockGLBExporter.return_value.export.assert_called_once()
        assert result.num_frames_extracted == 5
        assert result.num_points == 200
