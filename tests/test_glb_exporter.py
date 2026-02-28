"""Tests for GLBExporter."""

import os
import struct

import numpy as np
import pytest


def _make_minimal_mesh():
    """Return a simple Open3D triangle mesh (two triangles)."""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    # Attach simple per-triangle UVs.
    uvs = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
    return mesh


def _make_texture(size: int = 64) -> np.ndarray:
    """Return a simple RGB checkerboard texture."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            if (i // 8 + j // 8) % 2 == 0:
                img[i : i + 8, j : j + 8] = [255, 128, 0]
    return img


class TestGLBExporter:
    def test_exports_valid_glb(self, tmp_path):
        from pipeline.glb_exporter import GLBExporter

        mesh = _make_minimal_mesh()
        texture = _make_texture()
        out = str(tmp_path / "test.glb")

        exporter = GLBExporter(texture_format="jpeg")
        result_path = exporter.export(mesh, texture, out)

        assert os.path.exists(result_path)
        assert result_path.endswith(".glb")

    def test_glb_has_valid_magic(self, tmp_path):
        from pipeline.glb_exporter import GLBExporter

        mesh = _make_minimal_mesh()
        texture = _make_texture()
        out = str(tmp_path / "test.glb")

        GLBExporter().export(mesh, texture, out)

        with open(out, "rb") as fh:
            magic = struct.unpack("<I", fh.read(4))[0]
        # glTF magic number is 0x46546C67 ('glTF')
        assert magic == 0x46546C67

    def test_glb_file_size_reasonable(self, tmp_path):
        from pipeline.glb_exporter import GLBExporter

        mesh = _make_minimal_mesh()
        texture = _make_texture(256)
        out = str(tmp_path / "test.glb")

        GLBExporter(texture_format="jpeg", jpeg_quality=70).export(mesh, texture, out)

        size_bytes = os.path.getsize(out)
        # Should be at least 1 KB (has geometry + texture) but < 5 MB for this tiny mesh.
        assert 1024 < size_bytes < 5 * 1024 * 1024

    def test_png_texture_format(self, tmp_path):
        from pipeline.glb_exporter import GLBExporter

        mesh = _make_minimal_mesh()
        texture = _make_texture()
        out = str(tmp_path / "test.glb")

        GLBExporter(texture_format="png").export(mesh, texture, out)
        assert os.path.exists(out)

    def test_creates_parent_dirs(self, tmp_path):
        from pipeline.glb_exporter import GLBExporter

        mesh = _make_minimal_mesh()
        texture = _make_texture()
        out = str(tmp_path / "subdir" / "nested" / "out.glb")

        GLBExporter().export(mesh, texture, out)
        assert os.path.exists(out)

    def test_raises_on_no_triangles(self, tmp_path):
        import open3d as o3d

        from pipeline.glb_exporter import GLBExporter

        mesh = o3d.geometry.TriangleMesh()
        texture = _make_texture()
        out = str(tmp_path / "empty.glb")

        with pytest.raises(RuntimeError, match="no triangles"):
            GLBExporter().export(mesh, texture, out)

    def test_raises_on_no_uvs(self, tmp_path):
        import open3d as o3d

        from pipeline.glb_exporter import GLBExporter

        mesh = _make_minimal_mesh()
        mesh.triangle_uvs = o3d.utility.Vector2dVector([])
        texture = _make_texture()
        out = str(tmp_path / "no_uv.glb")

        with pytest.raises(RuntimeError, match="no UV"):
            GLBExporter().export(mesh, texture, out)

    def test_invalid_texture_format(self):
        from pipeline.glb_exporter import GLBExporter

        with pytest.raises(ValueError, match="texture_format"):
            GLBExporter(texture_format="bmp")

    def test_nan_uvs_produce_valid_json(self, tmp_path):
        """GLB with NaN UV values must still produce valid, parseable JSON."""
        import json
        import open3d as o3d

        from pipeline.glb_exporter import GLBExporter

        mesh = _make_minimal_mesh()
        # Inject NaN into the triangle UVs to simulate degenerate xatlas output.
        uvs_with_nan = np.array(
            [
                [float("nan"), 0.0],
                [1.0, float("nan")],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs_with_nan)
        texture = _make_texture()
        out = str(tmp_path / "nan_uv.glb")

        GLBExporter().export(mesh, texture, out)

        with open(out, "rb") as fh:
            raw = fh.read()

        # GLB layout: 12-byte file header + 8-byte JSON chunk header = data at byte 20.
        # File header:  magic(4) + version(4) + total_length(4)
        # Chunk header: chunk_length(4) + chunk_type(4)
        _GLB_JSON_DATA_OFFSET = 20
        json_chunk_len = struct.unpack_from("<I", raw, 12)[0]
        json_text = raw[_GLB_JSON_DATA_OFFSET: _GLB_JSON_DATA_OFFSET + json_chunk_len].rstrip(b" ").decode("utf-8")

        # Must not contain the bare "NaN" token (invalid JSON).
        assert "NaN" not in json_text, "GLB JSON chunk contains NaN"
        # Must parse successfully with the strict Python json parser.
        parsed = json.loads(json_text)
        assert "accessors" in parsed
