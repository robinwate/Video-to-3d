"""Integration-level tests for TextureBaker."""

import numpy as np
import pytest


def _make_simple_mesh():
    """Build a flat quad mesh (2 triangles) with vertex colours."""
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
    colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    return mesh


class TestTextureBaker:
    def test_imports(self):
        from pipeline.texture_baker import TextureBaker

        assert TextureBaker is not None

    def test_invalid_texture_size(self):
        from pipeline.texture_baker import TextureBaker

        with pytest.raises(ValueError, match="texture_size"):
            TextureBaker(texture_size=300)

    def test_bakes_from_vertex_colors(self):
        from pipeline.texture_baker import TextureBaker

        mesh = _make_simple_mesh()
        baker = TextureBaker(texture_size=256)
        uv_mesh, texture = baker.bake(mesh)

        assert texture.shape == (256, 256, 3)
        assert texture.dtype == np.uint8

    def test_uv_mesh_has_uvs(self):
        import open3d as o3d

        from pipeline.texture_baker import TextureBaker

        mesh = _make_simple_mesh()
        baker = TextureBaker(texture_size=256)
        uv_mesh, _ = baker.bake(mesh)

        uvs = np.asarray(uv_mesh.triangle_uvs)
        assert len(uvs) == len(np.asarray(mesh.triangles)) * 3

    def test_raises_on_empty_mesh(self):
        import open3d as o3d

        from pipeline.texture_baker import TextureBaker

        mesh = o3d.geometry.TriangleMesh()
        baker = TextureBaker(texture_size=256)
        with pytest.raises(RuntimeError, match="no geometry"):
            baker.bake(mesh)

    def test_barycentric_inside(self):
        from pipeline.texture_baker import TextureBaker

        tri = np.array([[0, 0], [10, 0], [0, 10]])
        bary = TextureBaker._barycentric(tri, 2, 2)
        assert bary is not None
        assert abs(bary.sum() - 1.0) < 1e-5

    def test_barycentric_outside(self):
        from pipeline.texture_baker import TextureBaker

        tri = np.array([[0, 0], [10, 0], [0, 10]])
        bary = TextureBaker._barycentric(tri, 20, 20)
        assert bary is None
