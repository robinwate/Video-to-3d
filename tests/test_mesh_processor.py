"""Tests for MeshProcessor."""

import numpy as np
import pytest


def _sphere_point_cloud(n: int = 500):
    """Return a sphere-ish point cloud with colours."""
    rng = np.random.default_rng(42)
    phi = rng.uniform(0, 2 * np.pi, n)
    theta = rng.uniform(0, np.pi, n)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    return points, colors


class TestMeshProcessor:
    def test_imports(self):
        from pipeline.mesh_processor import MeshProcessor

        assert MeshProcessor is not None

    def test_raises_on_too_few_points(self):
        from pipeline.mesh_processor import MeshProcessor

        proc = MeshProcessor()
        points = np.zeros((5, 3), dtype=np.float32)
        colors = np.zeros((5, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Too few points"):
            proc.process(points, colors)

    def test_produces_mesh(self):
        import open3d as o3d

        from pipeline.mesh_processor import MeshProcessor

        points, colors = _sphere_point_cloud(1000)
        proc = MeshProcessor(
            poisson_depth=6,
            max_triangles=5000,
            smoothing_iterations=1,
        )
        mesh = proc.process(points, colors)

        assert isinstance(mesh, o3d.geometry.TriangleMesh)
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        assert len(verts) > 0
        assert len(tris) > 0

    def test_max_triangles_respected(self):
        from pipeline.mesh_processor import MeshProcessor

        points, colors = _sphere_point_cloud(2000)
        max_t = 1000
        proc = MeshProcessor(
            poisson_depth=6,
            max_triangles=max_t,
            smoothing_iterations=0,
        )
        mesh = proc.process(points, colors)
        tris = np.asarray(mesh.triangles)
        # Allow a small tolerance (decimation is approximate).
        assert len(tris) <= max_t * 1.1

    def test_mesh_has_normals(self):
        from pipeline.mesh_processor import MeshProcessor

        points, colors = _sphere_point_cloud(1000)
        proc = MeshProcessor(poisson_depth=6, max_triangles=5000)
        mesh = proc.process(points, colors)
        assert mesh.has_vertex_normals()

    def test_normalize_centers_mesh(self):
        from pipeline.mesh_processor import MeshProcessor

        points, colors = _sphere_point_cloud(1000)
        proc = MeshProcessor(poisson_depth=6, max_triangles=5000)
        mesh = proc.process(points, colors)
        verts = np.asarray(mesh.vertices)
        # After normalisation the base should be at y≈0.
        assert verts[:, 1].min() >= -0.05
