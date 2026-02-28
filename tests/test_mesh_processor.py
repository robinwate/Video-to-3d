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

    def test_normalize_survives_nan_vertex(self):
        """_normalize must not propagate NaN from a single bad vertex to all others."""
        import open3d as o3d

        from pipeline.mesh_processor import MeshProcessor

        # Build a minimal valid mesh then inject NaN into one vertex.
        mesh = o3d.geometry.TriangleMesh()
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
        tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)

        verts_nan = verts.copy()
        verts_nan[0] = [float("nan"), float("nan"), float("nan")]
        mesh.vertices = o3d.utility.Vector3dVector(verts_nan)

        result = MeshProcessor._normalize(mesh)
        result_verts = np.asarray(result.vertices)

        assert len(result_verts) > 0, "_normalize removed all vertices"
        assert np.isfinite(result_verts).all(), (
            "NaN from one bad vertex propagated to all others"
        )
