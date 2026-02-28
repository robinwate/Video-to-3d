"""Convert a point cloud to a clean, optimized mesh ready for texturing."""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MeshProcessor:
    """Convert a colored point cloud to a clean, simplified mesh.

    Pipeline:
    1. Create an Open3D point cloud from ``(points, colors)``.
    2. Estimate and orient normals.
    3. Run Poisson surface reconstruction to produce a watertight mesh.
    4. Crop the mesh to the bounding box of the original points to remove
       exterior Poisson artefacts.
    5. Remove small disconnected components (floating fragments).
    6. Simplify (decimate) to a target triangle count suitable for real-time
       rendering.
    7. Re-smooth normals for correct lighting behaviour.

    Args:
        poisson_depth: Octree depth for Poisson reconstruction.  Higher values
            produce finer detail at the cost of memory and time (8–10 is
            typical for product scans).
        max_triangles: Target triangle count after decimation.  Tuned for
            mobile AR rendering.
        min_component_ratio: Connected components whose triangle count is
            below ``min_component_ratio * total_triangles`` are removed as
            floating fragments.
        smoothing_iterations: Number of Taubin smoothing passes applied after
            simplification to reduce mesh noise.
    """

    def __init__(
        self,
        poisson_depth: int = 9,
        max_triangles: int = 50_000,
        min_component_ratio: float = 0.05,
        smoothing_iterations: int = 5,
    ) -> None:
        self.poisson_depth = poisson_depth
        self.max_triangles = max_triangles
        self.min_component_ratio = min_component_ratio
        self.smoothing_iterations = smoothing_iterations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ):
        """Build a clean mesh from a colored point cloud.

        Args:
            points: ``(N, 3)`` float32 XYZ array.
            colors: ``(N, 3)`` uint8 RGB array.

        Returns:
            An ``open3d.geometry.TriangleMesh`` with vertex colors and
            consistent normals.

        Raises:
            RuntimeError: If the point cloud is too sparse or reconstruction
                fails.
        """
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "open3d is required.  Install it with: pip install open3d"
            ) from exc

        if len(points) < 100:
            raise RuntimeError(
                f"Too few points ({len(points)}) for mesh reconstruction."
            )

        # ----------------------------------------------------------------
        # 1. Build point cloud
        # ----------------------------------------------------------------
        logger.info("Building point cloud (%d points) …", len(points))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if len(colors) == len(points):
            rgb_f = colors.astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(rgb_f)

        # ----------------------------------------------------------------
        # 2. Statistical outlier removal (clean up background noise)
        # ----------------------------------------------------------------
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        logger.info(
            "Point cloud after outlier removal: %d points", len(pcd.points)
        )

        # ----------------------------------------------------------------
        # 3. Estimate and orient normals
        # ----------------------------------------------------------------
        logger.info("Estimating normals …")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(100)

        # ----------------------------------------------------------------
        # 4. Poisson reconstruction
        # ----------------------------------------------------------------
        logger.info("Running Poisson reconstruction (depth=%d) …", self.poisson_depth)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.poisson_depth
        )

        # ----------------------------------------------------------------
        # 5. Remove low-density vertices (Poisson artefacts) BEFORE crop
        # ----------------------------------------------------------------
        densities_np = np.asarray(densities)
        if len(densities_np) > 0 and len(densities_np) == len(np.asarray(mesh.vertices)):
            density_threshold = np.percentile(densities_np, 15)
            vertices_to_remove = (densities_np < density_threshold).tolist()
            mesh.remove_vertices_by_mask(vertices_to_remove)
            logger.info(
                "Removed %d low-density vertices",
                int(sum(vertices_to_remove)),
            )

        # ----------------------------------------------------------------
        # 6. Crop to input bounding box to remove exterior artefacts
        # ----------------------------------------------------------------
        logger.info("Cropping mesh to input bounding box …")
        bbox = pcd.get_axis_aligned_bounding_box()
        # Crop exactly to the cleaned point cloud bounds so that Poisson
        # surface caps (low-density fill beyond the real surface) are removed.
        mesh = mesh.crop(bbox)

        # ----------------------------------------------------------------
        # 7. Remove small disconnected components
        # ----------------------------------------------------------------
        mesh = self._remove_small_components(mesh)

        # ----------------------------------------------------------------
        # 8. Simplification / decimation
        # ----------------------------------------------------------------
        current_tris = len(np.asarray(mesh.triangles))
        if current_tris > self.max_triangles:
            logger.info(
                "Simplifying mesh: %d → %d triangles …",
                current_tris,
                self.max_triangles,
            )
            mesh = mesh.simplify_quadric_decimation(self.max_triangles)

        # ----------------------------------------------------------------
        # 9. Smooth and recompute normals
        # ----------------------------------------------------------------
        if self.smoothing_iterations > 0:
            mesh = mesh.filter_smooth_taubin(
                number_of_iterations=self.smoothing_iterations
            )
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        # ----------------------------------------------------------------
        # 10. Centre and normalise scale
        # ----------------------------------------------------------------
        mesh = self._normalize(mesh)

        logger.info(
            "Mesh processing complete: %d vertices, %d triangles",
            len(np.asarray(mesh.vertices)),
            len(np.asarray(mesh.triangles)),
        )
        return mesh

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _remove_small_components(self, mesh):
        """Remove connected components smaller than *min_component_ratio*."""
        import open3d as o3d

        triangle_clusters, cluster_n_triangles, _ = (
            mesh.cluster_connected_triangles()
        )
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        total = int(triangle_clusters.shape[0])
        if total == 0:
            return mesh

        threshold = max(1, int(total * self.min_component_ratio))
        small_cluster_ids = np.where(cluster_n_triangles < threshold)[0]
        mask = np.isin(triangle_clusters, small_cluster_ids)
        mesh.remove_triangles_by_mask(mask)
        mesh.remove_unreferenced_vertices()

        removed = int(mask.sum())
        if removed:
            logger.info("Removed %d triangles in small components", removed)
        return mesh

    @staticmethod
    def _normalize(mesh):
        """Centre the mesh at the origin and normalise scale to unit cube."""
        import open3d as o3d

        vertices = np.asarray(mesh.vertices)
        if len(vertices) == 0:
            return mesh

        center = vertices.mean(axis=0)
        vertices -= center

        # Scale to unit cube while preserving aspect ratio.
        max_extent = np.abs(vertices).max()
        if max_extent > 0:
            vertices /= max_extent

        # Place base at y=0.
        min_y = vertices[:, 1].min()
        vertices[:, 1] -= min_y

        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        return mesh
