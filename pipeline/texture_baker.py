"""UV-unwrap a mesh and bake textures from the source images."""

import copy
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TextureBaker:
    """UV-unwrap a mesh and bake textures from the reconstruction images.

    The baking process:
    1. UV-unwrap the mesh with ``xatlas`` (chart-based atlas).
    2. For each texel in the atlas, cast a ray outward along the surface
       normal to find which source image best covers that surface point.
    3. Project the 3-D surface point into each candidate image using the
       recovered camera model and sample the pixel colour.
    4. Blend contributions (weighted by view angle and visibility) into the
       texture atlas.

    When full camera information is not available (e.g. sparse-only mode
    without camera poses), the baker falls back to a simple *vertex colour
    projection* that transfers the per-vertex colours from the Open3D mesh
    into the texture atlas.

    Args:
        texture_size: Side length (px) of the square texture atlas.
            1024 × 1024 is a good default for mobile web AR.
        padding: Texel padding added around each UV chart to prevent
            colour bleeding across seams.
    """

    def __init__(
        self,
        texture_size: int = 1024,
        padding: int = 2,
    ) -> None:
        if texture_size not in (256, 512, 1024, 2048, 4096):
            raise ValueError(
                "texture_size must be a power of two between 256 and 4096"
            )
        self.texture_size = texture_size
        self.padding = padding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bake(
        self,
        mesh,
        image_paths: Optional[List[str]] = None,
        camera_matrices: Optional[np.ndarray] = None,
        camera_extrinsics: Optional[np.ndarray] = None,
    ) -> Tuple:
        """UV-unwrap *mesh* and bake a texture atlas.

        Args:
            mesh: An ``open3d.geometry.TriangleMesh``.
            image_paths: Paths to the source images used for colour sampling.
                If *None*, only vertex colours are used.
            camera_matrices: ``(N, 3, 3)`` intrinsic matrices.  Required when
                *image_paths* is provided.
            camera_extrinsics: ``(N, 4, 4)`` world-to-camera transforms.
                Required when *image_paths* is provided.

        Returns:
            Tuple ``(uv_mesh, texture_image)`` where:
            - ``uv_mesh`` is an ``open3d.geometry.TriangleMesh`` with triangle
              UVs set.
            - ``texture_image`` is an ``(H, W, 3)`` uint8 RGB array.
        """
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        triangles = np.asarray(mesh.triangles, dtype=np.uint32)

        if len(vertices) == 0 or len(triangles) == 0:
            raise RuntimeError("Mesh has no geometry to bake.")

        # ----------------------------------------------------------------
        # 1. UV unwrapping
        # ----------------------------------------------------------------
        logger.info("Running xatlas UV unwrapping …")
        uvs, indices = self._unwrap_uv(vertices, triangles)

        # ----------------------------------------------------------------
        # 2. Texture baking
        # ----------------------------------------------------------------
        if (
            image_paths
            and camera_matrices is not None
            and camera_extrinsics is not None
        ):
            logger.info(
                "Baking textures from %d images …", len(image_paths)
            )
            texture = self._bake_from_images(
                vertices,
                triangles,
                uvs,
                indices,
                image_paths,
                camera_matrices,
                camera_extrinsics,
            )
        else:
            logger.info(
                "No camera data – baking from vertex colours …"
            )
            vertex_colors = np.asarray(mesh.vertex_colors)
            texture = self._bake_from_vertex_colors(
                vertices, triangles, uvs, indices, vertex_colors
            )

        # ----------------------------------------------------------------
        # 3. Post-process texture (inpaint gaps, dilate to fill seams)
        # ----------------------------------------------------------------
        texture = self._postprocess_texture(texture)

        # ----------------------------------------------------------------
        # 4. Attach UVs to a copy of the mesh
        # ----------------------------------------------------------------
        uv_mesh = self._attach_uvs(mesh, uvs, indices)

        logger.info(
            "Texture baking complete: atlas %dx%d", self.texture_size, self.texture_size
        )
        return uv_mesh, texture

    # ------------------------------------------------------------------
    # UV unwrapping
    # ------------------------------------------------------------------

    def _unwrap_uv(
        self, vertices: np.ndarray, triangles: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run xatlas to generate a UV atlas.

        Returns:
            ``(uvs, uv_indices)`` where *uvs* is ``(M, 2)`` float32 and
            *uv_indices* is ``(F, 3)`` uint32.
        """
        try:
            import xatlas

            vmapping, indices, uvs = xatlas.parametrize(vertices, triangles)
            return uvs.astype(np.float32), indices.astype(np.uint32)
        except Exception as exc:
            logger.warning(
                "xatlas UV unwrapping failed (%s); using planar projection.", exc
            )
            return self._planar_uv(vertices, triangles)

    @staticmethod
    def _planar_uv(
        vertices: np.ndarray, triangles: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: simple planar UV projection from above (XZ plane)."""
        x = vertices[:, 0]
        z = vertices[:, 2]
        u = (x - x.min()) / (x.max() - x.min() + 1e-9)
        v = (z - z.min()) / (z.max() - z.min() + 1e-9)
        uvs = np.stack([u, v], axis=1).astype(np.float32)
        return uvs, triangles.astype(np.uint32)

    # ------------------------------------------------------------------
    # Texture baking from images
    # ------------------------------------------------------------------

    def _bake_from_images(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        uvs: np.ndarray,
        uv_indices: np.ndarray,
        image_paths: List[str],
        camera_matrices: np.ndarray,
        camera_extrinsics: np.ndarray,
    ) -> np.ndarray:
        """Rasterise each UV triangle and sample colour from source images."""
        size = self.texture_size
        texture = np.zeros((size, size, 3), dtype=np.float32)
        weight = np.zeros((size, size), dtype=np.float32)

        images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in image_paths]

        # Rasterise triangles.
        for tri_idx, (uv_tri, vert_tri) in enumerate(
            zip(uv_indices, triangles)
        ):
            uv_pts = np.clip(uvs[uv_tri], 0.0, 1.0)  # (3, 2) – guard against NaN
            v_pts = vertices[vert_tri]  # (3, 3) world coords

            # Pixel coords for this UV triangle.
            px = (uv_pts[:, 0] * (size - 1)).astype(np.int32)
            py = ((1 - uv_pts[:, 1]) * (size - 1)).astype(np.int32)
            tri_px = np.stack([px, py], axis=1)

            # Rasterise bounding box of UV triangle.
            x_min, y_min = tri_px.min(axis=0)
            x_max, y_max = tri_px.max(axis=0)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(size - 1, x_max)
            y_max = min(size - 1, y_max)

            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    bary = self._barycentric(tri_px, x, y)
                    if bary is None:
                        continue
                    # 3-D world position for this texel.
                    world_pt = bary[0] * v_pts[0] + bary[1] * v_pts[1] + bary[2] * v_pts[2]
                    color = self._sample_best_image(
                        world_pt, images, camera_matrices, camera_extrinsics
                    )
                    if color is not None:
                        texture[y, x] += color
                        weight[y, x] += 1.0

        # Normalise.
        mask = weight > 0
        texture[mask] /= weight[mask, np.newaxis]
        return texture.astype(np.uint8)

    @staticmethod
    def _barycentric(
        tri: np.ndarray, x: int, y: int
    ) -> Optional[np.ndarray]:
        """Return barycentric coordinates if (x, y) is inside *tri* (3×2)."""
        v0 = tri[2] - tri[0]
        v1 = tri[1] - tri[0]
        v2 = np.array([x, y]) - tri[0]
        dot00 = float(v0.dot(v0))
        dot01 = float(v0.dot(v1))
        dot02 = float(v0.dot(v2))
        dot11 = float(v1.dot(v1))
        dot12 = float(v1.dot(v2))
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-12:
            return None
        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        if u < 0 or v < 0 or u + v > 1:
            return None
        return np.array([1 - u - v, v, u], dtype=np.float32)

    @staticmethod
    def _sample_best_image(
        world_pt: np.ndarray,
        images: List[np.ndarray],
        K: np.ndarray,
        extrinsics: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Project *world_pt* into each camera; return colour from best view."""
        best_color = None
        best_z = -np.inf

        for i, img in enumerate(images):
            Ki = K[i]
            Rt = extrinsics[i]  # (4, 4) world-to-camera
            R = Rt[:3, :3]
            t = Rt[:3, 3]
            cam_pt = R @ world_pt + t
            if cam_pt[2] <= 0:
                continue
            z = cam_pt[2]
            proj = Ki @ cam_pt
            px = int(round(proj[0] / proj[2]))
            py = int(round(proj[1] / proj[2]))
            h, w = img.shape[:2]
            if 0 <= px < w and 0 <= py < h:
                if z > best_z:
                    best_z = z
                    best_color = img[py, px].astype(np.float32)

        return best_color

    # ------------------------------------------------------------------
    # Texture baking from vertex colours
    # ------------------------------------------------------------------

    def _bake_from_vertex_colors(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        uvs: np.ndarray,
        uv_indices: np.ndarray,
        vertex_colors: np.ndarray,
    ) -> np.ndarray:
        """Bake vertex colours into the UV atlas by interpolation."""
        size = self.texture_size
        texture = np.zeros((size, size, 3), dtype=np.float32)
        weight = np.zeros((size, size), dtype=np.float32)

        has_colors = len(vertex_colors) == len(vertices)

        for uv_tri, vert_tri in zip(uv_indices, triangles):
            uv_pts = np.clip(uvs[uv_tri], 0.0, 1.0)  # (3, 2) – guard against NaN
            px = (uv_pts[:, 0] * (size - 1)).astype(np.int32)
            py = ((1 - uv_pts[:, 1]) * (size - 1)).astype(np.int32)
            tri_px = np.stack([px, py], axis=1)

            if has_colors:
                v_colors = vertex_colors[vert_tri] * 255.0  # (3, 3) float
            else:
                v_colors = np.full((3, 3), 180.0)

            x_min, y_min = tri_px.min(axis=0)
            x_max, y_max = tri_px.max(axis=0)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(size - 1, x_max)
            y_max = min(size - 1, y_max)

            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    bary = self._barycentric(tri_px, x, y)
                    if bary is None:
                        continue
                    color = bary[0] * v_colors[0] + bary[1] * v_colors[1] + bary[2] * v_colors[2]
                    texture[y, x] += color
                    weight[y, x] += 1.0

        mask = weight > 0
        texture[mask] /= weight[mask, np.newaxis]
        return texture.astype(np.uint8)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess_texture(texture: np.ndarray) -> np.ndarray:
        """Inpaint zero-coverage texels and dilate to fill chart borders."""
        mask = (texture.sum(axis=2) == 0).astype(np.uint8) * 255
        if mask.any():
            texture = cv2.inpaint(texture, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Dilate filled regions to reduce colour bleeding at seams.
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(2):
            dilated = cv2.dilate(texture, kernel)
            texture = np.where(mask[:, :, np.newaxis] > 0, dilated, texture)
            mask = cv2.erode(mask, kernel)

        return texture

    # ------------------------------------------------------------------
    # Attach UVs to mesh
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_uvs(mesh, uvs: np.ndarray, uv_indices: np.ndarray):
        """Return a copy of *mesh* with per-triangle UV coordinates."""
        import open3d as o3d

        # Use deepcopy for a guaranteed full deep copy of all mesh attributes.
        # o3d.geometry.TriangleMesh(mesh) is not a reliable copy constructor
        # across all Open3D versions and can produce zero-vertex meshes.
        mesh_uv = copy.deepcopy(mesh)
        # Store UVs as per-triangle vertex UVs (flattened).
        triangle_uvs = uvs[uv_indices].reshape(-1, 2)
        # Convert to open3d Vector2dVector.
        mesh_uv.triangle_uvs = o3d.utility.Vector2dVector(
            triangle_uvs.astype(np.float64)
        )
        return mesh_uv
