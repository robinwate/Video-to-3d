"""Export a textured mesh to the GLB format."""

import logging
import os
import struct
import tempfile
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class GLBExporter:
    """Export an Open3D textured mesh as a self-contained GLB file.

    The GLB format (Binary glTF) embeds geometry, UV coordinates,
    normals, and textures in a single ``.glb`` file that can be
    directly loaded by web viewers and AR runtimes.

    Args:
        texture_format: Image format for the embedded texture.
            ``"jpeg"`` gives smaller files; ``"png"`` is lossless.
        jpeg_quality: JPEG compression quality (1–100).  Only used when
            ``texture_format="jpeg"``.
    """

    def __init__(
        self,
        texture_format: str = "jpeg",
        jpeg_quality: int = 85,
    ) -> None:
        if texture_format not in ("jpeg", "png"):
            raise ValueError("texture_format must be 'jpeg' or 'png'")
        self.texture_format = texture_format
        self.jpeg_quality = jpeg_quality

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        mesh,
        texture_image: np.ndarray,
        output_path: str,
    ) -> str:
        """Write *mesh* and *texture_image* to a GLB file at *output_path*.

        Args:
            mesh: An ``open3d.geometry.TriangleMesh`` with triangle UVs set.
            texture_image: ``(H, W, 3)`` uint8 RGB texture array.
            output_path: Destination file path.  Parent directories are
                created automatically.

        Returns:
            Absolute path of the written file.

        Raises:
            RuntimeError: If the mesh has no triangles or UVs.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        triangles = np.asarray(mesh.triangles, dtype=np.uint32)
        triangle_uvs = np.asarray(mesh.triangle_uvs, dtype=np.float32)

        if len(triangles) == 0:
            raise RuntimeError("Mesh has no triangles; cannot export GLB.")
        if len(triangle_uvs) == 0:
            raise RuntimeError("Mesh has no UV coordinates; cannot export GLB.")

        # Compute per-vertex normals (flat shading if smooth not available).
        if mesh.has_vertex_normals():
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        else:
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        # Build an indexed geometry where every triangle corner is a unique
        # vertex so that per-face UV seams are handled correctly.
        n_corners = len(triangles) * 3
        pos_arr = vertices[triangles.flatten()].astype(np.float32)  # (N*3, 3)
        norm_arr = normals[triangles.flatten()].astype(np.float32)
        uv_arr = triangle_uvs  # (N*3, 2) – already per-corner

        # Sanitize all float arrays: NaN/Inf in positions propagates to the
        # accessor min/max fields in the glTF JSON, producing invalid JSON
        # (e.g. "NaN") that strict parsers in viewers reject.  NaN/Inf in
        # binary attribute data similarly corrupts the GPU upload.
        if not np.isfinite(pos_arr).all():
            logger.warning(
                "Mesh positions contain non-finite values (NaN/Inf); "
                "clamping to 0. This may indicate upstream point-cloud or "
                "mesh-processing quality issues."
            )
            pos_arr = np.nan_to_num(pos_arr, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(norm_arr).all():
            logger.warning(
                "Mesh normals contain non-finite values; clamping to 0."
            )
            norm_arr = np.nan_to_num(norm_arr, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(uv_arr).all() or uv_arr.min() < 0.0 or uv_arr.max() > 1.0:
            logger.warning(
                "UV coordinates contain non-finite or out-of-range values; "
                "clamping to [0, 1]."
            )
            uv_arr = np.clip(
                np.nan_to_num(uv_arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
            )

        # Index buffer: simple sequential indices.
        index_arr = np.arange(n_corners, dtype=np.uint32)

        # ----------------------------------------------------------------
        # Encode texture image
        # ----------------------------------------------------------------
        tex_bytes = self._encode_texture(texture_image)

        # ----------------------------------------------------------------
        # Build glTF binary buffers
        # ----------------------------------------------------------------
        pos_bytes = pos_arr.tobytes()
        norm_bytes = norm_arr.tobytes()
        uv_bytes = uv_arr.tobytes()
        idx_bytes = index_arr.tobytes()

        # Pad all buffers to 4-byte alignment.
        def _pad4(b: bytes) -> bytes:
            r = len(b) % 4
            return b + b"\x00" * ((4 - r) % 4)

        pos_bytes = _pad4(pos_bytes)
        norm_bytes = _pad4(norm_bytes)
        uv_bytes = _pad4(uv_bytes)
        idx_bytes = _pad4(idx_bytes)
        tex_bytes_padded = _pad4(tex_bytes)

        # Buffer view byte offsets.
        off_pos = 0
        off_norm = off_pos + len(pos_bytes)
        off_uv = off_norm + len(norm_bytes)
        off_idx = off_uv + len(uv_bytes)
        off_tex = off_idx + len(idx_bytes)
        total_buf = off_tex + len(tex_bytes_padded)

        binary_blob = (
            pos_bytes + norm_bytes + uv_bytes + idx_bytes + tex_bytes_padded
        )

        # ----------------------------------------------------------------
        # Bounding box for POSITION accessor
        # ----------------------------------------------------------------
        pos_min = pos_arr.min(axis=0).tolist()
        pos_max = pos_arr.max(axis=0).tolist()

        # ----------------------------------------------------------------
        # MIME type for texture
        # ----------------------------------------------------------------
        mime = "image/jpeg" if self.texture_format == "jpeg" else "image/png"

        # ----------------------------------------------------------------
        # glTF JSON
        # ----------------------------------------------------------------
        import json

        gltf_json = {
            "asset": {"version": "2.0", "generator": "Video-to-3D Pipeline"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": 0,
                                "NORMAL": 1,
                                "TEXCOORD_0": 2,
                            },
                            "indices": 3,
                            "material": 0,
                        }
                    ]
                }
            ],
            "materials": [
                {
                    "pbrMetallicRoughness": {
                        "baseColorTexture": {"index": 0},
                        "metallicFactor": 0.0,
                        "roughnessFactor": 1.0,
                    },
                    "doubleSided": True,
                }
            ],
            "textures": [{"source": 0, "sampler": 0}],
            "samplers": [
                {
                    "magFilter": 9729,  # LINEAR
                    "minFilter": 9987,  # LINEAR_MIPMAP_LINEAR
                    "wrapS": 10497,     # REPEAT
                    "wrapT": 10497,
                }
            ],
            "images": [{"bufferView": 4, "mimeType": mime}],
            "accessors": [
                {
                    "bufferView": 0,
                    "byteOffset": 0,
                    "componentType": 5126,  # FLOAT
                    "count": n_corners,
                    "type": "VEC3",
                    "min": pos_min,
                    "max": pos_max,
                },
                {
                    "bufferView": 1,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": n_corners,
                    "type": "VEC3",
                },
                {
                    "bufferView": 2,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": n_corners,
                    "type": "VEC2",
                },
                {
                    "bufferView": 3,
                    "byteOffset": 0,
                    "componentType": 5125,  # UNSIGNED_INT
                    "count": n_corners,
                    "type": "SCALAR",
                },
            ],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": off_pos,
                    "byteLength": len(pos_bytes),
                    "target": 34962,  # ARRAY_BUFFER
                },
                {
                    "buffer": 0,
                    "byteOffset": off_norm,
                    "byteLength": len(norm_bytes),
                    "target": 34962,
                },
                {
                    "buffer": 0,
                    "byteOffset": off_uv,
                    "byteLength": len(uv_bytes),
                    "target": 34962,
                },
                {
                    "buffer": 0,
                    "byteOffset": off_idx,
                    "byteLength": len(idx_bytes),
                    "target": 34963,  # ELEMENT_ARRAY_BUFFER
                },
                {
                    "buffer": 0,
                    "byteOffset": off_tex,
                    "byteLength": len(tex_bytes),
                },
            ],
            "buffers": [{"byteLength": total_buf}],
        }

        json_str = json.dumps(gltf_json, separators=(",", ":"))
        json_bytes = json_str.encode("utf-8")
        # Pad JSON chunk to 4-byte boundary with spaces.
        r = len(json_bytes) % 4
        if r:
            json_bytes += b" " * (4 - r)

        # ----------------------------------------------------------------
        # GLB container
        # ----------------------------------------------------------------
        # Header: magic (4), version (4), total length (4) = 12 bytes
        # Chunk 0 (JSON): length (4), type (4), data
        # Chunk 1 (BIN):  length (4), type (4), data

        chunk0_len = len(json_bytes)
        chunk1_len = len(binary_blob)
        total_len = 12 + 8 + chunk0_len + 8 + chunk1_len

        header = struct.pack("<III", 0x46546C67, 2, total_len)  # 'glTF'
        chunk0_header = struct.pack("<II", chunk0_len, 0x4E4F534A)  # 'JSON'
        chunk1_header = struct.pack("<II", chunk1_len, 0x004E4942)  # 'BIN\0'

        output_path = os.path.abspath(output_path)
        with open(output_path, "wb") as fh:
            fh.write(header)
            fh.write(chunk0_header)
            fh.write(json_bytes)
            fh.write(chunk1_header)
            fh.write(binary_blob)

        file_size_mb = os.path.getsize(output_path) / 1_048_576
        logger.info(
            "GLB written: %s (%.2f MB, %d triangles)",
            output_path,
            file_size_mb,
            len(triangles),
        )
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_texture(self, image: np.ndarray) -> bytes:
        """Encode an RGB uint8 image to JPEG or PNG bytes."""
        import cv2

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.texture_format == "jpeg":
            success, buf = cv2.imencode(
                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
        else:
            success, buf = cv2.imencode(".png", bgr)
        if not success:
            raise RuntimeError("Failed to encode texture image.")
        return buf.tobytes()
