# Video-to-3D

A production-ready AI-powered pipeline that reconstructs a textured 3D GLB asset from a product video.

## Overview

The pipeline takes a single product video as input and produces a self-contained `.glb` file with embedded textures, ready for use in web viewers and AR environments.

The pipeline now uses modern AI/neural models at every stage, which means it handles videos with **dark or solid-colour backgrounds** correctly — a known limitation of classical COLMAP-based photogrammetry.

### Processing Stages

1. **Frame Extraction** – samples the video at a configurable frame rate using OpenCV
2. **Frame Filtering** – removes blurry frames (Laplacian variance) and near-duplicates
3. **AI Background Removal** – removes the background from every frame using `rembg` (U2-Net) to produce RGBA foreground masks
4. **AI Depth Estimation** – estimates per-frame depth maps using a transformer model (Depth Anything V2) applied only to foreground pixels
5. **Neural Point Cloud Fusion** – back-projects each frame's depth map into 3-D, rotates frames around Y assuming 360° object rotation, and merges into a single colored point cloud
6. **Mesh Processing** – Poisson surface reconstruction, mesh cleanup, quadric decimation, normal smoothing, and scale normalisation
7. **Texture Baking** – UV-unwraps the mesh with `xatlas` and bakes colours from source images
8. **GLB Export** – packages geometry, normals, UVs, and compressed textures into a self-contained `.glb`

## Requirements

- Python 3.9+
- See `requirements.txt` for Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `transformers` are required for depth estimation.  Install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/) for your platform/CUDA version before running `pip install -r requirements.txt`.

## Usage

```bash
python main.py path/to/product_video.mp4 -o output.glb
```

### Options

```
usage: video-to-3d [-h] [-o OUTPUT] [--fps FPS] [--max-frames MAX_FRAMES]
                   [--blur-threshold BLUR_THRESHOLD]
                   [--similarity-threshold SIMILARITY_THRESHOLD]
                   [--max-triangles MAX_TRIANGLES]
                   [--texture-size {256,512,1024,2048}]
                   [--texture-format {jpeg,png}]
                   [--device {cpu,cuda}]
                   [--bg-model BG_MODEL]
                   [--depth-model DEPTH_MODEL]
                   [--work-dir WORK_DIR] [-v]
                   input_video

positional arguments:
  input_video           Path to the input video file.

optional arguments:
  -o OUTPUT             Output .glb path (default: output.glb)
  --fps FPS             Frame sampling rate (default: 2.0)
  --max-frames N        Maximum frames to extract (default: 150)
  --blur-threshold T    Laplacian variance blur threshold (default: 15.0)
  --device {cpu,cuda}   Compute device for AI models (default: cpu)
  --bg-model BG_MODEL   rembg model name for background removal (default: u2net)
  --depth-model MODEL   HuggingFace depth model (default: depth-anything/Depth-Anything-V2-Small-hf)
  --max-triangles N     Target triangle count (default: 50000)
  --texture-size N      Texture atlas size in pixels (default: 1024)
  --texture-format FMT  jpeg or png (default: jpeg)
  -v                    Verbose logging
```

### Python API

```python
from pipeline import Pipeline
from pipeline.pipeline import PipelineConfig

config = PipelineConfig(
    target_fps=2.0,
    max_triangles=50_000,
    texture_size=1024,
    device="cpu",           # or "cuda" for GPU
    bg_removal_model="u2net",
    depth_model="depth-anything/Depth-Anything-V2-Small-hf",
)
pipeline = Pipeline(config=config)
result = pipeline.run("product_video.mp4", "output.glb")
print(f"Done: {result.glb_path} ({result.num_triangles} triangles)")
```

## Output Specification

| Property | Value |
|----------|-------|
| Format | GLB (binary glTF 2.0) |
| Textures | Embedded JPEG or PNG |
| Materials | PBR metallic-roughness (diffuse only) |
| Target triangle count | ≤ 50 000 (mobile AR) |
| Texture resolution | 1024 × 1024 px (default) |
| Coordinate origin | Base centred at origin, Y-up |

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Input Requirements

- Static object captured from multiple angles (ideally 360° rotation)
- Works with dark/solid-colour backgrounds (handled by AI background removal)
- Clear visibility throughout the video
- Minimum ~10 usable frames after filtering
