# Video-to-3D

A production-ready pipeline that reconstructs a textured 3D GLB asset from a product video.

## Overview

The pipeline takes a single product video as input and produces a self-contained `.glb` file with embedded textures, ready for use in web viewers and AR environments.

### Processing Stages

1. **Frame Extraction** – samples the video at a configurable frame rate using OpenCV
2. **Frame Filtering** – removes blurry frames (Laplacian variance) and near-duplicates
3. **3D Reconstruction** – runs COLMAP feature extraction, matching, SfM, and dense MVS via `pycolmap`
4. **Mesh Processing** – Poisson surface reconstruction, mesh cleanup, quadric decimation, normal smoothing, and scale normalisation
5. **Texture Baking** – UV-unwraps the mesh with `xatlas` and bakes colours from source images
6. **GLB Export** – packages geometry, normals, UVs, and compressed textures into a self-contained `.glb`

## Requirements

- Python 3.9+
- See `requirements.txt` for Python dependencies

```bash
pip install -r requirements.txt
```

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
                   [--texture-format {jpeg,png}] [--no-dense] [--gpu]
                   [--work-dir WORK_DIR] [-v]
                   input_video

positional arguments:
  input_video           Path to the input video file.

optional arguments:
  -o OUTPUT             Output .glb path (default: output.glb)
  --fps FPS             Frame sampling rate (default: 2.0)
  --max-frames N        Maximum frames to extract (default: 150)
  --blur-threshold T    Laplacian variance blur threshold (default: 80.0)
  --no-dense            Skip dense MVS; use sparse point cloud only
  --gpu                 Use GPU in COLMAP feature extraction
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

- Static object captured from multiple angles
- Not transparent or highly reflective (MVP)
- Clear visibility throughout the video
- Minimum ~10 usable frames after filtering
