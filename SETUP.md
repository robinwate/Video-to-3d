# Setup Guide — Video-to-3D Pipeline

This guide lists every tool, application, and command you need to install and
run the pipeline on your own computer (Windows, macOS, or Linux).

---

## 1. Prerequisites

### Python 3.9 or newer

The pipeline requires Python 3.9+. Python 3.10, 3.11, or 3.12 are all fine.

**Check whether Python is already installed:**

```bash
python --version
# or on Linux/macOS:
python3 --version
```

**Install Python if needed:**

- **Windows** – download the installer from https://www.python.org/downloads/
  - During installation **tick "Add Python to PATH"**
- **macOS** – `brew install python` (requires Homebrew: https://brew.sh)
- **Ubuntu/Debian** – `sudo apt install python3 python3-pip python3-venv`

---

## 2. Create a virtual environment (strongly recommended)

A virtual environment keeps the pipeline's dependencies isolated from the rest
of your system.

```bash
# Navigate to the project directory first
cd C:\Dev\immersive-ar\video23d\Video-to-3d   # Windows example
# or
cd /path/to/Video-to-3d                        # macOS / Linux

# Create the virtual environment
python -m venv .venv

# Activate it
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal prompt after activation.

---

## 3. Install Python dependencies

With the virtual environment active, run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all required packages, including:

| Package | Purpose |
|---------|---------|
| `opencv-python-headless` | Video decoding, frame extraction, image processing |
| `numpy` | Array operations throughout the pipeline |
| `open3d` | Point cloud processing, Poisson surface reconstruction |
| `xatlas` | UV unwrapping (atlas generation) |
| `pycolmap` | COLMAP bindings — SIFT features, SfM, dense MVS |
| `scipy` | Scientific utilities |
| `Pillow` | Image loading/saving |
| `pygltflib` | (Optional) glTF validation utilities |
| `trimesh` | (Optional) mesh I/O utilities |

> **Note — GPU support:** `pycolmap` includes CPU-only binaries by default.
> Dense MVS (`patch_match_stereo`) requires CUDA. On CPU-only machines the
> pipeline automatically falls back to the sparse point cloud — this is safe
> and the `--no-dense` flag skips dense MVS entirely.

---

## 4. Windows-specific notes

### Microsoft Visual C++ Redistributable

Some Python wheels (especially `open3d` and `pycolmap`) require the
**Visual C++ Redistributable** runtime. If you see a DLL error, install it:

- https://aka.ms/vs/17/release/vc_redist.x64.exe

### PowerShell execution policy

If `.venv\Scripts\Activate.ps1` is blocked, run once as Administrator:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 5. Verify the installation

Run the built-in tests to confirm everything is working:

```bash
pip install pytest
python -m pytest tests/ -v
```

Expected output: **44 passed** (no failures).

---

## 6. Run the pipeline

```bash
python main.py apple.mp4 -o output.glb
```

### Useful flags

| Flag | Default | Description |
|------|---------|-------------|
| `-o PATH` | `output.glb` | Output GLB file path |
| `--fps N` | `2.0` | Frames per second to sample from the video |
| `--max-frames N` | `150` | Maximum number of frames to extract |
| `--no-dense` | off | Skip dense MVS (much faster; use on CPU-only machines) |
| `--max-triangles N` | `50000` | Target triangle count after simplification |
| `--texture-size N` | `1024` | Texture atlas size (256 / 512 / 1024 / 2048) |
| `-v` | off | Enable verbose logging |

**Recommended command for a first test (fast, CPU-only):**

```bash
python main.py apple.mp4 -o output.glb --no-dense -v
```

---

## 7. Troubleshooting common errors

### `ModuleNotFoundError: No module named 'cv2'`

The virtual environment is not active, or dependencies were not installed:

```bash
# Activate the venv first (see Section 2), then:
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'pipeline'`

Run the script from the project root directory, or use the full path:

```bash
cd C:\Dev\immersive-ar\video23d\Video-to-3d
python main.py apple.mp4 -o output.glb
```

### `AttributeError: 'pycolmap.SiftExtractionOptions' has no attribute 'use_gpu'`

Your pycolmap version is < 3.0. Upgrade it:

```bash
pip install --upgrade pycolmap
```

### `RuntimeError: SfM failed: no reconstructions produced`

The input video does not have enough overlap between frames. Try:

```bash
python main.py video.mp4 -o output.glb --fps 3 --no-dense
```

### `DLL load failed` (Windows)

Install the Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe

### `open3d` crashes silently on import (macOS Apple Silicon)

Use `open3d` 0.18+ which ships with ARM64 wheels:

```bash
pip install --upgrade open3d
```

---

## 8. Full dependency version reference

The pipeline was tested with the following versions:

| Package | Tested version |
|---------|---------------|
| Python | 3.12.3 |
| opencv-python-headless | 4.13.0 |
| numpy | 2.4.2 |
| open3d | 0.19.0 |
| xatlas | 0.0.11 |
| pycolmap | 3.13.0 |
| scipy | 1.17.0 |
| Pillow | 12.1.0 |

Any version that meets the `>=` constraints in `requirements.txt` should work.
