#!/usr/bin/env python3
"""Command-line interface for the Video-to-3D pipeline."""

import argparse
import logging
import os
import sys

# Ensure the pipeline package is importable regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure stdout to UTF-8 on Windows so Unicode characters don't crash.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video-to-3d",
        description=(
            "Reconstruct a production-ready textured 3-D GLB asset "
            "from a product video."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_video", help="Path to the input video file.")
    p.add_argument(
        "-o",
        "--output",
        default="output.glb",
        help="Path for the output .glb file.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frame sampling rate (frames per second of video to extract).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Maximum number of frames to extract from the video.",
    )
    p.add_argument(
        "--blur-threshold",
        type=float,
        default=15.0,
        help="Laplacian variance threshold for blur detection.",
    )
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=8.0,
        help="Pixel difference threshold for duplicate frame removal.",
    )
    p.add_argument(
        "--max-triangles",
        type=int,
        default=50_000,
        help="Target triangle count after mesh simplification.",
    )
    p.add_argument(
        "--texture-size",
        type=int,
        default=1024,
        choices=[256, 512, 1024, 2048],
        help="Texture atlas side length in pixels.",
    )
    p.add_argument(
        "--texture-format",
        default="jpeg",
        choices=["jpeg", "png"],
        help="Embedded texture image format.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device for AI models (cpu or cuda).",
    )
    p.add_argument(
        "--bg-model",
        default="u2net",
        help="rembg model for AI background removal (e.g. u2net, u2netp).",
    )
    p.add_argument(
        "--depth-model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="HuggingFace model for depth estimation.",
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help="Working directory for intermediate files.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    logger = logging.getLogger("video_to_3d")

    if not os.path.exists(args.input_video):
        logger.error("Input video not found: %s", args.input_video)
        return 1

    try:
        from pipeline import Pipeline
        from pipeline.pipeline import PipelineConfig
    except ImportError as exc:
        logger.error(
            "Failed to import pipeline package: %s\n"
            "Ensure all dependencies are installed: pip install -r requirements.txt",
            exc,
        )
        return 1

    config = PipelineConfig(
        target_fps=args.fps,
        max_frames=args.max_frames,
        blur_threshold=args.blur_threshold,
        similarity_threshold=args.similarity_threshold,
        max_triangles=args.max_triangles,
        texture_size=args.texture_size,
        texture_format=args.texture_format,
        device=args.device,
        bg_removal_model=args.bg_model,
        depth_model=args.depth_model,
    )

    pipeline = Pipeline(config=config, work_dir=args.work_dir)

    try:
        result = pipeline.run(args.input_video, args.output)
        print("\nDone in {:.1f}s".format(result.elapsed_seconds))
        print("  Frames extracted : {}".format(result.num_frames_extracted))
        print("  Frames used      : {}".format(result.num_frames_used))
        print("  Point cloud size : {:,}".format(result.num_points))
        print("  Mesh triangles   : {:,}".format(result.num_triangles))
        print("  Output           : {}".format(result.glb_path))
        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print("  - {}".format(w))
        return 0
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except ImportError as exc:
        logger.error(
            "Missing dependency: %s\n"
            "Install all requirements with: pip install -r requirements.txt",
            exc,
        )
        return 1
    except RuntimeError as exc:
        logger.error("Pipeline failed: %s", exc)
        return 2
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
