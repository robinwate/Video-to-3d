#!/usr/bin/env python3
"""Command-line interface for the Video-to-3D pipeline."""

import argparse
import logging
import os
import sys


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
        default=80.0,
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
        "--no-dense",
        action="store_true",
        help="Skip dense MVS reconstruction (faster, lower quality).",
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration in COLMAP feature extraction.",
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

    from pipeline import Pipeline
    from pipeline.pipeline import PipelineConfig

    config = PipelineConfig(
        target_fps=args.fps,
        max_frames=args.max_frames,
        blur_threshold=args.blur_threshold,
        similarity_threshold=args.similarity_threshold,
        max_triangles=args.max_triangles,
        texture_size=args.texture_size,
        texture_format=args.texture_format,
        dense_reconstruction=not args.no_dense,
        use_gpu=args.gpu,
    )

    pipeline = Pipeline(config=config, work_dir=args.work_dir)

    try:
        result = pipeline.run(args.input_video, args.output)
        print(f"\n✓ Done in {result.elapsed_seconds:.1f}s")
        print(f"  Frames extracted : {result.num_frames_extracted}")
        print(f"  Frames used      : {result.num_frames_used}")
        print(f"  Point cloud size : {result.num_points:,}")
        print(f"  Mesh triangles   : {result.num_triangles:,}")
        print(f"  Output           : {result.glb_path}")
        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  • {w}")
        return 0
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("Pipeline failed: %s", exc)
        return 2
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
