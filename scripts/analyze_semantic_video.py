"""Command-line entry point for canonical semantic five-pillar inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene_motion_llm.inference.semantic_five_pillar_inference import (
    SemanticFivePillarInferencer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse one <=15-second video with the LIRIS semantic five-pillar model."
    )
    parser.add_argument("video", type=Path, help="Path to an uploaded short video.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Canonical semantic checkpoint.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or omit for automatic selection.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for a JSON-safe full report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inferencer = SemanticFivePillarInferencer(args.checkpoint, device=args.device)
    result = inferencer.analyze_video(args.video)
    print(result["report"])
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nFull JSON report written to: {args.output_json}")


if __name__ == "__main__":
    main()
