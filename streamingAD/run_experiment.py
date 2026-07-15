#!/usr/bin/env python3
"""
run_experiment.py — Run interactive AD experiments with instruction insertions.

Simulates streaming video playback where user instructions are inserted at
random timestamps. For each insertion, generates AD before/after the instruction.
Instructions persist within a movie session (reset between movies).

Output: JSON file with insertion records (before/after texts, timing, metadata)

Usage:
    # Single movie:
    conda activate videollava
    python streamingAD/run_experiment.py \
        --movie "The Shawshank Redemption" \
        --num-insertions 5 \
        --gpu 2

    # Multiple movies:
    python streamingAD/run_experiment.py \
        --movie "The Shawshank Redemption" "The Godfather" "Harry Potter 1" \
        --num-insertions 3 4 5 \
        --gpu 2

    # All available movies, random 1-10 insertions each:
    python streamingAD/run_experiment.py --all-movies --gpu 2

    # Show/save instruction categories:
    python streamingAD/run_experiment.py --show-categories
    python streamingAD/run_experiment.py --save-default-config
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project paths & env vars (cache locally, not on system disk) ─────────────
STREAMING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STREAMING_ROOT.parent

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".hf_cache" / "sentence_transformers"))
os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT_ROOT / ".pip_cache"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch_cache"))

if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

from interactive_experiment import (
    InteractiveExperiment,
    load_instruction_categories,
    save_instruction_categories,
    save_experiment_result,
    DEFAULT_CATEGORIES,
    INSTRUCTION_CONFIG_PATH,
    FINAL_BY_MOVIE_DIR,
    AD_CLIPS_DIR,
)
from segment_db import scan_available_movies
from ad_engine import build_ad_engine

# ── Output directory ──────────────────────────────────────────────────────────
EXPERIMENT_OUTPUT_DIR = PROJECT_ROOT / "experiment_results"


def run_single_movie(
    experiment: InteractiveExperiment,
    movie_title: str,
    num_insertions: int = 3,
    insertion_strategy: str = "random",
    temperature: float = 0.2,
    seed: int = 42,
) -> Path:
    """Run experiment on one movie, save and return the JSON path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = movie_title.replace(" ", "_").replace("/", "_")[:60]
    output_path = EXPERIMENT_OUTPUT_DIR / f"{safe_name}_experiment_{timestamp}.json"

    result = experiment.run_movie_experiment(
        movie_title=movie_title,
        num_insertions=num_insertions,
        insertion_strategy=insertion_strategy,
        temperature=temperature,
        seed=seed,
    )
    save_experiment_result(result, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run interactive AD experiments with instruction insertions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single movie with 5 insertions:
  python streamingAD/run_experiment.py --movie "The Shawshank Redemption" --num-insertions 5 --gpu 2

  # Multiple movies:
  python streamingAD/run_experiment.py --movie "The Shawshank" "The Godfather" --num-insertions 3 4 --gpu 2

  # Manual timestamps:
  python streamingAD/run_experiment.py --movie "The Shawshank" --strategy manual --timestamps 120.5 300.0 600.0 --gpu 2
        """
    )

    # Required args
    parser.add_argument("--movie", nargs="+", default=None,
                        help="Movie title(s) to process")
    parser.add_argument("--all-movies", action="store_true",
                        help="Auto-discover all available movies from dataset (overrides --movie)")

    # Experiment config
    parser.add_argument("--num-insertions", nargs="+", type=int, default=None,
                        help="Number of instruction insertions per movie (default: 3)")
    parser.add_argument("--min-insertions", type=int, default=None,
                        help="Min insertions for random range (default: 1)")
    parser.add_argument("--max-insertions", type=int, default=None,
                        help="Max insertions for random range (default: 10)")
    parser.add_argument("--strategy", default="random",
                        choices=["random", "uniform", "manual"],
                        help="Instruction insertion strategy (default: random)")
    parser.add_argument("--timestamps", nargs="+", type=float, default=None,
                        help="Manual timestamps in seconds (required for --strategy manual)")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Specific category IDs to use (default: all)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Generation temperature (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    # Model config
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id (default: 0)")
    parser.add_argument("--instruction-config", type=str, default=None,
                        help="Path to instruction categories JSON config")

    # Category management
    parser.add_argument("--show-categories", action="store_true",
                        help="Show instruction categories and exit")
    parser.add_argument("--save-default-config", action="store_true",
                        help="Save default instruction categories config and exit")

    args = parser.parse_args()

    # ── Category management ───────────────────────────────────────────────
    if args.show_categories:
        categories = load_instruction_categories(
            Path(args.instruction_config) if args.instruction_config else None
        )
        print(f"\nInstruction Categories ({len(categories)}):")
        print("-" * 60)
        for cat in categories:
            print(f"\n  [{cat.category_id}] {cat.name} ({cat.name_cn})")
            print(f"    {cat.description}")
            print(f"    Weight: {cat.weight}")
            print(f"    Templates ({len(cat.templates)}):")
            for i, (en, cn) in enumerate(zip(cat.templates, cat.templates_cn)):
                print(f"      {i+1}. EN: {en}")
                print(f"         CN: {cn}")
        return

    if args.save_default_config:
        save_instruction_categories(DEFAULT_CATEGORIES, INSTRUCTION_CONFIG_PATH)
        print(f"\nSaved default config to {INSTRUCTION_CONFIG_PATH}")
        print("Edit this JSON file to add/modify/remove instruction categories.")
        print("\nStructure:")
        print('  {')
        print('    "categories": [')
        print('      {')
        print('        "category_id": "my_category",')
        print('        "name": "My Category",')
        print('        "name_cn": "我的类别",')
        print('        "description": "Description",')
        print('        "templates": ["Template 1", "Template 2"],')
        print('        "templates_cn": ["模板1", "模板2"],')
        print('        "weight": 1.0')
        print('      }')
        print('    ]')
        print('  }')
        return

    # ── Validate args ─────────────────────────────────────────────────────
    if not args.movie and not args.all_movies:
        parser.error("--movie or --all-movies is required (or use --show-categories / --save-default-config)")

    if args.strategy == "manual" and not args.timestamps:
        parser.error("--timestamps required with --strategy manual")

    # ── Discover movies if --all-movies ───────────────────────────────────
    import random as _random
    if args.all_movies:
        available = scan_available_movies(seg_dir=FINAL_BY_MOVIE_DIR, clip_dir=AD_CLIPS_DIR)
        args.movie = sorted(available.keys())
        print(f"[discover] Found {len(args.movie)} available movies")

    if not args.movie:
        parser.error("No movies found. Use --movie to specify or --all-movies to auto-discover.")

    # ── Determine insertion counts ────────────────────────────────────────
    use_random_range = args.min_insertions is not None or args.max_insertions is not None
    if args.num_insertions is None and not use_random_range:
        # Default: fixed 3 insertions per movie
        args.num_insertions = [3]

    rng_ins = _random.Random(args.seed)

    def _get_num_insertions(idx: int) -> int:
        """Get insertion count for movie at index idx."""
        if use_random_range:
            lo = args.min_insertions or 1
            hi = args.max_insertions or 10
            return rng_ins.randint(lo, hi)
        if args.num_insertions is not None:
            return args.num_insertions[idx] if idx < len(args.num_insertions) else args.num_insertions[-1]
        return 3

    # ── Initialize engine ─────────────────────────────────────────────────
    print("=" * 60)
    print("INTERACTIVE AD EXPERIMENT")
    print("=" * 60)
    print(f"Movies: {len(args.movie)}")
    if use_random_range:
        print(f"Insertions per movie: random [{args.min_insertions or 1}, {args.max_insertions or 10}]")
    else:
        print(f"Insertions per movie: {args.num_insertions}")
    print(f"Strategy: {args.strategy}")
    print(f"GPU: {args.gpu}")
    print(f"Seed: {args.seed}")
    print()

    print("[init] Loading AD Engine...")
    engine = build_ad_engine(gpu_id=args.gpu)
    print("[init] AD Engine ready.\n")

    # ── Load categories ───────────────────────────────────────────────────
    config_path = Path(args.instruction_config) if args.instruction_config else None
    categories = load_instruction_categories(config_path)
    print(f"[config] Loaded {len(categories)} instruction categories:")
    for cat in categories:
        print(f"  - {cat.category_id}: {cat.name} ({len(cat.templates)} templates)")

    # ── Create experiment instance ────────────────────────────────────────
    experiment = InteractiveExperiment(engine=engine, categories=categories, gpu_id=args.gpu)

    # ── Run experiments ───────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []

    for i, movie in enumerate(args.movie):
        n_ins = _get_num_insertions(i)

        print(f"\n{'#' * 60}")
        print(f"MOVIE {i+1}/{len(args.movie)}: {movie}")
        print(f"  Insertions: {n_ins}")
        print(f"  Strategy: {args.strategy}")
        if args.strategy == "manual":
            print(f"  Timestamps: {args.timestamps}")
        print(f"{'#' * 60}")

        try:
            json_path = run_single_movie(
                experiment=experiment,
                movie_title=movie,
                num_insertions=n_ins,
                insertion_strategy=args.strategy,
                temperature=args.temperature,
                seed=args.seed,
            )

            results.append({
                "movie": movie,
                "json_path": str(json_path),
                "num_insertions": n_ins,
                "status": "success",
            })
            print(f"\n✓ Saved: {json_path}")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "movie": movie,
                "error": str(e),
                "status": "failed",
            })

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        status = "✓" if r["status"] == "success" else "✗"
        print(f"  {status} {r['movie']}: {r.get('json_path', r.get('error', 'N/A'))}")

    print(f"\nOutput directory: {EXPERIMENT_OUTPUT_DIR}")
    print("\nNext step: Run evaluation with:")
    print("  python streamingAD/run_eval.py --experiment-json <path_to_json>")


if __name__ == "__main__":
    main()
