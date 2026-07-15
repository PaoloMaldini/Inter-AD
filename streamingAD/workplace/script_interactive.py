#!/usr/bin/env python3
"""
script_interactive.py — Interactive AD: Script-Based + User Instruction Insertion
===================================================================================

Based on pipeline_script_based.py. Uses time-aligned xlsx script for:
  1. Dialogue gap detection (same as baseline)
  2. Rich context building (characters, location, dialogue)
  3. Face gallery (PlotTree)
  4. AD generation via Video-LLaMA

Key addition: inserts user instructions at random gap positions.
  - Instructions accumulate: after an insertion, all subsequent gaps
    are influenced by the active instruction set.
  - Supports category filtering (--categories) or full random.

Output: _ad_output.json (baseline-compatible format) + insertion metadata.

Usage:
    conda activate videollava
    python streamingAD/workplace/script_interactive.py \
        --num-insertions 5 --gpu-id 3

    # Category-filtered:
    python streamingAD/workplace/script_interactive.py \
        --categories style detail --num-insertions 3 --gpu-id 3

    # Single movie:
    python streamingAD/workplace/script_interactive.py \
        --only-movie "Shawshank" --gpu-id 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STREAMING_ROOT))

from pipeline_script_based import (
    match_videos_to_xlsxs,
    parse_dialogue_rows,
    detect_dialogue_gaps,
    build_gap_context,
    extract_clip,
    _prepare_fast_video,
    _sec_to_timestamp,
    TASK_PROMPT,
    VIDEO_DIRS,
    XLSX_DIRS,
    GT_CSV,
    DEFAULT_GAP_SEC,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_GPU,
)
from ad_engine import build_ad_engine

# ── Instruction categories ────────────────────────────────────
# Inline definition so this script is self-contained.
# Each category: id, name, templates (EN only for now), weight.

INSTRUCTION_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "style": {
        "name": "Narrative Style",
        "templates": [
            "Describe in a poetic, literary style",
            "Use a suspenseful, thriller-like tone",
            "Describe in a calm, contemplative manner",
            "Use dramatic, cinematic language",
        ],
        "weight": 1.0,
    },
    "detail": {
        "name": "Detail Focus",
        "templates": [
            "Focus heavily on visual details like colors, lighting, and textures",
            "Describe the spatial layout and positioning of objects and characters",
            "Pay special attention to the main character's body language and gestures in detail",
            "Describe the background environment and atmosphere in detail",
        ],
        "weight": 1.0,
    },
    "character": {
        "name": "Character Focus",
        "templates": [
            "Pay special attention to the main character's emotional state",
            "Describe the main character's body language and gestures in detail",
            "Focus on the relationships and interactions between characters",
            "Describe how the main character's expression changes during this scene",
        ],
        "weight": 1.0,
    },
    "audio": {
        "name": "Audio Focus",
        "templates": [
            "Describe the ambient sounds and music that would be playing",
            "Focus on the emotional tone that the background music conveys",
            "Describe the soundscape of this scene in detail",
            "Pay attention to the rhythm and pacing implied by the visuals",
        ],
        "weight": 1.0,
    },
    "narrative": {
        "name": "Narrative Context",
        "templates": [
            "Describe what just happened leading up to this moment",
            "Explain what is likely to happen next based on the visual cues",
            "Provide context about why the characters might be in this situation",
            "Describe the significance of this scene in the overall story",
        ],
        "weight": 1.0,
    },
}

INSTRUCTION_CONFIG_PATH = STREAMING_ROOT / "instruction_categories.json"


def load_instruction_categories(config_path: Optional[Path] = None,
                                filter_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Load categories from JSON file or use built-in defaults."""
    path = config_path or INSTRUCTION_CONFIG_PATH
    cats: Dict[str, Dict[str, Any]] = {}

    if path.exists():
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            for cat in data.get("categories", []):
                cid = cat.get("category_id", "")
                if not cid:
                    continue
                cats[cid] = {
                    "name": cat.get("name", cid),
                    "templates": cat.get("templates", []),
                    "weight": cat.get("weight", 1.0),
                }
        except Exception:
            pass

    if not cats:
        cats = dict(INSTRUCTION_CATEGORIES)

    if filter_ids:
        cats = {k: v for k, v in cats.items() if k in filter_ids}

    return cats


def sample_instruction(categories: Dict[str, Dict[str, Any]],
                       rng: random.Random) -> Tuple[str, str]:
    """Sample one instruction from categories. Returns (template, category_id)."""
    if not categories:
        return "Describe what is happening in this clip in detail.", "default"

    # Weighted category selection
    ids = list(categories.keys())
    weights = [categories[cid].get("weight", 1.0) for cid in ids]
    cat_id = rng.choices(ids, weights=weights, k=1)[0]

    templates = categories[cat_id].get("templates", [])
    if not templates:
        return "Describe what is happening in this clip in detail.", cat_id

    template = rng.choice(templates)
    return template, cat_id


def select_insertion_gaps(num_gaps: int, num_insertions: int,
                          rng: random.Random) -> List[int]:
    """Select random gap indices for instruction insertion (sorted)."""
    # Insert at least at gap 0 or 1, leave room for before/after comparison
    max_idx = max(num_gaps - 1, 0)
    if num_insertions >= num_gaps:
        # Insert at every gap
        return list(range(num_gaps))
    chosen = sorted(rng.sample(range(max(1, num_gaps)), min(num_insertions, num_gaps)))
    return chosen


def run_one_movie(
    engine,
    video_path: Path,
    xlsx_path: Path,
    output_dir: Path,
    categories: Dict[str, Dict[str, Any]],
    num_insertions: int,
    rng: random.Random,
    gap_threshold_sec: float,
    temperature: float,
    max_new_tokens: int,
    gpu_id: int,
) -> Dict[str, Any]:
    """Run interactive experiment on one movie (script-based)."""
    movie_name = video_path.stem
    print(f"\n{'='*60}")
    print(f"[Script-Interactive] {movie_name}")
    print(f"  Video: {video_path}")
    print(f"  Script: {xlsx_path}")
    print(f"  Insertions: {num_insertions}")
    print(f"  Categories: {list(categories.keys()) if categories else 'ALL'}")
    print(f"{'='*60}")

    t_start = time.monotonic()

    # Parse script & detect gaps
    rows = parse_dialogue_rows(xlsx_path)
    if not rows:
        return {"movie": movie_name, "status": "no_dialogue", "ad_entries": []}

    candidates = detect_dialogue_gaps(rows, gap_threshold_sec)
    if not candidates:
        return {"movie": movie_name, "status": "no_gaps", "ad_entries": []}

    num_gaps = len(candidates)
    print(f"  {len(rows)} dialogue rows, {num_gaps} gaps")

    # Face gallery
    from face_gallery import load_gallery, lookup_face_images
    gallery_embs, gallery_people = load_gallery(movie_name)
    if gallery_embs is not None:
        print(f"  Face gallery: {len(gallery_people)} characters")
    else:
        print(f"  Face gallery: not available")

    # Select insertion points
    actual_insertions = min(num_insertions, num_gaps)
    insertion_indices = select_insertion_gaps(num_gaps, actual_insertions, rng)
    insertion_set = set(insertion_indices)

    print(f"  Insertion gaps: {insertion_indices}")

    # Prepare clips
    clip_tmp = output_dir / f".tmp_{movie_name}"
    clip_tmp.mkdir(parents=True, exist_ok=True)
    fast_video = _prepare_fast_video(video_path, clip_tmp)

    # Track active instructions (accumulating)
    active_instructions: List[Dict[str, str]] = []
    entries: List[Dict[str, Any]] = []
    insertion_events: List[Dict[str, Any]] = []
    inference_total = 0.0
    face_avatar_cache: Dict[str, List[Path]] = {}

    for idx, cand in enumerate(candidates):
        # Insert instruction if this gap is an insertion point
        if idx in insertion_set:
            template, cat_id = sample_instruction(categories, rng)
            active_instructions.append({
                "template": template,
                "category": cat_id,
                "gap_idx": idx,
            })
            insertion_events.append({
                "insertion_id": len(insertion_events) + 1,
                "gap_idx": idx,
                "timestamp_sec": round(cand.gap_start_sec, 3),
                "category": cat_id,
                "instruction_text": template,
            })
            print(f"\n  *** INSERT #{len(insertion_events)} at gap {idx}: "
                  f"[{cat_id}] {template}")

        # Extract clip
        clip_path = clip_tmp / f"gap{cand.gap_id:04d}.mp4"
        t0 = time.monotonic()
        if not extract_clip(fast_video, cand.gap_start_sec, cand.gap_end_sec, clip_path):
            continue
        extract_time = time.monotonic() - t0

        # Build context
        context_text = build_gap_context(cand)

        # Add active instructions to context
        if active_instructions:
            instr_block = "\n[User instructions - follow ALL of these]\n"
            for i, ai in enumerate(active_instructions, 1):
                instr_block += f"  {i}. {ai['template']}\n"
            context_text = instr_block + "\n" + context_text

        # Face gallery
        face_avatars: List[Path] = []
        if gallery_people is not None and cand.characters:
            cache_key = "|".join(sorted(cand.characters))
            if cache_key not in face_avatar_cache:
                face_avatar_cache[cache_key] = lookup_face_images(
                    cand.characters, gallery_people, movie_name,
                )
            face_avatars = face_avatar_cache[cache_key]

        # Generate AD
        try:
            ad_text, inf_time, _ = engine.infer_one_segment(
                clip_path=clip_path, context_text=context_text,
                task_prompt=TASK_PROMPT, temperature=temperature,
                max_new_tokens=max_new_tokens,
                face_avatars=face_avatars, character_names=cand.characters,
            )
        except Exception as e:
            ad_text = f"[ERROR: {e}]"
            inf_time = 0.0
        inference_total += inf_time

        ctx_before = cand.context_before[-5:] if cand.context_before else []
        ctx_after = cand.context_after[:8] if cand.context_after else []

        entries.append({
            "gap_id": cand.gap_id,
            "scene_index": cand.scene_index,
            "location": cand.location,
            "gap_start_sec": round(cand.gap_start_sec, 3),
            "gap_end_sec": round(cand.gap_end_sec, 3),
            "gap_duration_sec": round(cand.gap_duration_sec, 3),
            "characters": cand.characters,
            "context_before": ctx_before,
            "context_after": ctx_after,
            "ad_text": ad_text,
            "inference_time_sec": round(inf_time, 3),
            "active_instructions": list(active_instructions),
            "active_instruction_count": len(active_instructions),
        })

        print(f"  [{idx+1}/{num_gaps}] Gap {cand.gap_id} ({cand.gap_duration_sec:.1f}s, "
              f"instr={len(active_instructions)}): {ad_text[:80]}...")

        if clip_path.exists():
            clip_path.unlink()

    shutil.rmtree(str(clip_tmp), ignore_errors=True)

    t_total = time.monotonic() - t_start

    video_duration_sec = 0.0
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        video_duration_sec = float(result.stdout.strip())
    except Exception:
        pass

    result = {
        "movie": movie_name,
        "movie_title": movie_name,
        "method": "script_interactive",
        "video_path": str(video_path),
        "xlsx_path": str(xlsx_path),
        "video_duration_sec": video_duration_sec,
        "gap_threshold_sec": gap_threshold_sec,
        "total_gaps": num_gaps,
        "generated_count": len(entries),
        "num_insertions": len(insertion_events),
        "inference_total_time_sec": round(inference_total, 1),
        "total_time_sec": round(t_total, 1),
        "ad_entries": entries,
        "insertion_events": insertion_events,
        "categories_used": list(set(ie["category"] for ie in insertion_events)),
    }

    # Save outputs
    safe_name = movie_name.replace(" ", "_").replace("/", "_")[:60]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    ad_file = output_dir / f"{safe_name}_ad_output.json"
    with ad_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {ad_file}")
    print(f"  {len(entries)} ADs, {len(insertion_events)} insertions, "
          f"time={t_total:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Interactive AD: Script-Based + Instruction Insertion")
    parser.add_argument("--video-dir", nargs="*", default=VIDEO_DIRS)
    parser.add_argument("--xlsx-dir", nargs="*", default=XLSX_DIRS)
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: experiment_results/script_interactive_<ts>)")
    parser.add_argument("--gap-threshold", type=float, default=DEFAULT_GAP_SEC)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU)
    parser.add_argument("--only-movie", default=None)
    parser.add_argument("--no-gt-filter", action="store_true")

    # Instruction config
    parser.add_argument("--num-insertions", type=int, default=None,
                        help="Fixed number of insertions per movie (default: random 1-10)")
    parser.add_argument("--min-insertions", type=int, default=1,
                        help="Min insertions for random range (default: 1)")
    parser.add_argument("--max-insertions", type=int, default=10,
                        help="Max insertions for random range (default: 10)")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Category IDs to use (default: all). E.g.: style detail character")
    parser.add_argument("--instruction-config", default=None,
                        help="Path to instruction categories JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "experiment_results" / f"script_interactive_{ts}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    categories = load_instruction_categories(
        Path(args.instruction_config) if args.instruction_config else None,
        filter_ids=args.categories,
    )

    gt_csv = None if args.no_gt_filter else GT_CSV
    pairs = match_videos_to_xlsxs(args.video_dir, args.xlsx_dir,
                                   gt_csv=gt_csv, only_movie=args.only_movie)
    if not pairs:
        print("ERROR: No video-xlsx pairs found.")
        sys.exit(1)

    print("=" * 60)
    print("Interactive AD: Script-Based")
    print(f"  Movies:    {len(pairs)}")
    print(f"  Insertions: {'fixed ' + str(args.num_insertions) if args.num_insertions else f'random {args.min_insertions}-{args.max_insertions}'}")
    print(f"  Categories: {list(categories.keys())}")
    print(f"  Seed:      {args.seed}")
    print(f"  GPU:       {args.gpu_id}")
    print(f"  Output:    {output_dir}")
    print("=" * 60)

    print("\nLoading AD Engine...")
    engine = build_ad_engine(gpu_id=args.gpu_id)
    print("Engine ready.\n")

    all_results = []
    for vi, (vp, xp) in enumerate(pairs):
        n_ins = args.num_insertions if args.num_insertions else rng.randint(
            args.min_insertions, args.max_insertions)

        print(f"\n{'#'*60}")
        print(f"MOVIE {vi+1}/{len(pairs)}: {vp.stem} (insertions={n_ins})")
        print(f"{'#'*60}")

        try:
            r = run_one_movie(
                engine=engine, video_path=vp, xlsx_path=xp, output_dir=output_dir,
                categories=categories, num_insertions=n_ins, rng=rng,
                gap_threshold_sec=args.gap_threshold,
                temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                gpu_id=args.gpu_id,
            )
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"movie": vp.stem, "status": "error", "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        status = "OK" if r.get("generated_count", 0) > 0 else "FAIL"
        n = r.get("generated_count", 0)
        ins = r.get("num_insertions", 0)
        print(f"  {status} {r.get('movie', '?')}: {n} ADs, {ins} insertions")

    print(f"\nOutput: {output_dir}")
    print(f"\nNext: python streamingAD/run_experiment_eval.py "
          f"--baseline-dir <baseline_dir> --instructed-dir {output_dir}")


if __name__ == "__main__":
    main()
