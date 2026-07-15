#!/usr/bin/env python3
"""
Pipeline for running our method on DiDeMo benchmark.

DiDeMo task: given a short video (<30s), generate a natural language description.

Usage:
    conda activate videollava && python streamingAD/pipeline_didemo.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# ── Default Config ──────────────────────────────────────────

DEFAULT_GPU         = 0
DEFAULT_SPLIT       = "test"
DEFAULT_MAX_CLIPS   = 0               # 0 = all
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS  = 128

# ── Paths ───────────────────────────────────────────────────

STREAMING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT   = STREAMING_ROOT.parent

DIDEMO_DIR     = PROJECT_ROOT / "datasets" / "didemo"
VIDEOS_DIR     = DIDEMO_DIR / "videos"

OUTPUT_DIR     = PROJECT_ROOT / "compare" / "didemo"

TASK_PROMPT = (
    "Describe what is happening in this video concisely. "
    "Focus on visible actions, movements, and expressions. "
    "If people are visible, describe their appearance and actions."
)


# ── Helpers ─────────────────────────────────────────────────

def load_split(split: str) -> list:
    json_path = DIDEMO_DIR / f"{split}.json"
    if not json_path.exists():
        print(f"  [WARN] {json_path} not found")
        return []
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def find_video_path(video_field: str, split: str) -> Optional[Path]:
    """video_field is like 'test/52198061@N00_4175539607_26669cd634.mp4'"""
    filename = video_field.split("/")[-1]
    path = VIDEOS_DIR / split / filename
    if path.exists() and path.stat().st_size > 1000:
        return path
    return None


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--max-clips", type=int, default=DEFAULT_MAX_CLIPS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(STREAMING_ROOT))

    print(f"\n{'='*60}")
    print(f"[DiDeMo Pipeline] split={args.split}")
    print(f"  Data dir: {DIDEMO_DIR}")
    print(f"  Output:   {OUTPUT_DIR}")
    print(f"{'='*60}")

    # ── Load data ───────────────────────────────────────────
    print("\n[Phase 1] Loading data...")
    samples = load_split(args.split)
    print(f"  Total samples: {len(samples)}")

    # ── Find available videos ───────────────────────────────
    print("[Phase 2] Finding available videos...")
    entries = []
    for i, sample in enumerate(samples):
        video_path = find_video_path(sample["video"], args.split)
        if video_path is not None:
            entries.append((i, sample, video_path))

    if args.max_clips > 0:
        entries = entries[:args.max_clips]

    print(f"  Available: {len(entries)}/{len(samples)}")
    if not entries:
        print("  No videos found.")
        return

    # ── Build AD engine ─────────────────────────────────────
    print("\n[Phase 3] Building AD engine...")
    from ad_engine import build_ad_engine
    engine = build_ad_engine(gpu_id=0)
    print()

    # ── Inference ───────────────────────────────────────────
    print(f"[Phase 4] Running inference on {len(entries)} videos...")
    results = []
    total_time = 0.0

    for i, (idx, sample, video_path) in enumerate(entries):
        gt_caption = sample.get("caption", "")

        try:
            ad_text, inf_time, _ = engine.infer_one_segment(
                clip_path=video_path,
                context_text=f"[Task] {TASK_PROMPT}",
                task_prompt=TASK_PROMPT,
                temperature=args.temperature,
                max_new_tokens=args.max_tokens,
            )
            total_time += inf_time
        except Exception as e:
            ad_text = ""
            inf_time = 0.0
            print(f"    [ERROR] clip {i}: {e}")

        results.append({
            "clip_idx": idx,
            "video": sample["video"],
            "gt_caption": gt_caption,
            "generated_text": ad_text,
            "inference_time_sec": round(inf_time, 3),
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(entries):
            print(f"  [{i + 1}/{len(entries)}] "
                  f"avg_time={total_time / (i + 1):.2f}s")

    # ── Save results ────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "didemo_output.json"

    output = {
        "method": "pipeline_didemo",
        "dataset": "DiDeMo",
        "split": args.split,
        "total_samples": len(samples),
        "available_videos": len(entries),
        "generated_count": len(results),
        "inference_total_time_sec": round(total_time, 1),
        "total_time_sec": round(total_time, 1),
        "time_per_video_sec": round(total_time / max(len(results), 1), 2),
        "task_prompt": TASK_PROMPT,
        "ad_entries": [
            {
                "gap_id": r["clip_idx"],
                "ad_text": r["generated_text"],
                "gt_caption": r["gt_caption"],
                "video": r["video"],
                "gap_duration_sec": 0,
                "scene_index": "",
                "location": "",
                "characters": [],
                "context_before": [],
                "context_after": [],
                "preprocess_time_sec": 0,
                "inference_total_time_sec": r["inference_time_sec"],
                "total_time_sec": r["inference_time_sec"],
            }
            for r in results
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: {output_file}")
    print(f"  Generated: {len(results)}/{len(entries)}")
    print(f"  Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
