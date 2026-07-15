#!/usr/bin/env python3
"""
Pipeline for running our method on CMD-AD benchmark.

CMD-AD task: given a pre-defined video clip, generate one AD sentence.
No VAD/Whisper needed — clips are already extracted and timestamps are given.

Usage:
    conda activate videollava && python streamingAD/pipeline_cmdad.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Default Config (edit here instead of CLI) ───────────────

DEFAULT_GPU         = 0
DEFAULT_SPLIT       = "eval"          # "eval" or "train" or "all"
DEFAULT_MAX_CLIPS   = 0               # 0 = all
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS  = 128

# ── Paths ───────────────────────────────────────────────────

STREAMING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT   = STREAMING_ROOT.parent

CMDAD_DIR      = PROJECT_ROOT / "datasets" / "cmdad"
ANNO_CSV       = CMDAD_DIR / "cmd_ad_anno_v1.csv"
CLIPS_DIR      = CMDAD_DIR / "clips"
CHARBANK_JSON  = CMDAD_DIR / "AutoAD-Zero" / "resources" / "charbanks" / "cmdad_charbank.json"
CLIPS_MANIFEST = CMDAD_DIR / "clip_manifest.json"

OUTPUT_DIR     = PROJECT_ROOT / "compare" / "cmdad"

TASK_PROMPT = (
    "Describe what is happening in this clip concisely. "
    "Focus on visible actions, movements, and expressions. "
    "If character names are mentioned in the context, use them "
    "(e.g. 'Don Vito Corleone walks...' not 'A man walks...'). "
    "Do not quote dialogue."
)


# ── Helpers ─────────────────────────────────────────────────

def load_charbank() -> Dict[str, list]:
    if CHARBANK_JSON.exists():
        with open(CHARBANK_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_clip_manifest() -> set:
    if CLIPS_MANIFEST.exists():
        with open(CLIPS_MANIFEST) as f:
            manifest = json.load(f)
        return {Path(m["file"]).stem for m in manifest}
    return set()


def build_imdb_to_movie_map() -> Dict[str, str]:
    """Build reverse mapping: IMDB ID -> PlotTree movie name."""
    name2imdb_path = STREAMING_ROOT / "face_gallery_data" / "name2imdbid.json"
    if not name2imdb_path.exists():
        return {}
    with open(name2imdb_path, encoding="utf-8") as f:
        n2i = json.load(f)
    return {v: k for k, v in n2i.items()}


def find_clip_path(row: pd.Series, manifest_stems: set) -> Optional[Path]:
    split = row["split"]
    videoid = row["cmd_filename"].split("/")[-1]
    start = float(row["scaled_start"])
    end = float(row["scaled_end"])
    filename = f"{videoid}_{start:.1f}_{end:.1f}.mp4"

    path = CLIPS_DIR / split / filename
    if path.exists() and path.stat().st_size > 500:
        return path
    return None


def build_context(row: pd.Series, charbank: dict, nearby_ads: List[str],
                  detected_characters: Optional[List[str]] = None) -> str:
    """Build context text for this clip."""
    parts = []

    # Movie and character info from charbank
    imdbid = str(row.get("imdbid", "")).strip()
    movie_title = str(row.get("movie_title", "")).strip()

    if imdbid and imdbid in charbank:
        chars = charbank[imdbid]
        char_names = [c["role"] for c in chars[:10] if c.get("role")]
        if char_names:
            parts.append(f"[Movie: {movie_title}]")
            parts.append(f"[Characters in movie: {', '.join(char_names)}]")

    # Detected characters in this specific clip
    if detected_characters:
        parts.append(f"[Characters visible in this clip: {', '.join(detected_characters)}]")

    # Nearby ADs as dialogue context
    if nearby_ads:
        parts.append("[Previous ADs for context:]")
        for ad in nearby_ads[-3:]:
            parts.append(f"  {ad}")

    parts.append(f"\n[Task] {TASK_PROMPT}")
    return "\n".join(parts)


def get_nearby_ads(ad_df: pd.DataFrame, movie_ads: pd.DataFrame,
                   current_idx: int, window: int = 3) -> List[str]:
    """Get AD texts from nearby clips in the same movie."""
    # Find position of current_idx in the movie's AD list
    movie_indices = movie_ads.index.tolist()
    try:
        pos = movie_indices.index(current_idx)
    except ValueError:
        return []

    start = max(0, pos - window)
    return [str(movie_ads.iloc[i]["text"]) for i in range(start, pos)
            if pd.notna(movie_ads.iloc[i]["text"])]


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
    print(f"[CMD-AD Pipeline] split={args.split}")
    print(f"  Annotations: {ANNO_CSV}")
    print(f"  Clips dir:   {CLIPS_DIR}")
    print(f"  Output:      {OUTPUT_DIR}")
    print(f"{'='*60}")

    # ── Load annotations ────────────────────────────────────
    print("\n[Phase 1] Loading annotations...")
    ad_df = pd.read_csv(ANNO_CSV)
    total_all = len(ad_df)

    if args.split != "all":
        ad_df = ad_df[ad_df["split"] == args.split]
    print(f"  Total ADs: {total_all}, filtered ({args.split}): {len(ad_df)}")

    # ── Load charbank ───────────────────────────────────────
    print("[Phase 2] Loading charbank...")
    charbank = load_charbank()
    print(f"  {len(charbank)} movies in charbank")

    # ── Load clip manifest ──────────────────────────────────
    print("[Phase 3] Loading clip manifest...")
    manifest_stems = load_clip_manifest()
    print(f"  {len(manifest_stems)} clips in manifest")

    # ── Find available clips ────────────────────────────────
    print("[Phase 4] Finding available clips...")
    entries = []
    for idx, row in ad_df.iterrows():
        clip_path = find_clip_path(row, manifest_stems)
        if clip_path is not None:
            entries.append((idx, row, clip_path))

    if args.max_clips > 0:
        entries = entries[:args.max_clips]

    print(f"  Available clips: {len(entries)}/{len(ad_df)}")
    if not entries:
        print("  No clips found. Check CLIPS_DIR.")
        return

    # ── Load face gallery ───────────────────────────────────
    print("[Phase 5] Loading face gallery...")
    gallery_cache: Dict[str, Tuple] = {}
    imdb_to_movie = build_imdb_to_movie_map()
    print(f"  IMDB->movie mapping: {len(imdb_to_movie)} entries")
    try:
        from face_gallery import load_gallery
        print("  Face gallery module loaded.")
    except Exception:
        load_gallery = None
        print("  Face gallery not available, skipping.")

    # ── Build AD engine ─────────────────────────────────────
    print("\n[Phase 6] Building AD engine...")
    from ad_engine import build_ad_engine
    engine = build_ad_engine(gpu_id=0)
    print()

    # ── Inference ───────────────────────────────────────────
    print(f"[Phase 7] Running inference on {len(entries)} clips...")
    results = []
    total_time = 0.0

    for i, (idx, row, clip_path) in enumerate(entries):
        imdbid = str(row.get("imdbid", "")).strip()
        movie_title = str(row.get("movie_title", "")).strip()
        gt_text = str(row.get("text", "")).strip()
        split = str(row.get("split", "")).strip()
        videoid = row["cmd_filename"].split("/")[-1]

        # Nearby ADs for context
        movie_ads = ad_df[ad_df["imdbid"] == imdbid]
        nearby = get_nearby_ads(ad_df, movie_ads, idx, window=3)

        # Face detection
        chars: List[str] = []
        face_avatars: List[Path] = []
        if load_gallery is not None and imdbid:
            if imdbid not in gallery_cache:
                try:
                    movie_name = imdb_to_movie.get(imdbid, imdbid)
                    embs, meta = load_gallery(movie_name)
                    gallery_cache[imdbid] = (embs, meta)
                except Exception:
                    gallery_cache[imdbid] = (None, None)

            embs, meta = gallery_cache[imdbid]
            if embs is not None and meta is not None:
                try:
                    from face_gallery import detect_faces_in_clip
                    chars, face_avatars = detect_faces_in_clip(
                        clip_path, embs, meta, threshold=0.35,
                    )
                except Exception:
                    pass

        # Build context (with detected characters)
        context = build_context(row, charbank, nearby, detected_characters=chars)

        # Run inference
        try:
            ad_text, inf_time, _ = engine.infer_one_segment(
                clip_path=clip_path,
                context_text=context,
                task_prompt=TASK_PROMPT,
                temperature=args.temperature,
                max_new_tokens=args.max_tokens,
                face_avatars=face_avatars,
                character_names=chars,
            )
            total_time += inf_time
        except Exception as e:
            ad_text = ""
            inf_time = 0.0
            print(f"    [ERROR] clip {i}: {e}")

        results.append({
            "clip_idx": i,
            "imdbid": imdbid,
            "movie_title": movie_title,
            "videoid": videoid,
            "split": split,
            "cmd_filename": str(row.get("cmd_filename", "")),
            "scaled_start": float(row.get("scaled_start", 0)),
            "scaled_end": float(row.get("scaled_end", 0)),
            "duration": float(row.get("duration", 0)),
            "gt_text": gt_text,
            "ad_text": ad_text,
            "characters": chars,
            "inference_time_sec": round(inf_time, 3),
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(entries):
            print(f"  [{i + 1}/{len(entries)}] "
                  f"avg_time={total_time / (i + 1):.2f}s "
                  f"chars={len(chars) if chars else 0} "
                  f"movie={movie_title[:30]}")

    # ── Save results ────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "cmdad_output.json"

    output = {
        "method": "pipeline_cmdad",
        "dataset": "CMD-AD",
        "split": args.split,
        "total_ad_entries": len(ad_df),
        "available_clips": len(entries),
        "generated_count": len(results),
        "total_gaps": len(results),
        "total_generated": len(results),
        "inference_total_time_sec": round(total_time, 1),
        "preprocess_time_sec": 0,
        "total_time_sec": round(total_time, 1),
        "time_per_clip_sec": round(total_time / max(len(results), 1), 2),
        "task_prompt": TASK_PROMPT,
        "ad_entries": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: {output_file}")
    print(f"  Generated: {len(results)}/{len(entries)}")
    print(f"  Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
