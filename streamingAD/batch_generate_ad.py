#!/usr/bin/env python3
"""
batch_generate_ad.py — Batch AD generation from videos + Excel scripts.

自动匹配视频文件和 xlsx 剧本，检测对话间隙，
在每个合适的静默间隔中生成 Audio Description。

Usage:
    python batch_generate_ad.py \
        --video-dir /path/to/videos \
        --xlsx-dir /path/to/xlsxs \
        --output-dir /path/to/output \
        --gap-threshold 4.0 \
        --gpu-id 0
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ── 配置区：直接修改这里的路径 ──────────────────────────────────────
VIDEO_DIRS: List[str] = [
    "/mnt/disk1new/storyvideo/Movie",
]
XLSX_DIRS: List[str] = [
    "/mnt/disk1new/storyvideo/Alignedscript/Movie",
]
OUTPUT_DIR: str = "/mnt/disk1new/ylz/newAD/batch_ad_output"
GAP_THRESHOLD_SEC: float = 4.0
TEMPERATURE: float = 0.2
MAX_NEW_TOKENS: int = 256
GPU_ID: int = 2
ONLY_MOVIE: Optional[str] = None  # set a movie name to process only one, or None to process all
GT_CSV: str = "/mnt/disk1new/ylz/newAD/AutoAD3/cmd_ad_anno_v1.csv"  # filter to known movies
# ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMING_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(STREAMING_ROOT))

from preprocess_pipeline import (
    _find_result_sheet,
    _load_xlsx_rows,
    _parse_hhmmss_mmm,
    dedupe_keep_order,
)
# ad_engine imported lazily in main() to avoid torch import overhead

TASK_PROMPT = (
    "Describe what is happening in this clip concisely. "
    "Focus on visible actions, movements, and expressions. "
    "If character names are mentioned in the context, use them "
    "(e.g. 'Don Vito Corleone walks...' not 'A man walks...'). "
    "Do not quote dialogue."
)


@dataclass
class DialogueRow:
    row_index: int
    start_sec: float
    end_sec: float
    start_time: str
    end_time: str
    dialog: str
    scene_index: str
    characters: str
    location: str


@dataclass
class GapCandidate:
    gap_id: int
    scene_index: str
    location: str
    gap_start_sec: float
    gap_end_sec: float
    gap_start_time: str
    gap_end_time: str
    gap_duration_sec: float
    context_before: List[str]
    context_after: List[str]
    characters: List[str]


def _sec_to_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def match_video_to_xlsx(video_dir: Path, xlsx_dir: Path) -> List[Tuple[Path, Path]]:
    videos = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov"}])
    xlsxs = sorted([p for p in xlsx_dir.iterdir() if p.suffix.lower() in {".xlsx"}])

    if not videos:
        print(f"[WARN] No video files found in {video_dir}")
    if not xlsxs:
        print(f"[WARN] No xlsx files found in {xlsx_dir}")

    pairs: List[Tuple[Path, Path]] = []
    used_xlsx: set = set()

    for video in videos:
        video_stem = video.stem.lower()
        best: Optional[Path] = None
        best_score = 0

        for xlsx in xlsxs:
            if xlsx in used_xlsx:
                continue
            xlsx_stem = xlsx.stem.lower()

            if video_stem == xlsx_stem:
                best = xlsx
                break

            if video_stem in xlsx_stem or xlsx_stem in video_stem:
                score = len(set(video_stem) & set(xlsx_stem))
                if score > best_score:
                    best_score = score
                    best = xlsx

        if best is not None:
            pairs.append((video, best))
            used_xlsx.add(best)
            print(f"  Matched: {video.name}  <->  {best.name}")
        else:
            print(f"  [SKIP] No xlsx match for: {video.name}")

    return pairs


def gather_all_pairs() -> List[Tuple[Path, Path]]:
    all_pairs: List[Tuple[Path, Path]] = []
    for video_dir, xlsx_dir in zip(VIDEO_DIRS, XLSX_DIRS):
        vd = Path(video_dir).resolve()
        xd = Path(xlsx_dir).resolve()
        print(f"--- Scanning pairs: video={vd}  xlsx={xd} ---")
        pairs = match_video_to_xlsx(vd, xd)
        all_pairs.extend(pairs)
        print(f"  → {len(pairs)} pairs found")

    if GT_CSV and Path(GT_CSV).exists():
        import csv as _csv
        import re as _re
        gt_titles: set = set()
        with open(GT_CSV) as f:
            for row in _csv.DictReader(f):
                t = row.get('movie_title', '').strip().lower()
                if t:
                    gt_titles.add(t)
        def _clean(n: str) -> str:
            return _re.sub(r'^(IMDB-\d+-|douban-\d+-)', '', n).strip().lower()
        filtered = [(v, x) for v, x in all_pairs if _clean(v.stem) in gt_titles]
        print(f"  → {len(filtered)} pairs after GT filter (of {len(all_pairs)} total)")
        all_pairs = filtered

    if ONLY_MOVIE:
        all_pairs = [(v, x) for v, x in all_pairs if v.stem == ONLY_MOVIE]
        if not all_pairs:
            print(f"  [WARN] ONLY_MOVIE='{ONLY_MOVIE}' not found in scanned pairs!")
    return all_pairs


def parse_dialogue_rows(xlsx_path: Path) -> List[DialogueRow]:
    result_sheet = _find_result_sheet(xlsx_path)
    raw_rows = _load_xlsx_rows(xlsx_path, sheet_name=result_sheet)
    rows: List[DialogueRow] = []

    for i, raw in enumerate(raw_rows):
        start_sec = _parse_hhmmss_mmm(str(raw.get("start_time", "") or ""))
        end_sec = _parse_hhmmss_mmm(str(raw.get("end_time", "") or ""))
        if start_sec is None or end_sec is None or end_sec < start_sec:
            continue

        dialog = str(raw.get("dialog", "") or "").strip()
        if not dialog:
            continue

        rows.append(DialogueRow(
            row_index=i,
            start_sec=start_sec,
            end_sec=end_sec,
            start_time=str(raw.get("start_time", "") or ""),
            end_time=str(raw.get("end_time", "") or ""),
            dialog=dialog,
            scene_index=str(raw.get("scene_index", "") or "0"),
            characters=str(raw.get("characters", "") or ""),
            location=str(raw.get("location", "") or ""),
        ))

    rows.sort(key=lambda r: r.start_sec)
    return rows


def detect_dialogue_gaps(
    rows: List[DialogueRow],
    gap_threshold_sec: float = 4.0,
    max_context_lines: int = 5,
) -> List[GapCandidate]:
    scenes: Dict[str, List[DialogueRow]] = {}
    for r in rows:
        scenes.setdefault(r.scene_index, []).append(r)
    for srows in scenes.values():
        srows.sort(key=lambda r: r.start_sec)

    sorted_scene_keys = sorted(scenes.keys(), key=lambda k: int(k) if k.isdigit() else 10**9)
    candidates: List[GapCandidate] = []
    gap_id = 0

    def _add_gap(
        srows_before: List[DialogueRow], idx: int,
        srows_after: List[DialogueRow], jdx: int,
        scene_a: str, scene_b: str,
    ):
        nonlocal gap_id
        gap_start = srows_before[idx].end_sec
        gap_end = srows_after[jdx].start_sec
        duration = gap_end - gap_start
        if duration < gap_threshold_sec:
            return

        context_before = [r.dialog for r in srows_before[max(0, idx - max_context_lines + 1):idx + 1]]
        context_after = [r.dialog for r in srows_after[jdx:min(len(srows_after), jdx + max_context_lines)]]

        chars_set: set = set()
        for r in srows_before[max(0, idx - 2):min(len(srows_before), idx + 1)]:
            for c in r.characters.split(","):
                c = c.strip()
                if c:
                    chars_set.add(c)
        for r in srows_after[jdx:min(len(srows_after), jdx + 3)]:
            for c in r.characters.split(","):
                c = c.strip()
                if c:
                    chars_set.add(c)

        location = srows_before[idx].location or srows_after[jdx].location or ""
        scene_label = scene_a if scene_a == scene_b else f"{scene_a}→{scene_b}"

        gap_id += 1
        candidates.append(GapCandidate(
            gap_id=gap_id,
            scene_index=scene_label,
            location=location,
            gap_start_sec=gap_start,
            gap_end_sec=gap_end,
            gap_start_time=_sec_to_timestamp(gap_start),
            gap_end_time=_sec_to_timestamp(gap_end),
            gap_duration_sec=duration,
            context_before=context_before,
            context_after=context_after,
            characters=sorted(chars_set),
        ))

    # Within-scene gaps
    for scene_idx in sorted_scene_keys:
        srows = scenes[scene_idx]
        for i in range(len(srows) - 1):
            _add_gap(srows, i, srows, i + 1, scene_idx, scene_idx)

    # Cross-scene gaps
    for si in range(len(sorted_scene_keys) - 1):
        a_key = sorted_scene_keys[si]
        b_key = sorted_scene_keys[si + 1]
        _add_gap(scenes[a_key], len(scenes[a_key]) - 1,
                 scenes[b_key], 0, a_key, b_key)

    return candidates


def extract_clip(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
    ffmpeg_bin: str = "ffmpeg",
) -> bool:
    duration = end_sec - start_sec
    if duration <= 0:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-map", "0:v:0?", "-map", "0:a:0?",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _prepare_fast_video(
    src_video: Path,
    cache_dir: Path,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    fast_video = cache_dir / f"fast_{src_video.stem}.mp4"
    if fast_video.exists():
        return fast_video

    print(f"  Pre-converting to fast-seek MP4 ({src_video.stat().st_size / 1e9:.1f}GB) ...")
    t0 = time.monotonic()
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src_video),
        "-map", "0:v:0?", "-map", "0:a:0?",
        "-c", "copy",
        "-movflags", "+faststart",
        str(fast_video),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    size_gb = fast_video.stat().st_size / 1e9
    print(f"  Done in {elapsed:.1f}s → {size_gb:.1f}GB")
    return fast_video


def build_gap_context(candidate: GapCandidate) -> str:
    parts: List[str] = []

    parts.append("[Scene context]")
    if candidate.location:
        parts.append(f"Location: {candidate.location}")
    if candidate.characters:
        parts.append(f"Characters: {', '.join(candidate.characters)}")

    if candidate.context_before:
        parts.append("\n[Dialogue before the gap]")
        for d in candidate.context_before:
            parts.append(f"- {d}")

    if candidate.context_after:
        parts.append("\n[Dialogue after the gap]")
        for d in candidate.context_after:
            parts.append(f"- {d}")

    parts.append("\n[Gap description]")
    parts.append(f"This is a silent interval ({candidate.gap_duration_sec:.1f}s) between "
                  f"two dialogue segments in scene {candidate.scene_index}.")

    return "\n".join(parts)


def process_one_pair(
    engine: ADEngine,
    video_path: Path,
    xlsx_path: Path,
    output_dir: Path,
    gap_threshold_sec: float,
    temperature: float,
    max_new_tokens: int,
) -> Dict[str, Any]:
    movie_name = video_path.stem
    print(f"\n{'='*60}")
    print(f"Processing: {movie_name}")
    print(f"  Video: {video_path}")
    print(f"  XLSX:  {xlsx_path}")
    print(f"{'='*60}")

    rows = parse_dialogue_rows(xlsx_path)
    if not rows:
        print(f"  [WARN] No dialogue rows found")
        return {"movie": movie_name, "status": "no_dialogue", "ad_entries": []}

    print(f"  Found {len(rows)} dialogue rows in {len(set(r.scene_index for r in rows))} scenes")

    candidates = detect_dialogue_gaps(rows, gap_threshold_sec)
    if not candidates:
        print(f"  No gaps >= {gap_threshold_sec}s found")
        return {"movie": movie_name, "status": "no_gaps", "gap_threshold": gap_threshold_sec, "ad_entries": []}

    print(f"  Found {len(candidates)} dialogue gaps >= {gap_threshold_sec}s")

    clip_tmp_dir = output_dir / f".tmp_clips_{movie_name}"
    clip_tmp_dir.mkdir(parents=True, exist_ok=True)

    fast_video = _prepare_fast_video(video_path, clip_tmp_dir)

    entries: List[Dict[str, Any]] = []

    for idx, cand in enumerate(candidates):
        clip_path = clip_tmp_dir / f"gap{cand.gap_id:04d}.mp4"

        t0 = time.monotonic()
        if not extract_clip(fast_video, cand.gap_start_sec, cand.gap_end_sec, clip_path):
            print(f"  [{idx+1}/{len(candidates)}] Gap {cand.gap_id}: FFmpeg extract FAILED")
            continue
        extract_time = time.monotonic() - t0

        context_text = build_gap_context(cand)
        print(f"\n  [{idx+1}/{len(candidates)}] Gap {cand.gap_id} "
              f"({cand.gap_duration_sec:.1f}s) "
              f"scene={cand.scene_index} "
              f"chars={cand.characters}")

        try:
            ad_text, inference_time, _ = engine.infer_one_segment(
                clip_path=clip_path,
                context_text=context_text,
                task_prompt=TASK_PROMPT,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            print(f"    Inference ERROR: {exc}")
            ad_text = f"[ERROR: {exc}]"
            inference_time = 0.0

        print(f"    AD ({inference_time:.1f}s): {ad_text}")

        entries.append({
            "gap_id": cand.gap_id,
            "scene_index": cand.scene_index,
            "location": cand.location,
            "gap_start_time": cand.gap_start_time,
            "gap_end_time": cand.gap_end_time,
            "gap_start_sec": round(cand.gap_start_sec, 3),
            "gap_end_sec": round(cand.gap_end_sec, 3),
            "gap_duration_sec": round(cand.gap_duration_sec, 1),
            "characters": cand.characters,
            "context_before": cand.context_before,
            "context_after": cand.context_after,
            "ad_text": ad_text,
            "inference_time_sec": round(inference_time, 1),
            "extract_time_sec": round(extract_time, 1),
        })

        if clip_path.exists():
            clip_path.unlink()

    try:
        shutil.rmtree(str(clip_tmp_dir), ignore_errors=True)
    except OSError:
        pass

    result = {
        "movie": movie_name,
        "video_path": str(video_path),
        "xlsx_path": str(xlsx_path),
        "gap_threshold_sec": gap_threshold_sec,
        "total_gaps": len(candidates),
        "generated_count": len(entries),
        "ad_entries": entries,
    }

    out_file = output_dir / f"{movie_name}_ad_output.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {out_file}")

    return result


def write_summary_csv(all_results: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("movie,gap_id,scene_index,location,gap_start_time,gap_end_time,gap_duration_sec,characters,ad_text,inference_time_sec\n")
        for r in all_results:
            movie = r["movie"]
            for e in r.get("ad_entries", []):
                chars = "|".join(e["characters"])
                ad = e["ad_text"].replace('"', '""')
                f.write(f'{movie},{e["gap_id"]},{e["scene_index"]},{e["location"]},{e["gap_start_time"]},{e["gap_end_time"]},{e["gap_duration_sec"]},"{chars}","{ad}",{e["inference_time_sec"]}\n')
    print(f"\nSummary CSV: {csv_path}")


def main():
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Batch AD Generation ===")
    for vd, xd in zip(VIDEO_DIRS, XLSX_DIRS):
        print(f"Video dir: {vd}")
        print(f"XLSX dir:  {xd}")
    print(f"Output:    {output_dir}")
    print(f"Gap thld:  {GAP_THRESHOLD_SEC}s")
    print(f"Temp:      {TEMPERATURE}")
    print(f"GPU:       {GPU_ID}")
    print()

    pairs = gather_all_pairs()
    if not pairs:
        print("ERROR: No video-xlsx pairs found.")
        sys.exit(1)

    print(f"\n  {len(pairs)} total pairs to process\n")

    print("--- Loading Video-LLaMA model ---")
    from ad_engine import build_ad_engine
    engine = build_ad_engine(gpu_id=GPU_ID)
    print()

    all_results: List[Dict[str, Any]] = []

    for vi, (video_path, xlsx_path) in enumerate(pairs):
        try:
            result = process_one_pair(
                engine=engine,
                video_path=video_path,
                xlsx_path=xlsx_path,
                output_dir=output_dir,
                gap_threshold_sec=GAP_THRESHOLD_SEC,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            all_results.append(result)

            total_gaps = result.get("total_gaps", 0)
            generated = result.get("generated_count", 0)
            print(f"\n  [{vi+1}/{len(pairs)}] {result['movie']}: "
                  f"{generated}/{total_gaps} ADs generated")

        except Exception as exc:
            print(f"\n  [{vi+1}/{len(pairs)}] ERROR processing {video_path.stem}: {exc}")
            traceback.print_exc()

    write_summary_csv(all_results, output_dir)

    total_ads = sum(r.get("generated_count", 0) for r in all_results)
    total_gaps = sum(r.get("total_gaps", 0) for r in all_results)
    print(f"\n=== Done ===")
    print(f"Total: {len(pairs)} movies, {total_ads}/{total_gaps} ADs generated")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
