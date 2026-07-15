#!/usr/bin/env python3
"""
run_all_generate.py — Batch AD generation for ALL matching movies.
Only generates ADs, does NOT run evaluation.

Usage:
    conda activate videollava
    python streamingAD/run_all_generate.py
"""

from __future__ import annotations

import csv
import json
import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
try:
    import ctypes
    ctypes.CDLL("libavutil.so").av_log_set_level(16)
except Exception:
    pass
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Config ────────────────────────────────────────────────────
VIDEO_DIRS: List[str] = ["/mnt/disk1new/storyvideo/Movie"]
XLSX_DIRS: List[str] = ["/mnt/disk1new/storyvideo/Alignedscript/Movie"]
GT_CSV: str = "/mnt/disk1new/ylz/newAD/AutoAD3/cmd_ad_anno_v1.csv"
OUTPUT_BASE: str = "/mnt/disk1new/ylz/newAD/batch_ad_output"
GAP_THRESHOLD_SEC: float = 4.0
TEMPERATURE: float = 0.2
MAX_NEW_TOKENS: int = 256
GPU_ID: int = 0
START_FROM: int = 0
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def _clean_name(fname: str) -> str:
    name = os.path.splitext(fname)[0]
    name = re.sub(r'^(IMDB-\d+-|douban-\d+-)', '', name).strip().lower()
    return name


def _find_matching_movies() -> List[Tuple[str, str, str, str]]:
    gt: Dict[str, str] = {}
    with open(GT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row['imdbid']] = row['movie_title'].strip().lower()

    video_names: Dict[str, str] = {}
    for vd in VIDEO_DIRS:
        for f in os.listdir(vd):
            if f.endswith(('.mkv', '.mp4')):
                video_names[_clean_name(f)] = f

    xlsx_names: Dict[str, str] = {}
    for xd in XLSX_DIRS:
        for f in os.listdir(xd):
            if f.endswith('.xlsx'):
                xlsx_names[_clean_name(f)] = f

    matched: List[Tuple[str, str, str, str]] = []
    for vname, vfile in sorted(video_names.items()):
        if vname not in xlsx_names:
            continue
        for iid, gtitle in gt.items():
            if vname == gtitle:
                matched.append((vname, vfile, xlsx_names[vname], iid))
                break
    return matched


def _build_gt_for_alignment(
    ad_entries: List[Dict[str, Any]],
    gt_rows: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    refs: Dict[str, List[str]] = {}
    for e in ad_entries:
        gid = str(e['gap_id'])
        gap_s = e['gap_start_sec']
        gap_e = e['gap_end_sec']
        matched_texts: List[str] = []
        for gt_r in gt_rows:
            try:
                gt_s = float(gt_r.get('audiovault_start', 0))
                gt_e = float(gt_r.get('audiovault_end', 0))
            except (ValueError, TypeError):
                continue
            if max(gap_s, gt_s) < min(gap_e, gt_e):
                matched_texts.append(gt_r.get('text', ''))
        if matched_texts:
            refs[gid] = matched_texts
    return refs


def generate_one_movie(
    engine,
    movie_name: str,
    video_basename: str,
    xlsx_basename: str,
    imdbid: str,
    out_dir: Path,
) -> bool:
    from batch_generate_ad import (
        parse_dialogue_rows,
        detect_dialogue_gaps,
        extract_clip,
        _prepare_fast_video,
        build_gap_context,
        TASK_PROMPT,
    )

    ad_file = out_dir / f"{movie_name}_ad_output.json"
    ref_file = out_dir / f"{movie_name}_ref.json"

    if ad_file.exists() and ref_file.exists():
        print(f"  ⏭  Already done, skipping")
        return True

    video_path = None
    for vd in VIDEO_DIRS:
        p = Path(vd) / video_basename
        if p.exists():
            video_path = p.resolve()
            break
    if video_path is None:
        print(f"  ❌ Video not found: {video_basename}")
        return False

    xlsx_path = None
    for xd in XLSX_DIRS:
        p = Path(xd) / xlsx_basename
        if p.exists():
            xlsx_path = p.resolve()
            break
    if xlsx_path is None:
        print(f"  ❌ XLSX not found: {xlsx_basename}")
        return False

    rows = parse_dialogue_rows(xlsx_path)
    if not rows:
        print(f"  ❌ No dialogue rows")
        return False

    candidates = detect_dialogue_gaps(rows, GAP_THRESHOLD_SEC)
    if not candidates:
        print(f"  ❌ No gaps >= {GAP_THRESHOLD_SEC}s")
        return False

    print(f"    {len(rows)} dialogue rows → {len(candidates)} gaps")

    # ══════ Load face gallery ══════
    from face_gallery import load_gallery, lookup_face_images
    gallery_embs, gallery_people = load_gallery(movie_name)
    face_avatar_cache: Dict[str, List[Path]] = {}

    clip_tmp = out_dir / f".tmp_{movie_name}"
    clip_tmp.mkdir(parents=True, exist_ok=True)

    t0_prep = time.monotonic()
    fast_video = _prepare_fast_video(video_path, clip_tmp)

    entries: List[Dict[str, Any]] = []
    inference_total_sec: float = 0.0

    for cand in candidates:
        clip_path = clip_tmp / f"gap{cand.gap_id:04d}.mp4"
        if not extract_clip(fast_video, cand.gap_start_sec, cand.gap_end_sec, clip_path):
            continue

        context_text = build_gap_context(cand)

        face_avatars: List[Path] = []
        if gallery_people is not None and cand.characters:
            cache_key = "|".join(sorted(cand.characters))
            if cache_key not in face_avatar_cache:
                face_avatar_cache[cache_key] = lookup_face_images(
                    cand.characters, gallery_people, movie_name,
                )
            face_avatars = face_avatar_cache[cache_key]

        try:
            ad_text, inference_time, _ = engine.infer_one_segment(
                clip_path=clip_path,
                context_text=context_text,
                task_prompt=TASK_PROMPT,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
                face_avatars=face_avatars,
                character_names=cand.characters,
            )
        except Exception:
            ad_text = "[INFERENCE ERROR]"
            inference_time = 0.0

        inference_total_sec += inference_time

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
        })

        if clip_path.exists():
            clip_path.unlink()

    try:
        shutil.rmtree(str(clip_tmp), ignore_errors=True)
    except OSError:
        pass

    preprocess_time = time.monotonic() - t0_prep

    video_duration_sec = 0.0
    try:
        import subprocess
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            text=True,
        )
        video_duration_sec = float(out.strip())
    except Exception:
        pass

    total_time_sec = preprocess_time + inference_total_sec
    time_per_sec = total_time_sec / video_duration_sec if video_duration_sec > 0 else 0

    gt_rows: List[Dict[str, Any]] = []
    with open(GT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['imdbid'] == imdbid:
                gt_rows.append(row)

    refs = _build_gt_for_alignment(entries, gt_rows)
    with ref_file.open('w', encoding='utf-8') as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)

    gap_result = {
        "movie": movie_name,
        "imdbid": imdbid,
        "gap_threshold_sec": GAP_THRESHOLD_SEC,
        "total_gaps": len(candidates),
        "generated_count": len(entries),
        "video_duration_sec": round(video_duration_sec, 1),
        "preprocess_time_sec": round(preprocess_time, 1),
        "inference_total_time_sec": round(inference_total_sec, 1),
        "total_time_sec": round(total_time_sec, 1),
        "time_per_video_sec": round(time_per_sec, 3),
        "ad_entries": entries,
    }
    with ad_file.open('w', encoding='utf-8') as f:
        json.dump(gap_result, f, ensure_ascii=False, indent=2)

    return True


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

    global START_FROM
    if len(sys.argv) > 1 and sys.argv[1] == "--start-from":
        START_FROM = int(sys.argv[2])

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(OUTPUT_BASE) / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "run.log"
    log_fh = log_file.open('w', encoding='utf-8', buffering=1)

    class Tee:
        def __init__(self, fh):
            self.fh = fh
            self.stdout = sys.stdout
        def write(self, s):
            self.stdout.write(s)
            self.fh.write(s)
        def flush(self):
            self.stdout.flush()
            self.fh.flush()

    sys.stdout = Tee(log_fh)

    print(f"Run dir: {run_dir}\n")

    matched = _find_matching_movies()
    print(f"Matching movies (video + xlsx + GT): {len(matched)}")
    for i, (name, vf, xf, iid) in enumerate(matched):
        tag = " ← START" if i >= START_FROM else ""
        print(f"  {i:2d}. [{name[:45]:45s}] {iid}  {tag}")

    def log(msg: str):
        t = datetime.now().strftime("%H:%M:%S")
        print(f"[{t}] {msg}")

    log(f"Starting batch AD generation for {len(matched)} movies")
    log(f"GPU={GPU_ID} GapThld={GAP_THRESHOLD_SEC}s Temp={TEMPERATURE}")

    log("Loading Video-LLaMA AD engine ...")
    from ad_engine import build_ad_engine
    engine = build_ad_engine(gpu_id=GPU_ID)
    log("Engine loaded.")

    success = 0
    fail = 0

    for idx, (mname, vbasename, xbasename, imdbid) in enumerate(matched):
        if idx < START_FROM:
            continue

        log(f"[{idx+1}/{len(matched)}] {mname} ({imdbid})")
        t_start = time.monotonic()
        try:
            ok = generate_one_movie(engine, mname, vbasename, xbasename, imdbid, run_dir)
            elapsed = time.monotonic() - t_start
            if ok:
                success += 1
                log(f"  ✅ Done ({elapsed:.0f}s)")
            else:
                fail += 1
                log(f"  ❌ FAILED ({elapsed:.0f}s)")
        except Exception as e:
            elapsed = time.monotonic() - t_start
            log(f"  ❌ ERROR ({elapsed:.0f}s): {e}")
            traceback.print_exc(file=log_fh)
            fail += 1

    log(f"\nDone. Success: {success}, Fail: {fail}")

    log_fh.close()
    print(f"\nLog: {log_file}")
    print(f"Output: {run_dir}")
    print(f"\nTo run evaluation: conda run -n videollava python streamingAD/run_all_eval.py --run-dir {run_dir}")


if __name__ == "__main__":
    main()
