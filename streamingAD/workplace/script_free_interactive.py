#!/usr/bin/env python3
"""
script_free_interactive.py — Interactive AD: Script-Free + User Instruction Insertion
=====================================================================================

Based on pipeline_script_free.py. Uses audio/visual analysis only:
  1. VAD silence detection (ffmpeg silencedetect) for dialogue gaps
  2. Whisper transcription for nearby dialogue context
  3. Scene boundary detection
  4. Face gallery (PlotTree) + insightface detection
  5. AD generation via Video-LLaMA

Key addition: inserts user instructions at random gap positions.
  - Instructions accumulate: after an insertion, all subsequent gaps
    are influenced by the active instruction set.
  - Supports category filtering or full random.

Output: _ad_output.json (baseline-compatible format) + insertion metadata.

Usage:
    conda activate videollava
    python streamingAD/workplace/script_free_interactive.py \
        --num-insertions 5 --gpu-id 3

    # Category-filtered:
    python streamingAD/workplace/script_free_interactive.py \
        --categories style detail --gpu-id 3

    # Single movie:
    python streamingAD/workplace/script_free_interactive.py \
        --only-movie "Shawshank" --gpu-id 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STREAMING_ROOT))

from vad_gap_detector import (
    AudioGap, VideoSceneBoundary,
    detect_gaps_from_video, extract_clip, _prepare_fast_video, _sec_to_timestamp,
    _run_ffmpeg_silence_detect,
)
from speech_transcriber import (
    SpeakerTurn, TranscriptionResult,
    transcribe_video, turns_near_timestamp, turns_to_context_text,
    DEFAULT_WHISPER_MODEL,
)
from analysis_cache import load_analysis, save_analysis
from ad_engine import build_ad_engine
from face_gallery import load_gallery

# ── Instruction categories (same as script_interactive) ───────

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
TASK_PROMPT = (
    "Describe what is happening in this clip concisely. "
    "Focus on visible actions, movements, and expressions. "
    "If character names are mentioned in the context, use them "
    "(e.g. 'Don Vito Corleone walks...' not 'A man walks...'). "
    "Do not quote dialogue."
)

DEFAULT_VIDEO_DIR = "/mnt/disk1new/storyvideo/Movie"
DEFAULT_OUTPUT_DIR = "/mnt/disk1new/ylz/newAD/experiment_results/script_free_interactive"
DEFAULT_MIN_GAP_SEC = 4.0
DEFAULT_MAX_GAP_SEC = 60.0
DEFAULT_SILENCE_DB = -30.0
DEFAULT_SILENCE_DUR = 1.5


def load_instruction_categories(config_path: Optional[Path] = None,
                                filter_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
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
    if not categories:
        return "Describe what is happening in this clip in detail.", "default"
    ids = list(categories.keys())
    weights = [categories[cid].get("weight", 1.0) for cid in ids]
    cat_id = rng.choices(ids, weights=weights, k=1)[0]
    templates = categories[cat_id].get("templates", [])
    if not templates:
        return "Describe what is happening in this clip in detail.", cat_id
    return rng.choice(templates), cat_id


def select_insertion_gaps(num_gaps: int, num_insertions: int,
                          rng: random.Random) -> List[int]:
    max_idx = max(num_gaps - 1, 0)
    if num_insertions >= num_gaps:
        return list(range(num_gaps))
    return sorted(rng.sample(range(max(1, num_gaps)), min(num_insertions, num_gaps)))


def _get_video_duration(video_path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _find_nearest_scene(gap: AudioGap, scenes: List[VideoSceneBoundary],
                        window_sec: float = 5.0) -> Optional[float]:
    if not scenes:
        return None
    best_ts, best_dist = None, float("inf")
    mid = (gap.gap_start_sec + gap.gap_end_sec) / 2
    for b in scenes:
        dist = abs(b.timestamp_sec - mid)
        if dist < best_dist and dist < window_sec:
            best_dist, best_ts = dist, b.timestamp_sec
    return best_ts


def _build_context(
    gap: AudioGap,
    trans_turns: Optional[List[SpeakerTurn]],
    nearest_scene: Optional[float],
    scene_boundaries: List[VideoSceneBoundary],
    detected_characters: Optional[List[str]] = None,
) -> str:
    if trans_turns:
        nearby = turns_near_timestamp(
            trans_turns, (gap.gap_start_sec + gap.gap_end_sec) / 2,
            window_before_sec=45.0, window_after_sec=20.0,
        )
        ctx = turns_to_context_text(nearby, gap.gap_start_sec, gap.gap_end_sec)
    else:
        ctx = (
            f"[Scene context]\n"
            f"This is a silent pause between conversations ({gap.gap_duration_sec:.1f}s).\n\n"
            f"[Instructions]\n"
            f"Describe only what is VISIBLE on screen during this gap. "
            f"Focus on actions, gestures, facial expressions, camera movements, "
            f"and environmental changes."
        )

    if detected_characters:
        char_list = ", ".join(detected_characters)
        ctx = f"[Characters visible in this clip: {char_list}]\n\n{ctx}"

    if nearest_scene is not None:
        score = 0.0
        for b in scene_boundaries:
            if abs(b.timestamp_sec - nearest_scene) < 0.01:
                score = b.score
                break
        ctx += (f"\n\n[Visual cue]\nA visual scene transition occurs near this gap "
                f"(at {nearest_scene:.1f}s, score={score:.2f}).")
    return ctx


def _extract_nearby_turns(
    trans_turns: Optional[List[SpeakerTurn]],
    gap_start: float, gap_end: float,
    window_before: float = 45.0, window_after: float = 20.0,
) -> List[SpeakerTurn]:
    if not trans_turns:
        return []
    return turns_near_timestamp(
        trans_turns, (gap_start + gap_end) / 2,
        window_before_sec=window_before,
        window_after_sec=window_after,
    )


def _dialogue_before(turns: List[SpeakerTurn], gap_start: float,
                     max_lines: int = 5) -> List[str]:
    before = [t for t in turns if t.end_sec <= gap_start]
    before.sort(key=lambda t: t.start_sec)
    return [f"{t.speaker_label}: {t.text}" for t in before[-max_lines:]]


def _dialogue_after(turns: List[SpeakerTurn], gap_end: float,
                    max_lines: int = 5) -> List[str]:
    after = [t for t in turns if t.start_sec >= gap_end]
    after.sort(key=lambda t: t.start_sec)
    return [f"{t.speaker_label}: {t.text}" for t in after[:max_lines]]


def find_videos(video_dir: Path, only_movie: Optional[str] = None) -> List[Path]:
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts"}
    videos = sorted(f for f in video_dir.iterdir() if f.suffix.lower() in video_exts)
    if only_movie:
        videos = [v for v in videos if only_movie.lower() in v.stem.lower()]
    return videos


def run_one_movie(
    engine,
    video_path: Path,
    output_dir: Path,
    categories: Dict[str, Dict[str, Any]],
    num_insertions: int,
    rng: random.Random,
    min_gap_sec: float,
    max_gap_sec: float,
    silence_threshold_db: float,
    min_silence_dur: float,
    whisper_model: str,
    temperature: float,
    max_new_tokens: int,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Run interactive experiment on one movie (script-free)."""
    movie_name = video_path.stem
    video_duration_sec = _get_video_duration(video_path)

    print(f"\n{'='*60}")
    print(f"[Script-Free Interactive] {movie_name}")
    print(f"  Video: {video_path}")
    print(f"  Duration: {video_duration_sec:.0f}s ({video_duration_sec/60:.1f}min)")
    print(f"  Insertions: {num_insertions}")
    print(f"  Categories: {list(categories.keys()) if categories else 'ALL'}")
    print(f"{'='*60}")

    t_start = time.monotonic()
    trans_turns: Optional[List[SpeakerTurn]] = None
    whisper_lang: str = ""
    gaps: List[AudioGap] = []
    scene_boundaries: List[VideoSceneBoundary] = []

    analysis_params = {
        "min_gap_sec": min_gap_sec,
        "max_gap_sec": max_gap_sec,
        "silence_threshold_db": silence_threshold_db,
        "min_silence_dur": min_silence_dur,
    }

    # Phase A: Analysis (cached)
    cached_turns, cached_lang, cached_gaps, cached_scenes = load_analysis(
        output_dir, movie_name, min_gap_sec, silence_threshold_db, min_silence_dur,
    )

    if cached_turns is not None and cached_gaps is not None and not force_recompute:
        print(f"  [Cache] Loaded: {len(cached_turns)} turns, {len(cached_gaps)} gaps")
        trans_turns = cached_turns
        whisper_lang = cached_lang
        gaps.extend(cached_gaps)
        if cached_scenes:
            scene_boundaries.extend(cached_scenes)
    else:
        print(f"  [Cache] Not found, computing...")
        t0 = time.monotonic()
        print(f"  [VAD] Silence detection...")
        events = _run_ffmpeg_silence_detect(
            video_path=video_path,
            silence_threshold_db=silence_threshold_db,
            min_silence_dur=min_silence_dur,
        )
        print(f"    {len(events)} silence events in {time.monotonic()-t0:.1f}s")

        t0 = time.monotonic()
        print(f"  [Whisper] Transcribing...")
        try:
            trans_result = transcribe_video(
                video_path=video_path, model_size=whisper_model,
                silence_threshold_db=silence_threshold_db,
                min_silence_dur=min_silence_dur,
                silence_events=events,
            )
            trans_turns = trans_result.turns
            whisper_lang = trans_result.language
            print(f"    {len(trans_turns)} turns in {time.monotonic()-t0:.1f}s")
        except Exception as e:
            print(f"    Whisper ERROR: {e}")
            import traceback
            traceback.print_exc()

        if events is not None:
            import vad_gap_detector as _vgd
            gaps_list = _vgd._silence_events_to_gaps(
                events, min_gap_sec=min_gap_sec, max_gap_sec=max_gap_sec,
            )
            gaps.extend(gaps_list)

            print(f"  [Scene] Detecting boundaries...")
            scene_boundaries = _vgd._detect_scene_boundaries(video_path)
            gaps = _vgd._merge_gaps_with_scenes(gaps, scene_boundaries)

        print(f"    {len(gaps)} gaps, {len(scene_boundaries)} scene boundaries")

        if trans_turns is not None or gaps:
            save_analysis(
                output_dir, movie_name,
                whisper_turns=trans_turns, whisper_lang=whisper_lang,
                gaps=gaps, scene_boundaries=scene_boundaries,
                params=analysis_params,
            )

    if not gaps:
        return {"movie": movie_name, "status": "no_gaps", "ad_entries": []}

    num_gaps = len(gaps)

    # Face gallery
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

    # Track active instructions
    active_instructions: List[Dict[str, str]] = []
    entries: List[Dict[str, Any]] = []
    insertion_events: List[Dict[str, Any]] = []
    inference_total = 0.0

    for idx, gap in enumerate(gaps):
        # Insert instruction if this is an insertion point
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
                "timestamp_sec": round(gap.gap_start_sec, 3),
                "category": cat_id,
                "instruction_text": template,
            })
            print(f"\n  *** INSERT #{len(insertion_events)} at gap {idx}: "
                  f"[{cat_id}] {template}")

        # Extract clip
        clip_path = clip_tmp / f"gap{gap.gap_id:04d}.mp4"
        if not extract_clip(fast_video, gap.gap_start_sec, gap.gap_end_sec, clip_path):
            continue

        # Detect faces in clip
        chars: List[str] = []
        face_avatars: List[Path] = []
        if gallery_embs is not None and gallery_people is not None:
            from face_gallery import detect_faces_in_clip
            chars, face_avatars = detect_faces_in_clip(
                clip_path, gallery_embs, gallery_people,
            )

        # Build context
        nearest_scene = _find_nearest_scene(gap, scene_boundaries)
        context_text = _build_context(
            gap, trans_turns, nearest_scene, scene_boundaries,
            detected_characters=chars,
        )

        # Add active instructions to context
        if active_instructions:
            instr_block = "\n[User instructions - follow ALL of these]\n"
            for i, ai in enumerate(active_instructions, 1):
                instr_block += f"  {i}. {ai['template']}\n"
            context_text = instr_block + "\n" + context_text

        # Generate AD
        t0 = time.monotonic()
        try:
            ad_text, inf_time, _ = engine.infer_one_segment(
                clip_path=clip_path, context_text=context_text,
                task_prompt=TASK_PROMPT, temperature=temperature,
                max_new_tokens=max_new_tokens,
                face_avatars=face_avatars, character_names=chars,
            )
        except Exception as e:
            ad_text = f"[ERROR: {e}]"
            inf_time = 0.0
        inference_total += inf_time

        nearby_turns = _extract_nearby_turns(trans_turns, gap.gap_start_sec, gap.gap_end_sec)

        entries.append({
            "gap_id": gap.gap_id,
            "scene_index": "",
            "location": "UNKNOWN",
            "gap_start_sec": round(gap.gap_start_sec, 3),
            "gap_end_sec": round(gap.gap_end_sec, 3),
            "gap_duration_sec": round(gap.gap_duration_sec, 3),
            "characters": chars,
            "context_before": _dialogue_before(nearby_turns, gap.gap_start_sec),
            "context_after": _dialogue_after(nearby_turns, gap.gap_end_sec),
            "ad_text": ad_text,
            "inference_time_sec": round(inf_time, 3),
            "active_instructions": list(active_instructions),
            "active_instruction_count": len(active_instructions),
        })

        print(f"  [{idx+1}/{num_gaps}] Gap {gap.gap_id} ({gap.gap_duration_sec:.1f}s, "
              f"instr={len(active_instructions)}): {ad_text[:80]}...")

        if clip_path.exists():
            clip_path.unlink()

    shutil.rmtree(str(clip_tmp), ignore_errors=True)

    t_total = time.monotonic() - t_start

    result = {
        "movie": movie_name,
        "movie_title": movie_name,
        "method": "script_free_interactive",
        "video_path": str(video_path),
        "video_duration_sec": video_duration_sec,
        "min_gap_sec": min_gap_sec,
        "max_gap_sec": max_gap_sec,
        "total_gaps": num_gaps,
        "generated_count": len(entries),
        "num_insertions": len(insertion_events),
        "inference_total_time_sec": round(inference_total, 1),
        "total_time_sec": round(t_total, 1),
        "ad_entries": entries,
        "insertion_events": insertion_events,
        "categories_used": list(set(ie["category"] for ie in insertion_events)),
    }

    # Save output
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
        description="Interactive AD: Script-Free + Instruction Insertion")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-gap-sec", type=float, default=DEFAULT_MIN_GAP_SEC)
    parser.add_argument("--max-gap-sec", type=float, default=DEFAULT_MAX_GAP_SEC)
    parser.add_argument("--silence-threshold-db", type=float, default=DEFAULT_SILENCE_DB)
    parser.add_argument("--min-silence-dur", type=float, default=DEFAULT_SILENCE_DUR)
    parser.add_argument("--whisper-model", default="turbo")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--only-movie", default=None)

    # Instruction config
    parser.add_argument("--num-insertions", type=int, default=None,
                        help="Fixed number of insertions per movie (default: random 1-10)")
    parser.add_argument("--min-insertions", type=int, default=1)
    parser.add_argument("--max-insertions", type=int, default=10)
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Category IDs (default: all). E.g.: style detail character")
    parser.add_argument("--instruction-config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "experiment_results" / f"script_free_interactive_{ts}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    categories = load_instruction_categories(
        Path(args.instruction_config) if args.instruction_config else None,
        filter_ids=args.categories,
    )

    videos = find_videos(Path(args.video_dir), only_movie=args.only_movie)
    if not videos:
        print("ERROR: No videos found.")
        sys.exit(1)

    print("=" * 60)
    print("Interactive AD: Script-Free")
    print(f"  Movies:    {len(videos)}")
    print(f"  Insertions: {'fixed ' + str(args.num_insertions) if args.num_insertions else f'random {args.min_insertions}-{args.max_insertions}'}")
    print(f"  Categories: {list(categories.keys())}")
    print(f"  Seed:      {args.seed}")
    print(f"  GPU:       {args.gpu_id}")
    print(f"  Output:    {output_dir}")
    print("=" * 60)

    print("\nLoading AD Engine...")
    engine = build_ad_engine(gpu_id=args.gpu_id)
    print("Engine ready.\n")

    os.environ.setdefault("WHISPER_CACHE_DIR", str(PROJECT_ROOT / "models" / "whisper"))

    all_results = []
    for vi, vp in enumerate(videos):
        n_ins = args.num_insertions if args.num_insertions else rng.randint(
            args.min_insertions, args.max_insertions)

        print(f"\n{'#'*60}")
        print(f"MOVIE {vi+1}/{len(videos)}: {vp.stem} (insertions={n_ins})")
        print(f"{'#'*60}")

        try:
            r = run_one_movie(
                engine=engine, video_path=vp, output_dir=output_dir,
                categories=categories, num_insertions=n_ins, rng=rng,
                min_gap_sec=args.min_gap_sec, max_gap_sec=args.max_gap_sec,
                silence_threshold_db=args.silence_threshold_db,
                min_silence_dur=args.min_silence_dur,
                whisper_model=args.whisper_model,
                temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                force_recompute=args.force_recompute,
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
