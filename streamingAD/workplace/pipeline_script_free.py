#!/usr/bin/env python3
"""
pipeline_script_free.py — Method 2: Script-Free Audio Description Generation
=============================================================================

No aligned script required. Works on any raw video using only audio + visual analysis:

  Phase A — Analysis (one-time per video):
    A1. VAD silence detection: ffmpeg silencedetect finds dialogue gaps
    A2. Whisper transcription: transcribe all speech with timestamps
    A3. Speaker segmentation: map Whisper segments onto VAD speech intervals

  Phase B — Generation (per gap):
    B1. Extract video clip for each gap
    B2. Build context from nearby transcribed dialogue
    B3. Generate AD text via Video-LLaMA

Comparison with Method 1 (pipeline_script_based.py):
  - Same output JSON format → direct comparison
  - Extra phase A adds transcription time (~10-15 min for 2h film)
  - No xlsx / aligned script dependency
  - Context uses SPEAKER-N labels instead of named characters

Usage:
    python streamingAD/pipeline_script_free.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMING_ROOT = Path(__file__).resolve().parent
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

# ── Default Config (modify here, no CLI args needed) ─────────
DEFAULT_VIDEO_DIR: str = "/mnt/disk1new/storyvideo/Movie"
DEFAULT_OUTPUT_DIR: str = "/mnt/disk1new/ylz/newAD/compare/script_free"
DEFAULT_MIN_GAP_SEC: float = 4.0
DEFAULT_MAX_GAP_SEC: float = 60.0
DEFAULT_SILENCE_DB: float = -30.0
DEFAULT_SILENCE_DUR: float = 1.5
DEFAULT_WHISPER: str = "turbo"
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_MAX_TOKENS: int = 256
DEFAULT_GPU: int = 0
DEFAULT_ONLY_MOVIE: str = "Shawshank"
DEFAULT_NO_SCENE_DETECT: bool = True
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts"}

TASK_PROMPT = (
    "Describe what is happening in this clip concisely. "
    "Focus on visible actions, movements, and expressions. "
    "If character names are mentioned in the context, use them "
    "(e.g. 'Don Vito Corleone walks...' not 'A man walks...'). "
    "Do not quote dialogue."
)
# ──────────────────────────────────────────────────────────────


def _pct(a, b): return f"{a}/{b}" if b else "0"
def _fmt_sec(s): return f"{s:.1f}s"


# ── Video Discovery ──────────────────────────────────────────

def find_videos(
    video_paths: Optional[List[Path]] = None,
    video_dir: Optional[Path] = None,
    only_movie: Optional[str] = None,
) -> List[Path]:
    videos: List[Path] = []
    if video_paths:
        for vp in video_paths:
            vp = Path(vp).resolve()
            if vp.is_file() and vp.suffix.lower() in VIDEO_EXTS:
                videos.append(vp)
            elif vp.is_dir():
                for f in sorted(vp.iterdir()):
                    if f.suffix.lower() in VIDEO_EXTS:
                        videos.append(f)
    if video_dir:
        vd = Path(video_dir).resolve()
        for f in sorted(vd.iterdir()):
            if f.suffix.lower() in VIDEO_EXTS:
                if f not in videos:
                    videos.append(f)
    if only_movie:
        videos = [v for v in videos if only_movie.lower() in v.stem.lower()]
    return videos


# ── Helper: Video Duration ─────────────────────────────────

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


# ── Helper: Scene Boundary ──────────────────────────────────

def _find_nearest_scene(gap: AudioGap, scenes: List[VideoSceneBoundary],
                        window_sec: float = 5.0) -> Optional[float]:
    if not scenes: return None
    best_ts, best_dist = None, float("inf")
    mid = (gap.gap_start_sec + gap.gap_end_sec) / 2
    for b in scenes:
        dist = abs(b.timestamp_sec - mid)
        if dist < best_dist and dist < window_sec:
            best_dist, best_ts = dist, b.timestamp_sec
    return best_ts


# ── Main Pipeline ────────────────────────────────────────────

def process_one_video(
    engine,
    video_path: Path,
    output_dir: Path,
    min_gap_sec: float = DEFAULT_MIN_GAP_SEC,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
    silence_threshold_db: float = DEFAULT_SILENCE_DB,
    min_silence_dur: float = DEFAULT_SILENCE_DUR,
    use_scene_detect: bool = True,
    transcribe: bool = True,
    whisper_model: str = DEFAULT_WHISPER,
    temperature: float = DEFAULT_TEMPERATURE,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    gpu_id: int = DEFAULT_GPU,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    movie_name = video_path.stem
    pipeline_start = time.monotonic()

    video_duration_sec = _get_video_duration(video_path)

    print(f"\n{'='*60}")
    print(f"[Method 2: Script-Free] {movie_name}")
    print(f"  Video: {video_path}")
    print(f"  Size:  {video_path.stat().st_size / 1e9:.1f}GB")
    print(f"{'='*60}")

    timing: Dict[str, float] = {}
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

    # ══════ Phase A: Analysis (cached) ══════
    cached_turns, cached_lang, cached_gaps, cached_scenes = load_analysis(
        output_dir, movie_name, min_gap_sec, silence_threshold_db, min_silence_dur,
    )

    if cached_turns is not None and cached_gaps is not None and not force_recompute:
        print(f"  [Cache] ✅ Loaded analysis cache: {len(cached_turns)} turns, "
              f"{len(cached_gaps)} gaps, {len(cached_scenes or [])} scene boundaries")
        trans_turns = cached_turns
        whisper_lang = cached_lang
        gaps.extend(cached_gaps)
        if cached_scenes:
            scene_boundaries.extend(cached_scenes)
    else:
        print(f"  [Cache] Not found or params changed, computing...")
        silence_events_raw: Optional[List[Dict[str, float]]] = None

        t0 = time.monotonic()
        print(f"  [Phase A3] VAD silence detection (shared) ...")
        events = _run_ffmpeg_silence_detect(
            video_path=video_path,
            silence_threshold_db=silence_threshold_db,
            min_silence_dur=min_silence_dur,
        )
        silence_events_raw = events
        timing["vad_scan_sec"] = round(time.monotonic() - t0, 1)
        print(f"    Done in {_fmt_sec(timing['vad_scan_sec'])}: {len(events)} silence events")

        if transcribe:
            t0 = time.monotonic()
            print(f"  [Phase A1/A2] Whisper '{whisper_model}' transcription ...")
            try:
                trans_result = transcribe_video(
                    video_path=video_path, model_size=whisper_model,
                    silence_threshold_db=silence_threshold_db, min_silence_dur=min_silence_dur,
                    silence_events=silence_events_raw,
                )
                trans_turns = trans_result.turns
                whisper_lang = trans_result.language
                timing["whisper_transcribe_sec"] = round(time.monotonic() - t0, 1)
                print(f"    Done in {_fmt_sec(timing['whisper_transcribe_sec'])}: "
                      f"{len(trans_turns)} speaker turns, lang={whisper_lang}")
            except Exception as e:
                print(f"    Whisper ERROR: {e}")
                traceback.print_exc()
                timing["whisper_transcribe_sec"] = 0.0

        if silence_events_raw is not None:
            import vad_gap_detector as _vgd
            gaps_list = _vgd._silence_events_to_gaps(
                silence_events_raw, min_gap_sec=min_gap_sec, max_gap_sec=max_gap_sec,
            )
            gaps.extend(gaps_list)

            if use_scene_detect:
                t0 = time.monotonic()
                print(f"  [Phase A3b] Scene boundary detection ...")
                scene_boundaries = _vgd._detect_scene_boundaries(video_path)
                gaps = _vgd._merge_gaps_with_scenes(gaps, scene_boundaries)
                timing["scene_detect_sec"] = round(time.monotonic() - t0, 1)

        print(f"    Analysis complete: {len(gaps)} gaps >= {min_gap_sec}s, "
              f"{len(scene_boundaries)} scene boundaries")

        if trans_turns is not None or gaps:
            save_analysis(
                output_dir, movie_name,
                whisper_turns=trans_turns, whisper_lang=whisper_lang,
                gaps=gaps, scene_boundaries=scene_boundaries,
                params=analysis_params,
            )
            print(f"    📁 Analysis cache saved")

    if not gaps:
        print("  [WARN] No gaps found")
        return {"movie": movie_name, "method": "script_free", "status": "no_gaps",
                "gap_threshold_sec": min_gap_sec, "ad_entries": []}

    # ══════ Phase A4: Face gallery ══════
    from face_gallery import load_gallery
    gallery_embs, gallery_people = load_gallery(movie_name)
    if gallery_embs is not None:
        print(f"  [Phase A4] Face gallery: {len(gallery_people)} characters loaded")
    else:
        print(f"  [Phase A4] Face gallery: not available for {movie_name}")

    # ══════ Phase B: Generation ══════
    clip_tmp = output_dir / f".tmp_{movie_name}"
    clip_tmp.mkdir(parents=True, exist_ok=True)
    fast_video = _prepare_fast_video(video_path, clip_tmp)

    entries: List[Dict[str, Any]] = []
    inference_total = 0.0
    extract_total = 0.0

    for idx, gap in enumerate(gaps):
        clip_path = clip_tmp / f"gap{gap.gap_id:04d}.mp4"

        t0 = time.monotonic()
        if not extract_clip(fast_video, gap.gap_start_sec, gap.gap_end_sec, clip_path):
            print(f"  [{_pct(idx+1, len(gaps))}] Gap {gap.gap_id}: extract FAILED")
            continue
        extract_time = time.monotonic() - t0
        extract_total += extract_time

        nearest_scene = _find_nearest_scene(gap, scene_boundaries)

        chars: List[str] = []
        face_avatars: List[Path] = []
        if gallery_embs is not None and gallery_people is not None:
            from face_gallery import detect_faces_in_clip
            chars, face_avatars = detect_faces_in_clip(
                clip_path, gallery_embs, gallery_people,
            )
            if chars:
                print(f"    👤 {', '.join(chars)}")

        context_text = _build_context(
            gap, trans_turns, nearest_scene, scene_boundaries, detected_characters=chars,
        )

        try:
            ad_text, inf_time, _ = engine.infer_one_segment(
                clip_path=clip_path, context_text=context_text,
                task_prompt=TASK_PROMPT, temperature=temperature, max_new_tokens=max_new_tokens,
                face_avatars=face_avatars, character_names=chars,
            )
        except Exception as e:
            ad_text = f"[ERROR: {e}]"
            inf_time = 0.0
        inference_total += inf_time

        print(f"  [{_pct(idx+1, len(gaps))}] Gap {gap.gap_id} "
              f"({gap.gap_duration_sec:.1f}s): {ad_text}")

        nearby_turns = _extract_nearby_turns(trans_turns, gap.gap_start_sec, gap.gap_end_sec)

        entries.append({
            "gap_id": gap.gap_id,
            "gap_start_time": _sec_to_timestamp(gap.gap_start_sec),
            "gap_end_time": _sec_to_timestamp(gap.gap_end_sec),
            "gap_start_sec": round(gap.gap_start_sec, 3),
            "gap_end_sec": round(gap.gap_end_sec, 3),
            "gap_duration_sec": round(gap.gap_duration_sec, 1),
            "ad_text": ad_text,
            "scene_index": "",
            "location": "UNKNOWN",
            "characters": chars,
            "context_before": _dialogue_before(nearby_turns, gap.gap_start_sec),
            "context_after": _dialogue_after(nearby_turns, gap.gap_end_sec),
            "inference_time_sec": round(inf_time, 1),
            "extract_time_sec": round(extract_time, 1),
        })

        if clip_path.exists():
            clip_path.unlink()

    shutil.rmtree(str(clip_tmp), ignore_errors=True)

    timing["extract_total_sec"] = round(extract_total, 1)
    timing["inference_total_sec"] = round(inference_total, 1)
    timing["total_sec"] = round(time.monotonic() - pipeline_start, 1)

    gap_durations = [g.gap_duration_sec for g in gaps]
    result = {
        "movie": movie_name,
        "method": "script_free",
        "video_path": str(video_path),
        "params": {
            "min_gap_sec": min_gap_sec, "max_gap_sec": max_gap_sec,
            "silence_threshold_db": silence_threshold_db,
            "min_silence_dur": min_silence_dur,
            "whisper_model": whisper_model if transcribe else None,
            "whisper_language": whisper_lang,
            "scene_detect": use_scene_detect,
        },
        "timing": timing,
        "total_gaps": len(gaps),
        "generated_count": len(entries),
        "preprocess_time_sec": timing.get("vad_scan_sec", 0) + timing.get("whisper_transcribe_sec", 0),
        "inference_total_time_sec": timing.get("inference_total_sec", 0),
        "total_time_sec": timing["total_sec"],
        "time_per_video_sec": round(timing["total_sec"] / max(video_duration_sec, 1), 2),
        "video_duration_sec": video_duration_sec,
        "num_scene_boundaries": len(scene_boundaries),
        "num_speaker_turns": len(trans_turns) if trans_turns else 0,
        "ad_entries": entries,
    }

    out_file = output_dir / f"{movie_name}_ad_output.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Saved: {out_file}")
    print(f"  Summary: {len(entries)}/{len(gaps)} ADs")
    print(f"  Timing: total={_fmt_sec(timing['total_sec'])}, "
          f"whisper={_fmt_sec(timing.get('whisper_transcribe_sec', 0))}, "
          f"vad={_fmt_sec(timing.get('vad_scan_sec', 0))}, "
          f"infer={_fmt_sec(timing['inference_total_sec'])}")

    return result


def _extract_nearby_turns(
    trans_turns: Optional[List[SpeakerTurn]],
    gap_start: float,
    gap_end: float,
    window_before: float = 45.0,
    window_after: float = 20.0,
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


# ── Entry Point ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Method 2: Script-Free AD Generation")
    parser.add_argument("--video", nargs="*", default=None)
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-gap-sec", type=float, default=DEFAULT_MIN_GAP_SEC)
    parser.add_argument("--max-gap-sec", type=float, default=DEFAULT_MAX_GAP_SEC)
    parser.add_argument("--silence-threshold-db", type=float, default=DEFAULT_SILENCE_DB)
    parser.add_argument("--min-silence-dur", type=float, default=DEFAULT_SILENCE_DUR)
    parser.add_argument("--no-transcribe", action="store_true")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER)
    parser.add_argument("--no-scene-detect", action="store_true", default=DEFAULT_NO_SCENE_DETECT)
    parser.add_argument("--force-recompute", action="store_true", help="Skip cache, recompute all analysis")
    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--only-movie", default=DEFAULT_ONLY_MOVIE)
    args = parser.parse_args()

    video_list = [Path(v).resolve() for v in args.video] if args.video else []
    videos = find_videos(
        video_paths=video_list if video_list else None,
        video_dir=Path(args.video_dir) if args.video_dir else None,
        only_movie=args.only_movie,
    )
    if not videos:
        print("ERROR: No videos found."); sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    do_transcribe = not args.no_transcribe

    print("=" * 60)
    print("Method 2: Script-Free AD Generation")
    print(f"  Videos:      {len(videos)}")
    print(f"  Gap range:   {args.min_gap_sec}s – {args.max_gap_sec}s")
    print(f"  Silence:     {args.silence_threshold_db}dB / {args.min_silence_dur}s")
    print(f"  Transcribe:  {do_transcribe} ({args.whisper_model})")
    print(f"  Scene det:   {not args.no_scene_detect}")
    print(f"  Output:      {output_dir}")
    print(f"  GPU:         {args.gpu_id}")
    print("=" * 60)

    os.environ.setdefault("WHISPER_CACHE_DIR", str(PROJECT_ROOT / "models" / "whisper"))

    print("\n--- Loading Video-LLaMA ---")
    from ad_engine import build_ad_engine
    engine = build_ad_engine(gpu_id=args.gpu_id)
    print()

    all_results = []
    for vi, vp in enumerate(videos):
        print(f"\n[{_pct(vi+1, len(videos))}] {vp.stem}")
        try:
            r = process_one_video(
                engine, video_path=vp, output_dir=output_dir,
                min_gap_sec=args.min_gap_sec, max_gap_sec=args.max_gap_sec,
                silence_threshold_db=args.silence_threshold_db,
                min_silence_dur=args.min_silence_dur,
                use_scene_detect=not args.no_scene_detect,
                transcribe=do_transcribe, whisper_model=args.whisper_model,
                temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                gpu_id=args.gpu_id,
                force_recompute=args.force_recompute,
            )
            all_results.append(r)
            print(f"  [{_pct(vi+1, len(videos))}] {r['movie']}: "
                  f"{r['generated_count']}/{r['total_gaps']} ADs")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    total_g = sum(r.get("total_gaps", 0) for r in all_results)
    total_a = sum(r.get("generated_count", 0) for r in all_results)
    print(f"\n=== Done: {total_a}/{total_g} ADs across {len(all_results)} videos ===")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
