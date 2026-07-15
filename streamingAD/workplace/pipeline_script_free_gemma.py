#!/usr/bin/env python3
"""
pipeline_script_free_gemma.py — Script-Free Audio Description with Gemma4

Same flow as pipeline_script_free.py but uses Gemma4 26B-A4B instead of Video-LLaMA.
Imports post-processing, prompt, and engine from pipeline_enhanced_gemma4.py.

Usage:
    python streamingAD/workplace/pipeline_script_free_gemma.py
    python streamingAD/workplace/pipeline_script_free_gemma.py --fast
"""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image as PILImage
from io import BytesIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(STREAMING_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "streamingAD"))

# Import from pipeline_enhanced_gemma4
from pipeline_enhanced_gemma4 import (
    Gemma4ADEngine, postprocess_ad, TASK_PROMPT,
    build_gemma4_engine, GEMMA4_MODEL_PATH,
)

from vad_gap_detector import (
    AudioGap, VideoSceneBoundary,
    detect_gaps_from_video, extract_clip, _prepare_fast_video,
    _sec_to_timestamp, _run_ffmpeg_silence_detect,
)
from speech_transcriber import (
    SpeakerTurn, TranscriptionResult,
    transcribe_video, turns_near_timestamp, turns_to_context_text,
    DEFAULT_WHISPER_MODEL,
)
from analysis_cache import load_analysis, save_analysis

# ── Config ───────────────────────────────────────────────────
DEFAULT_VIDEO_DIR: str = "/mnt/disk1new/storyvideo/Movie"
DEFAULT_OUTPUT_DIR: str = "/mnt/disk1new/ylz/newAD/compare/script_free_gemma"
DEFAULT_MIN_GAP_SEC: float = 4.0
DEFAULT_MAX_GAP_SEC: float = 60.0
DEFAULT_SILENCE_DB: float = -30.0
DEFAULT_SILENCE_DUR: float = 1.5
DEFAULT_WHISPER: str = "turbo"
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_MAX_TOKENS: int = 128
DEFAULT_GPU: str = "0,2,3"
DEFAULT_ONLY_MOVIE: str = "Shawshank"
DEFAULT_NO_SCENE_DETECT: bool = True
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts"}


def _pct(a, b): return f"{a}/{b}" if b else "0"
def _fmt_sec(s): return f"{s:.1f}s"


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
    if not scenes: return None
    best_ts, best_dist = None, float("inf")
    mid = (gap.gap_start_sec + gap.gap_end_sec) / 2
    for b in scenes:
        dist = abs(b.timestamp_sec - mid)
        if dist < best_dist and dist < window_sec:
            best_dist, best_ts = dist, b.timestamp_sec
    return best_ts


def _extract_nearby_turns(trans_turns, gap_start, gap_end,
                          window_before=45.0, window_after=20.0):
    if not trans_turns: return []
    return turns_near_timestamp(trans_turns, (gap_start + gap_end) / 2,
                                window_before_sec=window_before,
                                window_after_sec=window_after)


def _dialogue_before(turns, gap_start, max_lines=5):
    before = [t for t in turns if t.end_sec <= gap_start]
    before.sort(key=lambda t: t.start_sec)
    return [f"{t.speaker_label}: {t.text}" for t in before[-max_lines:]]


def _dialogue_after(turns, gap_end, max_lines=5):
    after = [t for t in turns if t.start_sec >= gap_end]
    after.sort(key=lambda t: t.start_sec)
    return [f"{t.speaker_label}: {t.text}" for t in after[:max_lines]]


def _build_context(gap, trans_turns, nearest_scene, scene_boundaries,
                   detected_characters=None):
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
            f"Focus on actions, gestures, facial expressions, camera movements."
        )
    if detected_characters:
        ctx = f"[Characters visible in this clip: {', '.join(detected_characters)}]\n\n{ctx}"
    if nearest_scene is not None:
        score = 0.0
        for b in scene_boundaries:
            if abs(b.timestamp_sec - nearest_scene) < 0.01:
                score = b.score; break
        ctx += (f"\n\n[Visual cue]\nA visual scene transition occurs near this gap "
                f"(at {nearest_scene:.1f}s, score={score:.2f}).")
    return ctx


def process_one_video(
    engine: Gemma4ADEngine,
    video_path: Path,
    output_dir: Path,
    fast_mode: bool = False,
    max_gaps: int = 0,
    min_gap_sec: float = DEFAULT_MIN_GAP_SEC,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
    silence_threshold_db: float = DEFAULT_SILENCE_DB,
    min_silence_dur: float = DEFAULT_SILENCE_DUR,
    use_scene_detect: bool = True,
    transcribe: bool = True,
    whisper_model: str = DEFAULT_WHISPER,
    temperature: float = DEFAULT_TEMPERATURE,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    movie_name = video_path.stem
    pipeline_start = time.monotonic()
    video_duration_sec = _get_video_duration(video_path)

    print(f"\n{'='*60}")
    mode_str = "FAST (1 candidate, beam)" if fast_mode else "FULL (3 candidates)"
    print(f"[Script-Free Gemma4] {movie_name} [{mode_str}]")
    print(f"  Video: {video_path}")
    print(f"  Size:  {video_path.stat().st_size / 1e9:.1f}GB")
    print(f"{'='*60}")

    timing: Dict[str, float] = {}
    trans_turns: Optional[List[SpeakerTurn]] = None
    whisper_lang: str = ""
    gaps: List[AudioGap] = []
    scene_boundaries: List[VideoSceneBoundary] = []

    analysis_params = {
        "min_gap_sec": min_gap_sec, "max_gap_sec": max_gap_sec,
        "silence_threshold_db": silence_threshold_db,
        "min_silence_dur": min_silence_dur,
    }

    # ══════ Phase A: Analysis (cached) ══════
    cached_turns, cached_lang, cached_gaps, cached_scenes = load_analysis(
        output_dir, movie_name, min_gap_sec, silence_threshold_db, min_silence_dur,
    )
    if cached_turns is not None and cached_gaps is not None and not force_recompute:
        print(f"  [Cache] Loaded: {len(cached_turns)} turns, {len(cached_gaps)} gaps")
        trans_turns = cached_turns
        whisper_lang = cached_lang
        gaps.extend(cached_gaps)
        if cached_scenes: scene_boundaries.extend(cached_scenes)
    else:
        print(f"  [Cache] Not found, computing...")
        events = _run_ffmpeg_silence_detect(
            video_path=video_path, silence_threshold_db=silence_threshold_db,
            min_silence_dur=min_silence_dur,
        )
        timing["vad_scan_sec"] = round(time.monotonic() - time.monotonic(), 1)

        if transcribe:
            t0 = time.monotonic()
            print(f"  [Whisper] Transcribing with '{whisper_model}'...")
            try:
                trans_result = transcribe_video(
                    video_path=video_path, model_size=whisper_model,
                    silence_threshold_db=silence_threshold_db,
                    min_silence_dur=min_silence_dur, silence_events=events,
                )
                trans_turns = trans_result.turns
                whisper_lang = trans_result.language
                timing["whisper_sec"] = round(time.monotonic() - t0, 1)
                print(f"    Done: {len(trans_turns)} turns, lang={whisper_lang}")
            except Exception as e:
                print(f"    Whisper ERROR: {e}")
                traceback.print_exc()

        if events is not None:
            import vad_gap_detector as _vgd
            gaps_list = _vgd._silence_events_to_gaps(
                events, min_gap_sec=min_gap_sec, max_gap_sec=max_gap_sec)
            gaps.extend(gaps_list)
            if use_scene_detect:
                t0 = time.monotonic()
                scene_boundaries = _vgd._detect_scene_boundaries(video_path)
                gaps = _vgd._merge_gaps_with_scenes(gaps, scene_boundaries)
                timing["scene_sec"] = round(time.monotonic() - t0, 1)

        print(f"    {len(gaps)} gaps found")
        if trans_turns is not None or gaps:
            save_analysis(output_dir, movie_name, whisper_turns=trans_turns,
                          whisper_lang=whisper_lang, gaps=gaps,
                          scene_boundaries=scene_boundaries, params=analysis_params)

    if not gaps:
        print("  [WARN] No gaps found")
        return {"movie": movie_name, "status": "no_gaps", "ad_entries": []}

    if max_gaps > 0:
        gaps = gaps[:max_gaps]
        print(f"  [Limit] Processing first {max_gaps} gaps only")

    # ══════ Phase A4: Face gallery ══════
    try:
        from face_gallery import load_gallery
        gallery_embs, gallery_people = load_gallery(movie_name)
        if gallery_embs is not None:
            print(f"  [Face] {len(gallery_people)} characters loaded")
        else:
            print(f"  [Face] Not available")
    except Exception:
        gallery_embs, gallery_people = None, None

    # ══════ Phase B: Generation ══════
    clip_tmp = output_dir / f".tmp_{movie_name}"
    clip_tmp.mkdir(parents=True, exist_ok=True)
    fast_video = _prepare_fast_video(video_path, clip_tmp)

    entries: List[Dict[str, Any]] = []
    inference_total = 0.0

    for idx, gap in enumerate(gaps):
        clip_path = clip_tmp / f"gap{gap.gap_id:04d}.mp4"
        if not extract_clip(fast_video, gap.gap_start_sec, gap.gap_end_sec, clip_path):
            continue

        nearest_scene = _find_nearest_scene(gap, scene_boundaries)

        chars: List[str] = []
        face_avatars: List[Path] = []
        if gallery_embs is not None and gallery_people is not None:
            try:
                from face_gallery import detect_faces_in_clip
                chars, face_avatars = detect_faces_in_clip(
                    clip_path, gallery_embs, gallery_people)
            except Exception: pass

        context_text = _build_context(
            gap, trans_turns, nearest_scene, scene_boundaries,
            detected_characters=chars)

        # Generate AD: fast=1 beam, full=3 candidates
        if fast_mode:
            try:
                ad_text, inf_time, _ = engine.infer_one_segment(
                    clip_path=clip_path, context_text=context_text,
                    task_prompt=TASK_PROMPT, temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    face_avatars=face_avatars, character_names=chars,
                    num_beams=3)
                ad_text = postprocess_ad(ad_text)
            except Exception as e:
                ad_text, inf_time = f"[ERROR: {e}]", 0.0
        else:
            # Full mode: 3 candidates + self selection
            candidates = []
            total_inf = 0.0
            for c in range(3):
                try:
                    if c == 0:
                        txt, t, _ = engine.infer_one_segment(
                            clip_path=clip_path, context_text=context_text,
                            task_prompt=TASK_PROMPT, temperature=temperature,
                            max_new_tokens=max_new_tokens,
                            face_avatars=face_avatars, character_names=chars,
                            num_beams=3)
                        candidates.append(txt)
                    else:
                        temp = temperature + c * 0.2
                        txt, t, _ = engine.infer_one_segment(
                            clip_path=clip_path, context_text=context_text,
                            task_prompt=TASK_PROMPT, temperature=temp,
                            max_new_tokens=max_new_tokens,
                            face_avatars=face_avatars, character_names=chars,
                            num_beams=1)
                        candidates.append(txt)
                    total_inf += t
                except Exception as e:
                    print(f"      [WARN] candidate {c} failed: {e}")

            processed = [postprocess_ad(c) for c in candidates]
            processed = [c for c in processed if c]
            ad_text = engine.select_best_candidate(processed) if processed else ""
            inf_time = total_inf

        inference_total += inf_time

        nearby_turns = _extract_nearby_turns(trans_turns, gap.gap_start_sec, gap.gap_end_sec)
        entries.append({
            "gap_id": gap.gap_id,
            "gap_start_sec": round(gap.gap_start_sec, 3),
            "gap_end_sec": round(gap.gap_end_sec, 3),
            "gap_duration_sec": round(gap.gap_duration_sec, 1),
            "ad_text": ad_text,
            "characters": chars,
            "context_before": _dialogue_before(nearby_turns, gap.gap_start_sec),
            "context_after": _dialogue_after(nearby_turns, gap.gap_end_sec),
            "inference_time_sec": round(inf_time, 1),
        })

        print(f"  [{_pct(idx+1, len(gaps))}] Gap {gap.gap_id} "
              f"({gap.gap_duration_sec:.1f}s): {ad_text}")

        if clip_path.exists(): clip_path.unlink()

    shutil.rmtree(str(clip_tmp), ignore_errors=True)

    timing["inference_total_sec"] = round(inference_total, 1)
    timing["total_sec"] = round(time.monotonic() - pipeline_start, 1)

    result = {
        "movie": movie_name,
        "method": "script_free_gemma4",
        "mode": "fast" if fast_mode else "full",
        "video_path": str(video_path),
        "timing": timing,
        "total_gaps": len(gaps),
        "generated_count": len(entries),
        "inference_total_time_sec": timing["inference_total_sec"],
        "total_time_sec": timing["total_sec"],
        "ad_entries": entries,
    }
    out_file = output_dir / f"{movie_name}_gemma4_output.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {out_file}")
    print(f"  {len(entries)}/{len(gaps)} ADs, infer={_fmt_sec(inference_total)}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Script-Free AD with Gemma4")
    parser.add_argument("--video", nargs="*", default=None)
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only-movie", default=DEFAULT_ONLY_MOVIE)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--min-gap-sec", type=float, default=DEFAULT_MIN_GAP_SEC)
    parser.add_argument("--max-gap-sec", type=float, default=DEFAULT_MAX_GAP_SEC)
    parser.add_argument("--silence-threshold-db", type=float, default=DEFAULT_SILENCE_DB)
    parser.add_argument("--min-silence-dur", type=float, default=DEFAULT_SILENCE_DUR)
    parser.add_argument("--no-transcribe", action="store_true")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER)
    parser.add_argument("--no-scene-detect", action="store_true", default=DEFAULT_NO_SCENE_DETECT)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--model-path", default=GEMMA4_MODEL_PATH)
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 1 candidate beam search, no selector")
    parser.add_argument("--max-gaps", type=int, default=0,
                        help="Limit number of gaps to process (0 = all)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("WHISPER_CACHE_DIR", str(PROJECT_ROOT / "models" / "whisper"))

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

    print(f"\n{'='*60}")
    print(f"Script-Free AD — Gemma4 {'[FAST]' if args.fast else '[FULL]'}")
    print(f"  Videos: {len(videos)}, GPU: {args.gpu}")
    print(f"  Gap: {args.min_gap_sec}s – {args.max_gap_sec}s")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    print("\n--- Loading Gemma4 ---")
    engine = build_gemma4_engine(model_path=args.model_path)
    print()

    all_results = []
    for vi, vp in enumerate(videos):
        try:
            r = process_one_video(
                engine, video_path=vp, output_dir=output_dir,
                fast_mode=args.fast, max_gaps=args.max_gaps,
                min_gap_sec=args.min_gap_sec, max_gap_sec=args.max_gap_sec,
                silence_threshold_db=args.silence_threshold_db,
                min_silence_dur=args.min_silence_dur,
                use_scene_detect=not args.no_scene_detect,
                transcribe=not args.no_transcribe,
                whisper_model=args.whisper_model,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
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


if __name__ == "__main__":
    main()
