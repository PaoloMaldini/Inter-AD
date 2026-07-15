#!/usr/bin/env python3
"""
Gemma4 + Qwen interactive AD experiment (script-free).

Uses silence gaps + Whisper context + scene boundaries + face retrieval, while
simulating random user instructions during playback. Outputs evaluator-
compatible `*_ad_output.json`.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKPLACE_DIR = SCRIPT_DIR / "workplace"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WORKPLACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKPLACE_DIR))

from analysis_cache import load_analysis, save_analysis
from face_gallery import detect_faces_in_clip, load_gallery
from gemma_qwen_interactive_common import (
    DEFAULT_GEMMA4_MODEL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_WORD_BUDGET_RATIO,
    DEFAULT_AD_WORDS_PER_MINUTE,
    build_runtime_task_prompt,
    compute_dynamic_word_budget,
    InstructionState,
    QwenInstructionRewriter,
    build_composite_request,
    build_gemma4_engine_multi_gpu,
    generate_ad,
    get_video_duration,
    infer_primary_slot_for_request,
    load_instruction_categories,
    safe_movie_slug,
    save_experiment_result,
    select_insertion_indices,
)
from speech_transcriber import DEFAULT_WHISPER_MODEL, SpeakerTurn, transcribe_video
from vad_gap_detector import _prepare_fast_video, _run_ffmpeg_silence_detect, _silence_events_to_gaps, extract_clip
from workplace.pipeline_script_free_gemma import (
    DEFAULT_MAX_GAP_SEC,
    DEFAULT_MIN_GAP_SEC,
    DEFAULT_SILENCE_DB,
    DEFAULT_SILENCE_DUR,
    _build_context,
    _dialogue_after,
    _dialogue_before,
    _extract_nearby_turns,
    _find_nearest_scene,
    find_videos,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiment_results" / "gemma4_qwen_interactive_scriptfree"
DEFAULT_NUM_INSERTIONS = 12
DEFAULT_MIN_INSERTIONS = 10
DEFAULT_MAX_INSERTIONS = 15


def _fmt_sec(value: float) -> str:
    return f"{value:.1f}s"


def _preview(text: str, limit: int = 120) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@dataclass
class InsertionRecord:
    insertion_id: int
    movie_title: str
    insert_timestamp_sec: float
    segment_idx: int
    segment_start_sec: float
    segment_end_sec: float
    instruction_text: str
    category_id: str
    category_name: str
    instruction_language: str
    active_instructions_count: int
    instruction_before: str
    instruction_after: str
    all_active_instructions: List[Dict[str, Any]]
    text_before: str
    text_after: str
    latency_before_sec: float
    latency_after_sec: float
    ttff_before_sec: float
    ttff_after_sec: float
    context_text: str = ""
    characters: List[str] = field(default_factory=list)
    ref_ad: str = ""


@dataclass
class SegmentRecord:
    segment_idx: int
    segment_start_sec: float
    segment_end_sec: float
    clip_path: str
    generated_text: str
    instruction_text: str
    active_instructions_count: int
    latency_sec: float
    task_prompt: str
    context_text: str = ""
    characters: List[str] = field(default_factory=list)
    ref_ad: str = ""


@dataclass
class MovieExperimentResult:
    movie_title: str
    movie_duration_sec: float
    total_segments: int
    num_insertions: int
    insertion_records: List[InsertionRecord]
    segment_records: List[SegmentRecord]
    ad_entries: List[Dict[str, Any]]
    run_config: Dict[str, Any]


def _compute_gaps_and_context(
    *,
    video_path: Path,
    output_dir: Path,
    min_gap_sec: float,
    max_gap_sec: float,
    silence_threshold_db: float,
    min_silence_dur: float,
    whisper_model: str,
    force_recompute: bool,
):
    print(f"[analysis] movie={video_path.stem}", flush=True)
    trans_turns: Optional[List[SpeakerTurn]] = None
    whisper_lang = ""
    gaps = []
    scene_boundaries = []

    cached_turns, cached_lang, cached_gaps, cached_scenes = load_analysis(
        output_dir, video_path.stem, min_gap_sec, silence_threshold_db, min_silence_dur
    )
    if cached_turns is not None and cached_gaps is not None and not force_recompute:
        print("[analysis] using_cached_analysis", flush=True)
        trans_turns = cached_turns
        whisper_lang = cached_lang
        gaps = list(cached_gaps)
        scene_boundaries = list(cached_scenes or [])
        print(
            f"[analysis] cached_turns={len(trans_turns) if trans_turns else 0} cached_gaps={len(gaps)} "
            f"cached_scenes={len(scene_boundaries)} whisper_lang={whisper_lang or 'unknown'}",
            flush=True,
        )
        return trans_turns, whisper_lang, gaps, scene_boundaries

    print(
        f"[analysis] running_silence_detect threshold_db={silence_threshold_db} min_silence_dur={min_silence_dur}",
        flush=True,
    )
    events = _run_ffmpeg_silence_detect(
        video_path=video_path,
        silence_threshold_db=silence_threshold_db,
        min_silence_dur=min_silence_dur,
    )
    print(f"[analysis] silence_events={len(events)}", flush=True)
    try:
        print(f"[analysis] running_whisper model={whisper_model}", flush=True)
        trans_result = transcribe_video(
            video_path=video_path,
            model_size=whisper_model,
            silence_threshold_db=silence_threshold_db,
            min_silence_dur=min_silence_dur,
            silence_events=events,
        )
        trans_turns = trans_result.turns
        whisper_lang = trans_result.language
        print(
            f"[analysis] whisper_done turns={len(trans_turns) if trans_turns else 0} lang={whisper_lang or 'unknown'}",
            flush=True,
        )
    except Exception as exc:
        trans_turns = None
        whisper_lang = ""
        print(f"[analysis] whisper_failed error={exc}", flush=True)

    gaps = _silence_events_to_gaps(events, min_gap_sec=min_gap_sec, max_gap_sec=max_gap_sec)
    print(f"[analysis] raw_gaps={len(gaps)}", flush=True)

    import vad_gap_detector as _vgd

    print("[analysis] detecting_scene_boundaries", flush=True)
    scene_boundaries = _vgd._detect_scene_boundaries(video_path)
    print(f"[analysis] scene_boundaries={len(scene_boundaries)}", flush=True)
    gaps = _vgd._merge_gaps_with_scenes(gaps, scene_boundaries)
    print(f"[analysis] merged_gaps={len(gaps)}", flush=True)

    save_analysis(
        output_dir=output_dir,
        movie_name=video_path.stem,
        whisper_turns=trans_turns,
        whisper_lang=whisper_lang,
        gaps=gaps,
        scene_boundaries=scene_boundaries,
        params={
            "min_gap_sec": min_gap_sec,
            "max_gap_sec": max_gap_sec,
            "silence_threshold_db": silence_threshold_db,
            "min_silence_dur": min_silence_dur,
        },
    )
    print("[analysis] cache_saved", flush=True)
    return trans_turns, whisper_lang, gaps, scene_boundaries


def _resolve_num_insertions(args: argparse.Namespace, rng: random.Random, num_gaps: int) -> int:
    if args.min_insertions is not None or args.max_insertions is not None:
        low = args.min_insertions if args.min_insertions is not None else args.num_insertions
        high = args.max_insertions if args.max_insertions is not None else args.num_insertions
        if low > high:
            raise ValueError(f"min_insertions > max_insertions: {low} > {high}")
        chosen = rng.randint(low, high)
    else:
        chosen = args.num_insertions
    return max(0, min(chosen, num_gaps))


def run_one_movie(
    *,
    args: argparse.Namespace,
    video_path: Path,
    categories,
    qwen: QwenInstructionRewriter,
    gemma,
    rng: random.Random,
) -> MovieExperimentResult:
    movie_title = video_path.stem
    t0 = time.monotonic()
    print(f"[movie] {movie_title}", flush=True)
    print(f"[movie] video={video_path}", flush=True)
    video_duration_sec = get_video_duration(video_path)
    print(f"[movie] duration={_fmt_sec(video_duration_sec)}", flush=True)
    movie_output_dir = Path(args.output_dir)
    state = InstructionState()
    state.refresh_task_prompt(qwen)

    trans_turns, whisper_lang, gaps, scene_boundaries = _compute_gaps_and_context(
        video_path=video_path,
        output_dir=movie_output_dir,
        min_gap_sec=args.min_gap_sec,
        max_gap_sec=args.max_gap_sec,
        silence_threshold_db=args.silence_threshold_db,
        min_silence_dur=args.min_silence_dur,
        whisper_model=args.whisper_model,
        force_recompute=args.force_recompute,
    )
    if not gaps:
        raise RuntimeError(f"No script-free gaps found for {movie_title}")
    print(
        f"[movie] gaps={len(gaps)} whisper_lang={whisper_lang or 'unknown'} scenes={len(scene_boundaries)}",
        flush=True,
    )

    gallery_embs, gallery_people = load_gallery(movie_title)
    if gallery_embs is None or gallery_people is None:
        print("[movie] face_gallery=unavailable", flush=True)
    else:
        print(f"[movie] face_gallery={len(gallery_people)} characters", flush=True)
    num_insertions = _resolve_num_insertions(args, rng, len(gaps))
    insertion_indices = select_insertion_indices(len(gaps), num_insertions, rng)
    insertion_set = set(insertion_indices)
    print(f"[movie] num_insertions={num_insertions}", flush=True)
    print(f"[movie] insertion_indices={insertion_indices}", flush=True)
    print(f"[model] qwen_gpu={args.qwen_gpu} gemma_gpus={args.gemma_gpus}", flush=True)

    clip_tmp = movie_output_dir / f".tmp_scriptfree_{safe_movie_slug(movie_title)}"
    clip_tmp.mkdir(parents=True, exist_ok=True)
    print(f"[movie] preparing_fast_video tmp={clip_tmp}", flush=True)
    fast_video = _prepare_fast_video(video_path, clip_tmp)
    print("[movie] fast_video_ready", flush=True)

    insertion_records: List[InsertionRecord] = []
    segment_records: List[SegmentRecord] = []
    ad_entries: List[Dict[str, Any]] = []
    insertion_events: List[Dict[str, Any]] = []
    insertion_counter = 0
    all_characters: List[str] = []

    try:
        for seg_idx, gap in enumerate(gaps):
            seg_no = seg_idx + 1
            print(
                f"[segment {seg_no}/{len(gaps)}] gap_id={gap.gap_id} start={_fmt_sec(gap.gap_start_sec)} "
                f"end={_fmt_sec(gap.gap_end_sec)} dur={gap.gap_duration_sec:.1f}s active={len(state.snapshot())}",
                flush=True,
            )
            clip_path = clip_tmp / f"gap{gap.gap_id:04d}.mp4"
            print(f"[segment {seg_no}/{len(gaps)}] extracting_clip -> {clip_path.name}", flush=True)
            if not extract_clip(fast_video, gap.gap_start_sec, gap.gap_end_sec, clip_path):
                print(f"[segment {seg_no}/{len(gaps)}] extract_clip_failed", flush=True)
                continue
            print(f"[segment {seg_no}/{len(gaps)}] clip_ready", flush=True)

            chars: List[str] = []
            face_avatars: List[Path] = []
            if gallery_embs is not None and gallery_people is not None:
                try:
                    chars, face_avatars = detect_faces_in_clip(clip_path, gallery_embs, gallery_people)
                except Exception:
                    chars, face_avatars = [], []
            print(
                f"[segment {seg_no}/{len(gaps)}] detected_characters={chars if chars else '[]'}",
                flush=True,
            )
            for name in chars:
                if name and name not in all_characters:
                    all_characters.append(name)

            nearest_scene = _find_nearest_scene(gap, scene_boundaries)
            context_text = _build_context(
                gap,
                trans_turns,
                nearest_scene,
                scene_boundaries,
                detected_characters=chars,
            )
            print(
                f"[segment {seg_no}/{len(gaps)}] context={_preview(context_text, 100)}",
                flush=True,
            )

            final_text = ""
            final_latency = 0.0
            final_task_prompt = state.current_task_prompt
            dynamic_max_words = compute_dynamic_word_budget(
                gap.gap_duration_sec,
                words_per_minute=args.words_per_minute,
                budget_ratio=args.word_budget_ratio,
                min_words=args.min_words_per_gap,
                hard_cap=args.max_words if args.max_words and args.max_words > 0 else None,
            )
            runtime_task_prompt = build_runtime_task_prompt(
                final_task_prompt,
                gap_duration_sec=gap.gap_duration_sec,
                max_words=dynamic_max_words,
            )
            print(
                f"[segment {seg_no}/{len(gaps)}] word_budget={dynamic_max_words} "
                f"(gap={gap.gap_duration_sec:.1f}s, wpm={args.words_per_minute}, ratio={args.word_budget_ratio})",
                flush=True,
            )

            if seg_idx in insertion_set:
                insertion_counter += 1
                request_meta = build_composite_request(
                    categories=categories,
                    rng=rng,
                    all_characters=all_characters,
                    min_parts=args.min_request_parts,
                    max_parts=args.max_request_parts,
                )
                request_text = request_meta["request_text"]
                primary_slot = infer_primary_slot_for_request(request_meta)
                instruction_before = state.raw_instruction_text()
                active_before = state.snapshot()
                print(
                    f"*** INSERT #{insertion_counter} seg={seg_idx} t={_fmt_sec(gap.gap_start_sec)} "
                    f"cat={request_meta['category_id']} slot={primary_slot}",
                    flush=True,
                )
                print(f"  raw={request_text}", flush=True)
                print(f"  active_before={len(active_before)}", flush=True)

                print(f"[segment {seg_no}/{len(gaps)}] generating_baseline", flush=True)
                before_text, before_latency, _ = generate_ad(
                    gemma,
                    clip_path=clip_path,
                    context_text=context_text,
                    task_prompt=runtime_task_prompt,
                    face_avatars=face_avatars,
                    character_names=chars,
                    max_words=dynamic_max_words,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                )
                print(
                    f"[segment {seg_no}/{len(gaps)}] baseline_done latency={before_latency:.2f}s "
                    f"text={_preview(before_text)}",
                    flush=True,
                )

                print(f"[segment {seg_no}/{len(gaps)}] rewriting_request_with_qwen", flush=True)
                edit = qwen.plan_instruction_edit(
                    prompt_state=state.prompt_state,
                    latest_request=request_text,
                    primary_slot=primary_slot,
                )
                touched_slots = state.apply_edit(edit)
                effective_slots = [
                    slot
                    for slot in touched_slots
                    if slot == "others" or getattr(state.prompt_state, slot, "")
                ]
                state.sync_instruction_logs(
                    category_id=request_meta["category_id"],
                    category_name=request_meta["category_name"],
                    slot_ids=effective_slots,
                    raw_text=request_text,
                    language=request_meta["request_language"],
                    insert_timestamp_sec=gap.gap_start_sec,
                    insert_segment_idx=seg_idx,
                    character_name=None,
                )
                final_task_prompt = state.refresh_task_prompt(qwen)
                runtime_task_prompt = build_runtime_task_prompt(
                    final_task_prompt,
                    gap_duration_sec=gap.gap_duration_sec,
                    max_words=dynamic_max_words,
                )
                print(
                    f"[segment {seg_no}/{len(gaps)}] qwen_slots={effective_slots} focus={state.current_focus_line}",
                    flush=True,
                )
                print(f"[segment {seg_no}/{len(gaps)}] prompt_after:", flush=True)
                print(runtime_task_prompt, flush=True)

                print(f"[segment {seg_no}/{len(gaps)}] generating_after_insert", flush=True)
                after_text, after_latency, _ = generate_ad(
                    gemma,
                    clip_path=clip_path,
                    context_text=context_text,
                    task_prompt=runtime_task_prompt,
                    face_avatars=face_avatars,
                    character_names=chars,
                    max_words=dynamic_max_words,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                )

                final_text = after_text
                final_latency = after_latency
                print(
                    f"[segment {seg_no}/{len(gaps)}] after_done latency={after_latency:.2f}s "
                    f"text={_preview(after_text)}",
                    flush=True,
                )

                insertion_records.append(
                    InsertionRecord(
                        insertion_id=insertion_counter,
                        movie_title=movie_title,
                        insert_timestamp_sec=gap.gap_start_sec,
                        segment_idx=seg_idx,
                        segment_start_sec=gap.gap_start_sec,
                        segment_end_sec=gap.gap_end_sec,
                        instruction_text=request_text,
                        category_id=request_meta["category_id"],
                        category_name=request_meta["category_name"],
                        instruction_language=request_meta["request_language"],
                        active_instructions_count=len(state.snapshot()),
                        instruction_before=instruction_before,
                        instruction_after=state.raw_instruction_text(),
                        all_active_instructions=state.snapshot(),
                        text_before=before_text,
                        text_after=after_text,
                        latency_before_sec=round(before_latency, 4),
                        latency_after_sec=round(after_latency, 4),
                        ttff_before_sec=round(before_latency, 4),
                        ttff_after_sec=round(after_latency, 4),
                        context_text=context_text[:600],
                        characters=chars,
                        ref_ad="",
                    )
                )
                insertion_events.append(
                    {
                        "insertion_id": insertion_counter,
                        "gap_idx": seg_idx,
                        "segment_idx": seg_idx,
                        "timestamp_sec": round(gap.gap_start_sec, 3),
                        "category": request_meta["category_id"],
                        "category_id": request_meta["category_id"],
                        "category_name": request_meta["category_name"],
                        "instruction_text": request_text,
                        "language": request_meta["request_language"],
                        "components": request_meta["components"],
                        "task_prompt_after": final_task_prompt,
                        "prompt_state_after": state.prompt_state_dict(),
                        "focus_line_after": state.current_focus_line,
                        "active_before": active_before,
                        "all_active_instructions": state.snapshot(),
                        "whisper_language": whisper_lang,
                        "dynamic_max_words": dynamic_max_words,
                    }
                )
            else:
                print(f"[segment {seg_no}/{len(gaps)}] generating", flush=True)
                final_text, final_latency, _ = generate_ad(
                    gemma,
                    clip_path=clip_path,
                    context_text=context_text,
                    task_prompt=runtime_task_prompt,
                    face_avatars=face_avatars,
                    character_names=chars,
                    max_words=dynamic_max_words,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                )
                print(
                    f"[segment {seg_no}/{len(gaps)}] done latency={final_latency:.2f}s "
                    f"text={_preview(final_text)}",
                    flush=True,
                )

            nearby_turns = _extract_nearby_turns(trans_turns, gap.gap_start_sec, gap.gap_end_sec)

            segment_records.append(
                SegmentRecord(
                    segment_idx=seg_idx,
                    segment_start_sec=gap.gap_start_sec,
                    segment_end_sec=gap.gap_end_sec,
                    clip_path=str(clip_path),
                    generated_text=final_text,
                    instruction_text=state.raw_instruction_text(),
                    active_instructions_count=len(state.snapshot()),
                    latency_sec=round(final_latency, 4),
                    task_prompt=runtime_task_prompt,
                    context_text=context_text[:600],
                    characters=chars,
                    ref_ad="",
                )
            )

            ad_entries.append(
                {
                    "gap_id": gap.gap_id,
                    "scene_index": "",
                    "location": "UNKNOWN",
                    "gap_start_sec": round(gap.gap_start_sec, 3),
                    "gap_end_sec": round(gap.gap_end_sec, 3),
                    "gap_duration_sec": round(gap.gap_duration_sec, 3),
                    "characters": chars,
                    "context_before": _dialogue_before(nearby_turns, gap.gap_start_sec),
                    "context_after": _dialogue_after(nearby_turns, gap.gap_end_sec),
                    "ad_text": final_text,
                    "inference_time_sec": round(final_latency, 3),
                    "active_instructions": state.snapshot(),
                    "active_instruction_count": len(state.snapshot()),
                    "task_prompt": runtime_task_prompt,
                    "max_words": dynamic_max_words,
                }
            )
    finally:
        shutil.rmtree(str(clip_tmp), ignore_errors=True)
        print(f"[movie] cleaned_tmp={clip_tmp}", flush=True)

    elapsed = time.monotonic() - t0
    print(
        f"[movie] completed segments={len(ad_entries)} insertions={len(insertion_records)} elapsed={elapsed:.1f}s",
        flush=True,
    )

    return MovieExperimentResult(
        movie_title=movie_title,
        movie_duration_sec=video_duration_sec,
        total_segments=len(ad_entries),
        num_insertions=len(insertion_records),
        insertion_records=insertion_records,
        segment_records=segment_records,
        ad_entries=ad_entries,
        run_config={
            "method": "gemma4_qwen_interactive_scriptfree",
            "video_path": str(video_path),
            "min_gap_sec": args.min_gap_sec,
            "max_gap_sec": args.max_gap_sec,
            "silence_threshold_db": args.silence_threshold_db,
            "min_silence_dur": args.min_silence_dur,
            "whisper_model": args.whisper_model,
            "num_insertions": num_insertions,
            "min_insertions": args.min_insertions,
            "max_insertions": args.max_insertions,
            "min_request_parts": args.min_request_parts,
            "max_request_parts": args.max_request_parts,
            "temperature": args.temperature,
            "num_beams": args.num_beams,
            "max_words": args.max_words,
            "words_per_minute": args.words_per_minute,
            "word_budget_ratio": args.word_budget_ratio,
            "min_words_per_gap": args.min_words_per_gap,
            "seed": args.seed,
            "qwen_model": args.qwen_model,
            "gemma_model": args.gemma_model,
            "qwen_gpu": args.qwen_gpu,
            "gemma_gpus": args.gemma_gpus,
            "force_recompute": args.force_recompute,
            "insertion_events": insertion_events,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma4 + Qwen interactive AD experiment (script-free).")
    parser.add_argument("--video-dir", type=str, default="/mnt/disk1new/storyvideo/Movie")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--only-movie", type=str, default=None)
    parser.add_argument("--min-gap-sec", type=float, default=DEFAULT_MIN_GAP_SEC)
    parser.add_argument("--max-gap-sec", type=float, default=DEFAULT_MAX_GAP_SEC)
    parser.add_argument("--silence-threshold-db", type=float, default=DEFAULT_SILENCE_DB)
    parser.add_argument("--min-silence-dur", type=float, default=DEFAULT_SILENCE_DUR)
    parser.add_argument("--whisper-model", type=str, default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--instruction-config", type=str, default="")
    parser.add_argument("--specific-categories", nargs="*", default=None)
    parser.add_argument("--num-insertions", type=int, default=DEFAULT_NUM_INSERTIONS)
    parser.add_argument("--min-insertions", type=int, default=DEFAULT_MIN_INSERTIONS)
    parser.add_argument("--max-insertions", type=int, default=DEFAULT_MAX_INSERTIONS)
    parser.add_argument("--min-request-parts", type=int, default=1)
    parser.add_argument("--max-request-parts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=0)
    parser.add_argument("--words-per-minute", type=int, default=DEFAULT_AD_WORDS_PER_MINUTE)
    parser.add_argument("--word-budget-ratio", type=float, default=DEFAULT_WORD_BUDGET_RATIO)
    parser.add_argument("--min-words-per-gap", type=int, default=5)
    parser.add_argument("--gemma-model", type=str, default=DEFAULT_GEMMA4_MODEL)
    parser.add_argument("--qwen-model", type=str, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--gemma-gpus", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--gemma-max-memory-gib", type=int, nargs="+", default=None)
    parser.add_argument("--gemma-experts-implementation", choices=["eager", "batched_mm", "grouped_mm"], default=None)
    parser.add_argument("--qwen-gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] output_dir={output_dir}", flush=True)
    print(f"[run] seed={args.seed}", flush=True)

    categories = load_instruction_categories(
        Path(args.instruction_config) if args.instruction_config else None,
        filter_ids=args.specific_categories,
    )
    print(f"[run] categories={[cat.category_id for cat in categories]}", flush=True)
    print(f"[run] loading_qwen model={args.qwen_model} gpu={args.qwen_gpu}", flush=True)
    qwen = QwenInstructionRewriter(model_path=args.qwen_model, gpu=args.qwen_gpu)
    print(
        f"[run] loading_gemma model={args.gemma_model} gpus={args.gemma_gpus} "
        f"max_memory_gib={args.gemma_max_memory_gib} "
        f"experts_impl={args.gemma_experts_implementation}",
        flush=True,
    )
    gemma = build_gemma4_engine_multi_gpu(
        model_path=args.gemma_model,
        gpus=args.gemma_gpus,
        gpu_max_memory_gib_by_device=args.gemma_max_memory_gib,
        experts_implementation=args.gemma_experts_implementation,
    )

    videos = find_videos(video_dir=Path(args.video_dir), only_movie=args.only_movie)
    if not videos:
        raise RuntimeError("No videos found for script-free experiment.")
    print(f"[run] matched_movies={len(videos)}", flush=True)

    for video_path in videos:
        result = run_one_movie(
            args=args,
            video_path=video_path,
            categories=categories,
            qwen=qwen,
            gemma=gemma,
            rng=rng,
        )
        out_path = output_dir / f"{safe_movie_slug(result.movie_title)}_experiment.json"
        full_path, ad_path = save_experiment_result(result, out_path)
        print(f"[saved] full={full_path}", flush=True)
        print(f"[saved] ad_output={ad_path}", flush=True)


if __name__ == "__main__":
    main()
