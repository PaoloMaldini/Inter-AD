#!/usr/bin/env python3
"""
Gemma4 + Qwen interactive AD experiment (script-based).

Uses aligned script gaps as the playable timeline, but simulates interactive
user requests inserted during movie playback. Outputs evaluator-compatible
`*_ad_output.json`.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKPLACE_DIR = SCRIPT_DIR / "workplace"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WORKPLACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKPLACE_DIR))

from face_gallery import load_gallery, lookup_face_images
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
    sanitize_context_text,
    safe_movie_slug,
    save_experiment_result,
    select_insertion_indices,
)
from workplace.pipeline_script_based import (
    DEFAULT_GAP_SEC,
    GT_CSV,
    VIDEO_DIRS,
    XLSX_DIRS,
    _prepare_fast_video,
    build_gap_context,
    detect_dialogue_gaps,
    extract_clip,
    match_videos_to_xlsxs,
    parse_dialogue_rows,
)


DEFAULT_OUTPUT_DIR = Path("/mnt/disk6new/wzq/experiment/ghl/tmp/gemma4_qwen_interactive_script")
DEFAULT_NUM_INSERTIONS = 12
DEFAULT_MIN_INSERTIONS = 10
DEFAULT_MAX_INSERTIONS = 15
DEFAULT_TMP_ROOT = Path(os.environ.get("NEWAD_TMP_ROOT", "/mnt/disk6new/wzq/experiment/ghl/tmp/newAD_script_tmp"))


def _fmt_sec(value: float) -> str:
    return f"{value:.1f}s"


def _preview(text: str, limit: int = 120) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _resolve_clip_tmp_dir(output_dir: Path, movie_title: str) -> Path:
    preferred_root = DEFAULT_TMP_ROOT
    target_name = f".tmp_script_{safe_movie_slug(movie_title)}"
    try:
        preferred_root.mkdir(parents=True, exist_ok=True)
        return preferred_root / target_name
    except Exception:
        return output_dir / target_name


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


def _resolve_video_script_pairs(args: argparse.Namespace) -> List[tuple[Path, Path]]:
    gt_csv = None if args.no_gt_filter else GT_CSV
    return match_videos_to_xlsxs(
        video_dirs=args.video_dir,
        xlsx_dirs=args.xlsx_dir,
        gt_csv=gt_csv,
        only_movie=args.only_movie,
    )


def _resolve_num_insertions(args: argparse.Namespace, rng: random.Random, num_candidates: int) -> int:
    if args.min_insertions is not None or args.max_insertions is not None:
        low = args.min_insertions if args.min_insertions is not None else args.num_insertions
        high = args.max_insertions if args.max_insertions is not None else args.num_insertions
        if low > high:
            raise ValueError(f"min_insertions > max_insertions: {low} > {high}")
        chosen = rng.randint(low, high)
    else:
        chosen = args.num_insertions
    return max(0, min(chosen, num_candidates))


def run_one_movie(
    *,
    args: argparse.Namespace,
    video_path: Path,
    xlsx_path: Path,
    categories,
    qwen: QwenInstructionRewriter,
    gemma,
    rng: random.Random,
) -> MovieExperimentResult:
    movie_title = video_path.stem
    t0 = time.monotonic()
    print(f"[movie] {movie_title}", flush=True)
    print(f"[movie] video={video_path}", flush=True)
    print(f"[movie] script={xlsx_path}", flush=True)
    rows = parse_dialogue_rows(xlsx_path)
    if not rows:
        raise RuntimeError(f"No dialogue rows found: {xlsx_path}")
    print(f"[movie] dialogue_rows={len(rows)}", flush=True)

    candidates = detect_dialogue_gaps(rows, args.gap_threshold)
    if not candidates:
        raise RuntimeError(f"No dialogue gaps found for {movie_title}")

    video_duration_sec = get_video_duration(video_path)
    print(
        f"[movie] candidate_gaps={len(candidates)} duration={_fmt_sec(video_duration_sec)} gap_threshold={args.gap_threshold}",
        flush=True,
    )
    state = InstructionState()
    state.refresh_task_prompt(qwen)

    gallery_embs, gallery_people = load_gallery(movie_title)
    if gallery_embs is None or gallery_people is None:
        print("[movie] face_gallery=unavailable", flush=True)
    else:
        print(f"[movie] face_gallery={len(gallery_people)} characters", flush=True)
    num_insertions = _resolve_num_insertions(args, rng, len(candidates))
    insertion_indices = select_insertion_indices(len(candidates), num_insertions, rng)
    insertion_set = set(insertion_indices)
    print(f"[movie] num_insertions={num_insertions}", flush=True)
    print(f"[movie] insertion_indices={insertion_indices}", flush=True)
    print(f"[model] qwen_gpu={args.qwen_gpu} gemma_gpus={args.gemma_gpus}", flush=True)

    clip_tmp = _resolve_clip_tmp_dir(Path(args.output_dir), movie_title)
    clip_tmp.mkdir(parents=True, exist_ok=True)
    print(f"[movie] preparing_fast_video tmp={clip_tmp}", flush=True)
    fast_video = _prepare_fast_video(video_path, clip_tmp)
    print(f"[movie] fast_video_ready path={fast_video}", flush=True)

    insertion_records: List[InsertionRecord] = []
    segment_records: List[SegmentRecord] = []
    ad_entries: List[Dict[str, Any]] = []
    insertion_events: List[Dict[str, Any]] = []
    insertion_counter = 0

    all_characters: List[str] = []
    for cand in candidates:
        for name in cand.characters:
            if name and name not in all_characters:
                all_characters.append(name)

    try:
        for seg_idx, cand in enumerate(candidates):
            seg_no = seg_idx + 1
            print(
                f"[segment {seg_no}/{len(candidates)}] gap_id={cand.gap_id} start={_fmt_sec(cand.gap_start_sec)} "
                f"end={_fmt_sec(cand.gap_end_sec)} dur={cand.gap_duration_sec:.1f}s active={len(state.snapshot())}",
                flush=True,
            )
            clip_path = clip_tmp / f"gap{cand.gap_id:04d}.mp4"
            print(f"[segment {seg_no}/{len(candidates)}] extracting_clip -> {clip_path.name}", flush=True)
            if not extract_clip(fast_video, cand.gap_start_sec, cand.gap_end_sec, clip_path):
                print(f"[segment {seg_no}/{len(candidates)}] extract_clip_failed", flush=True)
                continue
            print(f"[segment {seg_no}/{len(candidates)}] clip_ready", flush=True)

            raw_context = build_gap_context(cand)
            context_text = sanitize_context_text(raw_context) if args.sanitize_context else raw_context
            print(
                f"[segment {seg_no}/{len(candidates)}] context={_preview(context_text, 100)}",
                flush=True,
            )

            face_avatars: List[Path] = []
            if gallery_embs is not None and gallery_people is not None and cand.characters:
                face_avatars = lookup_face_images(cand.characters, gallery_people, movie_title)
            if cand.characters:
                print(
                    f"[segment {seg_no}/{len(candidates)}] characters={cand.characters}",
                    flush=True,
                )

            final_text = ""
            final_latency = 0.0
            final_task_prompt = state.current_task_prompt
            dynamic_max_words = compute_dynamic_word_budget(
                cand.gap_duration_sec,
                words_per_minute=args.words_per_minute,
                budget_ratio=args.word_budget_ratio,
                min_words=args.min_words_per_gap,
                hard_cap=args.max_words if args.max_words and args.max_words > 0 else None,
            )
            runtime_task_prompt = build_runtime_task_prompt(
                final_task_prompt,
                gap_duration_sec=cand.gap_duration_sec,
                max_words=dynamic_max_words,
            )
            print(
                f"[segment {seg_no}/{len(candidates)}] word_budget={dynamic_max_words} "
                f"(gap={cand.gap_duration_sec:.1f}s, wpm={args.words_per_minute}, ratio={args.word_budget_ratio})",
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
                    f"*** INSERT #{insertion_counter} seg={seg_idx} t={_fmt_sec(cand.gap_start_sec)} "
                    f"cat={request_meta['category_id']} slot={primary_slot}",
                    flush=True,
                )
                print(f"  raw={request_text}", flush=True)
                print(f"  active_before={len(active_before)}", flush=True)

                print(f"[segment {seg_no}/{len(candidates)}] generating_baseline", flush=True)
                before_text, before_latency, _ = generate_ad(
                    gemma,
                    clip_path=clip_path,
                    context_text=context_text,
                    task_prompt=runtime_task_prompt,
                    face_avatars=face_avatars,
                    character_names=cand.characters,
                    max_words=dynamic_max_words,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                )
                print(
                    f"[segment {seg_no}/{len(candidates)}] baseline_done latency={before_latency:.2f}s "
                    f"text={_preview(before_text)}",
                    flush=True,
                )

                print(f"[segment {seg_no}/{len(candidates)}] rewriting_request_with_qwen", flush=True)
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
                    insert_timestamp_sec=cand.gap_start_sec,
                    insert_segment_idx=seg_idx,
                    character_name=None,
                )
                final_task_prompt = state.refresh_task_prompt(qwen)
                runtime_task_prompt = build_runtime_task_prompt(
                    final_task_prompt,
                    gap_duration_sec=cand.gap_duration_sec,
                    max_words=dynamic_max_words,
                )
                print(
                    f"[segment {seg_no}/{len(candidates)}] qwen_slots={effective_slots} focus={state.current_focus_line}",
                    flush=True,
                )
                print(f"[segment {seg_no}/{len(candidates)}] prompt_after:", flush=True)
                print(runtime_task_prompt, flush=True)

                print(f"[segment {seg_no}/{len(candidates)}] generating_after_insert", flush=True)
                after_text, after_latency, _ = generate_ad(
                    gemma,
                    clip_path=clip_path,
                    context_text=context_text,
                    task_prompt=runtime_task_prompt,
                    face_avatars=face_avatars,
                    character_names=cand.characters,
                    max_words=dynamic_max_words,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                )

                final_text = after_text
                final_latency = after_latency
                print(
                    f"[segment {seg_no}/{len(candidates)}] after_done latency={after_latency:.2f}s "
                    f"text={_preview(after_text)}",
                    flush=True,
                )

                record = InsertionRecord(
                    insertion_id=insertion_counter,
                    movie_title=movie_title,
                    insert_timestamp_sec=cand.gap_start_sec,
                    segment_idx=seg_idx,
                    segment_start_sec=cand.gap_start_sec,
                    segment_end_sec=cand.gap_end_sec,
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
                    characters=list(cand.characters),
                    ref_ad="",
                )
                insertion_records.append(record)
                insertion_events.append(
                    {
                        "insertion_id": insertion_counter,
                        "gap_idx": seg_idx,
                        "segment_idx": seg_idx,
                        "timestamp_sec": round(cand.gap_start_sec, 3),
                        "category": request_meta["category_id"],
                        "category_id": request_meta["category_id"],
                        "category_name": request_meta["category_name"],
                        "instruction_text": request_text,
                        "language": request_meta["request_language"],
                        "components": request_meta["components"],
                        "task_prompt_after": runtime_task_prompt,
                        "prompt_state_after": state.prompt_state_dict(),
                        "focus_line_after": state.current_focus_line,
                        "active_before": active_before,
                        "all_active_instructions": state.snapshot(),
                        "dynamic_max_words": dynamic_max_words,
                    }
                )
            else:
                print(f"[segment {seg_no}/{len(candidates)}] generating", flush=True)
                final_text, final_latency, _ = generate_ad(
                    gemma,
                    clip_path=clip_path,
                    context_text=context_text,
                    task_prompt=runtime_task_prompt,
                    face_avatars=face_avatars,
                    character_names=cand.characters,
                    max_words=dynamic_max_words,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                )
                print(
                    f"[segment {seg_no}/{len(candidates)}] done latency={final_latency:.2f}s "
                    f"text={_preview(final_text)}",
                    flush=True,
                )

            segment_records.append(
                SegmentRecord(
                    segment_idx=seg_idx,
                    segment_start_sec=cand.gap_start_sec,
                    segment_end_sec=cand.gap_end_sec,
                    clip_path=str(clip_path),
                    generated_text=final_text,
                    instruction_text=state.raw_instruction_text(),
                    active_instructions_count=len(state.snapshot()),
                    latency_sec=round(final_latency, 4),
                    task_prompt=runtime_task_prompt,
                    context_text=context_text[:600],
                    characters=list(cand.characters),
                    ref_ad="",
                )
            )

            ad_entries.append(
                {
                    "gap_id": cand.gap_id,
                    "scene_index": cand.scene_index,
                    "location": cand.location,
                    "gap_start_sec": round(cand.gap_start_sec, 3),
                    "gap_end_sec": round(cand.gap_end_sec, 3),
                    "gap_duration_sec": round(cand.gap_duration_sec, 3),
                    "characters": list(cand.characters),
                    "context_before": cand.context_before[-5:],
                    "context_after": cand.context_after[:8],
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
            "method": "gemma4_qwen_interactive_script",
            "video_path": str(video_path),
            "xlsx_path": str(xlsx_path),
            "gap_threshold_sec": args.gap_threshold,
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
            "sanitize_context": args.sanitize_context,
            "insertion_events": insertion_events,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma4 + Qwen interactive AD experiment (script-based).")
    parser.add_argument("--video-dir", nargs="*", default=VIDEO_DIRS)
    parser.add_argument("--xlsx-dir", nargs="*", default=XLSX_DIRS)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--only-movie", type=str, default=None)
    parser.add_argument("--no-gt-filter", action="store_true")
    parser.add_argument("--gap-threshold", type=float, default=DEFAULT_GAP_SEC)
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
    parser.add_argument("--sanitize-context", action="store_true", default=True)
    parser.add_argument("--no-sanitize-context", dest="sanitize_context", action="store_false")
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

    pairs = _resolve_video_script_pairs(args)
    if not pairs:
        raise RuntimeError("No matched video/xlsx pairs found.")
    print(f"[run] matched_movies={len(pairs)}", flush=True)

    for video_path, xlsx_path in pairs:
        result = run_one_movie(
            args=args,
            video_path=video_path,
            xlsx_path=xlsx_path,
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
