#!/usr/bin/env python3
"""
interactive_experiment.py — Interactive AD Experiment Framework.

Simulates a streaming video playback where user instructions are inserted at
random timestamps. For each insertion:
  1. Generate AD *before* the instruction (baseline)
  2. Insert the instruction into the prompt
  3. Generate AD *after* the instruction (instructed)
  4. Record timing (latency, TTFF) and metadata

Instructions are persistent within a single movie session (reset between movies).
Multiple instructions can be inserted at different timestamps in the same movie.

Instruction categories are configurable via JSON file or built-in defaults.
"""

from __future__ import annotations

import copy
import json
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Project paths & env vars (cache locally, not on system disk) ─────────────
STREAMING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STREAMING_ROOT.parent

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".hf_cache" / "sentence_transformers"))
os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT_ROOT / ".pip_cache"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch_cache"))

if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

from segment_db import SegmentDB, load_segment_db, extract_face_data, to_float
from context_builder import (
    build_prompt_context,
    build_task_prompt,
    AVAILABLE_MODULES,
    AVAILABLE_PLOT_SUBMODULES,
)
from ad_engine import ADEngine, build_ad_engine

# ── Paths ─────────────────────────────────────────────────────────────────────
INSTRUCTION_CONFIG_PATH = STREAMING_ROOT / "instruction_categories.json"
FINAL_BY_MOVIE_DIR = Path("/mnt/disk1new/ylz/newAD/Step04_RunTest/step04_final_by_movie_new")
AD_CLIPS_DIR = Path("/mnt/disk1new/ylz/newAD/Step04_RunTest/ad_clips_final")
FACE_JSON_ROOT = Path("/mnt/disk1new/ylz/newAD/Step04_RunTest/step04_03_face_align/json")

# ── Instruction Category Definition ──────────────────────────────────────────

@dataclass
class InstructionCategory:
    """A single instruction category with templates."""
    category_id: str
    name: str
    name_cn: str
    description: str
    templates: List[str]         # English templates with {placeholder} support
    templates_cn: List[str]      # Chinese counterparts
    weight: float = 1.0          # Sampling weight for random selection

    def sample_template(self, lang: str = "auto") -> Tuple[str, str]:
        """Return (template, language_used)."""
        idx = random.randint(0, len(self.templates) - 1)
        if lang == "cn" or (lang == "auto" and random.random() < 0.5):
            return self.templates_cn[idx % len(self.templates_cn)], "cn"
        return self.templates[idx % len(self.templates)], "en"


# ── Default Instruction Categories ────────────────────────────────────────────

DEFAULT_CATEGORIES: List[InstructionCategory] = [
    InstructionCategory(
        category_id="style",
        name="Language Style",
        name_cn="语言风格",
        description="Adjust the writing style / tone of the AD",
        templates=[
            "Use a more poetic and literary style",
            "Use a dramatic, cinematic narration style",
            "Use a casual, conversational tone",
            "Use formal, documentary-style language",
            "Make it sound like a thriller novel narration",
        ],
        templates_cn=[
            "使用更诗意、文学化的风格描述",
            "使用戏剧化的电影解说风格",
            "使用轻松随意的语气描述",
            "使用正式的纪录片风格",
            "像悬疑小说旁白一样描述",
        ],
    ),
    InstructionCategory(
        category_id="character",
        name="Character Focus",
        name_cn="角色聚焦",
        description="Increase description frequency of a specific character",
        templates=[
            "Focus more on {character}'s expressions and reactions",
            "Describe {character}'s body language and gestures in detail",
            "Pay special attention to {character}'s emotional state",
            "Emphasize {character}'s presence and actions",
        ],
        templates_cn=[
            "更多关注{character}的表情和反应",
            "详细描述{character}的肢体语言和手势",
            "特别注意{character}的情绪状态",
            "强调{character}的存在和动作",
        ],
    ),
    InstructionCategory(
        category_id="event_romance",
        name="Romantic Events",
        name_cn="浪漫情节",
        description="Focus on romantic or emotional scenes",
        templates=[
            "Focus on romantic moments and emotional connections between characters",
            "Describe tender interactions and intimate gestures",
            "Highlight the emotional atmosphere of romantic scenes",
        ],
        templates_cn=[
            "聚焦浪漫时刻和角色之间的情感联系",
            "描述温柔的互动和亲密的举动",
            "突出浪漫场景的情感氛围",
        ],
    ),
    InstructionCategory(
        category_id="event_action",
        name="Action Events",
        name_cn="动作情节",
        description="Focus on action, fighting, or intense scenes",
        templates=[
            "Focus on action sequences and physical movements",
            "Describe fighting choreography and dynamic motions in detail",
            "Highlight the intensity and speed of action scenes",
        ],
        templates_cn=[
            "聚焦动作场景和肢体运动",
            "详细描述打斗编排和动态动作",
            "突出动作场景的紧张感和速度感",
        ],
    ),
    InstructionCategory(
        category_id="event_drama",
        name="Dramatic Events",
        name_cn="戏剧情节",
        description="Focus on dramatic or tense moments",
        templates=[
            "Focus on dramatic tension and suspenseful moments",
            "Describe the characters' psychological states during tense scenes",
            "Highlight the dramatic lighting and mood shifts",
        ],
        templates_cn=[
            "聚焦戏剧张力和悬疑时刻",
            "描述紧张场景中角色的心理状态",
            "突出戏剧性的光影和情绪变化",
        ],
    ),
    InstructionCategory(
        category_id="lang_switch",
        name="Language Switch",
        name_cn="语言切换",
        description="Switch output language",
        templates=[
            "Describe in Chinese (中文描述)",
            "Describe in English",
        ],
        templates_cn=[
            "用中文描述",
            "用英文描述",
        ],
    ),
    InstructionCategory(
        category_id="detail_detailed",
        name="More Detailed",
        name_cn="更详细",
        description="Make the description more detailed and comprehensive",
        templates=[
            "Describe every visible detail in the frame comprehensively",
            "Provide a thorough, detailed description of the entire scene",
            "Include all visual elements: objects, colors, textures, spatial layout",
        ],
        templates_cn=[
            "全面描述画面中每一个可见细节",
            "提供完整、详细的场景描述",
            "包含所有视觉元素：物体、颜色、纹理、空间布局",
        ],
    ),
    InstructionCategory(
        category_id="detail_concise",
        name="More Concise",
        name_cn="更简洁",
        description="Make the description shorter and more concise",
        templates=[
            "Be very concise, describe in a single sentence only",
            "Use minimal words, focus only on the most important action",
            "Keep it short and punchy, no more than 10 words",
        ],
        templates_cn=[
            "非常简洁，只用一句话描述",
            "用最少的词，只关注最重要的动作",
            "简短有力，不超过十个词",
        ],
    ),
]


def load_instruction_categories(config_path: Optional[Path] = None) -> List[InstructionCategory]:
    """Load instruction categories from JSON file, or return defaults."""
    if config_path and config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        categories = []
        for item in data.get("categories", []):
            categories.append(InstructionCategory(**item))
        return categories if categories else DEFAULT_CATEGORIES
    return DEFAULT_CATEGORIES


def save_instruction_categories(categories: List[InstructionCategory], config_path: Path):
    """Save instruction categories to JSON file for editing."""
    data = {"categories": [asdict(c) for c in categories]}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[config] Saved {len(categories)} instruction categories to {config_path}")


# ── Instruction State ─────────────────────────────────────────────────────────

@dataclass
class ActiveInstruction:
    """An instruction that has been inserted and remains active."""
    instruction_id: int
    category_id: str
    category_name: str
    template: str
    language: str
    insert_timestamp_sec: float
    insert_segment_idx: int
    character_name: Optional[str] = None   # filled for 'character' category


@dataclass
class InsertionRecord:
    """Record of a single instruction insertion experiment."""
    insertion_id: int
    movie_title: str
    insert_timestamp_sec: float
    segment_idx: int
    segment_start_sec: float
    segment_end_sec: float

    # Instruction info
    instruction_text: str            # the NEW instruction inserted at this point
    category_id: str
    category_name: str
    instruction_language: str
    active_instructions_count: int   # how many instructions were active at this point

    # Full combined instruction strings (what actually went into the prompt)
    instruction_before: str          # full instruction for text_before (previous instructions only)
    instruction_after: str           # full instruction for text_after  (previous + new instruction)

    # All active instructions as structured list (for detailed evaluation)
    all_active_instructions: List[Dict[str, str]]  # [{id, category, template, lang}, ...]

    # Generated texts
    text_before: str                 # AD generated WITHOUT the new instruction
    text_after: str                  # AD generated WITH the new instruction

    # Timing
    latency_before_sec: float        # inference time for "before"
    latency_after_sec: float         # inference time for "after"
    ttff_before_sec: float           # time-to-first-frame for "before" (same as latency here)
    ttff_after_sec: float            # time-to-first-frame for "after"

    # Context (for evaluation)
    context_text: str = ""
    characters: List[str] = field(default_factory=list)
    ref_ad: str = ""                 # ground-truth AD if available


@dataclass
class SegmentRecord:
    """Record of AD generation for a single segment."""
    segment_idx: int
    segment_start_sec: float
    segment_end_sec: float
    clip_path: str
    generated_text: str
    instruction_text: str          # combined active instructions at this point
    active_instructions_count: int
    latency_sec: float
    context_text: str = ""
    characters: List[str] = field(default_factory=list)
    ref_ad: str = ""


@dataclass
class MovieExperimentResult:
    """Complete result of one movie experiment run."""
    movie_title: str
    movie_duration_sec: float
    total_segments: int
    num_insertions: int
    insertion_records: List[InsertionRecord]
    segment_records: List[SegmentRecord]
    ad_entries: List[Dict[str, Any]]   # baseline-format ad_entries for eval compatibility
    run_config: Dict[str, Any]


# ── Utility ───────────────────────────────────────────────────────────────────

def _resolve_dir(base: Path, movie_title: str) -> Optional[Path]:
    if base.is_dir():
        return base
    for p in base.parent.iterdir():
        if p.is_dir() and movie_title.lower().replace(" ", "_") in p.name.lower():
            return p
    return None


def _parse_clip_stem(ad_id: str, clip_index: Any, ad_order: int) -> str:
    match = re.search(r"__clip(\d+)__ad(\d+)$", str(ad_id or ""))
    if match:
        return f"clip{match.group(1).zfill(4)}_ad{match.group(2).zfill(4)}"
    return f"clip{int(to_float(clip_index, 0)):04d}_ad{ad_order:04d}"


def _fill_character_template(template: str, characters: List[str]) -> Tuple[str, Optional[str]]:
    """Return (filled_template, chosen_character_name)."""
    if "{character}" in template and characters:
        chosen = random.choice(characters)
        return template.replace("{character}", chosen), chosen
    return template.replace("{character}", "the main character"), None


# ── Core Experiment Class ─────────────────────────────────────────────────────

class InteractiveExperiment:
    """
    Simulates streaming AD generation with random instruction insertions.

    Workflow for a single movie:
      1. Load segment DB and precomputed data
      2. Iterate through segments in order (streaming simulation)
      3. At random timestamps, insert an instruction:
         a. Generate baseline AD (no extra instruction)
         b. Insert instruction into prompt
         c. Generate instructed AD
         d. Record before/after texts + timing
      4. Instruction persists for all subsequent segments
      5. Reset when switching to a new movie
    """

    def __init__(
        self,
        engine: ADEngine,
        categories: Optional[List[InstructionCategory]] = None,
        config_path: Optional[Path] = None,
        gpu_id: int = 0,
    ):
        self.engine = engine
        self.categories = categories or load_instruction_categories(config_path)
        self.gpu_id = gpu_id

        # Per-movie state (reset between movies)
        self._active_instructions: List[ActiveInstruction] = []
        self._next_instr_id = 0
        self._movie_title: str = ""

    def reset(self):
        """Reset instruction state for a new movie."""
        self._active_instructions.clear()
        self._next_instr_id = 0
        self._movie_title = ""

    def add_instruction(self, category_id: str, character_name: Optional[str] = None) -> str:
        """Manually add an instruction (for testing or interactive use)."""
        cat = next((c for c in self.categories if c.category_id == category_id), None)
        if cat is None:
            raise ValueError(f"Unknown category: {category_id}")
        template, lang = cat.sample_template()
        if character_name:
            template = template.replace("{character}", character_name)
        instr = ActiveInstruction(
            instruction_id=self._next_instr_id,
            category_id=category_id,
            category_name=cat.name,
            template=template,
            language=lang,
            insert_timestamp_sec=0.0,
            insert_segment_idx=0,
            character_name=character_name,
        )
        self._active_instructions.append(instr)
        self._next_instr_id += 1
        return template

    def remove_instruction(self, instruction_id: int):
        """Remove an active instruction by ID."""
        self._active_instructions = [
            i for i in self._active_instructions if i.instruction_id != instruction_id
        ]

    def list_active_instructions(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": i.instruction_id,
                "category": i.category_name,
                "template": i.template,
                "language": i.language,
                "insert_at": i.insert_timestamp_sec,
            }
            for i in self._active_instructions
        ]

    def _build_combined_instruction(self, new_instruction: str = "") -> str:
        """Combine all active instructions + new one into a single prompt addition."""
        parts = []
        for instr in self._active_instructions:
            parts.append(instr.template)
        if new_instruction:
            parts.append(new_instruction)
        return " | ".join(parts) if parts else ""

    def _generate_for_segment(
        self,
        clip_path: Path,
        context_text: str,
        instruction: str,
        temperature: float = 0.2,
        face_avatars: Optional[List[Path]] = None,
        character_names: Optional[List[str]] = None,
    ) -> Tuple[str, float, str]:
        """Generate AD for a segment with given instruction. Returns (text, latency, prompt)."""
        task_prompt = build_task_prompt(custom_instruction=instruction)
        text, latency, prompt = self.engine.infer_one_segment(
            clip_path=clip_path,
            context_text=context_text,
            task_prompt=task_prompt,
            temperature=temperature,
            face_avatars=face_avatars,
            character_names=character_names,
        )
        return text, latency, prompt

    def run_movie_experiment(
        self,
        movie_title: str,
        video_path: str = "",
        num_insertions: int = 3,
        insertion_strategy: str = "random",
        specific_timestamps: Optional[List[float]] = None,
        specific_categories: Optional[List[str]] = None,
        temperature: float = 0.2,
        seed: int = 42,
    ) -> MovieExperimentResult:
        """
        Run interactive experiment on a single movie.

        Args:
            movie_title: Movie to process.
            video_path: Path to video file (for clip extraction).
            num_insertions: How many instructions to insert.
            insertion_strategy: 'random', 'uniform', or 'manual'.
            specific_timestamps: Manual timestamps (for 'manual' strategy).
            specific_categories: Categories to use (None = random from all).
            temperature: Generation temperature.
            seed: Random seed for reproducibility.

        Returns:
            MovieExperimentResult with all insertion records.
        """
        rng = random.Random(seed)
        np.random.seed(seed)

        self.reset()
        self._movie_title = movie_title

        # ── Load movie data ────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"[experiment] Loading movie: {movie_title}")
        seg_db = load_segment_db(movie_title, final_by_movie_dir=FINAL_BY_MOVIE_DIR)
        segments = seg_db.segments
        total_dur = seg_db.total_duration
        n_seg = len(segments)
        print(f"  Segments: {n_seg}, Duration: {total_dur:.1f}s")

        clips_dir = _resolve_dir(AD_CLIPS_DIR / movie_title, movie_title)
        face_dir = _resolve_dir(FACE_JSON_ROOT / movie_title, movie_title)

        if clips_dir is None:
            raise FileNotFoundError(f"Clips directory not found for {movie_title}")

        # ── Determine insertion points ─────────────────────────────────────
        if insertion_strategy == "manual" and specific_timestamps:
            insertion_indices = []
            for ts in specific_timestamps:
                for si, seg in enumerate(segments):
                    s = to_float(seg.get("ad_movie_start_sec"), 0.0)
                    e = to_float(seg.get("ad_movie_end_sec"), 0.0)
                    if s <= ts <= e:
                        insertion_indices.append(si)
                        break
                else:
                    # Find nearest segment
                    dists = [abs(ts - to_float(seg.get("ad_movie_start_sec"), 0.0)) for seg in segments]
                    insertion_indices.append(int(np.argmin(dists)))
        elif insertion_strategy == "uniform":
            step = max(1, n_seg // (num_insertions + 1))
            insertion_indices = [step * (i + 1) for i in range(num_insertions)]
            insertion_indices = [min(idx, n_seg - 1) for idx in insertion_indices]
        else:  # random
            available = list(range(1, n_seg))  # don't insert at segment 0
            k = min(num_insertions, len(available))
            insertion_indices = sorted(rng.sample(available, k))

        print(f"  Insertion points: {len(insertion_indices)} at segments {insertion_indices}")

        # ── Select categories for each insertion ───────────────────────────
        available_cats = self.categories
        if specific_categories:
            available_cats = [c for c in self.categories if c.category_id in specific_categories]
            if not available_cats:
                available_cats = self.categories

        cat_weights = [c.weight for c in available_cats]
        selected_cats = rng.choices(available_cats, weights=cat_weights, k=len(insertion_indices))

        # ── Extract character names from segments ──────────────────────────
        all_characters: List[str] = []
        for seg in segments:
            for c in seg.get("characters", []) if isinstance(seg.get("characters"), list) else []:
                name = str(c).strip()
                if name and name not in all_characters:
                    all_characters.append(name)

        # ── Stream through ALL segments ────────────────────────────────────
        insertion_set = set(insertion_indices)
        insertion_records: List[InsertionRecord] = []
        segment_records: List[SegmentRecord] = []
        ad_entries: List[Dict[str, Any]] = []
        insertion_counter = 0

        for seg_idx, seg in enumerate(segments):
            seg_start = to_float(seg.get("ad_movie_start_sec"), 0.0)
            seg_end = to_float(seg.get("ad_movie_end_sec"), 0.0)
            ad_id = str(seg.get("ad_id", "")).strip()

            # Resolve clip path
            clip_stem = _parse_clip_stem(ad_id=ad_id, clip_index=seg.get("clip_index"), ad_order=seg_idx + 1)
            clip_path = clips_dir / f"{clip_stem}.mp4"

            if not clip_path.is_file():
                print(f"  [skip] seg {seg_idx}: clip not found at {clip_path}")
                continue

            # Resolve face data
            face_matches: List[Dict[str, Any]] = []
            face_avatars: Optional[List[Path]] = None
            character_names: Optional[List[str]] = None
            if face_dir:
                face_json = face_dir / f"{clip_stem}.json"
                face_matches, _ = extract_face_data(face_json, max_face_records=4)
                if face_matches:
                    character_names = [m.get("role_name", "") for m in face_matches if m.get("role_name")]
                    face_avatars = [Path(m.get("avatar_path", "")) for m in face_matches if m.get("avatar_path")]

            # Build context
            context_text, _ = build_prompt_context(
                segment=seg,
                modules=AVAILABLE_MODULES,
                plot_submodules=AVAILABLE_PLOT_SUBMODULES,
                face_matches=face_matches,
                max_description_lines=5,
                max_dialog_lines=8,
            )
            ref_ad = str(seg.get("cmdqa", {}).get("text", "")).strip()

            # ── Generate AD for this segment with current active instructions ──
            current_instr = self._build_combined_instruction("")
            t_seg = time.monotonic()
            seg_text, seg_latency, _ = self._generate_for_segment(
                clip_path=clip_path,
                context_text=context_text,
                instruction=current_instr,
                temperature=temperature,
                face_avatars=face_avatars,
                character_names=character_names,
            )
            seg_latency = time.monotonic() - t_seg

            segment_records.append(SegmentRecord(
                segment_idx=seg_idx,
                segment_start_sec=seg_start,
                segment_end_sec=seg_end,
                clip_path=str(clip_path),
                generated_text=seg_text,
                instruction_text=current_instr,
                active_instructions_count=len(self._active_instructions),
                latency_sec=round(seg_latency, 4),
                context_text=context_text[:500],
                characters=all_characters[:10],
                ref_ad=ref_ad,
            ))

            progress_str = f"  [{seg_idx+1}/{n_seg}] t={seg_start:.0f}s"
            if self._active_instructions:
                progress_str += f" (instr×{len(self._active_instructions)})"
            print(f"{progress_str} → {seg_text[:60]}...")

            # ── If this segment has an instruction insertion, do before/after ──
            if seg_idx in insertion_set:
                cat = selected_cats[insertion_counter]
                insertion_counter += 1

                template, lang = cat.sample_template()
                chosen_character = None
                if cat.category_id == "character":
                    template, chosen_character = _fill_character_template(template, all_characters)

                print(f"\n  [insert #{insertion_counter}] seg {seg_idx}, t={seg_start:.1f}s, "
                      f"cat={cat.name}, instr='{template}'")

                # 1. BEFORE: generate with current instructions (before adding new one)
                instr_before = current_instr
                active_snapshot_before = [
                    {"id": str(ai.instruction_id), "category": ai.category_id,
                     "template": ai.template, "lang": ai.language}
                    for ai in self._active_instructions
                ]
                t0 = time.monotonic()
                text_before, latency_before, _ = self._generate_for_segment(
                    clip_path=clip_path,
                    context_text=context_text,
                    instruction=instr_before,
                    temperature=temperature,
                    face_avatars=face_avatars,
                    character_names=character_names,
                )
                latency_before = time.monotonic() - t0

                # 2. Add new instruction
                new_active = ActiveInstruction(
                    instruction_id=self._next_instr_id,
                    category_id=cat.category_id,
                    category_name=cat.name,
                    template=template,
                    language=lang,
                    insert_timestamp_sec=seg_start,
                    insert_segment_idx=seg_idx,
                    character_name=chosen_character,
                )
                self._active_instructions.append(new_active)
                self._next_instr_id += 1

                # 3. AFTER: generate with all instructions including new one
                instr_after = self._build_combined_instruction("")
                active_snapshot_after = [
                    {"id": str(ai.instruction_id), "category": ai.category_id,
                     "template": ai.template, "lang": ai.language}
                    for ai in self._active_instructions
                ]
                t1 = time.monotonic()
                text_after, latency_after, _ = self._generate_for_segment(
                    clip_path=clip_path,
                    context_text=context_text,
                    instruction=instr_after,
                    temperature=temperature,
                    face_avatars=face_avatars,
                    character_names=character_names,
                )
                latency_after = time.monotonic() - t1

                record = InsertionRecord(
                    insertion_id=insertion_counter,
                    movie_title=movie_title,
                    insert_timestamp_sec=seg_start,
                    segment_idx=seg_idx,
                    segment_start_sec=seg_start,
                    segment_end_sec=seg_end,
                    instruction_text=template,
                    category_id=cat.category_id,
                    category_name=cat.name,
                    instruction_language=lang,
                    active_instructions_count=len(self._active_instructions),
                    instruction_before=instr_before,
                    instruction_after=instr_after,
                    all_active_instructions=active_snapshot_after,
                    text_before=text_before,
                    text_after=text_after,
                    latency_before_sec=round(latency_before, 4),
                    latency_after_sec=round(latency_after, 4),
                    ttff_before_sec=round(latency_before, 4),
                    ttff_after_sec=round(latency_after, 4),
                    context_text=context_text[:500],
                    characters=all_characters[:10],
                    ref_ad=ref_ad,
                )
                insertion_records.append(record)

                print(f"    BEFORE: {text_before[:80]}...")
                print(f"    AFTER:  {text_after[:80]}...")
                print(f"    Latency: {latency_before:.2f}s → {latency_after:.2f}s")

            # ── Build ad_entry for baseline-format compatibility ───────────
            ctx_before: List[str] = []
            ctx_after: List[str] = []
            for row in (seg.get("matched_rows_selected") or []):
                dialog = str(row.get("align_dialog") or row.get("dialog") or "").strip()
                if dialog:
                    row_start = to_float(row.get("start_time_sec"), seg_start)
                    if row_start <= (seg_start + seg_end) / 2:
                        ctx_before.append(dialog)
                    else:
                        ctx_after.append(dialog)

            ad_entries.append({
                "gap_id": seg_idx + 1,
                "scene_index": str(seg.get("scene_index", seg.get("index_result", ""))),
                "location": str(seg.get("location", "")),
                "gap_start_sec": round(seg_start, 3),
                "gap_end_sec": round(seg_end, 3),
                "gap_duration_sec": round(seg_end - seg_start, 3),
                "characters": [str(c) for c in (seg.get("characters") or []) if str(c).strip()],
                "context_before": ctx_before[-5:],
                "context_after": ctx_after[:8],
                "ad_text": seg_text,
                "inference_time_sec": round(seg_latency, 3),
                "active_instructions": [
                    {"id": ai.instruction_id, "category": ai.category_name,
                     "template": ai.template, "language": ai.language}
                    for ai in self._active_instructions
                ],
                "active_instruction_count": len(self._active_instructions),
            })

        # ── Build insertion_events for metadata ────────────────────────────
        insertion_events: List[Dict[str, Any]] = []
        for ir in insertion_records:
            insertion_events.append({
                "insertion_id": ir.insertion_id,
                "segment_idx": ir.segment_idx,
                "timestamp_sec": ir.insert_timestamp_sec,
                "category_id": ir.category_id,
                "category_name": ir.category_name,
                "instruction_text": ir.instruction_text,
                "language": ir.instruction_language,
                "all_active_instructions": ir.all_active_instructions,
            })

        result = MovieExperimentResult(
            movie_title=movie_title,
            movie_duration_sec=total_dur,
            total_segments=n_seg,
            num_insertions=len(insertion_records),
            insertion_records=insertion_records,
            segment_records=segment_records,
            ad_entries=ad_entries,
            run_config={
                "num_insertions": num_insertions,
                "insertion_strategy": insertion_strategy,
                "temperature": temperature,
                "seed": seed,
                "categories_used": list(set(r.category_id for r in insertion_records)),
                "insertion_events": insertion_events,
            },
        )
        print(f"\n[experiment] Done. {len(ad_entries)} ADs generated, "
              f"{len(insertion_records)} insertions.")
        return result


def save_experiment_result(result: MovieExperimentResult, output_path: Path):
    """Save experiment result to JSON.

    Outputs two files:
    - output_path: full experiment data with segment/insertion records
    - output_path with _ad_output suffix: baseline-compatible format for eval
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Full experiment JSON
    data = asdict(result)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[experiment] Full result saved to {output_path}")

    # 2. Baseline-compatible _ad_output.json (for run_all_eval.py)
    ad_output_path = output_path.with_name(
        output_path.stem.replace("_experiment", "_ad_output").replace("_instructed", "_ad_output")
        + ".json"
    )
    if "_ad_output" not in ad_output_path.stem:
        ad_output_path = output_path.with_name(output_path.stem + "_ad_output.json")

    ad_payload = {
        "movie": result.movie_title,
        "movie_title": result.movie_title,
        "total_gaps": result.total_segments,
        "generated_count": len(result.ad_entries),
        "video_duration_sec": result.movie_duration_sec,
        "ad_entries": result.ad_entries,
        "insertion_events": result.run_config.get("insertion_events", []),
        "num_insertions": result.num_insertions,
        "run_config": result.run_config,
    }
    with ad_output_path.open("w", encoding="utf-8") as f:
        json.dump(ad_payload, f, ensure_ascii=False, indent=2)
    print(f"[experiment] Baseline-format output saved to {ad_output_path}")


def load_experiment_result(json_path: Path) -> MovieExperimentResult:
    """Load experiment result from JSON."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    records = [InsertionRecord(**r) for r in data["insertion_records"]]
    seg_records = [SegmentRecord(**r) for r in data.get("segment_records", [])]
    ad_entries = data.get("ad_entries", [])
    return MovieExperimentResult(
        movie_title=data["movie_title"],
        movie_duration_sec=data["movie_duration_sec"],
        total_segments=data["total_segments"],
        num_insertions=data["num_insertions"],
        insertion_records=records,
        segment_records=seg_records,
        ad_entries=ad_entries,
        run_config=data.get("run_config", {}),
    )
