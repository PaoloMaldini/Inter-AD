#!/usr/bin/env python3
"""
Gemma4 + Qwen interactive AD experiment.

Design goals:
1. Keep generation isolated from evaluation.
2. Reuse the existing random insertion behavior from interactive_experiment.py.
3. Use Qwen to classify user requests into structured prompt slots.
4. Preserve multi-turn requests while overriding conflicting ones by slot.
5. Save evaluator-compatible *_ad_output.json for eval_interactive.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKPLACE_DIR = SCRIPT_DIR / "workplace"

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".hf_cache" / "sentence_transformers"))
os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT_ROOT / ".pip_cache"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch_cache"))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WORKPLACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKPLACE_DIR))

from context_builder import AVAILABLE_MODULES, AVAILABLE_PLOT_SUBMODULES, build_prompt_context
from segment_db import extract_face_data, load_segment_db, to_float
from workplace.pipeline_enhanced_gemma4 import extract_frames, postprocess_ad


FINAL_BY_MOVIE_DIR = Path("/mnt/disk1new/ylz/newAD/Step04_RunTest/step04_final_by_movie_new")
AD_CLIPS_DIR = Path("/mnt/disk1new/ylz/newAD/Step04_RunTest/ad_clips_final")
FACE_JSON_ROOT = Path("/mnt/disk1new/ylz/newAD/Step04_RunTest/step04_03_face_align/json")

DEFAULT_GEMMA4_MODEL = "/mnt/disk1new/ylz/newAD/models/gemma-4-26b-a4b-it"
DEFAULT_QWEN_MODEL = "/mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiment_results" / "gemma4_qwen_interactive"

PROMPT_LEAK_PATTERNS = (
    "Plot Database",
    "character mapping",
    "Target AD Clip",
    "[CMDQA AD text]",
    "[Nearby screenplay descriptions]",
    "[Scene indices]",
    "[Record types]",
    "[Related shot structured info]",
)


@dataclass
class InstructionCategory:
    category_id: str
    name: str
    name_cn: str
    description: str
    templates: List[str]
    templates_cn: List[str]
    weight: float = 1.0

    def sample_template(self, rng: random.Random, lang: str = "auto") -> Tuple[str, str]:
        idx = rng.randint(0, len(self.templates) - 1)
        if lang == "cn" or (lang == "auto" and rng.random() < 0.5):
            return self.templates_cn[idx % len(self.templates_cn)], "cn"
        return self.templates[idx % len(self.templates)], "en"


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
            "Describe in French",
            "Keep the current output language unchanged",
        ],
        templates_cn=[
            "用中文描述",
            "用英文描述",
            "用法语描述",
            "保持当前语言不变",
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


@dataclass
class ActiveInstruction:
    instruction_id: int
    category_id: str
    category_name: str
    slot_id: str
    raw_text: str
    language: str
    insert_timestamp_sec: float
    insert_segment_idx: int
    character_name: Optional[str] = None


@dataclass
class InsertionRecord:
    insertion_id: int
    movie_title: str
    insert_timestamp_sec: float
    segment_idx: int
    segment_start_sec: float
    segment_end_sec: float
    category_id: str
    category_name: str
    instruction_language: str
    instruction_text: str
    instruction_slot: str
    instruction_before: str
    instruction_after: str
    task_prompt_before: str
    task_prompt_after: str
    prompt_state_before: Dict[str, Any]
    prompt_state_after: Dict[str, Any]
    focus_line_after: str
    active_instructions_before: List[Dict[str, Any]]
    active_instructions_after: List[Dict[str, Any]]
    text_before: str
    text_after: str
    latency_before_sec: float
    latency_after_sec: float
    rewrite_latency_sec: float
    context_text: str = ""
    ref_ad: str = ""
    characters: List[str] = field(default_factory=list)


@dataclass
class SegmentRecord:
    segment_idx: int
    segment_start_sec: float
    segment_end_sec: float
    clip_path: str
    generated_text: str
    task_prompt: str
    raw_instruction_text: str
    active_instructions_count: int
    latency_sec: float
    context_text: str = ""
    ref_ad: str = ""
    characters: List[str] = field(default_factory=list)


@dataclass
class MovieExperimentResult:
    movie_title: str
    movie_duration_sec: float
    total_segments: int
    num_insertions: int
    insertion_records: List[InsertionRecord]
    segment_records: List[SegmentRecord]
    ad_entries: List[Dict[str, Any]]
    insertion_events: List[Dict[str, Any]]
    run_config: Dict[str, Any]


PROMPT_STATE_FIELDS = (
    "objective",
    "language",
    "verbosity",
    "style",
    "focus_primary",
    "focus_character",
    "focus_action",
    "focus_emotion",
    "focus_environment",
    "focus_interaction",
    "naming",
    "dialogue",
    "format",
    "constraints",
)

FOCUS_FIELDS = (
    "focus_primary",
    "focus_character",
    "focus_action",
    "focus_emotion",
    "focus_environment",
    "focus_interaction",
)


@dataclass
class PromptState:
    objective: str = "Describe what is happening in this clip concisely."
    language: str = ""
    verbosity: str = ""
    style: str = ""
    focus_primary: str = "Focus on visible actions, movements, and expressions."
    focus_character: str = ""
    focus_action: str = ""
    focus_emotion: str = ""
    focus_environment: str = ""
    focus_interaction: str = ""
    naming: str = (
        "If character names are mentioned in the context, use them "
        "(e.g. 'Don Vito Corleone walks...' not 'A man walks...')."
    )
    dialogue: str = "Do not quote dialogue."
    format: str = ""
    constraints: str = ""
    others: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = {key: getattr(self, key) for key in PROMPT_STATE_FIELDS}
        data["others"] = list(self.others)
        return data


def sanitize_context_text(context_text: str) -> str:
    cleaned: List[str] = []
    for raw in context_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in PROMPT_LEAK_PATTERNS:
            continue
        if line.startswith("Plot Database (The Subtext):"):
            continue
        if line.startswith("character mapping"):
            continue
        if line.startswith("Target AD Clip:"):
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        line = re.sub(r"^-+\s*", "", line).strip()
        if not line or line == "None":
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def normalize_suffix(text: str) -> str:
    text = " ".join(str(text).strip().split())
    text = re.sub(r'^[\'"]|[\'"]$', "", text)
    text = re.sub(r"^(Also:\s*)", "", text, flags=re.IGNORECASE)
    text = text.strip().rstrip(". ")
    return text


def normalize_prompt_line(text: str) -> str:
    text = " ".join(str(text).strip().split())
    text = re.sub(r'^[\'"]|[\'"]$', "", text)
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def cleanup_qwen_control_line(text: str) -> str:
    text = " ".join(str(text or "").strip().split())
    if not text:
        return ""
    text = re.sub(r"^[-*]\s*", "", text)
    text = re.sub(r"^(?:language|verbosity|style|focus_primary|focus_character|focus_action|focus_emotion|focus_environment|focus_interaction|format|constraints)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r'^[\'"]|[\'"]$', "", text)
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def build_task_prompt_from_state(prompt_state: PromptState, focus_line: str) -> str:
    lines: List[str] = []
    for key in ("objective", "language", "verbosity", "style"):
        value = normalize_prompt_line(getattr(prompt_state, key))
        if value:
            lines.append(value)
    focus_line = normalize_prompt_line(focus_line)
    if focus_line:
        lines.append(focus_line)
    for key in ("naming", "dialogue", "format", "constraints"):
        value = normalize_prompt_line(getattr(prompt_state, key))
        if value:
            lines.append(value)
    if prompt_state.others:
        merged_others = "; ".join(normalize_suffix(item) for item in prompt_state.others if normalize_suffix(item))
        if merged_others:
            lines.append(f"Also: {merged_others}.")
    return "\n".join(lines)


def build_prompt_surface_payload(prompt_state: PromptState, focus_line: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key in ("objective", "language", "verbosity", "style"):
        value = normalize_prompt_line(getattr(prompt_state, key))
        if value:
            payload[key] = value
    focus_line = normalize_prompt_line(focus_line)
    if focus_line:
        payload["focus_line"] = focus_line
    for key in ("naming", "dialogue", "format", "constraints"):
        value = normalize_prompt_line(getattr(prompt_state, key))
        if value:
            payload[key] = value
    others = [normalize_suffix(item) for item in prompt_state.others if normalize_suffix(item)]
    if others:
        payload["others"] = others
    return payload


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def _clean_prompt_surface_output(raw: str) -> List[str]:
    lines: List[str] = []
    for raw_line in str(raw or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("```"):
            continue
        line = re.sub(r"^\d+\.\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r'^[\'"]|[\'"]$', "", line)
        line = normalize_prompt_line(line)
        if line:
            lines.append(line)
    return lines


def _line_matches_rule(line: str, field: str, expected: Any) -> bool:
    low = line.lower()
    if field == "language":
        expected_low = str(expected).lower()
        if "chinese" in expected_low:
            return "chinese" in low
        if "english" in expected_low:
            return "english" in low
        if "french" in expected_low:
            return "french" in low
        if "current output language unchanged" in expected_low:
            return "current" in low and "language" in low and "unchanged" in low
    if field == "dialogue":
        return "dialogue" in low and "quote" in low
    if field == "naming":
        return "character name" in low or ("use" in low and "name" in low)
    if field == "focus_line":
        return low.startswith("focus") or "pay special attention" in low
    return True


def _validate_prompt_surface_lines(lines: List[str], payload: Dict[str, Any]) -> bool:
    expected_fields = list(payload.keys())
    if len(lines) != len(expected_fields):
        return False
    joined = "\n".join(lines)
    if _contains_cjk(joined):
        return False
    banned_markers = ("prompt", "template", "slot", "json", "metadata", "field order")
    if any(marker in joined.lower() for marker in banned_markers):
        return False
    for field, line in zip(expected_fields, lines):
        if not _line_matches_rule(line, field, payload[field]):
            return False
    return True


def load_instruction_categories(config_path: Optional[Path]) -> List[InstructionCategory]:
    if config_path and config_path.is_file():
        with config_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        categories: List[InstructionCategory] = []
        for item in payload.get("categories", []):
            categories.append(InstructionCategory(**item))
        if categories:
            return categories
    return DEFAULT_CATEGORIES


def canonical(text: str) -> str:
    return " ".join(str(text or "").strip().lower().replace("_", " ").split())


def _resolve_dir(base_dir: Path, movie_title: str) -> Optional[Path]:
    direct = base_dir / movie_title
    if direct.is_dir():
        return direct
    target = canonical(movie_title)
    if base_dir.is_dir():
        for p in base_dir.iterdir():
            if p.is_dir() and canonical(p.name) == target:
                return p
    return None


def _parse_clip_stem(ad_id: str, clip_index: Any, ad_order: int) -> str:
    ad_id = str(ad_id or "").strip()
    if ad_id:
        stem = Path(ad_id).stem
        if stem:
            match = re.search(r"(clip\d{4})_+(ad\d{4})", stem)
            if match:
                return f"{match.group(1)}_{match.group(2)}"
            match = re.search(r"__clip(\d+)__ad(\d+)$", stem)
            if match:
                return f"clip{match.group(1).zfill(4)}_ad{match.group(2).zfill(4)}"
            return stem
    try:
        idx = int(float(clip_index))
        return f"clip{idx:04d}_ad{ad_order:04d}"
    except Exception:
        return f"clip{ad_order:04d}_ad{ad_order:04d}"


def _fill_character_template(
    template: str,
    characters: Sequence[str],
    rng: random.Random,
) -> Tuple[str, Optional[str]]:
    if "{character}" in template and characters:
        chosen = rng.choice(list(characters))
        return template.replace("{character}", chosen), chosen
    return template.replace("{character}", "the main character"), None


def infer_instruction_slot(category_id: str, instruction_text: str) -> str:
    category_slot_map = {
        "lang_switch": "language",
        "detail_detailed": "verbosity",
        "detail_concise": "verbosity",
        "style": "style",
        "character": "focus_character",
        "event_action": "focus_action",
        "event_romance": "focus_interaction",
        "event_drama": "focus_emotion",
    }
    if category_id in category_slot_map:
        return category_slot_map[category_id]

    low = instruction_text.lower()
    if any(tok in low for tok in ("chinese", "english", "french", "bilingual", "same language", "current language", "中文", "英文", "法语", "双语", "当前语言")):
        return "language"
    if any(tok in low for tok in ("concise", "single sentence", "short", "简洁", "一句话")):
        return "verbosity"
    if any(tok in low for tok in ("detail", "detailed", "更多", "详细")):
        return "verbosity"
    return "others"


class InstructionState:
    def __init__(self):
        self._active: List[ActiveInstruction] = []
        self._next_id = 0
        self.prompt_state = PromptState()
        self.current_focus_line = normalize_prompt_line(self.prompt_state.focus_primary)
        self.current_task_prompt = build_task_prompt_from_state(self.prompt_state, self.current_focus_line)

    def snapshot(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": item.instruction_id,
                "category_id": item.category_id,
                "category_name": item.category_name,
                "slot_id": item.slot_id,
                "raw_text": item.raw_text,
                "language": item.language,
                "insert_timestamp_sec": item.insert_timestamp_sec,
                "insert_segment_idx": item.insert_segment_idx,
                "character_name": item.character_name,
            }
            for item in self._active
        ]

    def raw_instruction_text(self) -> str:
        texts: List[str] = []
        seen = set()
        for item in self._active:
            text = str(item.raw_text or "").strip()
            if not text or text in seen:
                continue
            texts.append(text)
            seen.add(text)
        return " | ".join(texts)

    def prompt_state_dict(self) -> Dict[str, Any]:
        return self.prompt_state.to_dict()

    def add_instruction_log(
        self,
        *,
        category_id: str,
        category_name: str,
        slot_id: str,
        raw_text: str,
        language: str,
        insert_timestamp_sec: float,
        insert_segment_idx: int,
        character_name: Optional[str],
    ) -> ActiveInstruction:
        instr = ActiveInstruction(
            instruction_id=self._next_id,
            category_id=category_id,
            category_name=category_name,
            slot_id=slot_id,
            raw_text=raw_text,
            language=language,
            insert_timestamp_sec=insert_timestamp_sec,
            insert_segment_idx=insert_segment_idx,
            character_name=character_name,
        )
        self._active.append(instr)
        self._next_id += 1
        return instr

    def sync_instruction_logs(
        self,
        *,
        category_id: str,
        category_name: str,
        slot_ids: Sequence[str],
        raw_text: str,
        language: str,
        insert_timestamp_sec: float,
        insert_segment_idx: int,
        character_name: Optional[str],
    ) -> List[ActiveInstruction]:
        unique_slots: List[str] = []
        for slot in slot_ids:
            if slot not in unique_slots:
                unique_slots.append(slot)

        removable_slots = {slot for slot in unique_slots if slot != "others"}
        if removable_slots:
            self._active = [item for item in self._active if item.slot_id not in removable_slots]
        if "others" in unique_slots and not self.prompt_state.others:
            self._active = [item for item in self._active if item.slot_id != "others"]

        created: List[ActiveInstruction] = []
        for slot in unique_slots:
            if slot == "others":
                if self.prompt_state.others:
                    created.append(
                        self.add_instruction_log(
                            category_id=category_id,
                            category_name=category_name,
                            slot_id=slot,
                            raw_text=raw_text,
                            language=language,
                            insert_timestamp_sec=insert_timestamp_sec,
                            insert_segment_idx=insert_segment_idx,
                            character_name=character_name,
                        )
                    )
                continue

            current_value = getattr(self.prompt_state, slot, "") if slot in PROMPT_STATE_FIELDS else ""
            if current_value:
                created.append(
                    self.add_instruction_log(
                        category_id=category_id,
                        category_name=category_name,
                        slot_id=slot,
                        raw_text=raw_text,
                        language=language,
                        insert_timestamp_sec=insert_timestamp_sec,
                        insert_segment_idx=insert_segment_idx,
                        character_name=character_name,
                    )
                )
        return created

    def apply_edit(self, edit: Dict[str, Any]) -> List[str]:
        updates = edit.get("updates", {}) if isinstance(edit, dict) else {}
        remove = edit.get("remove", []) if isinstance(edit, dict) else []
        append_others = edit.get("append_others", []) if isinstance(edit, dict) else []
        touched: List[str] = []

        for slot in remove:
            if slot == "others":
                self.prompt_state.others = []
                touched.append(slot)
            elif slot in PROMPT_STATE_FIELDS:
                setattr(self.prompt_state, slot, "")
                touched.append(slot)

        for slot, value in updates.items():
            if slot == "others":
                if value:
                    self.prompt_state.others = [normalize_suffix(str(value))]
                    touched.append(slot)
                continue
            if slot in PROMPT_STATE_FIELDS:
                setattr(self.prompt_state, slot, normalize_prompt_line(str(value)))
                touched.append(slot)

        for item in append_others:
            text = normalize_suffix(str(item))
            if text and text not in self.prompt_state.others:
                self.prompt_state.others.append(text)
                touched.append("others")

        return touched

    def refresh_task_prompt(self, qwen: "QwenInstructionRewriter") -> str:
        self.current_focus_line = qwen.compose_focus_line(self.prompt_state)
        self.current_task_prompt = qwen.compose_final_task_prompt(self.prompt_state, self.current_focus_line)
        return self.current_task_prompt


class QwenInstructionRewriter:
    def __init__(self, model_path: str, gpu: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        ).to(self.device).eval()

    def generate(self, prompt: str, max_new_tokens: int = 96) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        if not raw.strip():
            return {}
        candidates = []
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        candidates.extend(fenced)
        candidates.append(raw)
        for candidate in candidates:
            candidate = candidate.strip()
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                continue
            try:
                return json.loads(candidate[start:end + 1])
            except Exception:
                continue
        return {}

    def _fallback_edit(self, latest_request: str, primary_slot: str) -> Dict[str, Any]:
        latest_low = latest_request.lower()
        fallback_map = {
            "language": (
                "Keep the current output language unchanged."
                if any(tok in latest_low for tok in ("当前语言", "保持当前", "same language", "current language", "unchanged"))
                else "Describe in French."
                if any(tok in latest_low for tok in ("法语", "french"))
                else "Describe in Chinese."
                if any(tok in latest_low for tok in ("中文", "chinese"))
                else "Describe in English."
            ),
            "verbosity": "Use minimal words and focus on the most important visible action.",
            "focus_action": "Emphasize visible actions, gestures, and body movements.",
            "focus_character": "Pay special attention to the main character's presence, actions, and expressions.",
            "focus_emotion": "Emphasize facial expressions and visible emotional changes.",
            "focus_environment": "Describe the environment, lighting, and atmosphere more clearly.",
            "focus_interaction": "Pay special attention to interactions between characters.",
        }
        line = fallback_map.get(primary_slot, normalize_prompt_line(latest_request))
        updates: Dict[str, Any] = {}
        if primary_slot in PROMPT_STATE_FIELDS:
            updates[primary_slot] = line
            return {"updates": updates, "remove": [], "append_others": []}
        return {"updates": {}, "remove": [], "append_others": [normalize_suffix(latest_request)]}

    def plan_instruction_edit(
        self,
        prompt_state: PromptState,
        latest_request: str,
        primary_slot: str,
    ) -> Dict[str, Any]:
        prompt = f"""You are a prompt editor for interactive movie audio description.

Your job is to convert one informal user request into structured edits to a fixed task-prompt template.

You are NOT writing the final audio description.
You are NOT explaining your reasoning.
You are NOT copying the user's raw wording into the prompt.

You must produce short, natural English control lines that can be inserted directly into a movie-AD task prompt.

Current prompt state JSON:
{json.dumps(prompt_state.to_dict(), ensure_ascii=False, indent=2)}

New user request:
{latest_request}

Available editable slots:
- language
- verbosity
- style
- focus_primary
- focus_character
- focus_action
- focus_emotion
- focus_environment
- focus_interaction
- format
- constraints
- others

Rules:
- Split multi-intent user requests across the correct slots.
- Output English prompt-control lines only, even if the user speaks Chinese.
- Do not copy raw Chinese fragments into updates.
- Do not write explanations.
- Preserve the user's important semantic modifiers and translate them faithfully into English.
- If the user specifies a particular mood, atmosphere, tone, or intensity, keep that specificity instead of replacing it with a broader generic term.
- For explicitly requested semantics such as oppressive, tense, gloomy, romantic, poetic, documentary, playful, gentle, female lead, or background crowd, preserve that meaning in the update line.
- Preserve visible-only AD behavior.
- Do not edit objective, naming, or dialogue unless the user explicitly asks.
- If the new request conflicts with an existing slot, overwrite that slot.
- Use append_others only for content that truly does not fit any named slot.
- If one request clearly maps to several focus slots, update several slots.
- Keep each update concise and directly usable in a task prompt.
- If the request is ambiguous, use the most likely slot based on AD semantics.
- The instruction should guide description style, focus, or constraints. It must NOT narrate any scene content.
- Prefer slot mappings like:
  - "改成中文" -> language -> "Describe in Chinese."
  - "改成英文" -> language -> "Describe in English."
  - "改成法语" -> language -> "Describe in French."
  - "保持当前语言不变" -> language -> "Keep the current output language unchanged."
  - "更简洁" -> verbosity -> "Use minimal words and focus on the most important visible action."
  - "多描述动作" -> focus_action -> "Emphasize visible actions, gestures, and body movements."
  - "聚焦女主角/主角/某角色" -> focus_character -> "Pay special attention to the female lead's presence, actions, and expressions."
  - "多描述环境/光线/压抑氛围" -> focus_environment -> "Describe the environment, lighting, and oppressive atmosphere more clearly."
  - "多描述情绪/表情" -> focus_emotion -> "Emphasize facial expressions and visible emotional changes."
  - "多描述互动" -> focus_interaction -> "Pay special attention to interactions between characters."
- Good outputs:
  - "Describe in Chinese."
  - "Describe in French."
  - "Keep the current output language unchanged."
  - "Use minimal words and focus on the most important visible action."
  - "Emphasize visible actions, gestures, and body movements."
  - "Pay special attention to the female lead's presence, actions, and expressions."
  - "Describe the environment, lighting, and oppressive atmosphere more clearly."
- Bad outputs:
  - "中文."
  - "更多描述动作."
  - "Focus on 女主角."
  - "Describe the environment and overall atmosphere more clearly." when the user explicitly asked for an oppressive or tense atmosphere
  - long explanations
  - invented scene details
- The most likely primary slot for this request is: {primary_slot}

Return strict JSON only:
{{
  "updates": {{}},
  "remove": [],
  "append_others": []
}}
"""
        raw = self.generate(prompt, max_new_tokens=256)
        parsed = self._extract_json(raw)
        if not parsed:
            return self._fallback_edit(latest_request, primary_slot)
        parsed.setdefault("updates", {})
        parsed.setdefault("remove", [])
        parsed.setdefault("append_others", [])
        cleaned_updates: Dict[str, Any] = {}
        for slot, value in parsed["updates"].items():
            if slot in PROMPT_STATE_FIELDS:
                cleaned = cleanup_qwen_control_line(value)
                if cleaned:
                    cleaned_updates[slot] = cleaned
        parsed["updates"] = cleaned_updates
        parsed["remove"] = [slot for slot in parsed["remove"] if slot == "others" or slot in PROMPT_STATE_FIELDS]
        parsed["append_others"] = [normalize_suffix(item) for item in parsed["append_others"] if normalize_suffix(item)]
        return parsed

    def compose_focus_line(self, prompt_state: PromptState) -> str:
        focus_items = [
            cleanup_qwen_control_line(getattr(prompt_state, key))
            for key in FOCUS_FIELDS
            if cleanup_qwen_control_line(getattr(prompt_state, key))
        ]
        if not focus_items:
            return ""
        if len(focus_items) == 1:
            return focus_items[0]
        prompt = f"""You are merging focus-related prompt controls for movie audio description.

Focus instructions:
{json.dumps(focus_items, ensure_ascii=False, indent=2)}

Write exactly ONE short English line for the final task prompt.

Rules:
- Keep all compatible meaning.
- Remove redundancy.
- Keep the line natural and concise.
- Do not mix Chinese and English.
- Do not invent scene details.
- Keep it suitable for movie audio description.
- Preserve specific semantic modifiers from the source items, especially mood or atmosphere words such as oppressive, tense, gloomy, romantic, playful, or tender.
- Do not weaken a specific modifier into a generic phrase like "overall atmosphere" or "emotion."
- Start with "Focus" or "Pay special attention".
- Good examples:
  - "Focus on visible actions, body movements, and the female lead's expressions."
  - "Pay special attention to character interactions, visible emotions, and body language."
  - "Focus on visible actions, expressions, and the oppressive atmosphere created by the environment and lighting."
- Bad examples:
  - "Focus on 女主角."
  - "More action."
  - "Focus on the overall atmosphere." if the source specifically mentions an oppressive or tense mood
  - explanations
  - multiple lines

Output only the merged line.
"""
        raw = self.generate(prompt, max_new_tokens=96)
        line = raw.splitlines()[0].strip() if raw.strip() else focus_items[0]
        return cleanup_qwen_control_line(line)

    def compose_final_task_prompt(self, prompt_state: PromptState, focus_line: str) -> str:
        fallback = build_task_prompt_from_state(prompt_state, focus_line)
        payload = build_prompt_surface_payload(prompt_state, focus_line)
        if not payload:
            return fallback

        prompt = f"""You are polishing a task prompt for movie audio description.

Your input is a structured set of prompt-control fields.
Rewrite them into a clean final multi-line task prompt.

Rules:
- Preserve the meaning of every non-empty field.
- Preserve specific modifiers exactly in meaning whenever possible.
- Keep the original order of fields.
- Output one short natural English line per non-empty field.
- Do not drop hard constraints.
- Do not invent scene details.
- Do not mention prompts, templates, slots, JSON, or metadata.
- Keep the style concise, direct, and suitable for controlling AD generation.
- Normalize awkward fragments into fluent English.
- Merge only when necessary for fluency, but do not remove constraints.
- Do not replace specific qualifiers like oppressive, tense, romantic, poetic, documentary, female lead, or background crowd with broader generic wording.
- Output only the final prompt text, no explanation.

Field order:
1. objective
2. language
3. verbosity
4. style
5. focus_line
6. naming
7. dialogue
8. format
9. constraints
10. others

Input JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
        raw = self.generate(prompt, max_new_tokens=192)
        lines = _clean_prompt_surface_output(raw)
        if not _validate_prompt_surface_lines(lines, payload):
            return fallback
        return "\n".join(lines)


class SingleGPUGemma4ADEngine:
    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device

    def infer_one_segment(
        self,
        clip_path: Path,
        context_text: str,
        task_prompt: str,
        temperature: float = 0.2,
        max_new_tokens: int = 96,
        face_avatars: Optional[List[Path]] = None,
        character_names: Optional[List[str]] = None,
        num_beams: int = 3,
    ) -> Tuple[str, float, str]:
        torch.cuda.empty_cache()
        frames = extract_frames(clip_path, num_frames=8)
        if not frames:
            return "", 0.0, task_prompt

        from PIL import Image as PILImage

        avatar_images: List[PILImage.Image] = []
        if face_avatars:
            for avatar_path in face_avatars:
                try:
                    avatar_images.append(PILImage.open(str(avatar_path)).convert("RGB"))
                except Exception:
                    continue

        content: List[Dict[str, Any]] = []
        if context_text:
            content.append({"type": "text", "text": context_text + "\n"})
        if avatar_images and character_names:
            for img in avatar_images:
                content.append({"type": "image", "image": img})
            content.append({
                "type": "text",
                "text": f"[These faces belong to: {', '.join(character_names)}]\n",
            })
        for img in frames:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": task_prompt})

        messages = [{"role": "user", "content": content}]
        full_prompt = f"{context_text}\n{task_prompt}".strip() if context_text else task_prompt
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        all_images = avatar_images + frames
        inputs = self.processor(
            text=[text],
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if num_beams > 1:
            gen_kwargs.update(num_beams=num_beams, do_sample=False)
        else:
            if temperature > 0.01:
                gen_kwargs.update(do_sample=True, temperature=temperature)
            else:
                gen_kwargs.update(do_sample=False)

        start_time = time.monotonic()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.monotonic() - start_time

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        answer = self.processor.decode(generated, skip_special_tokens=True).strip()
        return answer, elapsed, full_prompt


def build_gemma4_engine_multi_gpu(
    model_path: str,
    gpus: Sequence[int],
    offload_dir: Optional[Path] = None,
    gpu_max_memory_gib: int = 46,
    gpu_max_memory_gib_by_device: Optional[Sequence[int]] = None,
    experts_implementation: Optional[str] = None,
) -> SingleGPUGemma4ADEngine:
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    gpu_list = list(gpus)
    if torch.cuda.is_available() and not gpu_list:
        raise ValueError("At least one Gemma GPU must be provided.")
    if gpu_max_memory_gib_by_device:
        max_memory_values = list(gpu_max_memory_gib_by_device)
        if len(max_memory_values) == 1:
            max_memory_values = max_memory_values * len(gpu_list)
        if len(max_memory_values) != len(gpu_list):
            raise ValueError(
                "gpu_max_memory_gib_by_device must provide either 1 value or one value per Gemma GPU."
            )
    else:
        max_memory_values = [gpu_max_memory_gib] * len(gpu_list)
    device = f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if experts_implementation:
        load_kwargs["experts_implementation"] = experts_implementation
    if torch.cuda.is_available():
        if offload_dir is not None:
            offload_dir.mkdir(parents=True, exist_ok=True)
            load_kwargs["offload_folder"] = str(offload_dir)
        max_memory_map = {
            gpu: f"{max_memory_gib}GiB"
            for gpu, max_memory_gib in zip(gpu_list, max_memory_values)
        }
        print(f"[gemma] max_memory={max_memory_map}", flush=True)
        if experts_implementation:
            print(f"[gemma] experts_implementation={experts_implementation}", flush=True)
        load_kwargs.update(
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory=max_memory_map,
        )
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_path,
        **load_kwargs,
    ).eval()
    return SingleGPUGemma4ADEngine(model=model, processor=processor, device=device)


def generate_ad(
    engine: SingleGPUGemma4ADEngine,
    *,
    clip_path: Path,
    context_text: str,
    task_prompt: str,
    face_avatars: Optional[List[Path]],
    character_names: Optional[List[str]],
    max_words: int,
    temperature: float,
    num_beams: int,
) -> Tuple[str, float, str]:
    text, latency, full_prompt = engine.infer_one_segment(
        clip_path=clip_path,
        context_text=context_text,
        task_prompt=task_prompt,
        temperature=temperature,
        max_new_tokens=96,
        face_avatars=face_avatars,
        character_names=character_names,
        num_beams=num_beams,
    )
    cleaned = postprocess_ad(text, max_words=max_words) or text.strip()
    return cleaned, latency, full_prompt


def select_insertion_indices(
    *,
    segments: Sequence[Dict[str, Any]],
    num_insertions: int,
    insertion_strategy: str,
    specific_timestamps: Optional[Sequence[float]],
    rng: random.Random,
) -> List[int]:
    n_seg = len(segments)
    if n_seg <= 1:
        return []

    if insertion_strategy == "manual" and specific_timestamps:
        insertion_indices: List[int] = []
        for ts in specific_timestamps:
            found = None
            for seg_idx, seg in enumerate(segments):
                start = to_float(seg.get("ad_movie_start_sec"), 0.0)
                end = to_float(seg.get("ad_movie_end_sec"), 0.0)
                if start <= ts <= end:
                    found = seg_idx
                    break
            if found is None:
                dists = [abs(ts - to_float(seg.get("ad_movie_start_sec"), 0.0)) for seg in segments]
                found = int(np.argmin(dists))
            insertion_indices.append(found)
        return sorted(set(idx for idx in insertion_indices if 0 < idx < n_seg))

    if insertion_strategy == "uniform":
        step = max(1, n_seg // (num_insertions + 1))
        indices = [min(step * (i + 1), n_seg - 1) for i in range(num_insertions)]
        return sorted(set(idx for idx in indices if idx > 0))

    available = list(range(1, n_seg))
    k = min(num_insertions, len(available))
    return sorted(rng.sample(available, k))


def save_experiment_result(result: MovieExperimentResult, output_path: Path) -> Tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    ad_output_path = output_path.with_name(
        output_path.stem.replace("_experiment", "_ad_output") + ".json"
    )
    if "_ad_output" not in ad_output_path.stem:
        ad_output_path = output_path.with_name(output_path.stem + "_ad_output.json")

    payload = {
        "movie": result.movie_title,
        "movie_title": result.movie_title,
        "total_gaps": result.total_segments,
        "generated_count": len(result.ad_entries),
        "video_duration_sec": result.movie_duration_sec,
        "ad_entries": result.ad_entries,
        "insertion_events": result.insertion_events,
        "num_insertions": result.num_insertions,
        "run_config": result.run_config,
    }
    with ad_output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path, ad_output_path


def resolve_segment_resources(
    *,
    movie_title: str,
    segment: Dict[str, Any],
    seg_idx: int,
) -> Dict[str, Any]:
    seg_start = to_float(segment.get("ad_movie_start_sec"), 0.0)
    seg_end = to_float(segment.get("ad_movie_end_sec"), 0.0)
    ad_id = str(segment.get("ad_id", "")).strip()
    clip_stem = _parse_clip_stem(ad_id=ad_id, clip_index=segment.get("clip_index"), ad_order=seg_idx + 1)

    clips_dir = _resolve_dir(AD_CLIPS_DIR, movie_title)
    face_dir = _resolve_dir(FACE_JSON_ROOT, movie_title)
    if clips_dir is None:
        raise FileNotFoundError(f"Clips directory not found for {movie_title}")

    clip_path = clips_dir / f"{clip_stem}.mp4"
    if not clip_path.is_file():
        raise FileNotFoundError(f"Clip not found: {clip_path}")

    face_matches: List[Dict[str, Any]] = []
    face_avatars: Optional[List[Path]] = None
    character_names: Optional[List[str]] = None
    if face_dir is not None:
        face_json = face_dir / f"{clip_stem}.json"
        face_matches, avatars = extract_face_data(face_json, max_face_records=4)
        if avatars:
            face_avatars = avatars
        if face_matches:
            character_names = [m.get("role_name", "") for m in face_matches if m.get("role_name")]

    raw_context, _ = build_prompt_context(
        segment=segment,
        modules=AVAILABLE_MODULES,
        plot_submodules=AVAILABLE_PLOT_SUBMODULES,
        face_matches=face_matches,
        max_description_lines=5,
        max_dialog_lines=8,
    )

    return {
        "seg_start": seg_start,
        "seg_end": seg_end,
        "clip_stem": clip_stem,
        "clip_path": clip_path,
        "face_matches": face_matches,
        "face_avatars": face_avatars,
        "character_names": character_names,
        "raw_context": raw_context,
        "context_text": sanitize_context_text(raw_context),
        "ref_ad": str(segment.get("cmdqa", {}).get("text", "")).strip(),
        "characters": [m.get("role_name", "") for m in face_matches if m.get("role_name")],
    }


def run_single_clip_test(args: argparse.Namespace) -> Dict[str, Any]:
    qwen = QwenInstructionRewriter(model_path=args.qwen_model, gpu=args.qwen_gpu)
    gemma = build_gemma4_engine_multi_gpu(
        model_path=args.gemma_model,
        gpus=args.gemma_gpus,
        gpu_max_memory_gib_by_device=args.gemma_max_memory_gib,
        experts_implementation=args.gemma_experts_implementation,
    )
    state = InstructionState()
    state.refresh_task_prompt(qwen)

    seg_db = load_segment_db(args.movie_title, final_by_movie_dir=FINAL_BY_MOVIE_DIR)
    segments = list(seg_db.segments)
    if not segments:
        raise RuntimeError(f"No segments found for movie: {args.movie_title}")

    seg_idx = args.test_segment_idx
    if seg_idx < 0 or seg_idx >= len(segments):
        raise IndexError(f"test segment idx out of range: {seg_idx}, total={len(segments)}")
    seg = segments[seg_idx]
    resources = resolve_segment_resources(movie_title=args.movie_title, segment=seg, seg_idx=seg_idx)
    context_text = resources["context_text"] if args.sanitize_context else resources["raw_context"]
    print(f"[single-clip] resolved seg={seg_idx} clip={resources['clip_stem']}")
    print("[single-clip] generating baseline...")

    before_text, before_latency, _ = generate_ad(
        gemma,
        clip_path=resources["clip_path"],
        context_text=context_text,
        task_prompt=state.current_task_prompt,
        face_avatars=resources["face_avatars"],
        character_names=resources["character_names"],
        max_words=args.max_words,
        temperature=args.temperature,
        num_beams=args.num_beams,
    )

    applied_edits: List[Dict[str, Any]] = []
    for request in args.test_requests:
        print(f"[single-clip] rewriting request: {request}")
        primary_slot = infer_instruction_slot("manual", request)
        structured_edit = qwen.plan_instruction_edit(
            prompt_state=state.prompt_state,
            latest_request=request,
            primary_slot=primary_slot,
        )
        touched_slots = state.apply_edit(structured_edit)
        effective_slots = [
            slot
            for slot in touched_slots
            if slot == "others" or (slot in PROMPT_STATE_FIELDS and getattr(state.prompt_state, slot, ""))
        ]
        state.sync_instruction_logs(
            category_id="manual_test",
            category_name="Manual Test",
            slot_ids=effective_slots,
            raw_text=request,
            language="auto",
            insert_timestamp_sec=resources["seg_start"],
            insert_segment_idx=seg_idx,
            character_name=None,
        )
        state.refresh_task_prompt(qwen)
        applied_edits.append(
            {
                "request": request,
                "primary_slot": primary_slot,
                "structured_edit": structured_edit,
                "effective_slots": effective_slots,
                "prompt_state": state.prompt_state_dict(),
                "focus_line": state.current_focus_line,
                "task_prompt": state.current_task_prompt,
            }
        )

    print("[single-clip] generating final output...")
    after_text, after_latency, _ = generate_ad(
        gemma,
        clip_path=resources["clip_path"],
        context_text=context_text,
        task_prompt=state.current_task_prompt,
        face_avatars=resources["face_avatars"],
        character_names=resources["character_names"],
        max_words=args.max_words,
        temperature=args.temperature,
        num_beams=args.num_beams,
    )

    payload = {
        "movie_title": args.movie_title,
        "segment_idx": seg_idx,
        "clip_path": str(resources["clip_path"]),
        "segment_start_sec": round(resources["seg_start"], 3),
        "segment_end_sec": round(resources["seg_end"], 3),
        "context_text": context_text,
        "reference_ad": resources["ref_ad"],
        "baseline_task_prompt": InstructionState().current_task_prompt,
        "baseline_output": before_text,
        "baseline_latency_sec": round(before_latency, 4),
        "applied_edits": applied_edits,
        "final_prompt_state": state.prompt_state_dict(),
        "final_focus_line": state.current_focus_line,
        "final_task_prompt": state.current_task_prompt,
        "final_output": after_text,
        "final_latency_sec": round(after_latency, 4),
        "active_instructions": state.snapshot(),
    }

    movie_slug = re.sub(r"[^A-Za-z0-9]+", "_", args.movie_title).strip("_")
    output_path = Path(args.output_dir) / f"{movie_slug}_single_clip_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[single-clip] movie={args.movie_title} seg={seg_idx} clip={resources['clip_stem']}")
    for idx, edit in enumerate(applied_edits, start=1):
        print(f"[request {idx}] {edit['request']}")
        print(f"  slots={edit['effective_slots']}")
        print(f"  focus={edit['focus_line']}")
    print("[baseline prompt]")
    print(payload["baseline_task_prompt"])
    print("[final prompt]")
    print(payload["final_task_prompt"])
    print(f"[baseline output] {before_text}")
    print(f"[final output] {after_text}")
    print(f"[saved] single_clip={output_path}")
    return payload


def run_qwen_only_test(args: argparse.Namespace) -> Dict[str, Any]:
    qwen = QwenInstructionRewriter(model_path=args.qwen_model, gpu=args.qwen_gpu)
    state = InstructionState()
    state.refresh_task_prompt(qwen)

    applied_edits: List[Dict[str, Any]] = []
    for request in args.test_requests:
        primary_slot = infer_instruction_slot("manual", request)
        structured_edit = qwen.plan_instruction_edit(
            prompt_state=state.prompt_state,
            latest_request=request,
            primary_slot=primary_slot,
        )
        touched_slots = state.apply_edit(structured_edit)
        effective_slots = [
            slot
            for slot in touched_slots
            if slot == "others" or (slot in PROMPT_STATE_FIELDS and getattr(state.prompt_state, slot, ""))
        ]
        state.sync_instruction_logs(
            category_id="manual_test",
            category_name="Manual Test",
            slot_ids=effective_slots,
            raw_text=request,
            language="auto",
            insert_timestamp_sec=0.0,
            insert_segment_idx=0,
            character_name=None,
        )
        state.refresh_task_prompt(qwen)
        applied_edits.append(
            {
                "request": request,
                "primary_slot": primary_slot,
                "structured_edit": structured_edit,
                "effective_slots": effective_slots,
                "prompt_state": state.prompt_state_dict(),
                "focus_line": state.current_focus_line,
                "task_prompt": state.current_task_prompt,
            }
        )

    payload = {
        "test_requests": list(args.test_requests),
        "baseline_task_prompt": InstructionState().current_task_prompt,
        "applied_edits": applied_edits,
        "final_prompt_state": state.prompt_state_dict(),
        "final_focus_line": state.current_focus_line,
        "final_task_prompt": state.current_task_prompt,
        "active_instructions": state.snapshot(),
    }

    output_path = Path(args.output_dir) / "qwen_only_prompt_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[qwen-only] requests")
    for idx, request in enumerate(args.test_requests, start=1):
        print(f"[request {idx}] {request}")
    print("[baseline prompt]")
    print(payload["baseline_task_prompt"])
    print("[final focus]")
    print(payload["final_focus_line"])
    print("[final prompt]")
    print(payload["final_task_prompt"])
    print(f"[saved] qwen_only={output_path}")
    return payload


def run_movie_experiment(args: argparse.Namespace) -> MovieExperimentResult:
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    categories = load_instruction_categories(
        Path(args.instruction_config) if args.instruction_config else None
    )
    qwen = QwenInstructionRewriter(model_path=args.qwen_model, gpu=args.qwen_gpu)
    gemma = build_gemma4_engine_multi_gpu(
        model_path=args.gemma_model,
        gpus=args.gemma_gpus,
        gpu_max_memory_gib_by_device=args.gemma_max_memory_gib,
        experts_implementation=args.gemma_experts_implementation,
    )
    state = InstructionState()
    state.refresh_task_prompt(qwen)

    seg_db = load_segment_db(args.movie_title, final_by_movie_dir=FINAL_BY_MOVIE_DIR)
    segments = list(seg_db.segments)
    if args.max_segments > 0:
        segments = segments[:args.max_segments]
    if not segments:
        raise RuntimeError(f"No segments found for movie: {args.movie_title}")

    total_duration = max(to_float(seg.get("ad_movie_end_sec"), 0.0) for seg in segments)
    insertion_indices = select_insertion_indices(
        segments=segments,
        num_insertions=args.num_insertions,
        insertion_strategy=args.insertion_strategy,
        specific_timestamps=args.specific_timestamps,
        rng=rng,
    )
    available_cats = categories
    if args.specific_categories:
        selected_ids = set(args.specific_categories)
        available_cats = [cat for cat in categories if cat.category_id in selected_ids] or categories
    cat_weights = [cat.weight for cat in available_cats]
    selected_cats = rng.choices(available_cats, weights=cat_weights, k=len(insertion_indices))

    print(f"[movie] {args.movie_title}")
    print(f"[movie] segments={len(segments)} duration={total_duration:.1f}s")
    print(f"[movie] insertion_indices={insertion_indices}")
    print(f"[model] qwen_gpu={args.qwen_gpu} gemma_gpus={args.gemma_gpus}")

    insertion_records: List[InsertionRecord] = []
    segment_records: List[SegmentRecord] = []
    ad_entries: List[Dict[str, Any]] = []
    insertion_events: List[Dict[str, Any]] = []

    all_characters: List[str] = []
    for seg in segments:
        chars = seg.get("characters") or []
        if isinstance(chars, list):
            for name in chars:
                text = str(name).strip()
                if text and text not in all_characters:
                    all_characters.append(text)

    insertion_lookup = {seg_idx: selected_cats[pos] for pos, seg_idx in enumerate(insertion_indices)}
    insertion_counter = 0

    for seg_idx, seg in enumerate(segments):
        try:
            resources = resolve_segment_resources(movie_title=args.movie_title, segment=seg, seg_idx=seg_idx)
        except FileNotFoundError as exc:
            print(f"[skip] seg={seg_idx} {exc}")
            continue
        seg_start = resources["seg_start"]
        seg_end = resources["seg_end"]
        clip_path = resources["clip_path"]
        face_avatars = resources["face_avatars"]
        character_names = resources["character_names"]
        context_text = resources["context_text"] if args.sanitize_context else resources["raw_context"]
        ref_ad = resources["ref_ad"]

        if seg_idx in insertion_lookup:
            cat = insertion_lookup[seg_idx]
            insertion_counter += 1
            template, lang = cat.sample_template(rng=rng)
            chosen_character = None
            if cat.category_id == "character":
                template, chosen_character = _fill_character_template(template, all_characters, rng)
            slot_id = infer_instruction_slot(cat.category_id, template)

            instruction_before = state.raw_instruction_text()
            active_before = state.snapshot()
            task_prompt_before = state.current_task_prompt
            prompt_state_before = state.prompt_state_dict()

            text_before, latency_before, _ = generate_ad(
                gemma,
                clip_path=clip_path,
                context_text=context_text,
                task_prompt=task_prompt_before,
                face_avatars=face_avatars,
                character_names=character_names,
                max_words=args.max_words,
                temperature=args.temperature,
                num_beams=args.num_beams,
            )

            rewrite_start = time.monotonic()
            structured_edit = qwen.plan_instruction_edit(
                prompt_state=state.prompt_state,
                latest_request=template,
                primary_slot=slot_id,
            )
            touched_slots = state.apply_edit(structured_edit)
            effective_slots = [
                slot
                for slot in touched_slots
                if slot == "others" or (slot in PROMPT_STATE_FIELDS and getattr(state.prompt_state, slot, ""))
            ]
            state.sync_instruction_logs(
                category_id=cat.category_id,
                category_name=cat.name,
                slot_ids=effective_slots,
                raw_text=template,
                language=lang,
                insert_timestamp_sec=seg_start,
                insert_segment_idx=seg_idx,
                character_name=chosen_character,
            )
            task_prompt_after = state.refresh_task_prompt(qwen)
            rewrite_latency = time.monotonic() - rewrite_start
            prompt_state_after = state.prompt_state_dict()
            instruction_after = state.raw_instruction_text()
            active_after = state.snapshot()

            text_after, latency_after, _ = generate_ad(
                gemma,
                clip_path=clip_path,
                context_text=context_text,
                task_prompt=task_prompt_after,
                face_avatars=face_avatars,
                character_names=character_names,
                max_words=args.max_words,
                temperature=args.temperature,
                num_beams=args.num_beams,
            )

            insertion_records.append(
                InsertionRecord(
                    insertion_id=insertion_counter,
                    movie_title=args.movie_title,
                    insert_timestamp_sec=seg_start,
                    segment_idx=seg_idx,
                    segment_start_sec=seg_start,
                    segment_end_sec=seg_end,
                    category_id=cat.category_id,
                    category_name=cat.name,
                    instruction_language=lang,
                    instruction_text=template,
                    instruction_slot="multi" if len(touched_slots) > 1 else (touched_slots[0] if touched_slots else slot_id),
                    instruction_before=instruction_before,
                    instruction_after=instruction_after,
                    task_prompt_before=task_prompt_before,
                    task_prompt_after=task_prompt_after,
                    prompt_state_before=prompt_state_before,
                    prompt_state_after=prompt_state_after,
                    focus_line_after=state.current_focus_line,
                    active_instructions_before=active_before,
                    active_instructions_after=active_after,
                    text_before=text_before,
                    text_after=text_after,
                    latency_before_sec=round(latency_before, 4),
                    latency_after_sec=round(latency_after, 4),
                    rewrite_latency_sec=round(rewrite_latency, 4),
                    context_text=context_text[:500],
                    ref_ad=ref_ad,
                    characters=all_characters[:10],
                )
            )
            insertion_events.append(
                {
                    "insertion_id": insertion_counter,
                    "gap_idx": seg_idx,
                    "timestamp_sec": round(seg_start, 3),
                    "category": cat.category_id,
                    "category_id": cat.category_id,
                    "category_name": cat.name,
                    "instruction_text": template,
                    "instruction_slot": slot_id,
                    "instruction_before": instruction_before,
                    "instruction_after": instruction_after,
                    "task_prompt_after": task_prompt_after,
                    "prompt_state_after": prompt_state_after,
                    "focus_line_after": state.current_focus_line,
                    "language": lang,
                    "active_instructions": active_after,
                }
            )

            final_text = text_after
            final_task_prompt = task_prompt_after
            final_raw_instruction = instruction_after
            final_active_count = len(active_after)

            print(
                f"[insert {insertion_counter}] seg={seg_idx} t={seg_start:.1f}s "
                f"cat={cat.category_id} slot={slot_id}"
            )
            print(f"  raw={template}")
            print(f"  focus={state.current_focus_line}")
            print(f"  before={text_before[:120]}")
            print(f"  after ={text_after[:120]}")
        else:
            final_text, final_latency, _ = generate_ad(
                gemma,
                clip_path=clip_path,
                context_text=context_text,
                task_prompt=state.current_task_prompt,
                face_avatars=face_avatars,
                character_names=character_names,
                max_words=args.max_words,
                temperature=args.temperature,
                num_beams=args.num_beams,
            )
            latency_after = final_latency
            final_task_prompt = state.current_task_prompt
            final_raw_instruction = state.raw_instruction_text()
            final_active_count = len(state.snapshot())

        segment_latency = latency_after if seg_idx in insertion_lookup else final_latency
        segment_records.append(
            SegmentRecord(
                segment_idx=seg_idx,
                segment_start_sec=seg_start,
                segment_end_sec=seg_end,
                clip_path=str(clip_path),
                generated_text=final_text,
                task_prompt=final_task_prompt,
                raw_instruction_text=final_raw_instruction,
                active_instructions_count=final_active_count,
                latency_sec=round(segment_latency, 4),
                context_text=context_text[:500],
                ref_ad=ref_ad,
                characters=all_characters[:10],
            )
        )

        ctx_before: List[str] = []
        ctx_after: List[str] = []
        for row in (seg.get("matched_rows_selected") or []):
            dialog = str(row.get("align_dialog") or row.get("dialog") or "").strip()
            if not dialog:
                continue
            row_start = to_float(row.get("start_time_sec"), seg_start)
            if row_start <= (seg_start + seg_end) / 2:
                ctx_before.append(dialog)
            else:
                ctx_after.append(dialog)

        ad_entries.append(
            {
                "gap_id": seg_idx + 1,
                "scene_index": str(seg.get("scene_index", seg.get("index_result", ""))),
                "location": str(seg.get("location", "")),
                "gap_start_sec": round(seg_start, 3),
                "gap_end_sec": round(seg_end, 3),
                "gap_duration_sec": round(seg_end - seg_start, 3),
                "characters": [str(c) for c in (seg.get("characters") or []) if str(c).strip()],
                "context_before": ctx_before[-5:],
                "context_after": ctx_after[:8],
                "ad_text": final_text,
                "inference_time_sec": round(segment_latency, 3),
                "active_instructions": state.snapshot(),
                "active_instruction_count": len(state.snapshot()),
                "task_prompt": final_task_prompt,
            }
        )

        print(
            f"[segment {seg_idx + 1}/{len(segments)}] "
            f"t={seg_start:.1f}s active={len(state.snapshot())} text={final_text[:100]}"
        )

    return MovieExperimentResult(
        movie_title=args.movie_title,
        movie_duration_sec=total_duration,
        total_segments=len(segments),
        num_insertions=len(insertion_records),
        insertion_records=insertion_records,
        segment_records=segment_records,
        ad_entries=ad_entries,
        insertion_events=insertion_events,
        run_config={
            "movie_title": args.movie_title,
            "max_segments": args.max_segments,
            "num_insertions": args.num_insertions,
            "insertion_strategy": args.insertion_strategy,
            "specific_timestamps": args.specific_timestamps,
            "specific_categories": args.specific_categories,
            "temperature": args.temperature,
            "num_beams": args.num_beams,
            "seed": args.seed,
            "sanitize_context": args.sanitize_context,
            "gemma_model": args.gemma_model,
            "qwen_model": args.qwen_model,
            "gemma_gpus": args.gemma_gpus,
            "qwen_gpu": args.qwen_gpu,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma4 + Qwen interactive AD experiment.")
    parser.add_argument("--movie-title", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--instruction-config", type=str, default="")
    parser.add_argument("--gemma-model", type=str, default=DEFAULT_GEMMA4_MODEL)
    parser.add_argument("--qwen-model", type=str, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--gemma-gpus", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--gemma-max-memory-gib", type=int, nargs="+", default=None)
    parser.add_argument("--gemma-experts-implementation", choices=["eager", "batched_mm", "grouped_mm"], default=None)
    parser.add_argument("--qwen-gpu", type=int, default=0)
    parser.add_argument("--num-insertions", type=int, default=3)
    parser.add_argument("--insertion-strategy", choices=("random", "uniform", "manual"), default="random")
    parser.add_argument("--specific-timestamps", type=float, nargs="*", default=None)
    parser.add_argument("--specific-categories", type=str, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=22)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--sanitize-context", action="store_true", default=True)
    parser.add_argument("--no-sanitize-context", dest="sanitize_context", action="store_false")
    parser.add_argument("--single-clip-test", action="store_true")
    parser.add_argument("--qwen-only-test", action="store_true")
    parser.add_argument("--test-segment-idx", type=int, default=2)
    parser.add_argument(
        "--test-requests",
        type=str,
        nargs="+",
        default=["改成中文，更多描述动作，聚焦于女主角"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.qwen_only_test:
        run_qwen_only_test(args)
        return

    if args.single_clip_test:
        run_single_clip_test(args)
        return

    result = run_movie_experiment(args)
    movie_slug = re.sub(r"[^A-Za-z0-9]+", "_", args.movie_title).strip("_")
    output_path = output_dir / f"{movie_slug}_experiment.json"
    full_path, ad_output_path = save_experiment_result(result, output_path)
    print(f"[saved] full={full_path}")
    print(f"[saved] ad_output={ad_output_path}")


if __name__ == "__main__":
    main()
