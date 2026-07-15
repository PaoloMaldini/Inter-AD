#!/usr/bin/env python3
"""
Shared helpers for Gemma4 + Qwen interactive AD experiments.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

STREAMING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STREAMING_ROOT.parent
WORKPLACE_DIR = STREAMING_ROOT / "workplace"

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".hf_cache" / "sentence_transformers"))
os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT_ROOT / ".pip_cache"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch_cache"))

if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))
if str(WORKPLACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKPLACE_DIR))

from pipeline_enhanced_gemma4 import extract_frames, postprocess_ad as base_postprocess_ad


DEFAULT_GEMMA4_MODEL = "/mnt/disk1new/ylz/newAD/models/gemma-4-26b-a4b-it"
DEFAULT_QWEN_MODEL = "/mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct"
DEFAULT_AD_WORDS_PER_MINUTE = 150
DEFAULT_WORD_BUDGET_RATIO = 0.85
DEFAULT_MIN_WORD_BUDGET = 5

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
        payload = {key: getattr(self, key) for key in PROMPT_STATE_FIELDS}
        payload["others"] = list(self.others)
        return payload


@dataclass
class ActivePromptInstruction:
    instruction_id: int
    category_id: str
    category_name: str
    slot_id: str
    raw_text: str
    language: str
    insert_timestamp_sec: float
    insert_segment_idx: int
    character_name: Optional[str] = None


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
        if line and line != "None":
            cleaned.append(line)
    return "\n".join(cleaned)


def normalize_suffix(text: str) -> str:
    text = " ".join(str(text or "").strip().split())
    text = re.sub(r'^[\'"]|[\'"]$', "", text)
    text = re.sub(r"^(Also:\s*)", "", text, flags=re.IGNORECASE)
    return text.strip().rstrip(". ")


def normalize_prompt_line(text: str) -> str:
    text = " ".join(str(text or "").strip().split())
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
    text = re.sub(
        r"^(?:language|verbosity|style|focus_primary|focus_character|focus_action|focus_emotion|focus_environment|focus_interaction|format|constraints)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
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
        merged = "; ".join(
            normalize_suffix(item) for item in prompt_state.others if normalize_suffix(item)
        )
        if merged:
            lines.append(f"Also: {merged}.")
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


def load_instruction_categories(
    config_path: Optional[Path] = None,
    filter_ids: Optional[Sequence[str]] = None,
) -> List[InstructionCategory]:
    categories = DEFAULT_CATEGORIES
    if config_path and config_path.is_file():
        with config_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        loaded = [InstructionCategory(**item) for item in payload.get("categories", [])]
        if loaded:
            categories = loaded
    if filter_ids:
        wanted = set(filter_ids)
        filtered = [cat for cat in categories if cat.category_id in wanted]
        if filtered:
            categories = filtered
    return categories


def safe_movie_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def get_video_duration(video_path: Path) -> float:
    import subprocess

    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(proc.stdout.strip())
    except Exception:
        return 0.0


def compute_dynamic_word_budget(
    gap_duration_sec: float,
    *,
    words_per_minute: int = DEFAULT_AD_WORDS_PER_MINUTE,
    budget_ratio: float = DEFAULT_WORD_BUDGET_RATIO,
    min_words: int = DEFAULT_MIN_WORD_BUDGET,
    hard_cap: Optional[int] = None,
) -> int:
    if gap_duration_sec <= 0:
        budget = max(1, min_words)
    else:
        raw_budget = math.floor((gap_duration_sec * words_per_minute / 60.0) * budget_ratio)
        budget = max(min_words, raw_budget)
    if hard_cap is not None and hard_cap > 0:
        budget = min(budget, hard_cap)
    return max(1, int(budget))


def build_runtime_task_prompt(
    base_task_prompt: str,
    *,
    gap_duration_sec: float,
    max_words: int,
) -> str:
    lines = [line.strip() for line in str(base_task_prompt or "").splitlines() if line.strip()]
    lines.append(
        f"Keep it to one complete sentence that fits within this {gap_duration_sec:.1f}-second silent gap."
    )
    lines.append(f"Use no more than {max_words} words.")
    return "\n".join(lines)


def _soft_trim_to_word_budget(text: str, max_words: int) -> str:
    if not text or max_words <= 0:
        return text.strip()

    words = text.split()
    if len(words) <= max_words:
        return text.strip()

    trimmed_words = words[:max_words]
    trimmed = " ".join(trimmed_words).strip()

    # Prefer ending on a natural clause/sentence boundary if one appears near the limit.
    best_boundary = max(trimmed.rfind(ch) for ch in ".!?;,:")
    if best_boundary >= int(len(trimmed) * 0.6):
        trimmed = trimmed[: best_boundary + 1].strip()
    else:
        dangling_tokens = {
            "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "at",
            "with", "for", "from", "by", "through", "into", "onto", "over", "under",
            "his", "her", "their", "its", "this", "that", "these", "those",
        }
        while trimmed_words and trimmed_words[-1].lower().strip(".,;:!?") in dangling_tokens:
            trimmed_words.pop()
        if trimmed_words:
            trimmed = " ".join(trimmed_words).rstrip(",;:")

    trimmed = trimmed.strip()
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed


def _fill_character_template(
    template: str,
    characters: Sequence[str],
    rng: random.Random,
) -> Tuple[str, Optional[str]]:
    if "{character}" in template and characters:
        chosen = rng.choice(list(characters))
        return template.replace("{character}", chosen), chosen
    return template.replace("{character}", "the main character"), None


def weighted_sample_without_replacement(
    categories: Sequence[InstructionCategory],
    k: int,
    rng: random.Random,
) -> List[InstructionCategory]:
    pool = list(categories)
    chosen: List[InstructionCategory] = []
    while pool and len(chosen) < k:
        weights = [cat.weight for cat in pool]
        cat = rng.choices(pool, weights=weights, k=1)[0]
        chosen.append(cat)
        pool.remove(cat)
    return chosen


def build_composite_request(
    categories: Sequence[InstructionCategory],
    rng: random.Random,
    all_characters: Sequence[str],
    min_parts: int = 1,
    max_parts: int = 3,
) -> Dict[str, Any]:
    if not categories:
        raise ValueError("No instruction categories available.")

    max_parts = max(1, min(max_parts, len(categories)))
    min_parts = max(1, min(min_parts, max_parts))
    num_parts = rng.randint(min_parts, max_parts)
    selected = weighted_sample_without_replacement(categories, num_parts, rng)
    request_lang = "cn" if rng.random() < 0.5 else "en"

    clauses: List[str] = []
    parts: List[Dict[str, Any]] = []
    for cat in selected:
        text, lang = cat.sample_template(rng=rng, lang=request_lang)
        chosen_character = None
        if cat.category_id == "character":
            text, chosen_character = _fill_character_template(text, all_characters, rng)
        clauses.append(text.strip())
        parts.append(
            {
                "category_id": cat.category_id,
                "category_name": cat.name,
                "language": lang,
                "text": text.strip(),
                "character_name": chosen_character,
            }
        )

    if request_lang == "cn":
        request_text = "，".join(clauses)
    else:
        request_text = "; ".join(clauses)

    if len(parts) == 1:
        category_id = parts[0]["category_id"]
        category_name = parts[0]["category_name"]
    else:
        category_id = "composite"
        category_name = "Composite Request"

    return {
        "request_text": request_text,
        "request_language": request_lang,
        "category_id": category_id,
        "category_name": category_name,
        "components": parts,
    }


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


def infer_primary_slot_for_request(request_meta: Dict[str, Any]) -> str:
    parts = request_meta.get("components", [])
    slots = [infer_instruction_slot(part["category_id"], part["text"]) for part in parts]
    unique_slots = list(dict.fromkeys(slots))
    if not unique_slots:
        return infer_instruction_slot(request_meta.get("category_id", ""), request_meta.get("request_text", ""))
    if len(unique_slots) == 1:
        return unique_slots[0]
    if "language" in unique_slots:
        return "language"
    for slot in (
        "focus_character",
        "focus_action",
        "focus_emotion",
        "focus_environment",
        "focus_interaction",
        "verbosity",
        "style",
    ):
        if slot in unique_slots:
            return slot
    return "others"


def select_insertion_indices(
    total_segments: int,
    num_insertions: int,
    rng: random.Random,
) -> List[int]:
    if total_segments <= 1 or num_insertions <= 0:
        return []
    available = list(range(1, total_segments))
    k = min(num_insertions, len(available))
    return sorted(rng.sample(available, k))


class InstructionState:
    def __init__(self):
        self._active: List[ActivePromptInstruction] = []
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
            if text and text not in seen:
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
    ) -> ActivePromptInstruction:
        item = ActivePromptInstruction(
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
        self._active.append(item)
        self._next_id += 1
        return item

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
    ) -> None:
        unique_slots = list(dict.fromkeys(slot_ids))
        removable = {slot for slot in unique_slots if slot != "others"}
        if removable:
            self._active = [item for item in self._active if item.slot_id not in removable]
        if "others" in unique_slots and not self.prompt_state.others:
            self._active = [item for item in self._active if item.slot_id != "others"]

        for slot in unique_slots:
            if slot == "others":
                if self.prompt_state.others:
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
                continue

            current_value = getattr(self.prompt_state, slot, "") if slot in PROMPT_STATE_FIELDS else ""
            if current_value:
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
                text = normalize_suffix(str(value))
                if text:
                    self.prompt_state.others = [text]
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
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        if not raw.strip():
            return {}
        candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
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
        if primary_slot in PROMPT_STATE_FIELDS:
            return {"updates": {primary_slot: line}, "remove": [], "append_others": []}
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
        from PIL import Image as PILImage

        torch.cuda.empty_cache()
        frames = extract_frames(clip_path, num_frames=8)
        if not frames:
            return "", 0.0, task_prompt

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
            content.append({"type": "text", "text": f"[These faces belong to: {', '.join(character_names)}]\n"})
        for img in frames:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": task_prompt})

        messages = [{"role": "user", "content": content}]
        full_prompt = f"{context_text}\n{task_prompt}".strip() if context_text else task_prompt
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    model = Gemma4ForConditionalGeneration.from_pretrained(model_path, **load_kwargs).eval()
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
    dynamic_max_new_tokens = max(24, min(128, int(max_words * 3.2))) if max_words > 0 else 96
    text, latency, full_prompt = engine.infer_one_segment(
        clip_path=clip_path,
        context_text=context_text,
        task_prompt=task_prompt,
        temperature=temperature,
        max_new_tokens=dynamic_max_new_tokens,
        face_avatars=face_avatars,
        character_names=character_names,
        num_beams=num_beams,
    )
    cleaned = base_postprocess_ad(text, max_words=10000) or text.strip()
    cleaned = _soft_trim_to_word_budget(cleaned, max_words) or cleaned
    return cleaned, latency, full_prompt


def save_experiment_result(result, output_path: Path) -> Tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    ad_output_path = output_path.with_name(
        output_path.stem.replace("_experiment", "_ad_output").replace("_instructed", "_ad_output") + ".json"
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
        "insertion_events": result.run_config.get("insertion_events", []),
        "num_insertions": result.num_insertions,
        "run_config": result.run_config,
    }
    with ad_output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path, ad_output_path
