#!/usr/bin/env python3
"""
Prompt context builders.

Builds per-segment prompt context in order:
  Plot Database → character mapping → scene info → Video

Uses single-line task prompt (multi-line/bracket prompts confuse this model).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


TASK_PROMPT_TEMPLATE = (
    "Describe what is happening in this clip concisely. "
    "Focus on visible actions, movements, and expressions. "
    "If character names are mentioned in the context, use them "
    "(e.g. 'Don Vito Corleone walks...' not 'A man walks...'). "
    "Do not quote dialogue. "
    "{}"
)

CN_INSTRUCTION_MAP = {
    "加入环境氛围的描述": "Describe the environment, setting and atmosphere in detail",
    "多描述一下主要人物的表情变化": "Focus on facial expressions and emotional changes of the characters",
    "重点关注人物之间的互动": "Focus on interactions and body language between characters",
    "描述场景中的光线和色彩": "Describe the lighting, colors and visual mood of the scene",
    "详细描述画面中的每个细节": "Describe every visible detail in the frame",
    "用简洁的语言描述": "Use concise language",
    "增加更多动作解读，描述肢体语言": "Describe actions, gestures and body language in more detail",
    "关注镜头运动和场景切换": "Pay attention to camera movements and scene transitions",
    "用更简洁的语言描述，不要超过一句话": "Be very concise, describe in a single sentence",
}

AVAILABLE_MODULES = ("plot", "character", "scene")
AVAILABLE_PLOT_SUBMODULES = ("ad_text", "descriptions", "scene_indices", "record_types")


def dedupe_keep_order(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for v in values:
        t = str(v or "").strip()
        if not t:
            continue
        tok = t.lower()
        if tok in seen:
            continue
        seen.add(tok)
        out.append(t)
    return out


def trim_lines(items: Sequence[str], max_lines: int) -> List[str]:
    if max_lines <= 0:
        return []
    return list(items[:max_lines])


def build_plot_text(
    segment: Dict[str, Any],
    max_description_lines: int,
    plot_submodules: Sequence[str],
) -> Tuple[str, Dict[str, Any]]:
    cmdqa = segment.get("cmdqa", {}) if isinstance(segment.get("cmdqa", {}), dict) else {}
    aggregated = segment.get("aggregated", {}) if isinstance(segment.get("aggregated", {}), dict) else {}

    payload: Dict[str, Any] = {}
    sections: List[str] = []

    if "ad_text" in plot_submodules:
        ad_text = str(cmdqa.get("text", "")).strip()
        payload["ad_text"] = ad_text
        sections.append(f"[CMDQA AD text]\n- {ad_text if ad_text else 'None'}")

    if "descriptions" in plot_submodules:
        descriptions = dedupe_keep_order(aggregated.get("descriptions", []) or [])
        descriptions = trim_lines(descriptions, max_description_lines)
        payload["descriptions"] = descriptions
        if descriptions:
            sections.append("[Nearby screenplay descriptions]\n" + "\n".join(f"- {x}" for x in descriptions))
        else:
            sections.append("[Nearby screenplay descriptions]\n- None")

    if "scene_indices" in plot_submodules:
        scene_indices = dedupe_keep_order(aggregated.get("scene_indices", []) or [])
        payload["scene_indices"] = scene_indices
        text_val = ", ".join(str(x) for x in scene_indices) if scene_indices else "None"
        sections.append(f"[Scene indices]\n- {text_val}")

    if "record_types" in plot_submodules:
        record_types = dedupe_keep_order(aggregated.get("record_types", []) or [])
        payload["record_types"] = record_types
        text_val = ", ".join(str(x) for x in record_types) if record_types else "None"
        sections.append(f"[Record types]\n- {text_val}")

    if not sections:
        return "- None", payload
    return "\n\n".join(sections), payload


def build_character_text(face_matches: List[Dict[str, Any]]) -> str:
    if not face_matches:
        return "- None"
    lines: List[str] = []
    for item in face_matches:
        role_name = str(item.get("role_name", "")).strip() or "Unknown"
        lines.append(f"<rolename>{role_name}</rolename>")
    return "\n".join(lines)


def build_scene_text(segment: Dict[str, Any], max_dialog_lines: int) -> str:
    aggregated = segment.get("aggregated", {}) if isinstance(segment.get("aggregated", {}), dict) else {}

    locations = dedupe_keep_order(aggregated.get("locations", []) or [])
    dialogs = dedupe_keep_order(aggregated.get("dialogs", []) or [])
    align_dialogs = dedupe_keep_order(aggregated.get("align_dialogs", []) or [])

    dialogs = trim_lines(dialogs, max_dialog_lines)
    align_dialogs = trim_lines(align_dialogs, max_dialog_lines)

    def _join(vals: Sequence[str]) -> str:
        return ", ".join(vals) if vals else "None"

    lines = [
        f"Locations: {_join(locations)}",
        f"Dialogs (context only, do not quote): {_join(dialogs)}",
        f"Align dialogs (context only, do not quote): {_join(align_dialogs)}",
    ]
    return "\n".join(lines)


def build_prompt_context(
    segment: Dict[str, Any],
    modules: Sequence[str],
    plot_submodules: Sequence[str],
    face_matches: List[Dict[str, Any]],
    max_description_lines: int = 5,
    max_dialog_lines: int = 8,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build prompt context for a SINGLE segment.
    Order: plot -> character -> scene
    No <ImageHere> markers — upload methods handle those.
    """
    module_payload: Dict[str, Any] = {}
    parts: List[str] = []

    if "plot" in modules:
        text, plot_payload = build_plot_text(
            segment=segment,
            max_description_lines=max_description_lines,
            plot_submodules=plot_submodules,
        )
        module_payload["plot"] = plot_payload
        parts.append(f"Plot Database (The Subtext):\n{text}")

    if "character" in modules:
        text = build_character_text(face_matches=face_matches)
        module_payload["character"] = text
        parts.append(f"character mapping (name + avatar):\n{text}")

    if "scene" in modules:
        text = build_scene_text(segment=segment, max_dialog_lines=max_dialog_lines)
        module_payload["scene"] = text
        parts.append(f"[Related shot structured info]\n{text}")

    parts.append("Target AD Clip:")
    context = "\n\n".join(parts).strip()
    return context, module_payload


def _translate_instruction(instruction: str) -> str:
    stripped = instruction.strip()
    if not stripped:
        return ""
    if stripped in CN_INSTRUCTION_MAP:
        return CN_INSTRUCTION_MAP[stripped]
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in stripped)
    if has_cjk:
        return stripped
    return stripped


def build_task_prompt(custom_instruction: str = "") -> str:
    translated = _translate_instruction(custom_instruction)
    if not translated:
        return TASK_PROMPT_TEMPLATE.format("")
    return TASK_PROMPT_TEMPLATE.format(
        f"Also: {translated}. "
    )
