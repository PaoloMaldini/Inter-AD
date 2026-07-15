#!/usr/bin/env python3
"""
analysis_cache.py — Cache intermediate analysis results so they survive across runs.

Saves/loads:
  - Whisper transcription turns (from speech_transcriber)
  - VAD gaps (from vad_gap_detector)
  - Scene boundaries (from vad_gap_detector)

Cache key: movie_name (video stem)
Cache dir:  {output_dir}/_analysis_cache/
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vad_gap_detector import AudioGap, VideoSceneBoundary

CACHE_SUBDIR = "_analysis_cache"


@dataclass
class CachedAnalysis:
    movie_name: str
    whisper_turns: List[Dict[str, Any]]
    whisper_lang: str
    gaps: List[Dict[str, Any]]
    scene_boundaries: List[Dict[str, Any]]
    params: Dict[str, Any]
    cached_at: float


def _turn_to_dict(turn: Any) -> Dict[str, Any]:
    return {
        "turn_id": turn.turn_id,
        "speaker_label": turn.speaker_label,
        "start_sec": turn.start_sec,
        "end_sec": turn.end_sec,
        "text": turn.text,
    }


def _dict_to_turns(raw: List[Dict[str, Any]]) -> List[Any]:
    from speech_transcriber import SpeakerTurn
    return [
        SpeakerTurn(
            turn_id=r["turn_id"],
            speaker_label=r["speaker_label"],
            start_sec=r["start_sec"],
            end_sec=r["end_sec"],
            text=r["text"],
        )
        for r in raw
    ]


def _gap_to_dict(gap: AudioGap) -> Dict[str, Any]:
    return {
        "gap_id": gap.gap_id,
        "gap_start_sec": gap.gap_start_sec,
        "gap_end_sec": gap.gap_end_sec,
        "gap_duration_sec": gap.gap_duration_sec,
    }


def _dict_to_gaps(raw: List[Dict[str, Any]]) -> List[AudioGap]:
    return [
        AudioGap(
            gap_id=r["gap_id"],
            gap_start_sec=r["gap_start_sec"],
            gap_end_sec=r["gap_end_sec"],
            gap_duration_sec=r["gap_duration_sec"],
            audio_before_end_sec=r["gap_start_sec"],
            audio_after_start_sec=r["gap_end_sec"],
        )
        for r in raw
    ]


def _scene_to_dict(b: VideoSceneBoundary) -> Dict[str, Any]:
    return {"timestamp_sec": b.timestamp_sec, "score": b.score}


def _dict_to_scenes(raw: List[Dict[str, Any]]) -> List[VideoSceneBoundary]:
    return [VideoSceneBoundary(timestamp_sec=r["timestamp_sec"], score=r["score"]) for r in raw]


def _cache_dir(output_dir: Path) -> Path:
    d = output_dir / CACHE_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(output_dir: Path, movie_name: str) -> Path:
    safe = movie_name.replace(" ", "_").replace("/", "_")
    return _cache_dir(output_dir) / f"{safe}.json"


def save_analysis(
    output_dir: Path,
    movie_name: str,
    whisper_turns: Optional[List[Any]],
    whisper_lang: str,
    gaps: List[AudioGap],
    scene_boundaries: List[VideoSceneBoundary],
    params: Dict[str, Any],
) -> Path:
    data = CachedAnalysis(
        movie_name=movie_name,
        whisper_turns=[_turn_to_dict(t) for t in (whisper_turns or [])],
        whisper_lang=whisper_lang,
        gaps=[_gap_to_dict(g) for g in gaps],
        scene_boundaries=[_scene_to_dict(b) for b in scene_boundaries],
        params=params,
        cached_at=time.time(),
    )
    cp = _cache_path(output_dir, movie_name)
    with cp.open("w", encoding="utf-8") as f:
        json.dump(asdict(data), f, ensure_ascii=False, indent=2)
    return cp


def load_analysis(
    output_dir: Path,
    movie_name: str,
    min_gap_sec: float,
    silence_threshold_db: float,
    min_silence_dur: float,
) -> Tuple[Optional[List[Any]], str, Optional[List[AudioGap]], Optional[List[VideoSceneBoundary]]]:
    cp = _cache_path(output_dir, movie_name)
    if not cp.is_file():
        return None, "", None, None

    try:
        raw = json.loads(cp.read_text(encoding="utf-8"))
        cached_params = raw.get("params", {})
        if (
            cached_params.get("min_gap_sec") != min_gap_sec
            or cached_params.get("silence_threshold_db") != silence_threshold_db
            or cached_params.get("min_silence_dur") != min_silence_dur
        ):
            return None, "", None, None

        turns = _dict_to_turns(raw.get("whisper_turns", [])) if raw.get("whisper_turns") else None
        lang = raw.get("whisper_lang", "")
        gaps = _dict_to_gaps(raw.get("gaps", [])) if raw.get("gaps") else None
        scenes = _dict_to_scenes(raw.get("scene_boundaries", [])) if raw.get("scene_boundaries") else None
        return turns, lang, gaps, scenes
    except Exception:
        return None, "", None, None
