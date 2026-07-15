#!/usr/bin/env python3
"""
Segment database — load and query per-movie AD segment data.

Uses step04-01 final JSON output.
Provides:
- SegmentDB: query segments by time
- load_segment_db(): load from step04_final_by_movie_new/
- extract_face_data(): parse step04_03 face alignment JSON
- Utility: canonical, to_float, read_json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def canonical(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def read_json(path: Path) -> Any:
    with open(str(path), "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class SegmentDB:
    segments: List[Dict[str, Any]] = field(default_factory=list)
    movie_title: str = ""

    def current_segment(self, time_sec: float) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        best_dist = float("inf")
        for s in self.segments:
            start = to_float(s.get("ad_movie_start_sec"), 0.0)
            end = to_float(s.get("ad_movie_end_sec"), 0.0)
            if start <= time_sec <= end:
                return s
            dist = min(abs(time_sec - start), abs(time_sec - end))
            if dist < best_dist:
                best_dist = dist
                best = s
        return best

    @property
    def total_duration(self) -> float:
        if not self.segments:
            return 0.0
        return max(to_float(s.get("ad_movie_end_sec"), 0.0) for s in self.segments)


def load_segment_db(movie: str, final_by_movie_dir: Path) -> SegmentDB:
    candidates = list(final_by_movie_dir.glob(f"*{movie.replace(' ', '_')}*.json"))
    if not candidates:
        candidates = [p for p in final_by_movie_dir.glob("*.json")
                      if canonical(movie) in canonical(p.stem.replace("_", " "))]
    if not candidates:
        raise FileNotFoundError(f"Segment JSON not found for '{movie}' in {final_by_movie_dir}")
    data = read_json(candidates[0])
    segments = data.get("ad_segments", []) if isinstance(data, dict) else []
    title = (data.get("movie_summary", {}) or {}).get("movie_title", movie) if isinstance(data, dict) else movie
    return SegmentDB(segments=segments, movie_title=str(title))


def scan_available_movies(
    seg_dir: Path,
    clip_dir: Path,
    face_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Path]]:
    clip_names: Dict[str, str] = {}
    for p in clip_dir.iterdir():
        if p.is_dir():
            clip_names[canonical(p.name)] = p.name

    # face_dir is optional: when using PlotTree face gallery (face_gallery.py),
    # per-clip face JSONs are not required for movie discovery.
    face_names: Dict[str, str] = {}
    if face_dir and face_dir.is_dir():
        for p in face_dir.iterdir():
            if p.is_dir():
                face_names[canonical(p.name)] = p.name

    available: Dict[str, Dict[str, Path]] = {}
    for jf in sorted(seg_dir.glob("*.json")):
        key = canonical(jf.stem.replace("_", " "))
        if key not in clip_names:
            continue
        if face_names and key not in face_names:
            continue
        title = jf.stem.replace("_", " ")
        entry: Dict[str, Path] = {
            "seg_json": jf,
            "clip_dir": clip_dir / clip_names[key],
        }
        if face_names and key in face_names:
            entry["face_dir"] = face_dir / face_names[key]
        available[title] = entry
    return available


def extract_face_data(
    face_json_path: Optional[Path],
    max_face_records: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    if face_json_path is None or not face_json_path.is_file():
        return [], []
    payload = read_json(face_json_path)
    detections = payload.get("detections", []) if isinstance(payload, dict) else []
    if not isinstance(detections, list):
        return [], []

    by_person: Dict[str, Dict[str, Any]] = {}
    for det in detections:
        if not isinstance(det, dict):
            continue
        match = det.get("match", {})
        if not isinstance(match, dict) or not bool(match.get("matched", False)):
            continue

        person_id = str(match.get("person_id", "")).strip()
        display_name = str(match.get("display_name", "")).strip()
        role_name = str(match.get("role_name", "")).strip()
        gallery_image = str(match.get("gallery_image", "")).strip()
        similarity = to_float(match.get("similarity"), 0.0)
        clip_face_image = str(det.get("crop_file", "")).strip()
        avatar_image = clip_face_image or gallery_image

        dedupe_key = person_id.lower() or display_name.lower() or role_name.lower()
        record = {
            "person_id": person_id,
            "display_name": display_name,
            "role_name": role_name,
            "similarity": similarity,
            "avatar_image": avatar_image,
            "clip_face_image": clip_face_image,
            "gallery_image": gallery_image,
        }
        cur = by_person.get(dedupe_key)
        if cur is None or similarity > to_float(cur.get("similarity"), 0.0):
            by_person[dedupe_key] = record

    records = sorted(by_person.values(), key=lambda x: to_float(x.get("similarity"), 0.0), reverse=True)
    if max_face_records > 0:
        records = records[:max_face_records]

    avatar_paths: List[Path] = []
    for r in records:
        avatar = str(r.get("avatar_image", "") or r.get("clip_face_image", "") or r.get("gallery_image", ""))
        if avatar:
            p = Path(avatar)
            if p.is_file():
                avatar_paths.append(p)
    return records, avatar_paths
