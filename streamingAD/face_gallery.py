#!/usr/bin/env python3
"""
face_gallery.py — Shared face gallery loader + character lookup.

Uses PlotTree's full face feature database (81 movies, 1693 faces) as primary source.
Falls back to step04_03 cache_gallery for movies not in PlotTree.

Provides:
  - Gallery lookup by movie name → face embeddings + character metadata
  - Face avatar image lookup by character name
  - Face detection on video clips (insightface detect → gallery match)
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

STREAMING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STREAMING_ROOT.parent
DATA_DIR = STREAMING_ROOT / "face_gallery_data"

FACES_DIR = PROJECT_ROOT / "Step04_RunTest" / "faces" / "movies"
OLD_CACHE_DIR = PROJECT_ROOT / "Step04_RunTest" / "step04_03_face_align" / "cache_gallery"

_PLOTREE_PKL_PATH = DATA_DIR / "plotree_features.pkl"
_PLOTREE_MOVIE_JSON = DATA_DIR / "plotree_movie.json"
_NAME2IMDB_PATH = DATA_DIR / "name2imdbid.json"


def _strip_movie_prefix(name: str) -> str:
    name = str(name)
    name = re.sub(r'^(IMDB|douban)-\d+-', '', name)
    return name.strip().replace(" ", "_").replace("-", "_")


# ── PlotTree database loaders ───────────────────────────────

def _load_plotree_db() -> Tuple[dict, dict, dict]:
    if not hasattr(_load_plotree_db, "_cache"):
        with open(_PLOTREE_PKL_PATH, "rb") as f:
            pkl = pickle.load(f)
        with open(_PLOTREE_MOVIE_JSON, encoding="utf-8") as f:
            movie_json = json.load(f)
        with open(_NAME2IMDB_PATH, encoding="utf-8") as f:
            name2imdb = json.load(f)
        _load_plotree_db._cache = (pkl, movie_json, name2imdb)
    return _load_plotree_db._cache


def _movie_to_imdbid(movie_name: str) -> Optional[str]:
    _, _, name2imdb = _load_plotree_db()
    clean = movie_name.strip()
    for k, v in name2imdb.items():
        if k == clean:
            return v
    search = _strip_movie_prefix(clean).lower()
    for k, v in name2imdb.items():
        if _strip_movie_prefix(k).lower() == search:
            return v
    return None


# ── Public API ──────────────────────────────────────────────

def load_gallery(
    movie_name: str,
) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, str]]]]:
    imdb_id = _movie_to_imdbid(movie_name)
    if imdb_id:
        emb, meta = _load_from_plotree(movie_name, imdb_id)
        if emb is not None:
            return emb, meta

    return _load_from_old_cache(movie_name)


def _load_from_plotree(
    movie_name: str, imdb_id: str,
) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, str]]]]:
    pkl, movie_json, _ = _load_plotree_db()

    if imdb_id not in pkl or imdb_id not in movie_json:
        return None, None

    feats = pkl[imdb_id]["features"]
    pkl_names = pkl[imdb_id]["names"]
    json_chars = movie_json[imdb_id]

    if isinstance(feats, np.ndarray):
        embeddings = feats.astype(np.float32)
    else:
        embeddings = np.asarray(feats.cpu().numpy(), dtype=np.float32)

    if embeddings.ndim != 2:
        return None, None

    face_image_map = _build_face_image_map(movie_name, json_chars)

    people_meta: List[Dict[str, str]] = []
    for i in range(min(embeddings.shape[0], len(json_chars))):
        char_info = json_chars[i]
        role = str(char_info.get("role", "")).strip()
        actor = str(char_info.get("name", "")).strip()
        nm_id = str(char_info.get("id", "")).strip()

        display = role if role else actor
        if not display:
            display = pkl_names[i] if i < len(pkl_names) else f"character_{i+1}"

        source_image = _find_face_image(movie_name, nm_id, role, face_image_map)

        people_meta.append({
            "display_name": display,
            "role_name": role,
            "actor_name": actor,
            "nm_id": nm_id,
            "source_image": source_image,
        })

    return embeddings, people_meta


def _build_face_image_map(
    movie_name: str, json_chars: list,
) -> Dict[str, List[Path]]:
    candidate_dirs = []
    for name in [movie_name, _strip_movie_prefix(movie_name)]:
        d = FACES_DIR / name
        if d.is_dir():
            candidate_dirs.append(d)
    if not candidate_dirs:
        return {}

    image_files: Dict[str, List[Path]] = {}
    for d in candidate_dirs:
        for f in d.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                stem_lower = f.stem.lower()
                image_files.setdefault(stem_lower, []).append(f)
    return image_files


def _find_face_image(
    movie_name: str,
    nm_id: str,
    role: str,
    image_map: Dict[str, List[Path]],
) -> str:
    if nm_id:
        for stem, paths in image_map.items():
            if nm_id.lower() in stem:
                return str(paths[0])
    if role:
        role_key = role.lower().replace(" ", "_").replace("'", "").replace('"', "")
        for stem, paths in image_map.items():
            if role_key in stem:
                return str(paths[0])
        role_first = role.lower().split()[0] if role.lower().split() else ""
        if role_first:
            for stem, paths in image_map.items():
                if role_first in stem:
                    return str(paths[0])
    return ""


def _load_from_old_cache(
    movie_name: str,
) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, str]]]]:
    safe = _strip_movie_prefix(movie_name)
    direct = OLD_CACHE_DIR / f"{safe}.npz"
    if not direct.exists():
        try:
            for f in sorted(OLD_CACHE_DIR.iterdir()):
                if f.suffix == ".npz" and safe.lower() in f.stem.lower():
                    direct = f
                    break
        except OSError:
            return None, None
    if not direct.exists():
        return None, None

    json_path = direct.with_suffix(".json")
    if not json_path.exists():
        return None, None

    try:
        payload = np.load(str(direct))
        emb = np.asarray(payload["embeddings"], dtype=np.float32)
        people_meta: List[Dict[str, str]] = json.loads(
            json_path.read_text(encoding="utf-8")
        )
    except Exception:
        return None, None

    if emb.ndim != 2 or emb.shape[0] != len(people_meta):
        return None, None

    return emb, people_meta


def lookup_face_images(
    character_names: List[str],
    people_meta: List[Dict[str, str]],
    movie_name: str = "",
) -> List[Path]:
    name_to_image: Dict[str, str] = {}

    for p in people_meta:
        source = str(p.get("source_image", "")).strip()
        if not source:
            continue
        for field in ("display_name", "role_name"):
            key = str(p.get(field, "")).strip().lower()
            if key and key not in name_to_image:
                name_to_image[key] = source

    result: List[Path] = []
    seen: set = set()
    for cname in character_names:
        key = str(cname).strip().lower()
        img_path = name_to_image.get(key, "")
        if img_path and img_path not in seen:
            p = Path(img_path)
            if p.is_file():
                result.append(p)
                seen.add(img_path)
            else:
                fallback = _find_image_by_name(Path(img_path).stem, movie_name)
                if fallback and str(fallback) not in seen:
                    result.append(fallback)
                    seen.add(str(fallback))
    return result


def _find_image_by_name(stem: str, movie_name: str) -> Optional[Path]:
    if not movie_name:
        return None
    search_name = Path(movie_name).stem
    candidates = [
        FACES_DIR / search_name / f"{stem}.jpg",
        FACES_DIR / search_name / f"{stem}.png",
    ]
    for c in candidates:
        if c.is_file():
            return c
    try:
        face_dir = FACES_DIR / search_name
        if face_dir.is_dir():
            for f in face_dir.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"} and stem.lower() in f.stem.lower():
                    return f
    except OSError:
        pass
    return None


def detect_faces_in_clip(
    clip_path: Path,
    gallery_embs: np.ndarray,
    people_meta: List[Dict[str, str]],
    threshold: float = 0.35,
    max_frames: int = 8,
) -> Tuple[List[str], List[Path]]:
    import cv2
    from insightface.app import FaceAnalysis

    if not hasattr(detect_faces_in_clip, "_app"):
        try:
            app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception:
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
        detect_faces_in_clip._app = app

    face_app = detect_faces_in_clip._app

    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return [], []

    step = max(1, total_frames // max_frames)
    chars_seen: Dict[str, float] = {}
    char_to_image: Dict[str, str] = {}

    gallery_normed = gallery_embs / (
        np.linalg.norm(gallery_embs, axis=1, keepdims=True) + 1e-12
    )

    for fn in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_app.get(frame)
        for face in faces:
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                continue
            emb_np = np.asarray(emb, dtype=np.float32)
            emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-12)
            sim = emb_np @ gallery_normed.T
            best_idx = int(np.argmax(sim))
            best_score = float(sim[best_idx])
            if best_score < threshold:
                continue

            person = people_meta[best_idx]
            name = str(
                person.get("display_name") or person.get("role_name") or ""
            ).strip()
            if not name:
                continue

            if name not in chars_seen or best_score > chars_seen[name]:
                chars_seen[name] = best_score
                char_to_image[name] = str(person.get("source_image", ""))

    cap.release()

    sorted_chars = sorted(
        chars_seen.keys(), key=lambda n: chars_seen[n], reverse=True
    )

    avatar_paths: List[Path] = []
    seen_imgs: set = set()
    for name in sorted_chars:
        img = char_to_image.get(name, "")
        if img and img not in seen_imgs:
            p = Path(img)
            if p.is_file():
                avatar_paths.append(p)
                seen_imgs.add(img)

    return sorted_chars, avatar_paths
