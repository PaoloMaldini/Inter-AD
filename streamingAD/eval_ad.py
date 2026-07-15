"""
eval_ad.py — Evaluation metrics for Audio Description generation.

Supported metrics:
  CIDEr / SPICE / BertScore  — classic caption metrics (need GT references)
  R@k/N                       — retrieval-based assessment (need clip database)
  CRITIC                      — character co-reference accuracy
  Redundancy (R)              — semantic redundancy vs. context
  Audio Overlap               — physical feasibility (AD duration vs. gap)
  Depth & Density             — controllability & richness

Usage:
    conda run -n videollava python streamingAD/eval_ad.py \
        --input batch_ad_output/IMDB-001-The\ Shawshank\ Redemption_ad_gaps.json \
        --output batch_ad_output/eval_result.json

Reference format (for CIDEr/SPICE/BertScore/R@k):
    {
      "gap_id_1": ["reference caption 1", "reference caption 2", ...],
      ...
    }
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AdEntry:
    gap_id: int
    ad_text: str
    gap_duration_sec: float
    scene_index: str
    location: str
    characters: List[str]
    context_before: List[str]
    context_after: List[str]


@dataclass
class EvalResult:
    metric: str
    score: float
    per_sample: List[float]
    details: Dict[str, Any] = field(default_factory=dict)


def _load_entries(input_path: Path) -> List[AdEntry]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    entries = []
    for e in data.get("ad_entries", []):
        entries.append(AdEntry(
            gap_id=e["gap_id"],
            ad_text=e.get("ad_text", ""),
            gap_duration_sec=float(e.get("gap_duration_sec", 0)),
            scene_index=str(e.get("scene_index", "")),
            location=str(e.get("location", "")),
            characters=[str(c).strip() for c in e.get("characters", []) if str(c).strip()],
            context_before=[str(s) for s in e.get("context_before", [])],
            context_after=[str(s) for s in e.get("context_after", [])],
        ))
    return entries


def _load_references(ref_path: Optional[Path]) -> Optional[Dict[str, List[str]]]:
    if ref_path is None or not ref_path.is_file():
        return None
    data = json.loads(ref_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        refs = {}
        for i, item in enumerate(data):
            refs[str(i + 1)] = [str(item)] if isinstance(item, str) else [str(x) for x in item]
        return refs
    return {str(k): [str(v)] if isinstance(v, str) else [str(x) for x in v]
            for k, v in data.items()}


# ═══════════════════════════════════════════════════════════════════
# 1. Audio Overlap (NarrAD, WACV 2025)
# ═══════════════════════════════════════════════════════════════════

WORDS_PER_MINUTE = 150  # standard AD narration pace


def eval_audio_overlap(entries: List[AdEntry], wpm: int = WORDS_PER_MINUTE) -> EvalResult:
    per_sample: List[float] = []
    violations = 0
    total_overlap_sec = 0.0

    for e in entries:
        word_count = len(e.ad_text.split())
        ad_duration = (word_count / wpm) * 60
        gap = e.gap_duration_sec
        overlap_ratio = min(1.0, gap / max(ad_duration, 0.01))
        per_sample.append(round(overlap_ratio, 4))
        if ad_duration > gap:
            violations += 1
            total_overlap_sec += ad_duration - gap

    return EvalResult(
        metric="Audio Overlap",
        score=round(float(np.mean(per_sample)), 4),
        per_sample=per_sample,
        details={
            "speaking_rate_wpm": wpm,
            "fits_within_gap_ratio": round(float(np.mean(per_sample)), 4),
            "violations": violations,
            "violation_pct": round(violations / max(len(entries), 1) * 100, 1),
            "total_overlap_seconds": round(total_overlap_sec, 1),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# 2. Redundancy / Semantic Redundancy (FocusedAD, arXiv 2025)
# ═══════════════════════════════════════════════════════════════════

def eval_redundancy(entries: List[AdEntry]) -> EvalResult:
    per_sample: List[float] = []

    for e in entries:
        ad_words = set(_tokenize(e.ad_text))
        ctx_words = set()
        for ctx_line in e.context_before + e.context_after:
            ctx_words.update(_tokenize(ctx_line))
        if not ad_words:
            per_sample.append(0.0)
            continue
        overlap = len(ad_words & ctx_words)
        ratio = overlap / len(ad_words)
        per_sample.append(round(ratio, 4))

    return EvalResult(
        metric="Redundancy",
        score=round(float(np.mean(per_sample)), 4),
        per_sample=per_sample,
        details={
            "avg_word_overlap_ratio": round(float(np.mean(per_sample)), 4),
            "high_redundancy_pct": round(
                sum(1 for p in per_sample if p > 0.6) / max(len(per_sample), 1) * 100, 1
            ),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# 3. Depth & Density (CVPR 2024 controllable captioning)
# ═══════════════════════════════════════════════════════════════════

def eval_depth_density(entries: List[AdEntry]) -> EvalResult:
    sent_lens = []
    word_lens = []
    tt_ratios = []
    unique_all = set()

    item_stats = []
    for e in entries:
        words = _tokenize(e.ad_text)
        if not words:
            item_stats.append({"sent_len": 0, "word_len": 0, "ttr": 0})
            continue
        sl = len(words)
        wl = np.mean([len(w) for w in words])
        ttr = len(set(words)) / len(words)
        sent_lens.append(sl)
        word_lens.append(wl)
        tt_ratios.append(ttr)
        unique_all.update(w.lower() for w in words)
        item_stats.append({"sent_len": sl, "word_len": round(float(wl), 2), "ttr": round(ttr, 4)})

    return EvalResult(
        metric="Depth & Density",
        score=round(float(np.mean(tt_ratios)) if tt_ratios else 0, 4),
        per_sample=tt_ratios,
        details={
            "avg_sentence_length_words": round(float(np.mean(sent_lens)) if sent_lens else 0, 1),
            "avg_word_length_chars": round(float(np.mean(word_lens)) if word_lens else 0, 2),
            "avg_type_token_ratio": round(float(np.mean(tt_ratios)) if tt_ratios else 0, 4),
            "total_unique_words": len(unique_all),
            "total_word_count": sum(sent_lens),
            "item_stats": item_stats[:10],
        },
    )


# ═══════════════════════════════════════════════════════════════════
# 4. CRITIC (AutoAD III, CVPR 2024)
# ═══════════════════════════════════════════════════════════════════

def _get_ner_model():
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import logging
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=device)


def eval_critic(entries: List[AdEntry], use_ner: bool = True) -> EvalResult:
    per_sample: List[float] = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    eligible = 0

    ner = _get_ner_model() if use_ner else None

    _PLACEHOLDER_CHARS = {"?", "unknown", "", "none", "n/a", "na"}

    for e in entries:
        expected = set(c.lower() for c in e.characters if c.lower().strip() not in _PLACEHOLDER_CHARS)
        if not expected:
            per_sample.append(-1.0)
            continue
        eligible += 1

        detected_text = e.ad_text
        if ner:
            ner_results = ner(detected_text)
            detected = set()
            for entity in ner_results:
                if entity.get("entity_group") == "PER":
                    detected.add(entity["word"].lower())
        else:
            detected = set(w.lower() for w in _tokenize(e.ad_text) if w[0].isupper())

        if not detected:
            per_sample.append(0.0)
            fn_total += len(expected)
            continue

        tp = sum(1 for d in detected for c in expected if c in d or d in c)
        fp = sum(1 for d in detected for c in expected if c not in d and d not in c)
        fn = sum(1 for c in expected for d in detected if c not in d and d not in c)

        tp = min(tp, len(expected))
        fn = max(0, len(expected) - tp)
        fp = max(0, len(detected) - tp)

        tp_total += tp
        fp_total += fp
        fn_total += fn

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.01)
        per_sample.append(round(f1, 4))

    valid = [p for p in per_sample if p >= 0]
    avg_f1 = float(np.mean(valid)) if valid else None

    return EvalResult(
        metric="CRITIC",
        score=round(avg_f1, 4) if avg_f1 is not None else None,
        per_sample=per_sample,
        details={
            "eligible_entries": eligible,
            "avg_character_f1": round(avg_f1, 4) if avg_f1 is not None else None,
            "total_tp": tp_total,
            "total_fp": fp_total,
            "total_fn": fn_total,
            "macro_precision": round(tp_total / max(tp_total + fp_total, 1), 4),
            "macro_recall": round(tp_total / max(tp_total + fn_total, 1), 4),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# 5. BertScore (ICLR 2020)
# ═══════════════════════════════════════════════════════════════════

def _get_bert_model():
    from transformers import AutoModel, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model


def _bert_embed(texts: List[str], tokenizer, model) -> np.ndarray:
    import torch
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        output = model(**{k: v for k, v in encoded.items() if k != "token_type_ids"})
    cls_emb = output.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(cls_emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return cls_emb / norms


def eval_bertscore(
    entries: List[AdEntry],
    ref_path: Optional[Path] = None,
    use_context_as_ref: bool = True,
) -> EvalResult:
    per_sample: List[float] = []
    refs = _load_references(ref_path)

    if refs is not None:
        tokenizer, model = _get_bert_model()
        for e in entries:
            gt_list = refs.get(str(e.gap_id), [])
            if not gt_list:
                per_sample.append(-1.0)
                continue
            cand_emb = _bert_embed([e.ad_text], tokenizer, model)
            ref_emb = _bert_embed(gt_list, tokenizer, model)
            sims = np.dot(cand_emb, ref_emb.T).flatten()
            per_sample.append(round(float(np.mean(sims)), 4) if len(sims) else 0.0)
    elif use_context_as_ref:
        tokenizer, model = _get_bert_model()
        for e in entries:
            ctx = " ".join(e.context_before[-3:] + e.context_after[:3])
            if not ctx.strip():
                per_sample.append(-1.0)
                continue
            cand_emb = _bert_embed([e.ad_text], tokenizer, model)
            ref_emb = _bert_embed([ctx], tokenizer, model)
            sim = float(np.dot(cand_emb, ref_emb.T)[0, 0])
            per_sample.append(round(sim, 4))
    else:
        return EvalResult(
            metric="BertScore",
            score=0,
            per_sample=[],
            details={"status": "skipped", "reason": "No references or context available"},
        )

    valid = [p for p in per_sample if p >= 0]
    return EvalResult(
        metric="BertScore",
        score=round(float(np.mean(valid)) if valid else 0, 4),
        per_sample=per_sample,
        details={
            "samples_evaluated": len(valid),
            "mode": "reference" if refs else "context_as_pseudo_ref",
            "avg_similarity": round(float(np.mean(valid)) if valid else 0, 4),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# 6. R@k/N (AutoAD II, ICCV 2023)
# ═══════════════════════════════════════════════════════════════════

def eval_retrieval_r_at_k(
    entries: List[AdEntry],
    ref_path: Optional[Path] = None,
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> EvalResult:
    refs = _load_references(ref_path)
    if refs is None:
        return EvalResult(
            metric="R@k/N",
            score=0,
            per_sample=[],
            details={"status": "skipped", "reason": "No reference GT provided"},
        )

    tokenizer, model = _get_bert_model()

    all_cand_embs = []
    all_ref_embs_list = []
    valid_indices = []

    for i, e in enumerate(entries):
        gt_list = refs.get(str(e.gap_id), [])
        if not gt_list:
            continue
        valid_indices.append(i)
        all_cand_embs.append(_bert_embed([e.ad_text], tokenizer, model)[0])
        all_ref_embs_list.append(_bert_embed(gt_list, tokenizer, model))

    if not all_cand_embs:
        return EvalResult(
            metric="R@k/N",
            score=0,
            per_sample=[],
            details={"status": "skipped", "reason": "No valid entries with GT"},
        )

    cand_mat = np.stack(all_cand_embs, axis=0)
    results = {}

    for k in k_values:
        hits = 0
        for i in range(len(cand_mat)):
            scores = np.dot(cand_mat[i:i+1], cand_mat.T).flatten()
            top_k = np.argsort(-scores)[:min(k + 1, len(scores))]
            if i in top_k:
                hits += 1
        results[f"R@{k}"] = round(hits / len(cand_mat), 4)

    return EvalResult(
        metric="R@k/N",
        score=results.get("R@1", 0),
        per_sample=[],
        details={"status": "computed", **results, "num_samples": len(cand_mat)},
    )


# ═══════════════════════════════════════════════════════════════════
# 7. CIDEr (CVPR 2015) — simplified TF-IDF n-gram consensus
# ═══════════════════════════════════════════════════════════════════

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def eval_cider(
    entries: List[AdEntry],
    ref_path: Optional[Path] = None,
    ngram_max: int = 4,
) -> EvalResult:
    refs = _load_references(ref_path)
    if refs is None:
        return EvalResult(
            metric="CIDEr",
            score=0,
            per_sample=[],
            details={"status": "skipped", "reason": "No reference GT provided"},
        )

    all_docs = []
    for e in entries:
        gt_list = refs.get(str(e.gap_id), [e.ad_text])
        all_docs.extend(gt_list)

    all_tokens = [_tokenize(doc) for doc in all_docs]
    df: Dict[int, Dict[Tuple, int]] = {}
    for n in range(1, ngram_max + 1):
        df[n] = Counter()
        for tokens in all_tokens:
            df[n].update(_ngrams(tokens, n))

    num_docs = len(all_docs) or 1
    idf: Dict[int, Dict[Tuple, float]] = {}
    for n in range(1, ngram_max + 1):
        idf[n] = {ng: np.log(max(num_docs / max(df[n][ng], 1), 1.0)) + 1.0
                  for ng in df[n]}

    per_sample: List[float] = []
    for e in entries:
        gt_list = refs.get(str(e.gap_id), [])
        if not gt_list:
            per_sample.append(-1.0)
            continue
        cand_tokens = _tokenize(e.ad_text)
        scores = []
        for gt in gt_list:
            gt_tokens = _tokenize(gt)
            total = 0.0
            for n in range(1, ngram_max + 1):
                cand_ng = _ngrams(cand_tokens, n)
                gt_ng = _ngrams(gt_tokens, n)
                if not gt_ng:
                    continue
                weighted = 0.0
                for ng, count in cand_ng.items():
                    weight = idf.get(n, {}).get(ng, 1.0) if num_docs > 1 else 1.0
                    weighted += min(count, gt_ng.get(ng, 0)) * weight
                denom = sum(gt_ng.values()) or 1
                total += weighted / denom
            scores.append(total / ngram_max)
        per_sample.append(round(float(np.mean(scores)) if scores else 0, 4))

    valid = [p for p in per_sample if p >= 0]
    return EvalResult(
        metric="CIDEr",
        score=round(float(np.mean(valid)) if valid else 0, 4),
        per_sample=per_sample,
        details={
            "samples_evaluated": len(valid),
            "ngram_max": ngram_max,
            "total_documents": num_docs,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# 8. SPICE (ECCV 2016) — simplified semantic proposition scoring
# ═══════════════════════════════════════════════════════════════════

def _extract_semantic_tuples(text: str) -> set:
    tokens = _tokenize(text)
    tuples: set = set()
    for i in range(len(tokens) - 1):
        tuples.add(("bigram", tokens[i], tokens[i+1]))
        tuples.add(("unigram", tokens[i],))
    if tokens:
        tuples.add(("unigram", tokens[-1],))
    for i in range(len(tokens) - 2):
        tuples.add(("trigram", tokens[i], tokens[i+1], tokens[i+2]))
    return tuples


def eval_spice(
    entries: List[AdEntry],
    ref_path: Optional[Path] = None,
) -> EvalResult:
    refs = _load_references(ref_path)
    if refs is None:
        return EvalResult(
            metric="SPICE",
            score=0,
            per_sample=[],
            details={"status": "skipped", "reason": "No reference GT provided"},
        )

    per_sample: List[float] = []
    for e in entries:
        gt_list = refs.get(str(e.gap_id), [])
        if not gt_list:
            per_sample.append(-1.0)
            continue
        cand_tuples = _extract_semantic_tuples(e.ad_text)
        scores = []
        for gt in gt_list:
            gt_tuples = _extract_semantic_tuples(gt)
            if not gt_tuples:
                continue
            tp = len(cand_tuples & gt_tuples)
            fp = len(cand_tuples - gt_tuples)
            fn = len(gt_tuples - cand_tuples)
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 0.01)
            scores.append(f1)
        per_sample.append(round(float(np.mean(scores)) if scores else 0, 4))

    valid = [p for p in per_sample if p >= 0]
    return EvalResult(
        metric="SPICE",
        score=round(float(np.mean(valid)) if valid else 0, 4),
        per_sample=per_sample,
        details={"samples_evaluated": len(valid)},
    )


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _format_result(r: EvalResult) -> str:
    lines = [f"\n{'─' * 60}", f"  {r.metric}: {r.score}"]
    for k, v in r.details.items():
        if k in ("item_stats",):
            continue
        lines.append(f"    {k}: {v}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def run_eval(
    input_path: Path,
    output_path: Path,
    ref_path: Optional[Path] = None,
    skip_heavy: bool = False,
) -> Dict[str, Any]:
    entries = _load_entries(input_path)
    print(f"Loaded {len(entries)} AD entries from {input_path}")
    if ref_path:
        refs = _load_references(ref_path)
        has_gt = refs is not None and len(refs) > 0
        print(f"References: {'loaded' if has_gt else 'none found'}")
    else:
        has_gt = False
        print("No reference file provided. GT-dependent metrics (CIDEr/SPICE/R@k) will skip.")

    results: Dict[str, Any] = {
        "source": str(input_path),
        "num_entries": len(entries),
    }

    # Always-run metrics
    print("\nComputing Audio Overlap ...")
    r = eval_audio_overlap(entries)
    results[r.metric] = {"score": r.score, "details": r.details}
    print(_format_result(r))

    print("\nComputing Redundancy ...")
    r = eval_redundancy(entries)
    results[r.metric] = {"score": r.score, "details": r.details}
    print(_format_result(r))

    print("\nComputing Depth & Density ...")
    r = eval_depth_density(entries)
    results[r.metric] = {"score": r.score, "details": r.details}
    print(_format_result(r))

    if not skip_heavy:
        print("\nComputing CRITIC (character NER) ...")
        r = eval_critic(entries)
        results[r.metric] = {"score": r.score, "details": r.details}
        print(_format_result(r))

        print("\nComputing BertScore ...")
        r = eval_bertscore(entries, ref_path=ref_path if has_gt else None, use_context_as_ref=True)
        results[r.metric] = {"score": r.score, "details": r.details}
        print(_format_result(r))

        print("\nComputing CIDEr ...")
        r = eval_cider(entries, ref_path=ref_path if has_gt else None)
        results[r.metric] = {"score": r.score, "details": r.details}
        print(_format_result(r))

        print("\nComputing SPICE ...")
        r = eval_spice(entries, ref_path=ref_path if has_gt else None)
        results[r.metric] = {"score": r.score, "details": r.details}
        print(_format_result(r))

        print("\nComputing R@k/N ...")
        r = eval_retrieval_r_at_k(entries, ref_path=ref_path if has_gt else None)
        results[r.metric] = {"score": r.score, "details": r.details}
        print(_format_result(r))
    else:
        for m in ("CRITIC", "BertScore", "CIDEr", "SPICE", "R@k/N"):
            results[m] = {"score": 0, "details": {"status": "skipped", "reason": "--skip-heavy"}}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'═' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"{'═' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate AD generation results")
    parser.add_argument("--input", required=True, help="Path to ad_gaps.json file")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--ref", default=None, help="Path to reference GT JSON (optional)")
    parser.add_argument("--skip-heavy", action="store_true", help="Skip model-heavy metrics (CRITIC/BertScore)")
    args = parser.parse_args()

    run_eval(
        input_path=Path(args.input),
        output_path=Path(args.output),
        ref_path=Path(args.ref) if args.ref else None,
        skip_heavy=args.skip_heavy,
    )


if __name__ == "__main__":
    main()
