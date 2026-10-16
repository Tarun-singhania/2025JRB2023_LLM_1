#!/usr/bin/env python3
"""
Part B sanitizer evaluation for Trustworthy LLM Assignment 1.

Required CLI supported:
python src/sanitize_eval.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path adapters/best_sanitizer \
  --test_file data/partB_test.jsonl \
  --system {finetuned,regex,presidio,zeroshot} \
  --output_predictions results/partB_predictions.jsonl \
  --report_file results/partB_metrics.json

Design notes
------------
* Finetuned / zeroshot systems emit masked text, then recover source spans by
  deterministic literal-skeleton alignment using difflib.SequenceMatcher.
* Regex and Presidio know their detector spans directly; these are used as the
  predicted spans, and their masked text is created from those spans.
* Exact and IoU>=0.5 matching are one-to-one. IoU pairs are consumed in
  descending IoU order ("maximum-IoU" greedy matching).
* Leak checking is case-insensitive and whitespace-normalized.
* Over-masking is the fraction of gold non-PII source characters that do not
  participate in an exact SequenceMatcher equality block with the output.
"""

import argparse
import difflib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Fixed assignment label vocabulary.
# Keeping this fixed avoids using test-set labels as an oracle.
# ---------------------------------------------------------------------------
DATASET_LABELS = {
    "BOD", "BUILDING", "CARDISSUER", "CITY", "COUNTRY", "DATE",
    "DRIVERLICENSE", "EMAIL", "GEOCOORD", "GIVENNAME1", "GIVENNAME2",
    "IDCARD", "IP", "LASTNAME1", "LASTNAME2", "LASTNAME3", "PASS",
    "PASSPORT", "POSTCODE", "SECADDRESS", "SEX", "SOCIALNUMBER", "STATE",
    "STREET", "TEL", "TIME", "TITLE", "USERNAME",
}

SYSTEM_PROMPT = (
    "You are a privacy-preserving text sanitizer. "
    "Replace personally identifiable information with the appropriate "
    "placeholder labels while preserving all non-sensitive text. "
    "Return only the sanitized text."
)
USER_PREFIX = "Sanitize the following text:\n"

PLACEHOLDER_RE = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")
BRACKET_TOKEN_RE = re.compile(r"\[[^\]\n]{1,80}\]")
WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# I/O + reproducibility
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--test_file", required=True)
    p.add_argument(
        "--system",
        required=True,
        choices=["finetuned", "regex", "presidio", "zeroshot"],
    )
    p.add_argument("--output_predictions", required=True)
    p.add_argument("--report_file", required=True)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=260)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug only. 0 means evaluate the full supplied file.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Used by finetuned/zeroshot systems.",
    )
    p.add_argument(
        "--presidio_spacy_model",
        type=str,
        default="en_core_web_sm",
    )
    args = p.parse_args()

    if args.batch_size <= 0:
        p.error("--batch_size must be > 0")
    if args.max_new_tokens <= 0:
        p.error("--max_new_tokens must be > 0")
    if args.limit < 0:
        p.error("--limit must be >= 0")
    return args


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_label(label: str) -> str:
    return str(label).strip().upper()


def normalize_ws_case(text: str) -> str:
    return WS_RE.sub(" ", str(text)).strip().casefold()


def gold_spans_from_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "start": int(ent["start"]),
            "end": int(ent["end"]),
            "label": normalize_label(ent["label"]),
        }
        for ent in row.get("privacy_mask", [])
    ]


# ---------------------------------------------------------------------------
# Span recovery for generated masked text
# ---------------------------------------------------------------------------
def parse_prediction_skeleton(
    source_text: str,
    predicted_text: str,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
    """Remove newly introduced placeholder-like tokens and keep source literals.

    Known assignment labels are always treated as placeholders.
    An unknown ALL-CAPS bracket token is treated as a prediction only when the
    exact token was not already present literally in the source. This prevents
    legitimate source strings such as [RESET_LINK] from becoming false entities,
    while still counting hallucinated unknown labels as false positives.
    """
    skeleton_parts: List[str] = []
    placeholders: List[Dict[str, Any]] = []
    cursor = 0
    skel_pos = 0
    unknown_new = 0

    for m in PLACEHOLDER_RE.finditer(predicted_text):
        token = m.group(0)
        label = normalize_label(m.group(1))

        treat_as_placeholder = (
            label in DATASET_LABELS or token not in source_text
        )
        if not treat_as_placeholder:
            continue

        literal = predicted_text[cursor:m.start()]
        skeleton_parts.append(literal)
        skel_pos += len(literal)

        if label not in DATASET_LABELS:
            unknown_new += 1

        placeholders.append(
            {
                "pos": skel_pos,
                "label": label,
                "token": token,
            }
        )
        cursor = m.end()

    tail = predicted_text[cursor:]
    skeleton_parts.append(tail)

    return (
        "".join(skeleton_parts),
        placeholders,
        {"unknown_new_placeholders": unknown_new},
    )


def _split_source_span(
    start: int,
    end: int,
    n: int,
    source: str,
) -> List[Tuple[int, int]]:
    """Fallback for multiple adjacent placeholders mapping to one source region."""
    if n <= 1:
        return [(start, end)]
    if end <= start:
        return [(start, start)] * n

    segment = source[start:end]
    tokens = list(re.finditer(r"\S+", segment))

    if len(tokens) >= n:
        pieces: List[Tuple[int, int]] = []
        boundaries = np.linspace(0, len(tokens), n + 1, dtype=int)
        for k in range(n):
            lo, hi = int(boundaries[k]), int(boundaries[k + 1])
            if hi <= lo:
                hi = min(lo + 1, len(tokens))
            s = start + tokens[lo].start()
            e = start + tokens[hi - 1].end()
            pieces.append((s, e))
        return pieces

    cuts = [start + round((end - start) * k / n) for k in range(n + 1)]
    return [(cuts[k], cuts[k + 1]) for k in range(n)]


def recover_predicted_spans(
    source_text: str,
    predicted_text: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Recover (start,end,label) triples from model-produced masked text.

    Alignment rules:
      1) Parse placeholder-like tokens from generated text.
      2) Delete those placeholders to form a literal skeleton.
      3) Character-align source -> skeleton with SequenceMatcher(autojunk=False).
      4) Each placeholder maps to the source interval between the nearest equal
         literal block on its left and nearest equal literal block on its right.
      5) If adjacent placeholders collapse to the same unresolved source region,
         split that region monotonically over whitespace tokens; fall back to
         an even character split.

    Failure modes:
      * large rewrites of non-PII text,
      * reordered text,
      * repeated/ambiguous anchors,
      * multiple adjacent placeholders with no separating literal anchor,
      * malformed placeholders.
    """
    skeleton, placeholders, diag = parse_prediction_skeleton(
        source_text, predicted_text
    )

    diag = dict(diag)
    diag.update(
        {
            "n_placeholders": len(placeholders),
            "adjacent_group_splits": 0,
            "zero_length_dropped": 0,
        }
    )

    if not placeholders:
        return [], diag

    sm = difflib.SequenceMatcher(None, source_text, skeleton, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]

    raw: List[Dict[str, Any]] = []
    for ph in placeholders:
        p = int(ph["pos"])

        prev_blocks = [b for b in blocks if b.b + b.size <= p]
        next_blocks = [b for b in blocks if b.b >= p]

        if prev_blocks:
            prev = max(prev_blocks, key=lambda b: b.b + b.size)
            start = prev.a + prev.size
        else:
            start = 0

        if next_blocks:
            nxt = min(next_blocks, key=lambda b: b.b)
            end = nxt.a
        else:
            end = len(source_text)

        start = max(0, min(int(start), len(source_text)))
        end = max(start, min(int(end), len(source_text)))

        raw.append(
            {
                "start": start,
                "end": end,
                "label": ph["label"],
            }
        )

    grouped: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, span in enumerate(raw):
        grouped[(span["start"], span["end"])].append(idx)

    for (start, end), idxs in grouped.items():
        if len(idxs) <= 1:
            continue
        diag["adjacent_group_splits"] += 1
        pieces = _split_source_span(start, end, len(idxs), source_text)
        for idx, (s, e) in zip(idxs, pieces):
            raw[idx]["start"] = int(s)
            raw[idx]["end"] = int(e)

    clean: List[Dict[str, Any]] = []
    for x in raw:
        if int(x["end"]) <= int(x["start"]):
            diag["zero_length_dropped"] += 1
            continue
        clean.append(
            {
                "start": int(x["start"]),
                "end": int(x["end"]),
                "label": normalize_label(x["label"]),
            }
        )
    return clean, diag


# ---------------------------------------------------------------------------
# Entity matching + metrics
# ---------------------------------------------------------------------------
def span_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    inter = max(
        0,
        min(int(a["end"]), int(b["end"]))
        - max(int(a["start"]), int(b["start"])),
    )
    union = max(int(a["end"]), int(b["end"])) - min(
        int(a["start"]), int(b["start"])
    )
    return float(inter / union) if union > 0 else 0.0


def match_entities(
    pred: Sequence[Dict[str, Any]],
    gold: Sequence[Dict[str, Any]],
    criterion: str,
) -> List[Tuple[int, int]]:
    """Deterministic one-to-one matching.

    exact:
      same label + identical [start,end)

    iou50:
      same label + IoU>=0.5; candidate pairs are greedily consumed from the
      largest IoU downward, so each predicted/gold entity participates once.
    """
    if criterion not in {"exact", "iou50"}:
        raise ValueError(f"Unknown criterion: {criterion}")

    candidates: List[Tuple[float, int, int]] = []

    for pi, p in enumerate(pred):
        for gi, g in enumerate(gold):
            if normalize_label(p["label"]) != normalize_label(g["label"]):
                continue

            if criterion == "exact":
                valid = (
                    int(p["start"]) == int(g["start"])
                    and int(p["end"]) == int(g["end"])
                )
                score = 1.0 if valid else 0.0
            else:
                score = span_iou(p, g)
                valid = score >= 0.5

            if valid:
                candidates.append((score, pi, gi))

    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matched: List[Tuple[int, int]] = []

    for _score, pi, gi in candidates:
        if pi in used_pred or gi in used_gold:
            continue
        used_pred.add(pi)
        used_gold.add(gi)
        matched.append((pi, gi))

    return matched


def _safe_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return float(p), float(r), float(f1)


def over_mask_counts(
    source_text: str,
    predicted_text: str,
    gold_spans: Sequence[Dict[str, Any]],
) -> Tuple[int, int]:
    """(changed non-PII chars, total non-PII chars)."""
    n = len(source_text)
    gold_mask = np.zeros(n, dtype=bool)

    for span in gold_spans:
        s = max(0, min(int(span["start"]), n))
        e = max(s, min(int(span["end"]), n))
        gold_mask[s:e] = True

    preserved = np.zeros(n, dtype=bool)
    sm = difflib.SequenceMatcher(
        None, source_text, predicted_text, autojunk=False
    )
    for block in sm.get_matching_blocks():
        if block.size > 0:
            preserved[block.a:block.a + block.size] = True

    non_pii = ~gold_mask
    total = int(non_pii.sum())
    changed = int((non_pii & ~preserved).sum())
    return changed, total


def leaked_values_for_row(
    row: Dict[str, Any],
    predicted_text: str,
) -> List[str]:
    norm_pred = normalize_ws_case(predicted_text)
    leaked: List[str] = []

    for ent in row.get("privacy_mask", []):
        value = str(ent.get("value", ""))
        norm_value = normalize_ws_case(value)
        if norm_value and norm_value in norm_pred:
            leaked.append(value)

    return leaked


def is_well_formed_output(source_text: str, predicted_text: str) -> bool:
    """Check placeholder syntax while allowing surrounding source structure.

    Valid placeholders are exact ``[LABEL]`` tokens anywhere in the output,
    even when they are nested inside source punctuation/arrays, e.g.:
        [[EMAIL]]
        ["[EMAIL]"]
        [[BUILDING], [STREET], [CITY]]

    A label-looking construction such as ``['SEX']`` is *not* a valid
    sanitizer placeholder: the required placeholder token ``[SEX]`` is absent.

    The check is intentionally syntax-only. Correctness of the label and span
    is handled by the entity metrics.
    """
    if not predicted_text.strip():
        return False

    # Find occurrences of the known assignment labels.  Ordinary occurrences
    # outside bracket-like placeholder context are ignored; occurrences that
    # look like attempted placeholders must be exact [LABEL] tokens.
    label_pattern = re.compile(
        r"\b(" + "|".join(
            sorted((re.escape(x) for x in DATASET_LABELS), key=len, reverse=True)
        ) + r")\b"
    )

    for m in label_pattern.finditer(predicted_text):
        s, e = m.span()
        label = m.group(1)

        # Correct placeholder syntax.  This also works inside [[LABEL]],
        # ["[LABEL]"], and other source-preserving wrappers.
        if (
            s > 0
            and e < len(predicted_text)
            and predicted_text[s - 1] == "["
            and predicted_text[e] == "]"
        ):
            continue

        left = predicted_text[max(0, s - 4):s]
        right = predicted_text[e:min(len(predicted_text), e + 4)]

        # If this label is sitting in nearby square-bracket syntax, it is an
        # attempted/malformed placeholder, e.g. ['SEX'], [EMAIL, EMAIL].
        looks_placeholder_like = ("[" in left) or ("]" in right)

        if looks_placeholder_like:
            # Do not penalize an identical literal bracket fragment that was
            # already present in the source text.
            local_start = predicted_text.rfind("[", max(0, s - 4), s)
            local_end = predicted_text.find("]", e, min(len(predicted_text), e + 4))
            if local_start != -1 and local_end != -1:
                fragment = predicted_text[local_start:local_end + 1]
                if fragment in source_text:
                    continue

            return False

    return True


def compute_metrics(
    rows: Sequence[Dict[str, Any]],
    evaluated: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(rows) != len(evaluated):
        raise ValueError("rows/evaluated length mismatch")

    criteria = ("exact", "iou50")
    aggregate = {
        c: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "per_class": defaultdict(
                lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
            ),
        }
        for c in criteria
    }

    leak_num = 0
    leak_den = 0
    over_changed = 0
    over_total = 0
    well_formed = 0

    error_taxonomy = Counter()
    alignment_diagnostics = Counter()

    for row, out in zip(rows, evaluated):
        pred_text = out["predicted_text"]
        pred_spans = out["predicted_spans"]
        gold_spans = gold_spans_from_row(row)

        for key, value in out.get("alignment_diagnostics", {}).items():
            alignment_diagnostics[key] += int(value)

        if is_well_formed_output(row["source_text"], pred_text):
            well_formed += 1
        else:
            error_taxonomy["malformed_output"] += 1

        gold_support = Counter(normalize_label(g["label"]) for g in gold_spans)

        matched_by_criterion = {}

        for criterion in criteria:
            pairs = match_entities(pred_spans, gold_spans, criterion)
            matched_by_criterion[criterion] = pairs

            matched_pred = {pi for pi, _ in pairs}
            matched_gold = {gi for _, gi in pairs}

            agg = aggregate[criterion]
            agg["tp"] += len(pairs)
            agg["fp"] += len(pred_spans) - len(pairs)
            agg["fn"] += len(gold_spans) - len(pairs)

            for label, support in gold_support.items():
                agg["per_class"][label]["support"] += int(support)

            for _pi, gi in pairs:
                label = normalize_label(gold_spans[gi]["label"])
                agg["per_class"][label]["tp"] += 1

            for pi, p in enumerate(pred_spans):
                if pi not in matched_pred:
                    label = normalize_label(p["label"])
                    agg["per_class"][label]["fp"] += 1

            for gi, g in enumerate(gold_spans):
                if gi not in matched_gold:
                    label = normalize_label(g["label"])
                    agg["per_class"][label]["fn"] += 1

        # Error taxonomy based mainly on IoU50.
        iou_pairs = matched_by_criterion["iou50"]
        exact_pairs = matched_by_criterion["exact"]
        iou_pred = {pi for pi, _ in iou_pairs}
        iou_gold = {gi for _, gi in iou_pairs}
        exact_pair_set = set(exact_pairs)

        # Boundary mismatch among successful IoU50 matches.
        for pair in iou_pairs:
            if pair not in exact_pair_set:
                error_taxonomy["boundary_error_but_iou50_match"] += 1

        # Misses: inspect whether an unmatched gold has overlap with any prediction.
        for gi, g in enumerate(gold_spans):
            if gi in iou_gold:
                continue

            overlapping = []
            for pi, p in enumerate(pred_spans):
                iou = span_iou(p, g)
                if iou > 0:
                    overlapping.append((iou, pi, p))

            if not overlapping:
                error_taxonomy["missed_entity_no_overlap"] += 1
            else:
                overlapping.sort(key=lambda x: -x[0])
                best_iou, _pi, best_p = overlapping[0]
                if normalize_label(best_p["label"]) != normalize_label(g["label"]):
                    error_taxonomy["wrong_label"] += 1
                elif best_iou < 0.5:
                    error_taxonomy["boundary_error_iou_below_0.5"] += 1
                else:
                    # Usually this means one-to-one competition.
                    error_taxonomy["one_to_one_conflict"] += 1

        for pi, _p in enumerate(pred_spans):
            if pi not in iou_pred:
                error_taxonomy["unmatched_predicted_entity"] += 1

        leaked = leaked_values_for_row(row, pred_text)
        leak_num += len(leaked)
        leak_den += sum(
            1
            for ent in row.get("privacy_mask", [])
            if normalize_ws_case(str(ent.get("value", "")))
        )
        error_taxonomy["leaked_gold_entity"] += len(leaked)

        changed, total = over_mask_counts(
            row["source_text"], pred_text, gold_spans
        )
        over_changed += changed
        over_total += total
        if changed > 0:
            error_taxonomy["examples_with_overmasking"] += 1

    reports: Dict[str, Any] = {}
    per_class_all: Dict[str, Dict[str, Any]] = {}

    for criterion in criteria:
        agg = aggregate[criterion]

        micro_p, micro_r, micro_f1 = _safe_prf(
            int(agg["tp"]), int(agg["fp"]), int(agg["fn"])
        )

        class_report: Dict[str, Dict[str, Any]] = {}
        macro_f1_values: List[float] = []

        labels = sorted(
            set(DATASET_LABELS)
            | set(agg["per_class"].keys())
        )

        for label in labels:
            stats = agg["per_class"][label]
            p, r, f1 = _safe_prf(
                int(stats["tp"]), int(stats["fp"]), int(stats["fn"])
            )

            class_report[label] = {
                "p": p,
                "r": r,
                "f1": f1,
                "support": int(stats["support"]),
                "tp": int(stats["tp"]),
                "fp": int(stats["fp"]),
                "fn": int(stats["fn"]),
            }

            # Macro over PII classes actually supported in this test set.
            if int(stats["support"]) > 0:
                macro_f1_values.append(f1)

        reports[criterion] = {
            "micro_p": micro_p,
            "micro_r": micro_r,
            "micro_f1": micro_f1,
            "macro_f1": (
                float(np.mean(macro_f1_values))
                if macro_f1_values
                else 0.0
            ),
            "tp": int(agg["tp"]),
            "fp": int(agg["fp"]),
            "fn": int(agg["fn"]),
        }

        per_class_all[criterion] = class_report

    # Ten most frequent gold classes.
    support_counts = Counter()
    for row in rows:
        for ent in row.get("privacy_mask", []):
            support_counts[normalize_label(ent["label"])] += 1

    top10_labels = [
        label for label, _count in support_counts.most_common(10)
    ]

    top10 = {}
    for label in top10_labels:
        iou = per_class_all["iou50"][label]
        exact = per_class_all["exact"][label]
        top10[label] = {
            # Direct p/r/f1 are IoU>=0.5 values for compatibility with the
            # assignment's example JSON.
            "p": iou["p"],
            "r": iou["r"],
            "f1": iou["f1"],
            "support": iou["support"],
            "exact": exact,
            "overlap_iou50": iou,
        }

    return {
        "exact": reports["exact"],
        "overlap_iou50": reports["iou50"],
        "per_class": top10,
        "per_class_exact_all": per_class_all["exact"],
        "per_class_iou50_all": per_class_all["iou50"],
        "leak_rate": float(leak_num / leak_den) if leak_den else 0.0,
        "over_masking_rate": (
            float(over_changed / over_total)
            if over_total else 0.0
        ),
        "well_formed_rate": (
            float(well_formed / len(rows))
            if rows else 0.0
        ),
        "n_gold_values": int(leak_den),
        "n_leaked_values": int(leak_num),
        "n_non_pii_chars": int(over_total),
        "n_overmasked_non_pii_chars": int(over_changed),
        "error_taxonomy": dict(error_taxonomy),
        "alignment_diagnostics": dict(alignment_diagnostics),
    }


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------
def build_messages(source_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PREFIX + source_text},
    ]


def load_generation_model(
    base_model: str,
    adapter_path: str,
    system: str,
    device: str,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
    )

    if system == "finetuned":
        from peft import PeftModel

        adapter_cfg = Path(adapter_path) / "adapter_config.json"
        adapter_weights = Path(adapter_path) / "adapter_model.safetensors"
        if not adapter_cfg.exists():
            raise FileNotFoundError(f"Missing {adapter_cfg}")
        if not adapter_weights.exists():
            raise FileNotFoundError(f"Missing {adapter_weights}")

        model = PeftModel.from_pretrained(model, adapter_path)

    model = model.to(device)
    model.eval()
    model.config.use_cache = True

    # Sampling-only defaults are irrelevant under do_sample=False.
    # Clear them only to suppress harmless Transformers warnings.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    return tokenizer, model


def generate_masked_texts(
    rows: Sequence[Dict[str, Any]],
    base_model: str,
    adapter_path: str,
    system: str,
    batch_size: int,
    max_new_tokens: int,
    device: str,
) -> Tuple[List[str], List[float]]:
    import torch
    from tqdm.auto import tqdm

    tokenizer, model = load_generation_model(
        base_model=base_model,
        adapter_path=adapter_path,
        system=system,
        device=device,
    )

    predictions: List[str] = []
    latencies_ms: List[float] = []

    for start in tqdm(
        range(0, len(rows), batch_size),
        desc=f"{system} generation",
    ):
        batch = rows[start:start + batch_size]

        prompts = [
            tokenizer.apply_chat_template(
                build_messages(row["source_text"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        with torch.inference_mode():
            out = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_example_ms = elapsed_ms / len(batch)

        prompt_width = enc["input_ids"].shape[1]
        generated = out[:, prompt_width:]

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        predictions.extend([x.strip() for x in decoded])
        latencies_ms.extend([per_example_ms] * len(batch))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions, latencies_ms


# ---------------------------------------------------------------------------
# Regex baseline
# ---------------------------------------------------------------------------
REGEX_RULES: List[Tuple[str, re.Pattern[str]]] = [
    (
        "EMAIL",
        re.compile(
            r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])",
            re.IGNORECASE,
        ),
    ),
    (
        "URL",
        re.compile(
            r"\b(?:https?://|www\.)[^\s<>()]+",
            re.IGNORECASE,
        ),
    ),
    (
        "IP",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
        ),
    ),
    (
        "GEOCOORD",
        re.compile(
            r"(?<!\d)[+-]?(?:90(?:\.0+)?|[0-8]?\d(?:\.\d+)?)"
            r"\s*[,;]\s*"
            r"[+-]?(?:180(?:\.0+)?|1[0-7]\d(?:\.\d+)?|"
            r"\d?\d(?:\.\d+)?)(?!\d)"
        ),
    ),
    (
        "SOCIALNUMBER",
        re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"),
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    ),
    (
        "TEL",
        re.compile(
            r"(?<!\w)(?:\+?\d{1,3}[\s().-]*)?"
            r"(?:\(?\d{2,4}\)?[\s.-]*)?"
            r"\d{3,4}[\s.-]\d{3,4}(?!\w)"
        ),
    ),
    (
        "DATE",
        re.compile(
            r"\b(?:"
            r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
            r"|"
            r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
            r"|"
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
            r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
            r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*|\s+)\d{4}"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "TIME",
        re.compile(
            r"\b(?:[01]?\d|2[0-3]):[0-5]\d"
            r"(?:\s?(?:am|pm))?\b"
            r"|\b(?:1[0-2]|0?[1-9])\s?(?:am|pm)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "POSTCODE",
        re.compile(
            r"\b(?:\d{5}(?:-\d{4})?|\d{6}|[A-Z]\d[A-Z]\s?\d[A-Z]\d)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "USERNAME",
        re.compile(r"(?<![\w@])@[A-Za-z0-9_][A-Za-z0-9_.-]{1,31}\b"),
    ),
]


def select_non_overlapping_spans(
    candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prefer earlier start, then longer span, then detector score."""
    ordered = sorted(
        candidates,
        key=lambda x: (
            int(x["start"]),
            -(int(x["end"]) - int(x["start"])),
            -float(x.get("score", 0.0)),
            normalize_label(x["label"]),
        ),
    )

    selected: List[Dict[str, Any]] = []

    for cand in ordered:
        s, e = int(cand["start"]), int(cand["end"])
        if e <= s:
            continue

        conflict = any(
            max(s, int(x["start"])) < min(e, int(x["end"]))
            for x in selected
        )
        if not conflict:
            selected.append(
                {
                    "start": s,
                    "end": e,
                    "label": normalize_label(cand["label"]),
                }
            )

    return sorted(selected, key=lambda x: (x["start"], x["end"]))


def mask_source_with_spans(
    source: str,
    spans: Sequence[Dict[str, Any]],
) -> str:
    pieces: List[str] = []
    cursor = 0

    for span in sorted(spans, key=lambda x: (x["start"], x["end"])):
        s, e = int(span["start"]), int(span["end"])
        if s < cursor:
            continue
        pieces.append(source[cursor:s])
        pieces.append(f"[{normalize_label(span['label'])}]")
        cursor = e

    pieces.append(source[cursor:])
    return "".join(pieces)


def regex_predict(source: str) -> Tuple[str, List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []

    for label, pattern in REGEX_RULES:
        for m in pattern.finditer(source):
            candidates.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "label": label,
                }
            )

    spans = select_non_overlapping_spans(candidates)
    return mask_source_with_spans(source, spans), spans


# ---------------------------------------------------------------------------
# Presidio baseline
# ---------------------------------------------------------------------------
PRESIDIO_TO_DATASET = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "TEL",
    "IP_ADDRESS": "IP",
    "DATE_TIME": "DATE",
    "US_SSN": "SOCIALNUMBER",
    "US_DRIVER_LICENSE": "DRIVERLICENSE",
    "US_PASSPORT": "PASSPORT",
    "URL": "URL",
    "CREDIT_CARD": "CREDIT_CARD",
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
}


def create_presidio_analyzer(spacy_model: str):
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except Exception as e:
        raise RuntimeError(
            "Presidio import failed. Install presidio-analyzer first."
        ) from e

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {
                "lang_code": "en",
                "model_name": spacy_model,
            }
        ],
    }

    try:
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
    except Exception as e:
        raise RuntimeError(
            f"Could not load spaCy model '{spacy_model}'. "
            f"Install it with: python -m spacy download {spacy_model}"
        ) from e

    # AnalyzerEngine creates its default recognizer registry when no custom
    # registry is supplied. That satisfies the assignment's "default
    # recognizers" baseline requirement.
    return AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )


def presidio_predict(
    source: str,
    analyzer,
) -> Tuple[str, List[Dict[str, Any]]]:
    results = analyzer.analyze(
        text=source,
        language="en",
    )

    candidates: List[Dict[str, Any]] = []

    for r in results:
        entity = str(r.entity_type).upper()
        mapped = PRESIDIO_TO_DATASET.get(entity, entity)

        candidates.append(
            {
                "start": int(r.start),
                "end": int(r.end),
                "label": mapped,
                "score": float(getattr(r, "score", 0.0)),
            }
        )

    spans = select_non_overlapping_spans(candidates)
    return mask_source_with_spans(source, spans), spans


# ---------------------------------------------------------------------------
# Evaluation orchestration
# ---------------------------------------------------------------------------
def evaluate_system(
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    evaluated: List[Dict[str, Any]] = []

    if args.system in {"finetuned", "zeroshot"}:
        texts, latencies = generate_masked_texts(
            rows=rows,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            system=args.system,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )

        for row, pred_text, latency_ms in zip(rows, texts, latencies):
            spans, diag = recover_predicted_spans(
                row["source_text"], pred_text
            )
            evaluated.append(
                {
                    "predicted_text": pred_text,
                    "predicted_spans": spans,
                    "latency_ms": float(latency_ms),
                    "alignment_diagnostics": diag,
                }
            )

        return evaluated

    if args.system == "regex":
        for row in rows:
            t0 = time.perf_counter()
            pred_text, spans = regex_predict(row["source_text"])
            latency_ms = (time.perf_counter() - t0) * 1000.0

            evaluated.append(
                {
                    "predicted_text": pred_text,
                    "predicted_spans": spans,
                    "latency_ms": float(latency_ms),
                    "alignment_diagnostics": {},
                }
            )
        return evaluated

    if args.system == "presidio":
        analyzer = create_presidio_analyzer(args.presidio_spacy_model)

        from tqdm.auto import tqdm

        for row in tqdm(rows, desc="presidio evaluation"):
            t0 = time.perf_counter()
            pred_text, spans = presidio_predict(
                row["source_text"], analyzer
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            evaluated.append(
                {
                    "predicted_text": pred_text,
                    "predicted_spans": spans,
                    "latency_ms": float(latency_ms),
                    "alignment_diagnostics": {},
                }
            )
        return evaluated

    raise ValueError(args.system)


def prediction_records(
    rows: Sequence[Dict[str, Any]],
    evaluated: Sequence[Dict[str, Any]],
    system: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for row, out in zip(rows, evaluated):
        leaked = leaked_values_for_row(
            row, out["predicted_text"]
        )

        records.append(
            {
                "id": row["id"],
                "language": row.get("language", "English"),
                "source_text": row["source_text"],
                "predicted_text": out["predicted_text"],
                "predicted_spans": out["predicted_spans"],
                "gold_spans": gold_spans_from_row(row),
                "leaked_values": leaked,
                "latency_ms": round(float(out["latency_ms"]), 3),
                "system": system,
            }
        )

    return records


def merge_report_file(
    report_file: str | Path,
    report: Dict[str, Any],
    is_ood: bool,
) -> None:
    """Keep top-level ID finetuned metrics + aggregate all systems/OOD runs.

    This preserves the assignment's example top-level fields while also
    collecting the baseline and OOD comparisons in one final JSON.
    """
    path = Path(report_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    systems_key = "ood_systems" if is_ood else "systems"

    # Initialize container from an existing report if present.
    merged = dict(existing)
    merged.setdefault("systems", {})
    merged.setdefault("ood_systems", {})

    merged[systems_key][report["system"]] = report

    # For assignment compatibility, keep the ID finetuned report at top level
    # when it exists; otherwise use the first ID report as top-level.
    id_finetuned = merged.get("systems", {}).get("finetuned")
    if id_finetuned is not None:
        top = dict(id_finetuned)
        top["systems"] = merged["systems"]
        top["ood_systems"] = merged["ood_systems"]
        merged = top
    elif not is_ood:
        top = dict(report)
        top["systems"] = merged["systems"]
        top["ood_systems"] = merged["ood_systems"]
        merged = top

    path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    rows = read_jsonl(args.test_file)

    if args.limit:
        rows = rows[:args.limit]

    required_fields = {
        "id", "language", "source_text", "target_text", "privacy_mask", "split"
    }

    for i, row in enumerate(rows):
        missing = required_fields - set(row)
        if missing:
            raise ValueError(f"row {i} missing fields: {sorted(missing)}")

        src = row["source_text"]
        for ent in row["privacy_mask"]:
            s, e = int(ent["start"]), int(ent["end"])
            if src[s:e] != ent["value"]:
                raise ValueError(
                    f"Gold offset mismatch in {row['id']} for {ent}"
                )

    print("=" * 80)
    print("PART B SANITIZER EVALUATION")
    print("=" * 80)
    print("system          :", args.system)
    print("test_file       :", args.test_file)
    print("n               :", len(rows))
    print("base_model      :", args.base_model)
    print("adapter_path    :", args.adapter_path)
    print("batch_size      :", args.batch_size)
    print("max_new_tokens  :", args.max_new_tokens)
    print("seed            :", args.seed)
    print("deterministic   : do_sample=False")
    print("Gold offset check: PASS")

    evaluated = evaluate_system(rows, args)

    pred_records = prediction_records(
        rows, evaluated, args.system
    )
    write_jsonl(args.output_predictions, pred_records)

    metrics = compute_metrics(rows, evaluated)

    report: Dict[str, Any] = {
        "system": args.system,
        "adapter": (
            args.adapter_path
            if args.system == "finetuned"
            else None
        ),
        "base_model": args.base_model,
        "test_set": Path(args.test_file).name,
        "n": len(rows),
        "generation": (
            {
                "prompt": "same Part-A sanitizer prompt",
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "seed": args.seed,
            }
            if args.system in {"finetuned", "zeroshot"}
            else None
        ),
        **metrics,
    }

    is_ood = "ood" in Path(args.test_file).stem.lower()
    merge_report_file(
        args.report_file,
        report,
        is_ood=is_ood,
    )

    print("\n=== RUN COMPLETE ===")
    print("system:", args.system)
    print("n:", len(rows))
    print(
        "exact micro P/R/F1:",
        f"{report['exact']['micro_p']:.6f}",
        f"{report['exact']['micro_r']:.6f}",
        f"{report['exact']['micro_f1']:.6f}",
    )
    print("exact macro F1:", f"{report['exact']['macro_f1']:.6f}")
    print(
        "IoU50 micro P/R/F1:",
        f"{report['overlap_iou50']['micro_p']:.6f}",
        f"{report['overlap_iou50']['micro_r']:.6f}",
        f"{report['overlap_iou50']['micro_f1']:.6f}",
    )
    print(
        "IoU50 macro F1:",
        f"{report['overlap_iou50']['macro_f1']:.6f}",
    )
    print("leak_rate:", f"{report['leak_rate']:.8f}")
    print(
        "over_masking_rate:",
        f"{report['over_masking_rate']:.8f}",
    )
    print(
        "well_formed_rate:",
        f"{report['well_formed_rate']:.8f}",
    )
    print("predictions:", args.output_predictions)
    print("report:", args.report_file)


if __name__ == "__main__":
    main()
