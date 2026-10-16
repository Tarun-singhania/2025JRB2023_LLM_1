#!/usr/bin/env python3
"""Part A PEFT sweep for the Trustworthy LLM assignment.

One generic script runs A1-A6 by changing CLI arguments. It:
- fine-tunes Qwen2.5-1.5B-Instruct with LoRA or QLoRA on 1 or 2+ GPUs,
- uses completion-only loss (prompt labels = -100),
- preserves supplied row order during training,
- measures trainable parameters via model.print_trainable_parameters(),
- measures exact per-device peak allocated memory via torch.cuda.max_memory_allocated(),
- measures training wall-clock time and non-padding token throughput,
- evaluates completion-only validation loss,
- deterministically generates validation sanitizations,
- recovers masked spans with difflib-based alignment,
- computes IoU>=0.5 micro entity F1 and leak rate,
- appends exactly one row to results/partA_sweep.csv.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import gc
import hashlib
import inspect
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, SequentialSampler
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


SYSTEM_PROMPT = (
    "You are a privacy-preserving text sanitizer. "
    "Replace personally identifiable information with the appropriate "
    "placeholder labels while preserving all non-sensitive text. "
    "Return only the sanitized text."
)
USER_PREFIX = "Sanitize the following text:\n"

REQUIRED_CSV_COLUMNS = [
    "run_id",
    "method",
    "target_modules",
    "lora_r",
    "lora_alpha",
    "quant",
    "trainable_params",
    "total_params",
    "trainable_pct",
    "peak_gpu_mem_gb",
    "train_wallclock_s",
    "tokens_per_s",
    "final_train_loss",
    "val_loss",
    "val_entity_f1",
    "val_leak_rate",
]

# Current assignment PDF (release 31/08/2026) sweep authority.
# The professor data bundle contains an older draft assignment.md with an
# inconsistent A4/A5 table; the newer PDF and its required A4 CLI both use
# A4=(r32, alpha64) and A5=(r32, alpha16).
EXPECTED_SWEEP = {
    "A1": {
        "target_modules": {"q_proj", "v_proj"},
        "lora_r": 8, "lora_alpha": 16, "quant": "none",
    },
    "A2": {
        "target_modules": {"q_proj", "k_proj", "v_proj", "o_proj"},
        "lora_r": 8, "lora_alpha": 16, "quant": "none",
    },
    "A3": {
        "target_modules": {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        },
        "lora_r": 8, "lora_alpha": 16, "quant": "none",
    },
    "A4": {
        "target_modules": {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        },
        "lora_r": 32, "lora_alpha": 64, "quant": "none",
    },
    "A5": {
        "target_modules": {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        },
        "lora_r": 32, "lora_alpha": 16, "quant": "none",
    },
    "A6": {
        "target_modules": {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        },
        "lora_r": 32, "lora_alpha": 64, "quant": "4bit",
    },
}

PLACEHOLDER_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*)\]")
WS_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one Part-A LoRA/QLoRA sanitizer experiment.")

    # Required assignment CLI
    p.add_argument("--model", required=True, type=str)
    p.add_argument("--train_file", required=True, type=str)
    p.add_argument("--val_file", required=True, type=str)
    p.add_argument("--target_modules", required=True, type=str)
    p.add_argument("--lora_r", required=True, type=int)
    p.add_argument("--lora_alpha", required=True, type=int)
    p.add_argument("--lora_dropout", required=True, type=float)
    p.add_argument("--quant", required=True, choices=["none", "4bit"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--run_id", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--results_csv", required=True, type=str)

    # Extra controls; keep identical across A1-A6
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--generation_batch_size", type=int, default=4)
    p.add_argument(
        "--expected_global_effective_batch_size",
        type=int,
        default=0,
        help=(
            "Optional safety check. In multi-GPU data parallel training the global "
            "effective batch is per_device_train_batch_size * world_size * "
            "gradient_accumulation_steps. Use 16 for the recommended T4x2 sweep."
        ),
    )
    p.add_argument(
        "--generation_max_new_tokens",
        type=int,
        default=0,
        help="0 = automatically use max validation completion length + 16.",
    )
    p.add_argument(
        "--strict_assignment_counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require 20,000 train and 2,000 validation rows.",
    )
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--replace_existing_row",
        action="store_true",
        help="Replace an existing run_id row in results_csv instead of failing.",
    )
    p.add_argument(
        "--skip_manifest_check",
        action="store_true",
        help=(
            "Skip data/manifest.jsonl integrity verification. By default, if "
            "a manifest exists beside the supplied Part-A files, it is verified."
        ),
    )
    args = p.parse_args()

    if not (1 <= args.epochs <= 10):
        p.error("--epochs must be an integer between 1 and 10.")
    if args.max_length <= 0:
        p.error("--max_length must be positive.")
    if args.generation_batch_size <= 0:
        p.error("--generation_batch_size must be positive.")
    return args


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return rows



def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest_for_files(
    train_file: str,
    val_file: str,
    skip: bool = False,
) -> None:
    """Verify professor-supplied manifest entries when manifest.jsonl exists."""
    if skip:
        print("Manifest check: SKIPPED by CLI")
        return

    train_path = Path(train_file).resolve()
    val_path = Path(val_file).resolve()

    # Normally both files live in the same professor-supplied data/ directory.
    candidate_dirs = {train_path.parent, val_path.parent}
    manifest_paths = [d / "manifest.jsonl" for d in candidate_dirs if (d / "manifest.jsonl").exists()]

    if not manifest_paths:
        print(
            "[warning] data/manifest.jsonl not found beside Part-A files; "
            "continuing because private/grader inputs may not ship a manifest."
        )
        return

    # If both resolve to the same data directory this naturally de-duplicates.
    for manifest_path in sorted(set(manifest_paths)):
        records: Dict[str, Dict[str, Any]] = {}
        with manifest_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {manifest_path}, line {line_no}: {exc}"
                    ) from exc
                records[str(item["file"])] = item

        for path in (train_path, val_path):
            if path.parent != manifest_path.parent.resolve():
                continue
            item = records.get(path.name)
            if item is None:
                raise ValueError(
                    f"{path.name} is not listed in professor manifest {manifest_path}"
                )
            payload_bytes = path.stat().st_size
            payload_rows = sum(1 for line in path.open("rb") if line.strip())
            payload_sha = sha256_file(path)

            expected_bytes = int(item["bytes"])
            expected_rows = int(item["rows"])
            expected_sha = str(item["sha256"])

            if payload_bytes != expected_bytes:
                raise ValueError(
                    f"Manifest byte-count mismatch for {path.name}: "
                    f"{payload_bytes} != {expected_bytes}"
                )
            if payload_rows != expected_rows:
                raise ValueError(
                    f"Manifest row-count mismatch for {path.name}: "
                    f"{payload_rows} != {expected_rows}"
                )
            if payload_sha != expected_sha:
                raise ValueError(
                    f"Manifest SHA256 mismatch for {path.name}: "
                    f"{payload_sha} != {expected_sha}"
                )

            print(
                f"Manifest PASS: {path.name} | rows={payload_rows} | "
                f"sha256={payload_sha}"
            )


def validate_requested_sweep_config(args: argparse.Namespace) -> None:
    """Catch accidental A1-A6 misconfiguration before expensive GPU training."""
    cfg = EXPECTED_SWEEP.get(str(args.run_id).upper())
    if cfg is None:
        # Extra runs are permitted by the assignment.
        return

    modules = {m.strip() for m in args.target_modules.split(",") if m.strip()}
    mismatches: List[str] = []
    if modules != cfg["target_modules"]:
        mismatches.append(
            f"target_modules={sorted(modules)} expected={sorted(cfg['target_modules'])}"
        )
    if int(args.lora_r) != int(cfg["lora_r"]):
        mismatches.append(f"lora_r={args.lora_r} expected={cfg['lora_r']}")
    if int(args.lora_alpha) != int(cfg["lora_alpha"]):
        mismatches.append(
            f"lora_alpha={args.lora_alpha} expected={cfg['lora_alpha']}"
        )
    if str(args.quant) != str(cfg["quant"]):
        mismatches.append(f"quant={args.quant} expected={cfg['quant']}")

    if mismatches:
        raise ValueError(
            f"{args.run_id} does not match the current required sweep:\n  - "
            + "\n  - ".join(mismatches)
        )


def validate_part_a_rows(rows: Sequence[Dict[str, Any]], name: str) -> None:
    """Validate Part-A schema plus professor-declared character-offset invariant."""
    required = {"id", "language", "source_text", "target_text", "privacy_mask", "split"}
    ids: List[str] = []

    for i, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{name} row {i} missing fields: {sorted(missing)}")
        if not isinstance(row["id"], str):
            raise TypeError(f"{name} row {i}: id must be str")
        if not isinstance(row["source_text"], str):
            raise TypeError(f"{name} row {i}: source_text must be str")
        if not isinstance(row["target_text"], str):
            raise TypeError(f"{name} row {i}: target_text must be str")
        if not isinstance(row["privacy_mask"], list):
            raise TypeError(f"{name} row {i}: privacy_mask must be list")

        source = row["source_text"]
        for j, ent in enumerate(row["privacy_mask"]):
            if not isinstance(ent, dict):
                raise TypeError(f"{name} row {i} privacy_mask[{j}] must be dict")
            missing_ent = {"start", "end", "label", "value"} - ent.keys()
            if missing_ent:
                raise ValueError(
                    f"{name} row {i} privacy_mask[{j}] missing "
                    f"{sorted(missing_ent)}"
                )
            try:
                s, e = int(ent["start"]), int(ent["end"])
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{name} row {i} privacy_mask[{j}] offsets must be integers"
                ) from exc
            if not (0 <= s < e <= len(source)):
                raise ValueError(
                    f"{name} row {i} privacy_mask[{j}] invalid span ({s}, {e}) "
                    f"for source length {len(source)}"
                )
            value = str(ent["value"])
            if source[s:e] != value:
                raise ValueError(
                    f"{name} row {i} privacy_mask[{j}] violates invariant "
                    "source_text[start:end] == value"
                )
            if not isinstance(ent["label"], str) or not ent["label"]:
                raise TypeError(
                    f"{name} row {i} privacy_mask[{j}] label must be non-empty str"
                )

        ids.append(row["id"])

    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs found in {name}")


def build_messages(source_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PREFIX + source_text},
    ]


def encode_row(
    row: Dict[str, Any],
    tokenizer: Any,
) -> Dict[str, Any]:
    """Canonical completion-only serialization; intentionally no truncation."""
    prompt_text = tokenizer.apply_chat_template(
        build_messages(row["source_text"]),
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    target_ids = tokenizer(
        row["target_text"],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")

    completion_ids = target_ids + [int(tokenizer.eos_token_id)]
    input_ids = prompt_ids + completion_ids
    labels = ([-100] * len(prompt_ids)) + completion_ids

    if not target_ids:
        raise ValueError(f"Example {row['id']} has an empty tokenized target_text.")
    if len(input_ids) != len(labels):
        raise AssertionError("input_ids / labels length mismatch")
    if not all(x == -100 for x in labels[: len(prompt_ids)]):
        raise AssertionError("Prompt labels are not fully masked")

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_length": len(prompt_ids),
        "completion_length": len(completion_ids),
        "total_length": len(input_ids),
    }


def build_prompt_ids(row: Dict[str, Any], tokenizer: Any) -> List[int]:
    prompt_text = tokenizer.apply_chat_template(
        build_messages(row["source_text"]),
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]


class EncodedDataset(Dataset):
    def __init__(self, examples: Sequence[Dict[str, Any]]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        return {
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "labels": ex["labels"],
        }


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for ex in batch:
            n = len(ex["input_ids"])
            pad_n = max_len - n
            input_ids.append(ex["input_ids"] + [self.pad_token_id] * pad_n)
            attention_mask.append(ex["attention_mask"] + [0] * pad_n)
            labels.append(ex["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class OrderedTrainer(Trainer):
    """Trainer that preserves the supplied row order instead of random shuffling."""

    def _get_train_sampler(self, train_dataset=None):  # compatible with recent Trainer APIs
        ds = train_dataset if train_dataset is not None else self.train_dataset
        if ds is None:
            return None
        return SequentialSampler(ds)


def distributed_env() -> Tuple[int, int, int, bool]:
    """Return (rank, local_rank, world_size, is_main_process).

    `accelerate launch --multi_gpu --num_processes 2 ...` sets these variables.
    The function also works for ordinary single-GPU `python ...` execution.
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size, rank == 0


def native_bf16_available(device_index: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(device_index)
    return major >= 8


def _from_pretrained_with_dtype(model_name: str, dtype: torch.dtype, **kwargs):
    """Transformers v4/v5 compatible model loading (`dtype` vs `torch_dtype`)."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            **kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            **kwargs,
        )


def load_peft_model(
    args: argparse.Namespace,
    dtype: torch.dtype,
    local_rank: int,
    is_main_process: bool,
):
    """Load one full model replica on this process's local GPU and attach PEFT.

    This is data parallelism (DDP): with two T4s there is one model replica per
    GPU. Do NOT use device_map="auto" here; that is model-parallel inference,
    not the desired two-GPU data-parallel training setup.
    """
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    if not target_modules:
        raise ValueError("No target modules supplied.")

    device = torch.device(f"cuda:{local_rank}")

    if args.quant == "4bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        # Each distributed process owns one quantized model replica on its GPU.
        model = _from_pretrained_with_dtype(
            args.model,
            dtype,
            quantization_config=qconfig,
            device_map={"": local_rank},
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        model = _from_pretrained_with_dtype(
            args.model,
            dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Explicit assignment requirement. Print once to avoid duplicate DDP logs.
    if is_main_process:
        model.print_trainable_parameters()

    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.config.use_cache = False
    return model


def _distributed_max(value: float, device: torch.device) -> float:
    """Exact maximum of a scalar across ranks; identity in single-GPU mode."""
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def count_parameters(model: torch.nn.Module) -> Tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    return int(trainable), int(total), float(pct)


def normalize_ws_case(text: str) -> str:
    """Case-insensitive, whitespace-normalized string used for leak checks."""
    return WS_RE.sub(" ", str(text)).strip().casefold()


def normalize_label(label: str) -> str:
    """Dataset labels are authoritative; preserve semantic numeric suffixes."""
    return str(label).strip().upper()


def allowed_labels_from_rows(rows: Sequence[Dict[str, Any]]) -> set[str]:
    return {
        normalize_label(ent["label"])
        for row in rows
        for ent in row.get("privacy_mask", [])
        if isinstance(ent, dict) and ent.get("label") is not None
    }


def parse_prediction_skeleton(
    predicted_text: str,
    allowed_labels: set[str],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Remove only valid PII placeholders; preserve unrelated bracketed text.

    The supplied corpus can legitimately contain literal bracketed strings such
    as [RESET_LINK] or [TITLE_A] that are NOT privacy-mask labels. Treating all
    bracketed text as a predicted entity would create false positives.
    """
    skeleton_parts: List[str] = []
    placeholders: List[Dict[str, Any]] = []
    cursor = 0
    skel_pos = 0

    for m in PLACEHOLDER_RE.finditer(predicted_text):
        label = normalize_label(m.group(1))
        if label not in allowed_labels:
            continue

        literal = predicted_text[cursor : m.start()]
        skeleton_parts.append(literal)
        skel_pos += len(literal)
        placeholders.append({"pos": skel_pos, "label": label})
        cursor = m.end()

    tail = predicted_text[cursor:]
    skeleton_parts.append(tail)
    return "".join(skeleton_parts), placeholders


def _split_source_span(
    start: int,
    end: int,
    n: int,
    source: str,
) -> List[Tuple[int, int]]:
    if n <= 1:
        return [(start, end)]
    if end <= start:
        return [(start, start)] * n

    segment = source[start:end]
    tokens = list(re.finditer(r"\S+", segment))
    if len(tokens) >= n:
        groups: List[Tuple[int, int]] = []
        boundaries = np.linspace(0, len(tokens), n + 1, dtype=int)
        for k in range(n):
            lo, hi = boundaries[k], boundaries[k + 1]
            if hi <= lo:
                hi = min(lo + 1, len(tokens))
            s = start + tokens[lo].start()
            e = start + tokens[hi - 1].end()
            groups.append((s, e))
        return groups

    cuts = [start + round((end - start) * k / n) for k in range(n + 1)]
    return [(cuts[k], cuts[k + 1]) for k in range(n)]


def recover_predicted_spans(
    source_text: str,
    predicted_text: str,
    allowed_labels: set[str],
) -> List[Dict[str, Any]]:
    """Recover predicted source spans using literal-anchor alignment.

    Rules:
      1. Only placeholders belonging to the dataset PII label vocabulary are
         treated as entity predictions.
      2. Remove those placeholders to form a literal skeleton.
      3. Align source_text to that skeleton with difflib.SequenceMatcher.
      4. Map each placeholder to the source region between nearest equal
         literal anchors.
      5. If multiple adjacent placeholders map to one unresolved region, split
         that region monotonically across whitespace tokens, then by chars.

    This alignment is deliberately deterministic. Its failure modes should be
    documented in the report (large model rewrites, reordered text, missing
    literal anchors, or ambiguous adjacent replacements).
    """
    skeleton, placeholders = parse_prediction_skeleton(
        predicted_text, allowed_labels
    )
    if not placeholders:
        return []

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
                "pos": p,
            }
        )

    grouped: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, span in enumerate(raw):
        grouped[(span["start"], span["end"])].append(idx)

    for (start, end), idxs in grouped.items():
        if len(idxs) <= 1:
            continue
        pieces = _split_source_span(start, end, len(idxs), source_text)
        for idx, (s, e) in zip(idxs, pieces):
            raw[idx]["start"] = s
            raw[idx]["end"] = e

    return [
        {
            "start": int(x["start"]),
            "end": int(x["end"]),
            "label": normalize_label(x["label"]),
        }
        for x in raw
        if int(x["end"]) > int(x["start"])
    ]


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
    """Deterministic one-to-one greedy bipartite matching.

    For IoU matching, valid pairs are same-label with IoU >= 0.5 and are
    consumed from highest IoU downward, matching the assignment's
    maximum-IoU/one-to-one requirement.
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


def gold_spans_from_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "start": int(ent["start"]),
            "end": int(ent["end"]),
            "label": normalize_label(ent["label"]),
        }
        for ent in row.get("privacy_mask", [])
    ]


def _safe_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * p * r / (p + r) if (p + r) else 0.0
    return float(p), float(r), float(f1)


def over_mask_counts(
    source_text: str,
    predicted_text: str,
    gold_spans: Sequence[Dict[str, Any]],
) -> Tuple[int, int]:
    """Return (changed_non_pii_chars, total_non_pii_chars).

    A source character is considered preserved only if it participates in an
    exact/equal SequenceMatcher block. Gold PII character positions are removed
    from the denominator, so changes inside true PII spans are not penalized.
    """
    n = len(source_text)
    gold_mask = np.zeros(n, dtype=bool)
    for span in gold_spans:
        s = max(0, min(int(span["start"]), n))
        e = max(s, min(int(span["end"]), n))
        gold_mask[s:e] = True

    preserved = np.zeros(n, dtype=bool)
    sm = difflib.SequenceMatcher(None, source_text, predicted_text, autojunk=False)
    for block in sm.get_matching_blocks():
        if block.size > 0:
            preserved[block.a : block.a + block.size] = True

    non_pii = ~gold_mask
    total = int(non_pii.sum())
    changed = int((non_pii & ~preserved).sum())
    return changed, total


def compute_validation_metrics(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    allowed_labels: set[str],
) -> Dict[str, Any]:
    """Compute the Part-B-style validation metrics used for Part-A selection."""
    if len(rows) != len(predictions):
        raise ValueError("rows/predictions length mismatch")

    criteria = ("exact", "iou50")
    aggregate = {
        c: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "per_class": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0}),
        }
        for c in criteria
    }

    leak_num = leak_den = 0
    over_changed = over_total = 0

    for row, pred_text in zip(rows, predictions):
        pred_spans = recover_predicted_spans(
            row["source_text"], pred_text, allowed_labels
        )
        gold_spans = gold_spans_from_row(row)

        gold_support = defaultdict(int)
        for g in gold_spans:
            gold_support[normalize_label(g["label"])] += 1

        for criterion in criteria:
            pairs = match_entities(pred_spans, gold_spans, criterion)
            matched_pred = {pi for pi, _gi in pairs}
            matched_gold = {gi for _pi, gi in pairs}

            agg = aggregate[criterion]
            agg["tp"] += len(pairs)
            agg["fp"] += len(pred_spans) - len(pairs)
            agg["fn"] += len(gold_spans) - len(pairs)

            for label, support in gold_support.items():
                agg["per_class"][label]["support"] += support

            for pi, gi in pairs:
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

        norm_pred = normalize_ws_case(pred_text)
        for ent in row.get("privacy_mask", []):
            value = ent.get("value", "")
            if value is None:
                continue
            norm_value = normalize_ws_case(str(value))
            if not norm_value:
                continue
            leak_den += 1
            if norm_value in norm_pred:
                leak_num += 1

        changed, total = over_mask_counts(
            row["source_text"], pred_text, gold_spans
        )
        over_changed += changed
        over_total += total

    reports: Dict[str, Any] = {}
    per_class_reports: Dict[str, Dict[str, Any]] = {}

    for criterion in criteria:
        agg = aggregate[criterion]
        micro_p, micro_r, micro_f1 = _safe_prf(
            int(agg["tp"]), int(agg["fp"]), int(agg["fn"])
        )

        class_report: Dict[str, Dict[str, Any]] = {}
        macro_f1_values: List[float] = []
        for label in sorted(agg["per_class"]):
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
        per_class_reports[criterion] = class_report

    # Assignment asks for the 10 most frequent classes. Use IoU50 values in
    # the convenience table while retaining exact per-class metrics separately.
    iou_classes = per_class_reports["iou50"]
    top10_labels = sorted(
        iou_classes,
        key=lambda label: (-int(iou_classes[label]["support"]), label),
    )[:10]
    top10 = {label: iou_classes[label] for label in top10_labels}

    return {
        "exact": reports["exact"],
        "overlap_iou50": reports["iou50"],
        "per_class": top10,
        "per_class_exact_all": per_class_reports["exact"],
        "per_class_iou50_all": per_class_reports["iou50"],
        "leak_rate": float(leak_num / leak_den) if leak_den else 0.0,
        "over_masking_rate": (
            float(over_changed / over_total) if over_total else 0.0
        ),
        "n_gold_values": int(leak_den),
        "n_leaked_values": int(leak_num),
        "n_non_pii_chars": int(over_total),
        "n_overmasked_non_pii_chars": int(over_changed),
    }


def left_pad_prompts(prompt_batch: Sequence[List[int]], pad_id: int, device: torch.device):
    max_len = max(len(x) for x in prompt_batch)
    bs = len(prompt_batch)
    input_ids = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(prompt_batch):
        t = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[i, -len(ids) :] = t
        attn[i, -len(ids) :] = 1
    return input_ids, attn


def generate_validation_predictions_distributed(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[Dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    rank: int,
    world_size: int,
    is_main_process: bool,
) -> List[str] | None:
    """Deterministic validation generation sharded across all GPU processes.

    Rank r generates validation examples r, r+world_size, ... . Predictions are
    gathered and restored to the original validation-file order on rank 0.
    This makes the expensive 2,000-example generation step use both T4 GPUs.
    """
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define pad/eos IDs for generation.")

    gen_cfg = GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        typical_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        early_stopping=False,
        pad_token_id=int(pad_id),
        eos_token_id=int(tokenizer.eos_token_id),
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
    )

    local_indices = list(range(rank, len(rows), world_size))
    local_pairs: List[Tuple[int, str]] = []

    iterator = range(0, len(local_indices), batch_size)
    iterator = tqdm(
        iterator,
        desc=f"Validation generation rank {rank}",
        disable=not is_main_process,
    )

    with torch.inference_mode():
        for start in iterator:
            batch_indices = local_indices[start : start + batch_size]
            batch_rows = [rows[i] for i in batch_indices]
            prompt_ids = [build_prompt_ids(r, tokenizer) for r in batch_rows]
            input_ids, attention_mask = left_pad_prompts(
                prompt_ids, int(pad_id), device
            )
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_cfg,
            )
            new_tokens = output[:, input_ids.shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            local_pairs.extend(
                (idx, text) for idx, text in zip(batch_indices, texts)
            )

    if world_size == 1 or not (dist.is_available() and dist.is_initialized()):
        local_pairs.sort(key=lambda x: x[0])
        return [text for _idx, text in local_pairs]

    gathered: List[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_pairs)

    if not is_main_process:
        return None

    merged: List[Tuple[int, str]] = []
    for part in gathered:
        merged.extend(part)
    merged.sort(key=lambda x: x[0])

    if len(merged) != len(rows):
        raise RuntimeError(
            f"Distributed generation produced {len(merged)} predictions for "
            f"{len(rows)} validation rows."
        )
    indices = [idx for idx, _text in merged]
    if indices != list(range(len(rows))):
        raise RuntimeError("Distributed validation prediction indices are incomplete/duplicated.")

    return [text for _idx, text in merged]


def append_csv_row(path: str, row: Dict[str, Any], replace_existing: bool) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    existing: List[Dict[str, str]] = []
    if out.exists():
        with out.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
        matching = [r for r in existing if r.get("run_id") == str(row["run_id"])]
        if matching and not replace_existing:
            raise ValueError(
                f"run_id={row['run_id']} already exists in {path}. "
                "Use --replace_existing_row to replace it."
            )
        if matching:
            existing = [r for r in existing if r.get("run_id") != str(row["run_id"])]

    existing.append({k: row[k] for k in REQUIRED_CSV_COLUMNS})
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(existing)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this assignment run.")

    rank, local_rank, world_size, is_main_process = distributed_env()
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible."
        )

    # Critical: make each Accelerate/DDP process load its model on a different T4.
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_all_seeds(args.seed)

    global_effective_batch = (
        args.per_device_train_batch_size
        * world_size
        * args.gradient_accumulation_steps
    )
    if (
        args.expected_global_effective_batch_size > 0
        and global_effective_batch != args.expected_global_effective_batch_size
    ):
        raise ValueError(
            "Global effective batch-size check failed: "
            f"per_device={args.per_device_train_batch_size} * "
            f"world_size={world_size} * "
            f"grad_accum={args.gradient_accumulation_steps} = "
            f"{global_effective_batch}, expected "
            f"{args.expected_global_effective_batch_size}."
        )

    if is_main_process:
        print("=== DISTRIBUTED SETUP ===")
        print(f"Visible CUDA GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(
                f"GPU {i}: {torch.cuda.get_device_name(i)} | "
                f"compute={torch.cuda.get_device_capability(i)} | "
                f"VRAM={torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
            )
        print(f"world_size: {world_size}")
        print(f"per-device train batch: {args.per_device_train_batch_size}")
        print(f"gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"GLOBAL EFFECTIVE BATCH: {global_effective_batch}")
        if world_size == 1:
            print(
                "[warning] Single-process execution detected. To use both Kaggle T4 GPUs, "
                "launch with: accelerate launch --multi_gpu --num_processes 2 ..."
            )

    validate_requested_sweep_config(args)
    verify_manifest_for_files(
        args.train_file,
        args.val_file,
        skip=args.skip_manifest_check,
    )

    train_rows = read_jsonl(args.train_file)
    val_rows = read_jsonl(args.val_file)
    validate_part_a_rows(train_rows, "train")
    validate_part_a_rows(val_rows, "validation")

    if args.strict_assignment_counts:
        if len(train_rows) != 20_000:
            raise ValueError(f"Expected 20,000 training rows, found {len(train_rows)}")
        if len(val_rows) != 2_000:
            raise ValueError(f"Expected 2,000 validation rows, found {len(val_rows)}")

    allowed_labels = allowed_labels_from_rows(train_rows + val_rows)
    if is_main_process:
        print(
            f"PII label vocabulary ({len(allowed_labels)} classes): "
            + ", ".join(sorted(allowed_labels))
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")

    if is_main_process:
        print("Encoding training set (no truncation)...")
    encoded_train = [
        encode_row(r, tokenizer)
        for r in tqdm(train_rows, disable=not is_main_process)
    ]
    if is_main_process:
        print("Encoding validation set (no truncation)...")
    encoded_val = [
        encode_row(r, tokenizer)
        for r in tqdm(val_rows, disable=not is_main_process)
    ]

    all_lengths = [x["total_length"] for x in encoded_train + encoded_val]
    longest = max(all_lengths)
    n_over = sum(1 for x in all_lengths if x > args.max_length)
    if is_main_process:
        print(
            f"Sequence lengths: mean={np.mean(all_lengths):.2f}, "
            f"p95={np.percentile(all_lengths,95):.1f}, "
            f"p99={np.percentile(all_lengths,99):.1f}, max={longest}"
        )
    if n_over:
        raise ValueError(
            f"{n_over} examples exceed --max_length={args.max_length}; longest={longest}. "
            "This script refuses silent truncation."
        )

    train_ds = EncodedDataset(encoded_train)
    val_ds = EncodedDataset(encoded_val)
    collator = CompletionOnlyCollator(tokenizer.pad_token_id)

    dtype = torch.bfloat16 if native_bf16_available(local_rank) else torch.float16
    precision_name = "bf16" if dtype == torch.bfloat16 else "fp16"
    if is_main_process:
        print(f"Training precision: {precision_name}")

    gc.collect()
    torch.cuda.empty_cache()
    model = load_peft_model(
        args=args,
        dtype=dtype,
        local_rank=local_rank,
        is_main_process=is_main_process,
    )
    trainable_params, total_params, trainable_pct = count_parameters(model)

    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_drop_last=False,
        seed=args.seed,
        data_seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=1.0,
    )

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in ta_params:
        training_kwargs["warmup_ratio"] = args.warmup_ratio
    elif "warmup_steps" in ta_params:
        # Transformers v5 accepts a float in [0,1) as a warmup ratio here.
        training_kwargs["warmup_steps"] = float(args.warmup_ratio)
    else:
        raise RuntimeError(
            "This Transformers version exposes neither warmup_ratio nor warmup_steps."
        )

    if "train_sampling_strategy" in ta_params:
        training_kwargs["train_sampling_strategy"] = "sequential"
    if "local_rank" in ta_params:
        training_kwargs["local_rank"] = local_rank
    if world_size > 1 and "ddp_find_unused_parameters" in ta_params:
        # LoRA has a fixed known trainable graph; avoids unnecessary DDP traversal.
        training_kwargs["ddp_find_unused_parameters"] = False
    if world_size > 1 and "ddp_backend" in ta_params:
        training_kwargs["ddp_backend"] = "nccl"

    train_args = TrainingArguments(**training_kwargs)

    trainer = OrderedTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    # Global exact dataset-token count. With the supplied 20,000 rows and the
    # recommended T4x2 batches, Accelerate/DDP shards the sequential dataloader
    # across ranks without requiring padding duplicates.
    nonpad_tokens_per_epoch = sum(x["total_length"] for x in encoded_train)
    exact_train_tokens = nonpad_tokens_per_epoch * args.epochs

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize(device)
    local_wallclock_s = time.perf_counter() - t0
    local_peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # Report actual job time and the worst per-GPU allocated-memory peak.
    train_wallclock_s = _distributed_max(local_wallclock_s, device)
    peak_gpu_mem_gb = _distributed_max(local_peak_gb, device)
    tokens_per_s = (
        exact_train_tokens / train_wallclock_s if train_wallclock_s > 0 else 0.0
    )

    final_train_loss = float(train_result.metrics.get("train_loss", math.nan))

    # Trainer/Accelerate performs distributed teacher-forced evaluation.
    eval_metrics = trainer.evaluate(eval_dataset=val_ds)
    val_loss = float(eval_metrics["eval_loss"])

    max_val_completion = max(x["completion_length"] for x in encoded_val)
    max_new_tokens = (
        args.generation_max_new_tokens
        if args.generation_max_new_tokens > 0
        else max_val_completion + 16
    )
    if is_main_process:
        print(
            f"Validation generation max_new_tokens={max_new_tokens}; "
            f"using {world_size} GPU process(es)"
        )

    # Unwrap DDP before model.generate/save_pretrained. Each process keeps its
    # own synchronized model replica and generates only its shard of validation.
    wrapped_model = getattr(trainer, "model_wrapped", trainer.model)
    generation_model = trainer.accelerator.unwrap_model(wrapped_model)
    torch.cuda.empty_cache()
    predictions = generate_validation_predictions_distributed(
        model=generation_model,
        tokenizer=tokenizer,
        rows=val_rows,
        batch_size=args.generation_batch_size,
        max_new_tokens=max_new_tokens,
        rank=rank,
        world_size=world_size,
        is_main_process=is_main_process,
    )

    if is_main_process:
        assert predictions is not None
        val_metrics = compute_validation_metrics(
            val_rows,
            predictions,
            allowed_labels=allowed_labels,
        )
        val_entity_f1 = val_metrics["overlap_iou50"]["micro_f1"]
        val_leak_rate = val_metrics["leak_rate"]

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        generation_model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        run_metrics = {
            "run_id": args.run_id,
            "distributed_world_size": world_size,
            "gpu_names": gpu_names,
            "precision": precision_name,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_effective_batch_size": global_effective_batch,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "generation_batch_size_per_gpu": args.generation_batch_size,
            "peak_gpu_mem_gb_definition": "maximum torch.cuda.max_memory_allocated() across participating GPUs",
            "throughput_definition": "global non-padding training tokens / distributed wall-clock seconds",
            "max_validation_completion_tokens": max_val_completion,
            "generation_max_new_tokens": max_new_tokens,
            "validation": val_metrics,
        }
        with (out_dir / "run_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(run_metrics, f, indent=2)

        method = "QLoRA" if args.quant == "4bit" else "LoRA"
        csv_row = {
            "run_id": args.run_id,
            "method": method,
            "target_modules": args.target_modules,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "quant": args.quant,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_pct": f"{trainable_pct:.8f}",
            "peak_gpu_mem_gb": f"{peak_gpu_mem_gb:.6f}",
            "train_wallclock_s": f"{train_wallclock_s:.3f}",
            "tokens_per_s": f"{tokens_per_s:.3f}",
            "final_train_loss": f"{final_train_loss:.8f}",
            "val_loss": f"{val_loss:.8f}",
            "val_entity_f1": f"{val_entity_f1:.8f}",
            "val_leak_rate": f"{val_leak_rate:.8f}",
        }
        append_csv_row(args.results_csv, csv_row, args.replace_existing_row)

        print("\n=== RUN COMPLETE ===")
        print(f"distributed_world_size: {world_size}")
        print(f"global_effective_batch_size: {global_effective_batch}")
        for k in REQUIRED_CSV_COLUMNS:
            print(f"{k}: {csv_row[k]}")
        print(f"Adapter saved to: {out_dir}")
        print(f"CSV updated: {args.results_csv}")

    # Ensure rank 0 finishes writing before every worker exits.
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
