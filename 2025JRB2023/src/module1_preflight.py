#!/usr/bin/env python3
"""Clean Module-1 pre-training checks for the Trustworthy LLM assignment.

This script DOES NOT train. It validates the environment/data, verifies
completion-only labels, measures token lengths, records the A1-A6 sweep, and
optionally performs a Qwen + LoRA dry-run.

Colab:
  %cd /content/drive/MyDrive/TrustworthyLLM_A1
  !python src/module1_preflight.py --project_root . --max_length 768 --model_check
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm.auto import tqdm

import accelerate
import bitsandbytes as bnb
import datasets
import peft
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a privacy-preserving text sanitizer. "
    "Replace personally identifiable information with the appropriate "
    "placeholder labels while preserving all non-sensitive text. "
    "Return only the sanitized text."
)
USER_PREFIX = "Sanitize the following text:\n"

REQUIRED_FIELDS = {
    "id", "language", "source_text", "target_text", "privacy_mask", "split"
}

EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "A1": {
        "method": "LoRA",
        "target_modules": ["q_proj", "v_proj"],
        "lora_r": 8,
        "lora_alpha": 16,
        "quant": "none",
    },
    "A2": {
        "method": "LoRA",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_r": 8,
        "lora_alpha": 16,
        "quant": "none",
    },
    "A3": {
        "method": "LoRA",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "lora_r": 8,
        "lora_alpha": 16,
        "quant": "none",
    },
    "A4": {
        "method": "LoRA",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "lora_r": 32,
        "lora_alpha": 64,
        "quant": "none",
    },
    "A5": {
        "method": "LoRA",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "lora_r": 32,
        "lora_alpha": 16,
        "quant": "none",
    },
    "A6": {
        "method": "QLoRA",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "lora_r": 32,
        "lora_alpha": 64,
        "quant": "4bit",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=Path, default=Path("."))
    p.add_argument("--model", type=str, default=MODEL_DEFAULT)
    p.add_argument("--train_file", type=Path, default=None)
    p.add_argument("--val_file", type=Path, default=None)
    p.add_argument("--expected_train_rows", type=int, default=20_000)
    p.add_argument("--expected_val_rows", type=int, default=2_000)
    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--show_example", action="store_true")
    p.add_argument("--model_check", action="store_true")
    return p.parse_args()


def print_environment() -> None:
    print("=" * 80)
    print("1. ENVIRONMENT")
    print("=" * 80)
    print("Python       :", platform.python_version())
    print("PyTorch      :", torch.__version__)
    print("Transformers :", transformers.__version__)
    print("PEFT         :", peft.__version__)
    print("Accelerate   :", accelerate.__version__)
    print("Datasets     :", datasets.__version__)
    print("bitsandbytes :", bnb.__version__)
    print("CUDA         :", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Enable a GPU runtime in Colab.")

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    native_bf16 = capability[0] >= 8

    print("GPU          :", name)
    print("Compute cap. :", capability)
    print("VRAM         :", f"{total_gb:.2f} GB")
    print("Native BF16  :", native_bf16)
    print("Precision    :", "bf16" if native_bf16 else "fp16")

    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        )
        print("\n", out.stdout)
    except Exception as exc:
        print("[warning] nvidia-smi failed:", exc)


def prepare_dirs(root: Path) -> Dict[str, Path]:
    root = root.expanduser().resolve()
    d = {
        "root": root,
        "data": root / "data",
        "src": root / "src",
        "adapters": root / "adapters",
        "results": root / "results",
        "logs": root / "logs",
    }
    for key, path in d.items():
        if key != "root":
            path.mkdir(parents=True, exist_ok=True)
    print("\nProject root:", root)
    return d


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name}, line {line_no}: {exc}"
                ) from exc
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def verify_professor_manifest(data_dir: Path) -> None:
    """Verify every professor-supplied dataset file against manifest.jsonl."""
    manifest_path = data_dir / "manifest.jsonl"
    if not manifest_path.exists():
        print(
            "[warning] data/manifest.jsonl not found. "
            "The professor bundle includes it; copy the complete data/ folder "
            "to enable exact integrity verification."
        )
        return

    print("\n" + "=" * 80)
    print("PROFESSOR DATA MANIFEST")
    print("=" * 80)

    with manifest_path.open("r", encoding="utf-8") as f:
        items = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    for item in items:
        path = data_dir / str(item["file"])
        if not path.exists():
            raise FileNotFoundError(
                f"Professor manifest lists a missing file: {path}"
            )

        actual_bytes = path.stat().st_size
        actual_rows = sum(1 for line in path.open("rb") if line.strip())
        actual_sha = sha256_file(path)

        if actual_bytes != int(item["bytes"]):
            raise ValueError(
                f"{path.name}: byte count mismatch "
                f"{actual_bytes} != {item['bytes']}"
            )
        if actual_rows != int(item["rows"]):
            raise ValueError(
                f"{path.name}: row count mismatch "
                f"{actual_rows} != {item['rows']}"
            )
        if actual_sha != str(item["sha256"]):
            raise ValueError(
                f"{path.name}: SHA256 mismatch "
                f"{actual_sha} != {item['sha256']}"
            )

        print(
            f"PASS {path.name:28s} rows={actual_rows:5d} "
            f"sha256={actual_sha}"
        )

    print("DATASET BUNDLE INTEGRITY: PASS")


def validate_rows(rows: List[Dict[str, Any]], name: str, expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"{name}: expected {expected} rows, got {len(rows)}")

    ids: List[str] = []
    for i, row in enumerate(rows):
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"{name} row {i} missing: {sorted(missing)}")
        if not isinstance(row["id"], str):
            raise TypeError(f"{name} row {i}: id must be str")
        if not isinstance(row["source_text"], str):
            raise TypeError(f"{name} row {i}: source_text must be str")
        if not isinstance(row["target_text"], str):
            raise TypeError(f"{name} row {i}: target_text must be str")
        if not isinstance(row["privacy_mask"], list):
            raise TypeError(f"{name} row {i}: privacy_mask must be list")

        source = row["source_text"]
        for j, entity in enumerate(row["privacy_mask"]):
            if not isinstance(entity, dict):
                raise TypeError(
                    f"{name} row {i} privacy_mask[{j}] must be dict"
                )

            missing_entity = {"start", "end", "label", "value"} - entity.keys()
            if missing_entity:
                raise ValueError(
                    f"{name} row {i} privacy_mask[{j}] missing "
                    f"{sorted(missing_entity)}"
                )

            try:
                start = int(entity["start"])
                end = int(entity["end"])
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{name} row {i} privacy_mask[{j}] offsets must be integers"
                ) from exc

            if not (0 <= start < end <= len(source)):
                raise ValueError(
                    f"{name} row {i} privacy_mask[{j}] invalid span "
                    f"({start}, {end}) for source length {len(source)}"
                )

            value = str(entity["value"])
            if source[start:end] != value:
                raise ValueError(
                    f"{name} row {i} privacy_mask[{j}] violates professor "
                    "invariant source_text[start:end] == value"
                )

            if not isinstance(entity["label"], str) or not entity["label"]:
                raise TypeError(
                    f"{name} row {i} privacy_mask[{j}] label must be non-empty str"
                )

        ids.append(row["id"])

    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate IDs found")

    print(f"{name}: PASS | rows={len(rows)} | unique_ids={len(set(ids))}")


def build_messages(source_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PREFIX + source_text},
    ]


def encode_example_no_truncation(
    row: Dict[str, Any], tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    """Canonical completion-only serialization for Part A."""
    prompt_text = tokenizer.apply_chat_template(
        build_messages(row["source_text"]),
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False, truncation=False
    )["input_ids"]

    target_ids = tokenizer(
        row["target_text"], add_special_tokens=False, truncation=False
    )["input_ids"]

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    completion_ids = target_ids + [eos_id]
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    attention_mask = [1] * len(input_ids)

    assert len(input_ids) == len(labels) == len(attention_mask)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_length": len(prompt_ids),
        "completion_length": len(completion_ids),
        "total_length": len(input_ids),
    }


def verify_completion_masking(row: Dict[str, Any], tokenizer: AutoTokenizer) -> None:
    enc = encode_example_no_truncation(row, tokenizer)
    prompt_ok = all(x == -100 for x in enc["labels"][: enc["prompt_length"]])
    completion_ok = all(
        x != -100 for x in enc["labels"][enc["prompt_length"] :]
    )
    if not prompt_ok or not completion_ok:
        raise AssertionError("Completion-only label mask is incorrect")

    supervised_ids = [
        token_id
        for token_id, label in zip(enc["input_ids"], enc["labels"])
        if label != -100
    ]

    print("\nCOMPLETION-ONLY LOSS CHECK")
    print("ID                         :", row["id"])
    print("Prompt tokens              :", enc["prompt_length"])
    print("Completion tokens          :", enc["completion_length"])
    print("All prompt labels ignored  :", prompt_ok)
    print("All completion supervised  :", completion_ok)
    print("Supervised token count      :", len(supervised_ids))
    decoded = tokenizer.decode(supervised_ids, skip_special_tokens=False)
    print("Supervised suffix preview   :", repr(decoded[:200]))


def describe(name: str, x: np.ndarray) -> None:
    print(f"\n{name}")
    print("  min   :", int(x.min()))
    print("  mean  :", round(float(x.mean()), 2))
    print("  median:", float(np.median(x)))
    print("  p90   :", float(np.percentile(x, 90)))
    print("  p95   :", float(np.percentile(x, 95)))
    print("  p99   :", float(np.percentile(x, 99)))
    print("  max   :", int(x.max()))


def analyze_lengths(
    rows: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int
) -> Dict[str, np.ndarray]:
    p_list, c_list, t_list = [], [], []
    for row in tqdm(rows, desc="Analyzing prompt/completion lengths"):
        enc = encode_example_no_truncation(row, tokenizer)
        p_list.append(enc["prompt_length"])
        c_list.append(enc["completion_length"])
        t_list.append(enc["total_length"])

    p = np.asarray(p_list)
    c = np.asarray(c_list)
    t = np.asarray(t_list)

    describe("PROMPT LENGTH", p)
    describe("COMPLETION LENGTH", c)
    describe("TOTAL LENGTH", t)

    print("\nExamples exceeding candidate limits:")
    for limit in [256, 384, 512, 768, 1024, 1536, 2048]:
        n = int((t > limit).sum())
        pct = 100.0 * n / len(t)
        print(f"  >{limit:4d}: {n:5d}/{len(t)} ({pct:.3f}%)")

    if int((c <= 1).sum()) != 0:
        raise ValueError("Found an example with no supervised target tokens")

    print("\n10 longest examples (IDs + lengths only):")
    for idx in np.argsort(t)[-10:][::-1]:
        row = rows[int(idx)]
        print(
            f"  {row['id']} | prompt={p[idx]} | "
            f"completion={c[idx]} | total={t[idx]}"
        )

    max_seen = int(t.max())
    print("\nPlanned common max_length:", max_length)
    print("Observed maximum length   :", max_seen)
    if max_seen > max_length:
        raise ValueError(
            f"max_length={max_length} is unsafe; observed max={max_seen}. "
            "Increase it and use the same value for A1-A6."
        )
    print("MAX_LENGTH CHECK: PASS")
    return {"prompt": p, "completion": c, "total": t}


def print_experiments() -> None:
    print("\n" + "=" * 80)
    print("A1-A6 SWEEP")
    print("=" * 80)
    for run_id, cfg in EXPERIMENTS.items():
        print(
            f"{run_id}: {cfg['method']} | modules={','.join(cfg['target_modules'])} | "
            f"r={cfg['lora_r']} | alpha={cfg['lora_alpha']} | quant={cfg['quant']}"
        )


def get_precision() -> Dict[str, Any]:
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8:
        return {"dtype": torch.bfloat16, "bf16": True, "fp16": False}
    return {"dtype": torch.float16, "bf16": False, "fp16": True}


def model_dry_run(model_name: str) -> None:
    print("\n" + "=" * 80)
    print("MODEL + LoRA DRY-RUN")
    print("=" * 80)
    dtype = get_precision()["dtype"]
    torch.cuda.empty_cache()

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        )
    model = model.cuda()

    wanted = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    found = {
        name.split(".")[-1]
        for name, _ in model.named_modules()
        if name.split(".")[-1] in wanted
    }
    missing = wanted - found
    print("Found target modules:", sorted(found))
    if missing:
        raise RuntimeError(f"Missing LoRA target modules: {sorted(missing)}")
    print("TARGET-MODULE CHECK: PASS")

    cfg = EXPERIMENTS["A1"]
    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    print("\nA1 parameter report (dry-run):")
    peft_model.print_trainable_parameters()

    non_lora = [
        name
        for name, param in peft_model.named_parameters()
        if param.requires_grad and "lora_" not in name
    ]
    if non_lora:
        raise RuntimeError(
            "Unexpected non-LoRA trainable parameters: " + str(non_lora[:20])
        )
    print("BASE-MODEL FREEZE CHECK: PASS")

    del peft_model, model
    gc.collect()
    torch.cuda.empty_cache()
    print(
        "GPU allocated after cleanup:",
        f"{torch.cuda.memory_allocated()/1024**3:.3f} GB",
    )


def save_metadata(
    results_dir: Path,
    model_name: str,
    train_file: Path,
    val_file: Path,
    train_rows: int,
    val_rows: int,
    max_length: int,
    stats: Dict[str, np.ndarray],
) -> None:
    precision = get_precision()
    data = {
        "model": model_name,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
        "bitsandbytes": bnb.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "precision": "bf16" if precision["bf16"] else "fp16",
        "train_sha256": sha256_file(train_file),
        "validation_sha256": sha256_file(val_file),
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "max_length": max_length,
        "total_length_mean": float(stats["total"].mean()),
        "total_length_p95": float(np.percentile(stats["total"], 95)),
        "total_length_p99": float(np.percentile(stats["total"], 99)),
        "total_length_max": int(stats["total"].max()),
        "system_prompt": SYSTEM_PROMPT,
        "user_prefix": USER_PREFIX,
        "experiments": EXPERIMENTS,
    }
    out = results_dir / "module1_preflight.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("Saved metadata:", out)


def main() -> None:
    args = parse_args()
    print_environment()
    dirs = prepare_dirs(args.project_root)

    # The professor bundle explicitly supplies data/manifest.jsonl. Verify the
    # complete frozen dataset bundle before using any rows.
    verify_professor_manifest(dirs["data"])

    train_file = (
        args.train_file.expanduser().resolve()
        if args.train_file
        else dirs["data"] / "partA_train.jsonl"
    )
    val_file = (
        args.val_file.expanduser().resolve()
        if args.val_file
        else dirs["data"] / "partA_validation.jsonl"
    )

    if not train_file.exists():
        raise FileNotFoundError(train_file)
    if not val_file.exists():
        raise FileNotFoundError(val_file)

    print("\n" + "=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)
    train_rows = read_jsonl(train_file)
    val_rows = read_jsonl(val_file)
    validate_rows(train_rows, "TRAIN", args.expected_train_rows)
    validate_rows(val_rows, "VALIDATION", args.expected_val_rows)
    print("Train SHA256     :", sha256_file(train_file))
    print("Validation SHA256:", sha256_file(val_file))

    if args.show_example:
        row = train_rows[0]
        print("\nExample ID:", row["id"])
        print("SOURCE:", row["source_text"])
        print("TARGET:", row["target_text"])
        print("MASK  :", row["privacy_mask"])

    print("\n" + "=" * 80)
    print("TOKENIZER")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer    :", args.model)
    print("Vocab size   :", len(tokenizer))
    print("EOS token    :", repr(tokenizer.eos_token), tokenizer.eos_token_id)
    print("PAD token    :", repr(tokenizer.pad_token), tokenizer.pad_token_id)

    verify_completion_masking(train_rows[0], tokenizer)
    stats = analyze_lengths(train_rows + val_rows, tokenizer, args.max_length)
    print_experiments()

    total, used, free = shutil.disk_usage(
        "/content" if Path("/content").exists() else "/"
    )
    print("\nRuntime disk free:", f"{free/1024**3:.2f} GB")

    if args.model_check:
        model_dry_run(args.model)
    else:
        print("\nModel dry-run skipped. Add --model_check to enable it.")

    save_metadata(
        dirs["results"], args.model, train_file, val_file,
        len(train_rows), len(val_rows), args.max_length, stats
    )

    print("\n" + "=" * 80)
    print("PREFLIGHT COMPLETE — NO TRAINING PERFORMED")
    print("=" * 80)
    print("Next: run src/peft_sweep.py for A1-A6.")


if __name__ == "__main__":
    main()
