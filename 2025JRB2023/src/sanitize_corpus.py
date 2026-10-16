#!/usr/bin/env python3
import argparse, hashlib, json, os, random, re, shutil, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm

# Frozen Part-B sanitizer prompt/config, embedded here so this script is
# self-contained under accelerate multi-process launch.
SYSTEM_PROMPT = (
    "You are a privacy-preserving text sanitizer. "
    "Replace personally identifiable information with the appropriate "
    "placeholder labels while preserving all non-sensitive text. "
    "Return only the sanitized text."
)
USER_PREFIX = "Sanitize the following text:\n"

def build_messages(source_text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PREFIX + source_text},
    ]

def _resolve_lora_module_name(raw_name, module_names):
    candidates = [raw_name]
    for prefix in ("base_model.model.", "base_model."):
        if raw_name.startswith(prefix):
            candidates.append(raw_name[len(prefix):])

    for cand in candidates:
        if cand in module_names:
            return cand

    matches = [name for name in module_names if name and raw_name.endswith(name)]
    if matches:
        return max(matches, key=len)

    raise KeyError(f"Could not map LoRA module path: {raw_name}")


def _merge_lora_adapter_into_model(model, adapter_path):
    from safetensors.torch import load_file

    adapter_path = Path(adapter_path)
    cfg_path = adapter_path / "adapter_config.json"
    weight_path = adapter_path / "adapter_model.safetensors"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing {weight_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    state = load_file(str(weight_path), device="cpu")

    default_r = int(cfg["r"])
    default_alpha = float(cfg["lora_alpha"])
    use_rslora = bool(cfg.get("use_rslora", False))
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    rank_pattern = cfg.get("rank_pattern") or {}
    alpha_pattern = cfg.get("alpha_pattern") or {}

    modules = dict(model.named_modules())
    module_names = set(modules)

    a_re = re.compile(r"^(.*)\.lora_A(?:\.[^.]+)?\.weight$")
    b_re = re.compile(r"^(.*)\.lora_B(?:\.[^.]+)?\.weight$")

    A_by_base = {}
    B_by_base = {}

    for key, tensor in state.items():
        ma = a_re.match(key)
        if ma:
            A_by_base[ma.group(1)] = tensor
            continue
        mb = b_re.match(key)
        if mb:
            B_by_base[mb.group(1)] = tensor

    if not A_by_base:
        raise RuntimeError("No LoRA A matrices found in adapter_model.safetensors.")

    if set(A_by_base) != set(B_by_base):
        missing_b = sorted(set(A_by_base) - set(B_by_base))[:10]
        missing_a = sorted(set(B_by_base) - set(A_by_base))[:10]
        raise RuntimeError(
            f"Unpaired LoRA matrices. missing_B={missing_b}, missing_A={missing_a}"
        )

    merged = 0
    unresolved = []

    with torch.no_grad():
        for raw_base in sorted(A_by_base):
            try:
                module_name = _resolve_lora_module_name(raw_base, module_names)
            except KeyError:
                unresolved.append(raw_base)
                continue

            module = modules[module_name]
            if not hasattr(module, "weight"):
                raise RuntimeError(f"Resolved module has no weight: {module_name}")

            A = A_by_base[raw_base]
            B = B_by_base[raw_base]

            r = default_r
            alpha = default_alpha
            for key, value in rank_pattern.items():
                if module_name.endswith(key):
                    r = int(value)
                    break
            for key, value in alpha_pattern.items():
                if module_name.endswith(key):
                    alpha = float(value)
                    break

            scale = alpha / ((r ** 0.5) if use_rslora else r)
            delta = torch.matmul(B.float(), A.float()) * scale
            if fan_in_fan_out:
                delta = delta.T

            if tuple(delta.shape) != tuple(module.weight.shape):
                raise RuntimeError(
                    f"Shape mismatch for {module_name}: "
                    f"delta={tuple(delta.shape)}, weight={tuple(module.weight.shape)}"
                )

            module.weight.add_(
                delta.to(device=module.weight.device, dtype=module.weight.dtype)
            )
            merged += 1

    if unresolved:
        raise RuntimeError(
            "Could not resolve some adapter modules: " + ", ".join(unresolved[:10])
        )

    print(
        f"Manual LoRA merge: PASS ({merged} modules merged; "
        f"r={default_r}, alpha={default_alpha}, rslora={use_rslora})"
    )
    return model


def load_generation_model(base_model, adapter_path, system, device):
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
    model = model.to(device)

    if system == "finetuned":
        model = _merge_lora_adapter_into_model(model, adapter_path)

    model.eval()
    model.config.use_cache = True

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    return tokenizer, model

REQ = {"doc_id","text","n_reps","is_canary","canary_id","repetition_index"}

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def norm(x):
    return re.sub(r"\s+", " ", str(x)).strip().casefold()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--canaries_file", required=True)
    p.add_argument("--audit_file", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=260)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    if a.max_new_tokens != 260:
        p.error("Part B was frozen at max_new_tokens=260; Part C must reuse it.")
    if a.seed != 42:
        p.error("Use the frozen seed 42.")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)

    acc = Accelerator()
    rank, world, device = acc.process_index, acc.num_processes, acc.device

    rows = read_jsonl(a.input_file)
    canaries = read_jsonl(a.canaries_file)

    if len(rows) != 8007:
        raise ValueError(f"Expected 8007 raw rows, got {len(rows)}")
    if len(canaries) != 20:
        raise ValueError(f"Expected 20 canaries, got {len(canaries)}")
    for i, row in enumerate(rows):
        miss = REQ - set(row)
        if miss:
            raise ValueError(f"row {i} missing {sorted(miss)}")

    out_path = Path(a.output_file)
    audit_path = Path(a.audit_file)
    shard_dir = out_path.parent / ".partC_sanitize_shards"

    if acc.is_main_process:
        if out_path.exists() and not a.overwrite:
            raise FileExistsError(f"{out_path} exists; use --overwrite intentionally.")
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True)
        print("="*80)
        print("PART C CORPUS SANITIZATION")
        print("="*80)
        print("rows             :", len(rows))
        print("processes / GPUs :", world)
        print("batch_size/GPU   :", a.batch_size)
        print("do_sample        : False")
        print("max_new_tokens   :", a.max_new_tokens)
        print("seed             :", a.seed)
        print("canary special-case: NO")
    acc.wait_for_everyone()

    tokenizer, model = load_generation_model(
        a.base_model, a.adapter_path, "finetuned", device
    )

    my_idx = list(range(rank, len(rows), world))
    shard_path = shard_dir / f"rank_{rank}.jsonl"

    t0_all = time.perf_counter()
    with shard_path.open("w", encoding="utf-8") as fout:
        for off in tqdm(
            range(0, len(my_idx), a.batch_size),
            desc=f"GPU/process {rank}",
            disable=not acc.is_local_main_process and rank != 1,
        ):
            idxs = my_idx[off:off+a.batch_size]
            batch = [rows[i] for i in idxs]
            prompts = [
                tokenizer.apply_chat_template(
                    build_messages(x["text"]),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for x in batch
            ]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=False, add_special_tokens=False
            )
            enc = {k:v.to(device) for k,v in enc.items()}

            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=a.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            w = enc["input_ids"].shape[1]
            gen = out[:, w:]
            texts = tokenizer.batch_decode(
                gen, skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            for j, (idx, txt) in enumerate(zip(idxs, texts)):
                ids = gen[j].tolist()
                eos = tokenizer.eos_token_id
                first_eos = ids.index(eos) if eos is not None and eos in ids else None
                hit_cap = first_eos is None and len(ids) >= a.max_new_tokens
                n_gen = first_eos if first_eos is not None else len(ids)
                fout.write(json.dumps({
                    "index": idx,
                    "text": txt.strip(),
                    "generated_tokens": int(n_gen),
                    "hit_max_new_tokens": bool(hit_cap),
                }, ensure_ascii=False) + "\n")

    del model
    torch.cuda.empty_cache()
    acc.wait_for_everyone()

    if not acc.is_main_process:
        return

    # Merge shards in original row order.
    pred = {}
    for r in range(world):
        for x in read_jsonl(shard_dir / f"rank_{r}.jsonl"):
            pred[int(x["index"])] = x
    if set(pred) != set(range(len(rows))):
        raise RuntimeError("Shard merge failed: missing/duplicate row indices.")

    san_rows, hit_cap_rows = [], []
    for i, raw in enumerate(rows):
        x = pred[i]
        san = dict(raw)
        san["text"] = x["text"]              # ONLY field allowed to change
        san_rows.append(san)
        if x["hit_max_new_tokens"]:
            hit_cap_rows.append({
                "index": i,
                "doc_id": raw["doc_id"],
                "is_canary": bool(raw["is_canary"]),
                "generated_tokens": x["generated_tokens"],
            })

    meta_fields = ["doc_id","n_reps","is_canary","canary_id","repetition_index"]
    for i, (raw, san) in enumerate(zip(rows, san_rows)):
        for k in meta_fields:
            if raw[k] != san[k]:
                raise RuntimeError(f"Metadata changed at row {i}, field {k}")
    if [x["doc_id"] for x in rows] != [x["doc_id"] for x in san_rows]:
        raise RuntimeError("Row order changed.")

    # Repeated identical inputs should sanitize identically under deterministic decode.
    raw_by_doc, san_by_doc = defaultdict(set), defaultdict(set)
    for raw, san in zip(rows, san_rows):
        raw_by_doc[str(raw["doc_id"])].add(raw["text"])
        san_by_doc[str(raw["doc_id"])].add(san["text"])
    repeat_fail = [
        d for d in raw_by_doc
        if len(raw_by_doc[d]) == 1 and len(san_by_doc[d]) != 1
    ]

    write_jsonl(out_path, san_rows)

    # Canary survival audit: do not remove survivors; only count them.
    cmeta = {str(x["canary_id"]): x for x in canaries}
    per_canary = {}
    for cid, x in cmeta.items():
        per_canary[cid] = {
            "canary_id": cid, "n_reps": int(x["n_reps"]),
            "rows": 0, "survivor_rows": 0, "any_survival": False
        }

    for raw, san in zip(rows, san_rows):
        if not raw["is_canary"]:
            continue
        cid = str(raw["canary_id"])
        secret = str(cmeta[cid]["secret"])
        survived = bool(norm(secret)) and norm(secret) in norm(san["text"])
        q = per_canary[cid]
        q["rows"] += 1
        q["survivor_rows"] += int(survived)
        q["any_survival"] = q["any_survival"] or survived

    by_rep = {}
    for rep in [1,4,16,64]:
        group = [x for x in per_canary.values() if x["n_reps"] == rep]
        by_rep[str(rep)] = {
            "n_canaries": len(group),
            "canaries_with_any_survival": sum(int(x["any_survival"]) for x in group),
            "total_rows": sum(x["rows"] for x in group),
            "survivor_rows": sum(x["survivor_rows"] for x in group),
        }

    changed = sum(r["text"] != s["text"] for r,s in zip(rows, san_rows))
    audit = {
        "n_rows": len(rows),
        "n_distinct_doc_ids": len({x["doc_id"] for x in rows}),
        "metadata_preserved": True,
        "row_order_preserved": True,
        "decode_config": {
            "base_model": a.base_model,
            "adapter_path": a.adapter_path,
            "do_sample": False,
            "max_new_tokens": a.max_new_tokens,
            "seed": a.seed,
            "prompt_source": "src/sanitize_eval.py::build_messages",
        },
        "n_changed_text_rows": changed,
        "n_unchanged_text_rows": len(rows)-changed,
        "n_hit_max_new_tokens": len(hit_cap_rows),
        "hit_max_new_tokens_rate": len(hit_cap_rows)/len(rows),
        "hit_cap_rows": hit_cap_rows,
        "repeated_identical_input_consistency": {
            "pass": len(repeat_fail) == 0,
            "n_failures": len(repeat_fail),
            "failure_doc_ids": repeat_fail,
        },
        "canary_survival": {
            "n_canaries": len(per_canary),
            "canaries_with_any_survival": sum(int(x["any_survival"]) for x in per_canary.values()),
            "total_canary_rows": sum(x["rows"] for x in per_canary.values()),
            "survivor_rows": sum(x["survivor_rows"] for x in per_canary.values()),
            "by_n_reps": by_rep,
            "per_canary": list(per_canary.values()),
        },
        "input_sha256": sha256(a.input_file),
        "output_sha256": sha256(out_path),
        "wallclock_s": time.perf_counter() - t0_all,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(shard_dir)

    print("\n" + "="*80)
    print("SANITIZATION COMPLETE")
    print("="*80)
    print("Rows raw/sanitized :", len(rows), "/", len(san_rows))
    print("Metadata preserved : PASS")
    print("Row order preserved: PASS")
    print("Repeated consistency:", "PASS" if not repeat_fail else "FAIL")
    print("Changed text rows  :", changed)
    print("Hit max_new_tokens :", len(hit_cap_rows))
    print("Canaries with any surviving secret:",
          audit["canary_survival"]["canaries_with_any_survival"], "/ 20")
    print("Canary rows with surviving secret:",
          audit["canary_survival"]["survivor_rows"], "/",
          audit["canary_survival"]["total_canary_rows"])
    print("Output:", out_path)
    print("Audit :", audit_path)
    print("Output SHA256:", audit["output_sha256"])

if __name__ == "__main__":
    main()
