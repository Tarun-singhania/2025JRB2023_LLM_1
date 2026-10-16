#!/usr/bin/env python3
"""Single-GPU, token-weighted held-out PPL for V-base/V-raw/V-san.

This avoids distributed DataLoader padding/duplication and evaluates all three
models on exactly the same held-out token set.
"""
import argparse, json, math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import attack_extract as ae


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--heldout", default="data/partC_heldout.jsonl")
    ap.add_argument("--raw_adapter", default="adapters/victim_raw")
    ap.add_argument("--san_adapter", default="adapters/victim_san")
    ap.add_argument("--output", default="results/partC_utility.json")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rows = read_jsonl(args.heldout)
    if len(rows) != 500:
        raise ValueError(f"Expected 500 held-out rows, got {len(rows)}")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    encoded = []
    for row in rows:
        ids = tok(
            str(row["text"]), add_special_tokens=False, truncation=True,
            max_length=max(1, args.max_length - 1)
        )["input_ids"]
        if tok.eos_token_id is not None:
            ids = ids[:args.max_length-1] + [tok.eos_token_id]
        else:
            ids = ids[:args.max_length]
        encoded.append(ids)

    def collate(batch):
        L = max(map(len, batch))
        ii, am, yy = [], [], []
        for ids in batch:
            n = L - len(ids)
            ii.append(ids + [tok.pad_token_id] * n)
            am.append([1] * len(ids) + [0] * n)
            yy.append(ids + [-100] * n)
        return {
            "input_ids": torch.tensor(ii, dtype=torch.long),
            "attention_mask": torch.tensor(am, dtype=torch.long),
            "labels": torch.tensor(yy, dtype=torch.long),
        }

    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    def evaluate(adapter_path, name):
        _tok, model = ae.load_model(args.model, adapter_path, args.device)
        total_nll = 0.0
        total_tokens = 0
        with torch.inference_mode():
            for b in loader:
                ids = b["input_ids"].to(args.device)
                mask = b["attention_mask"].to(args.device)
                labels = b["labels"].to(args.device)
                out = model(input_ids=ids, attention_mask=mask, use_cache=False)
                logits = out.logits[:, :-1, :].float()
                targets = labels[:, 1:]
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1), ignore_index=-100, reduction="sum"
                )
                ntok = (targets != -100).sum()
                total_nll += float(nll.item())
                total_tokens += int(ntok.item())
        mean_nll = total_nll / total_tokens
        ppl = math.exp(mean_nll)
        del model
        torch.cuda.empty_cache()
        print(name, "tokens=", total_tokens, "ppl=", ppl)
        return {"token_count": total_tokens, "ppl": ppl}

    result = {
        "v_base": evaluate(None, "V-base"),
        "v_raw": evaluate(args.raw_adapter, "V-raw"),
        "v_san": evaluate(args.san_adapter, "V-san"),
    }
    counts = {v["token_count"] for v in result.values()}
    if len(counts) != 1:
        raise RuntimeError(f"Held-out token counts differ: {counts}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("ALL THREE PPL TOKEN COUNTS MATCH: PASS")


if __name__ == "__main__":
    main()
