# Assignment 1 — Fine-Tune, Sanitize, Extract

ELL8299/COL8383/COL873: Trustworthy Large Language Models, Semester I 2026–27.

**Student:** Tarun Kumar  
**Enrollment No.:** 2025JRB2023  
**Department:** Robotics (M.Tech)

This ZIP is the final submission bundle. It contains the required source files, final result artifacts, the <=5-page PDF report, reproducibility scripts. The professor-supplied `data/` corpus is intentionally **not** included.

## Required submission files

- `report.pdf` — final 5-page report; extracted PII is redacted.
- `requirements.txt`
- `run_all.sh`
- `README.md`
- `src/peft_sweep.py`
- `src/sanitize_eval.py`
- `src/train_victim.py`
- `src/attack_extract.py`
- `results/partA_sweep.csv`
- `results/partA_fig.png` (PDF copy also included)
- `results/partB_metrics.json`
- `results/partB_predictions.jsonl`
- `results/partC_attacks.json`
- `results/partC_exposure.png`

Additional reproducibility/audit files are included under `src/`, `results/`, and `adapters_metadata/`.

## Data

Place the professor-provided dataset bundle under:

```text
data/
  partA_train.jsonl
  partA_validation.jsonl
  partB_test.jsonl
  partB_ood_test.jsonl
  partC_corpus_raw.jsonl
  partC_heldout.jsonl
  partC_canaries.jsonl
  partC_targeted.jsonl
```

Do not resplit, resample, or augment those files.

`data/partC_pii_index.jsonl` was not present in the supplied bundle used for the experiments. `src/build_partC_pii_index.py` reconstructs the documented detector-derived Regex ∪ Presidio index used for untargeted PII-recovery evaluation. Its limitation is recorded in the report and result audit.

## Models and final choices

- Part A/B sanitizer base: `Qwen/Qwen2.5-1.5B-Instruct`
- Selected sanitizer: **A3** (all attention + MLP projections, LoRA r=8, alpha=16). It was selected for the lowest validation leak rate with strong F1 and substantially fewer trainable parameters than rank-32 alternatives.
- Part C victim base: `Qwen/Qwen2.5-0.5B` **base LM**, not Instruct.
- V-raw and V-san: identical manual-LoRA settings (r=32, alpha=64, dropout=0.05, 4 epochs); only the training text differs.

Final token-weighted held-out PPL was recomputed on one GPU for all three models over exactly the same 64,958 prediction tokens:

- V-base: 14.1747098877
- V-raw: 8.7782664714
- V-san: 14.6109408023

The single-GPU utility recomputation avoids distributed evaluation padding/duplication and is implemented in `src/eval_partC_utility.py`.

## Reproduction

Reference experiment environment was Kaggle/Colab with NVIDIA T4 GPUs. Install dependencies and the English spaCy model used by Presidio, then run from the repository root:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
bash run_all.sh
```

`run_all.sh` reproduces the A1–A6 sweep, copies A3 as the selected sanitizer, runs Part B ID/OOD evaluations and baselines, sanitizes the complete Part-C corpus, trains V-raw/V-san, recomputes utility, constructs the local PII index, runs all four attacks against V-base/V-raw/V-san, reproduces the redacted sanitizer-failure traces, and writes the final Part-C comparison/plot.

The complete run is computationally expensive. The submitted `results/` directory contains the completed authoritative outputs.

## Adapter artifacts

Adapter weights are not stored in this repository. If `adapters/<name>` is missing, the scripts download it from the Hugging Face Hub.

| CLI path | Hugging Face repo |
| --- | --- |
| `adapters/A1` | https://huggingface.co/TarunExhaustice/llm-a1-a1 |
| `adapters/A2` | https://huggingface.co/TarunExhaustice/llm-a1-a2 |
| `adapters/A3` | https://huggingface.co/TarunExhaustice/llm-a1-a3 |
| `adapters/A4` | https://huggingface.co/TarunExhaustice/llm-a1-a4 |
| `adapters/A5` | https://huggingface.co/TarunExhaustice/llm-a1-a5 |
| `adapters/A6` | https://huggingface.co/TarunExhaustice/llm-a1-a6 |
| `adapters/best_sanitizer` | https://huggingface.co/TarunExhaustice/llm-a1-best-sanitizer |
| `adapters/victim_raw` | https://huggingface.co/TarunExhaustice/llm-a1-victim-raw |
| `adapters/victim_san` | https://huggingface.co/TarunExhaustice/llm-a1-victim-san |

Part-A configs and the weight manifest remain in `adapters_metadata/`.

## Important result notes

- `results/partB_metrics.json` contains corrected **A3** ID/OOD results and Regex/Presidio baselines.
- `results/partB_predictions.jsonl` is the corrected A3 in-distribution prediction file required by the assignment.
- `results/partC_attacks.json` includes assignment-compatible top-level V-raw keys and an additional `models` object containing V-base, V-raw, and V-san results.
- Canary exposure uses 9,999 alternatives plus the true secret; the sampled-rank estimator ceiling is ~13.29 bits.
- The C.3 traced-leak artifact stores only IDs/labels/boolean outcomes; PII values are `[REDACTED]`.

## Ethics

Only assignment-provided/synthetic/public data and models fine-tuned for this assignment were attacked. No third-party or production service was attacked. Raw Part-C extraction generations are not included in this ZIP.
