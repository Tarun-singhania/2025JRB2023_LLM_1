#!/usr/bin/env bash
set -euo pipefail

# Full reproducibility pipeline for Assignment 1.
# Run from the project root after placing the professor-supplied data/ files.
# Reference setup: 2x NVIDIA T4; seed 42.

MODEL_SAN="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_VIC="Qwen/Qwen2.5-0.5B"
SEED=42
mkdir -p adapters results/detail results/run_metrics logs

# ---------------------------------------------------------------------------
# Part A: A1-A6 PEFT sweep
# ---------------------------------------------------------------------------
COMMON=(
  --model "$MODEL_SAN"
  --train_file data/partA_train.jsonl
  --val_file data/partA_validation.jsonl
  --lora_dropout 0.05
  --seed "$SEED"
  --results_csv results/partA_sweep.csv
  --epochs 2
  --learning_rate 2e-4
  --per_device_train_batch_size 4
  --per_device_eval_batch_size 8
  --gradient_accumulation_steps 2
  --expected_global_effective_batch_size 16
  --max_length 768
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --generation_batch_size 8
  --replace_existing_row
)

runA () {
  accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 \
    src/peft_sweep.py "${COMMON[@]}" \
    --target_modules "$2" --lora_r "$3" --lora_alpha "$4" --quant "$5" \
    --run_id "$1" --output_dir "adapters/$1" 2>&1 | tee "logs/$1.log"
}

runA A1 "q_proj,v_proj" 8 16 none
runA A2 "q_proj,k_proj,v_proj,o_proj" 8 16 none
runA A3 "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" 8 16 none
runA A4 "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" 32 64 none
runA A5 "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" 32 16 none
runA A6 "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" 32 64 4bit
python src/make_partA_fig.py

# A3 is the frozen Part-B/Part-C sanitizer checkpoint.
rm -rf adapters/best_sanitizer
cp -a adapters/A3 adapters/best_sanitizer

# ---------------------------------------------------------------------------
# Part B: ID + baselines + OOD
# ---------------------------------------------------------------------------
rm -f results/partB_metrics.json
python src/sanitize_eval.py --base_model "$MODEL_SAN" --adapter_path adapters/best_sanitizer \
  --test_file data/partB_test.jsonl --system finetuned \
  --output_predictions results/partB_predictions.jsonl --report_file results/partB_metrics.json \
  --batch_size 8 --max_new_tokens 260 --seed "$SEED" --device cuda:0
python src/sanitize_eval.py --base_model "$MODEL_SAN" --adapter_path adapters/best_sanitizer \
  --test_file data/partB_test.jsonl --system regex \
  --output_predictions results/detail/partB_predictions_regex.jsonl --report_file results/partB_metrics.json --seed "$SEED"
python src/sanitize_eval.py --base_model "$MODEL_SAN" --adapter_path adapters/best_sanitizer \
  --test_file data/partB_test.jsonl --system presidio \
  --output_predictions results/detail/partB_predictions_presidio.jsonl --report_file results/partB_metrics.json --seed "$SEED"

for SYS in finetuned regex presidio; do
  python src/sanitize_eval.py --base_model "$MODEL_SAN" --adapter_path adapters/best_sanitizer \
    --test_file data/partB_ood_test.jsonl --system "$SYS" \
    --output_predictions "results/detail/partB_ood_predictions_${SYS}.jsonl" \
    --report_file results/partB_metrics.json --batch_size 8 --max_new_tokens 260 --seed "$SEED" --device cuda:0
done

# ---------------------------------------------------------------------------
# Part C.1: sanitize full corpus, train identical victims, utility
# ---------------------------------------------------------------------------
rm -f data/partC_corpus_sanitized.jsonl results/partC_sanitization_audit.json
accelerate launch --num_processes 2 --num_machines 1 --multi_gpu --mixed_precision no --dynamo_backend no \
  src/sanitize_corpus.py --base_model "$MODEL_SAN" --adapter_path adapters/best_sanitizer \
  --input_file data/partC_corpus_raw.jsonl --output_file data/partC_corpus_sanitized.jsonl \
  --canaries_file data/partC_canaries.jsonl --audit_file results/partC_sanitization_audit.json \
  --batch_size 8 --max_new_tokens 260 --seed "$SEED" --overwrite

rm -rf adapters/victim_raw adapters/victim_san
for WHICH in raw san; do
  if [[ "$WHICH" == "raw" ]]; then CORPUS=data/partC_corpus_raw.jsonl; else CORPUS=data/partC_corpus_sanitized.jsonl; fi
  accelerate launch --num_processes 2 --num_machines 1 --multi_gpu --mixed_precision fp16 --dynamo_backend no \
    src/train_victim.py --model "$MODEL_VIC" --corpus "$CORPUS" --heldout data/partC_heldout.jsonl \
    --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --epochs 4 --lr 2e-4 \
    --max_length 512 --per_device_batch_size 4 --gradient_accumulation_steps 2 --seed "$SEED" \
    --output_dir "adapters/victim_${WHICH}"
done

CUDA_VISIBLE_DEVICES=0 python src/eval_partC_utility.py
python src/build_partC_pii_index.py

# ---------------------------------------------------------------------------
# Part C.2: all four attacks against V-base, V-raw, V-san
# ---------------------------------------------------------------------------
rm -f results/detail/partC_attacks_base.json results/detail/partC_attacks_raw.json results/detail/partC_attacks_san.json

run_attack () {
  local NAME="$1"; local ADAPTER="$2"; local CORPUS="$3"; local OUT="$4"
  CUDA_VISIBLE_DEVICES=0 python src/attack_extract.py \
    --victim_model "$MODEL_VIC" --adapter_path "$ADAPTER" --ref_model "$MODEL_VIC" \
    --corpus "$CORPUS" --heldout data/partC_heldout.jsonl --canaries data/partC_canaries.jsonl \
    --pii_index data/partC_pii_index.jsonl --targeted_file data/partC_targeted.jsonl \
    --attack all --num_samples 5000 --seed "$SEED" --victim_name "$NAME" --report_file "$OUT"
}

run_attack v_base none data/partC_corpus_raw.jsonl results/detail/partC_attacks_base.json
run_attack v_raw adapters/victim_raw data/partC_corpus_raw.jsonl results/detail/partC_attacks_raw.json
run_attack v_san adapters/victim_san data/partC_corpus_sanitized.jsonl results/detail/partC_attacks_san.json

# C.3 redacted trace and final aggregate/plot.
CUDA_VISIBLE_DEVICES=0 python src/trace_partC_failures.py
python src/finalize_partC_results.py

echo "============================================================"
echo "RUN_ALL COMPLETE"
echo "============================================================"
