#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 EVAL_JSONL OUTPUT_JSON KOTHA_MODEL LILTII_MODEL TRAINED_DIR" >&2
  exit 1
fi

EVAL_DATA="$1"
OUTPUT_JSON="$2"
KOTHA_MODEL="$3"
LILTII_MODEL="$4"
TRAINED_DIR="$5"

python scripts/evaluate_tokenizers.py \
  --eval-data "$EVAL_DATA" \
  --trained-dir "$TRAINED_DIR" \
  --sp-model "Kotha-1-32K=$KOTHA_MODEL" \
  --sp-model "LilTii-v0.2=$LILTII_MODEL" \
  --hf-model "Qwen-2.5=Qwen/Qwen2.5-0.5B" \
  --hf-model "BLOOM-560m=bigscience/bloom-560m" \
  --hf-model "TinyLlama=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" \
  --hf-model "Phi-2=microsoft/phi-2" \
  --hf-model "StableLM-2=stabilityai/stablelm-2-1_6b" \
  --hf-model "BanglaBERT=csebuetnlp/banglabert" \
  --hf-model "BanglaT5=csebuetnlp/banglat5" \
  --output "$OUTPUT_JSON" \
  --max-docs 3000 \
  --min-chars 200
