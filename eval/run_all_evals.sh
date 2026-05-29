#!/usr/bin/env bash
# Run zero-shot + fine-tuned evaluation for all 3 models.
set -e
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.hf_cache"

MODELS=(
  "HuggingFaceTB/SmolLM2-360M smollm2"
  "Qwen/Qwen2.5-1.5B qwen25"
  "google/gemma-4-E2B gemma4"
)

ADAPTERS=(
  "models/smollm2_360m_qlora/final_adapter"
  "models/qwen25_15b_qlora/final_adapter"
  "models/gemma4_e2b_qlora/final_adapter"
)

for i in "${!MODELS[@]}"; do
  read -r MODEL_ID TAG <<<"${MODELS[$i]}"
  ADAPTER="${ADAPTERS[$i]}"

  echo ""
  echo "=== [$((i+1))/3] ZERO-SHOT: $TAG ==="
  .venv/bin/python eval/evaluate.py --model_id "$MODEL_ID" --tag "${TAG}_zeroshot" || echo "[!] zeroshot $TAG failed"

  if [ -d "$ADAPTER" ]; then
    echo ""
    echo "=== [$((i+1))/3] FT: $TAG ==="
    .venv/bin/python eval/evaluate.py --model_id "$MODEL_ID" --adapter "$ADAPTER" --tag "${TAG}_ft" || echo "[!] ft $TAG failed"
  else
    echo "[!] no adapter for $TAG at $ADAPTER"
  fi
done

echo ""
echo "=== All evals done ==="
ls -la eval/results/
