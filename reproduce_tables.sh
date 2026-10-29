#!/usr/bin/env bash
# Reproduce the main experimental tables from the ACUN paper.
# Author: Nitish Kumar Roy <Nitishkumarrpy526@gmail.com>
#
# Usage:
#   bash scripts/reproduce_tables.sh
#
# Requires a machine with a modern NVIDIA GPU (tested on A100 80GB).
# Full suite takes roughly 6–8 hours.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "============================================================"
echo " ACUN – Reproducing paper tables"
echo " Author: Nitish Kumar Roy"
echo "============================================================"

EPSILONS=(1.0 3.0 8.0)

# ---------- E2E GPT-2 ----------
for eps in "${EPSILONS[@]}"; do
  echo ""
  echo ">>> E2E GPT-2 | ε=${eps}"
  python experiments/run_e2e_gpt2.py \
    --epsilon "$eps" \
    --epochs 5 \
    --batch-size 16 \
    --output-dir "runs/e2e_eps${eps}"
done

# ---------- Sentiment DistilBERT ----------
for eps in "${EPSILONS[@]}"; do
  echo ""
  echo ">>> Sentiment DistilBERT | ε=${eps}"
  python experiments/run_sentiment_distilbert.py \
    --epsilon "$eps" \
    --epochs 4 \
    --batch-size 64 \
    --output-dir "runs/sentiment_eps${eps}"
done

# ---------- Clinical LLaMA (optional – needs HF token & large GPU) ----------
# Uncomment if you have access to Meta-Llama-3-8B and sufficient VRAM
# for eps in "${EPSILONS[@]}"; do
#   echo ""
#   echo ">>> Clinical LLaMA-3-8B LoRA | ε=${eps}"
#   python experiments/run_clinical_llama.py \
#     --epsilon "$eps" \
#     --epochs 3 \
#     --output-dir "runs/clinical_eps${eps}"
# done

echo ""
echo "============================================================"
echo " Done. Metrics written under runs/"
echo "============================================================"
