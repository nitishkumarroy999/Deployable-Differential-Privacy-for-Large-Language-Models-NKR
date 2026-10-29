# ACUN: Adaptive Clipping with Utility-Aware Noise for Deployable DP-LLMs

**Author:** Nitish Kumar Roy  
**Email:** Nitishkumarrpy526@gmail.com  
**Paper:** *Deployable Differential Privacy for Large Language Models: Adaptive Gradient Clipping with Formal Utility Guarantees*  
**Venue:** 5th Workshop on Deployable AI (DAI 2027), AAAI-27, Montreal, Canada

---

## Overview

ACUN is a practical, drop-in replacement for standard DP-SGD that **jointly optimizes the clipping threshold and noise scale** online. It maintains a differentially private estimate of the median gradient norm and derives a closed-form noise multiplier that minimizes a local quadratic approximation of excess risk.

**Key results from the paper (A100 80GB, 5 seeds):**

| Task | Model | Metric | ε=1 Gain over best baseline |
|------|-------|--------|-----------------------------|
| E2E NLG | GPT-2 (124M) | ROUGE-L | +3.1 pts |
| Clinical QA | LLaMA-3-8B + LoRA | Exact Match | +2.6 pts |
| Sentiment | DistilBERT | Macro-F1 | +2.3 pts |

Wall-clock training time is reduced by **15–19%** with virtually identical peak memory.

---

## Installation

```bash
git clone https://github.com/<your-username>/acun-dp-llm.git
cd acun-dp-llm

# Recommended: create a fresh environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, CUDA 12.x recommended (tested on A100 80GB).

---

## Quick Start

### 1. GPT-2 on E2E NLG (public benchmark)

```bash
python experiments/run_e2e_gpt2.py \
  --epsilon 3.0 \
  --delta 1e-5 \
  --epochs 5 \
  --batch-size 16 \
  --output-dir runs/e2e_eps3
```

### 2. LLaMA-3-8B + LoRA on clinical data (simulated private setting)

```bash
python experiments/run_clinical_llama.py \
  --epsilon 3.0 \
  --lora-rank 16 \
  --epochs 3 \
  --output-dir runs/clinical_eps3
```

> **Note:** The real MIMIC-III derived set used in the paper is not redistributed.  
> The script falls back to a public clinical-style instruction dataset (or a synthetic proxy) so that the pipeline remains fully runnable.

### 3. DistilBERT sentiment classification

```bash
python experiments/run_sentiment_distilbert.py \
  --epsilon 3.0 \
  --epochs 4 \
  --batch-size 64 \
  --output-dir runs/sentiment_eps3
```

---

## Core Algorithm (ACUN)

The heart of the method lives in `acun/adaptive_clipping.py`:

```python
from acun import ACUNPrivacyEngine

engine = ACUNPrivacyEngine(
    module=model,
    batch_size=batch_size,
    sample_size=len(train_dataset),
    epochs=epochs,
    target_epsilon=3.0,
    target_delta=1e-5,
    median_budget_fraction=0.05,   # only extra hyper-parameter
)

optimizer = engine.make_private(optimizer)
```

ACUN automatically:
1. Maintains a private median estimate of per-example gradient norms (exponential mechanism).
2. Allocates remaining Rényi budget.
3. Sets the optimal clipping threshold \(C_t^\star\) and noise multiplier \(\sigma_t^\star\) via the closed-form solution derived in the paper.

---

## Repository Structure

```
acun-dp-llm/
├── acun/                       # Core library
│   ├── __init__.py
│   ├── adaptive_clipping.py    # ACUN logic + private median estimator
│   ├── privacy_engine.py       # Thin Opacus-compatible wrapper
│   └── utils.py
├── experiments/
│   ├── run_e2e_gpt2.py
│   ├── run_clinical_llama.py
│   └── run_sentiment_distilbert.py
├── configs/
│   ├── e2e_gpt2.yaml
│   ├── clinical_llama.yaml
│   └── sentiment_distilbert.yaml
├── scripts/
│   └── reproduce_tables.sh     # Reproduce main tables from the paper
├── requirements.txt
└── README.md
```

---

## Reproducing Paper Results

```bash
# Full reproduction script (runs all three tasks for ε ∈ {1,3,8})
bash scripts/reproduce_tables.sh
```

Results are written to `runs/` and summarized in `runs/summary.csv`.  
Expected runtime on one A100 80GB: ~6–8 hours for the full suite.

---

## Citation

If you use this code or the ACUN method, please cite:

```bibtex
@inproceedings{roy2027acun,
  title     = {Deployable Differential Privacy for Large Language Models: 
               Adaptive Gradient Clipping with Formal Utility Guarantees},
  author    = {Roy, Nitish Kumar},
  booktitle = {5th Workshop on Deployable AI (DAI 2027) at AAAI-27},
  year      = {2027},
  address   = {Montreal, Canada}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This implementation builds on [Opacus](https://opacus.ai/), [Hugging Face Transformers](https://huggingface.co/docs/transformers), and [PEFT](https://github.com/huggingface/peft).  
The author thanks the open-source communities that made real-world private LLM experiments possible.

---

**Contact:** Nitish Kumar Roy — Nitishkumarrpy526@gmail.com
