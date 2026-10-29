#!/usr/bin/env python
"""
Experiment 1 – GPT-2 (124M) fine-tuning on E2E NLG under DP.

Reproduces Table 1 / Figure 1 of the ACUN paper.

Author: Nitish Kumar Roy <Nitishkumarrpy526@gmail.com>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm.auto import tqdm

# Make the package importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acun import ACUNPrivacyEngine
from acun.utils import set_seed, save_json, get_device, count_parameters


def parse_args():
    p = argparse.ArgumentParser(description="ACUN – GPT-2 on E2E NLG")
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--epsilon", type=float, default=3.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--median-budget-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="runs/e2e_gpt2")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="For quick debugging; None = full set")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[ACUN] Device: {device}")
    print(f"[ACUN] Target (ε, δ) = ({args.epsilon}, {args.delta})")

    # ------------------------------------------------------------------
    # Data – E2E NLG
    # ------------------------------------------------------------------
    raw = load_dataset("e2e_nlg", split="train")
    if args.max_train_samples:
        raw = raw.select(range(min(args.max_train_samples, len(raw))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        # Simple concatenation of meaning representation + reference
        texts = [
            f"MR: {mr} REF: {ref}"
            for mr, ref in zip(examples["meaning_representation"], examples["human_reference"])
        ]
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized = raw.map(preprocess, batched=True, remove_columns=raw.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print(f"[ACUN] Trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # Privacy engine (ACUN)
    # ------------------------------------------------------------------
    engine = ACUNPrivacyEngine(
        module=model,
        batch_size=args.batch_size,
        sample_size=len(tokenized),
        epochs=args.epochs,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        median_budget_fraction=args.median_budget_fraction,
        max_grad_norm=args.max_grad_norm,
    )

    model, optimizer, train_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=args.max_grad_norm,
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    global_step = 0
    history = []

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            # Optional: collect norms for adaptive logic / logging
            # (In a full production integration one would replace Opacus clipping)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        eps_spent = engine.get_epsilon()
        history.append(
            {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "epsilon": eps_spent,
            }
        )
        print(f"  → Epoch {epoch+1} | loss={avg_loss:.4f} | ε≈{eps_spent:.3f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    final_eps = engine.get_epsilon()
    metrics = {
        "method": "ACUN",
        "model": args.model_name,
        "dataset": "e2e_nlg",
        "epsilon_target": args.epsilon,
        "epsilon_spent": final_eps,
        "delta": args.delta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "final_train_loss": history[-1]["avg_loss"] if history else None,
        "history": history,
        "clipper_state": engine.get_clipper_state(),
    }
    save_json(metrics, Path(args.output_dir) / "metrics.json")
    print(f"\n[ACUN] Finished. Spent ε ≈ {final_eps:.3f}. Results → {args.output_dir}")


if __name__ == "__main__":
    main()
