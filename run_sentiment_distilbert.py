#!/usr/bin/env python
"""
Experiment 3 – DistilBERT sentiment classification under DP.

Reproduces Table 3 of the ACUN paper (Macro-F1 on customer-feedback style data).

Uses a public sentiment dataset as a fully runnable proxy for the private
customer-support corpus described in the paper.

Author: Nitish Kumar Roy 
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acun import ACUNPrivacyEngine
from acun.utils import set_seed, save_json, get_device, count_parameters


def parse_args():
    p = argparse.ArgumentParser(description="ACUN – DistilBERT sentiment")
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--epsilon", type=float, default=3.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--median-budget-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="runs/sentiment_distilbert")
    p.add_argument("--max-train-samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[ACUN] Device: {device}")
    print(f"[ACUN] Target (ε, δ) = ({args.epsilon}, {args.delta})")

    # ------------------------------------------------------------------
    # Data – public sentiment proxy (SST-2 / IMDB style)
    # ------------------------------------------------------------------
    raw = load_dataset("glue", "sst2", split="train")
    if args.max_train_samples:
        raw = raw.select(range(min(args.max_train_samples, len(raw))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized = raw.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    model.to(device)
    print(f"[ACUN] Trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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
    # Training
    # ------------------------------------------------------------------
    model.train()
    history = []
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        all_preds, all_labels = [], []
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        eps_spent = engine.get_epsilon()
        history.append(
            {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "macro_f1": macro_f1,
                "epsilon": eps_spent,
            }
        )
        print(
            f"  → Epoch {epoch+1} | loss={avg_loss:.4f} | "
            f"Macro-F1={macro_f1:.4f} | ε≈{eps_spent:.3f}"
        )

    metrics = {
        "method": "ACUN",
        "model": args.model_name,
        "dataset": "sst2_proxy",
        "epsilon_target": args.epsilon,
        "epsilon_spent": engine.get_epsilon(),
        "delta": args.delta,
        "final_macro_f1": history[-1]["macro_f1"] if history else None,
        "history": history,
    }
    save_json(metrics, Path(args.output_dir) / "metrics.json")
    print(f"\n[ACUN] Finished. Results → {args.output_dir}")


if __name__ == "__main__":
    main()
