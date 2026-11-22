"""
Train one of the authors' official xLSTM models (library `xlstm`) on a character-level dataset.

What happens:
    1. load the dataset with a train / validation split (see training/data.py)
    2. build the model: sLSTM, mLSTM, xLSTM sLSTM->mLSTM or xLSTM mLSTM->sLSTM (see models/)
    3. train with AdamW on random text pieces, evaluate on the validation text after every epoch
    4. save to results/<model>/<dataset>/:
         results.json   all settings, the vocabulary, the loss per epoch and the final validation metrics
         checkpoint.pt  the trained weights (used by the introspection scripts)
         loss.png       train and validation loss per epoch

c: rewritten. Before: only the training loss was logged (no validation, so overfitting was invisible),
   logs of several runs were appended to the same file, there was no seed, and one epoch went over
   every overlapping window of the text (~17,000 steps for Shakespeare).

Examples:
    python -m training.train --model slstm --dataset copy
    python -m training.train --model xlstm_sm --dataset dyck --epochs 10
    python -m training.train --model slstm --dataset shakespeare --backend cuda   # Linux + CUDA only
"""
import argparse
import json
import math
import os
import platform

import torch
import torch.nn as nn

from models import build
from training.data import get_batch, load_dataset_split


@torch.no_grad()
def evaluate(model, data, args, device, generator, steps):
    """Average loss, perplexity, bits per character and accuracy on random pieces of `data`."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, tokens = 0.0, 0, 0
    for _ in range(steps):
        x, y = get_batch(data, args.batch, args.seq_len, device, generator)
        logits = model(x)  # [B, T, V]
        total_loss += crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        correct += (logits.argmax(-1) == y).sum().item()
        tokens += y.numel()
    model.train()
    loss = total_loss / tokens
    return {"loss": loss, "ppl": math.exp(loss), "bpc": loss / math.log(2), "accuracy": correct / tokens}


def plot(history, path, title):
    import matplotlib
    matplotlib.use("Agg")  # save the image without opening a window
    import matplotlib.pyplot as plt
    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [r["train_loss"] for r in history], marker="o", markersize=4, label="train")
    plt.plot(epochs, [r["val_loss"] for r in history], marker="o", markersize=4, label="validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["slstm", "mlstm", "xlstm_sm", "xlstm_ms"], required=True)
    p.add_argument("--dataset", choices=["python", "shakespeare", "dyck", "copy"], required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps_per_epoch", type=int, default=None,
                   help="n: default = one pass over the training characters")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0, help="n: gradient clipping (max norm), 0 = off")
    p.add_argument("--seed", type=int, default=0)  # n: reproducible runs
    p.add_argument("--eval_steps", type=int, default=50)
    # n: sLSTM backend. "cuda" = fast fused kernels, compiled on first use (needs Linux + CUDA toolkit).
    #    "vanilla" = same math in plain PyTorch, works everywhere. "auto" = cuda on Linux with a GPU.
    p.add_argument("--backend", choices=["auto", "cuda", "vanilla"], default="auto")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    if args.backend == "auto":
        args.backend = "cuda" if torch.cuda.is_available() and platform.system() == "Linux" else "vanilla"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # the seed fixes the initial weights and which batches are drawn
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    train_ids, val_ids, vocab = load_dataset_split(args.dataset)
    model = build(args.model, len(vocab), args.seq_len, args.embed_dim, args.num_heads, args.backend).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] {args.model} on {args.dataset}: vocab {len(vocab)}, {n_params:,} parameters, "
          f"device {device}, sLSTM backend {args.backend}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    steps = args.steps_per_epoch or max(1, len(train_ids) // (args.batch * args.seq_len))

    history = []
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for _ in range(steps):
            x, y = get_batch(train_ids, args.batch, args.seq_len, device, generator)
            logits = model(x)  # [B, T, V]: a score for every possible next character
            loss = crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss):  # n: stop right away instead of training on with NaN
                raise RuntimeError(f"Loss became {loss.item()} in epoch {epoch}")
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += loss.item()

        # after every epoch: check on the validation text, which the model never trains on
        val = evaluate(model, val_ids, args, device, generator, args.eval_steps)
        history.append({"epoch": epoch, "train_loss": total / steps, **{f"val_{k}": v for k, v in val.items()}})
        print(f"epoch {epoch:2d} | train loss {total / steps:.4f} | val loss {val['loss']:.4f} | "
              f"val ppl {val['ppl']:.3f} | val acc {val['accuracy']:.2%}")

    final = evaluate(model, val_ids, args, device, generator, steps=200)
    print(f"\nFINAL (validation) loss {final['loss']:.4f} | ppl {final['ppl']:.3f} | "
          f"bpc {final['bpc']:.4f} | acc {final['accuracy']:.2%}")

    # c: one folder per run, nothing is appended to old logs
    out_dir = os.path.join(args.out, args.model, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "dataset": args.dataset, "params": n_params, "config": vars(args),
                   "vocab": vocab, "history": history, "final_val": final}, f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "checkpoint.pt"))
    plot(history, os.path.join(out_dir, "loss.png"), f"{args.model} on {args.dataset}")
    print(f"[ok] saved results.json, checkpoint.pt and loss.png to {out_dir}")


if __name__ == "__main__":
    main()
