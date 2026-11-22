"""
Evaluate a trained checkpoint on the validation set.

c: uses the same data split and vocabulary as training (training/data.py) and the evaluation
   function of training/train.py, so the numbers are computed exactly the same way.

    python -m training.evaluate --model slstm --dataset copy
"""
import argparse
import json
import os

import torch

from models import build
from training.data import load_dataset_split
from training.train import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["slstm", "mlstm", "xlstm_sm", "xlstm_ms"], required=True)
    p.add_argument("--dataset", choices=["python", "shakespeare", "dyck", "copy"], required=True)
    p.add_argument("--results", default="results")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--backend", choices=["cuda", "vanilla"], default="vanilla")
    args = p.parse_args()

    # c: model settings are read from results.json, so they always match the checkpoint
    run_dir = os.path.join(args.results, args.model, args.dataset)
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as f:
        cfg = json.load(f)["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_ids, vocab = load_dataset_split(args.dataset)
    model = build(args.model, len(vocab), cfg["seq_len"], cfg["embed_dim"], cfg["num_heads"], args.backend)
    model.load_state_dict(torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location=device))
    model.to(device)

    class A:  # the settings evaluate() needs
        batch, seq_len = cfg["batch"], cfg["seq_len"]

    generator = torch.Generator().manual_seed(cfg["seed"])
    m = evaluate(model, val_ids, A, device, generator, args.steps)
    print(f"[eval] {args.model} on {args.dataset}: loss {m['loss']:.4f} | ppl {m['ppl']:.3f} | "
          f"bpc {m['bpc']:.4f} | acc {m['accuracy']:.2%}")


if __name__ == "__main__":
    main()
