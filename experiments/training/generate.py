"""
Generate text with a trained checkpoint, one character at a time.

    python -m training.generate --model slstm --dataset shakespeare --seed_text "ROMEO:"
"""
import argparse
import json
import os

import torch

from models import build
from training.data import load_dataset_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["slstm", "mlstm", "xlstm_sm", "xlstm_ms"], required=True)
    p.add_argument("--dataset", choices=["dyck", "copy", "shakespeare", "python"], required=True)
    p.add_argument("--results", default="results")
    p.add_argument("--seed_text", default="(", help="text to start from")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--backend", choices=["cuda", "vanilla"], default="vanilla")
    args = p.parse_args()

    # c: model settings and vocabulary come from results.json, so they always match the checkpoint
    run_dir = os.path.join(args.results, args.model, args.dataset)
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as f:
        saved = json.load(f)
    cfg, vocab = saved["config"], saved["vocab"]
    stoi = {ch: i for i, ch in enumerate(vocab)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build(args.model, len(vocab), cfg["seq_len"], cfg["embed_dim"], cfg["num_heads"], args.backend)
    model.load_state_dict(torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location=device))
    model.to(device).eval()

    ids = [stoi[ch] for ch in args.seed_text]
    with torch.no_grad():
        for _ in range(args.steps):
            # f: the model reads the whole text generated so far (last seq_len characters).
            #    Before, if model.step() failed, a bare `except` fed only the LAST character
            #    without any memory, so the text had no context at all.
            x = torch.tensor([ids[-cfg["seq_len"]:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1] / args.temperature  # scores for the next character
            ids.append(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())

    text = "".join(vocab[i] for i in ids)
    with open(os.path.join(run_dir, "generate.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n[saved to] {run_dir}/generate.txt")


if __name__ == "__main__":
    main()
