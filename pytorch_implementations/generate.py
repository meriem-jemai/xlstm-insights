"""
Generate text with a trained checkpoint.

Example:
    python generate.py --model slstm --ckpt runs/shakespeare_slstm.pt --seed "romeo:"
"""
import argparse

import torch

from training.checkpoints import load_checkpoint
from training.data import prepare_dataset
from models import build_model, run_name
from training.generation import generate_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="shakespeare.txt")
    ap.add_argument("--model", type=str, default="lstm", choices=["lstm", "slstm", "mlstm", "xlstm"])
    ap.add_argument("--blocks", type=str, default="s,m")
    ap.add_argument("--mlstm_variant", type=str, default="base", choices=["base", "global", "separate"])
    ap.add_argument("--embedding", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--seed", type=str, default="f", help="text to start from")
    ap.add_argument("--length", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, vocab_size, char_to_idx, idx_to_char = prepare_dataset(args.data, device=device)
    # c: models are built in models.py (shared with train.py)
    model = build_model(args.model, vocab_size, args.embedding, args.hidden, args.heads,
                        args.blocks, args.mlstm_variant).to(device)
    load_checkpoint(model, args.ckpt, device=device)

    text = generate_text(model, args.seed.lower(), char_to_idx, idx_to_char,
                         length=args.length, temperature=args.temperature)
    print(text)
    out_path = f"generated_{run_name(args.model, args.mlstm_variant, args.blocks)}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
