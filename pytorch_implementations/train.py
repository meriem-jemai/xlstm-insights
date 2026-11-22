"""
Train one of the PyTorch models.

What happens:
    1. load the text, split it into 90 % training and 10 % validation
    2. build the chosen model (LSTM, sLSTM, mLSTM or xLSTM)
    3. train it (see training/loop.py), evaluating on the validation text after every epoch
    4. save everything to runs/: <name>_results.json (all numbers), <name>_loss.png, <name>.pt (weights)
    The run name is <dataset>_<model>, plus _h<hidden> if the hidden size is not 256.

Examples:
    python train.py --model lstm
    python train.py --model slstm
    python train.py --model mlstm --mlstm_variant global
    python train.py --model xlstm --blocks s,m
    python train.py --model lstm --data dyck_train.txt --wandb
"""
import argparse
import json
import os

import torch

from training.checkpoints import save_checkpoint
from training.data import get_batch, prepare_dataset
from models import build_model, run_name
from training.plotting import plot_loss_curve
from training.loop import evaluate, train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="shakespeare.txt")
    ap.add_argument("--model", type=str, default="lstm", choices=["lstm", "slstm", "mlstm", "xlstm"])
    ap.add_argument("--blocks", type=str, default="s,m", help="xLSTM block order, e.g. s,m or m,s")
    ap.add_argument("--mlstm_variant", type=str, default="base", choices=["base", "global", "separate"])
    ap.add_argument("--embedding", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)                       # n: reproducible runs
    ap.add_argument("--steps_per_epoch", type=int, default=None, help="n: fixed steps per epoch (quick tests)")
    ap.add_argument("--out", type=str, default="runs")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")  # n: optional now
    args = ap.parse_args()

    # the seed fixes the initial weights and which batches are drawn, so a run can be repeated
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)  # controls which batches are drawn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU if available

    # 90 % of the text for training, the last 10 % for validation
    train_data, val_data, vocab_size, _, _ = prepare_dataset(args.data, device=device)
    model = build_model(args.model, vocab_size, args.embedding, args.hidden, args.heads,
                        args.blocks, args.mlstm_variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # run name, e.g. shakespeare_mlstm_separate_h320 (used for all output files)
    dataset = os.path.splitext(os.path.basename(args.data))[0]
    name = f"{dataset}_{run_name(args.model, args.mlstm_variant, args.blocks)}"
    if args.hidden != 256:
        name += f"_h{args.hidden}"  # n: hidden size in the name, so size-matched runs don't overwrite others
    print(f"{name}: {n_params:,} parameters, device {device}")

    # settings passed to the training loop
    config = dict(learning_rate=args.lr, batch_size=args.batch, seq_length=args.seq,
                  num_epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                  **{k: v for k, v in vars(args).items() if k != "steps_per_epoch"})

    log_fn = None
    if args.wandb:
        import wandb
        wandb.init(project="xlstm-pytorch", name=name, config=config)
        log_fn = wandb.log

    history = train_model(model, train_data, val_data, config, get_batch, log_fn=log_fn, generator=generator)

    # n: final metrics on more validation batches, saved together with the history
    final = evaluate(model, val_data, config, get_batch, steps=200, generator=generator)
    print(f"\nFINAL (validation) loss {final['loss']:.4f} | ppl {final['ppl']:.3f} | "
          f"bpc {final['bpc']:.4f} | acc {final['accuracy']:.2%}")

    # save: all numbers (for collect_results.py), the loss plot and the trained weights
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{name}_results.json"), "w") as f:
        json.dump({"name": name, "params": n_params, "config": vars(args),
                   "history": history, "final_val": final}, f, indent=2)
    plot_loss_curve(history, title=f"{name}", filename=os.path.join(args.out, f"{name}_loss.png"))
    save_checkpoint(model, os.path.join(args.out, f"{name}.pt"))

    if args.wandb:
        wandb.summary.update({f"final/{k}": v for k, v in final.items()})
        wandb.finish()


if __name__ == "__main__":
    main()

