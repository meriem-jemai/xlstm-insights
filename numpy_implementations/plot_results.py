"""
n: new file. Plots the training curves saved by train.py and prints a results table.

Usage (after training):
    python plot_results.py

Runs that were not trained yet are skipped.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")  # save images without opening a window
import matplotlib.pyplot as plt

VOCAB = 65  # unique characters in shakespeare.txt

# c: all runs of the size comparison: (label, history file, hidden size, heads; heads=None means LSTM)
RUNS = [
    ("LSTM", "runs/lstm_history.json", 256, None),
    ("sLSTM, 1 head", "runs/slstm_1heads_history.json", 256, 1),
    ("sLSTM, 4 heads (hidden 448)", "runs/slstm_4heads_h448_history.json", 448, 4),
    ("sLSTM, 4 heads (hidden 256)", "runs/slstm_4heads_history.json", 256, 4),
]
OUT = "figures"


def n_params(hidden, heads, vocab=VOCAB):
    """Parameter count: 4 gates x (input weights + recurrent weights + bias) + output layer."""
    recurrent = hidden * hidden if heads is None else heads * (hidden // heads) ** 2  # block-diagonal for sLSTM
    return 4 * (hidden * vocab + recurrent + hidden) + vocab * hidden + vocab


def style(ax, title):
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg NLL per char")
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = []
    for label, path, hidden, heads in RUNS:
        if os.path.exists(path):
            with open(path) as f:
                runs.append((label, json.load(f), n_params(hidden, heads)))
        else:
            print(f"skipped (not trained yet): {label}")
    if not runs:
        raise SystemExit("No results found in runs/. Train the models first.")

    # one plot per run, with the best value marked
    for label, hist, _ in runs:
        epochs = [r["epoch"] for r in hist]
        losses = [r["loss"] for r in hist]
        best = min(range(len(losses)), key=losses.__getitem__)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, losses, linewidth=2)
        ax.scatter(epochs[best], losses[best], s=28, zorder=3)
        ax.annotate(f"best {losses[best]:.3f}", (epochs[best], losses[best]),
                    xytext=(8, 10), textcoords="offset points", fontsize=9)
        style(ax, f"{label}: average loss per epoch")
        fname = label.lower().replace(", ", "_").replace(" ", "_").replace("(", "").replace(")", "") + "_loss.png"
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150)
        plt.close(fig)

    # all runs in one plot, to compare directly
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, hist, params in runs:
        ax.plot([r["epoch"] for r in hist], [r["loss"] for r in hist], linewidth=2,
                label=f"{label} ({params / 1000:.0f}k params)")
    style(ax, "LSTM vs sLSTM (NumPy, Shakespeare)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "loss_comparison.png"), dpi=150)
    plt.close(fig)

    # results table (markdown), ready to paste into the README
    print("| Model | Parameters | Loss | Perplexity | Bits per char | Tokens/s |")
    print("|---|---|---|---|---|---|")
    for label, hist, params in runs:
        r = hist[-1]
        print(f"| {label} | {params / 1000:.0f}k | {r['loss']:.3f} | {r['perplexity']:.2f} | "
              f"{r['bits_per_char']:.3f} | {r['tokens_per_s']:.0f} |")
    print(f"\nSaved plots to {OUT}/")


if __name__ == "__main__":
    main()
