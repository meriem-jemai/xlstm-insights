"""
n: new file. Collects all *_results.json files from runs/ into markdown tables and comparison plots.

What it produces:
    - tables (printed, ready to paste into the README):
        1. Shakespeare, size-matched models (~575k parameters each)
        2. Dyck, size-matched models
        3. Shakespeare at hidden size 256 (models of different sizes, for reference)
        4. mLSTM variants at hidden size 256 on Shakespeare, Copy and Dyck
    - plots in figures/: validation loss and accuracy per epoch, one line per model,
      for Shakespeare and Dyck

Every run of train.py writes runs/<name>_results.json; this script only reads those files.
Runs that do not exist yet are shown as "-".

    python collect_results.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")  # save images without opening a window
import matplotlib.pyplot as plt

RUNS, OUT = "runs", "figures"

# size-matched comparison (all ~575k parameters): (label, run name without the dataset prefix)
MODELS = [
    ("LSTM", "lstm"),
    ("sLSTM", "slstm_h320"),
    ("mLSTM (base)", "mlstm_base_h356"),
    ("mLSTM + global h (own)", "mlstm_global"),
    ("mLSTM + per-head h (own)", "mlstm_separate_h320"),
    ("xLSTM (sLSTM → mLSTM)", "xlstm_s-m_h128"),
]
DATASETS = [("Shakespeare", "shakespeare"), ("Dyck", "dyck_train")]

# extra: the same models at hidden size 256 (different sizes), and the large xLSTM
EXTRA = [
    ("sLSTM, hidden 256", "slstm"),
    ("mLSTM (base), hidden 256", "mlstm_base"),
    ("mLSTM + per-head h, hidden 256", "mlstm_separate"),
    ("xLSTM, hidden 256", "xlstm_s-m"),
]


def load(name):
    # name = run name, e.g. "shakespeare_lstm" -> reads runs/shakespeare_lstm_results.json
    path = os.path.join(RUNS, f"{name}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def table(title, rows):
    # rows = [(label, results dict or None), ...] -> one markdown table row per model
    print(f"\n### {title}\n")
    print("| Model | Parameters | Val loss ↓ | Val perplexity ↓ | Val accuracy ↑ |")
    print("|---|---|---|---|---|")
    for label, r in rows:
        if r is None:
            print(f"| {label} | – | – | – | – |")
            continue
        v = r["final_val"]
        print(f"| {label} | {r['params'] / 1000:.0f}k | {v['loss']:.3f} | {v['ppl']:.2f} | {v['accuracy']:.1%} |")


def plot(dataset_label, rows, key, ylabel, filename):
    # one line per model: the validation metric `key` (e.g. "val_loss") after every epoch
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, r in rows:
        if r is None:
            continue
        hist = r["history"]
        ax.plot([h["epoch"] for h in hist], [h[key] for h in hist], linewidth=2, marker="o", markersize=3,
                label=f"{label} ({r['params'] / 1000:.0f}k)")
    ax.set_title(f"{dataset_label}: {ylabel} (validation)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    for ds_label, prefix in DATASETS:
        rows = [(label, load(f"{prefix}_{name}")) for label, name in MODELS]
        table(f"{ds_label} (size-matched, ~575k parameters)", rows)
        plot(ds_label, rows, "val_loss", "Loss", f"{prefix.split('_')[0]}_val_loss.png")
        plot(ds_label, rows, "val_accuracy", "Accuracy", f"{prefix.split('_')[0]}_val_accuracy.png")

    table("Shakespeare, hidden size 256 (different sizes)",
          [(label, load(f"shakespeare_{name}")) for label, name in EXTRA])

    variants = [("base", "mlstm_base"), ("global h", "mlstm_global"), ("per-head h", "mlstm_separate")]
    print("\n### mLSTM variants, hidden size 256, val loss / accuracy\n")
    print("| Variant | Shakespeare | Copy | Dyck |")
    print("|---|---|---|---|")
    for label, name in variants:
        cells = []
        for prefix in ("shakespeare", "copy_train", "dyck_train"):
            r = load(f"{prefix}_{name}")
            cells.append("–" if r is None else f"{r['final_val']['loss']:.3f} / {r['final_val']['accuracy']:.1%}")
        print(f"| {label} | " + " | ".join(cells) + " |")

    print(f"\nSaved plots to {OUT}/")


if __name__ == "__main__":
    main()
