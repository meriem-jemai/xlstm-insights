import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import hooks
from config import RESULTS_DIR  

OUTDIR = os.path.join(RESULTS_DIR, "matrix_type5_lifespan")
os.makedirs(OUTDIR, exist_ok=True)


def _ensure(d: str):
    os.makedirs(d, exist_ok=True)

@torch.no_grad()
def _compute(delta_q: float = 0.9, sustain_q: float = 0.7):
    C_SEQ = hooks.C_SEQ
    if C_SEQ is None:
        raise RuntimeError("C_SEQ is None. Run the runner first.")
    B, S, NH, DH, _ = C_SEQ.shape
    Cb = C_SEQ.mean(0)  # [S, NH, DH, DH]

    # Delta over time (Frobenius norm per head)
    delta = torch.zeros(S, NH, device=Cb.device)
    delta[1:] = ((Cb[1:] - Cb[:-1]) ** 2).sum((2, 3)).sqrt()

    thresh  = torch.quantile(delta, delta_q, dim=0, keepdim=True)    # [1, NH]
    sustain = torch.quantile(delta, sustain_q, dim=0, keepdim=True)  # [1, NH]

    lifespans = []
    for h in range(NH):
        head = []
        t = 1
        while t < S:
            if delta[t, h] > thresh[0, h]:
                start = t
                while t < S and delta[t, h] >= sustain[0, h]:
                    t += 1
                head.append({"start": int(start), "len": int(t - start)})
            else:
                t += 1
        lifespans.append(head)

    return delta.cpu().numpy(), lifespans  # delta shape [S, NH]

def _plot_delta(delta: np.ndarray, path: str):
    with plt.rc_context({"axes.grid": False}):
        S, NH = delta.shape
        fig, ax = plt.subplots(figsize=(6.2, 3.1), dpi=200)

        im = ax.imshow(
            delta.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="bilinear",    
            filternorm=True,
            filterrad=1.5
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\|\Delta C_t\|_F$", rotation=90, labelpad=8)

        ax.set_title(r"$\Delta C_t$ Frobenius per head", pad=6)
        ax.set_xlabel("time", labelpad=4)
        ax.set_ylabel("head", labelpad=4)

        ax.set_ylim(-0.5, NH - 0.5)
        ax.set_yticks(np.arange(NH))
        ax.yaxis.set_major_locator(mticker.FixedLocator(np.arange(NH)))
        ax.set_yticklabels([str(i) for i in range(NH)])

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        fig.tight_layout(pad=0.3)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

def run(delta_q: float = 0.9, sustain_q: float = 0.7):
    _ensure(OUTDIR)
    delta, lifespans = _compute(delta_q, sustain_q)
    np.savez(os.path.join(OUTDIR, "lifespan.npz"), delta=delta)
    with open(os.path.join(OUTDIR, "lifespans.json"), "w") as f:
        json.dump(lifespans, f, indent=2)
    _plot_delta(delta, os.path.join(OUTDIR, "delta.png"))

if __name__ == "__main__":
    run()
