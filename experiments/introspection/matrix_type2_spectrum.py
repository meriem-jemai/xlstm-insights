import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import hooks
from config import RESULTS_DIR  

OUTDIR = os.path.join(RESULTS_DIR, "matrix_type2_spectrum")
os.makedirs(OUTDIR, exist_ok=True)


def _ensure(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def _style_axes(ax, title: str, xlabel: str = "time"):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(title, pad=6)
    ax.set_xlabel(xlabel, labelpad=4)
    ax.margins(x=0)

@torch.no_grad()
def _compute(step_stride: int = 10, topk: int = 10):
    C_SEQ = hooks.C_SEQ
    if C_SEQ is None:
        raise RuntimeError("C_SEQ is None. Run the runner to collect it first.")
    B, S, NH, DH, _ = C_SEQ.shape
    Cb = C_SEQ.mean(0)  # [S,NH,DH,DH]

    steps, sigmas, nuclear = [], [], []
    for t in range(0, S, step_stride):
        M = Cb[t].reshape(NH * DH, DH)
        svals = torch.linalg.svdvals(M)
        steps.append(t)
        svals = svals[:topk]
        sigmas.append(svals.cpu().numpy())
        nuclear.append(svals.sum().item())

    return np.array(steps), np.stack(sigmas, 0), np.array(nuclear)

def _plot_sigmas(steps: np.ndarray, sigmas: np.ndarray, path: str,
                 figsize=(5.6, 2.9), linewidth=1.2):
    fig, ax = plt.subplots(figsize=figsize)
    k = sigmas.shape[1]
    for i in range(k):
        ax.plot(steps, sigmas[:, i], linewidth=linewidth, label=fr"$\sigma_{{{i+1}}}$")
    _style_axes(ax, r"Top singular values of $C_t$")
    ax.legend(ncol=min(5, k), frameon=False, handlelength=1.6, borderaxespad=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def _plot_line(steps: np.ndarray, y: np.ndarray, title: str, path: str,
               figsize=(5.6, 2.9), linewidth=1.4):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(steps, y, linewidth=linewidth)
    _style_axes(ax, title)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def run(step_stride: int = 10, topk: int = 10, out_base: str = None):
    outdir = OUTDIR if out_base is None else os.path.join(BASE_OUT, out_base)
    _ensure(outdir)
    steps, sigmas, nuclear = _compute(step_stride=step_stride, topk=topk)

    np.savez(os.path.join(outdir, "spectrum.npz"),
             steps=steps, sigmas=sigmas, nuclear=nuclear)

    _plot_sigmas(steps, sigmas, os.path.join(outdir, "sigmas.png"))
    _plot_line(steps, nuclear, r"Nuclear norm $\sum \sigma_i$",
               os.path.join(outdir, "nuclear.png"))

if __name__ == "__main__":
    run()
