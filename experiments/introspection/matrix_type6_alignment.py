import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import hooks
from config import RESULTS_DIR  

OUTDIR = os.path.join(RESULTS_DIR, "matrix_type6_alignment")
os.makedirs(OUTDIR, exist_ok=True)

def _ensure(d: str):
    os.makedirs(d, exist_ok=True)

@torch.no_grad()
def _compute():
    C_SEQ = hooks.C_SEQ
    if C_SEQ is None:
        raise RuntimeError("C_SEQ is None. Run the runner first.")
    B, S, NH, DH, _ = C_SEQ.shape
    Cb = C_SEQ.mean(0)                 # [S, NH, DH, DH]
    dM = torch.zeros_like(Cb); dM[1:] = Cb[1:] - Cb[:-1]

    align = torch.zeros(S, NH, device=Cb.device)
    for t in range(1, S):
        for h in range(NH):
            M  = Cb[t, h]
            d  = dM[t, h]
            if torch.norm(d) < 1e-8 or torch.norm(M) < 1e-8:
                align[t, h] = 0.0
                continue
            # Principal right singular vectors (first row of Vh)
            V_M  = torch.linalg.svd(M, full_matrices=False).Vh[0]
            V_dM = torch.linalg.svd(d, full_matrices=False).Vh[0]
            num = torch.dot(V_M, V_dM).abs()
            den = (V_M.norm() * V_dM.norm() + 1e-12)
            align[t, h] = (num / den).clamp(0, 1)
    return align.detach().cpu().numpy()  # shape [S, NH]

def _plot(align: np.ndarray, path: str):
    with plt.rc_context({"axes.grid": False}):
        S, NH = align.shape
        fig, ax = plt.subplots(figsize=(6.2, 3.1), dpi=200)
        im = ax.imshow(
            align.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="bilinear",   
            filternorm=True,
            filterrad=1.5
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("cosine", rotation=90, labelpad=8)

        ax.set_title(r"Alignment between $M_t$ and $\Delta M_t$", pad=6)
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

def run():
    _ensure(OUTDIR)
    align = _compute()
    np.savez(os.path.join(OUTDIR, "alignment.npz"), alignment=align)
    _plot(align, os.path.join(OUTDIR, "alignment.png"))

if __name__ == "__main__":
    run()
