import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import hooks
from config import RESULTS_DIR

OUTDIR = os.path.join(RESULTS_DIR, "matrix_type11_attn_vs_spectrum")
os.makedirs(OUTDIR, exist_ok=True)

def _get_attn4d():
    A = getattr(hooks, "AUX", None)
    if not isinstance(A, dict):
        raise RuntimeError("hooks.AUX missing")
    attn = A.get("attn", None)
    if not torch.is_tensor(attn):
        raise RuntimeError("AUX['attn'] missing")
    if attn.dim() != 4:
        raise RuntimeError(f"AUX['attn'] must be 4D [B,S,NH,S], got {tuple(attn.shape)}")
    return attn.detach()

@torch.no_grad()
def _entropy_min_over_heads(attn4):
    P = attn4.clamp_min(1e-12)
    H = -(P * P.log()).sum(-1)
    Hmin = H.min(dim=2).values
    return Hmin.mean(dim=0)

@torch.no_grad()
def _spectrum_ratio():
    C = getattr(hooks, "C_SEQ", None)
    if C is None:
        raise RuntimeError("C_SEQ is None. Run replay first.")
    if C.dim() != 5:
        raise RuntimeError(f"C_SEQ must be [B,S,NH,DH,DH], got {tuple(C.shape)}")
    Cb = C.mean(0)
    B,S,NH,DH,_ = C.shape
    steps = np.arange(S)
    ratio = np.empty(S, dtype=float)
    for t in range(S):
        M = Cb[t].reshape(NH * DH, DH)
        s = torch.linalg.svdvals(M)
        s1 = s[0].item()
        s2 = s[1].item() if s.numel() > 1 else 0.0
        ratio[t] = (s1 / s2) if s2 > 1e-8 else np.inf
    return steps, ratio

def _plot(steps, H, ratio, out_png):
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(steps, H, label="attention entropy", linewidth=1.5)
    ax1.set_xlabel("time")
    ax1.set_ylabel("entropy")
    ax2 = ax1.twinx()
    ax2.plot(steps, ratio, label="spectrum ratio", linewidth=1.5, alpha=0.9)
    ax2.set_ylabel("spectrum ratio")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    fig.suptitle("Attention entropy vs spectrum ratio")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def run():
    attn4 = _get_attn4d()
    H = _entropy_min_over_heads(attn4).cpu().numpy()
    steps, ratio = _spectrum_ratio()
    np.savez(os.path.join(OUTDIR, "attn_vs_spectrum.npz"), steps=steps, entropy=H, sigma_ratio=ratio)
    _plot(steps, H, ratio, os.path.join(OUTDIR, "attn_vs_spectrum.png"))
    with open(os.path.join(OUTDIR, "info.txt"), "w") as f:
        f.write(f"attn shape={tuple(attn4.shape)}\n")
        f.write(f"entropy len={len(H)} steps={len(steps)}\n")
    print("[B11] wrote", OUTDIR)

if __name__ == "__main__":
    run()
