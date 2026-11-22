
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from config import RESULTS_DIR
import hooks_matrix as hooks

OUTDIR = os.path.join(RESULTS_DIR, "matrix_type12_leadlag")
os.makedirs(OUTDIR, exist_ok=True)

def _ensure(d):
    os.makedirs(d, exist_ok=True)

@torch.no_grad()
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
def _write_intensity():
    C = getattr(hooks, "C_SEQ", None)
    if C is None:
        raise RuntimeError("C_SEQ is None. Run replay first.")
    if C.dim() != 5:
        raise RuntimeError(f"C_SEQ must be [B,S,NH,DH,DH], got {tuple(C.shape)}")
    Cb = C.mean(0)
    S, NH, DH, _ = Cb.shape
    W = torch.zeros(S, NH, device=Cb.device)
    W[1:] = ((Cb[1:] - Cb[:-1]) ** 2).sum((2, 3)).sqrt()
    return W

def _pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x -= x.mean()
    y -= y.mean()
    den = (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
    return float((x @ y) / den)

@torch.no_grad()
def _xcorr_leadlag(max_lag=64):
    attn = _get_attn4d()
    W = _write_intensity().cpu().numpy()
    A = attn.mean(0).cpu().numpy()
    S, NH, _ = A.shape
    L = min(max_lag, max(1, S - 1))
    lags = np.arange(1, L + 1, dtype=int)
    corr = np.zeros(L, dtype=float)
    for i, d in enumerate(lags):
        xs, ys = [], []
        for h in range(NH):
            for t in range(0, S - d):
                xs.append(W[t, h])
                ys.append(A[t + d, h, t])
        corr[i] = _pearson(xs, ys) if xs else np.nan
    return lags, corr

def _plot(lags, corr, out_png):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(lags, corr, marker="o", linewidth=1.4)
    ax.set_title("Lead-lag cross-correlation")
    ax.set_xlabel("lag (steps)")
    ax.set_ylabel("rho (Pearson)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def run(max_lag: int = 64):
    _ensure(OUTDIR)
    lags, corr = _xcorr_leadlag(max_lag=max_lag)
    peak = int(lags[np.nanargmax(corr)]) if corr.size else -1
    np.savez(os.path.join(OUTDIR, "leadlag.npz"), lags=lags, corr=corr, peak_lag=peak)
    _plot(lags, corr, os.path.join(OUTDIR, "leadlag.png"))
    print(f"[B12] peak revisit lag={peak}")

if __name__ == "__main__":
    run()
