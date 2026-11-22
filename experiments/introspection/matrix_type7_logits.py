import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

import hooks_matrix as hooks

from config import RESULTS_DIR, PATTERN_PROVIDER  
from patterns_common import load_spans_from_provider, spans_dict_to_masks

OUTDIR = os.path.join(RESULTS_DIR, "matrix_type7_logits")  
os.makedirs(OUTDIR, exist_ok=True)

def _ensure(d): os.makedirs(d, exist_ok=True)

def _ridge_fit(X, y, lam=1e-3):
    Xc = X - X.mean(0, keepdims=True)
    yc = y - y.mean()
    XtX = Xc.T @ Xc
    beta = np.linalg.solve(XtX + lam*np.eye(XtX.shape[0]), Xc.T @ yc)
    yhat = Xc @ beta + y.mean()
    ssr = np.sum((y - yhat)**2)
    sst = np.sum((y - y.mean())**2) + 1e-12
    r2  = 1.0 - ssr/sst
    return r2, yhat, beta

def _pca_components(A, k):
    A0 = A - A.mean(0, keepdims=True)
    U, Sg, Vt = np.linalg.svd(A0, full_matrices=False)
    Vt = Vt[:k]                 
    Z  = (A0 @ Vt.T)            # [S, k]
    return Z, Vt                

@torch.no_grad()
def _memory_features(C, topk_sv=3):

    Cb = C.mean(0)  # [S,NH,DH,DH]
    S, NH, DH, _ = Cb.shape
    feats = []
    prev = None
    for t in range(S):
        ft = []
        for h in range(NH):
            M = Cb[t, h].cpu().numpy()
            fn = np.linalg.norm(M)                  # Frobenius
            if prev is None:
                fd = 0.0
            else:
                fd = np.linalg.norm(M - prev[h])    # Delta Frobenius
            svals = np.linalg.svd(M, compute_uv=False)
            top = svals[:topk_sv]
            nuc = float(np.sum(svals))
            ft.extend([fn, fd, *top, nuc])
        feats.append(ft)
        prev = Cb[t].cpu().numpy()
    F = np.asarray(feats, dtype=float)  
    mu = F.mean(0, keepdims=True)
    sd = F.std(0, keepdims=True) + 1e-9
    return (F - mu)/sd

@torch.no_grad()
def run(model: torch.nn.Module, ids: torch.Tensor, text: str = "", k_pca: int = 16, topk_sv: int = 3):
    _ensure(OUTDIR)

    C = getattr(hooks, "C_SEQ", None)     # [B,S,NH,DH,DH]
    A = getattr(hooks, "ACT_SEQ", None)   # [B,S,H]
    if C is None:
        raise RuntimeError("C_SEQ is None. Run replay_cell_sequence() first.")
    if A is None:
        A = getattr(hooks, "LAYER_IN", None)
    if A is None:
        raise RuntimeError("No ACT_SEQ/LAYER_IN found.")

    device = next(model.parameters()).device
    ids = ids.to(device)

    logits = model(ids)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if ids.size(1) >= 2:
        tgt    = ids[:, 1:]
        logits = logits[:, :-1, :]
    else:
        tgt = ids
    with torch.no_grad():
        cc = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  
        y  = cc.mean(0).detach().cpu().numpy()                 
    S_y = y.shape[0]

    Fm = _memory_features(C, topk_sv=topk_sv)       # [S, Fm]
    Fa_src = (A.mean(0).detach().cpu().numpy())     # [S, H]
    k_pca = int(min(k_pca, Fa_src.shape[0]-2, Fa_src.shape[1]))
    k_pca = max(k_pca, 4)
    Fa, _ = _pca_components(Fa_src, k=k_pca)        

    S = Fa.shape[0]
    T = min(S, Fm.shape[0], S_y)
    y  = y[:T]; Fm = Fm[:T]; Fa = Fa[:T]

    r2_mem,  yhat_m,  b_m  = _ridge_fit(Fm, y, lam=1e-2)
    r2_act,  yhat_a,  b_a  = _ridge_fit(Fa, y, lam=1e-2)
    r2_both, yhat_b,  b_b  = _ridge_fit(np.concatenate([Fm, Fa], 1), y, lam=1e-2)

    metrics_rows = [("GLOBAL", r2_mem, r2_act, r2_both)]
    if text and PATTERN_PROVIDER:
        spans = load_spans_from_provider(text, T, PATTERN_PROVIDER)
        masks = spans_dict_to_masks(spans, T) if spans else {}
        for name, m in masks.items():
            if not m.any() or (m.sum() < 5):
                continue
            im = np.where(m)[0]
            io = np.where(~m)[0] if (~m).any() else None

            r2_m_in, _, _ = _ridge_fit(Fm[im], y[im])
            r2_a_in, _, _ = _ridge_fit(Fa[im], y[im])
            r2_b_in, _, _ = _ridge_fit(np.concatenate([Fm[im], Fa[im]], 1), y[im])

            if io is not None and io.size >= 5:
                r2_m_out, _, _ = _ridge_fit(Fm[io], y[io])
                r2_a_out, _, _ = _ridge_fit(Fa[io], y[io])
                r2_b_out, _, _ = _ridge_fit(np.concatenate([Fm[io], Fa[io]], 1), y[io])
            else:
                r2_m_out = r2_a_out = r2_b_out = np.nan

            metrics_rows.append((name, r2_m_in, r2_a_in, r2_b_in))
            metrics_rows.append((name+"__OUT", r2_m_out, r2_a_out, r2_b_out))

    with open(os.path.join(OUTDIR, "r2_summary.csv"), "w") as f:
        f.write("split,R2_mem,R2_act,R2_both\n")
        for row in metrics_rows:
            f.write("{},{},{},{}\n".format(*row))

    labels = [r[0] for r in metrics_rows if not r[0].endswith("__OUT")]
    mems   = [r[1] for r in metrics_rows if not r[0].endswith("__OUT")]
    acts   = [r[2] for r in metrics_rows if not r[0].endswith("__OUT")]
    both   = [r[3] for r in metrics_rows if not r[0].endswith("__OUT")]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(5.6, 1.6*len(labels)), 2.9), dpi=200)
    ax.bar(x - 0.25, mems, width=0.25, label="memory")
    ax.bar(x,         acts, width=0.25, label="activations")
    ax.bar(x + 0.25,  both, width=0.25, label="both")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"$R^2$")
    ax.set_title("Correct-class logit explainability", pad=6)
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUTDIR, "r2_bars.png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


    print("[A7] wrote", OUTDIR)
