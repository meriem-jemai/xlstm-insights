import os, numpy as np, torch, matplotlib.pyplot as plt
import hooks
from config import RESULTS_DIR, PATTERN_PROVIDER
from patterns_common import load_spans_from_provider, spans_dict_to_masks

OUTDIR = os.path.join(RESULTS_DIR, "neuron_type14_token_means")
os.makedirs(OUTDIR, exist_ok=True)

def _ensure(d): os.makedirs(d, exist_ok=True)

def _sizes_from_hooks():
    C = getattr(hooks, "C_SEQ", None)
    A = getattr(hooks, "ACT_SEQ", None)
    if A is None and hasattr(hooks, "LAYER_IN"): A = hooks.LAYER_IN
    if C is None or A is None:
        raise RuntimeError("Need both C_SEQ and ACT_SEQ/LAYER_IN. Run replay first.")
    NH = int(C.shape[2])
    if A.dim() == 4:
        DHn = int(A.shape[3]); H = NH * DHn
    else:
        H = int(A.shape[-1])
        if H % NH != 0:
            raise RuntimeError(f"H={H} not divisible by NH={NH}.")
        DHn = H // NH
    return NH, H, DHn

@torch.no_grad()
def run(text: str, pattern_select=None, top_k: int = 32, eps: float = 1e-6):
    _ensure(OUTDIR)
    A = getattr(hooks, "ACT_SEQ", None)
    if A is None: A = getattr(hooks, "LAYER_IN", None)
    if A is None: raise RuntimeError("No ACT_SEQ/LAYER_IN. Run replay first.")
    NH, H, DHn = _sizes_from_hooks()
    if A.dim() == 4:
        B,S,_,_ = A.shape
        X = A.mean(0).reshape(S, NH * DHn).cpu().numpy()
    else:
        B,S,_ = A.shape
        X = A.mean(0).cpu().numpy()
    text = text[:S] if text else ""
    gmu = X.mean(0)
    gsd = X.std(0) + eps
    if not (PATTERN_PROVIDER and text):
        raise RuntimeError("PATTERN_PROVIDER and nonempty text are required.")
    spans = load_spans_from_provider(text, S, PATTERN_PROVIDER, pattern_select)
    masks = spans_dict_to_masks(spans, S) if spans else {}
    names = [k for k, v in masks.items() if v.any()]
    if not names:
        raise RuntimeError("No pattern masks produced by provider.")
    with open(os.path.join(OUTDIR, "pattern_means.csv"), "w") as f:
        f.write("neuron," + ",".join(names) + ",global\n")
        for i in range(H):
            cols = []
            for n in names:
                m = masks[n]
                cols.append(str(X[m].mean(0)[i]))
            cols.append(str(gmu[i]))
            f.write(str(i) + "," + ",".join(cols) + "\n")
    for n in names:
        m = masks[n]
        Xin = X[m]; Xout = X[~m] if (~m).any() else X[m]
        mu_in = Xin.mean(0); mu_out = Xout.mean(0)
        sd_in = Xin.std(0);  sd_out = Xout.std(0)
        delta = mu_in - mu_out
        effect = delta / (np.sqrt(0.5 * (sd_in**2 + sd_out**2)) + eps)
        np.save(os.path.join(OUTDIR, f"pattern_effect_{n}.npy"), effect)
        order = np.argsort(-effect)[:top_k]
        with open(os.path.join(OUTDIR, f"pattern_top_{n}.csv"), "w") as f:
            f.write("rank,neuron,head,offset_in_head,effect,delta,mu_in,mu_out,sd_in,sd_out,z_global\n")
            for r, i in enumerate(order, start=1):
                head = int(i // DHn); offset = int(i % DHn)
                zglob = (mu_in[i] - gmu[i]) / gsd[i]
                f.write(f"{r},{i},{head},{offset},{effect[i]},{delta[i]},{mu_in[i]},{mu_out[i]},{sd_in[i]},{sd_out[i]},{zglob}\n")
        import numpy as _np
        plt.figure(figsize=(10, 4))
        plt.bar(_np.arange(order.size), effect[order])
        plt.xticks(_np.arange(order.size), order, rotation=90)
        plt.ylabel("Cohen d"); plt.title(f"Top neurons for pattern: {n}")
        plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, f"pattern_top_{n}.png"), dpi=150); plt.close()
        plt.figure(figsize=(7, 3)); plt.hist(effect, bins=50)
        plt.title(f"Effect histogram: {n}"); plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"pattern_effect_hist_{n}.png"), dpi=150); plt.close()
        plt.figure(figsize=(7, 3)); plt.hist(_np.abs(effect), bins=50)
        plt.title(f"|Effect| histogram: {n}"); plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"pattern_effect_abs_hist_{n}.png"), dpi=150); plt.close()
    print("[N14] wrote", OUTDIR)
