import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import hooks
from config import RESULTS_DIR  

OUTDIR = os.path.join(RESULTS_DIR, "matrix_type3_redundancy")
os.makedirs(OUTDIR, exist_ok=True)


def _ensure(d): os.makedirs(d, exist_ok=True)

@torch.no_grad()
def _compute():
    C_SEQ = hooks.C_SEQ
    if C_SEQ is None:
        raise RuntimeError("C_SEQ is None. Run the runner first.")
    B,S,NH,DH,_ = C_SEQ.shape
    Cb = C_SEQ.mean(0)  # [S,NH,DH,DH]

    red = torch.zeros(S, NH, device=Cb.device)
    eye = torch.eye(DH, dtype=torch.bool, device=Cb.device)
    for t in range(S):
        Mt = Cb[t]  # [NH,DH,DH]
        row_norm = Mt / (Mt.norm(dim=2, keepdim=True) + 1e-8)
        cos = row_norm @ row_norm.transpose(1,2)  # [NH,DH,DH]
        off = cos.masked_select(~eye)
        red[t] = off.abs().view(NH, -1).mean(dim=1)
    return red.cpu().numpy()

def _plot_mean(y, path):
    plt.figure(figsize=(8,4)); plt.plot(y); plt.title("Redundancy")
    plt.xlabel("time"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def run():
    _ensure(OUTDIR)
    red = _compute()
    np.savez(os.path.join(OUTDIR, "redundancy.npz"), redundancy=red)
    _plot_mean(red.mean(1), os.path.join(OUTDIR, "redundancy_mean.png"))

if __name__ == "__main__":
    run()
