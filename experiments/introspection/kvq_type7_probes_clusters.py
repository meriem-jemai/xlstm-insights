import os, re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple
from config import RESULTS_DIR, DEVICE
from kvq_utils import get_QKV_first
OUTDIR = os.path.join(RESULTS_DIR, 'kvq_type7_probes')

def _char_class_labels(text: str) -> np.ndarray:
    y = np.zeros(len(text), dtype=np.int64)
    for i, ch in enumerate(text):
        if ch.isalpha():
            y[i] = 0
        elif ch.isdigit():
            y[i] = 1
        elif ch.isspace():
            y[i] = 2
        elif re.match('[^\\w\\s]', ch):
            y[i] = 3
        else:
            y[i] = 4
    return y

def _train_probe(X: torch.Tensor, y: torch.Tensor, nclass: int=5, steps: int=400, lr: float=0.1) -> float:
    model = nn.Linear(X.size(1), nclass, bias=True).to(X.device)
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (model(X).argmax(-1) == y).float().mean().item()
    return acc

def _pca(X: np.ndarray, k: int=2) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :k] * S[:k]

def _kmeans(X: np.ndarray, k: int=8, iters: int=50) -> Tuple[np.ndarray, np.ndarray]:
    N, D = X.shape
    rng = np.random.default_rng(0)
    cent = X[rng.choice(N, size=k, replace=False)]
    for _ in range(iters):
        d2 = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        lbl = d2.argmin(axis=1)
        for j in range(k):
            sel = X[lbl == j]
            if sel.size:
                cent[j] = sel.mean(0)
    return (cent, lbl)

def run(text: str, sample: int=20000):
    os.makedirs(OUTDIR, exist_ok=True)
    Q, K, V = get_QKV_first()
    XQ = Q.reshape(-1, Q.size(-1))
    XK = K.reshape(-1, K.size(-1))
    XV = V.reshape(-1, V.size(-1))
    y = _char_class_labels(text[:XQ.size(0)])
    y_t = torch.tensor(y, device=DEVICE)
    idx = np.arange(min(sample, XQ.size(0)))
    XQ_t = XQ[idx].to(DEVICE)
    XK_t = XK[idx].to(DEVICE)
    XV_t = XV[idx].to(DEVICE)
    y_t = y_t[idx]
    acc_q = _train_probe(XQ_t, y_t)
    acc_k = _train_probe(XK_t, y_t)
    acc_v = _train_probe(XV_t, y_t)
    with open(os.path.join(OUTDIR, 'probe_accuracy.txt'), 'w') as f:
        f.write(f'Q->class acc: {acc_q}\nK->class acc: {acc_k}\nV->class acc: {acc_v}\n')
    X = XQ.cpu().numpy()
    Z = _pca(X, k=2)
    _, lbl = _kmeans(Z, k=8)
    plt.figure(figsize=(6, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=lbl, s=6)
    plt.title('Q PCA + k-means clusters')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'Q_pca_kmeans.png'))
    plt.close()