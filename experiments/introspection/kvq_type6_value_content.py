import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple
from config import RESULTS_DIR
from kvq_utils import get_QKV_first, infer_layout_from_Q, compute_attention, eval_loss, kvq_inplace_modifiers
OUTDIR = os.path.join(RESULTS_DIR, 'kvq_type6_value')

def run(model: torch.nn.Module, ids: torch.Tensor, topk_dims: int=32):
    os.makedirs(OUTDIR, exist_ok=True)
    device = next(model.parameters()).device
    ids = ids.to(device)
    Q, K, V = get_QKV_first()
    nh, hd = infer_layout_from_Q(Q)
    A = compute_attention(Q.to(device), K.to(device), head_dim=hd, causal=True)
    A_mean = A.mean(dim=1)
    C = torch.einsum('bts,bsh->bth', A_mean, V.to(device))
    contrib = C.abs().mean(dim=(0, 1)).detach().cpu().numpy()
    order = np.argsort(-contrib)
    top = order[:topk_dims]
    with open(os.path.join(OUTDIR, 'v_dim_contribution_ranking.csv'), 'w') as fh:
        fh.write('dim,score\n')
        for d in order:
            fh.write(f'{int(d)},{float(contrib[d])}\n')

    def zero_top_dims(x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        if y.dim() == 2:
            y = y.unsqueeze(1)
        y[..., top] = 0.0
        return y
    base = eval_loss(model, ids)
    with kvq_inplace_modifiers(model, fV=zero_top_dims):
        ab = eval_loss(model, ids)
    with open(os.path.join(OUTDIR, 'topk_v_dim_ablation.txt'), 'w') as f:
        f.write(f'topk_dims: {topk_dims}\n')
        f.write(f"baseline_loss: {base['loss']}\n")
        f.write(f"ablated_loss:  {ab['loss']}\n")
        f.write(f"delta_loss:    {ab['loss'] - base['loss']}\n")
    plt.figure(figsize=(8, 4))
    plt.plot(contrib[order[:min(128, len(order))]])
    plt.title('Top V dims by contribution (|AV|)')
    plt.xlabel('rank')
    plt.ylabel('score')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'v_dim_contrib_top.png'))
    plt.close()