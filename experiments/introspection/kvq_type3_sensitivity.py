import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from config import RESULTS_DIR
from kvq_utils import kvq_inplace_modifiers, scale_tensor, eval_loss, get_QKV_first, infer_layout_from_Q, compute_attention, attention_entropy
OUTDIR = os.path.join(RESULTS_DIR, 'kvq_type3_sensitivity')

def _attn_entropy_under_current(model) -> float:
    Q, K, _ = get_QKV_first()
    nh, hd = infer_layout_from_Q(Q)
    print('[type3]', 'heads =', nh, 'head_dim =', hd)
    A = compute_attention(Q, K, head_dim=hd, causal=True)
    return float(attention_entropy(A).mean().item())

def run(model: torch.nn.Module, ids: torch.Tensor, q_scales: Sequence[float]=(0.25, 0.5, 1.0, 2.0, 4.0), k_scales: Sequence[float]=(0.25, 0.5, 1.0, 2.0, 4.0), q_noise: Sequence[float]=(0.0, 0.01, 0.05, 0.1), k_noise: Sequence[float]=(0.0, 0.01, 0.05, 0.1)):
    os.makedirs(OUTDIR, exist_ok=True)
    device = next(model.parameters()).device
    ids = ids.to(device)
    results = []
    
    for a in q_scales:
        with kvq_inplace_modifiers(model, fQ=lambda x, aa=a: scale_tensor(x, aa)):
            m = eval_loss(model, ids)
            H = _attn_entropy_under_current(model)
            results.append(('scaleQ', a, m['loss'], m['ppl'], m['acc'], H))
    for b in k_scales:
        with kvq_inplace_modifiers(model, fK=lambda x, bb=b: scale_tensor(x, bb)):
            m = eval_loss(model, ids)
            H = _attn_entropy_under_current(model)
            results.append(('scaleK', b, m['loss'], m['ppl'], m['acc'], H))
    for s in q_noise:
        with kvq_inplace_modifiers(model, fQ=lambda x, ss=s: x + torch.randn_like(x) * ss):
            m = eval_loss(model, ids)
            H = _attn_entropy_under_current(model)
            results.append(('noiseQ', s, m['loss'], m['ppl'], m['acc'], H))
    for s in k_noise:
        with kvq_inplace_modifiers(model, fK=lambda x, ss=s: x + torch.randn_like(x) * ss):
            m = eval_loss(model, ids)
            H = _attn_entropy_under_current(model)
            results.append(('noiseK', s, m['loss'], m['ppl'], m['acc'], H))
    with open(os.path.join(OUTDIR, 'sensitivity.csv'), 'w') as fh:
        fh.write('kind,param,loss,ppl,acc,attn_entropy\n')
        for k, p, los, ppl, acc, H in results:
            fh.write(f'{k},{p},{los},{ppl},{acc},{H}\n')

    def plot_series(kind_tag):
        xs, ys = ([], [])
        for kind, p, los, *_ in results:
            if kind == kind_tag:
                xs.append(p)
                ys.append(los)
        xs, ys = zip(*sorted(zip(xs, ys)))
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, marker='o')
        plt.title(f'{kind_tag}: loss vs param')
        plt.xlabel('param')
        plt.ylabel('loss')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'{kind_tag}_loss.png'))
        plt.close()
    for tag in ['scaleQ', 'scaleK', 'noiseQ', 'noiseK']:
        plot_series(tag)