import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from config import RESULTS_DIR
from kvq_utils import kvq_inplace_modifiers, shuffle_time, randomize_like, scale_tensor, get_QKV_first, infer_layout_from_Q, compute_attention, attention_entropy, eval_loss
OUTDIR = os.path.join(RESULTS_DIR, 'kvq_type2_causal')

def _eval_with(model, ids, fQ=None, fK=None, fV=None) -> Dict[str, float]:
    with kvq_inplace_modifiers(model, fQ=fQ, fK=fK, fV=fV):
        return eval_loss(model, ids)

def run(model: torch.nn.Module, ids: torch.Tensor, v_scale: float=1.5, topk_v_zero: int=5):
    os.makedirs(OUTDIR, exist_ok=True)
    device = next(model.parameters()).device
    ids = ids.to(device)
    base = eval_loss(model, ids)
    res_k = _eval_with(model, ids, fK=shuffle_time)
    res_q = _eval_with(model, ids, fQ=shuffle_time)

    def _randV(x):
        return randomize_like(x, scale=float(x.std().item() if x.numel() else 1.0))
    res_vrand = _eval_with(model, ids, fV=_randV)
    res_vscale = _eval_with(model, ids, fV=lambda x: scale_tensor(x, v_scale))
    Q, K, V = get_QKV_first()
    nh, hd = infer_layout_from_Q(Q)
    print('[type2 casu]', 'heads =', nh, 'head_dim =', hd)
    with torch.no_grad():
        A = compute_attention(Q.to(device), K.to(device), head_dim=hd, causal=True)
        attn_src = A.mean(dim=(0, 1, 2))
        top_idx = torch.topk(attn_src, k=min(topk_v_zero, attn_src.numel())).indices
        top_idx = top_idx.cpu()

    def zero_V_rows(x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        if y.dim() == 2:
            y = y.unsqueeze(1)
        y[:, top_idx, :] = 0.0
        return y
    res_vzero = _eval_with(model, ids, fV=zero_V_rows)

    def _dump(name, d):
        with open(os.path.join(OUTDIR, f'{name}.txt'), 'w') as f:
            for k, v in d.items():
                f.write(f'{k}: {v}\n')
    _dump('baseline', base)
    _dump('shuffle_k', res_k)
    _dump('shuffle_q', res_q)
    _dump('randomize_v', res_vrand)
    _dump('scale_v', res_vscale)
    _dump('zero_topV', res_vzero)
    labels = ['base', 'shuffle_k', 'shuffle_q', 'rand_v', 'scale_v', 'zeroTopV']
    vals = [base['loss'], res_k['loss'], res_q['loss'], res_vrand['loss'], res_vscale['loss'], res_vzero['loss']]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(vals)), vals)
    plt.xticks(np.arange(len(vals)), labels, rotation=30)
    plt.title('KVQ causal interventions - loss')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'interventions_loss.png'))
    plt.close()