from __future__ import annotations
import os
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from config import RESULTS_DIR
from kvq_utils import eval_loss, kvq_inplace_modifiers, get_QKV_first, infer_layout_from_Q
OUTDIR = os.path.join(RESULTS_DIR, 'kvq_type8_counterfactuals')

@torch.no_grad()
def _per_step_ce(model: torch.nn.Module, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = model(ids)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    inp = ids[:, :-1]
    tgt = ids[:, 1:]
    logits = logits[:, :inp.size(1), :]
    B, T, V = logits.shape
    assert B == 1, 'This analysis expects batch size 1.'
    ce = F.cross_entropy(logits.reshape(B * T, V), tgt.reshape(B * T), reduction='none').view(B, T)[0]
    pred = logits.argmax(-1)[0]
    tgt = tgt[0]
    return (ce, pred, tgt)

def _hard_steps(step_ce: torch.Tensor, topN: int) -> torch.Tensor:
    N = min(topN, step_ce.numel())
    return torch.topk(step_ce, k=N, largest=True).indices

def _choose_donor(pred: torch.Tensor, tgt: torch.Tensor, step_ce: torch.Tensor, t: int) -> int:
    correct = pred == tgt
    same_id = (tgt == tgt[t]) & correct
    if same_id.any():
        idx = torch.nonzero(same_id, as_tuple=False).view(-1)
        u = int(idx[torch.argmin(torch.abs(idx - t))])
        return u
    if correct.any():
        idx = torch.nonzero(correct, as_tuple=False).view(-1)
        u = int(idx[torch.argmin(torch.abs(idx - t))])
        return u
    return int(torch.argmin(step_ce).item())

def _window(a: int, b: int, S: int) -> Tuple[int, int]:
    lo = max(0, a)
    hi = min(S - 1, b)
    return (lo, hi)

def run(model: torch.nn.Module, ids: torch.Tensor, topN: int=50, k_window: int=16, adv_len: int=32, distractor_id: Optional[int]=None):
    os.makedirs(OUTDIR, exist_ok=True)
    device = next(model.parameters()).device
    ids = ids.to(device)
    model.eval()
    base_global = eval_loss(model, ids)
    step_ce, pred, tgt = _per_step_ce(model, ids)
    hard_idx = _hard_steps(step_ce, topN=topN)
    S_minus_1 = step_ce.numel()
    S = S_minus_1 + 1
    Q, K, V = get_QKV_first()
    assert Q.size(1) >= S and K.size(1) >= S, 'Captured Q/K length does not match input length.'
    if distractor_id is None:
        hist = torch.bincount(ids[0])
        distractor_id = int(torch.argmax(hist).item())
    rows: List[str] = []
    header = ','.join(['t', 'donor_u', 'target_id', 'base_step_ce', 'base_loss', 'qswap_step_ce', 'qswap_loss', 'qswap_delta_step', 'qswap_delta_loss', 'kswap_step_ce', 'kswap_loss', 'kswap_delta_step', 'kswap_delta_loss', 'adv_step_ce', 'adv_loss', 'adv_delta_step', 'adv_delta_loss'])
    rows.append(header)

    def evaluate_pair(for_ids: torch.Tensor, focus_t: int) -> Tuple[float, float]:
        sc, _, _ = _per_step_ce(model, for_ids)
        glob = eval_loss(model, for_ids)
        t_eval = max(0, min(sc.numel() - 1, focus_t))
        return (float(sc[t_eval].item()), float(glob['loss']))
    for t in hard_idx.tolist():
        t = int(t)
        u = _choose_donor(pred, tgt, step_ce, t)
        base_step = float(step_ce[t].item())
        base_loss = float(base_global['loss'])
        donor_q = Q[:, u, :].clone()

        def fQ_swap(x: torch.Tensor) -> torch.Tensor:
            y = x.clone()
            if y.dim() == 2:
                y[...] = donor_q
                return y
            y[:, t, :] = donor_q
            return y
        with kvq_inplace_modifiers(model, fQ=fQ_swap):
            q_step, q_loss = evaluate_pair(ids, focus_t=t)
        lo_t, hi_t = _window(t - k_window, t + k_window, S_minus_1)
        span = hi_t - lo_t + 1
        lo_u = max(0, u - span // 2)
        hi_u = min(S - 1, lo_u + span - 1)
        lo_u = hi_u - span + 1
        donor_k_band = K[:, lo_u:hi_u + 1, :].clone()

        def fK_swap(x: torch.Tensor) -> torch.Tensor:
            y = x.clone()
            if y.dim() == 2:
                return y
            y[:, lo_t:hi_t + 1, :] = donor_k_band
            return y
        with kvq_inplace_modifiers(model, fK=fK_swap):
            k_step, k_loss = evaluate_pair(ids, focus_t=t)
        ids_adv = ids.clone()
        a = max(0, t - adv_len)
        b = t
        if a < b:
            ids_adv[0, a:b] = distractor_id
        adv_step, adv_loss = evaluate_pair(ids_adv, focus_t=t)
        q_dstep = q_step - base_step
        q_dloss = q_loss - base_loss
        k_dstep = k_step - base_step
        k_dloss = k_loss - base_loss
        a_dstep = adv_step - base_step
        a_dloss = adv_loss - base_loss
        rows.append(','.join(map(str, [t, u, int(tgt[t].item()), base_step, base_loss, q_step, q_loss, q_dstep, q_dloss, k_step, k_loss, k_dstep, k_dloss, adv_step, adv_loss, a_dstep, a_dloss])))
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, 'counterfactuals_complete.csv'), 'w') as fh:
        fh.write('\n'.join(rows))
    header_cols = rows[0].split(',')
    col = {name: i for i, name in enumerate(header_cols)}

    def _avg(col_name: str) -> float:
        i = col[col_name]
        vals = []
        for r in rows[1:]:
            parts = r.split(',')
            if len(parts) <= i:
                continue
            try:
                vals.append(float(parts[i]))
            except ValueError:
                pass
        return float(np.mean(vals)) if vals else 0.0
    with open(os.path.join(OUTDIR, 'summary.txt'), 'w') as f:
        f.write('Type-8 Counterfactuals (complete)\n')
        f.write(f'hard_steps: {max(0, len(rows) - 1)}\n')
        f.write(f"avg_base_step_ce: {_avg('base_step_ce'):.4f}\n")
        f.write(f"avg_qswap_delta_step: {_avg('qswap_delta_step'):.4f}\n")
        f.write(f"avg_kswap_delta_step: {_avg('kswap_delta_step'):.4f}\n")
        f.write(f"avg_adv_delta_step:   {_avg('adv_delta_step'):.4f}\n")
        f.write(f"avg_qswap_delta_loss: {_avg('qswap_delta_loss'):.4f}\n")
        f.write(f"avg_kswap_delta_loss: {_avg('kswap_delta_loss'):.4f}\n")
        f.write(f"avg_adv_delta_loss:   {_avg('adv_delta_loss'):.4f}\n")