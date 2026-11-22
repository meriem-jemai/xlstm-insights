"""
Head ablation: how much worse does the model get if the gates of one head are switched off?

For every head: switch its input and forget gate off (see `ablate_heads` in hooks_slstm.py / hooks_mlstm.py),
run the model again, and compare the loss with the normal model. A large increase means the head matters.
    freeze: input gate closed, forget gate open -> the head keeps its memory but cannot write anything new
    erase:  input and forget gate closed        -> the head's memory is cleared at every step

f: the old version found the first module with `.igate` and `.fgate`. For the sLSTM this is the layer, whose
   `igate` / `fgate` projections are passed to the cell in swapped order (and contain only W·x), so the
   "input gate" ablation actually changed the forget gate. Now the ablation is done by the hooks module of
   the model, which sets the real gate pre-activations of the chosen head.
"""
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

import hooks  # hooks_slstm or hooks_mlstm, chosen by the run script
from config import NUM_HEADS, RESULTS_DIR

OUTDIR = os.path.join(RESULTS_DIR, "gates_type7_ablations")


@torch.no_grad()
def _forward_logits(model: torch.nn.Module, inp_ids: torch.Tensor):
    out = model(inp_ids)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    return logits


def _split_inputs(ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # inputs = all characters except the last, targets = all except the first (next-character prediction)
    if ids.size(1) >= 2:
        return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
    return ids.contiguous(), ids.contiguous()


def _nll_and_ppl(logits: torch.Tensor, tgt_ids: torch.Tensor) -> Tuple[float, float]:
    B, S, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * S, V), tgt_ids.reshape(B * S), reduction="mean")
    return float(loss.item()), float(torch.exp(loss).item())


@torch.no_grad()
def run(model: torch.nn.Module, ids: torch.Tensor, heads: Optional[List[int]] = None, mode: str = "freeze"):
    print("[type7_ablations] start")
    os.makedirs(OUTDIR, exist_ok=True)
    device = next(model.parameters()).device
    inp_ids, tgt_ids = _split_inputs(ids.to(device))

    # baseline: the normal model
    base_nll, base_ppl = _nll_and_ppl(_forward_logits(model, inp_ids), tgt_ids)

    rows = []
    for h in (heads if heads is not None else range(NUM_HEADS)):
        with hooks.ablate_heads(model, [h], mode=mode):  # c: real gates of head h switched off
            nll, ppl = _nll_and_ppl(_forward_logits(model, inp_ids), tgt_ids)
        rows.append((h, nll, ppl, nll - base_nll, ppl - base_ppl))

    csv_path = os.path.join(OUTDIR, f"type7_head_ablation_{mode}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("head,nll,ppl,delta_nll,delta_ppl,mode\n")
        for h, nll, ppl, dn, dp in rows:
            f.write(f"{h},{nll},{ppl},{dn},{dp},{mode}\n")

    rows_sorted = sorted(rows, key=lambda r: r[3], reverse=True)
    msg = "Top heads by contribution (delta NLL): " + ", ".join(f"h{h}:{dn:.4f}" for h, _, _, dn, _ in rows_sorted[:8])
    print(f"[type7] baseline nll={base_nll:.4f}, ppl={base_ppl:.3f}")
    print("[type7] " + msg)
    print(f"[type7] wrote {csv_path}")
