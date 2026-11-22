from __future__ import annotations
import contextlib
from typing import Callable, Optional, Tuple, Dict, List
import torch
import torch.nn.functional as F
import numpy as np
import hooks_kvq as hk

def shift_inp_tgt(ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if ids.size(1) >= 2:
        return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
    return ids, ids

def eval_loss(model: torch.nn.Module, ids: torch.Tensor, ignore_index: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(ids)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        inp, tgt = shift_inp_tgt(ids)
        logits = logits[:, :inp.size(1), :]
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B*T, V), tgt.reshape(B*T),
                               reduction="mean",
                               ignore_index=(-100 if ignore_index is None else ignore_index))
        pred = logits.argmax(-1)
        acc = (pred == tgt).float().mean()
        ppl = float(torch.exp(loss))
        return {"loss": float(loss), "ppl": ppl, "acc": float(acc)}

def compute_attention(Q: torch.Tensor, K: torch.Tensor, head_dim: int, causal: bool = True, eps: float = 1e-9) -> torch.Tensor:
    B, S, H = Q.shape
    nh = H // head_dim
    q = Q.view(B, S, nh, head_dim).transpose(1, 2)
    k = K.view(B, S, nh, head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(head_dim + 1e-9)
    if causal:
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    A = torch.softmax(scores, dim=-1)
    A = torch.where(torch.isfinite(A), A, torch.zeros_like(A))
    return A

def attention_entropy(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(A.clamp_min(eps) * torch.log(A.clamp_min(eps))).sum(dim=-1)

def attention_gini(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return 1.0 - (A.clamp_min(eps) ** 2).sum(dim=-1)

def _find_qkv_modules(layer) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    assert hasattr(layer, "q_proj") and hasattr(layer, "k_proj") and hasattr(layer, "v_proj")
    return layer.q_proj, layer.k_proj, layer.v_proj

@contextlib.contextmanager
def kvq_inplace_modifiers(model: torch.nn.Module,
                          fQ: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                          fK: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                          fV: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
    assert hk.KVQ_LAYER is not None
    q_mod, k_mod, v_mod = _find_qkv_modules(hk.KVQ_LAYER)
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _mk_hook(fn):
        def _hook(_m, _inp, out):
            if fn is None:
                return None
            y = out
            if isinstance(y, (tuple, list)):
                y = y[0]
            if y.dim() == 2:
                y2 = fn(y.unsqueeze(1))
                return y2.squeeze(1)
            return fn(y)
        return _hook

    if fQ is not None:
        handles.append(q_mod.register_forward_hook(_mk_hook(fQ)))
    if fK is not None:
        handles.append(k_mod.register_forward_hook(_mk_hook(fK)))
    if fV is not None:
        handles.append(v_mod.register_forward_hook(_mk_hook(fV)))
    try:
        yield
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

def shuffle_time(x: torch.Tensor) -> torch.Tensor:
    B, S, H = x.shape
    idx = torch.argsort(torch.rand(B, S, device=x.device), dim=1)
    out = x.clone()
    for b in range(B):
        out[b] = x[b, idx[b]]
    return out

def randomize_like(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.randn_like(x) * scale

def scale_tensor(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return x * alpha

def add_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma

def get_QKV_first() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hk.Q_OUTS and hk.K_OUTS and hk.V_OUTS
    Q = hk.Q_OUTS[0]
    K = hk.K_OUTS[0].to(Q.device)
    V = hk.V_OUTS[0].to(Q.device)
    return Q, K, V

def infer_layout_from_Q(Q: torch.Tensor, grouping: Optional[str] = None) -> Tuple[int, int]:
    H = Q.shape[-1]
    grp = grouping or hk.DEFAULT_GROUPING
    nh, hd = hk._get_layout_for_grouping(H, grp)
    return nh, hd
