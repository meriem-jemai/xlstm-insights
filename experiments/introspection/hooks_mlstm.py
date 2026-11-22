"""
Records the gates of the mLSTM while the model runs, and allows switching gates off per head (ablation).

In the mLSTM the input and forget gate are ONE number per head (pre-activations from cell.igate / cell.fgate,
shape [B, S, heads]). Unlike in the sLSTM layer, the names igate / fgate match their roles here.

f: the old version applied a sigmoid to both pre-activations. That is right for the forget gate
   (the library uses logsigmoid), but the mLSTM input gate is EXPONENTIAL, so sigmoid(iraw) is not the gate.
   Now the stabilized gates are computed from the pre-activations with the stabilizer m, exactly like in the
   recurrent form of the mLSTM:
       m_t = max(logsigmoid(fraw_t) + m_(t-1), iraw_t)
       i'_t = exp(iraw_t - m_t),  f'_t = exp(logsigmoid(fraw_t) + m_(t-1) - m_t)      (both between 0 and 1)
   so the values are comparable to the sLSTM gates (hooks_slstm.py).

After a forward pass:
    IGATE_OUTS[0], FGATE_OUTS[0]          [B, S, heads]  stabilized gates i', f'
    IGATE_RAW_OUTS[0], FGATE_RAW_OUTS[0]  [B, S, heads]  pre-activations iraw, fraw
    LOGITS_OUT[0]                         model output
"""
import contextlib

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from config import NUM_HEADS, LOGIT_SAT, GATE_EPS

# ===== Public buffers (filled once per forward) =====
IGATE_OUTS: List[torch.Tensor] = []      # [B,S,NH] stabilized input gate i'
FGATE_OUTS: List[torch.Tensor] = []      # [B,S,NH] stabilized forget gate f'
IGATE_RAW_OUTS: List[torch.Tensor] = []  # [B,S,NH] pre-activation iraw
FGATE_RAW_OUTS: List[torch.Tensor] = []  # [B,S,NH] pre-activation fraw
LOGITS_OUT: List[torch.Tensor] = []

# Internal accumulators for per-step mode
_I_STEPS_RAW: List[torch.Tensor] = []
_F_STEPS_RAW: List[torch.Tensor] = []

# --------------------------------------------------------------------
def _clear_all():
    IGATE_OUTS.clear(); FGATE_OUTS.clear()
    IGATE_RAW_OUTS.clear(); FGATE_RAW_OUTS.clear()
    LOGITS_OUT.clear()
    _I_STEPS_RAW.clear(); _F_STEPS_RAW.clear()


def _find_cell(model: torch.nn.Module):
    for m in model.modules():
        if m.__class__.__name__ == "mLSTMCell":
            return m
    raise RuntimeError("No mLSTMCell found in model")


def stabilized_gates(i_raw: torch.Tensor, f_raw: torch.Tensor):
    """
    n: stabilized gates from the pre-activations [B, S, NH], step by step like the recurrent mLSTM.
    Returns i', f' with the same shape, both between 0 and 1.
    """
    log_f = F.logsigmoid(f_raw)
    m_prev = torch.zeros_like(i_raw[:, 0])  # the state m starts at 0
    i_out, f_out = [], []
    for t in range(i_raw.shape[1]):
        m_t = torch.maximum(log_f[:, t] + m_prev, i_raw[:, t])
        i_out.append(torch.exp(i_raw[:, t] - m_t))
        f_out.append(torch.exp(log_f[:, t] + m_prev - m_t))
        m_prev = m_t
    return torch.stack(i_out, dim=1), torch.stack(f_out, dim=1)

# --------------------------------------------------------------------
def register_gate_hooks(model: torch.nn.Module):
    """
    Attach hooks to igate / fgate inside the first mLSTMCell (they output the pre-activations).
    """
    _clear_all()
    cell = _find_cell(model)

    # per-step hook maker
    def make_hook(store):
        def hook(_m, _inp, out):
            with torch.no_grad():
                t = out.detach()
                if t.dim() == 3:   # [B,S,NH]
                    store.append(t.cpu())
                elif t.dim() == 2: # [B,NH]
                    store.append(t.unsqueeze(1).cpu())
        return hook

    h_i = cell.igate.register_forward_hook(make_hook(_I_STEPS_RAW))
    h_f = cell.fgate.register_forward_hook(make_hook(_F_STEPS_RAW))

    # finalize after full forward
    def post_hook(_m, _in, _out):
        if not _I_STEPS_RAW or not _F_STEPS_RAW: return
        i_seq = torch.cat(_I_STEPS_RAW, dim=1)  # [B,S,NH]
        f_seq = torch.cat(_F_STEPS_RAW, dim=1)
        IGATE_RAW_OUTS.append(i_seq)
        FGATE_RAW_OUTS.append(f_seq)
        i_stab, f_stab = stabilized_gates(i_seq, f_seq)  # f: real gate values instead of sigmoid
        IGATE_OUTS.append(i_stab)
        FGATE_OUTS.append(f_stab)
        _I_STEPS_RAW.clear(); _F_STEPS_RAW.clear()

    h_post = model.register_forward_hook(post_hook)

    # also capture model output (logits) at top level
    def logits_hook(_m, _inp, out):
        with torch.no_grad():
            y = out[0] if isinstance(out, (tuple, list)) else out
            LOGITS_OUT.append(y.detach().cpu())
    h_logits = model.register_forward_hook(logits_hook)

    return [h_i, h_f, h_post, h_logits]


@contextlib.contextmanager
def ablate_heads(model: torch.nn.Module, heads: List[int], mode: str = "freeze"):
    """
    n: switch off the gates of some heads during the forward passes inside the `with` block.
        freeze: input gate closed (iraw = -9), forget gate open (fraw = +9) -> memory stays frozen
        erase:  input and forget gate closed (both -9)                      -> memory is cleared
    """
    cell = _find_cell(model)
    i_val, f_val = (-9.0, 9.0) if mode == "freeze" else (-9.0, -9.0)

    def make(value):
        def hook(_m, _inp, out):
            out = out.clone()
            out[..., heads] = value  # the last dimension is the head
            return out
        return hook

    handles = [cell.igate.register_forward_hook(make(i_val)), cell.fgate.register_forward_hook(make(f_val))]
    try:
        yield
    finally:
        for h in handles:
            h.remove()

# --------------------------------------------------------------------
def remove_gate_hooks(handles):
    for h in handles:
        try: h.remove()
        except Exception: pass
    _clear_all()

# --------------------------------------------------------------------
def extract_series_per_head() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Returns:
      i_all [S], f_all [S]        : layer mean gate activations
      i_ph [S,H], f_ph [S,H]      : per-head means
      head_dim (int)
    """
    assert IGATE_OUTS and FGATE_OUTS, "No gates captured; run a forward after registering hooks."
    i_t = IGATE_OUTS[0]  # [B,S,NH]
    f_t = FGATE_OUTS[0]
    B,S,HIDDEN = i_t.shape
    assert HIDDEN % NUM_HEADS == 0
    head_dim = HIDDEN // NUM_HEADS

    i_all = i_t.mean(dim=(0,2)).numpy()
    f_all = f_t.mean(dim=(0,2)).numpy()
    i_ph = i_t.view(B,S,NUM_HEADS,head_dim).mean(dim=(0,3)).numpy()
    f_ph = f_t.view(B,S,NUM_HEADS,head_dim).mean(dim=(0,3)).numpy()
    return i_all,f_all,i_ph,f_ph,head_dim

# --------------------------------------------------------------------
def extract_raw_series_per_head() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if IGATE_RAW_OUTS and FGATE_RAW_OUTS:
        i_t = IGATE_RAW_OUTS[0]; f_t = FGATE_RAW_OUTS[0]
    else:
        eps=1e-6
        i_sig = IGATE_OUTS[0].clamp(eps,1-eps)
        f_sig = FGATE_OUTS[0].clamp(eps,1-eps)
        i_t = (i_sig/(1-i_sig)).log(); f_t=(f_sig/(1-f_sig)).log()

    B,S,HIDDEN = i_t.shape
    head_dim = HIDDEN//NUM_HEADS
    i_all=i_t.mean(dim=(0,2)).numpy(); f_all=f_t.mean(dim=(0,2)).numpy()
    i_ph=i_t.view(B,S,NUM_HEADS,head_dim).mean(dim=(0,3)).numpy()
    f_ph=f_t.view(B,S,NUM_HEADS,head_dim).mean(dim=(0,3)).numpy()
    return i_all,f_all,i_ph,f_ph,head_dim

# --------------------------------------------------------------------
def safe_corrcoef(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat,float)
    if mat.ndim==1: mat=mat[None,:]
    M,N=mat.shape
    if M<=1: return np.ones((M,M))
    mu=mat.mean(1,keepdims=True); sd=mat.std(1,keepdims=True); sd[sd==0]=1
    Z=(mat-mu)/sd
    C=(Z@Z.T)/max(1,N-1)
    return np.clip(C,-1,1)

# --------------------------------------------------------------------
def debug_snapshot(tag=""):
    if not IGATE_OUTS or not FGATE_OUTS:
        print("[debug-mlstm] No gates captured yet."); return
    i=IGATE_OUTS[0].numpy(); f=FGATE_OUTS[0].numpy()
    B,S,HIDDEN=i.shape; head_dim=HIDDEN//NUM_HEADS
    print(f"\n[debug-mlstm] {tag} B={B} S={S} H={HIDDEN} heads={NUM_HEADS}")
    print(f"  i mean={i.mean():.4f} std={i.std():.4f}")
    print(f"  f mean={f.mean():.4f} std={f.std():.4f}")
    i_heads=i.mean((0,1)).reshape(NUM_HEADS,head_dim).mean(1)
    f_heads=f.mean((0,1)).reshape(NUM_HEADS,head_dim).mean(1)
    print(f"  i heads {i_heads}")
    print(f"  f heads {f_heads}")
