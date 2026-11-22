"""
Records the mLSTM matrix memory C over time, and the mLSTM's retrieval weights.

f: AUX["attn"] used to be softmax(q?k / sqrt(d)), like in a Transformer. The mLSTM does not use a softmax.
   It reads its memory with   h_t = sum_s w_ts * v_s,   where the real weights come from the gates:
       D_ts = exp( sum_{j=s+1..t} logsigmoid(fraw_j) + iraw_s - m_t )        (decay since s, times write strength)
       w_ts = (q_t?k_s / sqrt(d)) * D_ts / max(|sum_s (q_t?k_s / sqrt(d)) * D_ts|, exp(-m_t))
   This is exactly the parallel form of the official implementation (parallel_stabilized_simple).
   Now:
       AUX["weights"]     the real weights w_ts (can be negative)        [B, S, NH, S]
       AUX["attn"]        |w_ts| normalized to sum 1 per row, i.e. how much each past step contributes,
                          so the entropy / argmax / distance analyses work on the real retrieval
       AUX["softmax_qk"]  the old softmax version, kept only for comparison
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import math
import torch
import numpy as np
from config import NUM_HEADS

LAYER_IN: Optional[torch.Tensor] = None
C_SEQ: Optional[torch.Tensor] = None
MATRIX_LAYER: Optional[torch.nn.Module] = None
ACT_SEQ: Optional[torch.Tensor] = None
AUX: Dict[str, torch.Tensor] = {}
Q_SEQ: Optional[torch.Tensor] = None
K_SEQ: Optional[torch.Tensor] = None
_Q_LIST: List[torch.Tensor] = []
_K_LIST: List[torch.Tensor] = []
_I_LIST: List[torch.Tensor] = []  # n: input gate pre-activations [B, S, NH]
_F_LIST: List[torch.Tensor] = []  # n: forget gate pre-activations [B, S, NH]
_HANDLES: List[torch.utils.hooks.RemovableHandle] = []

def _clear():
    global LAYER_IN, C_SEQ, ACT_SEQ, AUX, Q_SEQ, K_SEQ, _Q_LIST, _K_LIST
    LAYER_IN = None
    C_SEQ = None
    ACT_SEQ = None
    AUX = {}
    Q_SEQ = None
    K_SEQ = None
    _Q_LIST = []
    _K_LIST = []
    _I_LIST.clear()
    _F_LIST.clear()

def register_matrix_hooks(model: torch.nn.Module):
    _clear()
    global MATRIX_LAYER, _HANDLES
    _HANDLES.clear()
    MATRIX_LAYER = None
    target = None
    for m in model.modules():
        if m.__class__.__name__ == "mLSTMLayer":
            target = m
            break
    if target is None:
        raise RuntimeError("No mLSTMLayer found in model")
    MATRIX_LAYER = target
    def pre_hook(_m, inp):
        global LAYER_IN, _Q_LIST, _K_LIST
        x = inp[0]
        LAYER_IN = x.detach()
        _Q_LIST = []
        _K_LIST = []
        _I_LIST.clear()
        _F_LIST.clear()
    def _mk_store(which: str):
        def h(_m, _in, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if t.dim() == 2:
                t = t.unsqueeze(1)
            if t.dim() != 3:
                raise RuntimeError(f"{which} output must be [B,S,H] or [B,H]")
            if which == "q":
                _Q_LIST.append(t.detach())
            else:
                _K_LIST.append(t.detach())
        return h
    if not hasattr(target, "q_proj") or not hasattr(target, "k_proj"):
        raise RuntimeError("mLSTMLayer must expose q_proj and k_proj")
    h_pre = target.register_forward_pre_hook(pre_hook)
    hq = target.q_proj.register_forward_hook(_mk_store("q"))
    hk = target.k_proj.register_forward_hook(_mk_store("k"))

    # n: the gate pre-activations of the mLSTM cell (one value per head), needed for the real weights
    cell = next((m for m in target.modules() if m.__class__.__name__ == "mLSTMCell"), None)
    if cell is None:
        raise RuntimeError("No mLSTMCell inside the mLSTMLayer")

    def _mk_gate(lst):
        def h(_m, _in, out):
            t = out.detach()
            lst.append(t.unsqueeze(1) if t.dim() == 2 else t)
        return h

    hi = cell.igate.register_forward_hook(_mk_gate(_I_LIST))
    hf = cell.fgate.register_forward_hook(_mk_gate(_F_LIST))
    _HANDLES.extend([h_pre, hq, hk, hi, hf])
    return _HANDLES


@torch.no_grad()
def mlstm_retrieval_weights(Qv: torch.Tensor, Kv: torch.Tensor, i_raw: torch.Tensor, f_raw: torch.Tensor,
                            eps: float = 1e-6) -> torch.Tensor:
    """
    n: the real weights w_ts with which the mLSTM reads step s at time t (parallel form of the official code).
    Qv, Kv: [B, S, NH, DH]   i_raw, f_raw: [B, S, NH]   ->   w: [B, S(t), NH, S(s)]
    """
    B, S, NH, DH = Qv.shape
    q = Qv.permute(0, 2, 1, 3).float()                    # [B, NH, S, DH]
    k = Kv.permute(0, 2, 1, 3).float() / DH ** 0.5        # keys scaled by 1/sqrt(d)
    i = i_raw.permute(0, 2, 1).float()                    # [B, NH, S]
    log_f = torch.nn.functional.logsigmoid(f_raw.permute(0, 2, 1).float())

    # log of the total forget factor between s and t: sum_{j=s+1..t} log f_j
    cum = torch.cat([torch.zeros_like(log_f[..., :1]), torch.cumsum(log_f, dim=-1)], dim=-1)  # [B, NH, S+1]
    log_fg = cum[..., 1:, None] - cum[..., None, 1:]                                         # [B, NH, t, s]
    causal = torch.ones(S, S, dtype=torch.bool, device=q.device).tril()
    log_D = torch.where(causal, log_fg + i[..., None, :], torch.tensor(float("-inf"), device=q.device))

    m = log_D.max(dim=-1, keepdim=True).values            # stabilizer per row
    D = torch.exp(log_D - m)
    C = (q @ k.transpose(-2, -1)) * D                     # unnormalized weights
    normalizer = torch.maximum(C.sum(-1, keepdim=True).abs(), torch.exp(-m))
    w = C / (normalizer + eps)
    return w.permute(0, 2, 1, 3)                          # [B, S, NH, S]

@torch.no_grad()
def replay_cell_sequence() -> torch.Tensor:
    global LAYER_IN, C_SEQ, ACT_SEQ, MATRIX_LAYER, AUX, Q_SEQ, K_SEQ, _Q_LIST, _K_LIST
    if MATRIX_LAYER is None:
        raise RuntimeError("MATRIX_LAYER not set. Call register_matrix_hooks(model) first.")
    if LAYER_IN is None:
        raise RuntimeError("LAYER_IN is None. Run a normal forward once to capture the layer input.")
    if len(_Q_LIST) == 0 or len(_K_LIST) == 0:
        raise RuntimeError("Q/K not captured. Ensure a full forward after register_matrix_hooks.")
    Q_SEQ = torch.cat(_Q_LIST, dim=1)
    K_SEQ = torch.cat(_K_LIST, dim=1)
    layer = MATRIX_LAYER
    x = LAYER_IN
    B, S, D = x.shape
    if Q_SEQ.size(0) != B or Q_SEQ.size(1) != S:
        raise RuntimeError(f"Q/K batch or seq mismatch: Q={tuple(Q_SEQ.shape)} vs x={[B,S,D]}")
    H = Q_SEQ.size(2)
    nh = int(NUM_HEADS)
    if H % nh != 0:
        raise RuntimeError(f"hidden size {H} not divisible by NUM_HEADS={nh}")
    dh = H // nh
    Qv = Q_SEQ.view(B, S, nh, dh)
    Kv = K_SEQ.view(B, S, nh, dh)
    if len(_I_LIST) == 0 or len(_F_LIST) == 0:
        raise RuntimeError("Gate pre-activations not captured. Ensure a full forward after register_matrix_hooks.")
    I_SEQ = torch.cat(_I_LIST, dim=1)  # taken before the replay below adds more entries
    F_SEQ = torch.cat(_F_LIST, dim=1)
    mlstm_state = None
    conv_state = None
    C_frames = []
    Y_frames = []
    for t in range(S):
        y, states = layer.step(x[:, t:t + 1, :], mlstm_state=mlstm_state, conv_state=conv_state)
        mlstm_state = states["mlstm_state"]
        conv_state = states.get("conv_state", None)
        C_t = mlstm_state[0]
        if C_t.dim() != 4:
            raise RuntimeError(f"Expected mlstm_state[0] to be [B, NH, DH, DH], got {tuple(C_t.shape)}")
        if y.dim() != 3 or y.size(1) != 1:
            raise RuntimeError(f"Expected step y [B,1,H], got {tuple(y.shape)}")
        C_frames.append(C_t.unsqueeze(1))
        Y_frames.append(y.detach())
    C_SEQ = torch.cat(C_frames, dim=1)
    ACT_SEQ = torch.cat(Y_frames, dim=1)
    # f: the real mLSTM retrieval weights instead of a softmax (see the explanation at the top)
    weights = mlstm_retrieval_weights(Qv, Kv, I_SEQ, F_SEQ)
    contribution = weights.abs()
    contribution = contribution / contribution.sum(-1, keepdim=True).clamp_min(1e-12)

    scale = 1.0 / float(dh) ** 0.5  # old softmax version, only kept for comparison
    scores = torch.einsum("bthd,bshd->bths", Qv, Kv) * scale
    mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask.view(1, S, 1, S), float("-inf"))

    AUX = {"attn": contribution.detach(), "weights": weights.detach(),
           "softmax_qk": torch.softmax(scores, dim=-1).detach()}
    print(f"[matrix] after_replay shape=[B={B}, S={S}, NH={nh}, DH={dh}x{dh}]  mean={C_SEQ.mean().item():.4f} std={C_SEQ.std().item():.4f}")
    return C_SEQ

def remove_matrix_hooks():
    global _HANDLES
    for h in _HANDLES:
        try:
            h.remove()
        except Exception:
            pass
    _HANDLES.clear()

def extract_c_series_per_head(group: str = "proj") -> Tuple[np.ndarray, np.ndarray, int]:
    if C_SEQ is None:
        raise RuntimeError("No C sequence captured. Call replay_cell_sequence() first.")
    C = C_SEQ
    if C.dim() != 5:
        raise RuntimeError(f"Expected C_SEQ to be [B, S, NH, DH, DH], got {tuple(C.shape)}")
    B, S, NH, DH, DH2 = C.shape
    if DH != DH2:
        raise RuntimeError(f"Expected square matrices DH x DH, got {DH} x {DH2}")
    C_layer = C.mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    C_ph = C.mean(dim=(0, 3, 4)).detach().cpu().numpy()
    return C_layer, C_ph, DH

def safe_corrcoef(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    if mat.ndim == 1:
        mat = mat[None, :]
    M, N = mat.shape
    if M <= 1:
        return np.ones((M, M))
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True)
    sd[sd == 0.0] = 1.0
    Z = (mat - mu) / sd
    C = (Z @ Z.T) / max(1, N - 1)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C

def debug_snapshot(tag: str = ""):
    if C_SEQ is None:
        print("[matrix] No C captured yet.")
        return
    C = C_SEQ
    if C.dim() == 5:
        B, S, Nh, Dh, Dh2 = C.shape
        msg = (f"[matrix] {tag} shape=[B={B}, S={S}, NH={Nh}, DH={Dh}x{Dh2}]  mean={C.mean().item():.4f} std={C.std().item():.4f}")
    else:
        msg = f"[matrix] {tag} unsupported C shape {tuple(C.shape)}"
    print(msg)
