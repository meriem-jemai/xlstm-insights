import torch
from typing import List, Tuple, Optional
import numpy as np
DEFAULT_GROUPING = 'cell'

def set_default_grouping(mode: str):
    assert mode in ('proj', 'cell'), "grouping must be 'proj' or 'cell'"
    global DEFAULT_GROUPING
    DEFAULT_GROUPING = mode
Q_OUTS: List[torch.Tensor] = []
K_OUTS: List[torch.Tensor] = []
V_OUTS: List[torch.Tensor] = []
LOGITS_OUT: List[torch.Tensor] = []
QKV_NUM_HEADS: Optional[int] = None
QKV_HEAD_DIM: Optional[int] = None
KVQ_LAYER: Optional[torch.nn.Module] = None

def _clear():
    Q_OUTS.clear()
    K_OUTS.clear()
    V_OUTS.clear()
    LOGITS_OUT.clear()

def _find_mlstmlayer(model: torch.nn.Module):
    for m in model.modules():
        if hasattr(m, 'q_proj') and hasattr(m, 'k_proj') and hasattr(m, 'v_proj'):
            return m
    return None

def register_kvq_hooks(model: torch.nn.Module):
    _clear()
    layer = _find_mlstmlayer(model)
    if layer is None:
        raise RuntimeError('No mLSTMLayer with q_proj/k_proj/v_proj found.')
    global KVQ_LAYER
    KVQ_LAYER = layer
    global QKV_NUM_HEADS, QKV_HEAD_DIM
    QKV_NUM_HEADS = None
    QKV_HEAD_DIM = None
    if hasattr(layer, 'config'):
        cfg = layer.config
        head_dim = getattr(cfg, 'qkv_proj_blocksize', None)
        inner_dim = getattr(cfg, '_inner_embedding_dim', None)
        if isinstance(head_dim, int) and head_dim > 0 and isinstance(inner_dim, int) and (inner_dim > 0):
            QKV_HEAD_DIM = head_dim
            QKV_NUM_HEADS = int(round(inner_dim / head_dim))

    def _mk_store(lst: List[torch.Tensor]):

        def h(_m, _in, out):
            with torch.no_grad():
                t = out.detach()
                if t.dim() == 2:
                    lst.append(t.unsqueeze(1).cpu())
                elif t.dim() == 3:
                    lst.append(t.cpu())
        return h
    hq = layer.q_proj.register_forward_hook(_mk_store(Q_OUTS))
    hk = layer.k_proj.register_forward_hook(_mk_store(K_OUTS))
    hv = layer.v_proj.register_forward_hook(_mk_store(V_OUTS))

    def _root(_m, _inp, out):
        with torch.no_grad():
            y = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(y, torch.Tensor):
                if y.dim() == 2:
                    y = y.unsqueeze(1)
                LOGITS_OUT.append(y.detach())
    hlog = model.register_forward_hook(_root)
    return [hq, hk, hv, hlog]

def _infer_head_layout(hidden_size: int) -> Tuple[int, int]:
    global QKV_NUM_HEADS, QKV_HEAD_DIM
    if QKV_HEAD_DIM and QKV_NUM_HEADS and (QKV_NUM_HEADS * QKV_HEAD_DIM == hidden_size):
        return (QKV_NUM_HEADS, QKV_HEAD_DIM)
    for hd in (4, 8, 16, 32, 64):
        if hidden_size % hd == 0:
            return (hidden_size // hd, hd)
    return (1, hidden_size)

def _get_layout_for_grouping(hidden_size: int, grouping: str) -> Tuple[int, int]:
    global KVQ_LAYER
    if grouping == 'cell' and KVQ_LAYER is not None and hasattr(KVQ_LAYER, 'config'):
        inner_dim = getattr(KVQ_LAYER.config, '_inner_embedding_dim', None)
        nh = getattr(KVQ_LAYER.config, 'num_heads', None)
        if isinstance(inner_dim, int) and isinstance(nh, int) and (nh > 0) and (inner_dim % nh == 0):
            return (nh, inner_dim // nh)
    return _infer_head_layout(hidden_size)

def extract_kvq_norm_series(grouping: Optional[str]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    if grouping is None:
        grouping = DEFAULT_GROUPING
    assert Q_OUTS and K_OUTS and V_OUTS, 'Run a forward after register_kvq_hooks().'
    Q = Q_OUTS[0].numpy()
    K = K_OUTS[0].numpy()
    V = V_OUTS[0].numpy()
    B, S, HIDDEN = Q.shape
    nh, hd = _get_layout_for_grouping(HIDDEN, grouping)

    def per_head_norm(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xv = X.reshape(B, S, nh, hd)
        norms = np.linalg.norm(Xv, axis=-1)
        Snh = norms.mean(axis=0)
        layer = Snh.mean(axis=1)
        return (layer, Snh)
    q_layer, q_ph = per_head_norm(Q)
    k_layer, k_ph = per_head_norm(K)
    v_layer, v_ph = per_head_norm(V)
    return (q_layer, k_layer, v_layer, q_ph, k_ph, v_ph, nh, hd)

def safe_corrcoef(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, float)
    if mat.ndim == 1:
        mat = mat[0:1, :]
    M, N = mat.shape
    if M <= 1:
        return np.ones((M, M), dtype=float)
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True)
    sd[sd == 0.0] = 1.0
    Z = (mat - mu) / sd
    C = Z @ Z.T / max(1, N - 1)
    return np.clip(C, -1.0, 1.0)