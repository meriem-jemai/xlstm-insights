"""
Records the gates of the sLSTM while the model runs, and allows switching gates off per head (ablation).

f: the old version hooked the layer's projections `igate` / `fgate` and applied a sigmoid. That was wrong
   in three ways:
     1. the official layer passes its projections to the cell in the order (fgate, igate, z, o), but the cell
        reads them as (i, f, z, o). So the module called `igate` is really the forget gate and vice versa.
     2. the projections only contain W·x. The real gate pre-activation also contains R·h and the bias.
     3. the sLSTM input gate is exponential and the forget gate uses logsigmoid plus the stabilizer m,
        so sigmoid(projection) is not the gate value the model uses.
   Now the gates are taken directly from the cell's computation (`slstm_forward_pointwise` of the vanilla
   backend; the cell keeps it in `cell.pointwise`, which is temporarily replaced by a recording wrapper),
   so they are exactly the values the model uses:
       i' = min(exp(iraw - m), 1),  f' = min(exp(logsigmoid(fraw) + m_prev - m), 1)   (both between 0 and 1)
   This needs the "vanilla" sLSTM backend (the default when building models for introspection).

After a forward pass:
    IGATE_OUTS[0], FGATE_OUTS[0]          [B, S, H]  stabilized gates i', f' (between 0 and 1)
    IGATE_RAW_OUTS[0], FGATE_RAW_OUTS[0]  [B, S, H]  full pre-activations iraw, fraw (W·x + R·h + b)
    LOGITS_OUT[0]                         model output
"""
import contextlib
from typing import List, Optional, Tuple

import numpy as np
import torch

import xlstm.blocks.slstm.src.vanilla as vanilla
from config import NUM_HEADS

IGATE_OUTS: List[torch.Tensor] = []
FGATE_OUTS: List[torch.Tensor] = []
IGATE_RAW_OUTS: List[torch.Tensor] = []
FGATE_RAW_OUTS: List[torch.Tensor] = []
LOGITS_OUT: List[torch.Tensor] = []

# per-time-step values, collected while the cell runs, joined after the forward pass
_STEPS = {"i": [], "f": [], "iraw": [], "fraw": []}
_ORIGINAL = vanilla.slstm_pointwise_function_registry["slstm"]  # the library's own function
_ABLATION: dict = {}  # {"heads": [...], "mode": "freeze" | "erase"} while an ablation is active


def _clear_all():
    IGATE_OUTS.clear(); FGATE_OUTS.clear()
    IGATE_RAW_OUTS.clear(); FGATE_RAW_OUTS.clear()
    LOGITS_OUT.clear()
    for v in _STEPS.values():
        v.clear()


def _head_slices(H: int):
    hd = H // NUM_HEADS
    return [slice(h * hd, (h + 1) * hd) for h in range(NUM_HEADS)]


def _recording_pointwise(Wx, Ry, b, states, constants):
    """
    Wraps the library's slstm_forward_pointwise (called once per time step).
    raw = Wx + Ry + b is laid out as [i | f | z | o], each of size H (heads one after another).
    """
    H = states.shape[-1]
    if _ABLATION:
        # n: ablation: set the gate pre-activations of the chosen heads to fixed values by adjusting Wx,
        #    so that raw = Wx + Ry + b takes exactly the wanted value
        Wx = Wx.clone()
        raw = Wx + Ry + b
        i_val, f_val = (-9.0, 9.0) if _ABLATION["mode"] == "freeze" else (-9.0, -9.0)
        for h in _ABLATION["heads"]:
            s = _head_slices(H)[h]
            i_idx = slice(0 * H + s.start, 0 * H + s.stop)  # input gate slot
            f_idx = slice(1 * H + s.start, 1 * H + s.stop)  # forget gate slot
            Wx[:, i_idx] += i_val - raw[:, i_idx]
            Wx[:, f_idx] += f_val - raw[:, f_idx]

    new_states, gates = _ORIGINAL(Wx, Ry, b, states, constants)

    with torch.no_grad():
        raw = (Wx + Ry + b).view(Wx.shape[0], 4, -1)
        _STEPS["iraw"].append(raw[:, 0].detach().cpu())
        _STEPS["fraw"].append(raw[:, 1].detach().cpu())
        _STEPS["i"].append(gates[0].detach().cpu())  # stabilized input gate i'
        _STEPS["f"].append(gates[1].detach().cpu())  # stabilized forget gate f'
    return new_states, gates


def _slstm_cells(model: torch.nn.Module):
    # the vanilla sLSTM cell stores its pointwise function in `.pointwise` when it is created
    cells = [m for m in model.modules() if hasattr(m, "pointwise")]
    if not cells:
        raise RuntimeError("No vanilla sLSTM cell found. Build the model with backend='vanilla'.")
    return cells


class _PatchHandle:
    """Undo object, so the patch can be removed like a normal PyTorch hook (handle.remove())."""

    def __init__(self, cells):
        self.cells = cells

    def remove(self):
        for c in self.cells:
            c.pointwise = _ORIGINAL


def _finalize(_m, _inp, out):
    # after a full forward pass: join the per-step values into [B, S, H]
    if not _STEPS["i"] and not IGATE_OUTS:
        raise RuntimeError("No sLSTM steps were recorded. Is the model built with backend='vanilla'?")
    if _STEPS["i"]:
        IGATE_OUTS.append(torch.stack(_STEPS["i"], dim=1))
        FGATE_OUTS.append(torch.stack(_STEPS["f"], dim=1))
        IGATE_RAW_OUTS.append(torch.stack(_STEPS["iraw"], dim=1))
        FGATE_RAW_OUTS.append(torch.stack(_STEPS["fraw"], dim=1))
        for v in _STEPS.values():
            v.clear()
    with torch.no_grad():
        y = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(y, torch.Tensor):
            LOGITS_OUT.append(y.detach().cpu())


def register_gate_hooks(model: torch.nn.Module):
    _clear_all()
    cells = _slstm_cells(model)
    # f: the cell keeps its own reference to the function (cell.pointwise), so we replace it there.
    #    Replacing it in the library's registry had no effect, because the cell copied it at creation.
    for c in cells:
        c.pointwise = _recording_pointwise
    return [_PatchHandle(cells), model.register_forward_hook(_finalize)]


def remove_gate_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass
    _clear_all()


@contextlib.contextmanager
def ablate_heads(model: torch.nn.Module, heads: List[int], mode: str = "freeze"):
    """
    n: switch off the gates of some heads during the forward passes inside the `with` block.
        freeze: input gate closed, forget gate open  -> the memory of the head stays frozen
        erase:  input and forget gate closed           -> the memory of the head is cleared at every step
    """
    cells = _slstm_cells(model)
    previous = [c.pointwise for c in cells]
    _ABLATION.update(heads=list(heads), mode=mode)
    for c in cells:
        c.pointwise = _recording_pointwise
    try:
        yield
    finally:
        _ABLATION.clear()
        for c, fn in zip(cells, previous):
            c.pointwise = fn


def _per_head(t: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, int]:
    B, S, H = t.shape
    assert H % NUM_HEADS == 0
    hd = H // NUM_HEADS
    return t.mean(dim=(0, 2)).numpy(), t.view(B, S, NUM_HEADS, hd).mean(dim=(0, 3)).numpy(), hd


def extract_series_per_head() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Gate values: i_all [S], f_all [S] (layer mean), i_ph [S, heads], f_ph [S, heads], head_dim."""
    assert IGATE_OUTS and FGATE_OUTS, "No gates captured, run model(ids) after register_gate_hooks"
    i_all, i_ph, hd = _per_head(IGATE_OUTS[0])
    f_all, f_ph, _ = _per_head(FGATE_OUTS[0])
    return i_all, f_all, i_ph, f_ph, hd


def extract_raw_series_per_head() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Same as extract_series_per_head, but for the pre-activations iraw, fraw."""
    assert IGATE_RAW_OUTS and FGATE_RAW_OUTS, "No raw gates captured, run model(ids) after register_gate_hooks"
    i_all, i_ph, hd = _per_head(IGATE_RAW_OUTS[0])
    f_all, f_ph, _ = _per_head(FGATE_RAW_OUTS[0])
    return i_all, f_all, i_ph, f_ph, hd


def safe_corrcoef(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, float)
    if mat.ndim == 1:
        mat = mat[None, :]
    M, N = mat.shape
    if M <= 1:
        return np.ones((M, M))
    mu = mat.mean(1, keepdims=True)
    sd = mat.std(1, keepdims=True)
    sd[sd == 0] = 1
    Z = (mat - mu) / sd
    C = Z @ Z.T / max(1, N - 1)
    return np.clip(C, -1, 1)


def debug_snapshot(tag=""):
    if not IGATE_OUTS or not FGATE_OUTS:
        print("[debug-slstm] No gates captured yet")
        return
    i = IGATE_OUTS[0].numpy()
    f = FGATE_OUTS[0].numpy()
    B, S, H = i.shape
    head_dim = H // NUM_HEADS
    print(f"\n[debug-slstm] {tag} B={B} S={S} H={H} heads={NUM_HEADS}")
    print(f"  i mean={i.mean():.4f} std={i.std():.4f}")
    print(f"  f mean={f.mean():.4f} std={f.std():.4f}")
    print(f"  i heads {i.mean((0, 1)).reshape(NUM_HEADS, head_dim).mean(1)}")
    print(f"  f heads {f.mean((0, 1)).reshape(NUM_HEADS, head_dim).mean(1)}")
