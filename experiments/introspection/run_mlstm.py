import sys, importlib, inspect, torch
from config import print_config_summary, MODEL_BUILDER, MODEL_TAG, DEVICE, RESULTS_DIR, CKPT_PATH, EMBED_DIM, NUM_HEADS, USE_TRAINED
from data_utils import load_data

import hooks_mlstm as hooks
sys.modules["hooks"] = hooks

import type1_dynamics, type2_saturation, type3_alignment, type4_memory_retention, type6_predictive_power, type7_ablations

import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
    "grid.color": "#DDDDDD", "grid.linestyle": "-", "grid.linewidth": 0.6, "grid.alpha": 0.9,
    "axes.grid": True, "axes.axisbelow": True,
    "figure.autolayout": False,
    "mathtext.default": "regular",
})
import matplotlib as mpl
from matplotlib.figure import Figure
mpl.rcParams.update({
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
_ORIG_SAVEFIG = Figure.savefig
def _savefig_force_pdf(self, fname, *args, **kwargs):
    if isinstance(fname, str) and fname.lower().endswith(".png"):
        fname = fname[:-4] + ".pdf"
    kwargs.setdefault("format", "pdf")
    kwargs.setdefault("bbox_inches", mpl.rcParams.get("savefig.bbox", "tight"))
    kwargs.setdefault("pad_inches", mpl.rcParams.get("savefig.pad_inches", 0.05))
    return _ORIG_SAVEFIG(self, fname, *args, **kwargs)
Figure.savefig = _savefig_force_pdf

import matplotlib.axes as _mx_axes
__old_imshow = _mx_axes.Axes.imshow
def __imshow_nogrid(self, *args, **kwargs):
    im = __old_imshow(self, *args, **kwargs)
    self.grid(False)
    for _s in self.spines.values():
        _s.set_visible(False)
    return im
_mx_axes.Axes.imshow = __imshow_nogrid
__old_matshow = _mx_axes.Axes.matshow
def __matshow_nogrid(self, *args, **kwargs):
    im = __old_matshow(self, *args, **kwargs)
    self.grid(False)
    for _s in self.spines.values():
        _s.set_visible(False)
    return im
_mx_axes.Axes.matshow = __matshow_nogrid
__old_pcolor = _mx_axes.Axes.pcolormesh
def __pcolor_nogrid(self, *args, **kwargs):
    pc = __old_pcolor(self, *args, **kwargs)
    self.grid(False)
    for _s in self.spines.values():
        _s.set_visible(False)
    return pc
_mx_axes.Axes.pcolormesh = __pcolor_nogrid

def _import_builder(path: str):
    mod, func = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), func)

def _build_with_config(builder, vocab_size, seq_len):
    desired = {"vocab_size": vocab_size, "seq_len": seq_len, "embed_dim": EMBED_DIM, "num_heads": NUM_HEADS}
    sig = inspect.signature(builder)
    kwargs = {k: v for k, v in desired.items() if k in sig.parameters}
    try:
        return builder(**kwargs)
    except TypeError:
        return builder(vocab_size, seq_len)

def main():
    print_config_summary()
    text, ids, stoi = load_data()
    build_model = _import_builder(MODEL_BUILDER)
    model = _build_with_config(build_model, len(stoi), ids.shape[1])
    if USE_TRAINED:
        print(f"[INFO] USE_TRAINED=True => loading checkpoint: {CKPT_PATH!r}")
        state = torch.load(CKPT_PATH, map_location=DEVICE)
        state = state.get("model", state) if isinstance(state, dict) else state
        model.load_state_dict(state, strict=True)
        print("[INFO] Checkpoint loaded (strict=True).")
    else:
        print("[INFO] USE_TRAINED=False => skipping checkpoint.")
    model.to(DEVICE).eval()
    ids = ids.to(next(model.parameters()).device)
    handles = hooks.register_gate_hooks(model)
    with torch.no_grad():
        _ = model(ids)
    try:
        hooks.debug_snapshot("after_forward")
    except Exception:
        pass
    type1_dynamics.run(text, mark_patterns=["Source blocks", "Gap blocks", "Copy blocks"])
    type2_saturation.run()
    type3_alignment.run(text)
    type4_memory_retention.run()
    inp_ids = ids[:, :-1] if ids.size(1) >= 2 else ids
    tgt_ids = ids[:,  1:] if ids.size(1) >= 2 else ids
    type6_predictive_power.run(model, inp_ids.contiguous(), tgt_ids.contiguous())
    type7_ablations.run(model, ids)
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass
    print(f"Done. See: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
