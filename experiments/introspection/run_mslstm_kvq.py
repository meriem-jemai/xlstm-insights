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
_orig_savefig = Figure.savefig
def _savefig_force_pdf(self, fname, *args, **kwargs):
    if isinstance(fname, str) and fname.lower().endswith(".png"):
        fname = fname[:-4] + ".pdf"
    kwargs.setdefault("format", "pdf")
    kwargs.setdefault("bbox_inches", mpl.rcParams.get("savefig.bbox", "tight"))
    kwargs.setdefault("pad_inches", mpl.rcParams.get("savefig.pad_inches", 0.05))
    return _orig_savefig(self, fname, *args, **kwargs)
Figure.savefig = _savefig_force_pdf

import matplotlib.axes as maxes
_old_imshow = maxes.Axes.imshow
def _imshow_nogrid(self, *args, **kwargs):
    im = _old_imshow(self, *args, **kwargs)
    self.grid(False)
    for s in self.spines.values():
        s.set_visible(False)
    return im
maxes.Axes.imshow = _imshow_nogrid
_old_matshow = maxes.Axes.matshow
def _matshow_nogrid(self, *args, **kwargs):
    im = _old_matshow(self, *args, **kwargs)
    self.grid(False)
    for s in self.spines.values():
        s.set_visible(False)
    return im
maxes.Axes.matshow = _matshow_nogrid
_old_pcolormesh = maxes.Axes.pcolormesh
def _pcolormesh_nogrid(self, *args, **kwargs):
    pc = _old_pcolormesh(self, *args, **kwargs)
    self.grid(False)
    for s in self.spines.values():
        s.set_visible(False)
    return pc
maxes.Axes.pcolormesh = _pcolormesh_nogrid

import sys, importlib, inspect, torch
from config import print_config_summary, MODEL_BUILDER, DEVICE, RESULTS_DIR, CKPT_PATH, EMBED_DIM, NUM_HEADS
from data_utils import load_data

import hooks_kvq as kvq_hooks
sys.modules["hooks"] = kvq_hooks  

import kvq_type1
import kvq_type2_causal
import kvq_type3_sensitivity
import kvq_type5_lag_buckets
import kvq_type6_value_content
import kvq_type7_probes_clusters
import kvq_type8_counterfactuals
import kvq_type9_toggles
import kvq_more  

import hooks_matrix as matrix_hooks  

GROUPING = "cell"

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
    model = _build_with_config(_import_builder(MODEL_BUILDER), len(stoi), ids.shape[1])

    print(f"[INFO] loading checkpoint: {CKPT_PATH}")
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=True)
    print("[INFO] checkpoint loaded")

    model.to(DEVICE).eval()
    ids = ids.to(next(model.parameters()).device)

    kvq_hooks.set_default_grouping(GROUPING)
    kvq_handles = kvq_hooks.register_kvq_hooks(model)
    with torch.no_grad():
        _ = model(ids)
        from hooks_kvq import extract_kvq_norm_series
        for grp in ("proj", "cell"):
            qL, kL, vL, qPH, kPH, vPH, nh, hd = extract_kvq_norm_series(grouping=grp)
            print(f"[kvq] baseline captured ({grp}): S={qPH.shape[0]} heads={nh} head_dim={hd}")

    print("[RUN] kvq_type1"); kvq_type1.run()
    print("[RUN] kvq_type2_causal"); kvq_type2_causal.run(model, ids)
    print("[RUN] kvq_type3_sensitivity"); kvq_type3_sensitivity.run(model, ids)
    print("[RUN] kvq_type5_lag_buckets"); kvq_type5_lag_buckets.run(model, ids)
    print("[RUN] kvq_type6_value_content"); kvq_type6_value_content.run(model, ids)
    print("[RUN] kvq_type7_probes_clusters"); kvq_type7_probes_clusters.run(text)
    print("[RUN] kvq_type8_counterfactuals"); kvq_type8_counterfactuals.run(model, ids, topN=100, k_window=24, adv_len=64)
    print("[RUN] kvq_type9_toggles"); kvq_type9_toggles.run(model, ids)

    print("[kvq_more] capturing matrix stream")
    mat_handles = matrix_hooks.register_matrix_hooks(model)
    with torch.no_grad():
        _ = model(ids)
    with torch.no_grad():
        _ = matrix_hooks.replay_cell_sequence()
    try:
        matrix_hooks.debug_snapshot("after_replay")
    except Exception:
        pass

    print("[RUN] kvq_more")
    kvq_more.run(model, ids, text=None)

    for h in kvq_handles:
        try: h.remove()
        except Exception: pass
    for h in mat_handles:
        try: h.remove()
        except Exception: pass

    print(f"Done. See: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
