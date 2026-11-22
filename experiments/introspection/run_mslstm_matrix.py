import sys, importlib, inspect, torch
from config import print_config_summary, MODEL_BUILDER, DEVICE, RESULTS_DIR, CKPT_PATH, EMBED_DIM, NUM_HEADS, USE_TRAINED
from data_utils import load_data

import hooks_matrix as hooks
sys.modules['hooks'] = hooks

import matrix_type5_lifespan           

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


import matrix_type2_spectrum          
import matrix_type3_redundancy         
#import matrix_type4_heads              
import matrix_type6_alignment          
import matrix_type7_logits             
#import matrix_type10_semantic_axes     
import matrix_type11_attn_vs_spectrum as B11
import matrix_type12_leadlag          as B12
#import neuron_type14_token_means as N14
#import neuron_type15_corr_saliency as N15
#import neuron_type16_max_examples as N16


def _import_builder(path: str):
    mod, func = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), func)

def _build_with_config(builder, vocab_size, seq_len):
    desired = {"vocab_size": vocab_size, "seq_len": seq_len,
               "embed_dim": EMBED_DIM, "num_heads": NUM_HEADS}
    sig = inspect.signature(builder)
    kwargs = {k: v for k, v in desired.items() if k in sig.parameters}
    try:
        return builder(**kwargs)
    except TypeError:
        return builder(vocab_size, seq_len)

def main():
    print_config_summary()

    # 1) Data
    text, ids, stoi = load_data()

    # 2) Model
    build_model = _import_builder(MODEL_BUILDER)
    model = _build_with_config(build_model, len(stoi), ids.shape[1])

    # 3) Checkpoint
    if USE_TRAINED:
        print(f"[INFO] USE_TRAINED=True => loading checkpoint: {CKPT_PATH!r}")
        state = torch.load(CKPT_PATH, map_location=DEVICE)  
        state = state.get("model", state) if isinstance(state, dict) else state
        model.load_state_dict(state, strict=True)         
        print("[INFO] Checkpoint loaded (strict=True).")
    else:
        print("[INFO] USE_TRAINED=False => skipping checkpoint (random init).")

    model.to(DEVICE).eval()
    ids = ids.to(next(model.parameters()).device)

    handles = hooks.register_matrix_hooks(model)
    with torch.no_grad():
        _ = model(ids)

    with torch.no_grad():
        _ = hooks.replay_cell_sequence()
    try:
        hooks.debug_snapshot("after_replay")
    except Exception:
        pass

    print("[RUN] A2 spectrum");       matrix_type2_spectrum.run()
    print("[RUN] A3 redundancy");     matrix_type3_redundancy.run()
    print("[RUN] A5 lifespan");       matrix_type5_lifespan.run()
    print("[RUN] A6 alignment");      matrix_type6_alignment.run()
    print("[RUN] A7 logits");         matrix_type7_logits.run(model, ids, text=text)
    print("[RUN] B11 attn vs spectrum"); B11.run()
    print("[RUN] B12 lead–lag");          B12.run()

    #print("[RUN] N14 token means");    N14.run(text)
    #print("[RUN] N15 corr/saliency");  N15.run(model, ids, text=text)
    #print("[RUN] N16 pattern stats");  N16.run(model, ids, text)

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    print(f"Done. See: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
