import os, math, csv
from typing import Optional, Dict, Tuple, List
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS_DIR, NUM_HEADS, EMBED_DIM, PATTERN_PROVIDER
from patterns_common import load_spans_from_provider, spans_dict_to_masks

BASE_OUT = RESULTS_DIR
EXPECTED_NH = int(NUM_HEADS)
PROJ_FACTOR = 2
EXPECTED_TOTAL_PRE = int(EMBED_DIM * PROJ_FACTOR)
EXPECTED_TOTAL_POST = int(EMBED_DIM)
STREAM_MODE = "pre"
Z_SPARSITY_THRESH = 2.0
HEATMAP_NEURONS = 32
HEATMAP_PATTERNS = 24
OUTDIR = os.path.join(BASE_OUT, "neuron_type16_patterns")
os.makedirs(OUTDIR, exist_ok=True)

def _ensure(d): os.makedirs(d, exist_ok=True)

def _survey_module_shapes(model: torch.nn.Module, ids: torch.Tensor) -> Dict[str, list]:
    rec: Dict[str, list] = {}
    hooks = []
    def make(name):
        def h(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            rec[name] = list(t.shape)
        return h
    for name, m in model.named_modules():
        hooks.append(m.register_forward_hook(make(name)))
    with torch.no_grad():
        _ = model(ids)
    for h in hooks: h.remove()
    return rec

def _pick_module_name_by_shape(shapes: Dict[str, list]) -> str:
    if STREAM_MODE == "pre":
        for name, shp in shapes.items():
            if len(shp) == 4 and shp[2] == EXPECTED_NH and (shp[2] * shp[3] == EXPECTED_TOTAL_PRE):
                return name
        for name, shp in shapes.items():
            if len(shp) == 3 and shp[2] == EXPECTED_TOTAL_PRE:
                return name
    else:
        for name, shp in shapes.items():
            if len(shp) == 3 and shp[2] == EXPECTED_TOTAL_POST:
                return name
        for name, shp in shapes.items():
            if len(shp) == 4 and shp[2] == 1 and shp[3] == EXPECTED_TOTAL_POST:
                return name
    return ""

def _resolve_target_layer(model: torch.nn.Module, ids: torch.Tensor) -> str:
    shapes = _survey_module_shapes(model, ids)
    name = _pick_module_name_by_shape(shapes)
    if not name:
        msg  = "[N16] could not locate the stream tensor\n"
        msg += f"STREAM_MODE={STREAM_MODE}\n"
        msg += f"Expected (pre): NH={EXPECTED_NH}, total={EXPECTED_TOTAL_PRE} or (post) total={EXPECTED_TOTAL_POST}\n"
        for n,s in list(shapes.items())[:60]:
            msg += f"  - {n}: {s}\n"
        raise RuntimeError(msg)
    return name

def _to_stream_shape(y: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    if STREAM_MODE == "pre":
        if y.dim() == 4 and y.size(1) == EXPECTED_NH:
            return y.permute(0, 2, 1, 3), y.size(1), y.size(3)
        if y.dim() == 4 and y.size(2) == EXPECTED_NH:
            return y, y.size(2), y.size(3)
        if y.dim() == 3 and y.size(2) == EXPECTED_TOTAL_PRE:
            D = EXPECTED_TOTAL_PRE // EXPECTED_NH
            return y.reshape(y.size(0), y.size(1), EXPECTED_NH, D), EXPECTED_NH, D
        raise RuntimeError(f"[N16] wrong stream shape for pre: {list(y.shape)}")
    else:
        if y.dim() == 3 and y.size(2) == EXPECTED_TOTAL_POST:
            return y.unsqueeze(2), 1, y.size(2)
        if y.dim() == 4 and y.size(2) == 1 and y.size(3) == EXPECTED_TOTAL_POST:
            return y, 1, y.size(3)
        raise RuntimeError(f"[N16] wrong stream shape for post: {list(y.shape)}")

def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(x.mean()); sd = float(x.std() + 1e-9)
    return (x - mu) / sd, mu, sd

def _point_biserial(z: np.ndarray, mask: np.ndarray) -> float:
    n = z.shape[0]
    n1 = int(mask.sum()); n0 = n - n1
    if n1 == 0 or n0 == 0:
        return 0.0
    z1 = z[mask]; z0 = z[~mask]
    mu1 = float(z1.mean()); mu0 = float(z0.mean())
    sd = float(z.std() + 1e-9)
    r = (mu1 - mu0) / sd * math.sqrt((n1 * n0) / float(n * n))
    return float(np.clip(r, -1.0, 1.0))

def _cohens_d(z: np.ndarray, mask: np.ndarray) -> float:
    n1 = int(mask.sum()); n0 = z.shape[0] - n1
    if n1 == 0 or n0 == 0:
        return 0.0
    z1 = z[mask]; z0 = z[~mask]
    m1 = float(z1.mean()); m0 = float(z0.mean())
    v1 = float(z1.var());  v0 = float(z0.var())
    sp = math.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / max(1, (n1 + n0 - 2)))
    if sp <= 1e-12: return 0.0
    return float((m1 - m0) / sp)

def _esc_csv(s: str) -> str:
    return str(s).replace("\n", "\\n").replace("\t", "\\t").strip()

@torch.no_grad()
def run(model: torch.nn.Module, ids: torch.Tensor, text: Optional[str]) -> None:
    if not isinstance(text, str) or len(text) == 0:
        raise RuntimeError("[N16] this analysis needs the raw text string")
    os.makedirs(OUTDIR, exist_ok=True)
    target_name = _resolve_target_layer(model, ids)
    rec = {"y": None}
    def fwd_hook(_m, _inp, out):
        rec["y"] = out[0] if isinstance(out, (tuple, list)) else out
    layer = None
    for name, m in model.named_modules():
        if name == target_name:
            layer = m; break
    if layer is None:
        raise RuntimeError(f"[N16] hook: module '{target_name}' not found")
    h = layer.register_forward_hook(fwd_hook)
    _ = model(ids)
    h.remove()
    y_raw = rec["y"]
    y_stream, nh, d_head = _to_stream_shape(y_raw)
    y_stream = y_stream.detach().float()
    B, S, _, _ = y_stream.shape
    spans = {}
    if PATTERN_PROVIDER:
        spans = load_spans_from_provider(text, S, PATTERN_PROVIDER) or {}
    cov_csv = os.path.join(OUTDIR, "pattern_coverage.csv")
    with open(cov_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["pattern", "covered_chars", "covered_ratio"])
        if spans:
            for name, items in spans.items():
                cov = 0
                try:
                    for it in items:
                        if isinstance(it, (list, tuple)) and len(it) >= 2:
                            a, b = int(it[0]), int(it[1])
                            cov += max(0, min(S, b) - max(0, a))
                        else:
                            pos = int(it)
                            if 0 <= pos < S: cov += 1
                except Exception:
                    pass
                w.writerow([name, cov, f"{cov/max(1,S):.6f}"])
        else:
            w.writerow(["(none)", 0, "0.0"])
    if not spans:
        with open(os.path.join(OUTDIR, "stream_shape.txt"), "w") as f:
            f.write(f"y_stream [B={B}, S={S}, NH={nh}, D={d_head}] total={nh*d_head}\n")
            f.write(f"STREAM_MODE={STREAM_MODE} EXPECTED_NH={EXPECTED_NH} EXPECTED_TOTAL_PRE={EXPECTED_TOTAL_PRE} EXPECTED_TOTAL_POST={EXPECTED_TOTAL_POST}\n")
            f.write(f"PATTERN_PROVIDER={PATTERN_PROVIDER or '(none)'}\n")
            f.write("No spans produced by provider; see pattern_coverage.csv\n")
        print("[N16] no spans from provider; wrote coverage CSV and stream info.")
        return
    masks = spans_dict_to_masks(spans, S)
    X = y_stream.mean(dim=0).reshape(S, nh * d_head).cpu().numpy()
    H = X.shape[1]
    Z = np.empty_like(X, dtype=np.float32)
    mu = np.zeros(H, dtype=np.float32)
    sd = np.zeros(H, dtype=np.float32)
    for j in range(H):
        Z[:, j], mu[j], sd[j] = _zscore(X[:, j])
    sparsity = (Z > Z_SPARSITY_THRESH).mean(axis=0)
    sum_path = os.path.join(OUTDIR, "neuron_summary.csv")
    top_path = os.path.join(OUTDIR, "neuron_pattern_top.csv")
    with open(sum_path, "w", newline="", encoding="utf-8") as fs, open(top_path, "w", newline="", encoding="utf-8") as ft:
        sumw = csv.writer(fs); topw = csv.writer(ft)
        sumw.writerow(["neuron","head","idx_in_head","mean","std","sparsity(z>"+str(Z_SPARSITY_THRESH)+")","best_pattern_rpb","best_|rpb|","best_pattern_d","best_|d|"])
        topw.writerow(["neuron","head","idx_in_head","rank","pattern","covered_chars","r_pb","|r_pb|","d","|d|"])
        pnames = list(masks.keys())
        pmasks = [masks[k] for k in pnames]
        best_r_abs = np.zeros(H, dtype=np.float32)
        best_r_name = np.array([""]*H, dtype=object)
        best_d_abs = np.zeros(H, dtype=np.float32)
        best_d_name = np.array([""]*H, dtype=object)
        for j in range(H):
            h_idx = j // d_head
            d_idx = j %  d_head
            z = Z[:, j]
            rows = []
            for p_idx, m in enumerate(pmasks):
                r = _point_biserial(z, m)
                d = _cohens_d(z, m)
                cov = int(m.sum())
                rows.append((p_idx, r, abs(r), d, abs(d), cov))
            rows.sort(key=lambda r: -r[2])
            if rows:
                p_idx, r, rabs, d_at_r, _, _ = rows[0]
                best_r_abs[j] = rabs
                best_r_name[j] = pnames[p_idx]
            rows_d = sorted(rows, key=lambda r: -r[4])
            if rows_d:
                p_idx, r_at_d, _, dval, dabs, _ = rows_d[0]
                best_d_abs[j] = dabs
                best_d_name[j] = pnames[p_idx]
            K = min(15, len(rows))
            for rank, row in enumerate(rows[:K], start=1):
                p_idx, r, rabs, d, dabs, cov = row
                name = pnames[p_idx]
                topw.writerow([j, h_idx, d_idx, rank, _esc_csv(name), cov, f"{r:.6g}", f"{rabs:.6g}", f"{d:.6g}", f"{dabs:.6g}"])
            sumw.writerow([j, h_idx, d_idx, f"{mu[j]:.6g}", f"{sd[j]:.6g}", f"{float(sparsity[j]):.6g}", _esc_csv(best_r_name[j]), f"{float(best_r_abs[j]):.6g}", _esc_csv(best_d_name[j]), f"{float(best_d_abs[j]):.6g}"])
    plt.figure(figsize=(6,3))
    plt.hist(best_r_abs, bins=50)
    plt.xlabel("best |r_pb| per neuron vs patterns"); plt.ylabel("count")
    plt.title("Pattern association strength per neuron")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "hist_best_point_biserial.png"), dpi=150)
    plt.close()
    pat_scores = []
    masks_list = [masks[k] for k in masks.keys()]
    for pi, m in enumerate(masks_list):
        max_abs = 0.0
        for j in range(H):
            max_abs = max(max_abs, abs(_point_biserial(Z[:, j], m)))
        pat_scores.append((pi, max_abs))
    pat_scores.sort(key=lambda x: -x[1])
    pick_patterns = [pi for pi, _ in pat_scores[:max(1, HEATMAP_PATTERNS)]]
    neu_scores = []
    for j in range(H):
        sc = 0.0
        for pi in pick_patterns:
            sc = max(sc, abs(_point_biserial(Z[:, j], masks_list[pi])))
        neu_scores.append((j, sc))
    neu_scores.sort(key=lambda x: -x[1])
    pick_neurons = [j for j, _ in neu_scores[:max(1, HEATMAP_NEURONS)]]
    M = np.zeros((len(pick_neurons), len(pick_patterns)), dtype=np.float32)
    for ri, j in enumerate(pick_neurons):
        for ci, pi in enumerate(pick_patterns):
            M[ri, ci] = _point_biserial(Z[:, j], masks_list[pi])
    plt.figure(figsize=(max(6, 0.35*len(pick_patterns)), max(4, 0.28*len(pick_neurons))))
    im = plt.imshow(M, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="r_pb")
    plt.yticks(range(len(pick_neurons)), [f"h{(j//d_head)}:ch{(j%d_head)}" for j in pick_neurons], fontsize=8)
    xt = [list(masks.keys())[pi] for pi in pick_patterns]
    xt = [(t[:10] + "...") if len(t) > 11 else t for t in xt]
    plt.xticks(range(len(pick_patterns)), xt, rotation=90, fontsize=8)
    plt.title("Neuron x pattern association (point-biserial r)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "heatmap_neuron_vs_pattern.png"), dpi=180)
    plt.close()
    with open(os.path.join(OUTDIR, "stream_shape.txt"), "w") as f:
        f.write(f"y_stream [B={B}, S={S}, NH={nh}, D={d_head}] total={nh*d_head}\n")
        f.write(f"STREAM_MODE={STREAM_MODE} EXPECTED_NH={EXPECTED_NH} EXPECTED_TOTAL_PRE={EXPECTED_TOTAL_PRE} EXPECTED_TOTAL_POST={EXPECTED_TOTAL_POST}\n")
        f.write(f"PATTERN_PROVIDER={PATTERN_PROVIDER or '(none)'}\n")
        f.write("See pattern_coverage.csv, neuron_summary.csv, neuron_pattern_top.csv, and heatmap PNGs.\n")
    print("[N16] wrote", OUTDIR)
