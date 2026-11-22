import os
from typing import Optional, List, Dict, Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR, NUM_HEADS, EMBED_DIM
import hooks_matrix as hooks
import hooks_kvq as kvq

OUTDIR = os.path.join(RESULTS_DIR, "kvq_more")
EXPECTED_NH = int(NUM_HEADS)
EXPECTED_TOTAL_PRE = int(EMBED_DIM * 2)
SPEC_CLIP = 1e-12
GRID_SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]

def _ensure(d): os.makedirs(d, exist_ok=True)

def _save_csv(path: str, rows: List[List[object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

def _survey_shapes(model: torch.nn.Module, ids: torch.Tensor) -> Dict[str, list]:
    rec: Dict[str, list] = {}
    hs = []
    def mk(name):
        def h(_m, _i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            rec[name] = list(t.shape)
        return h
    for n, m in model.named_modules():
        hs.append(m.register_forward_hook(mk(n)))
    with torch.no_grad():
        _ = model(ids)
    for h in hs: h.remove()
    return rec

def _pick_stream(shapes: Dict[str, list]) -> str:
    c4 = []
    for n, s in shapes.items():
        if len(s) == 4 and (s[1] == EXPECTED_NH or s[2] == EXPECTED_NH):
            c4.append(n)
    if c4:
        for nm in c4:
            if "mlstm_cell" in nm or "outnorm" in nm:
                return nm
        return c4[0]
    c3 = []
    for n, s in shapes.items():
        if len(s) == 3 and (s[2] == EXPECTED_TOTAL_PRE or (s[2] % EXPECTED_NH == 0)):
            c3.append(n)
    if c3:
        for nm in c3:
            if "proj_up" in nm or "mlstm_cell" in nm:
                return nm
        return c3[0]
    raise RuntimeError("stream not found")

def _to_stream(y: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    if y.dim() == 4 and y.size(2) == EXPECTED_NH:
        return y, y.size(2), y.size(3)
    if y.dim() == 4 and y.size(1) == EXPECTED_NH:
        return y.permute(0, 2, 1, 3), y.size(1), y.size(3)
    if y.dim() == 3:
        H = y.size(2)
        if H % EXPECTED_NH != 0:
            raise RuntimeError("concat stream size not divisible by NUM_HEADS")
        D = H // EXPECTED_NH
        return y.reshape(y.size(0), y.size(1), EXPECTED_NH, D), EXPECTED_NH, D
    raise RuntimeError("unsupported stream shape")

def _capture_stream(model: torch.nn.Module, ids: torch.Tensor) -> Tuple[torch.Tensor, int, int, str]:
    shapes = _survey_shapes(model, ids)
    target = _pick_stream(shapes)
    rec = {"y": None}
    def fh(_m, _i, o):
        rec["y"] = o[0] if isinstance(o, (tuple, list)) else o
    layer = None
    for n, m in model.named_modules():
        if n == target:
            layer = m
            break
    if layer is None:
        raise RuntimeError("stream layer not found")
    with torch.no_grad():
        h = layer.register_forward_hook(fh)
        _ = model(ids)
        y = rec["y"]
        h.remove()
    ys, nh, d = _to_stream(y)
    return ys.detach().float(), nh, d, target

def _ensure_qk4d(Q: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    if Q.dim() == 4 and K.dim() == 4:
        B, S, NH, D = Q.shape
        return Q, K, NH, D
    if Q.dim() == 3 and K.dim() == 3:
        B, S, H = Q.shape
        NH = EXPECTED_NH
        if H % NH != 0:
            raise RuntimeError("Q/K last dim not divisible by NUM_HEADS")
        D = H // NH
        return Q.reshape(B, S, NH, D), K.reshape(B, S, NH, D), NH, D
    raise RuntimeError("Q/K shapes invalid")

def _attn_expected_distance(attn: torch.Tensor, outdir: str) -> None:
    if attn.dim() != 4: raise RuntimeError("attn must be [B,S,NH,S]")
    B, S, NH, _ = attn.shape
    A = attn[0].detach()
    t_idx = torch.arange(S, device=A.device).float()
    dist = []
    for h in range(NH):
        P = A[:, h, :]
        mask = torch.triu(torch.ones(S, S, device=A.device), 1).bool()
        P = P.masked_fill(mask, 0.0)
        expd = (P * (t_idx.view(1, -1) - t_idx.view(-1, 1))).sum(-1)
        dist.append(expd.cpu().numpy())
    dist = np.stack(dist, axis=0)
    np.save(os.path.join(outdir, "attn_expected_distance.npy"), dist)
    fig = plt.figure(figsize=(10, 2.6 * NH))
    for h in range(NH):
        ax = plt.subplot(NH, 1, h + 1)
        ax.plot(dist[h])
        ax.set_ylabel(f"h{h}")
    plt.xlabel("time")
    plt.suptitle("attention expected look-back distance")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "attn_expected_distance.png"), dpi=150)
    plt.close(fig)

def _attn_argmax_hist(attn: torch.Tensor, outdir: str) -> None:
    B, S, NH, _ = attn.shape
    A = attn[0].detach()
    peaks = torch.argmax(A, dim=-1)
    fig, ax = plt.subplots(NH, 1, figsize=(10, 2.4 * NH), sharex=True)
    if NH == 1: ax = [ax]
    for h in range(NH):
        lag = (torch.arange(S, device=A.device) - peaks[:, h]).clamp_min(0).cpu().numpy()
        ax[h].hist(lag, bins=min(60, S), range=(0, S))
        ax[h].set_ylabel(f"h{h}")
    ax[-1].set_xlabel("lag")
    fig.suptitle("attention argmax lag histograms")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "attn_argmax_lag_hist.png"), dpi=150)
    plt.close(fig)

def _attn_js_matrix(attn: torch.Tensor) -> np.ndarray:
    B, S, NH, _ = attn.shape
    A = attn[0].detach()
    M = np.zeros((NH, NH), dtype=np.float64)
    for i in range(NH):
        for j in range(NH):
            Pi = A[:, i, :]
            Pj = A[:, j, :]
            Pi = Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-12)
            Pj = Pj / (Pj.sum(dim=-1, keepdim=True) + 1e-12)
            Mij = 0.0
            for t in range(S):
                p = Pi[t]; q = Pj[t]
                m = 0.5 * (p + q)
                kl1 = (p * (p.add(1e-12).log() - m.add(1e-12).log())).sum()
                kl2 = (q * (q.add(1e-12).log() - m.add(1e-12).log())).sum()
                Mij += 0.5 * (kl1 + kl2).item()
            M[i, j] = Mij / S
    return M

def _plot_js_and_dendrogram(M: np.ndarray, outdir: str) -> None:
    fig = plt.figure(figsize=(4.2, 3.8))
    plt.imshow(M, cmap="viridis", aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar()
    plt.title("attention JS divergence")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "attn_js_matrix.png"), dpi=150)
    plt.close(fig)

    N = M.shape[0]
    D = (M + M.T) * 0.5
    np.fill_diagonal(D, 0.0)
    clusters = [{i} for i in range(N)]
    heights = []
    Z = []
    while len(clusters) > 1:
        best = (None, None, 1e9)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                sa = clusters[a]; sb = clusters[b]
                vals = [D[i, j] for i in sa for j in sb]
                d = float(np.mean(vals)) if vals else 0.0
                if d < best[2]:
                    best = (a, b, d)
        a, b, d = best
        sa = clusters[a]; sb = clusters[b]
        Z.append((sorted(list(sa))[0], sorted(list(sb))[0], d, len(sa) + len(sb)))
        clusters.append(sa | sb)
        heights.append(d)
        del clusters[b]; del clusters[a]
    xs = list(range(N))
    fig = plt.figure(figsize=(6, 3))
    plt.title("simple dendrogram")
    plt.plot([], [])
    plt.axis("off")
    ystep = max(heights + [1.0]) / (len(Z) + 1) if Z else 1.0
    ypos = 0.0
    used = set()
    for (i, j, d, sz) in Z:
        ypos += ystep
        xi = xs[i]; xj = xs[j]
        plt.plot([xi, xi], [ypos - ystep, ypos], color="k")
        plt.plot([xj, xj], [ypos - ystep, ypos], color="k")
        plt.plot([min(xi, xj), max(xi, xj)], [ypos, ypos], color="k")
        used.add(i); used.add(j)
    for k in range(N):
        if k not in used:
            plt.plot([xs[k], xs[k]], [0, ypos], color="k", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "attn_js_dendrogram.png"), dpi=150)
    plt.close(fig)

def _spectrum_per_head(C: torch.Tensor, outdir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if C.dim() != 5: raise RuntimeError("C must be [B,S,NH,DH,DH]")
    B, S, NH, DH, _ = C.shape
    s1 = np.zeros((NH, S), dtype=np.float64)
    s2 = np.zeros((NH, S), dtype=np.float64)
    for t in range(S):
        Ct = C[0, t]
        for h in range(NH):
            sv = torch.linalg.svdvals(Ct[h]).cpu().numpy()
            s1[h, t] = float(sv[0]) if sv.size >= 1 else 0.0
            s2[h, t] = float(sv[1]) if sv.size >= 2 else 0.0
    ratio = np.divide(s1, np.maximum(s2, SPEC_CLIP))
    np.savez(os.path.join(outdir, "spectrum_per_head.npz"), s1=s1, s2=s2, ratio=ratio)

    fig = plt.figure(figsize=(10, 2.6 * NH))
    for h in range(NH):
        ax = plt.subplot(NH, 1, h + 1)
        ax.plot(s1[h], label="s1", linewidth=1.0)
        ax.plot(s2[h], label="s2", linewidth=1.0, alpha=0.8)
        ax2 = ax.twinx(); ax2.plot(ratio[h], label="s1/s2", linewidth=1.0, alpha=0.8)
        ax.set_ylabel(f"h{h}")
        if h == 0: ax.legend(loc="upper left")
    plt.xlabel("time")
    plt.suptitle("per-head spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "spectrum_per_head.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 3.5))
    plt.imshow(ratio, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("time"); plt.ylabel("head")
    plt.title("s1/s2 ratio heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "spectrum_ratio_heatmap.png"), dpi=150)
    plt.close(fig)

    data = [ratio[h] for h in range(NH)]
    fig = plt.figure(figsize=(8, 3.2))
    plt.violinplot(data, showmeans=True, showextrema=True)
    plt.xlabel("head"); plt.ylabel("s1/s2")
    plt.title("s1/s2 distribution per head")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "spectrum_ratio_violin.png"), dpi=150)
    plt.close(fig)

    return s1, s2, ratio

def _entropy_per_head(attn: torch.Tensor) -> np.ndarray:
    B, S, NH, _ = attn.shape
    P = attn[0].clamp_min(1e-12)
    H = -(P * P.log()).sum(-1).cpu().numpy()
    return H.T

def _per_time_loss(model: torch.nn.Module, ids: torch.Tensor) -> np.ndarray:
    logits = model(ids)
    if isinstance(logits, (list, tuple)): logits = logits[0]
    if ids.size(1) >= 2:
        tgt = ids[:, 1:]; logits = logits[:, :-1, :]
    else:
        tgt = ids
    logp = torch.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return nll.mean(dim=0).detach().cpu().numpy()

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    da = np.linalg.norm(a); db = np.linalg.norm(b)
    if da == 0 or db == 0: return 0.0
    return float(np.dot(a, b) / (da * db))

def _ratio_correlations(ratio: np.ndarray, H: np.ndarray, nll: np.ndarray, outdir: str) -> None:
    NH, Sr = ratio.shape
    if H.shape == (Sr, NH): H = H.T
    if H.shape[0] != NH: raise RuntimeError("entropy head count mismatch")
    Sc = min(Sr, H.shape[1], nll.shape[0])
    r_e = [_corr(ratio[h, :Sc], H[h, :Sc]) for h in range(NH)]
    r_l = [_corr(ratio[h, :Sc], nll[:Sc]) for h in range(NH)]
    _save_csv(os.path.join(outdir, "ratio_corr.csv"),
              [["head", "corr_ratio_entropy", "corr_ratio_loss", "length_used", "S_ratio", "S_entropy", "S_loss"]]
              + [[i, r_e[i], r_l[i], Sc, Sr, H.shape[1], nll.shape[0]] for i in range(NH)])
    fig = plt.figure(figsize=(8, 3.2))
    plt.bar(np.arange(NH) - 0.15, r_e, width=0.3, label="ratio vs entropy")
    plt.bar(np.arange(NH) + 0.15, r_l, width=0.3, label="ratio vs loss")
    plt.legend(); plt.xlabel("head"); plt.ylabel("corr"); plt.title("correlations per head")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "ratio_corr_bar.png"), dpi=150); plt.close(fig)

def _select_qk_from_lists(Q_list, K_list, S_ref: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    if not (isinstance(Q_list, list) and len(Q_list) and isinstance(K_list, list) and len(K_list)):
        raise RuntimeError("Q/K not captured")
    def pick(L):
        for t in reversed(L):
            if t.dim() >= 2 and t.size(1) == S_ref:
                return t
        return L[-1]
    Q = pick(Q_list)
    K = pick(K_list)
    Q, K, NH, D = _ensure_qk4d(Q, K)
    if Q.size(1) != S_ref:
        Q = Q[:, :S_ref, :, :]
    if K.size(1) != S_ref:
        K = K[:, :S_ref, :, :]
    return Q, K, NH, D

def _qk_attn_time_heatmap(outdir: str) -> None:
    Q_list = kvq.Q_OUTS; K_list = kvq.K_OUTS; A = hooks.AUX
    attn = A["attn"]
    S_ref = attn.size(1)
    Q, K, NH, D = _select_qk_from_lists(Q_list, K_list, S_ref)
    dev = Q.device
    if attn.device != dev:
        attn = attn.to(dev)
    B, S = Q.size(0), Q.size(1)
    Qn = Q / (Q.norm(dim=-1, keepdim=True) + 1e-12)
    Kn = K / (K.norm(dim=-1, keepdim=True) + 1e-12)
    heat = np.zeros((NH, S), dtype=np.float64)
    for h in range(NH):
        cosmap = torch.einsum("bsd,btd->bst", Kn[:, :, h, :], Qn[:, :, h, :])
        cosmap = cosmap.triu(diagonal=0)
        A_h = attn[:, :, h, :]
        for t in range(S):
            c = cosmap[:, t, :].reshape(B, -1)
            a = A_h[:, t, :].reshape(B, -1)
            c = c - c.mean(dim=1, keepdim=True)
            a = a - a.mean(dim=1, keepdim=True)
            num = (c * a).sum(dim=1)
            den = torch.sqrt((c ** 2).sum(dim=1)) * torch.sqrt((a ** 2).sum(dim=1)) + 1e-12
            r = (num / den).mean().item()
            heat[h, t] = r
    np.save(os.path.join(outdir, "qk_attn_time.npy"), heat)
    fig = plt.figure(figsize=(10, 3.5))
    plt.imshow(heat, aspect="auto", origin="lower", vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(); plt.xlabel("time"); plt.ylabel("head"); plt.title("corr(attn, cos(Q,K)) per time")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "qk_attn_time_heatmap.png"), dpi=150); plt.close(fig)

def _knockout_curves(model: torch.nn.Module, ids: torch.Tensor, NH: int, D: int, outdir: str) -> None:
    base_logits = model(ids)
    if isinstance(base_logits, (list, tuple)): base_logits = base_logits[0]
    if ids.size(1) >= 2:
        base_tgt = ids[:, 1:]; base_logits = base_logits[:, :-1, :]
    else:
        base_tgt = ids
    base_cc = base_logits.gather(-1, base_tgt.unsqueeze(-1)).squeeze(-1).mean().item()
    drops = []
    for h in range(NH):
        def hq(_m, _i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if t.dim() == 3: t = t.reshape(t.size(0), t.size(1), NH, D)
            t[:, :, h, :] = 0.0
            return t.reshape(t.size(0), t.size(1), NH * D)
        def hk(_m, _i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if t.dim() == 3: t = t.reshape(t.size(0), t.size(1), NH, D)
            t[:, :, h, :] = 0.0
            return t.reshape(t.size(0), t.size(1), NH * D)
        r1 = kvq.KVQ_LAYER.q_proj.register_forward_hook(hq)
        r2 = kvq.KVQ_LAYER.k_proj.register_forward_hook(hk)
        logits = model(ids)
        if isinstance(logits, (list, tuple)): logits = logits[0]
        if ids.size(1) >= 2:
            tgt = ids[:, 1:]; logits = logits[:, :-1, :]
        else:
            tgt = ids
        cc = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()
        r1.remove(); r2.remove()
        drops.append(base_cc - cc)
    order = np.argsort(-np.array(drops))
    xs = list(range(NH + 1))
    ys = []
    for p in xs:
        heads = set(int(order[i]) for i in range(p) if i < NH)
        def hq(_m, _i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if t.dim() == 3: t = t.reshape(t.size(0), t.size(1), NH, D)
            for hh in heads: t[:, :, hh, :] = 0.0
            return t.reshape(t.size(0), t.size(1), NH * D)
        def hk(_m, _i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if t.dim() == 3: t = t.reshape(t.size(0), t.size(1), NH, D)
            for hh in heads: t[:, :, hh, :] = 0.0
            return t.reshape(t.size(0), t.size(1), NH * D)
        r1 = kvq.KVQ_LAYER.q_proj.register_forward_hook(hq)
        r2 = kvq.KVQ_LAYER.k_proj.register_forward_hook(hk)
        logits = model(ids)
        if isinstance(logits, (list, tuple)): logits = logits[0]
        if ids.size(1) >= 2:
            tgt = ids[:, 1:]; logits = logits[:, :-1, :]
        else:
            tgt = ids
        cc = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()
        r1.remove(); r2.remove()
        ys.append(cc)
    rows = [["p", "mean_correct_logit"]] + [[xs[i], ys[i]] for i in range(len(xs))]
    _save_csv(os.path.join(outdir, "knockout_curve.csv"), rows)
    fig = plt.figure(figsize=(6, 3.4))
    plt.plot(xs, ys, marker="o"); plt.xlabel("top-p heads zeroed"); plt.ylabel("mean correct logit"); plt.title("head knockout curve")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "knockout_curve.png"), dpi=150); plt.close(fig)

def _robustness_grid(model: torch.nn.Module, ids: torch.Tensor, NH: int, D: int, outdir: str) -> None:
    base_logits = model(ids)
    if isinstance(base_logits, (list, tuple)): base_logits = base_logits[0]
    if ids.size(1) >= 2:
        base_tgt = ids[:, 1:]; base_logits = base_logits[:, :-1, :]
    else:
        base_tgt = ids
    base_cc = base_logits.gather(-1, base_tgt.unsqueeze(-1)).squeeze(-1).mean().item()
    drops = []
    for h in range(NH):
        def hq(_m, _i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if t.dim() == 3: t = t.reshape(t.size(0), t.size(1), NH, D)
            t[:, :, h, :] = 0.0
            return t.reshape(t.size(0), t.size(1), NH * D)
        def hk(_m, _i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if t.dim() == 3: t = t.reshape(t.size(0), t.size(1), NH, D)
            t[:, :, h, :] = 0.0
            return t.reshape(t.size(0), t.size(1), NH * D)
        r1 = kvq.KVQ_LAYER.q_proj.register_forward_hook(hq)
        r2 = kvq.KVQ_LAYER.k_proj.register_forward_hook(hk)
        logits = model(ids)
        if isinstance(logits, (list, tuple)): logits = logits[0]
        if ids.size(1) >= 2:
            tgt = ids[:, 1:]; logits = logits[:, :-1, :]
        else:
            tgt = ids
        cc = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()
        r1.remove(); r2.remove()
        drops.append(base_cc - cc)
    h0 = int(np.argmax(np.array(drops)))
    M = np.zeros((len(GRID_SCALES), len(GRID_SCALES)), dtype=np.float64)
    for i, qs in enumerate(GRID_SCALES):
        for j, ks in enumerate(GRID_SCALES):
            def hq(_m, _i, o):
                t = o[0] if isinstance(o, (tuple, list)) else o
                if t.dim() == 3: t = t.reshape(t.size(0), t.size(1), NH, D)
                t[:, :, h0, :] = t[:, :, h0, :] * qs
                return t.reshape(t.size(0), t.size(1), NH * D)
            def hk(_m, _i, o):
                t = o[0] if isinstance(o, (tuple, list)) else o
                if t.dim() == 3: t = t.reshape(t.size(0), t.size(1), NH, D)
                t[:, :, h0, :] = t[:, :, h0, :] * ks
                return t.reshape(t.size(0), t.size(1), NH * D)
            r1 = kvq.KVQ_LAYER.q_proj.register_forward_hook(hq)
            r2 = kvq.KVQ_LAYER.k_proj.register_forward_hook(hk)
            logits = model(ids)
            if isinstance(logits, (list, tuple)): logits = logits[0]
            if ids.size(1) >= 2:
                tgt = ids[:, 1:]; logits = logits[:, :-1, :]
            else:
                tgt = ids
            cc = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()
            r1.remove(); r2.remove()
            M[i, j] = cc
    _save_csv(os.path.join(outdir, "robustness_grid.csv"),
              [["q_scale\\k_scale"] + GRID_SCALES] + [[GRID_SCALES[i]] + list(M[i]) for i in range(len(GRID_SCALES))])
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(M, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(); plt.xlabel("k scale"); plt.ylabel("q scale"); plt.title(f"grid head {h0}")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "robustness_grid.png"), dpi=150); plt.close(fig)

def run(model: torch.nn.Module, ids: torch.Tensor, text: Optional[str] = None) -> None:
    _ensure(OUTDIR)
    if hooks.C_SEQ is None: raise RuntimeError("C_SEQ missing")
    if hooks.AUX is None or "attn" not in hooks.AUX: raise RuntimeError("AUX['attn'] missing")

    attn = hooks.AUX["attn"].detach()
    C = hooks.C_SEQ.detach()
    y, nh, d, target = _capture_stream(model, ids)
    if nh != EXPECTED_NH: raise RuntimeError("NUM_HEADS mismatch")

    with open(os.path.join(OUTDIR, "info.txt"), "w") as f:
        B, S = y.size(0), y.size(1)
        f.write(f"y_stream=[B={B},S={S},NH={nh},D={d}]\n")
        f.write(f"C_seq={tuple(C.shape)} attn={tuple(attn.shape)} layer={target}\n")

    _attn_expected_distance(attn, OUTDIR)
    _attn_argmax_hist(attn, OUTDIR)
    M = _attn_js_matrix(attn); np.save(os.path.join(OUTDIR, "attn_js_matrix.npy"), M); _plot_js_and_dendrogram(M, OUTDIR)
    s1, s2, ratio = _spectrum_per_head(C, OUTDIR)
    H = _entropy_per_head(attn)
    nll = _per_time_loss(model, ids)
    _ratio_correlations(ratio, H, nll, OUTDIR)

    kvq.register_kvq_hooks(model)
    with torch.no_grad():
        _ = model(ids)
    _qk_attn_time_heatmap(OUTDIR)
    _knockout_curves(model, ids, nh, d, OUTDIR)
    _robustness_grid(model, ids, nh, d, OUTDIR)
    print("[VIZ++] wrote", OUTDIR)
