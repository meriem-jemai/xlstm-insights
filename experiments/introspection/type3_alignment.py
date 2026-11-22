import os, re, html, importlib
from typing import Optional, List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from config import RESULTS_DIR, NUM_HEADS, PATTERN_PROVIDER
from hooks import extract_series_per_head
from common_viz import ensure_outdir
OUTDIR = os.path.join(RESULTS_DIR, 'gates_type3_alignment')
os.makedirs(OUTDIR, exist_ok=True)
ANCHOR_POLICY = 'center'

def _save_heatmap_heads(mat: np.ndarray, title: str, outpath: str, xlabel: Optional[str]=None, xticks: Optional[np.ndarray]=None):
    H = mat.shape[0]
    plt.figure(figsize=(8, max(4, H * 0.6)))
    plt.imshow(mat, aspect='auto', origin='lower')
    plt.colorbar()
    plt.ylabel('Head index')
    if xlabel:
        plt.xlabel(xlabel)
    plt.yticks(np.arange(H), [str(i) for i in range(H)])
    if xticks is not None and xticks.size > 0:
        step = max(1, len(xticks) // 10)
        tick_pos = np.arange(0, len(xticks), step)
        plt.xticks(tick_pos, [str(int(xticks[i])) for i in tick_pos])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def _eta(series: np.ndarray, events: np.ndarray, W: int):
    events = np.asarray(events, dtype=int)
    if events.size == 0:
        return None
    S = series.shape[0]
    valid = events[(events >= W) & (events < S - W)]
    if valid.size == 0:
        return None
    if series.ndim == 1:
        return np.mean([series[t - W:t + W + 1] for t in valid], axis=0)
    return np.mean([series[t - W:t + W + 1, :] for t in valid], axis=0)

def _hist_at_events(series1d: np.ndarray, events: np.ndarray, nbins: int=60):
    events = np.asarray(events, dtype=int)
    if events.size == 0:
        return (None, None)
    events = events[(events >= 0) & (events < series1d.shape[0])]
    if events.size == 0:
        return (None, None)
    vals = series1d[events]
    hist, edges = np.histogram(vals, bins=nbins, range=(series1d.min(), series1d.max()), density=True)
    return (hist, edges)

def _safe_corrcoef(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, float)
    if mat.ndim == 1:
        mat = mat[None, :]
    M, N = mat.shape
    if M <= 1:
        return np.ones((M, M), dtype=float)
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True)
    sd[sd == 0.0] = 1.0
    Z = (mat - mu) / sd
    C = Z @ Z.T / max(1, N - 1)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C
Span = Tuple[int, int]

def _merge_spans(spans: List[Span]) -> List[Span]:
    if not spans:
        return []
    spans = sorted(((int(s), int(e)) for s, e in spans if e > s))
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

def _mask_to_spans(mask: np.ndarray) -> List[Span]:
    mask = np.asarray(mask).astype(bool)
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]

def _provider_to_spans(patt: Dict[str, np.ndarray], S: int) -> Dict[str, List[Span]]:
    out: Dict[str, List[Span]] = {}
    for k, v in patt.items():
        a = np.asarray(v)
        spans: List[Span] = []
        if a.ndim == 2 and a.shape[1] == 2:
            for s, e in a.astype(int):
                s = max(0, min(S, int(s)))
                e = max(0, min(S, int(e)))
                if e > s:
                    spans.append((s, e))
        elif a.ndim == 1 and a.size == S and (a.dtype == np.bool_ or np.all((a == 0) | (a == 1))):
            spans = _mask_to_spans(a[:S])
        else:
            idx = a.ravel().astype(int, copy=False)
            idx = idx[(idx >= 0) & (idx < S)]
            if idx.size:
                idx = np.unique(idx)
                spans = [(int(t), int(t) + 1) for t in idx]
        spans = [(max(0, s), min(S, e)) for s, e in spans if 0 <= s < e <= S]
        out[k] = _merge_spans(spans)
    return out

def _load_patterns_from_provider(text: str, S: int) -> Dict[str, List[Span]]:
    if not PATTERN_PROVIDER:
        raise RuntimeError('PATTERN_PROVIDER not set in config.py or env.')
    mod, func = PATTERN_PROVIDER.rsplit('.', 1)
    provider_fn = getattr(importlib.import_module(mod), func)
    patt = provider_fn(text)
    if not isinstance(patt, dict):
        raise RuntimeError(f'Pattern provider returned non-dict: {type(patt)}')
    return _provider_to_spans(patt, S)

def _auto_policy(spans: List[Span]) -> str:
    if not spans:
        return ANCHOR_POLICY
    L = np.asarray([e - s for s, e in spans], dtype=int)
    if L.size == 0:
        return ANCHOR_POLICY
    med = np.median(L)
    if med <= 1:
        return 'start'
    if med >= 8:
        return 'dense'
    return 'center'

def _spans_to_events(spans: List[Span], policy: str) -> np.ndarray:
    if not spans:
        return np.array([], dtype=int)
    s = np.asarray([a for a, _ in spans], dtype=int)
    e = np.asarray([b for _, b in spans], dtype=int)
    if policy == 'start':
        ev = s
    elif policy == 'end':
        ev = e - 1
    elif policy == 'dense':
        ev = np.concatenate([np.arange(si, ei) for si, ei in zip(s, e)]).astype(int)
    else:
        ev = ((s + e) // 2).astype(int)
    return np.unique(ev)

def _sanitize_name(name: str) -> str:
    s = re.sub('[^a-zA-Z0-9._-]+', '_', name).strip('_')
    return s[:120] or 'pattern'

def write_fulltext_masks_html(text: str, patt_spans: Dict[str, List[Span]], outdir: str, select: Optional[List[str]]=None):
    os.makedirs(outdir, exist_ok=True)
    N = len(text)
    items = list(patt_spans.items())
    if select:
        wanted = {s.lower() for s in select}
        items = [(k, v) for k, v in items if k.lower() in wanted]
    rows = []
    for name, spans in items:
        spans = [(max(0, s), min(N, e)) for s, e in spans if 0 <= s < e <= N]
        hit_spans: List[Span] = _merge_spans(spans)
        fname = _sanitize_name(name) + '.html'
        fpath = os.path.join(outdir, fname)
        out = []
        pos = 0
        for s, e in hit_spans:
            if pos < s:
                out.append(html.escape(text[pos:s], quote=False))
            out.append('<span class="hl">')
            out.append(html.escape(text[s:e], quote=False))
            out.append('</span>')
            pos = e
        if pos < N:
            out.append(html.escape(text[pos:N], quote=False))
        with open(fpath, 'w', encoding='utf-8') as fh:
            fh.write(f"""<!doctype html>\n<meta charset="utf-8">\n<title>Pattern: {html.escape(name)}</title>\n<style>\n  body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; margin: 24px; }}\n  h1 {{ margin: 0 0 10px 0; font-size: 20px; }}\n  .meta {{ color:#667; font-size:12px; margin-bottom:12px; }}\n  pre.full {{ margin:0; white-space: pre-wrap; word-break: break-word; }}\n  .hl {{\n    background: #fff3cd;\n    outline: 2px solid #e76f51;\n    border-radius: 4px;\n    box-shadow: 0 0 0 1px rgba(0,0,0,0.05) inset;\n  }}\n  a.back {{ text-decoration:none; font-size:12px; color:#0366d6; }}\n</style>\n<a class="back" href="index.html"> back to index</a>\n<h1>Pattern: {html.escape(name)}</h1>\n<div class="meta">highlighted spans: {len(hit_spans)} * text length: {N} chars</div>\n<pre class="full">{''.join(out)}</pre>\n""")
        rows.append((name, len(hit_spans), fname))
    rows.sort(key=lambda x: x[0].lower())
    with open(os.path.join(outdir, 'index.html'), 'w', encoding='utf-8') as fh:
        fh.write('<!doctype html>\n<meta charset="utf-8">\n<title>Pattern masks index</title>\n<style>\n  body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; margin: 24px; }\n  h1 { margin: 0 0 10px 0; font-size: 20px; }\n  table { border-collapse: collapse; margin-top: 10px; }\n  th, td { border: 1px solid #eee; padding: 6px 10px; }\n  th { background: #f6f8fa; text-align: left; }\n  a { text-decoration: none; color: #0366d6; }\n</style>\n<h1>Pattern masks index</h1>\n<table>\n  <tr><th>#</th><th>pattern</th><th># highlighted spans</th></tr>\n')
        for i, (name, nspans, fname) in enumerate(rows, 1):
            fh.write(f"  <tr><td>{i}</td><td><a href='{fname}'>{html.escape(name)}</a></td><td>{nspans}</td></tr>\n")
        fh.write('</table>')

def run(text: str, half_window: int=32, nbins: int=60):
    print('[type3_alignment] start')
    ensure_outdir(OUTDIR)
    i_all, f_all, i_ph, f_ph, _ = extract_series_per_head()
    S, H = i_ph.shape
    text = text[:S]
    patt_spans = _load_patterns_from_provider(text, S)
    patt_idx: Dict[str, np.ndarray] = {name: _spans_to_events(spans, _auto_policy(spans)) for name, spans in patt_spans.items()}
    with open(os.path.join(OUTDIR, 'patterns_counts.txt'), 'w', encoding='utf-8') as fh:
        for k, idx in patt_idx.items():
            fh.write(f'{k}: {int(np.asarray(idx).size)}\n')
    html_full = os.path.join(OUTDIR, 'html_fulltext')
    write_fulltext_masks_html(text, patt_spans, html_full)
    print(f'[type3_alignment] wrote HTML mask previews to: {html_full}')
    lags = np.arange(-half_window, half_window + 1)
    for name, ev_idx in patt_idx.items():
        ev_idx = np.asarray(ev_idx, dtype=int)
        i_eta = _eta(i_ph, ev_idx, half_window)
        f_eta = _eta(f_ph, ev_idx, half_window)
        np.save(os.path.join(OUTDIR, f'ETA_{name}_i.npy'), i_eta if i_eta is not None else np.array([]))
        np.save(os.path.join(OUTDIR, f'ETA_{name}_f.npy'), f_eta if f_eta is not None else np.array([]))
        _save_heatmap_heads(i_eta.T if i_eta is not None else np.zeros((1, 1)), f'ETA input gate around {name}', os.path.join(OUTDIR, f'ETA_{name}_i.png'), xlabel='Lag', xticks=lags)
        _save_heatmap_heads(f_eta.T if f_eta is not None else np.zeros((1, 1)), f'ETA forget gate around {name}', os.path.join(OUTDIR, f'ETA_{name}_f.png'), xlabel='Lag', xticks=lags)
        for gate, eta in (('i', i_eta), ('f', f_eta)):
            if eta is None:
                continue
            center = eta[len(lags) // 2]
            order = np.argsort(-center)
            plt.figure(figsize=(8, 4 + H * 0.06))
            plt.bar(np.arange(H), center[order])
            plt.xlabel('Head index')
            plt.ylabel(f'{gate} at event (lag 0)')
            plt.title(f'{name}: head sensitivity at event')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTDIR, f'{name}_{gate}_head_bars.png'))
            plt.close()
        for gate, arr in (('i', i_all), ('f', f_all)):
            non = np.setdiff1d(np.arange(S), ev_idx) if ev_idx.size else np.arange(S)
            h_ev, e = _hist_at_events(arr, ev_idx, nbins)
            h_nev, _ = _hist_at_events(arr, non[:ev_idx.size], nbins) if ev_idx.size else (None, None)
            plt.figure(figsize=(8, 4))
            if h_ev is None:
                plt.text(0.5, 0.5, 'No events', ha='center', va='center')
            else:
                mids = 0.5 * (e[:-1] + e[1:])
                plt.plot(mids, h_ev, label='events')
                if h_nev is not None:
                    plt.plot(mids, h_nev, label='non-events (matched N)')
                plt.legend()
            plt.xlabel(f'{gate} layer mean')
            plt.ylabel('Density')
            plt.title(f'{name}: layer distribution shift')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTDIR, f'{name}_{gate}_dist.png'))
            plt.close()
    patterns = list(patt_idx.keys())
    M = len(patterns)
    Z_i = np.zeros((M, H))
    Z_f = np.zeros((M, H))
    for m, name in enumerate(patterns):
        i_eta = np.load(os.path.join(OUTDIR, f'ETA_{name}_i.npy'), allow_pickle=True)
        f_eta = np.load(os.path.join(OUTDIR, f'ETA_{name}_f.npy'), allow_pickle=True)
        if i_eta.size:
            c = i_eta[i_eta.shape[0] // 2]
            Z_i[m] = (c - c.mean()) / (c.std() + 1e-08)
        if f_eta.size:
            c = f_eta[f_eta.shape[0] // 2]
            Z_f[m] = (c - c.mean()) / (c.std() + 1e-08)
    plt.figure(figsize=(10, max(5, M * 0.35)))
    plt.imshow(Z_i, aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel('Head index')
    plt.ylabel('Pattern')
    plt.yticks(np.arange(M), patterns)
    plt.title('Pattern by head sensitivity (z, input gate at lag 0)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'pattern_head_matrix_i.png'))
    plt.close()
    plt.figure(figsize=(10, max(5, M * 0.35)))
    plt.imshow(Z_f, aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel('Head index')
    plt.ylabel('Pattern')
    plt.yticks(np.arange(M), patterns)
    plt.title('Pattern by head sensitivity (z, forget gate at lag 0)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'pattern_head_matrix_f.png'))
    plt.close()
    for gate, Z in (('i', Z_i), ('f', Z_f)):
        C = _safe_corrcoef(Z)
        plt.figure(figsize=(8, max(4, M * 0.3)))
        plt.imshow(C, vmin=-1, vmax=1, origin='lower')
        plt.colorbar()
        plt.title(f'Cross-pattern correlation ({gate})')
        plt.xticks(np.arange(M), patterns, rotation=90)
        plt.yticks(np.arange(M), patterns)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'pattern_correlation_{gate}.png'))
        plt.close()
    X = Z_i.T
    mu = np.mean(X, 0, keepdims=True)
    Xc = X - mu
    if Xc.size and Xc.shape[0] >= 2:
        U, Ssvd, _ = np.linalg.svd(Xc, full_matrices=False)
        PC = U[:, :2] * Ssvd[:2]
        plt.figure(figsize=(6, 5))
        plt.scatter(PC[:, 0], PC[:, 1], s=30)
        for h in range(H):
            plt.annotate(str(h), (PC[h, 0], PC[h, 1]), fontsize=8)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Heads embedded by pattern profile (input gate)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'heads_pca_by_pattern_i.png'))
        plt.close()
    eta_sum_csv = os.path.join(OUTDIR, 'type3_eta_summary.csv')
    eta_head_csv = os.path.join(OUTDIR, 'type3_eta_by_head.csv')
    with open(eta_sum_csv, 'w', encoding='utf-8') as fs, open(eta_head_csv, 'w', encoding='utf-8') as fh:
        fs.write(','.join(['pattern', 'n_events', 'half_window', 'i_center_mean', 'f_center_mean', 'i_area_mean', 'f_area_mean', 'i_peak_lag', 'i_peak_value', 'f_peak_lag', 'f_peak_value', 'top3_i_heads_center', 'top3_i_values_center', 'top3_f_heads_center', 'top3_f_values_center']) + '\n')
        fh.write(','.join(['pattern', 'head', 'i_center', 'f_center', 'i_area', 'f_area', 'i_peak_lag', 'i_peak_value', 'f_peak_lag', 'f_peak_value']) + '\n')
        for name in patterns:
            i_eta = np.load(os.path.join(OUTDIR, f'ETA_{name}_i.npy'), allow_pickle=True)
            f_eta = np.load(os.path.join(OUTDIR, f'ETA_{name}_f.npy'), allow_pickle=True)

            def _stats(eta: np.ndarray):
                if eta.size == 0:
                    return (np.array([]), 0.0, np.nan, np.nan)
                center = eta[eta.shape[0] // 2]
                area = eta.mean(axis=0).sum()
                mean_eta = eta.mean(axis=1)
                k = int(np.argmax(mean_eta))
                peak_lag = int(k - eta.shape[0] // 2)
                peak_val = float(mean_eta[k])
                return (center, float(area), peak_lag, peak_val)
            i_center, i_area, i_pk_lag, i_pk_val = _stats(i_eta)
            f_center, f_area, f_pk_lag, f_pk_val = _stats(f_eta)

            def _top3(v: np.ndarray):
                if v.size == 0:
                    return ('', '')
                order = np.argsort(-v)[:min(3, v.size)]
                return ('|'.join(map(str, order.tolist())), '|'.join((f'{v[h]:.4f}' for h in order.tolist())))
            top_i_idx, top_i_val = _top3(i_center)
            top_f_idx, top_f_val = _top3(f_center)
            n_events = int(np.asarray(patt_idx.get(name, [])).size)
            fs.write(','.join(map(str, [name, n_events, half_window, float(i_center.mean()) if i_center.size else np.nan, float(f_center.mean()) if f_center.size else np.nan, i_area, f_area, i_pk_lag, i_pk_val, f_pk_lag, f_pk_val, top_i_idx, top_i_val, top_f_idx, top_f_val])) + '\n')
            for h in range(H):
                ci = float(i_center[h]) if i_center.size else np.nan
                cf = float(f_center[h]) if f_center.size else np.nan
                if i_eta.size:
                    ai = float(i_eta[:, h].sum())
                    k = int(np.argmax(i_eta[:, h]))
                    pli = int(k - i_eta.shape[0] // 2)
                    pvi = float(i_eta[k, h])
                else:
                    ai = np.nan
                    pli = np.nan
                    pvi = np.nan
                if f_eta.size:
                    af = float(f_eta[:, h].sum())
                    k = int(np.argmax(f_eta[:, h]))
                    plf = int(k - f_eta.shape[0] // 2)
                    pvf = float(f_eta[k, h])
                else:
                    af = np.nan
                    plf = np.nan
                    pvf = np.nan
                fh.write(','.join(map(str, [name, h, ci, cf, ai, af, pli, pvi, plf, pvf])) + '\n')
    print('[type3_alignment] wrote', OUTDIR)