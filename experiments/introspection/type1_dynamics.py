import os
from typing import Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from config import RESULTS_DIR, PATTERN_PROVIDER
from hooks import extract_series_per_head, safe_corrcoef
from itertools import cycle
from matplotlib.lines import Line2D
from type3_alignment import _load_patterns_from_provider, _spans_to_events, _auto_policy
OUTDIR = os.path.join(RESULTS_DIR, 'gates_type1_dynamics')

def _autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, float)
    x = (x - x.mean()) / (x.std() + 1e-08)
    ac = np.correlate(x, x, mode='full')
    mid = ac.size // 2
    ac = ac[mid:mid + max_lag + 1]
    n = np.arange(len(x), len(x) - max_lag - 1, -1)
    return ac / n

def _crosscorr_1d(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = (x - x.mean()) / (x.std() + 1e-08)
    y = (y - y.mean()) / (y.std() + 1e-08)
    cc = np.correlate(x, y, mode='full')
    mid = cc.size // 2
    denom = len(x) - np.abs(np.arange(-max_lag, max_lag + 1))
    return cc[mid - max_lag:mid + max_lag + 1] / denom

def _rolling_mean_std(x: np.ndarray, win: int):
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return (x, x)
    if win > n:
        win = n
    c = np.cumsum(np.r_[0.0, x])
    m = np.empty_like(x)
    m[win - 1:] = (c[win:] - c[:-win]) / win
    m[:win - 1] = m[win - 1]
    c2 = np.cumsum(np.r_[0.0, x * x])
    v = np.empty_like(x)
    v[win - 1:] = (c2[win:] - c2[:-win]) / win - m[win - 1:] ** 2
    v[:win - 1] = v[win - 1]
    return (m, np.sqrt(np.maximum(v, 0.0)))

def _get_events(text: Optional[str], S: int, select: Optional[List[str]]=None) -> Dict[str, np.ndarray]:
    if not text or not PATTERN_PROVIDER:
        return {}
    patt_spans = _load_patterns_from_provider(text[:S], S)
    if select:
        want = {s.lower() for s in select}
        patt_spans = {k: v for k, v in patt_spans.items() if k.lower() in want}
    return {name: _spans_to_events(spans, _auto_policy(spans)) for name, spans in patt_spans.items()}

def _save_lines_with_events(series: List[np.ndarray], labels: List[str], title: str, outpath: str, events: Dict[str, np.ndarray], max_vlines: int=400, event_colors: Optional[Dict[str, str]]=None, event_alpha: float=0.18, event_lw: float=0.8):
    plt.figure(figsize=(8, 4))
    for y, lab in zip(series, labels):
        plt.plot(y, label=lab)
    ax = plt.gca()
    if event_colors is None:
        palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'])
        color_cycle = cycle(palette)
        event_colors = {name: next(color_cycle) for name in events.keys()}
    proxies, proxy_labels = ([], [])
    for name, ev in events.items():
        if ev is None:
            continue
        col = event_colors.get(name)
        xs = np.asarray(ev, int)[:max_vlines]
        for xi in xs:
            ax.axvline(int(xi), color=col, alpha=event_alpha, linewidth=event_lw)
        proxies.append(Line2D([0], [0], color=col, lw=2))
        proxy_labels.append(name)
    handles, gate_labels = ax.get_legend_handles_labels()
    ax.legend(handles + proxies, gate_labels + proxy_labels, fontsize=8, ncol=1, loc='upper right')
    plt.title(title)
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def _save_heatmap_heads(mat: np.ndarray, title: str, outpath: str, xlabel: Optional[str]=None):
    H = mat.shape[0]
    plt.figure(figsize=(8, max(4, H * 0.6)))
    plt.imshow(mat, aspect='auto', origin='lower')
    plt.colorbar()
    plt.ylabel('Head index')
    if xlabel:
        plt.xlabel(xlabel)
    plt.yticks(np.arange(H), [str(i) for i in range(H)])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.xticks(range(H))
    plt.yticks(range(H))
    plt.close()

def _estimate_period(ac_row: np.ndarray, min_lag: int=2) -> float:
    if ac_row.size <= min_lag:
        return np.nan
    k = np.argmax(ac_row[min_lag:]) + min_lag
    return float(k) if ac_row[k] > 0 else np.nan

def _first_decay_lag(ac_row: np.ndarray, thr: float=np.exp(-1)) -> float:
    if ac_row.size <= 1:
        return np.nan
    arr = ac_row[1:]
    idx = np.where(arr < thr)[0]
    return float(idx[0] + 1) if idx.size else np.nan

def _offdiag_abs_mean(C: np.ndarray) -> float:
    if C.size == 0:
        return np.nan
    n = C.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = np.abs(C[mask])
    return float(vals.mean()) if vals.size else np.nan

def _safe_pcorr(x: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(x, float)
    b = np.asarray(b, float)
    if x.std() < 1e-08 or b.std() < 1e-08:
        return np.nan
    return float(np.corrcoef(x, b)[0, 1])

def run(text: Optional[str]=None, max_lag: int=256, roll_win: int=64, mark_patterns: Optional[List[str]]=None, max_vlines: int=10000, event_colors: Optional[Dict[str, str]]=None, event_alpha: float=0.18, event_lw: float=0.8):
    print('[type1_dynamics] start')
    os.makedirs(OUTDIR, exist_ok=True)
    i_all, f_all, i_ph, f_ph, _ = extract_series_per_head()
    S, H = i_ph.shape
    max_lag = int(min(max_lag, S - 1))
    events = _get_events(text, S, select=mark_patterns)
    _save_lines_with_events([i_all, f_all], ['i (layer)', 'f (layer)'], 'Layer mean gates over time', os.path.join(OUTDIR, 'layer_mean.png'), events, max_vlines=max_vlines, event_colors=event_colors, event_alpha=event_alpha, event_lw=event_lw)
    _save_heatmap_heads(i_ph.T, 'Input gate per head', os.path.join(OUTDIR, 'i_heatmap.png'))
    _save_heatmap_heads(f_ph.T, 'Forget gate per head', os.path.join(OUTDIR, 'f_heatmap.png'))
    i_corr = safe_corrcoef(i_ph.T)
    f_corr = safe_corrcoef(f_ph.T)
    np.save(os.path.join(OUTDIR, 'i_head_corr.npy'), i_corr)
    np.save(os.path.join(OUTDIR, 'f_head_corr.npy'), f_corr)
    _save_heatmap_heads(i_corr, 'Head correlation (input gate)', os.path.join(OUTDIR, 'i_head_corr.png'), xlabel='Head index')
    _save_heatmap_heads(f_corr, 'Head correlation (forget gate)', os.path.join(OUTDIR, 'f_head_corr.png'), xlabel='Head index')
    i_ac = np.stack([_autocorr_1d(i_ph[:, h], max_lag) for h in range(H)], axis=0)
    f_ac = np.stack([_autocorr_1d(f_ph[:, h], max_lag) for h in range(H)], axis=0)
    np.save(os.path.join(OUTDIR, 'i_autocorr.npy'), i_ac)
    np.save(os.path.join(OUTDIR, 'f_autocorr.npy'), f_ac)
    _save_heatmap_heads(i_ac, 'Autocorrelation (input gate)', os.path.join(OUTDIR, 'i_autocorr_heatmap.png'), xlabel='Lag')
    _save_heatmap_heads(f_ac, 'Autocorrelation (forget gate)', os.path.join(OUTDIR, 'f_autocorr_heatmap.png'), xlabel='Lag')
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.stack([_crosscorr_1d(i_ph[:, h], f_ph[:, h], max_lag) for h in range(H)], axis=0)
    np.save(os.path.join(OUTDIR, 'if_crosscorr.npy'), cc)
    np.save(os.path.join(OUTDIR, 'if_crosscorr_lags.npy'), lags)
    _save_heatmap_heads(cc, 'Input vs forget cross-correlation (per head)', os.path.join(OUTDIR, 'if_crosscorr_heatmap.png'), xlabel='Lag (i leads)')
    i_mean, i_std = _rolling_mean_std(i_all, roll_win)
    f_mean, f_std = _rolling_mean_std(f_all, roll_win)
    _save_lines_with_events([i_mean, f_mean], [f'i rolling (w={roll_win})', f'f rolling (w={roll_win})'], 'Rolling layer mean', os.path.join(OUTDIR, 'rolling_mean_layer.png'), events, max_vlines=max_vlines, event_colors=event_colors, event_alpha=event_alpha, event_lw=event_lw)
    _save_lines_with_events([i_std, f_std], [f'i std (w={roll_win})', f'f std (w={roll_win})'], 'Rolling layer standard deviation', os.path.join(OUTDIR, 'rolling_std_layer.png'), events, max_vlines=max_vlines, event_colors=event_colors, event_alpha=event_alpha, event_lw=event_lw)
    i_min = i_ph.min(axis=0)
    i_max = i_ph.max(axis=0)
    i_rng = i_max - i_min
    f_min = f_ph.min(axis=0)
    f_max = f_ph.max(axis=0)
    f_rng = f_max - f_min
    i_mu = i_ph.mean(axis=0)
    i_sd = i_ph.std(axis=0)
    f_mu = f_ph.mean(axis=0)
    f_sd = f_ph.std(axis=0)
    i_ac_period = np.array([_estimate_period(i_ac[h]) for h in range(H)])
    f_ac_period = np.array([_estimate_period(f_ac[h]) for h in range(H)])
    i_decay = np.array([_first_decay_lag(i_ac[h]) for h in range(H)])
    f_decay = np.array([_first_decay_lag(f_ac[h]) for h in range(H)])
    zero_band = 5
    idx0 = np.where(lags == 0)[0][0]
    cc_zero_mean = cc[:, max(0, idx0 - zero_band):idx0 + zero_band + 1].mean(axis=1)
    cc_min_idx = cc.argmin(axis=1)
    cc_max_idx = cc.argmax(axis=1)
    cc_min_lag = lags[cc_min_idx]
    cc_max_lag = lags[cc_max_idx]
    cc_min_val = cc[np.arange(H), cc_min_idx]
    cc_max_val = cc[np.arange(H), cc_max_idx]
    per_head_csv = os.path.join(OUTDIR, 'type1_summary_per_head.csv')
    with open(per_head_csv, 'w', encoding='utf-8') as fh:
        cols = ['head', 'i_mean', 'i_std', 'i_min', 'i_max', 'i_range', 'f_mean', 'f_std', 'f_min', 'f_max', 'f_range', 'i_period', 'f_period', 'i_decay_lag', 'f_decay_lag', 'if_corr_mean_near_zero', 'if_corr_lag_min', 'if_corr_min', 'if_corr_lag_max', 'if_corr_max']
        fh.write(','.join(cols) + '\n')
        for h in range(H):
            fh.write(','.join(map(str, [h, i_mu[h], i_sd[h], i_min[h], i_max[h], i_rng[h], f_mu[h], f_sd[h], f_min[h], f_max[h], f_rng[h], i_ac_period[h], f_ac_period[h], i_decay[h], f_decay[h], cc_zero_mean[h], cc_min_lag[h], cc_min_val[h], cc_max_lag[h], cc_max_val[h]])) + '\n')
    i_corr_off = _offdiag_abs_mean(i_corr)
    f_corr_off = _offdiag_abs_mean(f_corr)
    layer_csv = os.path.join(OUTDIR, 'type1_summary_layer.csv')
    with open(layer_csv, 'w', encoding='utf-8') as fh:
        fh.write('metric,value\n')
        fh.write(f'i_layer_mean,{float(i_all.mean())}\n')
        fh.write(f'f_layer_mean,{float(f_all.mean())}\n')
        fh.write(f'i_layer_std,{float(i_all.std())}\n')
        fh.write(f'f_layer_std,{float(f_all.std())}\n')
        fh.write(f'i_heads_avg_corr,{i_corr_off}\n')
        fh.write(f'f_heads_avg_corr,{f_corr_off}\n')
        fh.write(f'period_estimate_i_median,{float(np.nanmedian(i_ac_period))}\n')
        fh.write(f'period_estimate_f_median,{float(np.nanmedian(f_ac_period))}\n')
    if events:
        patt_sum_csv = os.path.join(OUTDIR, 'type1_patterns_summary.csv')
        with open(patt_sum_csv, 'w', encoding='utf-8') as fh:
            cols = ['pattern', 'n_events', 'event_fraction', 'i_layer_mean_at_events', 'i_layer_mean_at_nonevents', 'i_layer_diff_ev_minus_non', 'f_layer_mean_at_events', 'f_layer_mean_at_nonevents', 'f_layer_diff_ev_minus_non', 'i_layer_corr_with_events', 'f_layer_corr_with_events', 'top3_i_heads_up', 'top3_i_values_up', 'top3_i_heads_down', 'top3_i_values_down', 'top3_f_heads_up', 'top3_f_values_up', 'top3_f_heads_down', 'top3_f_values_down']
            fh.write(','.join(cols) + '\n')
            for name, idx in events.items():
                idx = np.asarray(idx, int)
                idx = idx[(idx >= 0) & (idx < S)]
                n = int(idx.size)
                if n == 0:
                    row = [name, 0, 0.0] + [np.nan] * (len(cols) - 3)
                    fh.write(','.join(map(str, row)) + '\n')
                    continue
                mask = np.zeros(S, dtype=bool)
                mask[idx] = True
                non_idx = np.where(~mask)[0]
                frac = float(n / S)
                i_ev = float(i_all[idx].mean())
                f_ev = float(f_all[idx].mean())
                i_ne = float(i_all[non_idx].mean()) if non_idx.size else np.nan
                f_ne = float(f_all[non_idx].mean()) if non_idx.size else np.nan
                i_diff = i_ev - i_ne if non_idx.size else np.nan
                f_diff = f_ev - f_ne if non_idx.size else np.nan
                b = mask.astype(float)
                i_corr_ev = _safe_pcorr(i_all, b)
                f_corr_ev = _safe_pcorr(f_all, b)
                i_head_ev = i_ph[idx, :].mean(axis=0)
                f_head_ev = f_ph[idx, :].mean(axis=0)
                if non_idx.size:
                    i_head_ne = i_ph[non_idx, :].mean(axis=0)
                    f_head_ne = f_ph[non_idx, :].mean(axis=0)
                else:
                    i_head_ne = np.full(H, np.nan)
                    f_head_ne = np.full(H, np.nan)
                i_head_diff = i_head_ev - i_head_ne
                f_head_diff = f_head_ev - f_head_ne

                def _tops(dv: np.ndarray, k: int=3):
                    order_up = np.argsort(-dv)[:min(k, len(dv))]
                    order_dn = np.argsort(dv)[:min(k, len(dv))]
                    up_idx = '|'.join(map(str, order_up.tolist()))
                    up_val = '|'.join((f'{dv[h]:.4f}' for h in order_up.tolist()))
                    dn_idx = '|'.join(map(str, order_dn.tolist()))
                    dn_val = '|'.join((f'{dv[h]:.4f}' for h in order_dn.tolist()))
                    return (up_idx, up_val, dn_idx, dn_val)
                i_up_idx, i_up_val, i_dn_idx, i_dn_val = _tops(i_head_diff, 3)
                f_up_idx, f_up_val, f_dn_idx, f_dn_val = _tops(f_head_diff, 3)
                row = [name, n, frac, i_ev, i_ne, i_diff, f_ev, f_ne, f_diff, i_corr_ev, f_corr_ev, i_up_idx, i_up_val, i_dn_idx, i_dn_val, f_up_idx, f_up_val, f_dn_idx, f_dn_val]
                fh.write(','.join(map(str, row)) + '\n')
        patt_head_csv = os.path.join(OUTDIR, 'type1_patterns_by_head.csv')
        with open(patt_head_csv, 'w', encoding='utf-8') as fh:
            cols = ['pattern', 'head', 'i_mean_at_events', 'i_mean_at_nonevents', 'i_diff', 'f_mean_at_events', 'f_mean_at_nonevents', 'f_diff', 'i_corr_with_events', 'f_corr_with_events']
            fh.write(','.join(cols) + '\n')
            for name, idx in events.items():
                idx = np.asarray(idx, int)
                idx = idx[(idx >= 0) & (idx < S)]
                if idx.size == 0:
                    continue
                mask = np.zeros(S, dtype=bool)
                mask[idx] = True
                non_idx = np.where(~mask)[0]
                b = mask.astype(float)
                for h in range(H):
                    i_ev_h = float(i_ph[idx, h].mean())
                    f_ev_h = float(f_ph[idx, h].mean())
                    if non_idx.size:
                        i_ne_h = float(i_ph[non_idx, h].mean())
                        f_ne_h = float(f_ph[non_idx, h].mean())
                        i_diff_h = i_ev_h - i_ne_h
                        f_diff_h = f_ev_h - f_ne_h
                    else:
                        i_ne_h = np.nan
                        f_ne_h = np.nan
                        i_diff_h = np.nan
                        f_diff_h = np.nan
                    i_corr_h = _safe_pcorr(i_ph[:, h], b)
                    f_corr_h = _safe_pcorr(f_ph[:, h], b)
                    row = [name, h, i_ev_h, i_ne_h, i_diff_h, f_ev_h, f_ne_h, f_diff_h, i_corr_h, f_corr_h]
                    fh.write(','.join(map(str, row)) + '\n')
    readme = os.path.join(OUTDIR, 'type1_summary.txt')
    with open(readme, 'w', encoding='utf-8') as fh:
        fh.write('Type-1 Dynamics Summary\n')
        fh.write('=======================\n')
        fh.write(f'S={S}, H={H}, max_lag={max_lag}, roll_win={roll_win}\n\n')
        fh.write(f'Layer means: i={i_all.mean():.3f} +/- {i_all.std():.3f}, f={f_all.mean():.3f} +/- {f_all.std():.3f}\n')
        i_corr_off = _offdiag_abs_mean(i_corr)
        f_corr_off = _offdiag_abs_mean(f_corr)
        fh.write(f'Head correlation off-diagonal mean (abs): i={i_corr_off:.3f}, f={f_corr_off:.3f}\n')
        fh.write(f'Period (median across heads, steps): i={np.nanmedian(i_ac_period):.0f}, f={np.nanmedian(f_ac_period):.0f}\n')
        order_decay = np.argsort(np.nan_to_num(-f_decay, nan=-1000000000.0))
        top = order_decay[:min(5, H)]
        fh.write('Top long-memory heads (by f autocorr decay lag): ' + ', '.join([f'h{int(h)}~{float(f_decay[h]):.0f}' for h in top]) + '\n')
        order_range = np.argsort(np.nan_to_num(-(i_rng + f_rng), nan=-1000000000.0))
        top2 = order_range[:min(5, H)]
        fh.write('Top oscillatory heads (by range i+f): ' + ', '.join([f'h{int(h)}~{float(i_rng[h] + f_rng[h]):.3f}' for h in top2]) + '\n')
        fh.write(f'Average input vs forget correlation near lag 0: {cc_zero_mean.mean():.3f}\n')
        fh.write('Notes: period uses first non-zero autocorr peak; decay lag is first tau where autocorr < e^-1. Pattern CSVs include layer-level and per-head event effects.\n')
    print('[type1_dynamics] wrote', OUTDIR)