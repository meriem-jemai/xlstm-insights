import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from config import RESULTS_DIR
from hooks import extract_kvq_norm_series, safe_corrcoef
OUTDIR = os.path.join(RESULTS_DIR, 'kvq_type1')

def _save_heatmap_heads(mat: np.ndarray, title: str, outpath: str, xlabel: str=None):
    H = mat.shape[0]
    plt.figure(figsize=(8, max(4, 0.6 * H)))
    plt.imshow(mat, aspect='auto', origin='lower')
    plt.colorbar()
    plt.ylabel('Head index')
    if xlabel:
        plt.xlabel(xlabel)
    plt.yticks(np.arange(H), [str(i) for i in range(H)])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def _autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = (x - x.mean()) / (x.std() + 1e-08)
    ac = np.correlate(x, x, mode='full')
    mid = ac.size // 2
    ac = ac[mid:mid + max_lag + 1]
    n = np.arange(len(x), len(x) - max_lag - 1, -1)
    return ac / n

def _rolling_mean_std(x: np.ndarray, win: int):
    x = np.asarray(x, float)
    n = len(x)
    win = min(win, n) if n else win
    if n == 0:
        return (x, x)
    c = np.cumsum(np.r_[0.0, x])
    m = np.empty_like(x)
    m[win - 1:] = (c[win:] - c[:-win]) / win
    m[:win - 1] = m[win - 1]
    c2 = np.cumsum(np.r_[0.0, x * x])
    v = np.empty_like(x)
    v[win - 1:] = (c2[win:] - c2[:-win]) / win - m[win - 1:] ** 2
    v[:win - 1] = v[win - 1]
    return (m, np.sqrt(np.maximum(v, 0.0)))

def run(max_lag: int=256, roll_win: int=64):
    os.makedirs(OUTDIR, exist_ok=True)
    qL, kL, vL, qPH, kPH, vPH, H, hd = extract_kvq_norm_series()
    S = qPH.shape[0]
    max_lag = int(min(max_lag, S - 1))
    print(f'[type1] S={qPH.shape[0]} heads={H} head_dim={hd}')
    plt.figure(figsize=(8, 4))
    plt.plot(qL, label='Q (layer)')
    plt.plot(kL, label='K (layer)')
    plt.plot(vL, label='V (layer)')
    plt.title('KVQ layer mean norms over time')
    plt.xlabel('Time step')
    plt.ylabel('L2 norm (mean over heads)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'layer_mean_norms.png'))
    plt.close()
    _save_heatmap_heads(qPH.T, 'Q head norms', os.path.join(OUTDIR, 'Q_heatmap.png'))
    _save_heatmap_heads(kPH.T, 'K head norms', os.path.join(OUTDIR, 'K_heatmap.png'))
    _save_heatmap_heads(vPH.T, 'V head norms', os.path.join(OUTDIR, 'V_heatmap.png'))
    qC = safe_corrcoef(qPH.T)
    kC = safe_corrcoef(kPH.T)
    vC = safe_corrcoef(vPH.T)
    _save_heatmap_heads(qC, 'Q head correlation', os.path.join(OUTDIR, 'Q_head_corr.png'), xlabel='Head index')
    _save_heatmap_heads(kC, 'K head correlation', os.path.join(OUTDIR, 'K_head_corr.png'), xlabel='Head index')
    _save_heatmap_heads(vC, 'V head correlation', os.path.join(OUTDIR, 'V_head_corr.png'), xlabel='Head index')
    qAC = np.stack([_autocorr_1d(qPH[:, h], max_lag) for h in range(H)], axis=0)
    kAC = np.stack([_autocorr_1d(kPH[:, h], max_lag) for h in range(H)], axis=0)
    vAC = np.stack([_autocorr_1d(vPH[:, h], max_lag) for h in range(H)], axis=0)
    _save_heatmap_heads(qAC, 'Q norm autocorrelation', os.path.join(OUTDIR, 'Q_autocorr.png'), xlabel='Lag')
    _save_heatmap_heads(kAC, 'K norm autocorrelation', os.path.join(OUTDIR, 'K_autocorr.png'), xlabel='Lag')
    _save_heatmap_heads(vAC, 'V norm autocorrelation', os.path.join(OUTDIR, 'V_autocorr.png'), xlabel='Lag')
    for label, L in (('Q', qL), ('K', kL), ('V', vL)):
        m, s = _rolling_mean_std(L, roll_win)
        plt.figure(figsize=(8, 4))
        plt.plot(m, label=f'{label} rolling mean')
        plt.title(f'{label} layer rolling mean (w={roll_win})')
        plt.xlabel('Time step')
        plt.ylabel('Norm')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'{label}_rolling_mean.png'))
        plt.close()
        plt.figure(figsize=(8, 4))
        plt.plot(s, label=f'{label} rolling std')
        plt.title(f'{label} layer rolling std (w={roll_win})')
        plt.xlabel('Time step')
        plt.ylabel('Norm')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'{label}_rolling_std.png'))
        plt.close()
    csv = os.path.join(OUTDIR, 'kvq_type1_summary_per_head.csv')
    with open(csv, 'w', encoding='utf-8') as fh:
        fh.write(','.join(['head', 'Q_mean', 'Q_std', 'Q_min', 'Q_max', 'Q_range', 'K_mean', 'K_std', 'K_min', 'K_max', 'K_range', 'V_mean', 'V_std', 'V_min', 'V_max', 'V_range']) + '\n')
        for h in range(H):
            q = qPH[:, h]
            k = kPH[:, h]
            v = vPH[:, h]
            fh.write(','.join(map(str, [h, q.mean(), q.std(), q.min(), q.max(), q.max() - q.min(), k.mean(), k.std(), k.min(), k.max(), k.max() - k.min(), v.mean(), v.std(), v.min(), v.max(), v.max() - v.min()])) + '\n')
    csv2 = os.path.join(OUTDIR, 'kvq_type1_summary_layer.csv')
    with open(csv2, 'w', encoding='utf-8') as fh:
        fh.write('metric,value\n')
        fh.write(f'Q_layer_mean,{float(qL.mean())}\n')
        fh.write(f'K_layer_mean,{float(kL.mean())}\n')
        fh.write(f'V_layer_mean,{float(vL.mean())}\n')