import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from config import RESULTS_DIR, GATE_EPS, LOGIT_SAT
from hooks import extract_series_per_head, extract_raw_series_per_head
OUTDIR = os.path.join(RESULTS_DIR, 'gates_type2_saturation')

def _save_bar(vals, title, outpath, ylabel='Value'):
    plt.figure(figsize=(8, 4))
    x = np.arange(len(vals))
    plt.bar(x, vals)
    plt.xticks(x, [str(i) for i in x])
    plt.title(title)
    plt.xlabel('Head index')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def _save_lines(series, labels, title, outpath):
    plt.figure(figsize=(8, 4))
    for y, lab in zip(series, labels):
        plt.plot(y, label=lab)
    plt.title(title)
    plt.xlabel('Time step')
    plt.ylabel('Fraction')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def run():
    print('[type2_saturation] start')
    os.makedirs(OUTDIR, exist_ok=True)
    i_all, f_all, i_ph, f_ph, _ = extract_series_per_head()
    S, H = i_ph.shape
    _, _, i_raw, f_raw, _ = extract_raw_series_per_head()
    i_sat_low = (i_ph <= GATE_EPS).mean(axis=0)
    i_sat_high = (i_ph >= 1.0 - GATE_EPS).mean(axis=0)
    f_sat_low = (f_ph <= GATE_EPS).mean(axis=0)
    f_sat_high = (f_ph >= 1.0 - GATE_EPS).mean(axis=0)
    i_sat_total = i_sat_low + i_sat_high
    f_sat_total = f_sat_low + f_sat_high
    i_raw_sat = (np.abs(i_raw) > LOGIT_SAT).mean(axis=0)
    f_raw_sat = (np.abs(f_raw) > LOGIT_SAT).mean(axis=0)
    frac_i_t = ((i_ph <= GATE_EPS) | (i_ph >= 1.0 - GATE_EPS)).mean(axis=1)
    frac_f_t = ((f_ph <= GATE_EPS) | (f_ph >= 1.0 - GATE_EPS)).mean(axis=1)
    csv_path = os.path.join(OUTDIR, 'type2_saturation_per_head.csv')
    with open(csv_path, 'w', encoding='utf-8') as fh:
        cols = ['head', 'i_sat_low', 'i_sat_high', 'i_sat_total', 'i_raw_sat', 'f_sat_low', 'f_sat_high', 'f_sat_total', 'f_raw_sat']
        fh.write(','.join(cols) + '\n')
        for h in range(H):
            fh.write(','.join(map(str, [h, i_sat_low[h], i_sat_high[h], i_sat_total[h], i_raw_sat[h], f_sat_low[h], f_sat_high[h], f_sat_total[h], f_raw_sat[h]])) + '\n')
    csv_layer = os.path.join(OUTDIR, 'type2_saturation_layer.csv')
    with open(csv_layer, 'w', encoding='utf-8') as fh:
        fh.write('metric,value\n')
        fh.write(f'i_sat_total_mean,{float(i_sat_total.mean())}\n')
        fh.write(f'f_sat_total_mean,{float(f_sat_total.mean())}\n')
        fh.write(f'i_raw_sat_mean,{float(i_raw_sat.mean())}\n')
        fh.write(f'f_raw_sat_mean,{float(f_raw_sat.mean())}\n')
    _save_bar(i_sat_total, 'Input gate saturation per head', os.path.join(OUTDIR, 'i_saturation_per_head.png'), 'Fraction')
    _save_bar(f_sat_total, 'Forget gate saturation per head', os.path.join(OUTDIR, 'f_saturation_per_head.png'), 'Fraction')
    _save_bar(i_raw_sat, 'Input gate raw-logit saturation per head', os.path.join(OUTDIR, 'i_raw_saturation_per_head.png'), 'Fraction')
    _save_bar(f_raw_sat, 'Forget gate raw-logit saturation per head', os.path.join(OUTDIR, 'f_raw_saturation_per_head.png'), 'Fraction')
    _save_lines([frac_i_t, frac_f_t], ['i', 'f'], 'Fraction of heads saturated over time', os.path.join(OUTDIR, 'saturation_over_time.png'))
    print('[type2_saturation] wrote', OUTDIR)