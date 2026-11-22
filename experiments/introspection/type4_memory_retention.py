"""
Memory retention: how many steps does old information survive in the memory?

f: uses the ABSOLUTE forget gate f = sigmoid(fraw) (the library computes log f = logsigmoid(fraw) in both
   sLSTM and mLSTM). Before, the stabilized gate f' was used. f' is only a RELATIVE weight (old memory vs.
   the new input at this step, after shifting by the stabilizer m), so a strong new input makes f' small
   even if the model keeps its memory. That gave meaningless half-lives below one step for the sLSTM.

For each head: tau = 1 / mean(-log f),  half-life = ln(2) * tau  (steps until old content is halved).
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple
from config import RESULTS_DIR, NUM_HEADS
import hooks
OUTDIR = os.path.join(RESULTS_DIR, 'gates_type4_memory_retention')

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

def run():
    print('[type4_memory_retention] start')
    os.makedirs(OUTDIR, exist_ok=True)
    # f: absolute forget gate from the full pre-activation, averaged over the units of each head
    f_abs = torch.sigmoid(hooks.FGATE_RAW_OUTS[0].float())          # [B, S, units]
    B, S_, U = f_abs.shape
    f_ph = f_abs.view(B, S_, NUM_HEADS, U // NUM_HEADS).mean(dim=(0, 3)).numpy()  # [S, heads]
    S, H = f_ph.shape
    eps = 1e-09
    neglog = -np.log(np.clip(f_ph, eps, 1.0))
    tau = 1.0 / np.clip(neglog.mean(axis=0), eps, None)
    half_life = np.log(2.0) * tau
    frac_f_gt_09 = (f_ph > 0.9).mean(axis=0)
    frac_f_gt_099 = (f_ph > 0.99).mean(axis=0)
    f_q10 = np.quantile(f_ph, 0.1, axis=0)
    f_q90 = np.quantile(f_ph, 0.9, axis=0)
    csv_head = os.path.join(OUTDIR, 'type4_retention_per_head.csv')
    with open(csv_head, 'w', encoding='utf-8') as fh:
        fh.write(','.join(['head', 'tau', 'half_life', 'frac_f_gt_0.9', 'frac_f_gt_0.99', 'f_q10', 'f_q90']) + '\n')
        for h in range(H):
            fh.write(','.join(map(str, [h, tau[h], half_life[h], frac_f_gt_09[h], frac_f_gt_099[h], f_q10[h], f_q90[h]])) + '\n')
    csv_layer = os.path.join(OUTDIR, 'type4_retention_layer.csv')
    with open(csv_layer, 'w', encoding='utf-8') as fh:
        fh.write('metric,value\n')
        fh.write(f'half_life_median,{float(np.median(half_life))}\n')
        fh.write(f'half_life_mean,{float(np.mean(half_life))}\n')
        fh.write(f'frac_heads_f_gt_0.9_mean,{float(frac_f_gt_09.mean())}\n')
        fh.write(f'frac_heads_f_gt_0.99_mean,{float(frac_f_gt_099.mean())}\n')
    _save_bar(half_life, 'Half-life from forget gate (per head)', os.path.join(OUTDIR, 'half_life_per_head.png'), 'Steps')
    plt.figure(figsize=(8, 4))
    plt.hist(half_life, bins=40)
    plt.title('Half-life distribution')
    plt.xlabel('Steps')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'half_life_hist.png'))
    plt.close()
    print('[type4_memory_retention] wrote', OUTDIR)
