import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
from kvq_utils import eval_loss
from config import RESULTS_DIR

OUTDIR = os.path.join(RESULTS_DIR, "kvq_type5_lag")
os.makedirs(OUTDIR, exist_ok=True)



def run(model: torch.nn.Module, ids: torch.Tensor, buckets: List[Tuple[int, int]]=((1, 8), (9, 32), (33, 128), (129, 512), (513, 2048))):
    os.makedirs(OUTDIR, exist_ok=True)
    device = next(model.parameters()).device
    ids = ids.to(device)
    S = ids.size(1)

    def loss_lastN(N):
        N = int(max(1, min(N, S)))
        sub = ids[:, -N:]
        return eval_loss(model, sub)['loss']
    rows = []
    for Lmin, Lmax in buckets:
        los_max = loss_lastN(Lmax)
        los_pre = loss_lastN(max(1, Lmin - 1))
        contrib = los_pre - los_max
        rows.append((Lmin, Lmax, los_pre, los_max, contrib))
    with open(os.path.join(OUTDIR, 'lag_buckets.csv'), 'w') as fh:
        fh.write('lag_min,lag_max,loss_up_to_prev,loss_up_to_max,contrib\n')
        for Lmin, Lmax, pre, mx, contrib in rows:
            fh.write(f'{Lmin},{Lmax},{pre},{mx},{contrib}\n')
    labels = [f'{a}-{b}' for a, b, *_ in rows]
    vals = [r[-1] for r in rows]
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(vals)), vals)
    plt.xticks(np.arange(len(vals)), labels, rotation=30)
    plt.title('Lag bucket contribution (lower is worse loss; positive contrib = helpful)')
    plt.ylabel('Delta Delta loss from adding bucket')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'lag_bucket_contrib.png'))
    plt.close()

