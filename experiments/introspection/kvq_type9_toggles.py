import os
import torch
from config import RESULTS_DIR
from kvq_utils import kvq_inplace_modifiers, eval_loss
OUTDIR = os.path.join(RESULTS_DIR, 'kvq_type9_toggles')

def run(model: torch.nn.Module, ids: torch.Tensor):
    os.makedirs(OUTDIR, exist_ok=True)
    device = next(model.parameters()).device
    ids = ids.to(device)
    base = eval_loss(model, ids)
    q0 = None
    with kvq_inplace_modifiers(model, fQ=lambda x: x * 0.0):
        q0 = eval_loss(model, ids)
    k0 = None
    with kvq_inplace_modifiers(model, fK=lambda x: x * 0.0):
        k0 = eval_loss(model, ids)
    with open(os.path.join(OUTDIR, 'attention_toggle.txt'), 'w') as f:
        f.write(f'baseline: {base}\n')
        f.write(f'Q_zero:   {q0}\n')
        f.write(f'K_zero:   {k0}\n')