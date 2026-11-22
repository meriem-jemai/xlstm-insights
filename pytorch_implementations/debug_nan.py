"""
n: new file. Finds where a NaN/inf comes from.

Why: when the loss suddenly becomes NaN ("not a number"), all weights break and training is lost.
It usually comes from one operation, e.g. exp() of a huge number (= inf) or log(0) (= -inf),
and the gradient then turns into 0 * inf = NaN. This script finds that operation.

How:
    train.py saves the exact batch and the weights (before the update that would break them)
    to nan_debug.pt when a NaN/inf appears. This script replays that batch:
    1. forward: reports the first layer whose output is not finite
    2. backward with anomaly detection: PyTorch prints the operation whose gradient became NaN/inf

    python debug_nan.py
"""
import torch

from models import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saved = torch.load("nan_debug.pt", map_location=device, weights_only=False)
cfg = saved["config"]
vocab = saved["model"]["fc.weight"].shape[0]  # the output layer has one row per character

# rebuild the same model and load the weights from the moment before the NaN
model = build_model(cfg["model"], vocab, cfg["embedding"], cfg["hidden"], cfg["heads"],
                    cfg["blocks"], cfg["mlstm_variant"]).to(device)
model.load_state_dict(saved["model"])
inputs, targets = saved["inputs"].to(device), saved["targets"].to(device)

# 1) forward: hooks report every layer whose output contains NaN/inf (only the first time)
reported = set()


def make_hook(name):
    # a "hook" is a small function PyTorch calls automatically after a layer has computed its output
    def hook(module, inp, out):
        tensors = out if isinstance(out, (tuple, list)) else (out,)
        for t in tensors:
            if torch.is_tensor(t) and not torch.isfinite(t).all() and name not in reported:
                reported.add(name)
                print(f"[forward] non-finite output in: {name} ({module.__class__.__name__})")
    return hook


for name, module in model.named_modules():
    if name:
        module.register_forward_hook(make_hook(name))

# 2) backward with anomaly detection: raises at the first operation that produces NaN/inf in its gradient
torch.autograd.set_detect_anomaly(True)
logits, _ = model(inputs, model.init_hidden(inputs.size(0)))
loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
print(f"loss: {loss.item()}")
if not reported:
    print("[forward] all outputs finite, so the problem is in the backward pass:")
loss.backward()

bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
print("parameters with non-finite gradients:", bad if bad else "none (not reproduced)")
