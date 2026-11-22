"""
n: new file. Quick checks for the PyTorch models (runs on CPU in a few seconds).

What is checked:
    1. sLSTM: the per-head recurrence equals ONE big block-diagonal matrix (zeros between heads).
       This would have caught the old bug where only the diagonal of each head's matrix was used.
    2. mLSTM: the stabilized implementation gives exactly the same output as the plain,
       unstabilized formula from the paper. This proves the stabilizer m changes nothing in the math.
    3. xLSTM: the mLSTM stabilizer has one value per head (the gates are one number per head).
    4. Every model trains: forward + backward run, and all gradients are finite (no NaN/inf).

    python test_models.py
"""
import torch

from models.lstm import LSTMModel
from models.mlstm import mLSTMCell, mLSTMModel
from models.slstm import sLSTMCell, sLSTMModel
from models.mlstm_global_hidden import mLSTMModel as mLSTMGlobal
from models.mlstm_separate_hidden import mLSTMModel as mLSTMSeparate
from models.xlstm import xLSTMModel

torch.manual_seed(0)


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return ok


def test_slstm_recurrence_is_block_diagonal():
    """Each head must use its FULL D x D matrix (not only the diagonal)."""
    cell = sLSTMCell(8, 8, num_heads=2).double()  # tiny model: 2 heads of size 4, float64 for exact comparison
    B, H, D = 3, 2, 4
    h = torch.randn(B, H, D, dtype=torch.float64)
    out = cell._rec_proj(h, cell.R_z).reshape(B, H * D)
    # reference: build the full (8 x 8) matrix with the head blocks on the diagonal and zeros elsewhere
    block_diag = torch.block_diag(*cell.R_z.detach())
    expected = h.reshape(B, H * D) @ block_diag
    return check("sLSTM recurrence = block-diagonal matrix product", torch.allclose(out, expected))


def test_mlstm_stabilized_equals_unstabilized():
    """The stabilized mLSTM must give the same output as the plain formula from the paper."""
    cell = mLSTMCell(8, 8, num_heads=2).double()
    B, H, D, T = 3, 2, 4, 12
    xs = torch.randn(T, B, 8, dtype=torch.float64)  # 12 random time steps

    # stabilized (the implementation)
    state = (torch.zeros(B, H, D, D, dtype=torch.float64), torch.zeros(B, H, D, dtype=torch.float64),
             torch.zeros(B, H, 1, dtype=torch.float64))
    outs = []
    for x in xs:
        out, state = cell(x, state)
        outs.append(out)

    # unstabilized reference, written exactly like the paper:
    # C = f C + i v k^T,  n = f n + i k,  h = o * C q / max(|n^T q|, 1)
    C = torch.zeros(B, H, D, D, dtype=torch.float64)
    n = torch.zeros(B, H, D, dtype=torch.float64)
    refs = []
    with torch.no_grad():
        for x in xs:
            q = cell._proj(x, cell.W_q, cell.b_q)
            k = cell._proj(x, cell.W_k, cell.b_k) / cell.scale
            v = cell._proj(x, cell.W_v, cell.b_v)
            o = torch.sigmoid(cell._proj(x, cell.W_o, cell.b_o))
            i = torch.exp(cell._proj(x, cell.W_i, cell.b_i))  # plain exponential gates, no stabilizer
            f = torch.exp(cell._proj(x, cell.W_f, cell.b_f))
            C = f.unsqueeze(-1) * C + i.unsqueeze(-1) * (v.unsqueeze(-1) @ k.unsqueeze(-2))
            n = f * n + i * k
            denom = torch.clamp((n * q).sum(-1, keepdim=True).abs(), min=1.0)
            h = o * (C @ q.unsqueeze(-1)).squeeze(-1) / denom
            refs.append(cell.group_norm(h).reshape(B, H * D))

    ok = all(torch.allclose(a, b, atol=1e-8) for a, b in zip(outs, refs))
    return check("mLSTM stabilized = unstabilized formula", ok)


def test_models_train_step():
    """Every model runs forward + backward, and all parameters get finite gradients."""
    V, B, T = 20, 2, 6  # tiny vocabulary, batch and sequence: only checks that everything runs
    models = {
        "LSTM": LSTMModel(V, 16, 32),
        "sLSTM": sLSTMModel(V, 16, 32, num_heads=4),
        "mLSTM": mLSTMModel(V, 16, 32, num_heads=4),
        "mLSTM (global h)": mLSTMGlobal(V, 16, 32, num_heads=4),
        "mLSTM (separate h)": mLSTMSeparate(V, 16, 32, num_heads=4),
        "xLSTM (s,m)": xLSTMModel(V, 16, 32, ["s", "m"], num_heads=4),
    }
    # the mLSTM state of the xLSTM must have one stabilizer per head (last dimension 1)
    ok = check("xLSTM: mLSTM stabilizer has one value per head",
               models["xLSTM (s,m)"].init_hidden(B)[1][2].shape[-1] == 1)
    for name, model in models.items():
        x = torch.randint(0, V, (B, T))  # random characters
        y = torch.randint(0, V, (B, T))
        logits, _ = model(x, model.init_hidden(B))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()
        finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        ok &= check(f"{name}: forward + backward, finite gradients", finite and logits.shape == (B, T, V))
    return ok


if __name__ == "__main__":
    results = [
        test_slstm_recurrence_is_block_diagonal(),
        test_mlstm_stabilized_equals_unstabilized(),
        test_models_train_step(),
    ]
    print("\nAll tests passed." if all(results) else "\nSome tests failed.")
