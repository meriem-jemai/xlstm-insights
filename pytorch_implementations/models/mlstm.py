"""
mLSTM (Beck et al., 2024) in PyTorch.

- Matrix memory C per head, written with the outer product v k^T and read with the query q
- Exponential input and forget gates (one number per head), stabilized by the state m
- Normalizer state n, so that h = o * (C q) / max(|n^T q|, 1)
- No recurrent connection from h: everything depends on the input x only (this makes mLSTM parallelizable)
"""
import torch
import torch.nn as nn


class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # D, e.g. 256 / 4 = 64

        H, D, I = num_heads, self.head_dim, input_size

        def per_head_affine(out_dim):
            W = nn.Parameter(torch.empty(H, I, out_dim))
            b = nn.Parameter(torch.zeros(H, out_dim))
            # c: Xavier init for each head separately (on the 3D tensor PyTorch computed the wrong fan-in/out)
            for h in range(H):
                nn.init.xavier_uniform_(W.data[h])
            return W, b

        self.W_q, self.b_q = per_head_affine(D)  # query: what to read
        self.W_k, self.b_k = per_head_affine(D)  # key: where to write
        self.W_v, self.b_v = per_head_affine(D)  # value: what to write
        self.W_o, self.b_o = per_head_affine(D)  # output gate

        # f: input and forget gates are ONE number per head (as in the paper), not one per dimension.
        #    They scale the whole matrix memory C of the head. Before, each of the D dimensions had its
        #    own gate, which scaled C row by row and no longer matched the normalizer n.
        self.W_i, self.b_i = per_head_affine(1)
        self.W_f, self.b_f = per_head_affine(1)

        self.group_norm = nn.LayerNorm(D)  # LayerNorm over the units of each head
        self.scale = D ** 0.5              # keys are divided by sqrt(D)

    @staticmethod
    def _proj(x, W, b):
        # x: [B, I], W: [H, I, out] -> [B, H, out]
        return torch.einsum("bi,hio->bho", x, W) + b

    def forward(self, x, states):
        C_prev, n_prev, m_prev = states  # [B,H,D,D], [B,H,D], [B,H,1]

        q_t = self._proj(x, self.W_q, self.b_q)               # [B,H,D]
        k_t = self._proj(x, self.W_k, self.b_k) / self.scale  # [B,H,D]
        v_t = self._proj(x, self.W_v, self.b_v)               # [B,H,D]
        o_t = torch.sigmoid(self._proj(x, self.W_o, self.b_o))
        tilde_i = self._proj(x, self.W_i, self.b_i)           # [B,H,1]
        tilde_f = self._proj(x, self.W_f, self.b_f)           # [B,H,1]

        return self._memory_update(q_t, k_t, v_t, o_t, tilde_i, tilde_f, C_prev, n_prev, m_prev)

    # c: the memory update is a separate method, so the mLSTM variants can reuse it
    def _memory_update(self, q_t, k_t, v_t, o_t, tilde_i, tilde_f, C_prev, n_prev, m_prev):
        B = q_t.size(0)

        # stabilizer: keeps the exponential gates from overflowing
        m_t = torch.maximum(tilde_f + m_prev, tilde_i)        # [B,H,1]
        i_t = torch.exp(tilde_i - m_t)                        # [B,H,1]
        f_t = torch.exp(tilde_f + m_prev - m_t)               # [B,H,1]

        # write: C = f * C + i * v k^T (outer product), the same gate for the whole matrix
        outer_vk = v_t.unsqueeze(-1) @ k_t.unsqueeze(-2)      # [B,H,D,D]
        C_t = f_t.unsqueeze(-1) * C_prev + i_t.unsqueeze(-1) * outer_vk
        n_t = f_t * n_prev + i_t * k_t                        # [B,H,D]

        # read: C q, normalized by max(|n^T q|, lower bound)
        dot = (n_t * q_t).sum(dim=-1, keepdim=True)           # [B,H,1]
        # f: the lower bound is exp(-m) instead of 1, because C and n are stored scaled by exp(-m)
        #    (stabilized). Then this equals max(|n^T q|, 1) of the unstabilized formula from the paper.
        #    It is computed in log space, 1 / max(a, b) = exp(-max(log a, log b)), because exp(-m) alone
        #    overflowed to inf in float32 (NaN gradients); exp(-max(...)) can only become very small.
        # f: log(0) gives -inf and a NaN gradient (0 * inf) when n^T q is exactly 0 (e.g. the input gate
        #    underflowed to 0). Exact zeros skip the log; the result is the same as the paper's formula.
        abs_dot = dot.abs()
        nonzero = abs_dot > 0
        log_abs_dot = torch.where(nonzero, torch.log(torch.where(nonzero, abs_dot, torch.ones_like(abs_dot))),
                                  torch.full_like(abs_dot, float("-inf")))
        log_denom = torch.maximum(log_abs_dot, -m_t)
        h_tilde = (C_t @ q_t.unsqueeze(-1)).squeeze(-1) * torch.exp(-log_denom)  # [B,H,D]
        h_t = o_t * h_tilde

        out = self.group_norm(h_t).reshape(B, self.hidden_size)  # [B, hidden]
        return out, (C_t, n_t, m_t)


class mLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_heads=4):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, hidden_size)
        self.cell = mLSTMCell(hidden_size, hidden_size, num_heads=num_heads)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, states):
        x = self.proj(self.embedding(x))
        outputs = []
        for t in range(x.size(1)):
            h, states = self.cell(x[:, t], states)
            outputs.append(h.unsqueeze(1))
        logits = self.fc(torch.cat(outputs, dim=1))  # [B, T, vocab]
        return logits, states

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        H, D = self.num_heads, self.head_dim
        C = torch.zeros(batch_size, H, D, D, device=device)
        n = torch.zeros(batch_size, H, D, device=device)
        # f: one stabilizer per head, because the gates are one number per head
        m = torch.zeros(batch_size, H, 1, device=device)
        return (C, n, m)


