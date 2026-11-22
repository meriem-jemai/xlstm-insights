"""
mLSTM variant with a SEPARATE (per-head) hidden-state recurrence.

Every head sees x_t and only ITS OWN slice of the previous output h_{t-1}:
the projections of head i get [x_t ; h_{t-1}^(i)] as input. Heads stay independent.
The memory update itself is the same as in the standard mLSTM (see mlstm.py).
"""
import torch

from .mlstm import mLSTMCell as BaseCell
from .mlstm import mLSTMModel as BaseModel


class mLSTMCell(BaseCell):
    def __init__(self, input_size, hidden_size, num_heads=4):
        # c: reuses the standard mLSTM cell; each head's projections get [x ; own h slice] as input
        super().__init__(input_size + hidden_size // num_heads, hidden_size, num_heads)

    @staticmethod
    def _proj_heads(xh, W, b):
        # xh: [B, H, in] (a different input per head), W: [H, in, out] -> [B, H, out]
        return torch.einsum("bhi,hio->bho", xh, W) + b

    def forward(self, x, states):
        h_prev, C_prev, n_prev, m_prev = states
        B, H, D = x.size(0), self.num_heads, self.head_dim

        x_heads = x.unsqueeze(1).expand(-1, H, -1)        # [B, H, input]: same x for every head
        h_heads = h_prev.view(B, H, D)                    # [B, H, D]: each head's own slice
        xh = torch.cat([x_heads, h_heads], dim=-1)        # [B, H, input + D]

        q_t = self._proj_heads(xh, self.W_q, self.b_q)
        k_t = self._proj_heads(xh, self.W_k, self.b_k) / self.scale
        v_t = self._proj_heads(xh, self.W_v, self.b_v)
        o_t = torch.sigmoid(self._proj_heads(xh, self.W_o, self.b_o))
        tilde_i = self._proj_heads(xh, self.W_i, self.b_i)   # f: one gate value per head
        tilde_f = self._proj_heads(xh, self.W_f, self.b_f)

        # f: same corrected memory update as the standard mLSTM (scalar gates, exp(-m) bound)
        out, (C_t, n_t, m_t) = self._memory_update(q_t, k_t, v_t, o_t, tilde_i, tilde_f, C_prev, n_prev, m_prev)
        return out, (out, C_t, n_t, m_t)   # the output is fed back as h in the next step


class mLSTMModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_heads=4):
        super().__init__(vocab_size, embedding_dim, hidden_size, num_heads)
        self.cell = mLSTMCell(hidden_size, hidden_size, num_heads=num_heads)

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        return (h, *super().init_hidden(batch_size, device))  # (h, C, n, m)
