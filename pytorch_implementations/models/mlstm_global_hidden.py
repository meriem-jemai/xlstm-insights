"""
mLSTM variant with a GLOBAL hidden-state recurrence.

Unlike the standard mLSTM (which only sees x_t), every head here also sees the full previous
output h_{t-1} of all heads: the projections get [x_t ; h_{t-1}] as input.
The memory update itself is the same as in the standard mLSTM (see mlstm.py).
"""
import torch

from .mlstm import mLSTMCell as BaseCell
from .mlstm import mLSTMModel as BaseModel


class mLSTMCell(BaseCell):
    def __init__(self, input_size, hidden_size, num_heads=4):
        # c: reuses the standard mLSTM cell; the projections just get a bigger input [x ; h_prev]
        super().__init__(input_size + hidden_size, hidden_size, num_heads)

    def forward(self, x, states):
        h_prev, C_prev, n_prev, m_prev = states   # h_prev: [B, hidden]
        xh = torch.cat([x, h_prev], dim=-1)        # every head sees x and the FULL h_prev

        q_t = self._proj(xh, self.W_q, self.b_q)
        k_t = self._proj(xh, self.W_k, self.b_k) / self.scale
        v_t = self._proj(xh, self.W_v, self.b_v)
        o_t = torch.sigmoid(self._proj(xh, self.W_o, self.b_o))
        tilde_i = self._proj(xh, self.W_i, self.b_i)   # f: one gate value per head (via the base cell)
        tilde_f = self._proj(xh, self.W_f, self.b_f)

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
