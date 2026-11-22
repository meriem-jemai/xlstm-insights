"""
sLSTM (Beck et al., 2024) in PyTorch.

- Exponential input and forget gates, stabilized by the state m
- Normalizer state n, so that h = o * c / n
- Multiple heads: each head has its own recurrent matrix (block-diagonal recurrence)
"""
import torch
import torch.nn as nn


class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4, eps=1e-6):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # units per head, e.g. 256 / 4 = 64
        self.eps = eps

        H, D, I = num_heads, self.head_dim, input_size

        def per_head_affine():
            # input weights: every head sees the full input x and produces D values
            W = nn.Parameter(torch.empty(H, I, D))
            b = nn.Parameter(torch.zeros(H, D))
            # c: Xavier init for each head separately (on the 3D tensor PyTorch computed the wrong fan-in/out)
            for h in range(H):
                nn.init.xavier_uniform_(W.data[h])
            return W, b

        self.W_z, self.b_z = per_head_affine()  # cell input
        self.W_i, self.b_i = per_head_affine()  # input gate
        self.W_f, self.b_f = per_head_affine()  # forget gate
        self.W_o, self.b_o = per_head_affine()  # output gate

        def per_head_recurrent():
            # recurrent weights: one D x D matrix per head (= the blocks of a block-diagonal matrix)
            R = nn.Parameter(torch.empty(H, D, D))
            # c: Xavier init for each head separately
            for h in range(H):
                nn.init.xavier_uniform_(R.data[h])
            return R

        self.R_z = per_head_recurrent()
        self.R_i = per_head_recurrent()
        self.R_f = per_head_recurrent()
        self.R_o = per_head_recurrent()

        # LayerNorm over the units of each head ("group norm")
        self.group_norm = nn.LayerNorm(D)

    @staticmethod
    def _in_proj(x, W, b):
        # x: [B, I], W: [H, I, D] -> [B, H, D]
        return torch.einsum("bi,hid->bhd", x, W) + b

    @staticmethod
    def _rec_proj(h, R):
        # f: was einsum("bhd,hdd->bhd"). Repeating the letter d makes einsum take only the
        #    DIAGONAL of R, so every unit only saw its own previous value (no mixing inside a head).
        #    Now each head multiplies its slice of h with its full D x D matrix:
        #    h: [B, H, D], R: [H, D, E] -> [B, H, E]
        return torch.einsum("bhd,hde->bhe", h, R)

    def forward(self, x, states):
        h_prev, c_prev, n_prev, m_prev = states  # each [B, hidden]
        B = x.size(0)
        H, D = self.num_heads, self.head_dim

        # split the flat states into heads: [B, hidden] -> [B, H, D]
        h_prev = h_prev.view(B, H, D)
        c_prev = c_prev.view(B, H, D)
        n_prev = n_prev.view(B, H, D)
        m_prev = m_prev.view(B, H, D)

        # pre-activations = input part + recurrent part (per head)
        z_t = torch.tanh(self._in_proj(x, self.W_z, self.b_z) + self._rec_proj(h_prev, self.R_z))
        tilde_i = self._in_proj(x, self.W_i, self.b_i) + self._rec_proj(h_prev, self.R_i)
        tilde_f = self._in_proj(x, self.W_f, self.b_f) + self._rec_proj(h_prev, self.R_f)
        tilde_o = self._in_proj(x, self.W_o, self.b_o) + self._rec_proj(h_prev, self.R_o)

        # Exponential gates can overflow. The stabilizer m (largest exponent so far) keeps them small.
        m_t = torch.maximum(tilde_f + m_prev, tilde_i)
        i_t = torch.exp(tilde_i - m_t)              # stabilized input gate
        f_t = torch.exp(tilde_f + m_prev - m_t)     # stabilized forget gate
        o_t = torch.sigmoid(tilde_o)                # output gate

        c_t = f_t * c_prev + i_t * z_t              # cell state
        n_t = f_t * n_prev + i_t                    # normalizer
        h_t = o_t * c_t / (n_t + self.eps)          # hidden state

        # c: the group norm is applied only to the OUTPUT. The recurrent state keeps the raw h_t,
        #    as in the paper (before, the normalized h was fed back into the next step).
        out = self.group_norm(h_t).reshape(B, H * D)

        new_states = tuple(s.reshape(B, H * D) for s in (h_t, c_t, n_t, m_t))
        return out, new_states


class sLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_heads=4):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # character -> vector
        self.proj = nn.Linear(embedding_dim, hidden_size)
        self.cell = sLSTMCell(hidden_size, hidden_size, num_heads=num_heads)
        self.fc = nn.Linear(hidden_size, vocab_size)              # vector -> scores per character

    def forward(self, x, states):
        x = self.proj(self.embedding(x))  # [B, T, hidden]
        outputs = []
        for t in range(x.size(1)):        # recurrent: one time step after the other
            h, states = self.cell(x[:, t], states)
            outputs.append(h.unsqueeze(1))
        logits = self.fc(torch.cat(outputs, dim=1))  # [B, T, vocab]
        return logits, states

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        # h, c, n, m all start at zero
        return tuple(torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(4))
