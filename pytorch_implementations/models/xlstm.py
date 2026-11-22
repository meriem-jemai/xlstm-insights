"""
xLSTM in PyTorch: a stack of residual sLSTM and mLSTM blocks.

The block order is given as a list, e.g. ["s", "m"] = sLSTM block, then mLSTM block.

Simplified compared to the paper (Beck et al., 2024):
- sLSTM block: LayerNorm -> (optional conv) -> sLSTM -> MLP, with a residual connection.
  The paper uses a gated MLP ("post up-projection").
- mLSTM block: LayerNorm -> MLP -> mLSTM -> MLP, gated, with a residual connection.
  The paper first projects UP to a larger dimension and runs the mLSTM there ("pre up-projection");
  here the mLSTM runs at the model dimension.
"""
import torch
import torch.nn as nn

from .slstm import sLSTMCell
from .mlstm import mLSTMCell


class sLSTMBlock(nn.Module):
    """Runs an sLSTM cell over the whole sequence."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.cell = sLSTMCell(dim, dim, num_heads=num_heads)

    def forward(self, x, states):
        outputs = []
        for t in range(x.size(1)):  # one time step after the other
            h, states = self.cell(x[:, t], states)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1), states


class mLSTMBlock(nn.Module):
    """Runs an mLSTM cell over the whole sequence (recurrent form)."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.cell = mLSTMCell(dim, dim, num_heads=num_heads)

    def forward(self, x, states):
        outputs = []
        for t in range(x.size(1)):
            h, states = self.cell(x[:, t], states)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1), states


class MLP(nn.Module):
    """Two-layer feed-forward network: dim -> 4*dim -> dim."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return self.net(x)


class ResidualSBlock(nn.Module):
    """x + MLP(sLSTM(LayerNorm(x)))"""

    def __init__(self, dim, num_heads=4, use_conv=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.use_conv = use_conv
        # optional causal-ish 1D convolution over time before the sLSTM (off by default)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1) if use_conv else nn.Identity()
        self.sblock = sLSTMBlock(dim, num_heads=num_heads)
        self.mlp = MLP(dim)

    def forward(self, x, s_states):
        x_norm = self.norm(x)
        x_conv = self.conv(x_norm.transpose(1, 2)).transpose(1, 2) if self.use_conv else x_norm
        x_s, s_states = self.sblock(x_conv, s_states)
        x_out = self.mlp(x_s)
        return x + x_out, s_states  # residual connection: the block only adds a change to x


class ResidualMBlock(nn.Module):
    """x + scale * gate(x) * MLP(mLSTM(MLP(LayerNorm(x))))"""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pre_mlp = MLP(dim)
        self.mblock = mLSTMBlock(dim, num_heads=num_heads)
        self.post_mlp = MLP(dim)
        self.output_gate = nn.Linear(dim, dim)          # decides how much of the block output is used
        self.skip_scale = nn.Parameter(torch.ones(1))   # learnable scale of the residual update

    def forward(self, x, m_states):
        x_norm = self.norm(x)
        x_pre = self.pre_mlp(x_norm)
        x_m, m_states = self.mblock(x_pre, m_states)
        x_post = self.post_mlp(x_m)
        gate = torch.sigmoid(self.output_gate(x_norm))
        out = x + self.skip_scale * gate * x_post       # residual connection
        return out, m_states


class xLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, block_types, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # character -> vector
        self.proj = nn.Linear(embedding_dim, hidden_size)

        # build the stack in the given order, e.g. ["s", "m"]
        self.blocks = nn.ModuleList()
        for b in block_types:
            if b == 's':
                self.blocks.append(ResidualSBlock(hidden_size, num_heads=num_heads))
            elif b == 'm':
                self.blocks.append(ResidualMBlock(hidden_size, num_heads=num_heads))
            else:
                raise ValueError(f"Invalid block type: {b}")

        self.fc = nn.Linear(hidden_size, vocab_size)  # vector -> scores per character

    def forward(self, x, all_states):
        x = self.embedding(x)
        x = self.proj(x)

        # every block has its own memory state
        new_states = []
        for block, state in zip(self.blocks, all_states):
            if isinstance(block, ResidualSBlock):
                x, state = block(x, state)
            elif isinstance(block, ResidualMBlock):
                x, state = block(x, state)
            new_states.append(state)

        logits = self.fc(x)
        return logits, new_states

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        H = self.proj.out_features
        assert H % self.num_heads == 0, "hidden_size must be divisible by num_heads"

        states = []
        for block in self.blocks:
            if isinstance(block, ResidualSBlock):
                # sLSTM state: (h, c, n, m) each [B, H]
                h = torch.zeros(batch_size, H, device=device)
                c = torch.zeros(batch_size, H, device=device)
                n = torch.zeros(batch_size, H, device=device)
                m = torch.zeros(batch_size, H, device=device)
                states.append((h, c, n, m))
            else:
                # mLSTM state: (C, n, m) with per-head shapes
                D = H // self.num_heads
                C = torch.zeros(batch_size, self.num_heads, D, D, device=device)
                n = torch.zeros(batch_size, self.num_heads, D, device=device)
                m = torch.zeros(batch_size, self.num_heads, 1, device=device)  # f: one stabilizer per head
                states.append((C, n, m))
        return states
