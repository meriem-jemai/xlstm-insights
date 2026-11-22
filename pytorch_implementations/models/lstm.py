"""
Standard LSTM (Hochreiter & Schmidhuber, 1997) in PyTorch, written out gate by gate.

Used as the baseline for the sLSTM, mLSTM and xLSTM models.
"""
import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # input weights W (with bias): how the current input x affects each gate
        self.w_z = nn.Linear(input_size, hidden_size)   # cell input
        self.w_i = nn.Linear(input_size, hidden_size)   # input gate
        self.w_f = nn.Linear(input_size, hidden_size)   # forget gate
        self.w_o = nn.Linear(input_size, hidden_size)   # output gate
        # recurrent weights R (no bias): how the previous hidden state h affects each gate.
        # These are full matrices, so every unit mixes with every other unit ("memory mixing").
        self.r_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.r_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.r_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.r_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, states):
        h_prev, c_prev = states
        z = torch.tanh(self.w_z(x) + self.r_z(h_prev))     # new content, between -1 and 1
        i = torch.sigmoid(self.w_i(x) + self.r_i(h_prev))  # how much new content to write (0..1)
        f = torch.sigmoid(self.w_f(x) + self.r_f(h_prev))  # how much old memory to keep (0..1)
        o = torch.sigmoid(self.w_o(x) + self.r_o(h_prev))  # how much memory to show (0..1)
        c_t = f * c_prev + i * z                            # cell state: keep part of the old, add new
        h_t = o * torch.tanh(c_t)                           # hidden state: squashed memory, gated
        return h_t, (h_t, c_t)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # character -> vector
        self.proj = nn.Linear(embedding_dim, hidden_size)
        self.cell = LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)              # vector -> scores per character

    def forward(self, x, states):
        x = self.embedding(x)  # [B, T, embedding]
        x = self.proj(x)       # [B, T, hidden]
        outputs = []
        for t in range(x.size(1)):  # recurrent: one time step after the other
            h, states = self.cell(x[:, t], states)
            outputs.append(h.unsqueeze(1))
        logits = self.fc(torch.cat(outputs, dim=1))  # [B, T, vocab]
        return logits, states

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        H = self.hidden_size
        h = torch.zeros(batch_size, H, device=device)  # hidden state starts at zero
        c = torch.zeros(batch_size, H, device=device)  # cell state starts at zero
        return (h, c)
