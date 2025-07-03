import torch
from torch import nn


class LstmRnn(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)

        # learnable start states (1, H) – will be broadcast to batch
        self.h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(1, hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lstm.weight_ih)
        nn.init.orthogonal_(self.lstm.weight_hh)
        nn.init.zeros_(self.lstm.bias_ih)
        nn.init.zeros_(self.lstm.bias_hh)

    def forward(self, x, mask, backward: bool = False):
        """
        x    : (B, T, D)   batch-first
        mask : (B, T)      1 = keep token, 0 = padding
        """

        B, T, _ = x.shape
        prev_h  = self.h0.expand(B, -1).to(x.device)   # ensure same device
        prev_c  = self.c0.expand(B, -1).to(x.device)

        steps = range(T-1, -1, -1) if backward else range(T)
        h_seq = []

        for t in steps:
            m = mask[:, t].unsqueeze(1)        # (B,1) float / bool
            h_t, c_t = self.lstm(x[:, t], (prev_h, prev_c))

            prev_h = torch.where(m.bool(), h_t, prev_h)
            prev_c = torch.where(m.bool(), c_t, prev_c)
            h_seq.append(prev_h)

        # put time dimension back to (B, T, H)
        h_seq = h_seq[::-1] if backward else h_seq
        return torch.stack(h_seq, dim=1)