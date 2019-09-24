import torch
import torch.nn as nn

class Marginal(nn.Module):
    def __init__(self, N):
        super(Marginal, self).__init__()
        self.N = N
        self.w = nn.Parameter(torch.zeros(N, dtype=torch.float64))
    
    def forward(self, inputs):
        # log p(A) / log p(B)
        cste = torch.logsumexp(self.w, dim=0)
        return self.w[inputs] - cste

    def init_parameters(self):
        self.w.data.zero_()

class Conditional(nn.Module):
    def __init__(self, N):
        super(Conditional, self).__init__()
        self.N = N
        self.w = nn.Parameter(torch.zeros((N, N), dtype=torch.float64))
    
    def forward(self, conds, inputs):
        # log p(B | A) / log p(A | B)
        cste = torch.logsumexp(self.w[conds], dim=1)
        return self.w[conds, inputs] - cste

    def init_parameters(self):
        self.w.data.zero_()
