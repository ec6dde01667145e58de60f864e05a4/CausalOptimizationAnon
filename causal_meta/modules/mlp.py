import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


class MLPModel(nn.Module):
    def __init__(self, num_categories, hidden_size=8, num_nodes=2, alpha=1):
        super(MLPModel, self).__init__()

        assert (alpha == 0) or (alpha == 1)
        assert num_nodes == 2
        self.num_categories = num_categories
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.alpha = alpha

        self.W_0 = nn.Parameter(torch.FloatTensor(self.num_categories, self.hidden_size))
        self.b_0 = nn.Parameter(torch.FloatTensor(self.num_nodes, self.hidden_size))
        self.W_1 = nn.Parameter(torch.FloatTensor(self.num_nodes, self.hidden_size, self.num_categories))
        self.b_1 = nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_categories))

        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.W_0)
        for j in range(self.num_nodes):
            nn.init.kaiming_normal_(self.W_1[j])
        nn.init.uniform_(self.b_0, -1., 1.)
        nn.init.uniform_(self.b_1, -1., 1.)

    def logits(self, samples):
        batch_size = samples.size(0)
        # First MLP
        hidden_0 = self.b_0[0].unsqueeze(0).expand(batch_size, self.hidden_size)
        # Second MLP
        hidden_1 = torch.index_select(self.W_0, 0, samples[:, 1 - self.alpha]) + self.b_0[1]

        hidden = torch.stack([hidden_0, hidden_1] if self.alpha else [hidden_1, hidden_0], dim=1)
        hidden = F.leaky_relu(hidden)
        logits = torch.einsum('bij,ijk->bik', hidden, self.W_1) + self.b_1

        return logits

    def forward(self, samples):
        dist = Categorical(logits=self.logits(samples))
        return dist.log_prob(samples)
