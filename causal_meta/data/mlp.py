import torch
import torch.nn as nn
import contextlib

from torch.distributions import Categorical
from copy import deepcopy

from causal_meta.modules.mlp import MLPModel


class GroundTruthMLPModel(MLPModel):
    def __init__(self, num_categories, hidden_size=8, num_nodes=2):
        # The ground truth model is A -> B
        super(GroundTruthMLPModel, self).__init__(num_categories,
            hidden_size=hidden_size, num_nodes=num_nodes, alpha=1)
        self._fix_parameters()

    def _fix_parameters(self):
        for param in self.parameters():
            param.requires_grad_(False)

    @contextlib.contextmanager
    def save_params(self):
        state_dict = deepcopy(self.state_dict())
        yield
        self.load_state_dict(state_dict)

    def intervene(self):
        nn.init.kaiming_normal_(self.W_1[0])
        nn.init.uniform_(self.b_0[0], -0.1, 0.1)
        nn.init.uniform_(self.b_1[0], -0.1, 0.1)

    def sample_iter(self, batch_size=1, num_batches=1000):
        for _ in range(num_batches):
            with torch.no_grad():
                samples = torch.zeros((batch_size, self.num_nodes), dtype=torch.long)
                # QKFIX: this should be in topological order
                for j in range(self.num_nodes):
                    logits = self.logits(samples)
                    dist = Categorical(logits=logits[:, j])
                    samples[:, j] = dist.sample()
            yield samples
