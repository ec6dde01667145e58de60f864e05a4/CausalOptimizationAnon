import torch
from causal_meta.modules.mlp import MLPModel
from causal_meta.bivariate.structural import BivariateStructuralModel

class StructuralModel(BivariateStructuralModel):
    def __init__(self, num_categories, hidden_size=8, num_nodes=2):
        model_A_B = MLPModel(num_categories, hidden_size=hidden_size,
            num_nodes=num_nodes, alpha=1)
        model_B_A = MLPModel(num_categories, hidden_size=hidden_size,
            num_nodes=num_nodes, alpha=0)
        super(StructuralModel, self).__init__(model_A_B, model_B_A,
            dtype=torch.float32)
