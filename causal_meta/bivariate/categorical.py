import numpy as np

import torch
import torch.nn as nn

from causal_meta.modules.categorical import Marginal, Conditional
from causal_meta.bivariate.structural import BivariateStructuralModel

class Model(nn.Module):
    def __init__(self, N):
        super(Model, self).__init__()
        self.N = N

    def set_maximum_likelihood(self, inputs):
        inputs_A, inputs_B = np.split(inputs.numpy(), 2, axis=1)
        num_samples = inputs_A.shape[0]
        pi_A = np.zeros((self.N,), dtype=np.float64)
        pi_B_A = np.zeros((self.N, self.N), dtype=np.float64)
        
        # Empirical counts for p(A)
        for i in range(num_samples):
            pi_A[inputs_A[i, 0]] += 1
        pi_A /= float(num_samples)
        assert np.isclose(np.sum(pi_A, axis=0), 1.)
        
        # Empirical counts for p(B | A)
        for i in range(num_samples):
            pi_B_A[inputs_A[i, 0], inputs_B[i, 0]] += 1
        pi_B_A /= np.maximum(np.sum(pi_B_A, axis=1, keepdims=True), 1.)
        sum_pi_B_A = np.sum(pi_B_A, axis=1)
        assert np.allclose(sum_pi_B_A[sum_pi_B_A > 0], 1.)

        return self.set_analytical_maximum_likelihood(pi_A, pi_B_A)

class Model1(Model):
    def __init__(self, N):
        super(Model1, self).__init__(N=N)
        self.p_A = Marginal(N)
        self.p_B_A = Conditional(N)
    
    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
        # decomposition p(x_A, x_B) = p(x_A)p(x_B | x_A)
        inputs_A, inputs_B = torch.split(inputs, 1, dim=1)
        inputs_A, inputs_B = inputs_A.squeeze(1), inputs_B.squeeze(1)

        return self.p_A(inputs_A) + self.p_B_A(inputs_A, inputs_B)

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        
        self.p_A.w.data = torch.log(pi_A_th)
        self.p_B_A.w.data = torch.log(pi_B_A_th)

    def init_parameters(self):
        self.p_A.init_parameters()
        self.p_B_A.init_parameters()

class Model2(Model):
    def __init__(self, N):
        super(Model2, self).__init__(N=N)
        self.p_B = Marginal(N)
        self.p_A_B = Conditional(N)

    def forward(self, inputs):
        # Compute the (negative) log-likelihood with the
        # decomposition p(x_A, x_B) = p(x_B)p(x_A | x_B)
        inputs_A, inputs_B = torch.split(inputs, 1, dim=1)
        inputs_A, inputs_B = inputs_A.squeeze(1), inputs_B.squeeze(1)
        
        return self.p_B(inputs_B) + self.p_A_B(inputs_B, inputs_A)

    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        
        log_joint = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_B = torch.logsumexp(log_joint, dim=0)
        
        self.p_B.w.data = log_p_B
        self.p_A_B.w.data = log_joint.t() - log_p_B.unsqueeze(1)

    def init_parameters(self):
        self.p_B.init_parameters()
        self.p_A_B.init_parameters()

class StructuralModel(BivariateStructuralModel):
    def __init__(self, num_categories):
        model_A_B = Model1(num_categories)
        model_B_A = Model2(num_categories)
        super(StructuralModel, self).__init__(model_A_B, model_B_A)
        self.w = nn.Parameter(torch.tensor(0., dtype=torch.float64))
    
    def set_analytical_maximum_likelihood(self, pi_A, pi_B_A):
        self.model_A_B.set_analytical_maximum_likelihood(pi_A, pi_B_A)
        self.model_B_A.set_analytical_maximum_likelihood(pi_A, pi_B_A)
    
    def set_maximum_likelihood(self, inputs):
        self.model_A_B.set_maximum_likelihood(inputs)
        self.model_B_A.set_maximum_likelihood(inputs)

    def reset_modules_parameters(self):
        self.model_A_B.init_parameters()
        self.model_B_A.init_parameters()
