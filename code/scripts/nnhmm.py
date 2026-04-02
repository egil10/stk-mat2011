import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NNHMM(nn.Module):
    def __init__(self, n_states=2, input_dim=1, hidden_dim=32):
        """
        Heavyweight Neural HMM Implementation in PyTorch.
        Supports state-dependent means/variances and learned transitions.
        """
        super(NNHMM, self).__init__()
        self.K = n_states
        self.D = input_dim

        # 1. Transition Matrix (parameterized as unconstrained log-probs)
        # We learn the log-transitions to ensure they sum to 1 via softmax
        self.unnormalized_trans = nn.Parameter(torch.randn(n_states, n_states))

        # 2. Initial State Distribution (pi)
        self.unnormalized_pi = nn.Parameter(torch.randn(n_states))

        # 3. Emission Parameters (State-dependent Gaussian)
        # Means: one per state
        self.means = nn.Parameter(torch.randn(n_states, input_dim) * 0.01)
        # Log-variances: ensure positivity by learning log_std
        self.log_std = nn.Parameter(torch.zeros(n_states, input_dim))

    @property
    def transition_matrix(self):
        return F.softmax(self.unnormalized_trans, dim=-1)

    @property
    def pi(self):
        return F.softmax(self.unnormalized_pi, dim=-1)

    def log_prob_gaussian(self, x):
        """
        Calculates log-likelihood of data x under each state's Gaussian.
        x shape: [batch_size, input_dim]
        returns shape: [batch_size, K]
        """
        std = torch.exp(self.log_std)
        # log P(x|mu, sigma) = -0.5 * log(2pi) - log(sigma) - 0.5 * ((x-mu)/sigma)^2
        log_2pi = np.log(2 * np.pi)
        
        # Reshape for broadcasting
        # x: [B, 1, D], means: [1, K, D], std: [1, K, D]
        x_expanded = x.unsqueeze(1)
        means = self.means.unsqueeze(0)
        std = std.unsqueeze(0)
        
        log_pdf = -0.5 * log_2pi - torch.log(std) - 0.5 * ((x_expanded - means)/std)**2
        return log_pdf.sum(dim=-1) # Sum over input dimensions

    def forward(self, x_seq):
        """
        Forward Algorithm in Log-Space to compute the Total Log-Likelihood.
        x_seq shape: [seq_len, input_dim]
        """
        T = x_seq.shape[0]
        K = self.K
        
        # Get log-params
        log_trans = torch.log_softmax(self.unnormalized_trans, dim=-1)
        log_pi = torch.log_softmax(self.unnormalized_pi, dim=-1)
        log_emissions = self.log_prob_gaussian(x_seq) # [T, K]

        # Initial forward Log-alpha: [K]
        alpha = log_pi + log_emissions[0]

        # Scan through sequence
        for t in range(1, T):
            # log-sum-exp for stable transition update
            # alpha_t(j) = emission_t(j) * sum_i [alpha_t-1(i) * transition(i,j)]
            # In log space: alpha_t(j) = log_emiss_t(j) + LSE_i(alpha_t-1(i) + log_trans(i,j))
            combined = alpha.unsqueeze(1) + log_trans # [K, K]
            alpha = log_emissions[t] + torch.logsumexp(combined, dim=0)

        # Total marginal log-likelihood (the target for our 50k epochs)
        return torch.logsumexp(alpha, dim=0)

    def predict_states(self, x_seq):
        """Viterbi or Posterior decoded states (Placeholder for hw3)"""
        # (Implementation of Viterbi logic for sequence decoding)
        pass
