import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NNHMM(nn.Module):
    def __init__(self, n_states=2, input_dim=1, hidden_dim=32):
        """
        Heavyweight Neural HMM in PyTorch.

        - Transition matrix and initial probs are learned in unconstrained form.
        - Emissions are state-conditional diagonal Gaussians.
        """
        super(NNHMM, self).__init__()
        self.K = n_states
        self.D = input_dim

        # Unconstrained transition logits [K, K]
        self.unnormalized_trans = nn.Parameter(torch.randn(n_states, n_states))

        # Unconstrained initial-state logits [K]
        self.unnormalized_pi = nn.Parameter(torch.randn(n_states))

        # Emission params: mean and log-std per state
        self.means = nn.Parameter(torch.randn(n_states, input_dim) * 0.01)
        self.log_std = nn.Parameter(torch.zeros(n_states, input_dim))

    @property
    def transition_matrix(self):
        return F.softmax(self.unnormalized_trans, dim=-1)

    @property
    def pi(self):
        return F.softmax(self.unnormalized_pi, dim=-1)

    def _safe_log_std(self):
        # Numerical guard against exploding/vanishing std
        return torch.clamp(self.log_std, min=-8.0, max=5.0)

    def log_prob_gaussian(self, x):
        """
        Log p(x_t | z_t = k) for all t,k.

        Args:
            x: [T, D]
        Returns:
            log_emissions: [T, K]
        """
        log_std = self._safe_log_std()
        std = torch.exp(log_std)
        log_2pi = np.log(2.0 * np.pi)

        x_expanded = x.unsqueeze(1)          # [T, 1, D]
        means = self.means.unsqueeze(0)      # [1, K, D]
        std = std.unsqueeze(0)               # [1, K, D]

        log_pdf = -0.5 * log_2pi - torch.log(std) - 0.5 * ((x_expanded - means) / std) ** 2
        return log_pdf.sum(dim=-1)           # [T, K]

    def forward_log_alpha(self, x_seq):
        """
        Forward algorithm in log-space.

        Args:
            x_seq: [T, D]
        Returns:
            log_alpha: [T, K]
        """
        T = x_seq.shape[0]
        K = self.K

        log_trans = torch.log_softmax(self.unnormalized_trans, dim=-1)   # [K, K]
        log_pi = torch.log_softmax(self.unnormalized_pi, dim=-1)         # [K]
        log_emissions = self.log_prob_gaussian(x_seq)                     # [T, K]

        log_alpha = torch.empty((T, K), device=x_seq.device, dtype=x_seq.dtype)
        log_alpha[0] = log_pi + log_emissions[0]

        for t in range(1, T):
            # from i -> j: alpha_{t-1}(i) + log_trans(i,j)
            combined = log_alpha[t - 1].unsqueeze(1) + log_trans          # [K, K]
            log_alpha[t] = log_emissions[t] + torch.logsumexp(combined, dim=0)

        return log_alpha

    def log_likelihood(self, x_seq):
        """
        Total sequence log-likelihood log p(x_1:T).
        """
        log_alpha = self.forward_log_alpha(x_seq)
        return torch.logsumexp(log_alpha[-1], dim=0)

    def forward(self, x_seq):
        """
        Keeps training API unchanged: model(x_seq) -> scalar log-likelihood.
        """
        return self.log_likelihood(x_seq)

    def viterbi(self, x_seq):
        """
        Viterbi decoding (most likely latent path).

        Args:
            x_seq: [T, D]
        Returns:
            z: [T] int64 tensor with decoded states
        """
        T = x_seq.shape[0]
        K = self.K

        log_trans = torch.log_softmax(self.unnormalized_trans, dim=-1)    # [K, K]
        log_pi = torch.log_softmax(self.unnormalized_pi, dim=-1)          # [K]
        log_emissions = self.log_prob_gaussian(x_seq)                      # [T, K]

        delta = torch.empty((T, K), device=x_seq.device, dtype=x_seq.dtype)
        psi = torch.zeros((T, K), device=x_seq.device, dtype=torch.long)

        delta[0] = log_pi + log_emissions[0]

        for t in range(1, T):
            scores = delta[t - 1].unsqueeze(1) + log_trans                 # [K, K]
            best_prev_scores, best_prev_idx = torch.max(scores, dim=0)     # per current state j
            delta[t] = log_emissions[t] + best_prev_scores
            psi[t] = best_prev_idx

        z = torch.zeros(T, device=x_seq.device, dtype=torch.long)
        z[-1] = torch.argmax(delta[-1])

        for t in range(T - 2, -1, -1):
            z[t] = psi[t + 1, z[t + 1]]

        return z

    def backward_log_beta(self, x_seq):
        """
        Backward algorithm in log-space.

        Args:
            x_seq: [T, D]
        Returns:
            log_beta: [T, K]
        """
        T = x_seq.shape[0]
        K = self.K

        log_trans = torch.log_softmax(self.unnormalized_trans, dim=-1)    # [K, K]
        log_emissions = self.log_prob_gaussian(x_seq)                      # [T, K]

        log_beta = torch.zeros((T, K), device=x_seq.device, dtype=x_seq.dtype)
        # log_beta[T-1] = 0 already (log(1))

        for t in range(T - 2, -1, -1):
            # beta_t(i) = sum_j trans(i,j) * emiss_{t+1}(j) * beta_{t+1}(j)
            term = log_trans + (log_emissions[t + 1] + log_beta[t + 1]).unsqueeze(0)  # [K, K]
            log_beta[t] = torch.logsumexp(term, dim=1)

        return log_beta

    def posterior_probs(self, x_seq):
        """
        Smoothed posterior p(z_t=k | x_1:T).

        Args:
            x_seq: [T, D]
        Returns:
            gamma: [T, K], rows sum to 1
        """
        log_alpha = self.forward_log_alpha(x_seq)
        log_beta = self.backward_log_beta(x_seq)

        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
        return torch.exp(log_gamma)

    def predict_states(self, x_seq, method="viterbi"):
        """
        Decode latent states.

        Args:
            x_seq: [T, D]
            method: "viterbi" or "posterior"
        Returns:
            states: [T] int64 tensor
        """
        method = method.lower()
        if method == "viterbi":
            return self.viterbi(x_seq)
        if method == "posterior":
            gamma = self.posterior_probs(x_seq)
            return torch.argmax(gamma, dim=1)
        raise ValueError("method must be 'viterbi' or 'posterior'")