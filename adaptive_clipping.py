"""
Core ACUN adaptive clipping logic.

Implements:
  - Differentially private median estimation of gradient norms
    (exponential mechanism on a geometric histogram)
  - Closed-form optimal (C_t*, σ_t*) from the local excess-risk proxy
    derived in the paper (Eq. 5).

Author: Nitish Kumar Roy <Nitishkumarrpy526@gmail.com>
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


def private_median_estimate(
    norms: torch.Tensor,
    eps_med: float,
    n_bins: int = 64,
    min_norm: float = 1e-3,
    max_norm: float = 50.0,
) -> float:
    """
    Differentially private estimate of the median of per-example gradient norms
    using the exponential mechanism on a geometric histogram.

    Args:
        norms: 1-D tensor of per-example L2 gradient norms.
        eps_med: Privacy budget allocated to the median estimator.
        n_bins: Number of bins in the geometric grid.
        min_norm / max_norm: Range of the geometric grid.

    Returns:
        Private median estimate (float).
    """
    if norms.numel() == 0:
        return 1.0

    norms = norms.detach().cpu().float().clamp(min=min_norm, max=max_norm)
    # Geometric grid
    log_min, log_max = math.log(min_norm), math.log(max_norm)
    grid = torch.exp(torch.linspace(log_min, log_max, n_bins))

    # Histogram counts
    # Assign each norm to nearest grid point
    idx = torch.bucketize(norms, grid) - 1
    idx = idx.clamp(0, n_bins - 1)
    counts = torch.bincount(idx, minlength=n_bins).float()

    # Exponential mechanism: utility = count, sensitivity = 1
    # Probability ∝ exp(ε * utility / (2 * Δu)) with Δu = 1
    utilities = counts
    logits = (eps_med / 2.0) * utilities
    # Numerical stability
    logits = logits - logits.max()
    probs = torch.softmax(logits, dim=0)

    # Sample bin
    sampled_idx = torch.multinomial(probs, 1).item()
    return float(grid[sampled_idx].item())


class AdaptiveClipper:
    """
    Maintains the adaptive clipping threshold and noise multiplier for ACUN.

    At every step the caller supplies:
      - current per-example gradient norms
      - remaining Rényi privacy budget ρ_t
      - current learning-rate estimate / Lipschitz estimate

    The object returns the recommended (C_t, σ_t).
    """

    def __init__(
        self,
        median_budget_fraction: float = 0.05,
        n_bins: int = 64,
        initial_median: float = 1.0,
        lipschitz_ema: float = 1.0,
        lipschitz_momentum: float = 0.9,
    ):
        self.median_budget_fraction = median_budget_fraction
        self.n_bins = n_bins
        self.current_median = initial_median
        self.lipschitz_ema = lipschitz_ema
        self.lipschitz_momentum = lipschitz_momentum

        # Book-keeping
        self.history_C = []
        self.history_sigma = []
        self.history_median = []

    def update_lipschitz(self, grad_norm_mean: float) -> None:
        """Cheap moving-average estimate of local Lipschitz constant."""
        self.lipschitz_ema = (
            self.lipschitz_momentum * self.lipschitz_ema
            + (1.0 - self.lipschitz_momentum) * max(grad_norm_mean, 1e-6)
        )

    def compute_C_and_sigma(
        self,
        norms: torch.Tensor,
        remaining_rho: float,
        batch_size: int,
        lr: float,
        dim: int,
        total_epsilon: float,
    ) -> Tuple[float, float]:
        """
        Compute the optimal clipping threshold and noise multiplier
        following Eq. (5) of the paper.

        C*_t = m̂_t * (1 + sqrt(2 ρ_t B² / (η L d)))
        σ*_t = sqrt(2 ρ_t) / (C*_t / B)
        """
        # Privacy budget for the median estimator
        eps_med = self.median_budget_fraction * total_epsilon
        # Private median
        self.current_median = private_median_estimate(
            norms, eps_med=eps_med, n_bins=self.n_bins
        )
        self.history_median.append(self.current_median)

        # Guard against zero / negative remaining budget
        rho = max(remaining_rho, 1e-8)
        L = max(self.lipschitz_ema, 1e-6)
        eta = max(lr, 1e-8)
        B = max(batch_size, 1)
        d = max(dim, 1)

        # Closed-form solution
        factor = math.sqrt((2.0 * rho * (B ** 2)) / (eta * L * d))
        C = self.current_median * (1.0 + factor)
        # Numerical safety
        C = max(C, 1e-3)

        sigma = math.sqrt(2.0 * rho) / (C / B)
        sigma = max(sigma, 1e-4)

        self.history_C.append(C)
        self.history_sigma.append(sigma)
        return C, sigma

    def state_dict(self) -> dict:
        return {
            "current_median": self.current_median,
            "lipschitz_ema": self.lipschitz_ema,
            "history_C": self.history_C,
            "history_sigma": self.history_sigma,
            "history_median": self.history_median,
        }
