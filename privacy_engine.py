"""
ACUNPrivacyEngine – thin, Opacus-compatible wrapper.

This class makes ACUN a near drop-in replacement for
opacus.PrivacyEngine while adding the adaptive clipping logic.

Author: Nitish Kumar Roy <Nitishkumarrpy526@gmail.com>
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

try:
    from opacus import PrivacyEngine
    from opacus.accountants import RDPAccountant
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    PrivacyEngine = object  # type: ignore

from .adaptive_clipping import AdaptiveClipper


class ACUNPrivacyEngine:
    """
    High-level interface used by the experiment scripts.

    Typical usage:

        engine = ACUNPrivacyEngine(
            module=model,
            batch_size=32,
            sample_size=len(train_ds),
            epochs=5,
            target_epsilon=3.0,
            target_delta=1e-5,
            median_budget_fraction=0.05,
        )
        model, optimizer, train_loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            max_grad_norm=1.0,          # initial value; will be adapted
        )
    """

    def __init__(
        self,
        module: nn.Module,
        batch_size: int,
        sample_size: int,
        epochs: int,
        target_epsilon: float = 3.0,
        target_delta: float = 1e-5,
        median_budget_fraction: float = 0.05,
        max_grad_norm: float = 1.0,
        secure_mode: bool = False,
    ):
        if not OPACUS_AVAILABLE:
            raise ImportError(
                "Opacus is required. Install with: pip install opacus>=1.4.1"
            )

        self.module = module
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.epochs = epochs
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.median_budget_fraction = median_budget_fraction
        self.max_grad_norm = max_grad_norm
        self.secure_mode = secure_mode

        # Internal Opacus engine (we still rely on its accountant & Poisson sampler)
        self._opacus_engine = PrivacyEngine(secure_mode=secure_mode)
        self.clipper = AdaptiveClipper(
            median_budget_fraction=median_budget_fraction
        )

        self._steps = 0
        self._accountant: Optional[RDPAccountant] = None

    def make_private(
        self,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        max_grad_norm: Optional[float] = None,
        **kwargs,
    ):
        """
        Attach ACUN + Opacus privacy to the training loop.
        Returns the privatised (module, optimizer, data_loader).
        """
        if max_grad_norm is None:
            max_grad_norm = self.max_grad_norm

        # Make the model compatible with Opacus
        module = ModuleValidator.fix(module)

        private_module, private_optimizer, private_loader = (
            self._opacus_engine.make_private_with_epsilon(
                module=module,
                optimizer=optimizer,
                data_loader=data_loader,
                epochs=self.epochs,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                max_grad_norm=max_grad_norm,
                **kwargs,
            )
        )

        # Keep a reference so we can adapt the clipping norm on the fly
        self._private_optimizer = private_optimizer
        self._accountant = self._opacus_engine.accountant
        return private_module, private_optimizer, private_loader

    def step_adapt(self, norms: torch.Tensor, lr: float, dim: int) -> None:
        """
        Call this *after* the normal backward pass but *before* optimizer.step()
        if you want to override the clipping threshold for the current step.

        In the reference experiments we let Opacus handle the actual clipping
        with a reasonably large fixed max_grad_norm and use the adaptive
        values mainly for logging / analysis.  A production integration can
        replace the Opacus clipper entirely by the values returned here.
        """
        if self._accountant is None:
            return

        # Rough remaining Rényi budget (simplified)
        # In a full implementation one would query the accountant more precisely.
        spent = self.get_epsilon()
        remaining_frac = max(0.0, 1.0 - spent / max(self.target_epsilon, 1e-8))
        remaining_rho = remaining_frac * 1.0  # placeholder scale

        C, sigma = self.clipper.compute_C_and_sigma(
            norms=norms,
            remaining_rho=remaining_rho,
            batch_size=self.batch_size,
            lr=lr,
            dim=dim,
            total_epsilon=self.target_epsilon,
        )
        self._steps += 1
        # Optionally update the optimizer's max_grad_norm
        if hasattr(self._private_optimizer, "max_grad_norm"):
            self._private_optimizer.max_grad_norm = C

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        if delta is None:
            delta = self.target_delta
        if self._accountant is None:
            return 0.0
        return self._accountant.get_epsilon(delta)

    def get_clipper_state(self) -> dict:
        return self.clipper.state_dict()
