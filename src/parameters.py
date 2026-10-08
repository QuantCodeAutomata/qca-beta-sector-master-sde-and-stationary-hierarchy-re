"""
Parameter validation and derived quantity computation for Beta-sector SDE hierarchy.

All parameter mappings follow the paper exactly:
  beta_1 = 1/kappa_1^2  (when kappa_1 > 0)
  beta_2 = kappa^2/kappa_2^2  (when kappa_2 > 0 and kappa > 0)
  p = 2*gamma*theta/kappa^2  (for branches with kappa > 0)
  q = 2*gamma*(1-kappa_1^2*theta)/(kappa^2*kappa_1^2+kappa_2^2)  (full mB branch)
  beta = 2*gamma*theta/kappa_2^2  (IGa1/IGa branches)
  lambda = kappa^2/(2*gamma)  (Ga branch: theta/p)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BetaSectorParams:
    """
    Primitive parameters for the master mean-reverting SDE:
      dx = -gamma*(x-theta)*dt + sqrt((1-kappa_1^2*x)*(kappa^2*x + kappa_2^2*x^2))*dW

    Equivalently:
      dx = -gamma*(x-theta)*dt + sqrt(kappa^2*x*(1-x/beta_1)*(1+x/beta_2))*dW

    Derived quantities are computed automatically on construction.
    """
    gamma: float
    theta: float
    kappa: float = 0.0
    kappa_1: float = 0.0
    kappa_2: float = 0.0

    # Derived (computed in __post_init__)
    beta_1: float = field(init=False)
    beta_2: float = field(init=False)
    p: float = field(init=False)
    q: float = field(init=False)
    beta: float = field(init=False)   # for IGa1/IGa branches: 2*gamma*theta/kappa_2^2
    lam: float = field(init=False)    # lambda for Ga branch: kappa^2/(2*gamma)

    def __post_init__(self) -> None:
        assert self.gamma > 0, "gamma must be positive"
        assert self.theta > 0, "theta must be positive"

        # beta_1 = 1/kappa_1^2; inf when kappa_1 = 0 (unbounded upper support)
        self.beta_1 = 1.0 / self.kappa_1**2 if self.kappa_1 > 0 else np.inf

        # beta_2 = kappa^2/kappa_2^2; inf when either kappa or kappa_2 is 0
        self.beta_2 = (self.kappa**2 / self.kappa_2**2
                       if (self.kappa > 0 and self.kappa_2 > 0) else np.inf)

        # p = 2*gamma*theta/kappa^2  (branches with kappa > 0)
        self.p = (2 * self.gamma * self.theta / self.kappa**2
                  if self.kappa > 0 else np.nan)

        # q formula depends on branch
        if self.kappa > 0 and self.kappa_1 > 0:
            # Full mB or B1 branch
            denom = self.kappa**2 * self.kappa_1**2 + self.kappa_2**2
            self.q = 2 * self.gamma * (1 - self.kappa_1**2 * self.theta) / denom
        elif self.kappa > 0 and self.kappa_2 > 0 and self.kappa_1 == 0:
            # BP/B2 branch (kappa_1 = 0)
            # From the paper: q = 2*gamma/kappa_2^2 - p
            self.q = 2 * self.gamma / self.kappa_2**2 - self.p
        else:
            self.q = np.nan

        # beta = 2*gamma*theta/kappa_2^2  (IGa1 and IGa branches)
        self.beta = (2 * self.gamma * self.theta / self.kappa_2**2
                     if self.kappa_2 > 0 else np.nan)

        # lambda = kappa^2/(2*gamma) = theta/p  (Ga branch)
        self.lam = self.kappa**2 / (2 * self.gamma) if self.kappa > 0 else np.nan

    def validate_mB(self) -> None:
        """Validate admissibility for the full modified-Beta (mB) branch."""
        assert self.kappa > 0, "kappa must be positive for mB"
        assert self.kappa_1 > 0, "kappa_1 must be positive for mB"
        assert self.kappa_2 > 0, "kappa_2 must be positive for mB"
        assert self.p > 0, f"p={self.p} must be positive"
        assert self.q > 0, f"q={self.q} must be positive"
        assert self.theta < self.beta_1, f"theta={self.theta} must be < beta_1={self.beta_1}"

    def validate_IGa1(self) -> None:
        """Validate admissibility for IGa1 branch (kappa=0)."""
        assert self.kappa_1 > 0, "kappa_1 must be positive for IGa1"
        assert self.kappa_2 > 0, "kappa_2 must be positive for IGa1"
        assert self.theta < self.beta_1, f"theta={self.theta} must be < beta_1={self.beta_1}"

    def validate_BP_B2(self) -> None:
        """Validate admissibility for BP/B2 branch (kappa_1=0)."""
        assert self.kappa > 0, "kappa must be positive for BP/B2"
        assert self.kappa_2 > 0, "kappa_2 must be positive for BP/B2"
        assert self.p > 0, f"p={self.p} must be positive"
        assert self.q > 0, f"q={self.q} must be positive"

    def validate_B1(self) -> None:
        """Validate admissibility for B1 branch (kappa_2=0)."""
        assert self.kappa > 0, "kappa must be positive for B1"
        assert self.kappa_1 > 0, "kappa_1 must be positive for B1"
        assert self.p > 0, f"p={self.p} must be positive"
        assert self.q > 0, f"q={self.q} must be positive"
        assert self.theta < self.beta_1, f"theta={self.theta} must be < beta_1={self.beta_1}"

    def validate_IGa(self) -> None:
        """Validate admissibility for IGa terminal branch."""
        assert self.kappa_2 > 0, "kappa_2 must be positive for IGa"

    def validate_Ga(self) -> None:
        """Validate admissibility for Ga terminal branch."""
        assert self.kappa > 0, "kappa must be positive for Ga"
        assert self.p > 0, f"p={self.p} must be positive"


@dataclass
class GeneralizedParams:
    """
    Parameters for the generalized hierarchy (exp_2 dynamical route, exp_3 distributional route).
    Extends BetaSectorParams with alpha > 0.

    Additional derived auxiliary quantities for the dynamical route:
      delta = 1/alpha
      u = (p + alpha - 1) / alpha   (integrability: u > 0 required)
      v = q / alpha
      q1 = p + q
      w = 1 + q1/alpha
      lam2 = theta/p  (lambda_2 for GGa branch)
    """
    gamma: float
    theta: float
    kappa: float = 0.0
    kappa_1: float = 0.0
    kappa_2: float = 0.0
    alpha: float = 1.0

    beta_1: float = field(init=False)
    beta_2: float = field(init=False)
    p: float = field(init=False)
    q: float = field(init=False)
    beta: float = field(init=False)
    lam2: float = field(init=False)
    delta: float = field(init=False)
    u: float = field(init=False)
    v: float = field(init=False)
    q1: float = field(init=False)
    w: float = field(init=False)

    def __post_init__(self) -> None:
        assert self.gamma > 0, "gamma must be positive"
        assert self.theta > 0, "theta must be positive"
        assert self.alpha > 0, "alpha must be positive"

        self.beta_1 = 1.0 / self.kappa_1**2 if self.kappa_1 > 0 else np.inf
        self.beta_2 = (self.kappa**2 / self.kappa_2**2
                       if (self.kappa > 0 and self.kappa_2 > 0) else np.inf)
        self.p = (2 * self.gamma * self.theta / self.kappa**2
                  if self.kappa > 0 else np.nan)

        if self.kappa > 0 and self.kappa_1 > 0:
            denom = self.kappa**2 * self.kappa_1**2 + self.kappa_2**2
            self.q = 2 * self.gamma * (1 - self.kappa_1**2 * self.theta) / denom
        elif self.kappa > 0 and self.kappa_2 > 0 and self.kappa_1 == 0:
            self.q = 2 * self.gamma / self.kappa_2**2 - self.p
        else:
            self.q = np.nan

        self.beta = (2 * self.gamma * self.theta / self.kappa_2**2
                     if self.kappa_2 > 0 else np.nan)

        self.lam2 = (self.theta / self.p
                     if (self.kappa > 0 and not np.isnan(self.p) and self.p > 0)
                     else np.nan)

        # Auxiliary parameters for dynamical route
        self.delta = 1.0 / self.alpha
        self.u = ((self.p + self.alpha - 1.0) / self.alpha
                  if not np.isnan(self.p) else np.nan)
        self.v = self.q / self.alpha if not np.isnan(self.q) else np.nan
        self.q1 = (self.p + self.q
                   if (not np.isnan(self.p) and not np.isnan(self.q))
                   else np.nan)
        self.w = (1.0 + self.q1 / self.alpha
                  if not np.isnan(self.q1) else np.nan)

    def as_beta_sector(self) -> BetaSectorParams:
        """Return the alpha=1 projection as a BetaSectorParams object."""
        return BetaSectorParams(
            gamma=self.gamma,
            theta=self.theta,
            kappa=self.kappa,
            kappa_1=self.kappa_1,
            kappa_2=self.kappa_2,
        )

    def check_integrability(self) -> bool:
        """Check u > 0 condition required for GmB near-zero integrability."""
        return not np.isnan(self.u) and self.u > 0
