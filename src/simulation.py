"""
SDE simulators for the Beta-sector SDE hierarchy.

All simulators use Euler-Maruyama with explicit support enforcement.
For bounded processes, we apply projection/reflection to keep the state
in the admissible region.

Simulation protocol (implementation choices, not paper-specified):
  - dt: step size (user-specified, typically 0.001 to 0.01)
  - T_burn: burn-in period (user-specified, typically 50*mean-reversion-time)
  - T_sample: sampling period after burn-in (user-specified)
  - n_samples: number of samples drawn from the post-burn-in trajectory
  - Boundary enforcement: projection to [eps, upper - eps]
  - Random seed: user-specified for reproducibility

# Custom — Context7 found no library equivalent for bounded-support SDE simulation
# with these specific diffusion terms.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from src.parameters import BetaSectorParams, GeneralizedParams

_EPS_PROJ = 1e-8  # projection epsilon for bounded supports


def _euler_step(x: float, gamma: float, theta: float,
                mu_fn, sigma_fn, dt: float, dW: float) -> float:
    """Single Euler-Maruyama step."""
    mu = mu_fn(x)
    sigma = sigma_fn(x)
    return x + mu * dt + sigma * dW


def _project_bounded(x: float, lo: float, hi: float) -> float:
    """Project x into (lo, hi) with epsilon margin."""
    return float(np.clip(x, lo + _EPS_PROJ, hi - _EPS_PROJ))


def _project_positive(x: float) -> float:
    """Project x to be strictly positive."""
    return max(x, _EPS_PROJ)


def _run_simulation(
    x0: float,
    mu_fn,
    sigma_fn,
    project_fn,
    T_total: float,
    T_burn: float,
    dt: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Run Euler-Maruyama and return n_samples from the post-burn-in trajectory.

    Parameters
    ----------
    x0 : initial state
    mu_fn : drift function mu(x)
    sigma_fn : diffusion function sigma(x)
    project_fn : boundary enforcement function project(x) -> x_safe
    T_total : total simulation time (burn-in + sampling)
    T_burn : burn-in time (discarded)
    dt : step size
    n_samples : number of samples to return
    rng : numpy random generator
    """
    n_steps_burn = int(T_burn / dt)
    n_steps_sample = int((T_total - T_burn) / dt)
    n_steps_sample = max(n_steps_sample, n_samples)

    # Sample every (n_steps_sample // n_samples) steps
    stride = max(1, n_steps_sample // n_samples)

    x = x0
    sqrt_dt = np.sqrt(dt)

    # Burn-in
    for _ in range(n_steps_burn):
        dW = rng.normal(0, sqrt_dt)
        x_new = x + mu_fn(x) * dt + sigma_fn(x) * dW
        x = project_fn(x_new)

    # Sampling
    samples = []
    for i in range(n_steps_sample):
        dW = rng.normal(0, sqrt_dt)
        x_new = x + mu_fn(x) * dt + sigma_fn(x) * dW
        x = project_fn(x_new)
        if i % stride == 0 and len(samples) < n_samples:
            samples.append(x)

    return np.array(samples[:n_samples])


# ---------------------------------------------------------------------------
# Original Beta-sector SDE simulators
# ---------------------------------------------------------------------------

def simulate_mB_sde(
    p: BetaSectorParams,
    n_samples: int = 2000,
    T: float = 200.0,
    T_burn: float = 50.0,
    dt: float = 0.005,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate the master modified-Beta SDE.

    dx = -gamma*(x-theta)*dt + sqrt(kappa^2*x*(1-x/beta_1)*(1+x/beta_2))*dW

    Support: (0, beta_1).

    Parameters
    ----------
    p : BetaSectorParams with kappa, kappa_1, kappa_2 all > 0
    n_samples : number of post-burn-in samples
    T : total simulation time
    T_burn : burn-in time
    dt : step size
    seed : random seed
    """
    rng = np.random.default_rng(seed)
    gamma, theta = p.gamma, p.theta
    kappa, b1, b2 = p.kappa, p.beta_1, p.beta_2

    def mu(x: float) -> float:
        return -gamma * (x - theta)

    def sigma(x: float) -> float:
        val = kappa**2 * x * max(1 - x / b1, 0) * max(1 + x / b2, 0)
        return np.sqrt(max(val, 0.0))

    project = lambda x: _project_bounded(x, 0, b1)
    x0 = theta
    return _run_simulation(x0, mu, sigma, project, T, T_burn, dt, n_samples, rng)


def simulate_IGa1_sde(
    p: BetaSectorParams,
    n_samples: int = 2000,
    T: float = 200.0,
    T_burn: float = 50.0,
    dt: float = 0.005,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate the IGa1 SDE (kappa = 0).

    dx = -gamma*(x-theta)*dt + kappa_2*x*sqrt(1-x/beta_1)*dW

    Support: (0, beta_1).

    Parameters
    ----------
    p : BetaSectorParams with kappa_1 > 0, kappa_2 > 0, kappa = 0
    """
    rng = np.random.default_rng(seed)
    gamma, theta = p.gamma, p.theta
    kappa_2, b1 = p.kappa_2, p.beta_1

    def mu(x: float) -> float:
        return -gamma * (x - theta)

    def sigma(x: float) -> float:
        val = kappa_2**2 * x**2 * max(1 - x / b1, 0)
        return np.sqrt(max(val, 0.0))

    project = lambda x: _project_bounded(x, 0, b1)
    x0 = theta
    return _run_simulation(x0, mu, sigma, project, T, T_burn, dt, n_samples, rng)


def simulate_BP_B2_sde(
    p: BetaSectorParams,
    n_samples: int = 2000,
    T: float = 200.0,
    T_burn: float = 50.0,
    dt: float = 0.005,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate the BP/B2 SDE (kappa_1 = 0).

    dx = -gamma*(x-theta)*dt + sqrt(kappa^2*x*(1+x/beta_2))*dW

    Support: (0, infinity). Use soft truncation at large values.

    Parameters
    ----------
    p : BetaSectorParams with kappa > 0, kappa_2 > 0, kappa_1 = 0
    """
    rng = np.random.default_rng(seed)
    gamma, theta = p.gamma, p.theta
    kappa, b2 = p.kappa, p.beta_2

    def mu(x: float) -> float:
        return -gamma * (x - theta)

    def sigma(x: float) -> float:
        val = kappa**2 * x * max(1 + x / b2, 0)
        return np.sqrt(max(val, 0.0))

    # Upper truncation at 50*theta for numerical stability
    upper_trunc = max(50 * theta, 20 * b2)
    project = lambda x: _project_bounded(x, 0, upper_trunc)
    x0 = theta
    return _run_simulation(x0, mu, sigma, project, T, T_burn, dt, n_samples, rng)


def simulate_B1_sde(
    p: BetaSectorParams,
    n_samples: int = 2000,
    T: float = 200.0,
    T_burn: float = 50.0,
    dt: float = 0.005,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate the B1 SDE (kappa_2 = 0).

    dx = -gamma*(x-theta)*dt + sqrt(kappa^2*x*(1-x/beta_1))*dW

    Support: (0, beta_1).

    Parameters
    ----------
    p : BetaSectorParams with kappa > 0, kappa_1 > 0, kappa_2 = 0
    """
    rng = np.random.default_rng(seed)
    gamma, theta = p.gamma, p.theta
    kappa, b1 = p.kappa, p.beta_1

    def mu(x: float) -> float:
        return -gamma * (x - theta)

    def sigma(x: float) -> float:
        val = kappa**2 * x * max(1 - x / b1, 0)
        return np.sqrt(max(val, 0.0))

    project = lambda x: _project_bounded(x, 0, b1)
    x0 = theta
    return _run_simulation(x0, mu, sigma, project, T, T_burn, dt, n_samples, rng)


def simulate_IGa_sde(
    p: BetaSectorParams,
    n_samples: int = 2000,
    T: float = 200.0,
    T_burn: float = 50.0,
    dt: float = 0.005,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate the IGa SDE (kappa = 0, kappa_1 = 0).

    dx = -gamma*(x-theta)*dt + kappa_2*x*dW

    Support: (0, infinity).

    Parameters
    ----------
    p : BetaSectorParams with kappa_2 > 0  (kappa = 0, kappa_1 = 0)
    """
    rng = np.random.default_rng(seed)
    gamma, theta, kappa_2 = p.gamma, p.theta, p.kappa_2

    def mu(x: float) -> float:
        return -gamma * (x - theta)

    def sigma(x: float) -> float:
        return kappa_2 * max(x, 0.0)

    upper_trunc = 50 * theta
    project = lambda x: _project_bounded(x, 0, upper_trunc)
    x0 = theta
    return _run_simulation(x0, mu, sigma, project, T, T_burn, dt, n_samples, rng)


def simulate_Ga_sde(
    p: BetaSectorParams,
    n_samples: int = 2000,
    T: float = 200.0,
    T_burn: float = 50.0,
    dt: float = 0.005,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate the Gamma (Ga) SDE (kappa_1 = 0, kappa_2 = 0).

    dx = -gamma*(x-theta)*dt + kappa*sqrt(x)*dW

    Support: [0, infinity).

    Parameters
    ----------
    p : BetaSectorParams with kappa > 0  (kappa_1 = 0, kappa_2 = 0)
    """
    rng = np.random.default_rng(seed)
    gamma, theta, kappa = p.gamma, p.theta, p.kappa

    def mu(x: float) -> float:
        return -gamma * (x - theta)

    def sigma(x: float) -> float:
        return kappa * np.sqrt(max(x, 0.0))

    upper_trunc = 50 * theta
    project = lambda x: _project_bounded(x, 0, upper_trunc)
    x0 = theta
    return _run_simulation(x0, mu, sigma, project, T, T_burn, dt, n_samples, rng)


# ---------------------------------------------------------------------------
# Generalized SDE simulator (exp_2 dynamical route)
# ---------------------------------------------------------------------------

def simulate_generalized_sde(
    gp: GeneralizedParams,
    branch: str = "mGB",
    n_samples: int = 2000,
    T: float = 200.0,
    T_burn: float = 50.0,
    dt: float = 0.005,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate the generalized dynamical-route SDE.

    Master generalized SDE:
      dx = -gamma*(x - theta*x^(1-alpha))*dt
           + sqrt(kappa^2*x^(2-alpha)*(1-x^alpha/beta_1)*(1+x^alpha/beta_2))*dW

    Branch variants:
      'mGB'  : full SDE (kappa, kappa_1, kappa_2 all > 0)
      'GBP'  : kappa_1 = 0 (1-x^alpha/beta_1) -> 1
      'GB1'  : kappa_2 = 0 (1+x^alpha/beta_2) -> 1
      'GGa'  : kappa_1 = 0, kappa_2 = 0 -> CIR-type
      'GIGa' : kappa = 0, kappa_1 = 0 -> dx = -gamma*(x-theta)*dt + kappa_2*x*dW

    Support enforcement matches branch.

    Parameters
    ----------
    gp : GeneralizedParams
    branch : one of 'mGB', 'GBP', 'GB1', 'GGa', 'GIGa', 'GIGa1'
    n_samples : number of post-burn-in samples
    T : total simulation time
    T_burn : burn-in time
    dt : step size
    seed : random seed
    """
    rng = np.random.default_rng(seed)
    gamma, theta = gp.gamma, gp.theta
    kappa, kappa_2 = gp.kappa, gp.kappa_2
    alpha = gp.alpha
    b1 = gp.beta_1
    b2 = gp.beta_2

    if branch == "mGB":
        def mu(x: float) -> float:
            return -gamma * (x - theta * x ** (1 - alpha))

        def sigma(x: float) -> float:
            xa = max(x, _EPS_PROJ) ** alpha
            v = kappa**2 * max(x, _EPS_PROJ) ** (2 - alpha) * max(1 - xa / b1, 0) * max(1 + xa / b2, 0)
            return np.sqrt(max(v, 0))

        upper = b1 ** (1 / alpha)
        project = lambda x: _project_bounded(x, 0, upper)

    elif branch == "GBP":
        def mu(x: float) -> float:
            return -gamma * (x - theta * x ** (1 - alpha))

        def sigma(x: float) -> float:
            xa = max(x, _EPS_PROJ) ** alpha
            v = kappa**2 * max(x, _EPS_PROJ) ** (2 - alpha) * max(1 + xa / b2, 0)
            return np.sqrt(max(v, 0))

        upper_trunc = 50 * theta
        project = lambda x: _project_bounded(x, 0, upper_trunc)

    elif branch == "GB1":
        def mu(x: float) -> float:
            return -gamma * (x - theta * x ** (1 - alpha))

        def sigma(x: float) -> float:
            xa = max(x, _EPS_PROJ) ** alpha
            v = kappa**2 * max(x, _EPS_PROJ) ** (2 - alpha) * max(1 - xa / b1, 0)
            return np.sqrt(max(v, 0))

        upper = b1 ** (1 / alpha)
        project = lambda x: _project_bounded(x, 0, upper)

    elif branch == "GGa":
        def mu(x: float) -> float:
            return -gamma * (x - theta * x ** (1 - alpha))

        def sigma(x: float) -> float:
            v = kappa**2 * max(x, _EPS_PROJ) ** (2 - alpha)
            return np.sqrt(max(v, 0))

        upper_trunc = 50 * theta
        project = lambda x: _project_bounded(x, 0, upper_trunc)

    elif branch == "GIGa":
        # kappa = 0, kappa_1 = 0: dx = -gamma*(x-theta)*dt + kappa_2*x*dW
        def mu(x: float) -> float:
            return -gamma * (x - theta)

        def sigma(x: float) -> float:
            return kappa_2 * max(x, _EPS_PROJ)

        upper_trunc = 50 * theta
        project = lambda x: _project_bounded(x, 0, upper_trunc)

    elif branch == "GIGa1":
        # kappa = 0: dx = -gamma*(x-theta)*dt + kappa_2*x*sqrt(1-x^alpha/beta_1)*dW
        kappa_1 = gp.kappa_1

        def mu(x: float) -> float:
            return -gamma * (x - theta)

        def sigma(x: float) -> float:
            xa = max(x, _EPS_PROJ) ** alpha
            v = kappa_2**2 * max(x, _EPS_PROJ)**2 * max(1 - xa / b1, 0)
            return np.sqrt(max(v, 0))

        upper = b1 ** (1 / alpha)
        project = lambda x: _project_bounded(x, 0, upper)

    else:
        raise ValueError(f"Unknown branch: {branch}")

    x0 = float(theta)
    return _run_simulation(x0, mu, sigma, project, T, T_burn, dt, n_samples, rng)
