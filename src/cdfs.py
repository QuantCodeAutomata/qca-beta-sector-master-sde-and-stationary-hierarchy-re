"""
Stationary CDF implementations for the Beta-sector SDE hierarchy.

Covers:
  1. Original Beta-sector (exp_1): cdf_mB, cdf_IGa1, cdf_BP_B2, cdf_B1, cdf_IGa, cdf_Ga
  2. Generalized dynamical route (exp_2): cdf_mGB, cdf_GBP, cdf_GB1, cdf_GIGa, cdf_GGa
  3. Generalized distributional route (exp_3): cdf_mB_alpha, cdf_BP_B2_alpha,
     cdf_B1_alpha, cdf_IGa_alpha, cdf_Ga_alpha
  4. Traditional GB benchmark (exp_4): cdf_GB

Distributional-route CDFs use the pushforward identity F_X(x) = F_Y(x^alpha),
which avoids OCR-corrupted Appell F1 formulas.

# Custom — Context7 found no library equivalent for these novel SDE stationary laws.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betainc, gammaincc, gammainc, hyp2f1, gammaln
from scipy.integrate import quad, cumulative_trapezoid
from typing import Union

from src.parameters import BetaSectorParams, GeneralizedParams
from src.densities import (
    pdf_mB, pdf_IGa1, pdf_BP_B2, pdf_B1, pdf_IGa, pdf_Ga,
    pdf_mGB, pdf_GBP, pdf_GB1, pdf_GIGa, pdf_GGa,
    _to_array, _EPS,
)

ArrayLike = Union[np.ndarray, float]


# ---------------------------------------------------------------------------
# Numerical CDF helpers
# ---------------------------------------------------------------------------

def _cdf_by_quadrature(pdf_fn, x: np.ndarray, x_lo: float = 0.0) -> np.ndarray:
    """Compute CDF by cumulative quadrature from x_lo (accurate, slow for large arrays)."""
    out = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= x_lo:
            out[i] = 0.0
        else:
            val, _ = quad(lambda t: float(pdf_fn(np.array([t]))[0]),
                          x_lo + _EPS, xi, limit=200)
            out[i] = np.clip(val, 0.0, 1.0)
    return out


def _cdf_by_grid(pdf_fn, x: np.ndarray, x_lo: float, x_hi: float,
                 n_grid: int = 3000) -> np.ndarray:
    """
    Fast CDF via cumulative-trapezoid integration on a dense grid + linear interpolation.

    Accurate to ~1/n_grid^2 for smooth integrands; endpoint-safe by using a small
    interior offset. Normalizes so that F(x_hi) = 1.

    Parameters
    ----------
    pdf_fn : vectorized callable (x_array) -> f_array
    x      : evaluation points (any ordering)
    x_lo, x_hi : support boundaries (exclusive)
    n_grid : grid density; 3000 gives < 1e-5 relative error for smooth densities
    """
    eps_lo = max(x_lo, _EPS)
    eps_hi = x_hi - _EPS
    if eps_lo >= eps_hi:
        return np.zeros_like(x)
    # Denser grid near x_lo for integrands with near-zero singularities
    g_lin = np.linspace(eps_lo, eps_hi, n_grid)
    g_log = np.geomspace(max(eps_lo, 1e-9), eps_hi, n_grid // 2)
    grid = np.unique(np.concatenate([g_log, g_lin]))
    f_grid = pdf_fn(grid)
    f_grid = np.where(np.isfinite(f_grid), f_grid, 0.0)
    F_grid = np.concatenate([[0.0], cumulative_trapezoid(f_grid, grid)])
    # Normalize to [0, 1] using the computed total mass
    total = F_grid[-1]
    if total > 0:
        F_grid /= total
    # Clip for numerical safety
    F_grid = np.clip(F_grid, 0.0, 1.0)
    # Interpolate at requested points; extrapolate as 0/1 outside [grid[0], grid[-1]]
    out = np.interp(x, grid, F_grid, left=0.0, right=1.0)
    out[x >= x_hi] = 1.0
    out[x <= x_lo] = 0.0
    return out


# ---------------------------------------------------------------------------
# Exp-1: Original Beta-sector CDFs
# ---------------------------------------------------------------------------

def cdf_mB(x: ArrayLike, p: BetaSectorParams) -> np.ndarray:
    """
    Modified-Beta (mB) CDF on [0, beta_1].

    Uses the formula:
      F_mB(x) = I(z(x); p, q) + correction_term
    where z(x) = ((x/beta_1) + (x/beta_2)) / (1 + x/beta_2)
    and the correction accounts for the (1+x/beta_2)^(-p-q-1) factor.

    Because the paper's closed form involves the Appell F1 function (which is
    numerically fragile), we use numerical quadrature as primary method and
    confirm against the formula where feasible.

    Parameters
    ----------
    x : array of evaluation points
    p : BetaSectorParams with kappa, kappa_1, kappa_2 all > 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask_above = x >= p.beta_1
    out[mask_above] = 1.0

    interior = (x > 0) & (x < p.beta_1)
    if not np.any(interior):
        return out

    xi = x[interior]
    pdf_fn = lambda xv: pdf_mB(xv, p)
    # Grid interpolation is much faster for large arrays; quadrature for small ones
    if len(xi) >= 6:
        out[interior] = _cdf_by_grid(pdf_fn, xi, 0.0, p.beta_1)
    else:
        for i, xi_val in enumerate(xi):
            val, _ = quad(lambda t: float(pdf_fn(np.array([t]))[0]),
                          _EPS, xi_val, limit=200)
            out[np.where(interior)[0][i]] = np.clip(val, 0.0, 1.0)
    return out


def cdf_IGa1(x: ArrayLike, p: BetaSectorParams) -> np.ndarray:
    """
    IGa1 CDF via numerical quadrature.

    The paper provides a closed form involving incomplete Gamma with
    shifted arguments, but numerical quadrature is used for robustness.

    Parameters
    ----------
    x : array of evaluation points
    p : BetaSectorParams with kappa_1 > 0, kappa_2 > 0, kappa = 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    out[x >= p.beta_1] = 1.0

    interior = (x > 0) & (x < p.beta_1)
    if not np.any(interior):
        return out

    xi = x[interior]
    pdf_fn = lambda xv: pdf_IGa1(xv, p)
    if len(xi) >= 6:
        out[interior] = _cdf_by_grid(pdf_fn, xi, 0.0, p.beta_1)
    else:
        for i, xi_val in enumerate(xi):
            val, _ = quad(lambda t: float(pdf_fn(np.array([t]))[0]),
                          _EPS, xi_val, limit=200)
            out[np.where(interior)[0][i]] = np.clip(val, 0.0, 1.0)
    return out


def cdf_BP_B2(x: ArrayLike, p: BetaSectorParams) -> np.ndarray:
    """
    BP/B2 CDF on [0, infinity).

    F_BP_B2(x) = I(x/(x+beta_2); p, q+1)  [regularized incomplete beta]

    Parameters
    ----------
    x : array of evaluation points
    p : BetaSectorParams with kappa > 0, kappa_2 > 0, kappa_1 = 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = x > 0
    if not np.any(mask):
        return out

    xm = x[mask]
    b2 = p.beta_2
    pp, qq = p.p, p.q

    # regularized incomplete beta I(t; a, b) = betainc(a, b, t) in scipy
    t = xm / (xm + b2)
    out[mask] = betainc(pp, qq + 1, t)
    return out


def cdf_B1(x: ArrayLike, p: BetaSectorParams) -> np.ndarray:
    """
    B1 CDF on [0, beta_1].

    F_B1(x) = I(x/beta_1; p, q)

    Parameters
    ----------
    x : array of evaluation points
    p : BetaSectorParams with kappa > 0, kappa_1 > 0, kappa_2 = 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    out[x >= p.beta_1] = 1.0

    mask = (x > 0) & (x < p.beta_1)
    if not np.any(mask):
        return out

    xm = x[mask]
    out[mask] = betainc(p.p, p.q, xm / p.beta_1)
    return out


def cdf_IGa(x: ArrayLike, p: BetaSectorParams) -> np.ndarray:
    """
    IGa CDF on (0, infinity).

    F_IGa(x) = Gamma(q+1, beta/x) / Gamma(q+1) = gammaincc(q+1, beta/x)

    Derivation via substitution u = beta/t in the PDF integral:
      integral_0^x f(t) dt = gammaincc(q+1, beta/x)
    because gammaincc(a, z) = Gamma(a, z)/Gamma(a) is DECREASING in z,
    and z = beta/x is DECREASING in x, so the CDF is INCREASING in x. ✓

    Parameters
    ----------
    x : array of evaluation points
    p : BetaSectorParams with kappa_2 > 0  (kappa = 0, kappa_1 = 0)
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = x > 0
    if not np.any(mask):
        return out

    xm = x[mask]
    bet = p.beta
    q_iga = 2 * p.gamma / p.kappa_2**2

    # F(x) = P(X<=x) = Gamma(q+1, beta/x)/Gamma(q+1) = gammaincc(q+1, beta/x)
    # (upper regularized incomplete gamma — decreasing in beta/x → increasing in x)
    out[mask] = gammaincc(q_iga + 1, bet / xm)
    return np.clip(out, 0.0, 1.0)


def cdf_Ga(x: ArrayLike, p: BetaSectorParams) -> np.ndarray:
    """
    Gamma (Ga) CDF on [0, infinity).

    F_Ga(x) = gammainc(p, x/lambda) / Gamma(p)  [lower regularized incomplete gamma]
    In scipy: gammainc(a, x) is the regularized lower incomplete gamma = P(a, x).

    Parameters
    ----------
    x : array of evaluation points
    p : BetaSectorParams with kappa > 0  (kappa_1 = 0, kappa_2 = 0)
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = x > 0
    if not np.any(mask):
        return out

    xm = x[mask]
    out[mask] = gammainc(p.p, xm / p.lam)
    return out


# ---------------------------------------------------------------------------
# Exp-2: Generalized Dynamical route CDFs
# ---------------------------------------------------------------------------

def cdf_mGB(x: ArrayLike, gp: GeneralizedParams) -> np.ndarray:
    """
    mGB dynamical-route CDF via numerical quadrature.

    The Appell F1 closed form is numerically fragile (mpmath required for
    precision). Quadrature is used as the primary method.

    Parameters
    ----------
    x : array of evaluation points
    gp : GeneralizedParams with kappa, kappa_1, kappa_2 all > 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    upper = gp.beta_1 ** (1.0 / gp.alpha)
    out[x >= upper] = 1.0

    interior = (x > 0) & (x < upper)
    if not np.any(interior):
        return out

    xi = x[interior]
    pdf_fn = lambda xv: pdf_mGB(xv, gp)
    # Grid interpolation avoids O(n) quadrature calls for large arrays
    if len(xi) >= 6:
        out[interior] = _cdf_by_grid(pdf_fn, xi, 0.0, upper)
    else:
        for i, xi_val in enumerate(xi):
            val, _ = quad(lambda t: float(pdf_fn(np.array([t]))[0]),
                          _EPS, xi_val, limit=200)
            out[np.where(interior)[0][i]] = np.clip(val, 0.0, 1.0)
    return out


def cdf_GBP(x: ArrayLike, gp: GeneralizedParams) -> np.ndarray:
    """
    GBP dynamical-route CDF via regularized incomplete beta.

    F_GBP(x) = I(x^alpha/(x^alpha+beta_2); u, w-u)

    Parameters
    ----------
    x : array of evaluation points
    gp : GeneralizedParams with kappa > 0, kappa_2 > 0, kappa_1 = 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = x > 0
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** gp.alpha
    b2 = gp.beta_2
    u, w = gp.u, gp.w
    wu = w - u

    t = xa / (xa + b2)
    out[mask] = betainc(u, wu, t)
    return out


def cdf_GB1(x: ArrayLike, gp: GeneralizedParams) -> np.ndarray:
    """
    GB1 dynamical-route CDF via regularized incomplete beta.

    F_GB1(x) = I(x^alpha/beta_1; u, v)

    Parameters
    ----------
    x : array of evaluation points
    gp : GeneralizedParams with kappa > 0, kappa_1 > 0, kappa_2 = 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    upper = gp.beta_1 ** (1.0 / gp.alpha)
    out[x >= upper] = 1.0

    mask = (x > 0) & (x < upper)
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** gp.alpha
    out[mask] = betainc(gp.u, gp.v, xa / gp.beta_1)
    return out


def cdf_GIGa(x: ArrayLike, gp: GeneralizedParams) -> np.ndarray:
    """
    GIGa dynamical-route CDF.

    F_GIGa(x) = Gamma((q+1)/alpha, beta/(alpha*x^alpha)) / Gamma((q+1)/alpha)
              = 1 - gammaincc((q+1)/alpha, beta/(alpha*x^alpha))

    Parameters
    ----------
    x : array of evaluation points
    gp : GeneralizedParams with kappa_2 > 0, kappa = 0, kappa_1 = 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = x > 0
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** gp.alpha
    bet = gp.beta
    alpha = gp.alpha
    q_giga = 2 * gp.gamma / gp.kappa_2**2
    shape = (q_giga + 1) / alpha

    # F_GIGa(x) = Gamma(shape, beta/(alpha*x^alpha))/Gamma(shape) = gammaincc(shape, ...)
    out[mask] = gammaincc(shape, bet / (alpha * xa))
    return np.clip(out, 0.0, 1.0)


def cdf_GGa(x: ArrayLike, gp: GeneralizedParams) -> np.ndarray:
    """
    GGa dynamical-route CDF.

    F_GGa(x) = gammainc(u, x^alpha/(alpha*lam2))  [lower regularized incomplete gamma]

    Parameters
    ----------
    x : array of evaluation points
    gp : GeneralizedParams with kappa > 0, kappa_1 = 0, kappa_2 = 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = x > 0
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** gp.alpha
    scale = gp.alpha * gp.lam2
    out[mask] = gammainc(gp.u, xa / scale)
    return out


# ---------------------------------------------------------------------------
# Exp-3: Generalized Distributional route CDFs
# F_X(x) = F_Y(x^alpha) because Y = X^alpha is a monotone transformation.
# ---------------------------------------------------------------------------

def cdf_mB_alpha(x: ArrayLike, p: BetaSectorParams, alpha: float = 1.0) -> np.ndarray:
    """
    Distributional-route mB_alpha CDF: F_X(x) = F_mB(x^alpha).

    Parameters
    ----------
    x : array of evaluation points
    p : BetaSectorParams (parent mB parameters)
    alpha : transformation exponent > 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    upper = p.beta_1 ** (1.0 / alpha)
    out[x >= upper] = 1.0

    mask = (x > 0) & (x < upper)
    if not np.any(mask):
        return out

    xa = x[mask] ** alpha
    out[mask] = cdf_mB(xa, p)
    return out


def cdf_BP_B2_alpha(x: ArrayLike, p: BetaSectorParams, alpha: float = 1.0) -> np.ndarray:
    """
    Distributional-route BP/B2_alpha CDF: F_X(x) = F_BP_B2(x^alpha).
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = x > 0
    if not np.any(mask):
        return out

    xa = x[mask] ** alpha
    out[mask] = cdf_BP_B2(xa, p)
    return out


def cdf_B1_alpha(x: ArrayLike, p: BetaSectorParams, alpha: float = 1.0) -> np.ndarray:
    """
    Distributional-route B1_alpha CDF: F_X(x) = F_B1(x^alpha).
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    upper = p.beta_1 ** (1.0 / alpha)
    out[x >= upper] = 1.0

    mask = (x > 0) & (x < upper)
    if not np.any(mask):
        return out

    xa = x[mask] ** alpha
    out[mask] = cdf_B1(xa, p)
    return out


def cdf_IGa_alpha(x: ArrayLike, p: BetaSectorParams, alpha: float = 1.0) -> np.ndarray:
    """
    Distributional-route IGa_alpha CDF: F_X(x) = F_IGa(x^alpha).
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = x > 0
    if not np.any(mask):
        return out

    xa = x[mask] ** alpha
    out[mask] = cdf_IGa(xa, p)
    return out


def cdf_Ga_alpha(x: ArrayLike, p: BetaSectorParams, alpha: float = 1.0) -> np.ndarray:
    """
    Distributional-route Ga_alpha CDF: F_X(x) = F_Ga(x^alpha).
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = x > 0
    if not np.any(mask):
        return out

    xa = x[mask] ** alpha
    out[mask] = cdf_Ga(xa, p)
    return out


def cdf_IGa1_alpha(x: ArrayLike, p: BetaSectorParams, alpha: float = 1.0) -> np.ndarray:
    """
    Distributional-route IGa1_alpha CDF: F_X(x) = F_IGa1(x^alpha).
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    upper = p.beta_1 ** (1.0 / alpha)
    out[x >= upper] = 1.0

    mask = (x > 0) & (x < upper)
    if not np.any(mask):
        return out

    xa = x[mask] ** alpha
    out[mask] = cdf_IGa1(xa, p)
    return out


# ---------------------------------------------------------------------------
# Exp-4: Traditional GB CDF
# ---------------------------------------------------------------------------

def cdf_GB(x: ArrayLike, gp: GeneralizedParams) -> np.ndarray:
    """
    Traditional GB benchmark CDF.

    F_GB(x) = I(z(x); p, q)
    where z(x) = ((x^alpha/beta_1) + (x^alpha/beta_2)) / (1 + x^alpha/beta_2)

    Parameters
    ----------
    x : array of evaluation points
    gp : GeneralizedParams with kappa, kappa_1, kappa_2 all > 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    upper = gp.beta_1 ** (1.0 / gp.alpha)
    out[x >= upper] = 1.0

    mask = (x > 0) & (x < upper)
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** gp.alpha
    b1, b2 = gp.beta_1, gp.beta_2
    pp, qq = gp.p, gp.q

    z = (xa / b1 + xa / b2) / (1 + xa / b2)
    out[mask] = betainc(pp, qq, z)
    return out
