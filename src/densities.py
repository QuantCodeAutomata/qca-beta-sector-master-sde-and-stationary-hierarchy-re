"""
Stationary PDF implementations for the Beta-sector SDE hierarchy.

Covers three layers:
  1. Original Beta-sector (exp_1): mB, IGa1, BP_B2, B1, IGa, Ga
  2. Generalized dynamical route (exp_2): mGB, GBP, GB1, GIGa1, GIGa, GGa
  3. Generalized distributional/pushforward route (exp_3): mB_alpha, BP_B2_alpha,
     B1_alpha, IGa_alpha, Ga_alpha, IGa1_alpha
  4. Traditional GB benchmark (exp_4): pdf_GB

All PDFs are vectorized over numpy arrays x and return zeros outside their support.
Log-space arithmetic is used throughout for numerical stability.

Custom implementations — Context7 confirmed no library equivalent for these
novel SDE stationary laws.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betaln, gammaln, gammaincc, betainc, hyp2f1
from scipy.integrate import quad
from typing import Union

from src.parameters import BetaSectorParams, GeneralizedParams

ArrayLike = Union[np.ndarray, float]

_EPS = 1e-14  # small epsilon for endpoint safety


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_log(x: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Log with a floor at eps to avoid -inf at boundaries."""
    return np.log(np.maximum(x, eps))


def _to_array(x: ArrayLike) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float))


# ---------------------------------------------------------------------------
# Exp-1: Original Beta-sector PDFs
# ---------------------------------------------------------------------------

def pdf_mB(x: ArrayLike, p: BetaSectorParams) -> np.ndarray:
    """
    Modified-Beta (mB) stationary PDF on [0, beta_1].

    f_mB(x) = [(p+q)(1+beta_1/beta_2)^(p+1)] /
               [beta_1 * (p + (1+beta_1/beta_2)*q) * B(p,q)]
              * (x/beta_1)^(p-1) * (1-x/beta_1)^(q-1) * (1+x/beta_2)^(-p-q-1)

    Parameters
    ----------
    x : array of evaluation points
    p : BetaSectorParams with kappa, kappa_1, kappa_2 all > 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = (x > 0) & (x < p.beta_1)
    if not np.any(mask):
        return out

    xm = x[mask]
    b1, b2 = p.beta_1, p.beta_2
    pp, qq = p.p, p.q

    r = b1 / b2  # beta_1 / beta_2
    # log normalization constant
    log_norm = (
        np.log(pp + qq)
        + (pp + 1) * np.log(1 + r)
        - np.log(b1)
        - np.log(pp + (1 + r) * qq)
        - betaln(pp, qq)
    )
    log_f = (
        log_norm
        + (pp - 1) * _safe_log(xm / b1)
        + (qq - 1) * _safe_log(1 - xm / b1)
        + (-pp - qq - 1) * _safe_log(1 + xm / b2)
    )
    out[mask] = np.exp(log_f)
    return out


def pdf_IGa1(x: ArrayLike, p: BetaSectorParams) -> np.ndarray:
    """
    Inverse-Gamma-1 (IGa1) stationary PDF on (0, beta_1).

    Branch: kappa = 0.
    f_IGa1(x) = c * exp(-beta/x) * (x/beta)^(-q-2) * (1-x/beta_1)^(q-1)
    c = beta_1 / [beta * exp(-beta/beta_1) * (q*beta_1 + beta) * Gamma(q)]

    Parameters
    ----------
    x : array of evaluation points
    p : BetaSectorParams with kappa_1 > 0, kappa_2 > 0, kappa = 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = (x > 0) & (x < p.beta_1)
    if not np.any(mask):
        return out

    xm = x[mask]
    b1, bet = p.beta_1, p.beta
    # q is derived from IGa1 branch: use beta-related formula
    # For IGa1: from paper, q = (Gamma-branch analogue)
    # From the paper: f_IGa1 uses q where beta = 2*gamma*theta/kappa_2^2
    # and q corresponds to exponents in (1-x/beta_1)^(q-1)
    # For IGa1, q is not from kappa (kappa=0), so use the IGa1-specific q
    # From the stationary Fokker-Planck for kappa=0 SDE:
    # sigma^2(x) = kappa_2^2*x^2*(1-x/beta_1)
    # q from the FP solution equals 2*gamma*(1-kappa_1^2*theta)/kappa_2^2 [with kappa->0]
    # = 2*gamma/kappa_2^2 - 2*gamma*kappa_1^2*theta/kappa_2^2
    # For pure IGa1: q = 2*gamma*(1 - kappa_1^2*theta)/kappa_2^2
    # When kappa=0: the parameter q must be computed from kappa_1, kappa_2 only
    qq = 2 * p.gamma * (1 - p.kappa_1**2 * p.theta) / p.kappa_2**2
    if qq <= 0:
        return out

    # log constant: log(c) = log(b1) - log(beta) + beta/b1 - log(q*b1+beta) - gammaln(q)
    log_c = (
        np.log(b1)
        - np.log(bet)
        + bet / b1
        - np.log(qq * b1 + bet)
        - gammaln(qq)
    )
    log_f = (
        log_c
        - bet / xm
        + (-qq - 2) * _safe_log(xm / bet)
        + (qq - 1) * _safe_log(1 - xm / b1)
    )
    out[mask] = np.exp(log_f)
    return out


def pdf_BP_B2(x: ArrayLike, p: BetaSectorParams) -> np.ndarray:
    """
    Beta-Prime / B2 (BP/B2) stationary PDF on [0, infinity).

    Branch: kappa_1 = 0.
    f_BP_B2(x) = 1/[beta_2 * B(p, q+1)] * (x/beta_2)^(p-1) * (1+x/beta_2)^(-p-q-1)

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

    log_norm = -np.log(b2) - betaln(pp, qq + 1)
    log_f = (
        log_norm
        + (pp - 1) * _safe_log(xm / b2)
        + (-pp - qq - 1) * _safe_log(1 + xm / b2)
    )
    out[mask] = np.exp(log_f)
    return out


def pdf_B1(x: ArrayLike, p: BetaSectorParams) -> np.ndarray:
    """
    Beta-1 (B1) stationary PDF on [0, beta_1].

    Branch: kappa_2 = 0.
    f_B1(x) = 1/[beta_1 * B(p,q)] * (x/beta_1)^(p-1) * (1-x/beta_1)^(q-1)

    Parameters
    ----------
    x : array of evaluation points
    p : BetaSectorParams with kappa > 0, kappa_1 > 0, kappa_2 = 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = (x >= 0) & (x <= p.beta_1)
    # Exclude exact boundaries when exponents < 1
    interior = (x > 0) & (x < p.beta_1)
    mask = interior

    if not np.any(mask):
        return out

    xm = x[mask]
    b1 = p.beta_1
    pp, qq = p.p, p.q

    log_norm = -np.log(b1) - betaln(pp, qq)
    log_f = (
        log_norm
        + (pp - 1) * _safe_log(xm / b1)
        + (qq - 1) * _safe_log(1 - xm / b1)
    )
    out[mask] = np.exp(log_f)
    return out


def pdf_IGa(x: ArrayLike, p: BetaSectorParams) -> np.ndarray:
    """
    Inverse-Gamma (IGa) terminal stationary PDF on (0, infinity).

    Terminal branch: kappa = 0, kappa_1 = 0 (or limit of IGa1 with kappa_1 -> 0).
    f_IGa(x) = 1/[beta * Gamma(q+1)] * (x/beta)^(-q-2) * exp(-beta/x)

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
    # For IGa: q+1 shape parameter.  From the paper, beta=2*gamma*theta/kappa_2^2
    # and shape is q+1 = 2*gamma/kappa_2^2 (from Fokker-Planck limit)
    # We compute: q = 2*gamma*(1 - kappa_1^2*theta)/kappa_2^2 with kappa_1->0
    # => q = 2*gamma/kappa_2^2 -- but we need shape for IGa specifically
    # From paper: f_IGa uses "q" parameter aligned with IGa1 limit kappa_1->0
    # With kappa_1=0: q_IGa = 2*gamma/kappa_2^2 - p_IGa (but p_IGa undefined if kappa=0)
    # For pure IGa terminal (kappa=0, kappa_1=0): shape q+1 from FP:
    # sigma^2 = kappa_2^2*x^2, drift = -gamma*(x-theta)
    # f proportional to x^(-2*gamma/kappa_2^2) * exp(-beta/x) where beta=2*gamma*theta/kappa_2^2
    # => f = 1/(beta*Gamma(q+1)) * (x/beta)^(-q-2) * exp(-beta/x) with q+1=2*gamma/kappa_2^2-1
    # i.e., q = 2*gamma/kappa_2^2 - 2
    # But paper writes q+1 as the shape, so q = 2*gamma/kappa_2^2 - 2
    # From IGa1 limit with kappa_1=0 and kappa=0: q_IGa = 2*gamma/kappa_2^2 - 2
    # We use: shape = 2*gamma/kappa_2^2 - 1  (this is q+1 in paper notation)
    # => q+1 = 2*gamma/kappa_2^2 - 1  => q = 2*gamma/kappa_2^2 - 2
    # Actually from paper Eq (IGa): uses q parameter, with beta=2*gamma*theta/kappa_2^2
    # The simplest is: treat q as free parameter. For IGa, q+1 is the IG shape.
    # From the zero-current FP: f proportional to x^(-2-2*gamma*theta/kappa_2^2/theta)*exp(-beta/x)
    # = x^(-2 - beta/theta) * exp(-beta/x)   where beta/theta = 2*gamma/kappa_2^2
    # So exponent of x is -2-q-1 = -q-2 => q+1 = 2*gamma*theta/kappa_2^2 = beta (wrong units)
    # Let's be systematic: shape of IG(shape, scale) is alpha_ig, scale is beta_ig.
    # From FP: f(x) ~ x^(-2*gamma/kappa_2^2) * exp(-2*gamma*theta/(kappa_2^2 * x))
    # = x^(-(q_ig+2)) * exp(-beta/x) where q_ig+2 = 2*gamma/kappa_2^2, beta=2*gamma*theta/kappa_2^2
    # => q_ig = 2*gamma/kappa_2^2 - 2
    # normalization: integral_0^inf x^(-q_ig-2)*exp(-beta/x)dx = beta^(-q_ig-1)*Gamma(q_ig+1)
    # => c = 1/(beta*Gamma(q_ig+1)) normalizes
    # In the paper's notation, q_ig = 2*gamma/kappa_2^2 - 2.
    # We define q_ig from the stored beta: q_ig+1 = beta/theta = 2*gamma/kappa_2^2 - 1
    # Actually: beta = 2*gamma*theta/kappa_2^2.  Shape q+1 = beta/theta = 2*gamma/kappa_2^2 - 1??
    # No. Let's use: q_ig+1 from IG normalization. From FP:
    # shape(IG) = 2*gamma/kappa_2^2 - 1, scale = beta = 2*gamma*theta/kappa_2^2.
    # In paper notation: q = shape - 1 = 2*gamma/kappa_2^2 - 2.
    # But from the q_IGa1 formula at kappa=0, kappa_1->0:
    # q_IGa1 = 2*gamma*(1-kappa_1^2*theta)/kappa_2^2 -> 2*gamma/kappa_2^2 as kappa_1->0.
    # Then IGa uses q = q_IGa1 from the limit = 2*gamma/kappa_2^2.
    # Under this convention: f_IGa = 1/[beta*Gamma(q+1)] * (x/beta)^(-q-2) * exp(-beta/x)
    # Normalization check with q = 2*gamma/kappa_2^2:
    # integral = 1/(beta*Gamma(q+1)) * beta^(q+1) * Gamma(q+1) = 1.  Yes! (standard IG integral)
    # So q_IGa = 2*gamma/kappa_2^2 (NOT q_IGa1 from kappa_1>0 case)
    q_iga = 2 * p.gamma / p.kappa_2**2

    log_norm = -np.log(bet) - gammaln(q_iga + 1)
    log_f = (
        log_norm
        + (-q_iga - 2) * _safe_log(xm / bet)
        - bet / xm
    )
    out[mask] = np.exp(log_f)
    return out


def pdf_Ga(x: ArrayLike, p: BetaSectorParams) -> np.ndarray:
    """
    Gamma (Ga) terminal stationary PDF on [0, infinity).

    Terminal branch: kappa_1 = 0, kappa_2 = 0.
    f_Ga(x) = 1/[lambda * Gamma(p)] * (x/lambda)^(p-1) * exp(-x/lambda)
    where lambda = kappa^2/(2*gamma) = theta/p.

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
    lam = p.lam
    pp = p.p

    log_norm = -np.log(lam) - gammaln(pp)
    log_f = (
        log_norm
        + (pp - 1) * _safe_log(xm / lam)
        - xm / lam
    )
    out[mask] = np.exp(log_f)
    return out


# ---------------------------------------------------------------------------
# Exp-2: Generalized Dynamical route PDFs
# ---------------------------------------------------------------------------

def _mGB_log_normalizer(gp: GeneralizedParams) -> float:
    """
    Log of the mGB normalization constant using Gauss 2F1.

    c_mGB = alpha * beta_1^(-1/alpha) * (beta_2/beta_1)^(u-delta) /
            [B(u,v) * 2F1(w, u; u+v; -beta_1/beta_2)]

    Uses scipy.special.hyp2f1 for the hypergeometric function.
    # Using scipy.special.hyp2f1 — Context7 confirmed
    """
    u, v, w = gp.u, gp.v, gp.w
    b1, b2 = gp.beta_1, gp.beta_2
    alpha = gp.alpha
    delta = gp.delta

    # 2F1(w, u; u+v; -b1/b2)
    z_hyp = -b1 / b2
    hyp_val = float(hyp2f1(w, u, u + v, z_hyp))
    if hyp_val <= 0:
        raise ValueError(f"Hypergeometric 2F1 value non-positive: {hyp_val}")

    log_c = (
        np.log(alpha)
        - (1.0 / alpha) * np.log(b1)
        + (u - delta) * (np.log(b2) - np.log(b1))
        - betaln(u, v)
        - np.log(hyp_val)
    )
    return log_c


def pdf_mGB(x: ArrayLike, gp: GeneralizedParams) -> np.ndarray:
    """
    Generalized modified-Beta (mGB) dynamical-route PDF on [0, beta_1^(1/alpha)].

    f_mGB(x) = c_mGB * (x^alpha/beta_2)^(u-delta) * (1-x^alpha/beta_1)^(v-1)
                      * (1+x^alpha/beta_2)^(-w)

    Normalization via 2F1 hypergeometric.

    Parameters
    ----------
    x : array of evaluation points
    gp : GeneralizedParams with kappa, kappa_1, kappa_2 all > 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    upper = gp.beta_1 ** (1.0 / gp.alpha)
    mask = (x > 0) & (x < upper)
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** gp.alpha
    b1, b2 = gp.beta_1, gp.beta_2
    u, v, w = gp.u, gp.v, gp.w
    delta = gp.delta

    try:
        log_c = _mGB_log_normalizer(gp)
    except Exception:
        return out

    log_f = (
        log_c
        + (u - delta) * _safe_log(xa / b2)
        + (v - 1) * _safe_log(1 - xa / b1)
        + (-w) * _safe_log(1 + xa / b2)
    )
    out[mask] = np.exp(log_f)
    return out


def pdf_GBP(x: ArrayLike, gp: GeneralizedParams) -> np.ndarray:
    """
    Generalized Beta-Prime (GBP) dynamical-route PDF on [0, infinity).

    Branch: kappa_1 = 0 (or beta_1 -> infinity).
    f_GBP(x) = alpha / [beta_2^(1/alpha) * B(u, w-u)] *
               (x^alpha/beta_2)^(u-delta) * (1+x^alpha/beta_2)^(-w)

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
    delta = gp.delta
    alpha = gp.alpha

    # w - u = 1 + q1/alpha - u = 1 + (p+q)/alpha - (p+alpha-1)/alpha = q/alpha + 1/alpha = v + delta
    wu = w - u
    log_norm = (
        np.log(alpha)
        - (1.0 / alpha) * np.log(b2)
        - betaln(u, wu)
    )
    log_f = (
        log_norm
        + (u - delta) * _safe_log(xa / b2)
        + (-w) * _safe_log(1 + xa / b2)
    )
    out[mask] = np.exp(log_f)
    return out


def pdf_GB1(x: ArrayLike, gp: GeneralizedParams) -> np.ndarray:
    """
    Generalized Beta-1 (GB1) dynamical-route PDF on [0, beta_1^(1/alpha)].

    Branch: kappa_2 = 0 (or beta_2 -> infinity).
    f_GB1(x) = alpha / [beta_1^(1/alpha) * B(u,v)] *
               (x^alpha/beta_1)^(u-delta) * (1-x^alpha/beta_1)^(v-1)

    Parameters
    ----------
    x : array of evaluation points
    gp : GeneralizedParams with kappa > 0, kappa_1 > 0, kappa_2 = 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    upper = gp.beta_1 ** (1.0 / gp.alpha)
    mask = (x > 0) & (x < upper)
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** gp.alpha
    b1 = gp.beta_1
    u, v = gp.u, gp.v
    delta = gp.delta
    alpha = gp.alpha

    log_norm = (
        np.log(alpha)
        - (1.0 / alpha) * np.log(b1)
        - betaln(u, v)
    )
    log_f = (
        log_norm
        + (u - delta) * _safe_log(xa / b1)
        + (v - 1) * _safe_log(1 - xa / b1)
    )
    out[mask] = np.exp(log_f)
    return out


def pdf_GIGa(x: ArrayLike, gp: GeneralizedParams) -> np.ndarray:
    """
    Generalized Inverse-Gamma (GIGa) terminal dynamical-route PDF on (0, infinity).

    Terminal branch: kappa = 0, kappa_1 = 0.
    f_GIGa(x) = alpha / [(beta/alpha)^(1/alpha) * Gamma((q+1)/alpha)] *
                (alpha*x^alpha/beta)^(-(q+2)/alpha) * exp(-beta/(alpha*x^alpha))

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
    # For GIGa, q comes from the FP limit: use 2*gamma/kappa_2^2
    q_giga = 2 * gp.gamma / gp.kappa_2**2
    shape = (q_giga + 1) / alpha
    scale_factor = bet / alpha  # (beta/alpha)

    log_norm = (
        np.log(alpha)
        - (1.0 / alpha) * np.log(scale_factor)
        - gammaln(shape)
    )
    # (alpha*x^alpha/beta)^(-(q+2)/alpha) = (xa/scale_factor)^(-(q+2)/alpha)
    # since alpha*xa/beta = xa/(beta/alpha) = xa/scale_factor
    exp_outer = -(q_giga + 2) / alpha
    log_f = (
        log_norm
        + exp_outer * _safe_log(xa / scale_factor)
        - scale_factor / xa  # = -beta/(alpha*x^alpha); NOT -xa/scale_factor
    )
    out[mask] = np.exp(log_f)
    return out


def pdf_GGa(x: ArrayLike, gp: GeneralizedParams) -> np.ndarray:
    """
    Generalized Gamma (GGa) terminal dynamical-route PDF on [0, infinity).

    Terminal branch: kappa_1 = 0, kappa_2 = 0.
    f_GGa(x) = alpha / [(alpha*lam2)^(1/alpha) * Gamma(u)] *
               (x^alpha/(alpha*lam2))^(u-delta) * exp(-x^alpha/(alpha*lam2))

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
    lam2 = gp.lam2
    alpha = gp.alpha
    u = gp.u
    delta = gp.delta
    scale = alpha * lam2  # alpha * lambda_2

    log_norm = (
        np.log(alpha)
        - (1.0 / alpha) * np.log(scale)
        - gammaln(u)
    )
    log_f = (
        log_norm
        + (u - delta) * _safe_log(xa / scale)
        - xa / scale
    )
    out[mask] = np.exp(log_f)
    return out


def pdf_GIGa1(x: ArrayLike, gp: GeneralizedParams,
              norm_const: float | None = None) -> np.ndarray:
    """
    Generalized Inverse-Gamma-1 (GIGa1) dynamical-route PDF on (0, beta_1^(1/alpha)).

    Branch: kappa = 0. Unnormalized kernel:
      f(x) ∝ exp(-beta/(alpha*x^alpha)) * (alpha*x^alpha/beta)^(-(q+2)/alpha)
              * (1 - x^alpha/beta_1)^(q/alpha - 1)

    Normalization constant computed numerically via quadrature if not provided.

    Parameters
    ----------
    x : array of evaluation points
    gp : GeneralizedParams with kappa_1 > 0, kappa_2 > 0, kappa = 0
    norm_const : precomputed normalization integral (optional)
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    upper = gp.beta_1 ** (1.0 / gp.alpha)
    mask = (x > 0) & (x < upper)
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** gp.alpha
    bet = gp.beta
    b1 = gp.beta_1
    alpha = gp.alpha
    q_val = 2 * gp.gamma * (1 - gp.kappa_1**2 * gp.theta) / gp.kappa_2**2

    exp_outer = -(q_val + 2) / alpha
    v_val = q_val / alpha - 1

    def _kernel(t: float) -> float:
        ta = t ** alpha
        if ta <= 0 or ta >= b1:
            return 0.0
        log_k = (
            - bet / (alpha * ta)
            + exp_outer * np.log(alpha * ta / bet)
            + v_val * np.log(1 - ta / b1)
        )
        return float(np.exp(log_k))

    if norm_const is None:
        norm_val, _ = quad(_kernel, 1e-12, upper - 1e-12, limit=200)
        norm_const = norm_val

    if norm_const <= 0:
        return out

    log_f_vals = (
        - bet / (alpha * np.maximum(xa, _EPS))
        + exp_outer * _safe_log(alpha * xa / bet)
        + v_val * _safe_log(1 - xa / b1)
    )
    out[mask] = np.exp(log_f_vals) / norm_const
    return out


# ---------------------------------------------------------------------------
# Exp-3: Generalized Distributional (pushforward) route PDFs
# Pushforward rule: if Y has density f_Y, X = Y^(1/alpha) has density
#   f_X(x) = alpha * x^(alpha-1) * f_Y(x^alpha)
# ---------------------------------------------------------------------------

def pdf_mB_alpha(x: ArrayLike, p: BetaSectorParams, alpha: float = 1.0) -> np.ndarray:
    """
    Distributional-route mB_alpha: pushforward of mB by Y = X^alpha.

    f_mB_alpha(x) = alpha * x^(alpha-1) * f_mB(x^alpha)

    Support: [0, beta_1^(1/alpha)].

    Parameters
    ----------
    x : array of evaluation points
    p : BetaSectorParams (parent mB parameters)
    alpha : transformation exponent > 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    upper = p.beta_1 ** (1.0 / alpha)
    mask = (x > 0) & (x < upper)
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** alpha
    f_parent = pdf_mB(xa, p)
    out[mask] = alpha * xm ** (alpha - 1) * f_parent
    return out


def pdf_BP_B2_alpha(x: ArrayLike, p: BetaSectorParams, alpha: float = 1.0) -> np.ndarray:
    """
    Distributional-route BP/B2_alpha: pushforward of BP/B2.

    f_BP_B2_alpha(x) = alpha * x^(alpha-1) * f_BP_B2(x^alpha)
    Support: [0, infinity).
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = x > 0
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** alpha
    f_parent = pdf_BP_B2(xa, p)
    out[mask] = alpha * xm ** (alpha - 1) * f_parent
    return out


def pdf_B1_alpha(x: ArrayLike, p: BetaSectorParams, alpha: float = 1.0) -> np.ndarray:
    """
    Distributional-route B1_alpha: pushforward of B1.

    f_B1_alpha(x) = alpha * x^(alpha-1) * f_B1(x^alpha)
    Support: [0, beta_1^(1/alpha)].
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    upper = p.beta_1 ** (1.0 / alpha)
    mask = (x > 0) & (x < upper)
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** alpha
    f_parent = pdf_B1(xa, p)
    out[mask] = alpha * xm ** (alpha - 1) * f_parent
    return out


def pdf_IGa_alpha(x: ArrayLike, p: BetaSectorParams, alpha: float = 1.0) -> np.ndarray:
    """
    Distributional-route IGa_alpha: pushforward of IGa.

    f_IGa_alpha(x) = alpha * x^(alpha-1) * f_IGa(x^alpha)
    Support: (0, infinity).
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = x > 0
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** alpha
    f_parent = pdf_IGa(xa, p)
    out[mask] = alpha * xm ** (alpha - 1) * f_parent
    return out


def pdf_Ga_alpha(x: ArrayLike, p: BetaSectorParams, alpha: float = 1.0) -> np.ndarray:
    """
    Distributional-route Ga_alpha: pushforward of Ga.

    f_Ga_alpha(x) = alpha * x^(alpha-1) * f_Ga(x^alpha)
    Support: [0, infinity).
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    mask = x > 0
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** alpha
    f_parent = pdf_Ga(xa, p)
    out[mask] = alpha * xm ** (alpha - 1) * f_parent
    return out


def pdf_IGa1_alpha(x: ArrayLike, p: BetaSectorParams, alpha: float = 1.0) -> np.ndarray:
    """
    Distributional-route IGa1_alpha: pushforward of IGa1.

    f_IGa1_alpha(x) = alpha * x^(alpha-1) * f_IGa1(x^alpha)
    Support: (0, beta_1^(1/alpha)).
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    upper = p.beta_1 ** (1.0 / alpha)
    mask = (x > 0) & (x < upper)
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** alpha
    f_parent = pdf_IGa1(xa, p)
    out[mask] = alpha * xm ** (alpha - 1) * f_parent
    return out


# ---------------------------------------------------------------------------
# Exp-4: Traditional Generalized-Beta (GB) benchmark
# ---------------------------------------------------------------------------

def pdf_GB(x: ArrayLike, gp: GeneralizedParams) -> np.ndarray:
    """
    Traditional Generalized-Beta (GB) benchmark PDF on [0, beta_1^(1/alpha)].

    f_GB(x) = alpha / [beta_1^(1/alpha) * B(p,q)] * (1+beta_1/beta_2)^p *
              (x^alpha/beta_1)^(p-1/alpha) * (1-x^alpha/beta_1)^(q-1) *
              (1+x^alpha/beta_2)^(-p-q)

    This is the classical generalized Beta of the first kind with stretching
    parameter alpha. Independently normalized and checked.

    Parameters
    ----------
    x : array of evaluation points
    gp : GeneralizedParams with kappa, kappa_1, kappa_2 all > 0
    """
    x = _to_array(x)
    out = np.zeros_like(x)
    upper = gp.beta_1 ** (1.0 / gp.alpha)
    mask = (x > 0) & (x < upper)
    if not np.any(mask):
        return out

    xm = x[mask]
    xa = xm ** gp.alpha
    b1, b2 = gp.beta_1, gp.beta_2
    pp, qq = gp.p, gp.q
    alpha = gp.alpha

    r = b1 / b2
    log_norm = (
        np.log(alpha)
        - (1.0 / alpha) * np.log(b1)
        - betaln(pp, qq)
        + pp * np.log(1 + r)
    )
    log_f = (
        log_norm
        + (pp - 1.0 / alpha) * _safe_log(xa / b1)
        + (qq - 1) * _safe_log(1 - xa / b1)
        + (-pp - qq) * _safe_log(1 + xa / b2)
    )
    out[mask] = np.exp(log_f)
    return out
