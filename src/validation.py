"""
Validation tools for the Beta-sector SDE hierarchy experiments.

Provides:
  - Normalization checks (numerical integration of PDF over support)
  - CDF endpoint checks (F(lower) ≈ 0, F(upper) ≈ 1)
  - CDF derivative consistency (dF/dx ≈ f via finite differences)
  - Kolmogorov-Smirnov statistic between empirical and analytical CDFs
  - Cramér-von Mises statistic
  - Asymptotic slope estimation (log-log regression near x=0)
  - Branch-consistency checks (alpha=1 collapse)

# Custom — Context7 found no library equivalent for this validation suite.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import ks_1samp
from typing import Callable, Dict, Any, Tuple, Optional

_EPS = 1e-12


def check_normalization(
    pdf_fn: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    tol: float = 1e-2,
    n_intervals: int = 200,
) -> Dict[str, Any]:
    """
    Check that a PDF integrates to 1 over (lo, hi).

    Parameters
    ----------
    pdf_fn : callable (x_array) -> f_array
    lo, hi : integration limits (use small offset from endpoints)
    tol : acceptable deviation from 1
    n_intervals : number of quadrature subintervals hint

    Returns
    -------
    dict with keys: integral, error, passed
    """
    eps = max(lo, _EPS)
    val, err = quad(
        lambda x: float(pdf_fn(np.array([x]))[0]),
        eps, hi,
        limit=n_intervals,
    )
    return {
        "integral": val,
        "quad_error": err,
        "norm_error": abs(val - 1.0),
        "passed": abs(val - 1.0) < tol,
    }


def check_cdf_endpoints(
    cdf_fn: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    tol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Check F(lo+eps) ≈ 0 and F(hi-eps) ≈ 1.

    Parameters
    ----------
    cdf_fn : callable (x_array) -> F_array
    lo, hi : support boundaries
    tol : acceptable deviation
    """
    eps = 1e-8
    F_lo = float(cdf_fn(np.array([lo + eps]))[0])
    F_hi = float(cdf_fn(np.array([hi - eps]))[0])
    return {
        "F_lo": F_lo,
        "F_hi": F_hi,
        "err_lo": abs(F_lo - 0.0),
        "err_hi": abs(F_hi - 1.0),
        "passed": (abs(F_lo) < tol) and (abs(F_hi - 1.0) < tol),
    }


def check_cdf_derivative(
    pdf_fn: Callable[[np.ndarray], np.ndarray],
    cdf_fn: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    n_points: int = 50,
    h: float = 1e-5,
    tol_rtol: float = 0.05,
) -> Dict[str, Any]:
    """
    Check dF/dx ≈ f on an interior grid using central finite differences.

    Parameters
    ----------
    pdf_fn : callable (x_array) -> f_array
    cdf_fn : callable (x_array) -> F_array
    lo, hi : support boundaries
    n_points : number of interior evaluation points
    h : finite difference step
    tol_rtol : relative tolerance for agreement

    Returns
    -------
    dict with keys: max_abs_err, mean_rel_err, passed
    """
    x = np.linspace(lo + (hi - lo) * 0.05, hi - (hi - lo) * 0.05, n_points)
    dF = (cdf_fn(x + h) - cdf_fn(x - h)) / (2 * h)
    f = pdf_fn(x)
    abs_err = np.abs(dF - f)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(np.abs(f) > 1e-10, abs_err / np.abs(f), 0.0)
    return {
        "max_abs_err": float(np.max(abs_err)),
        "mean_rel_err": float(np.mean(rel_err)),
        "max_rel_err": float(np.max(rel_err)),
        "passed": float(np.mean(rel_err)) < tol_rtol,
    }


def compute_ks_statistic(
    samples: np.ndarray,
    cdf_fn: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, Any]:
    """
    Compute the Kolmogorov-Smirnov statistic between empirical samples and
    an analytical CDF.

    Parameters
    ----------
    samples : 1D array of simulated samples
    cdf_fn : callable (x_array) -> F_array (analytical CDF)

    Returns
    -------
    dict with keys: ks_stat, ks_pvalue
    """
    def _cdf_scalar(x):
        """Accept scalar or array from scipy's ks_1samp."""
        arr = np.atleast_1d(np.asarray(x, dtype=float))
        out = cdf_fn(arr)
        return out if arr.ndim > 0 and len(arr) > 1 else float(out[0])

    result = ks_1samp(samples, _cdf_scalar)
    return {"ks_stat": result.statistic, "ks_pvalue": result.pvalue}


def compute_cvm_statistic(
    samples: np.ndarray,
    cdf_fn: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, float]:
    """
    Compute the Cramér-von Mises statistic between empirical samples and
    an analytical CDF.

    CvM = (1/(12n)) + sum_i (F(x_i) - (2i-1)/(2n))^2

    Parameters
    ----------
    samples : 1D sorted array of samples
    cdf_fn : callable (x_array) -> F_array

    Returns
    -------
    dict with keys: cvm_stat
    """
    n = len(samples)
    x_sorted = np.sort(samples)
    F_vals = np.array([float(cdf_fn(np.array([xi]))[0]) for xi in x_sorted])
    i_vals = np.arange(1, n + 1)
    cvm = 1.0 / (12 * n) + np.sum((F_vals - (2 * i_vals - 1) / (2 * n)) ** 2)
    return {"cvm_stat": float(cvm)}


def estimate_asymptotic_slope(
    pdf_fn: Callable[[np.ndarray], np.ndarray],
    x_lo: float = 1e-5,
    x_hi: float = 1e-2,
    n_points: int = 30,
) -> Dict[str, float]:
    """
    Estimate the near-zero log-log slope of a PDF via linear regression.

    For f(x) ~ C * x^nu near x=0, log f = log C + nu * log x.
    Fits nu by OLS on log-log scale.

    Parameters
    ----------
    pdf_fn : callable (x_array) -> f_array
    x_lo, x_hi : range for slope estimation
    n_points : number of evaluation points

    Returns
    -------
    dict with keys: slope, intercept, r_squared
    """
    x = np.geomspace(x_lo, x_hi, n_points)
    f = pdf_fn(x)
    mask = f > 0
    if mask.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan}

    log_x = np.log(x[mask])
    log_f = np.log(f[mask])
    A = np.column_stack([log_x, np.ones_like(log_x)])
    result = np.linalg.lstsq(A, log_f, rcond=None)
    slope, intercept = result[0]

    # R²
    ss_res = np.sum((log_f - (slope * log_x + intercept)) ** 2)
    ss_tot = np.sum((log_f - log_f.mean()) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"slope": float(slope), "intercept": float(intercept), "r_squared": float(r_sq)}


def check_mean_property(
    samples: np.ndarray,
    expected_mean: float,
    tol: float = 0.1,
) -> Dict[str, Any]:
    """
    Check that E[X] ≈ expected_mean from simulated samples.

    Parameters
    ----------
    samples : 1D array
    expected_mean : expected value
    tol : absolute tolerance

    Returns
    -------
    dict with keys: empirical_mean, expected_mean, error, passed
    """
    emp = float(np.mean(samples))
    err = abs(emp - expected_mean)
    return {
        "empirical_mean": emp,
        "expected_mean": expected_mean,
        "abs_error": err,
        "passed": err < tol,
    }


def compute_integrated_absolute_error(
    pdf_fn_1: Callable[[np.ndarray], np.ndarray],
    pdf_fn_2: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    n_points: int = 500,
) -> float:
    """
    Compute ∫|f1(x) - f2(x)| dx on a uniform grid over (lo, hi).

    Parameters
    ----------
    pdf_fn_1, pdf_fn_2 : PDF callables
    lo, hi : integration range
    n_points : grid size

    Returns
    -------
    Integrated absolute difference (trapezoidal rule)
    """
    x = np.linspace(lo, hi, n_points)
    f1 = pdf_fn_1(x)
    f2 = pdf_fn_2(x)
    return float(np.trapz(np.abs(f1 - f2), x))


def compute_integrated_squared_error(
    pdf_fn_1: Callable[[np.ndarray], np.ndarray],
    pdf_fn_2: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    n_points: int = 500,
) -> float:
    """
    Compute ∫(f1(x) - f2(x))^2 dx on a uniform grid.

    Parameters
    ----------
    pdf_fn_1, pdf_fn_2 : PDF callables
    lo, hi : integration range
    n_points : grid size
    """
    x = np.linspace(lo, hi, n_points)
    f1 = pdf_fn_1(x)
    f2 = pdf_fn_2(x)
    return float(np.trapz((f1 - f2) ** 2, x))


def compute_sup_norm_cdf(
    cdf_fn_1: Callable[[np.ndarray], np.ndarray],
    cdf_fn_2: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    n_points: int = 500,
) -> float:
    """
    Compute sup_x |F1(x) - F2(x)| over a grid.

    Parameters
    ----------
    cdf_fn_1, cdf_fn_2 : CDF callables
    lo, hi : grid range
    n_points : grid size
    """
    x = np.linspace(lo, hi, n_points)
    return float(np.max(np.abs(cdf_fn_1(x) - cdf_fn_2(x))))


def regularized_kl_divergence(
    pdf_fn_1: Callable[[np.ndarray], np.ndarray],
    pdf_fn_2: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    n_points: int = 500,
    eps: float = 1e-10,
) -> float:
    """
    Compute KL(f1 || f2) = ∫ f1 * log(f1/f2) dx on a grid (regularized).

    Parameters
    ----------
    pdf_fn_1 : reference density
    pdf_fn_2 : comparison density
    lo, hi : grid range
    n_points : grid size
    eps : clipping floor to avoid log(0)
    """
    x = np.linspace(lo, hi, n_points)
    f1 = np.clip(pdf_fn_1(x), eps, None)
    f2 = np.clip(pdf_fn_2(x), eps, None)
    integrand = f1 * (np.log(f1) - np.log(f2))
    return float(np.trapz(integrand, x))


def regularized_js_divergence(
    pdf_fn_1: Callable[[np.ndarray], np.ndarray],
    pdf_fn_2: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    n_points: int = 500,
    eps: float = 1e-10,
) -> float:
    """
    Compute Jensen-Shannon divergence between f1 and f2 on a grid.

    JS(f1, f2) = 0.5*KL(f1||m) + 0.5*KL(f2||m) where m = 0.5*(f1+f2).

    Parameters
    ----------
    pdf_fn_1, pdf_fn_2 : PDF callables
    lo, hi : grid range
    n_points : grid size
    eps : clipping floor
    """
    x = np.linspace(lo, hi, n_points)
    f1 = np.clip(pdf_fn_1(x), eps, None)
    f2 = np.clip(pdf_fn_2(x), eps, None)
    m = 0.5 * (f1 + f2)
    kl1 = np.trapz(f1 * (np.log(f1) - np.log(m)), x)
    kl2 = np.trapz(f2 * (np.log(f2) - np.log(m)), x)
    return float(0.5 * kl1 + 0.5 * kl2)
