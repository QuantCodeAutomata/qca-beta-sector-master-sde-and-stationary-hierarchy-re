"""
Experiment 4: Noncommutativity and benchmark comparison.

Compares the dynamical-route generalized family (mGB) vs the distributional-route
family (mB_alpha) vs the traditional GB benchmark under identical parameters.
Tests the central structural claim that generalization and stationarization
do not commute for alpha != 1.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.parameters import BetaSectorParams, GeneralizedParams
from src.densities import (
    pdf_mB, pdf_mGB, pdf_mB_alpha, pdf_GB,
    pdf_GBP, pdf_GGa, pdf_GB1,
    pdf_BP_B2_alpha, pdf_Ga_alpha, pdf_B1_alpha,
    pdf_BP_B2, pdf_Ga, pdf_B1,
)
from src.cdfs import (
    cdf_mB, cdf_mGB, cdf_mB_alpha, cdf_GB,
)
from src.validation import (
    check_normalization, estimate_asymptotic_slope,
    compute_integrated_absolute_error, compute_integrated_squared_error,
    compute_sup_norm_cdf, regularized_js_divergence,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42

# Common parent parameters (fixed throughout)
PARENT = dict(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, kappa_2=0.8)
ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0]


def make_trio(alpha: float):
    """Return (gp, p, x_grid, upper) for the given alpha."""
    gp = GeneralizedParams(**PARENT, alpha=alpha)
    p = gp.as_beta_sector()
    upper = gp.beta_1 ** (1.0 / alpha)
    x = np.linspace(1e-6, upper - 1e-6, 500)
    return gp, p, x, upper


def compare_routes(alpha: float) -> dict:
    """Compute all comparison metrics for a given alpha."""
    gp, p, x, upper = make_trio(alpha)

    # PDFs on common grid
    try:
        f_mGB = pdf_mGB(x, gp)
    except Exception:
        f_mGB = np.full_like(x, np.nan)

    try:
        f_mBa = pdf_mB_alpha(x, p, alpha)
    except Exception:
        f_mBa = np.full_like(x, np.nan)

    try:
        f_GB = pdf_GB(x, gp)
    except Exception:
        f_GB = np.full_like(x, np.nan)

    f_mB = pdf_mB(x, p)

    # PDF differences
    def _iad(f1, f2):
        mask = np.isfinite(f1) & np.isfinite(f2)
        return float(np.trapz(np.abs(f1[mask] - f2[mask]), x[mask])) if mask.sum() > 1 else np.nan

    def _isd(f1, f2):
        mask = np.isfinite(f1) & np.isfinite(f2)
        return float(np.trapz((f1[mask] - f2[mask])**2, x[mask])) if mask.sum() > 1 else np.nan

    iad_dyn_dist = _iad(f_mGB, f_mBa)
    isd_dyn_dist = _isd(f_mGB, f_mBa)
    iad_dyn_GB   = _iad(f_mGB, f_GB)
    iad_dist_GB  = _iad(f_mBa, f_GB)

    # CDF sup-norm
    try:
        F_mGB = cdf_mGB(x, gp)
        F_mBa = cdf_mB_alpha(x, p, alpha)
        F_GB  = cdf_GB(x, gp)
        sup_dyn_dist = float(np.max(np.abs(F_mGB - F_mBa)))
        sup_dyn_GB   = float(np.max(np.abs(F_mGB - F_GB)))
    except Exception:
        sup_dyn_dist = np.nan
        sup_dyn_GB   = np.nan

    # Near-zero slopes
    try:
        slope_dyn = estimate_asymptotic_slope(
            lambda xv: pdf_mGB(xv, gp), x_lo=1e-6, x_hi=upper * 0.01)["slope"]
    except Exception:
        slope_dyn = np.nan

    try:
        slope_dist = estimate_asymptotic_slope(
            lambda xv: pdf_mB_alpha(xv, p, alpha), x_lo=1e-6, x_hi=upper * 0.01)["slope"]
    except Exception:
        slope_dist = np.nan

    expected_dyn = p.p + alpha - 2    # dynamical: x^(p+alpha-2)
    expected_dist = alpha * p.p - 1   # distributional: x^(alpha*p-1)

    # Normalization checks
    def _norm(pdf_fn):
        try:
            return check_normalization(pdf_fn, 1e-8, upper - 1e-8, tol=0.05)["integral"]
        except Exception:
            return np.nan

    return {
        "alpha": alpha,
        "norm_mGB": _norm(lambda xv: pdf_mGB(xv, gp)),
        "norm_mBa": _norm(lambda xv: pdf_mB_alpha(xv, p, alpha)),
        "norm_GB":  _norm(lambda xv: pdf_GB(xv, gp)),
        "iad_dyn_vs_dist": iad_dyn_dist,
        "isd_dyn_vs_dist": isd_dyn_dist,
        "iad_dyn_vs_GB":   iad_dyn_GB,
        "iad_dist_vs_GB":  iad_dist_GB,
        "sup_cdf_dyn_dist": sup_dyn_dist,
        "sup_cdf_dyn_GB":   sup_dyn_GB,
        "slope_dynamical":   slope_dyn,
        "expected_slope_dynamical": expected_dyn,
        "slope_distributional": slope_dist,
        "expected_slope_distributional": expected_dist,
        "routes_differ": iad_dyn_dist > 0.01 if not np.isnan(iad_dyn_dist) else None,
    }


def run_alpha1_equality() -> dict:
    """Verify mGB = mB_alpha = mB at alpha=1."""
    alpha = 1.0
    gp, p, x, upper = make_trio(alpha)

    f_mGB  = pdf_mGB(x, gp)
    f_mBa  = pdf_mB_alpha(x, p, alpha)
    f_mB_  = pdf_mB(x, p)

    mask = np.isfinite(f_mGB) & np.isfinite(f_mBa) & (f_mB_ > 1e-10)
    rel_diff_mGB_mB  = np.max(np.abs(f_mGB[mask] - f_mB_[mask]) / f_mB_[mask])
    rel_diff_mBa_mB  = np.max(np.abs(f_mBa[mask] - f_mB_[mask]) / f_mB_[mask])
    rel_diff_mGB_mBa = np.max(np.abs(f_mGB[mask] - f_mBa[mask]) / (np.abs(f_mBa[mask]) + 1e-12))

    return {
        "max_rel_diff_mGB_vs_mB":  float(rel_diff_mGB_mB),
        "max_rel_diff_mBa_vs_mB":  float(rel_diff_mBa_mB),
        "max_rel_diff_mGB_vs_mBa": float(rel_diff_mGB_mBa),
        "all_equal_at_alpha1": (rel_diff_mGB_mB < 0.05 and rel_diff_mBa_mB < 0.05),
    }


def plot_comparison(alpha: float) -> None:
    """Plot PDF and CDF overlays for mGB, mB_alpha, GB at a given alpha."""
    gp, p, x, upper = make_trio(alpha)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: PDF overlay
    ax = axes[0, 0]
    try:
        ax.plot(x, pdf_mGB(x, gp),      "b-",  lw=2,   label="mGB (dynamical)")
    except Exception:
        pass
    try:
        ax.plot(x, pdf_mB_alpha(x, p, alpha), "r--", lw=2, label="mB_alpha (distributional)")
    except Exception:
        pass
    try:
        ax.plot(x, pdf_GB(x, gp),       "g-.", lw=2,   label="GB (benchmark)")
    except Exception:
        pass
    ax.plot(x, pdf_mB(x, p), "k:", lw=1.5, label="mB (parent, alpha=1)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"PDF Comparison — alpha={alpha}")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # Panel 2: CDF overlay
    ax = axes[0, 1]
    try:
        ax.plot(x, cdf_mGB(x, gp),      "b-",  lw=2,   label="mGB CDF")
    except Exception:
        pass
    try:
        ax.plot(x, cdf_mB_alpha(x, p, alpha), "r--", lw=2, label="mB_alpha CDF")
    except Exception:
        pass
    try:
        ax.plot(x, cdf_GB(x, gp),       "g-.", lw=2,   label="GB CDF")
    except Exception:
        pass
    ax.plot(x, cdf_mB(x, p), "k:", lw=1.5, label="mB CDF (parent)")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_title(f"CDF Comparison — alpha={alpha}")
    ax.legend(fontsize=8)

    # Panel 3: PDF difference (mGB - mB_alpha)
    ax = axes[1, 0]
    try:
        f1 = pdf_mGB(x, gp)
        f2 = pdf_mB_alpha(x, p, alpha)
        ax.plot(x, f1 - f2, "purple", lw=2)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
    except Exception:
        pass
    ax.set_xlabel("x")
    ax.set_ylabel("f_mGB - f_mB_alpha")
    ax.set_title(f"PDF Difference (dynamical - distributional) — alpha={alpha}")

    # Panel 4: Log-log near zero
    ax = axes[1, 1]
    x_lo = np.geomspace(1e-6, upper * 0.2, 100)
    try:
        f_d = pdf_mGB(x_lo, gp)
        m = f_d > 0
        if np.any(m):
            ax.loglog(x_lo[m], f_d[m], "b-", lw=2, label=f"mGB slope~x^{p.p+alpha-2:.2f}")
    except Exception:
        pass
    try:
        f_p = pdf_mB_alpha(x_lo, p, alpha)
        m = f_p > 0
        if np.any(m):
            ax.loglog(x_lo[m], f_p[m], "r--", lw=2, label=f"mB_alpha slope~x^{alpha*p.p-1:.2f}")
    except Exception:
        pass
    ax.set_xlabel("x (log)")
    ax.set_ylabel("f(x) (log)")
    ax.set_title(f"Log-log near zero — alpha={alpha}")
    ax.legend(fontsize=8)

    plt.suptitle(f"Exp-4: Route comparison alpha={alpha}", fontsize=13)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"exp_4_comparison_alpha{alpha:.1f}.png")
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")


def plot_discrepancy_heatmap(rows: list) -> None:
    """Plot IAD(mGB, mB_alpha) as a function of alpha."""
    alphas = [r["alpha"] for r in rows if not np.isnan(r.get("iad_dyn_vs_dist", np.nan))]
    iads   = [r["iad_dyn_vs_dist"] for r in rows if not np.isnan(r.get("iad_dyn_vs_dist", np.nan))]

    if not alphas:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(alphas)), iads, tick_label=[f"{a:.1f}" for a in alphas], color="steelblue")
    ax.axvline(alphas.index(1.0) if 1.0 in alphas else -1, color="red", lw=2, ls="--",
               label="alpha=1 (routes should agree)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("IAD(mGB, mB_alpha)")
    ax.set_title("Exp-4: Noncommutativity — Integrated Absolute Difference")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "exp_4_iad_vs_alpha.png")
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")


def plot_slope_comparison(rows: list) -> None:
    """Compare dynamical vs distributional slopes across alpha values."""
    alphas = [r["alpha"] for r in rows]
    s_dyn  = [r.get("slope_dynamical", np.nan) for r in rows]
    s_dist = [r.get("slope_distributional", np.nan) for r in rows]
    e_dyn  = [r.get("expected_slope_dynamical", np.nan) for r in rows]
    e_dist = [r.get("expected_slope_distributional", np.nan) for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alphas, e_dyn,  "b-",  lw=2, label="Expected dynamic: p+alpha-2")
    ax.plot(alphas, e_dist, "r-",  lw=2, label="Expected distributional: alpha*p-1")
    ax.plot(alphas, s_dyn,  "b^--", ms=7, lw=1, label="Estimated dynamic")
    ax.plot(alphas, s_dist, "rs--", ms=7, lw=1, label="Estimated distributional")
    ax.axhline(-1, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Near-zero log-log slope")
    ax.set_title("Exp-4: Asymptotic slope comparison (routes separated for alpha != 1)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "exp_4_slope_comparison.png")
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")


def main() -> None:
    print("=" * 70)
    print("Experiment 4: Noncommutativity and benchmark comparison")
    print("=" * 70)

    # Alpha=1 equality test
    print("\n--- Alpha=1 equality check ---")
    eq = run_alpha1_equality()
    print(f"  mGB vs mB rel_diff: {eq['max_rel_diff_mGB_vs_mB']:.4f}")
    print(f"  mB_alpha vs mB rel_diff: {eq['max_rel_diff_mBa_vs_mB']:.4f}")
    print(f"  mGB vs mB_alpha rel_diff: {eq['max_rel_diff_mGB_vs_mBa']:.4f}")
    print(f"  All equal at alpha=1: {eq['all_equal_at_alpha1']}")

    rows = []
    for alpha in ALPHAS:
        print(f"\n--- alpha={alpha} ---")
        result = compare_routes(alpha)

        print(f"  Norm mGB={result['norm_mGB']:.4f},"
              f" mB_alpha={result['norm_mBa']:.4f},"
              f" GB={result['norm_GB']:.4f}")
        print(f"  IAD(mGB, mB_alpha)={result['iad_dyn_vs_dist']:.4f},"
              f" routes_differ={result['routes_differ']}")
        print(f"  Slope dynamical: est={result['slope_dynamical']:.3f},"
              f" expected={result['expected_slope_dynamical']:.3f}")
        print(f"  Slope distributional: est={result['slope_distributional']:.3f},"
              f" expected={result['expected_slope_distributional']:.3f}")
        print(f"  Sup-norm CDF(mGB, mB_alpha)={result['sup_cdf_dyn_dist']:.4f}")

        rows.append(result)

        try:
            plot_comparison(alpha)
        except Exception as e:
            print(f"  Plot error: {e}")

    # Summary plots
    plot_discrepancy_heatmap(rows)
    plot_slope_comparison(rows)

    # Print alpha=1 equality result in table
    rows.insert(0, {"alpha": "alpha=1 check", **eq})

    # Save
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "exp_4_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    print("\nExperiment 4 complete.")


if __name__ == "__main__":
    main()
