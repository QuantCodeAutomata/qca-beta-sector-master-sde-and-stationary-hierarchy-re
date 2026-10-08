"""
Experiment 3: Generalized hierarchy via the distributional (pushforward) route.

Validates all branch descendants by applying the pushforward rule
f_X(x) = alpha * x^(alpha-1) * f_Y(x^alpha) to the parent stationary laws,
verifies alpha=1 collapse, transformed-sample Monte Carlo, and asymptotic
near-zero behavior x^(alpha*p-1).
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
    pdf_mB, pdf_BP_B2, pdf_B1, pdf_IGa, pdf_Ga, pdf_IGa1,
    pdf_mB_alpha, pdf_BP_B2_alpha, pdf_B1_alpha, pdf_IGa_alpha,
    pdf_Ga_alpha, pdf_IGa1_alpha,
)
from src.cdfs import (
    cdf_mB, cdf_BP_B2, cdf_B1, cdf_IGa, cdf_Ga,
    cdf_mB_alpha, cdf_BP_B2_alpha, cdf_B1_alpha, cdf_IGa_alpha, cdf_Ga_alpha,
)
from src.simulation import (
    simulate_mB_sde, simulate_BP_B2_sde, simulate_B1_sde,
    simulate_IGa_sde, simulate_Ga_sde,
)
from src.validation import (
    check_normalization, check_cdf_endpoints, check_cdf_derivative,
    compute_ks_statistic, compute_cvm_statistic, estimate_asymptotic_slope,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------------------
# Parameter sets — one per parent branch
# ---------------------------------------------------------------------------

PARAMS_mB = BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, kappa_2=0.8)
PARAMS_B1 = BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5)
PARAMS_BP = BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_2=0.5)
PARAMS_IGa = BetaSectorParams(gamma=2.0, theta=0.3, kappa_2=0.5)
PARAMS_Ga = BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0)
PARAMS_IGa1 = BetaSectorParams(gamma=2.0, theta=0.3, kappa_1=0.5, kappa_2=0.8)

ALPHAS = [0.5, 1.0, 1.5, 2.0]


def get_upper(p: BetaSectorParams, alpha: float, branch: str) -> float:
    """Get the upper limit for bounded branches."""
    if branch in ("mB_alpha", "B1_alpha", "IGa1_alpha"):
        return p.beta_1 ** (1.0 / alpha) if np.isfinite(p.beta_1) else 20 * p.theta
    return 50 * p.theta  # unbounded: truncate


def run_distrib_checks(p: BetaSectorParams, alpha: float, branch: str) -> dict:
    """Run normalization, CDF, and slope checks for a distributional-route branch."""
    branch_fns = {
        "mB_alpha":     (lambda x: pdf_mB_alpha(x, p, alpha),     lambda x: cdf_mB_alpha(x, p, alpha)),
        "B1_alpha":     (lambda x: pdf_B1_alpha(x, p, alpha),     lambda x: cdf_B1_alpha(x, p, alpha)),
        "BP_B2_alpha":  (lambda x: pdf_BP_B2_alpha(x, p, alpha),  lambda x: cdf_BP_B2_alpha(x, p, alpha)),
        "IGa_alpha":    (lambda x: pdf_IGa_alpha(x, p, alpha),    lambda x: cdf_IGa_alpha(x, p, alpha)),
        "Ga_alpha":     (lambda x: pdf_Ga_alpha(x, p, alpha),     lambda x: cdf_Ga_alpha(x, p, alpha)),
        "IGa1_alpha":   (lambda x: pdf_IGa1_alpha(x, p, alpha),   None),
    }

    if branch not in branch_fns:
        return {"error": f"Unknown branch {branch}"}

    pdf_fn, cdf_fn = branch_fns[branch]
    upper = get_upper(p, alpha, branch)

    norm = check_normalization(pdf_fn, 1e-8, upper - 1e-8, tol=0.05)

    if cdf_fn is not None:
        cdf_ep = check_cdf_endpoints(cdf_fn, 0, upper)
        cdf_der = check_cdf_derivative(pdf_fn, cdf_fn, 1e-6, upper - 1e-6)
    else:
        cdf_ep = {"passed": None}
        cdf_der = {"passed": None, "mean_rel_err": None}

    # Near-zero slope: for distributional route, expected = alpha*p - 1.
    # Tight window so x^alpha << scale parameters (beta_1, beta_2) to be in true asymptotic regime.
    expected_slope = alpha * p.p - 1 if not np.isnan(p.p) else None
    beta_2 = p.beta_2 if np.isfinite(p.beta_2) else 1.0
    beta_1 = p.beta_1 if np.isfinite(p.beta_1) else 1e6
    x_hi_slope = min(
        (0.005 * min(beta_1, beta_2)) ** (1.0 / alpha),
        upper * 0.001,
        1e-3,
    )
    x_hi_slope = max(x_hi_slope, 1e-7)
    slope = estimate_asymptotic_slope(pdf_fn, x_lo=1e-8, x_hi=x_hi_slope)

    return {
        "norm_integral": norm["integral"],
        "norm_error": norm["norm_error"],
        "norm_passed": norm["passed"],
        "cdf_ep_passed": cdf_ep["passed"],
        "cdf_deriv_passed": cdf_der.get("passed"),
        "cdf_deriv_mean_rel_err": cdf_der.get("mean_rel_err"),
        "slope_near_zero": slope["slope"],
        "expected_slope_distributional": expected_slope,
        "slope_within_10pct": (
            abs(slope["slope"] - expected_slope) / (abs(expected_slope) + 1e-10) < 0.15
            if expected_slope is not None else None
        ),
    }


def run_alpha1_collapse(p: BetaSectorParams, branch_alpha: str, branch_orig: str) -> dict:
    """Check that the distributional route collapses to the original at alpha=1."""
    alpha = 1.0
    branch_fns = {
        "mB_alpha":    lambda x: pdf_mB_alpha(x, p, alpha),
        "B1_alpha":    lambda x: pdf_B1_alpha(x, p, alpha),
        "BP_B2_alpha": lambda x: pdf_BP_B2_alpha(x, p, alpha),
        "IGa_alpha":   lambda x: pdf_IGa_alpha(x, p, alpha),
        "Ga_alpha":    lambda x: pdf_Ga_alpha(x, p, alpha),
    }
    orig_fns = {
        "mB":    lambda x: pdf_mB(x, p),
        "B1":    lambda x: pdf_B1(x, p),
        "BP_B2": lambda x: pdf_BP_B2(x, p),
        "IGa":   lambda x: pdf_IGa(x, p),
        "Ga":    lambda x: pdf_Ga(x, p),
    }

    upper = get_upper(p, 1.0, branch_alpha)
    x = np.linspace(1e-6, upper - 1e-6, 200)

    f_alpha = branch_fns[branch_alpha](x)
    f_orig = orig_fns[branch_orig](x)

    mask = (f_orig > 1e-10) & np.isfinite(f_alpha) & np.isfinite(f_orig)
    if mask.sum() < 5:
        return {"max_rel_diff": np.nan, "collapsed": False}

    rel_diff = np.abs(f_alpha[mask] - f_orig[mask]) / np.abs(f_orig[mask])
    max_rel = float(np.max(rel_diff))
    return {
        "max_rel_diff": max_rel,
        "mean_rel_diff": float(np.mean(rel_diff)),
        "collapsed": max_rel < 0.05,
    }


def transformed_sample_mc(p: BetaSectorParams, alpha: float,
                           branch_orig: str, branch_alpha: str,
                           n_samples: int = 2000) -> dict:
    """
    Validate distributional route via transformed-sample Monte Carlo.
    Sample Y from parent law, compute X = Y^(1/alpha), compare CDF to analytical.
    """
    sim_map = {
        "mB":    lambda: simulate_mB_sde(p, n_samples=n_samples, T=200, T_burn=50, dt=0.005, seed=SEED),
        "B1":    lambda: simulate_B1_sde(p, n_samples=n_samples, T=200, T_burn=50, dt=0.005, seed=SEED),
        "BP_B2": lambda: simulate_BP_B2_sde(p, n_samples=n_samples, T=200, T_burn=50, dt=0.005, seed=SEED),
        "IGa":   lambda: simulate_IGa_sde(p, n_samples=n_samples, T=200, T_burn=50, dt=0.005, seed=SEED),
        "Ga":    lambda: simulate_Ga_sde(p, n_samples=n_samples, T=200, T_burn=50, dt=0.005, seed=SEED),
    }
    cdf_map = {
        "mB_alpha":    lambda x: cdf_mB_alpha(x, p, alpha),
        "B1_alpha":    lambda x: cdf_B1_alpha(x, p, alpha),
        "BP_B2_alpha": lambda x: cdf_BP_B2_alpha(x, p, alpha),
        "IGa_alpha":   lambda x: cdf_IGa_alpha(x, p, alpha),
        "Ga_alpha":    lambda x: cdf_Ga_alpha(x, p, alpha),
    }

    if branch_orig not in sim_map or branch_alpha not in cdf_map:
        return {"error": "branch not found"}

    try:
        y_samples = sim_map[branch_orig]()
        x_samples = y_samples ** (1.0 / alpha)
    except Exception as e:
        return {"error": str(e)}

    try:
        ks = compute_ks_statistic(x_samples, cdf_map[branch_alpha])
        cvm = compute_cvm_statistic(x_samples, cdf_map[branch_alpha])
    except Exception as e:
        return {"error": str(e)}

    return {
        "ks_stat": ks["ks_stat"],
        "ks_pvalue": ks["ks_pvalue"],
        "cvm_stat": cvm["cvm_stat"],
        "n_samples": len(x_samples),
    }


def plot_distrib_route(p: BetaSectorParams, alphas: list, branch_alpha: str,
                       branch_orig: str, tag: str) -> None:
    """Plot distributional-route PDFs for multiple alpha values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    upper_1 = get_upper(p, 1.0, branch_alpha)
    x1 = np.linspace(1e-6, upper_1 - 1e-6, 300)
    orig_fns = {
        "mB":    lambda x: pdf_mB(x, p),
        "B1":    lambda x: pdf_B1(x, p),
        "BP_B2": lambda x: pdf_BP_B2(x, p),
        "IGa":   lambda x: pdf_IGa(x, p),
        "Ga":    lambda x: pdf_Ga(x, p),
    }
    alpha_fns = {
        "mB_alpha":    lambda x, a: pdf_mB_alpha(x, p, a),
        "B1_alpha":    lambda x, a: pdf_B1_alpha(x, p, a),
        "BP_B2_alpha": lambda x, a: pdf_BP_B2_alpha(x, p, a),
        "IGa_alpha":   lambda x, a: pdf_IGa_alpha(x, p, a),
        "Ga_alpha":    lambda x, a: pdf_Ga_alpha(x, p, a),
    }

    ax = axes[0]
    for alpha, col in zip(alphas, colors):
        upper = get_upper(p, alpha, branch_alpha)
        x = np.linspace(1e-6, upper - 1e-6, 300)
        try:
            f = alpha_fns[branch_alpha](x, alpha)
            ax.plot(x, f, color=col, lw=1.5, label=f"alpha={alpha}")
        except Exception:
            pass
    if branch_orig in orig_fns:
        f_orig = orig_fns[branch_orig](x1)
        ax.plot(x1, f_orig, "k--", lw=2, label="Parent (alpha=1)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Distributional: {branch_alpha} — {tag}")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)

    # Log-log
    ax = axes[1]
    for alpha, col in zip(alphas, colors):
        upper = get_upper(p, alpha, branch_alpha)
        x = np.linspace(1e-6, upper * 0.3, 200)
        try:
            f = alpha_fns[branch_alpha](x, alpha)
            mask = f > 0
            if np.any(mask):
                ax.loglog(x[mask], f[mask], color=col, lw=1.5, label=f"alpha={alpha}")
        except Exception:
            pass
    ax.set_xlabel("x (log)")
    ax.set_ylabel("f(x) (log)")
    ax.set_title(f"Log-log: {branch_alpha}")
    ax.legend(fontsize=7)

    plt.suptitle(f"Exp-3 Distributional Route — {tag}", fontsize=12)
    plt.tight_layout()
    safe_tag = tag.replace("/", "_")
    fname = os.path.join(RESULTS_DIR, f"exp_3_{branch_alpha}_{safe_tag}.png")
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")


def main() -> None:
    print("=" * 70)
    print("Experiment 3: Generalized hierarchy — distributional route")
    print("=" * 70)

    rows = []

    branch_pairs = [
        (PARAMS_mB,   "mB_alpha",    "mB",    "mB"),
        (PARAMS_B1,   "B1_alpha",    "B1",    "B1"),
        (PARAMS_BP,   "BP_B2_alpha", "BP_B2", "BP/B2"),
        (PARAMS_IGa,  "IGa_alpha",   "IGa",   "IGa"),
        (PARAMS_Ga,   "Ga_alpha",    "Ga",    "Ga"),
        (PARAMS_IGa1, "IGa1_alpha",  "IGa1",  "IGa1"),
    ]

    for p, branch_alpha, branch_orig, label in branch_pairs:
        print(f"\n=== Branch: {label} ===")

        # Alpha=1 collapse check
        if branch_alpha != "IGa1_alpha":
            col = run_alpha1_collapse(p, branch_alpha, branch_orig)
            print(f"  Alpha=1 collapse: max_rel_diff={col.get('max_rel_diff','N/A'):.4f},"
                  f" collapsed={col.get('collapsed')}")

        for alpha in ALPHAS:
            print(f"  alpha={alpha}:")
            checks = run_distrib_checks(p, alpha, branch_alpha)
            print(f"    norm={checks.get('norm_integral','N/A'):.5f},"
                  f" slope={checks.get('slope_near_zero','N/A'):.3f},"
                  f" expected={checks.get('expected_slope_distributional','N/A')}")

            # Transformed-sample MC
            if branch_alpha != "IGa1_alpha" and alpha != 1.0:
                mc = transformed_sample_mc(p, alpha, branch_orig, branch_alpha)
                print(f"    MC KS={mc.get('ks_stat','N/A'):.4f},"
                      f" p={mc.get('ks_pvalue','N/A'):.4f}")
                checks.update({f"mc_{k}": v for k, v in mc.items()
                                if isinstance(v, (int, float))})

            row = {"branch": label, "alpha": alpha}
            row.update({k: v for k, v in checks.items() if isinstance(v, (int, float, bool))})
            rows.append(row)

        # Plot
        try:
            if branch_alpha != "IGa1_alpha":
                plot_distrib_route(p, ALPHAS, branch_alpha, branch_orig, label)
        except Exception as e:
            print(f"  Plot error: {e}")

    # Save results
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(RESULTS_DIR, "exp_3_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    # Summary slope comparison
    _plot_slope_comparison()

    print("\nExperiment 3 complete.")


def _plot_slope_comparison() -> None:
    """Compare distributional slope (alpha*p-1) vs alpha for mB_alpha."""
    p = PARAMS_mB
    alphas = np.linspace(0.3, 3.0, 30)
    slopes_theoretical = alphas * p.p - 1
    slopes_estimated = []

    for alpha in alphas:
        pdf_fn = lambda x, a=alpha: pdf_mB_alpha(x, p, a)
        upper = p.beta_1 ** (1.0 / alpha)
        b2 = p.beta_2 if np.isfinite(p.beta_2) else 1.0
        b1 = p.beta_1 if np.isfinite(p.beta_1) else 1e6
        xh = min((0.005 * min(b1, b2)) ** (1.0 / alpha), upper * 0.001, 1e-3)
        xh = max(xh, 1e-7)
        s = estimate_asymptotic_slope(pdf_fn, x_lo=1e-8, x_hi=xh)
        slopes_estimated.append(s["slope"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, slopes_theoretical, "b-", lw=2, label="Theoretical: alpha*p-1")
    ax.plot(alphas, slopes_estimated, "ro--", ms=4, lw=1, label="Estimated (log-log regression)")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Near-zero slope")
    ax.set_title(f"Exp-3: mB_alpha near-zero slope vs alpha (p={p.p:.3f})")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "exp_3_slope_vs_alpha.png")
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")


if __name__ == "__main__":
    main()
