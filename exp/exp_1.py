"""
Experiment 1: Beta-sector master SDE and stationary hierarchy reproduction.

Validates the top-level mB stationary law, its CDF, the one-parameter reductions
to IGa1, BP/B2, and B1, and the terminal two-parameter limits to IGa and Ga.
Also verifies that simulated long-run distributions match the analytical stationary
laws, and checks E[x]=theta for the mB branch.

All results are saved to results/exp_1_*.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.parameters import BetaSectorParams
from src.densities import (
    pdf_mB, pdf_IGa1, pdf_BP_B2, pdf_B1, pdf_IGa, pdf_Ga,
)
from src.cdfs import (
    cdf_mB, cdf_IGa1, cdf_BP_B2, cdf_B1, cdf_IGa, cdf_Ga,
)
from src.simulation import (
    simulate_mB_sde, simulate_IGa1_sde, simulate_BP_B2_sde,
    simulate_B1_sde, simulate_IGa_sde, simulate_Ga_sde,
)
from src.validation import (
    check_normalization, check_cdf_endpoints, check_cdf_derivative,
    compute_ks_statistic, compute_cvm_statistic, estimate_asymptotic_slope,
    check_mean_property,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    "mB_base": dict(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, kappa_2=0.8),
    "mB_stress_beta2_small": dict(gamma=2.0, theta=0.1, kappa=0.5, kappa_1=0.4, kappa_2=1.2),
    "mB_high_gamma": dict(gamma=5.0, theta=0.2, kappa=1.0, kappa_1=0.6, kappa_2=0.7),
    "IGa1_base": dict(gamma=2.0, theta=0.3, kappa_1=0.5, kappa_2=0.8),
    "BP_B2_base": dict(gamma=2.0, theta=0.3, kappa=1.0, kappa_2=0.5),
    "B1_base": dict(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5),
    "IGa_base": dict(gamma=2.0, theta=0.3, kappa_2=0.5),
    "Ga_base": dict(gamma=2.0, theta=0.3, kappa=1.0),
}

BRANCH_CONFIG = {
    "mB":     {"pdf": pdf_mB,    "cdf": cdf_mB,    "sim": simulate_mB_sde,    "bounded": True},
    "IGa1":   {"pdf": pdf_IGa1,  "cdf": cdf_IGa1,  "sim": simulate_IGa1_sde,  "bounded": True},
    "BP_B2":  {"pdf": pdf_BP_B2, "cdf": cdf_BP_B2, "sim": simulate_BP_B2_sde, "bounded": False},
    "B1":     {"pdf": pdf_B1,    "cdf": cdf_B1,    "sim": simulate_B1_sde,    "bounded": True},
    "IGa":    {"pdf": pdf_IGa,   "cdf": cdf_IGa,   "sim": simulate_IGa_sde,   "bounded": False},
    "Ga":     {"pdf": pdf_Ga,    "cdf": cdf_Ga,    "sim": simulate_Ga_sde,    "bounded": False},
}

SCENARIO_BRANCH_MAP = {
    "mB_base":              "mB",
    "mB_stress_beta2_small":"mB",
    "mB_high_gamma":        "mB",
    "IGa1_base":            "IGa1",
    "BP_B2_base":           "BP_B2",
    "B1_base":              "B1",
    "IGa_base":             "IGa",
    "Ga_base":              "Ga",
}


def get_grid(p: BetaSectorParams, branch: str, n: int = 300) -> np.ndarray:
    """Return an interior x-grid for the given branch."""
    if branch in ("mB", "IGa1", "B1"):
        return np.linspace(1e-6, p.beta_1 - 1e-6, n)
    elif branch in ("BP_B2", "IGa", "Ga"):
        # Truncate at 99.9th percentile of analytical CDF
        cdf_fn = BRANCH_CONFIG[branch]["cdf"]
        # Use a conservative upper bound based on theta
        upper = 20 * p.theta
        return np.linspace(1e-6, upper, n)
    return np.linspace(1e-6, 10 * p.theta, n)


def run_analytical_checks(p: BetaSectorParams, branch: str) -> dict:
    """Run all analytical validation checks for a branch."""
    cfg = BRANCH_CONFIG[branch]
    pdf_fn = lambda x: cfg["pdf"](x, p)
    cdf_fn = lambda x: cfg["cdf"](x, p)

    # Support limits
    if branch in ("mB", "IGa1", "B1"):
        lo, hi = 1e-8, p.beta_1 - 1e-8
    else:
        lo, hi = 1e-8, 50 * p.theta

    norm = check_normalization(pdf_fn, lo, hi)
    cdf_ep = check_cdf_endpoints(cdf_fn, lo, hi)

    if branch in ("mB", "IGa1"):
        # Use tighter hi for derivative check
        cdf_deriv = check_cdf_derivative(pdf_fn, cdf_fn, lo, min(p.beta_1 * 0.95, hi))
    else:
        cdf_deriv = check_cdf_derivative(pdf_fn, cdf_fn, lo, hi)

    # Asymptotic slope near zero
    slope = estimate_asymptotic_slope(pdf_fn, x_lo=1e-6, x_hi=1e-3)

    return {
        "norm_integral": norm["integral"],
        "norm_error": norm["norm_error"],
        "norm_passed": norm["passed"],
        "cdf_lo": cdf_ep["F_lo"],
        "cdf_hi": cdf_ep["F_hi"],
        "cdf_ep_passed": cdf_ep["passed"],
        "cdf_deriv_passed": cdf_deriv["passed"],
        "cdf_deriv_mean_rel_err": cdf_deriv["mean_rel_err"],
        "slope_near_zero": slope["slope"],
    }


def run_simulation_check(p: BetaSectorParams, branch: str, n_sim: int = 2000) -> dict:
    """Run simulation and compare to analytical CDF."""
    cfg = BRANCH_CONFIG[branch]
    try:
        samples = cfg["sim"](p, n_samples=n_sim, T=200.0, T_burn=50.0, dt=0.005, seed=SEED)
    except Exception as e:
        return {"error": str(e)}

    cdf_fn = lambda x: cfg["cdf"](x, p)
    ks = compute_ks_statistic(samples, cdf_fn)
    cvm = compute_cvm_statistic(samples, cdf_fn)

    result = {
        "n_samples": len(samples),
        "sample_mean": float(np.mean(samples)),
        "ks_stat": ks["ks_stat"],
        "ks_pvalue": ks["ks_pvalue"],
        "cvm_stat": cvm["cvm_stat"],
    }

    # Check E[x] = theta for mB branch
    if branch == "mB":
        mean_check = check_mean_property(samples, p.theta, tol=0.1)
        result["mean_check_passed"] = mean_check["passed"]
        result["mean_error"] = mean_check["abs_error"]

    return result


def plot_branch(p: BetaSectorParams, branch: str, samples: np.ndarray,
                tag: str, expected_slope: float | None = None) -> None:
    """Plot PDF comparison (empirical histogram vs analytical)."""
    cfg = BRANCH_CONFIG[branch]
    pdf_fn = lambda x: cfg["pdf"](x, p)
    cdf_fn = lambda x: cfg["cdf"](x, p)

    x = get_grid(p, branch)
    f = pdf_fn(x)
    F = cdf_fn(x)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: PDF overlay
    ax = axes[0]
    ax.plot(x, f, "b-", lw=2, label="Analytical PDF")
    ax.hist(samples, bins=50, density=True, alpha=0.4, color="orange", label="Simulated")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(f"{branch} PDF — {tag}")
    ax.legend(fontsize=8)

    # Panel 2: CDF overlay
    ax = axes[1]
    ax.plot(x, F, "b-", lw=2, label="Analytical CDF")
    sorted_s = np.sort(samples)
    ecdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    ax.step(sorted_s, ecdf, color="orange", alpha=0.7, label="Empirical CDF", where="post")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_title(f"{branch} CDF — {tag}")
    ax.legend(fontsize=8)

    # Panel 3: Log-log near zero (asymptotic slope)
    ax = axes[2]
    mask = f > 0
    if np.sum(mask) > 2:
        ax.loglog(x[mask], f[mask], "b-", lw=2, label="Analytical PDF")
        if expected_slope is not None:
            x_ref = x[mask]
            f_ref = np.exp(np.log(f[mask][5]) + expected_slope * (np.log(x_ref) - np.log(x[mask][5])))
            ax.loglog(x_ref, f_ref, "r--", lw=1.5,
                      label=f"Slope {expected_slope:.2f}")
    ax.set_xlabel("x (log)")
    ax.set_ylabel("f(x) (log)")
    ax.set_title(f"{branch} log-log — {tag}")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"exp_1_{branch}_{tag}.png")
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")


def run_stress_asymptotic(p: BetaSectorParams) -> dict:
    """
    For mB with beta_2 << beta_1, check intermediate slope x^(-q-2).
    """
    branch = "mB"
    pdf_fn = lambda x: pdf_mB(x, p)
    # Near zero: slope should be p-1
    slope_lo = estimate_asymptotic_slope(pdf_fn, x_lo=1e-6, x_hi=p.beta_2 * 0.01)
    # Intermediate: beta_2 << x << beta_1: slope should be -q-2
    x_mid_lo = p.beta_2 * 5 if p.beta_2 < p.beta_1 * 0.1 else p.beta_2 * 0.1
    x_mid_hi = min(p.beta_1 * 0.5, x_mid_lo * 100)
    slope_mid = estimate_asymptotic_slope(pdf_fn, x_lo=x_mid_lo, x_hi=x_mid_hi)
    return {
        "slope_near_zero": slope_lo["slope"],
        "expected_slope_near_zero": p.p - 1,
        "slope_intermediate": slope_mid["slope"],
        "expected_slope_intermediate": -(p.q + 2),
    }


def main() -> None:
    print("=" * 70)
    print("Experiment 1: Beta-sector master SDE and stationary hierarchy")
    print("=" * 70)

    rows = []

    for scen_name, params_dict in SCENARIOS.items():
        branch = SCENARIO_BRANCH_MAP[scen_name]
        print(f"\n--- Scenario: {scen_name} (branch: {branch}) ---")

        try:
            p = BetaSectorParams(**params_dict)
        except Exception as e:
            print(f"  Parameter error: {e}")
            continue

        # Analytical checks
        try:
            an = run_analytical_checks(p, branch)
            print(f"  Normalization: {an['norm_integral']:.6f} (err={an['norm_error']:.2e},"
                  f" passed={an['norm_passed']})")
            print(f"  CDF endpoints: lo={an['cdf_lo']:.4f}, hi={an['cdf_hi']:.4f},"
                  f" passed={an['cdf_ep_passed']}")
            print(f"  CDF derivative: passed={an['cdf_deriv_passed']},"
                  f" mean_rel_err={an['cdf_deriv_mean_rel_err']:.3f}")
            print(f"  Near-zero slope: {an['slope_near_zero']:.3f}"
                  f" (expected p-1={p.p-1:.3f} if applicable)")
        except Exception as e:
            print(f"  Analytical check error: {e}")
            an = {}

        # Simulation check
        try:
            sim = run_simulation_check(p, branch)
            print(f"  KS stat: {sim.get('ks_stat','N/A'):.4f},"
                  f" p-value: {sim.get('ks_pvalue','N/A'):.4f}")
            print(f"  CvM stat: {sim.get('cvm_stat','N/A'):.4f}")
            if "mean_check_passed" in sim:
                print(f"  E[x]=theta check: passed={sim['mean_check_passed']},"
                      f" error={sim.get('mean_error','-'):.4f}")
        except Exception as e:
            print(f"  Simulation error: {e}")
            sim = {}

        # Plot
        try:
            cfg = BRANCH_CONFIG[branch]
            samples = cfg["sim"](p, n_samples=2000, T=200.0, T_burn=50.0, dt=0.005, seed=SEED)
            exp_slope = p.p - 1 if not np.isnan(p.p) else None
            plot_branch(p, branch, samples, scen_name.replace(":", "_"), exp_slope)
        except Exception as e:
            print(f"  Plot error: {e}")

        row = {"scenario": scen_name, "branch": branch}
        row.update({k: v for k, v in an.items() if isinstance(v, (int, float, bool))})
        row.update({k: v for k, v in sim.items() if isinstance(v, (int, float, bool))})
        rows.append(row)

    # Stress asymptotic check for mB with beta_2 << beta_1
    print("\n--- Asymptotic stress check: mB with beta_2 << beta_1 ---")
    try:
        p_stress = BetaSectorParams(gamma=2.0, theta=0.05, kappa=0.5, kappa_1=0.3, kappa_2=2.0)
        print(f"  beta_1={p_stress.beta_1:.3f}, beta_2={p_stress.beta_2:.3f}, p={p_stress.p:.3f}, q={p_stress.q:.3f}")
        stress_res = run_stress_asymptotic(p_stress)
        print(f"  Slope near zero: {stress_res['slope_near_zero']:.3f}"
              f" (expected {stress_res['expected_slope_near_zero']:.3f})")
        print(f"  Slope intermediate: {stress_res['slope_intermediate']:.3f}"
              f" (expected {stress_res['expected_slope_intermediate']:.3f})")
        rows.append({"scenario": "mB_stress_asym", "branch": "mB", **stress_res})
    except Exception as e:
        print(f"  Stress error: {e}")

    # Reduction tree verification (parameter check)
    print("\n--- Reduction tree verification ---")
    # B1 from mB with kappa_2=0
    p_b1 = BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5)
    print(f"  B1: p={p_b1.p:.3f}, q={p_b1.q:.3f}, beta_1={p_b1.beta_1:.3f}")
    # BP/B2 from mB with kappa_1=0
    p_bp = BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_2=0.5)
    print(f"  BP/B2: p={p_bp.p:.3f}, q={p_bp.q:.3f}, beta_2={p_bp.beta_2:.3f}")
    # IGa1 from mB with kappa=0
    p_iga1 = BetaSectorParams(gamma=2.0, theta=0.3, kappa_1=0.5, kappa_2=0.8)
    print(f"  IGa1: beta={p_iga1.beta:.3f}, beta_1={p_iga1.beta_1:.3f}")
    # Ga from B1 with kappa_1=0
    p_ga = BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0)
    print(f"  Ga: p={p_ga.p:.3f}, lam={p_ga.lam:.3f} (expected theta/p={p_ga.theta/p_ga.p:.3f})")
    # IGa terminal
    p_iga = BetaSectorParams(gamma=2.0, theta=0.3, kappa_2=0.5)
    print(f"  IGa: beta={p_iga.beta:.3f}")

    # Save results
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(RESULTS_DIR, "exp_1_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    # Summary plot: PDF grid
    print("\nGenerating summary plot...")
    _plot_summary_grid()

    print("\nExperiment 1 complete.")


def _plot_summary_grid() -> None:
    """Plot a 2x3 grid of all branch PDFs for a standard parameter set."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    branches_params = [
        ("mB",    BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, kappa_2=0.8)),
        ("IGa1",  BetaSectorParams(gamma=2.0, theta=0.3, kappa_1=0.5, kappa_2=0.8)),
        ("BP_B2", BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_2=0.5)),
        ("B1",    BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5)),
        ("IGa",   BetaSectorParams(gamma=2.0, theta=0.3, kappa_2=0.5)),
        ("Ga",    BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0)),
    ]

    for ax, (branch, p) in zip(axes.flat, branches_params):
        cfg = BRANCH_CONFIG[branch]
        x = get_grid(p, branch, n=300)
        f = cfg["pdf"](x, p)
        ax.plot(x, f, "b-", lw=2)
        ax.set_title(f"{branch}\n(gamma={p.gamma}, theta={p.theta})")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_ylim(bottom=0)

    plt.suptitle("Exp-1: Beta-sector stationary PDFs", fontsize=14)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "exp_1_summary_grid.png")
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")


if __name__ == "__main__":
    main()
