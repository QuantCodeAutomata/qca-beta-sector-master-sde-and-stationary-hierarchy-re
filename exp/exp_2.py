"""
Experiment 2: Generalized hierarchy via the dynamical route.

Validates the top-level mGB stationary law, branch descendants (GBP, GB1,
GIGa1, GIGa, GGa), alpha=1 collapse back to original Beta-sector, and
Monte Carlo validation of the generalized SDE.
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
    pdf_mB, pdf_BP_B2, pdf_B1, pdf_IGa, pdf_Ga,
    pdf_mGB, pdf_GBP, pdf_GB1, pdf_GIGa, pdf_GGa, pdf_GIGa1,
)
from src.cdfs import (
    cdf_mB, cdf_BP_B2, cdf_B1, cdf_IGa, cdf_Ga,
    cdf_mGB, cdf_GBP, cdf_GB1, cdf_GIGa, cdf_GGa,
)
from src.simulation import simulate_generalized_sde
from src.validation import (
    check_normalization, check_cdf_endpoints, check_cdf_derivative,
    compute_ks_statistic, compute_cvm_statistic, estimate_asymptotic_slope,
    compute_integrated_absolute_error,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS_mGB = [
    dict(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, kappa_2=0.8, alpha=1.0),
    dict(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, kappa_2=0.8, alpha=0.5),
    dict(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, kappa_2=0.8, alpha=2.0),
    dict(gamma=3.0, theta=0.2, kappa=0.8, kappa_1=0.4, kappa_2=0.9, alpha=1.5),
]

SCENARIOS_GBP = [
    dict(gamma=2.0, theta=0.3, kappa=1.0, kappa_2=0.5, alpha=1.0),
    dict(gamma=2.0, theta=0.3, kappa=1.0, kappa_2=0.5, alpha=2.0),
]

SCENARIOS_GB1 = [
    dict(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, alpha=1.0),
    dict(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, alpha=2.0),
]

SCENARIOS_GGa = [
    dict(gamma=2.0, theta=0.3, kappa=1.0, alpha=1.0),
    dict(gamma=2.0, theta=0.3, kappa=1.0, alpha=2.0),
]

SCENARIOS_GIGa = [
    dict(gamma=2.0, theta=0.3, kappa_2=0.5, alpha=1.0),
    dict(gamma=2.0, theta=0.3, kappa_2=0.5, alpha=2.0),
]


def make_gp(d: dict) -> GeneralizedParams:
    return GeneralizedParams(**d)


def make_bp(d: dict) -> BetaSectorParams:
    """Alpha=1 projection."""
    d1 = {k: v for k, v in d.items() if k != "alpha"}
    return BetaSectorParams(**d1)


def run_mGB_checks(gp: GeneralizedParams) -> dict:
    """Run all checks for the mGB branch."""
    tag = f"alpha={gp.alpha:.2f}"
    upper = gp.beta_1 ** (1.0 / gp.alpha)

    print(f"  u={gp.u:.4f}, v={gp.v:.4f}, w={gp.w:.4f}, integrability_ok={gp.check_integrability()}")

    if not gp.check_integrability():
        return {"integrability": False}

    pdf_fn = lambda x: pdf_mGB(x, gp)
    cdf_fn = lambda x: cdf_mGB(x, gp)

    norm = check_normalization(pdf_fn, 1e-8, upper - 1e-8, tol=0.05)
    cdf_ep = check_cdf_endpoints(cdf_fn, 0, upper)
    cdf_der = check_cdf_derivative(pdf_fn, cdf_fn, 1e-6, upper - 1e-6)

    # Near-zero window: require x^alpha << beta_2 so correction terms are negligible.
    # x_hi = (0.005 * beta_2)^(1/alpha) but capped at min(1e-3, upper*0.001).
    beta_2 = gp.beta_2 if np.isfinite(gp.beta_2) else 1.0
    x_hi_slope = min((0.005 * beta_2) ** (1.0 / gp.alpha), upper * 0.001, 1e-3)
    x_hi_slope = max(x_hi_slope, 1e-6)
    slope = estimate_asymptotic_slope(pdf_fn, x_lo=1e-8, x_hi=x_hi_slope)
    expected_slope = gp.p + gp.alpha - 2  # dynamical route: x^(p+alpha-2)

    return {
        "integrability": True,
        "norm_integral": norm["integral"],
        "norm_error": norm["norm_error"],
        "norm_passed": norm["passed"],
        "cdf_ep_passed": cdf_ep["passed"],
        "cdf_deriv_passed": cdf_der["passed"],
        "slope_near_zero": slope["slope"],
        "expected_slope_dynamical": expected_slope,
    }


def run_alpha1_collapse_check(gp: GeneralizedParams, branch_gen: str, branch_orig: str) -> dict:
    """Verify that a generalized branch collapses to the original at alpha=1."""
    if abs(gp.alpha - 1.0) > 1e-10:
        return {"skipped": True, "reason": "alpha != 1"}

    bp = gp.as_beta_sector()
    upper = bp.beta_1 if np.isfinite(bp.beta_1) else 20 * bp.theta
    lo = 1e-6
    hi = upper - 1e-6 if np.isfinite(upper) else 20 * bp.theta

    x = np.linspace(lo, hi, 100)

    gen_pdfs = {
        "mGB": lambda x: pdf_mGB(x, gp),
        "GBP": lambda x: pdf_GBP(x, gp),
        "GB1": lambda x: pdf_GB1(x, gp),
        "GIGa": lambda x: pdf_GIGa(x, gp),
        "GGa": lambda x: pdf_GGa(x, gp),
    }
    orig_pdfs = {
        "mB": lambda x: pdf_mB(x, bp),
        "BP_B2": lambda x: pdf_BP_B2(x, bp),
        "B1": lambda x: pdf_B1(x, bp),
        "IGa": lambda x: pdf_IGa(x, bp),
        "Ga": lambda x: pdf_Ga(x, bp),
    }

    if branch_gen not in gen_pdfs or branch_orig not in orig_pdfs:
        return {"skipped": True, "reason": "branch not found"}

    f_gen = gen_pdfs[branch_gen](x)
    f_orig = orig_pdfs[branch_orig](x)

    # Relative difference
    mask = (f_orig > 1e-10) & np.isfinite(f_gen) & np.isfinite(f_orig)
    if mask.sum() < 5:
        return {"max_rel_diff": np.nan, "collapsed": False}

    rel_diff = np.abs(f_gen[mask] - f_orig[mask]) / np.abs(f_orig[mask])
    max_rel = float(np.max(rel_diff))

    return {
        "max_rel_diff": max_rel,
        "mean_rel_diff": float(np.mean(rel_diff)),
        "collapsed": max_rel < 0.05,
    }


def run_simulation_check_gen(gp: GeneralizedParams, branch: str) -> dict:
    """Simulate generalized SDE and compare to analytical."""
    cdf_map = {
        "mGB": lambda x: cdf_mGB(x, gp),
        "GBP": lambda x: cdf_GBP(x, gp),
        "GB1": lambda x: cdf_GB1(x, gp),
        "GGa": lambda x: cdf_GGa(x, gp),
        "GIGa": lambda x: cdf_GIGa(x, gp),
    }

    try:
        samples = simulate_generalized_sde(
            gp, branch=branch,
            n_samples=2000, T=200.0, T_burn=50.0, dt=0.005, seed=SEED
        )
    except Exception as e:
        return {"error": str(e)}

    if branch not in cdf_map:
        return {"error": f"No CDF for {branch}"}

    try:
        ks = compute_ks_statistic(samples, cdf_map[branch])
        cvm = compute_cvm_statistic(samples, cdf_map[branch])
    except Exception as e:
        return {"error": str(e)}

    return {
        "n_samples": len(samples),
        "sample_mean": float(np.mean(samples)),
        "ks_stat": ks["ks_stat"],
        "ks_pvalue": ks["ks_pvalue"],
        "cvm_stat": cvm["cvm_stat"],
    }


def plot_branch_comparison(gp_alpha: GeneralizedParams, gp_1: GeneralizedParams,
                           branch_gen: str, branch_orig: str, tag: str) -> None:
    """Plot generalized vs alpha=1 original PDF."""
    bp = gp_1.as_beta_sector()
    upper = bp.beta_1 if np.isfinite(bp.beta_1) else 20 * bp.theta
    x = np.linspace(1e-6, min(upper - 1e-6, 20 * bp.theta), 300)

    gen_pdf_map = {
        "mGB": lambda x: pdf_mGB(x, gp_alpha),
        "GBP": lambda x: pdf_GBP(x, gp_alpha),
        "GB1": lambda x: pdf_GB1(x, gp_alpha),
        "GGa": lambda x: pdf_GGa(x, gp_alpha),
        "GIGa": lambda x: pdf_GIGa(x, gp_alpha),
    }
    orig_pdf_map = {
        "mB": lambda x: pdf_mB(x, bp),
        "BP_B2": lambda x: pdf_BP_B2(x, bp),
        "B1": lambda x: pdf_B1(x, bp),
        "Ga": lambda x: pdf_Ga(x, bp),
        "IGa": lambda x: pdf_IGa(x, bp),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # PDF overlay
    ax = axes[0]
    if branch_gen in gen_pdf_map:
        x_gen = np.linspace(1e-6, gp_alpha.beta_1 ** (1 / gp_alpha.alpha) * 0.999, 300)
        f_gen = gen_pdf_map[branch_gen](x_gen)
        ax.plot(x_gen, f_gen, "b-", lw=2, label=f"{branch_gen} (alpha={gp_alpha.alpha})")
    if branch_orig in orig_pdf_map:
        f_orig = orig_pdf_map[branch_orig](x)
        ax.plot(x, f_orig, "r--", lw=2, label=f"{branch_orig} (alpha=1)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Dynamical route: {branch_gen} vs {branch_orig}")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # Log-log
    ax = axes[1]
    if branch_gen in gen_pdf_map:
        x_gen = np.linspace(1e-5, gp_alpha.beta_1 ** (1 / gp_alpha.alpha) * 0.5, 200)
        f_gen = gen_pdf_map[branch_gen](x_gen)
        mask = f_gen > 0
        if np.any(mask):
            ax.loglog(x_gen[mask], f_gen[mask], "b-", lw=2, label=f"{branch_gen} alpha={gp_alpha.alpha}")
    if branch_orig in orig_pdf_map:
        f_orig = orig_pdf_map[branch_orig](x)
        mask = f_orig > 0
        if np.any(mask):
            ax.loglog(x[mask], f_orig[mask], "r--", lw=2, label=f"{branch_orig} alpha=1")
    ax.set_xlabel("x (log)")
    ax.set_ylabel("f(x) (log)")
    ax.set_title("Log-log comparison")
    ax.legend(fontsize=8)

    plt.suptitle(f"Exp-2 Dynamical Route — {tag}", fontsize=12)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"exp_2_{branch_gen}_{tag}.png")
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")


def main() -> None:
    print("=" * 70)
    print("Experiment 2: Generalized hierarchy — dynamical route")
    print("=" * 70)

    rows = []

    # ---- mGB branch ----
    print("\n=== mGB branch ===")
    alpha1_gp = None
    for i, d in enumerate(SCENARIOS_mGB):
        gp = make_gp(d)
        print(f"\n  mGB scenario {i+1}: alpha={gp.alpha}, p={gp.p:.3f}, q={gp.q:.3f}")
        checks = run_mGB_checks(gp)
        print(f"  Checks: norm={checks.get('norm_integral','N/A'):.5f},"
              f" slope={checks.get('slope_near_zero','N/A'):.3f},"
              f" expected={checks.get('expected_slope_dynamical','N/A'):.3f}")

        if gp.alpha == 1.0:
            alpha1_gp = gp
            # Alpha=1 collapse
            collapse = run_alpha1_collapse_check(gp, "mGB", "mB")
            print(f"  Alpha=1 collapse mGB->mB: {collapse}")
            checks["collapse_mGB_mB"] = collapse.get("collapsed", False)

        # Simulation
        if checks.get("integrability", True):
            sim = run_simulation_check_gen(gp, "mGB")
            print(f"  Sim: KS={sim.get('ks_stat','N/A'):.4f}, p={sim.get('ks_pvalue','N/A'):.4f}")
            checks.update(sim)

        row = {"branch": "mGB", "alpha": gp.alpha}
        row.update({k: v for k, v in checks.items() if isinstance(v, (int, float, bool))})
        rows.append(row)

        # Plot
        if alpha1_gp and gp.alpha != 1.0:
            try:
                plot_branch_comparison(gp, alpha1_gp, "mGB", "mB", f"alpha{gp.alpha:.1f}")
            except Exception as e:
                print(f"  Plot error: {e}")

    # ---- GBP branch ----
    print("\n=== GBP branch ===")
    for i, d in enumerate(SCENARIOS_GBP):
        gp = make_gp(d)
        print(f"\n  GBP scenario {i+1}: alpha={gp.alpha}")
        pdf_fn = lambda x: pdf_GBP(x, gp)
        cdf_fn = lambda x: cdf_GBP(x, gp)
        try:
            norm = check_normalization(pdf_fn, 1e-8, 50 * gp.theta, tol=0.05)
            print(f"  Norm: {norm['integral']:.5f} (passed={norm['passed']})")
            if gp.alpha == 1.0:
                bp = gp.as_beta_sector()
                collapse = run_alpha1_collapse_check(gp, "GBP", "BP_B2")
                print(f"  Alpha=1 collapse GBP->BP/B2: {collapse}")
        except Exception as e:
            print(f"  Error: {e}")

    # ---- GB1 branch ----
    print("\n=== GB1 branch ===")
    for i, d in enumerate(SCENARIOS_GB1):
        gp = make_gp(d)
        print(f"\n  GB1 scenario {i+1}: alpha={gp.alpha}")
        upper = gp.beta_1 ** (1.0 / gp.alpha)
        pdf_fn = lambda x: pdf_GB1(x, gp)
        cdf_fn = lambda x: cdf_GB1(x, gp)
        try:
            norm = check_normalization(pdf_fn, 1e-8, upper - 1e-8, tol=0.05)
            print(f"  Norm: {norm['integral']:.5f} (passed={norm['passed']})")
            if gp.alpha == 1.0:
                collapse = run_alpha1_collapse_check(gp, "GB1", "B1")
                print(f"  Alpha=1 collapse GB1->B1: {collapse}")
        except Exception as e:
            print(f"  Error: {e}")

    # ---- GGa terminal ----
    print("\n=== GGa terminal branch ===")
    for i, d in enumerate(SCENARIOS_GGa):
        gp = make_gp(d)
        print(f"\n  GGa scenario {i+1}: alpha={gp.alpha}")
        pdf_fn = lambda x: pdf_GGa(x, gp)
        cdf_fn = lambda x: cdf_GGa(x, gp)
        try:
            norm = check_normalization(pdf_fn, 1e-8, 50 * gp.theta, tol=0.05)
            print(f"  Norm: {norm['integral']:.5f} (passed={norm['passed']})")
            if gp.alpha == 1.0:
                collapse = run_alpha1_collapse_check(gp, "GGa", "Ga")
                print(f"  Alpha=1 collapse GGa->Ga: {collapse}")
        except Exception as e:
            print(f"  Error: {e}")

    # ---- GIGa terminal ----
    print("\n=== GIGa terminal branch ===")
    for i, d in enumerate(SCENARIOS_GIGa):
        gp = make_gp(d)
        print(f"\n  GIGa scenario {i+1}: alpha={gp.alpha}")
        pdf_fn = lambda x: pdf_GIGa(x, gp)
        cdf_fn = lambda x: cdf_GIGa(x, gp)
        try:
            norm = check_normalization(pdf_fn, 1e-8, 50 * gp.theta, tol=0.05)
            cdf_ep = check_cdf_endpoints(cdf_fn, 1e-8, 50 * gp.theta)
            print(f"  Norm: {norm['integral']:.5f}, CDF endpoints passed={cdf_ep['passed']}")
            if gp.alpha == 1.0:
                collapse = run_alpha1_collapse_check(gp, "GIGa", "IGa")
                print(f"  Alpha=1 collapse GIGa->IGa: {collapse}")
        except Exception as e:
            print(f"  Error: {e}")

    # Summary figure: mGB PDFs for different alpha
    print("\nGenerating exp_2 summary figure...")
    _plot_mGB_alpha_grid()

    # Save results
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(RESULTS_DIR, "exp_2_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    print("\nExperiment 2 complete.")


def _plot_mGB_alpha_grid() -> None:
    """Plot mGB PDF for multiple alpha values."""
    d_base = dict(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, kappa_2=0.8)
    alphas = [0.5, 1.0, 1.5, 2.0]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, alpha in zip(axes, alphas):
        gp = GeneralizedParams(**d_base, alpha=alpha)
        upper = gp.beta_1 ** (1.0 / alpha)
        x = np.linspace(1e-6, upper - 1e-6, 300)
        try:
            f = pdf_mGB(x, gp)
            ax.plot(x, f, "b-", lw=2)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes)
        ax.set_title(f"mGB alpha={alpha}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_ylim(bottom=0)

    plt.suptitle("Exp-2: Generalized mGB — dynamical route", fontsize=13)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "exp_2_mGB_alpha_grid.png")
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")


if __name__ == "__main__":
    main()
