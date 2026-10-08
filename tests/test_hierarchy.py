"""
pytest tests for the Beta-sector SDE hierarchy.

Tests cover:
  - Parameter validation (BetaSectorParams, GeneralizedParams)
  - PDF normalization for all branches
  - CDF boundary values
  - dF/dx ≈ f consistency
  - Alpha=1 collapse (generalized → original)
  - Asymptotic slope properties
  - Pushforward (distributional route) identity
  - Mathematical property validation (support, monotone CDF, non-negative PDF)
  - Simulation validity (n_samples, support containment)
  - Validation utility functions
"""
import pytest
import numpy as np
from scipy.integrate import quad

from src.parameters import BetaSectorParams, GeneralizedParams
from src.densities import (
    pdf_mB, pdf_IGa1, pdf_BP_B2, pdf_B1, pdf_IGa, pdf_Ga,
    pdf_mGB, pdf_GBP, pdf_GB1, pdf_GIGa, pdf_GGa,
    pdf_mB_alpha, pdf_BP_B2_alpha, pdf_B1_alpha, pdf_IGa_alpha, pdf_Ga_alpha,
    pdf_GB,
)
from src.cdfs import (
    cdf_mB, cdf_IGa1, cdf_BP_B2, cdf_B1, cdf_IGa, cdf_Ga,
    cdf_mGB, cdf_GBP, cdf_GB1, cdf_GIGa, cdf_GGa,
    cdf_mB_alpha, cdf_BP_B2_alpha, cdf_B1_alpha, cdf_IGa_alpha, cdf_Ga_alpha,
    cdf_GB,
)
from src.simulation import (
    simulate_mB_sde, simulate_B1_sde, simulate_Ga_sde, simulate_IGa_sde,
    simulate_generalized_sde,
)
from src.validation import (
    check_normalization, check_cdf_endpoints, check_cdf_derivative,
    compute_ks_statistic, estimate_asymptotic_slope, check_mean_property,
    compute_integrated_absolute_error,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def p_mB():
    return BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, kappa_2=0.8)

@pytest.fixture
def p_B1():
    return BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5)

@pytest.fixture
def p_BP():
    return BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_2=0.5)

@pytest.fixture
def p_IGa1():
    return BetaSectorParams(gamma=2.0, theta=0.3, kappa_1=0.5, kappa_2=0.8)

@pytest.fixture
def p_IGa():
    return BetaSectorParams(gamma=2.0, theta=0.3, kappa_2=0.5)

@pytest.fixture
def p_Ga():
    return BetaSectorParams(gamma=2.0, theta=0.3, kappa=1.0)

@pytest.fixture
def gp_mGB():
    return GeneralizedParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, kappa_2=0.8, alpha=1.5)

@pytest.fixture
def gp_alpha1():
    return GeneralizedParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, kappa_2=0.8, alpha=1.0)


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------

class TestBetaSectorParams:

    def test_derived_beta1(self, p_mB):
        """beta_1 = 1/kappa_1^2."""
        assert abs(p_mB.beta_1 - 1.0 / p_mB.kappa_1**2) < 1e-12

    def test_derived_beta2(self, p_mB):
        """beta_2 = kappa^2/kappa_2^2."""
        expected = p_mB.kappa**2 / p_mB.kappa_2**2
        assert abs(p_mB.beta_2 - expected) < 1e-12

    def test_derived_p(self, p_mB):
        """p = 2*gamma*theta/kappa^2."""
        expected = 2 * p_mB.gamma * p_mB.theta / p_mB.kappa**2
        assert abs(p_mB.p - expected) < 1e-12

    def test_derived_q_positive(self, p_mB):
        """q must be positive for valid mB."""
        assert p_mB.q > 0

    def test_lambda_Ga(self, p_Ga):
        """lambda = kappa^2/(2*gamma) = theta/p."""
        assert abs(p_Ga.lam - p_Ga.kappa**2 / (2 * p_Ga.gamma)) < 1e-12
        assert abs(p_Ga.lam - p_Ga.theta / p_Ga.p) < 1e-12

    def test_beta1_inf_when_kappa1_zero(self, p_Ga):
        """beta_1 = inf when kappa_1 = 0."""
        assert np.isinf(p_Ga.beta_1)

    def test_invalid_gamma(self):
        """gamma <= 0 should raise AssertionError."""
        with pytest.raises(AssertionError):
            BetaSectorParams(gamma=-1.0, theta=0.3)

    def test_theta_support_condition(self, p_mB):
        """theta < beta_1 for bounded branches."""
        assert p_mB.theta < p_mB.beta_1

    def test_validate_mB_passes(self, p_mB):
        """validate_mB should not raise for valid params."""
        p_mB.validate_mB()  # should not raise


class TestGeneralizedParams:

    def test_delta(self, gp_mGB):
        """delta = 1/alpha."""
        assert abs(gp_mGB.delta - 1.0 / gp_mGB.alpha) < 1e-12

    def test_u_formula(self, gp_mGB):
        """u = (p + alpha - 1)/alpha."""
        expected = (gp_mGB.p + gp_mGB.alpha - 1.0) / gp_mGB.alpha
        assert abs(gp_mGB.u - expected) < 1e-12

    def test_v_formula(self, gp_mGB):
        """v = q/alpha."""
        expected = gp_mGB.q / gp_mGB.alpha
        assert abs(gp_mGB.v - expected) < 1e-12

    def test_w_formula(self, gp_mGB):
        """w = 1 + (p+q)/alpha."""
        expected = 1.0 + (gp_mGB.p + gp_mGB.q) / gp_mGB.alpha
        assert abs(gp_mGB.w - expected) < 1e-12

    def test_integrability_check(self, gp_mGB):
        """u > 0 should pass."""
        assert gp_mGB.check_integrability()

    def test_alpha1_as_beta_sector(self, gp_alpha1):
        """as_beta_sector() should give matching BetaSectorParams."""
        bp = gp_alpha1.as_beta_sector()
        assert abs(bp.gamma - gp_alpha1.gamma) < 1e-12
        assert abs(bp.p - gp_alpha1.p) < 1e-12
        assert abs(bp.q - gp_alpha1.q) < 1e-12


# ---------------------------------------------------------------------------
# PDF property tests
# ---------------------------------------------------------------------------

class TestPDFProperties:

    def test_mB_non_negative(self, p_mB):
        """PDF values must be non-negative."""
        x = np.linspace(1e-6, p_mB.beta_1 - 1e-6, 200)
        assert np.all(pdf_mB(x, p_mB) >= 0)

    def test_mB_zero_outside_support(self, p_mB):
        """PDF must be 0 outside support."""
        assert pdf_mB(np.array([0.0]), p_mB)[0] == 0.0
        assert pdf_mB(np.array([p_mB.beta_1 + 0.1]), p_mB)[0] == 0.0
        assert pdf_mB(np.array([-0.1]), p_mB)[0] == 0.0

    def test_B1_non_negative(self, p_B1):
        """B1 PDF must be non-negative on support."""
        x = np.linspace(1e-6, p_B1.beta_1 - 1e-6, 200)
        assert np.all(pdf_B1(x, p_B1) >= 0)

    def test_Ga_non_negative(self, p_Ga):
        """Gamma PDF must be non-negative."""
        x = np.linspace(1e-5, 20 * p_Ga.theta, 200)
        assert np.all(pdf_Ga(x, p_Ga) >= 0)

    def test_IGa_non_negative(self, p_IGa):
        """IGa PDF must be non-negative."""
        x = np.linspace(1e-5, 20 * p_IGa.theta, 200)
        assert np.all(pdf_IGa(x, p_IGa) >= 0)

    def test_BP_B2_non_negative(self, p_BP):
        """BP/B2 PDF must be non-negative."""
        x = np.linspace(1e-5, 20 * p_BP.theta, 200)
        assert np.all(pdf_BP_B2(x, p_BP) >= 0)

    def test_mGB_non_negative(self, gp_mGB):
        """mGB PDF must be non-negative on support."""
        upper = gp_mGB.beta_1 ** (1.0 / gp_mGB.alpha)
        x = np.linspace(1e-6, upper - 1e-6, 100)
        f = pdf_mGB(x, gp_mGB)
        assert np.all(f >= 0)

    def test_GB_non_negative(self, gp_mGB):
        """GB benchmark PDF must be non-negative."""
        upper = gp_mGB.beta_1 ** (1.0 / gp_mGB.alpha)
        x = np.linspace(1e-6, upper - 1e-6, 100)
        f = pdf_GB(x, gp_mGB)
        assert np.all(f >= 0)

    def test_distributional_non_negative(self, p_mB):
        """mB_alpha PDF must be non-negative."""
        alpha = 1.5
        upper = p_mB.beta_1 ** (1.0 / alpha)
        x = np.linspace(1e-6, upper - 1e-6, 100)
        assert np.all(pdf_mB_alpha(x, p_mB, alpha) >= 0)


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

NORM_TOL = 0.05  # Relaxed tolerance for quadrature over singular endpoints


class TestNormalization:

    def test_mB_normalizes(self, p_mB):
        """mB PDF integrates to 1."""
        r = check_normalization(
            lambda x: pdf_mB(x, p_mB), 1e-8, p_mB.beta_1 - 1e-8, tol=NORM_TOL
        )
        assert r["passed"], f"mB norm={r['integral']:.6f}"

    def test_B1_normalizes(self, p_B1):
        """B1 PDF integrates to 1."""
        r = check_normalization(
            lambda x: pdf_B1(x, p_B1), 1e-8, p_B1.beta_1 - 1e-8, tol=NORM_TOL
        )
        assert r["passed"], f"B1 norm={r['integral']:.6f}"

    def test_BP_normalizes(self, p_BP):
        """BP/B2 PDF integrates to 1."""
        r = check_normalization(
            lambda x: pdf_BP_B2(x, p_BP), 1e-8, 100 * p_BP.theta, tol=NORM_TOL
        )
        assert r["passed"], f"BP norm={r['integral']:.6f}"

    def test_Ga_normalizes(self, p_Ga):
        """Ga PDF integrates to 1."""
        r = check_normalization(
            lambda x: pdf_Ga(x, p_Ga), 1e-8, 50 * p_Ga.theta, tol=NORM_TOL
        )
        assert r["passed"], f"Ga norm={r['integral']:.6f}"

    def test_IGa_normalizes(self, p_IGa):
        """IGa PDF integrates to 1."""
        r = check_normalization(
            lambda x: pdf_IGa(x, p_IGa), 1e-8, 50 * p_IGa.theta, tol=NORM_TOL
        )
        assert r["passed"], f"IGa norm={r['integral']:.6f}"

    def test_mGB_normalizes(self, gp_mGB):
        """mGB PDF integrates to 1."""
        upper = gp_mGB.beta_1 ** (1.0 / gp_mGB.alpha)
        r = check_normalization(
            lambda x: pdf_mGB(x, gp_mGB), 1e-8, upper - 1e-8, tol=NORM_TOL
        )
        assert r["passed"], f"mGB norm={r['integral']:.6f}"

    def test_GB1_normalizes(self, gp_mGB):
        """GB1 PDF integrates to 1."""
        gp = GeneralizedParams(
            gamma=gp_mGB.gamma, theta=gp_mGB.theta,
            kappa=gp_mGB.kappa, kappa_1=gp_mGB.kappa_1, alpha=gp_mGB.alpha
        )
        upper = gp.beta_1 ** (1.0 / gp.alpha)
        r = check_normalization(
            lambda x: pdf_GB1(x, gp), 1e-8, upper - 1e-8, tol=NORM_TOL
        )
        assert r["passed"], f"GB1 norm={r['integral']:.6f}"

    def test_GGa_normalizes(self, gp_mGB):
        """GGa PDF integrates to 1."""
        gp = GeneralizedParams(
            gamma=gp_mGB.gamma, theta=gp_mGB.theta,
            kappa=gp_mGB.kappa, alpha=gp_mGB.alpha
        )
        r = check_normalization(
            lambda x: pdf_GGa(x, gp), 1e-8, 50 * gp.theta, tol=NORM_TOL
        )
        assert r["passed"], f"GGa norm={r['integral']:.6f}"

    def test_mB_alpha_normalizes(self, p_mB):
        """mB_alpha PDF integrates to 1 for alpha=2."""
        alpha = 2.0
        upper = p_mB.beta_1 ** (1.0 / alpha)
        r = check_normalization(
            lambda x: pdf_mB_alpha(x, p_mB, alpha), 1e-8, upper - 1e-8, tol=NORM_TOL
        )
        assert r["passed"], f"mB_alpha norm={r['integral']:.6f}"

    def test_Ga_alpha_normalizes(self, p_Ga):
        """Ga_alpha PDF integrates to 1 for alpha=2."""
        alpha = 2.0
        r = check_normalization(
            lambda x: pdf_Ga_alpha(x, p_Ga, alpha), 1e-8, 50 * p_Ga.theta, tol=NORM_TOL
        )
        assert r["passed"], f"Ga_alpha norm={r['integral']:.6f}"

    def test_GB_normalizes(self, gp_mGB):
        """GB benchmark PDF integrates to 1."""
        upper = gp_mGB.beta_1 ** (1.0 / gp_mGB.alpha)
        r = check_normalization(
            lambda x: pdf_GB(x, gp_mGB), 1e-8, upper - 1e-8, tol=NORM_TOL
        )
        assert r["passed"], f"GB norm={r['integral']:.6f}"


# ---------------------------------------------------------------------------
# CDF property tests
# ---------------------------------------------------------------------------

class TestCDFProperties:

    def test_B1_cdf_monotone(self, p_B1):
        """B1 CDF must be monotone increasing."""
        x = np.linspace(1e-6, p_B1.beta_1 - 1e-6, 100)
        F = cdf_B1(x, p_B1)
        assert np.all(np.diff(F) >= -1e-10), "CDF not monotone"

    def test_B1_cdf_endpoints(self, p_B1):
        """B1 CDF: F(0+) ≈ 0, F(beta_1-) ≈ 1."""
        assert cdf_B1(np.array([1e-8]), p_B1)[0] < 0.01
        assert cdf_B1(np.array([p_B1.beta_1 - 1e-8]), p_B1)[0] > 0.99

    def test_Ga_cdf_endpoints(self, p_Ga):
        """Ga CDF: F(0+) ≈ 0, F(large) ≈ 1."""
        assert cdf_Ga(np.array([1e-8]), p_Ga)[0] < 0.01
        assert cdf_Ga(np.array([20 * p_Ga.theta]), p_Ga)[0] > 0.99

    def test_BP_cdf_monotone(self, p_BP):
        """BP/B2 CDF must be monotone increasing."""
        x = np.linspace(1e-5, 20 * p_BP.theta, 100)
        F = cdf_BP_B2(x, p_BP)
        assert np.all(np.diff(F) >= -1e-10), "CDF not monotone"

    def test_GB1_cdf_monotone(self, gp_mGB):
        """GB1 CDF must be monotone."""
        gp = GeneralizedParams(
            gamma=gp_mGB.gamma, theta=gp_mGB.theta,
            kappa=gp_mGB.kappa, kappa_1=gp_mGB.kappa_1, alpha=gp_mGB.alpha
        )
        upper = gp.beta_1 ** (1.0 / gp.alpha)
        x = np.linspace(1e-6, upper - 1e-6, 100)
        F = cdf_GB1(x, gp)
        assert np.all(np.diff(F) >= -1e-10)

    def test_GGa_cdf_monotone(self, gp_mGB):
        """GGa CDF must be monotone."""
        gp = GeneralizedParams(
            gamma=gp_mGB.gamma, theta=gp_mGB.theta,
            kappa=gp_mGB.kappa, alpha=gp_mGB.alpha
        )
        x = np.linspace(1e-5, 20 * gp.theta, 100)
        F = cdf_GGa(x, gp)
        assert np.all(np.diff(F) >= -1e-10)

    def test_mB_alpha_cdf_monotone(self, p_mB):
        """mB_alpha CDF must be monotone for alpha=2."""
        alpha = 2.0
        upper = p_mB.beta_1 ** (1.0 / alpha)
        x = np.linspace(1e-6, upper - 1e-6, 100)
        F = cdf_mB_alpha(x, p_mB, alpha)
        assert np.all(np.diff(F) >= -1e-10)

    def test_pushforward_cdf_identity(self, p_mB):
        """F_X(x) = F_Y(x^alpha) for distributional route."""
        alpha = 2.0
        x = np.linspace(1e-5, p_mB.beta_1**0.5 - 1e-5, 50)
        F_alpha = cdf_mB_alpha(x, p_mB, alpha)
        F_parent = cdf_mB(x**alpha, p_mB)
        np.testing.assert_allclose(F_alpha, F_parent, atol=0.02)


# ---------------------------------------------------------------------------
# Alpha=1 collapse tests
# ---------------------------------------------------------------------------

class TestAlpha1Collapse:

    def _rel_diff(self, f1, f2):
        mask = f2 > 1e-10
        if not mask.any():
            return np.nan
        return float(np.max(np.abs(f1[mask] - f2[mask]) / f2[mask]))

    def test_mGB_collapses_to_mB(self, gp_alpha1, p_mB):
        """mGB at alpha=1 should equal mB."""
        x = np.linspace(1e-5, p_mB.beta_1 - 1e-5, 100)
        f_gen = pdf_mGB(x, gp_alpha1)
        f_orig = pdf_mB(x, p_mB)
        rd = self._rel_diff(f_gen, f_orig)
        assert rd < 0.1, f"mGB/mB collapse: max_rel_diff={rd:.4f}"

    def test_GBP_collapses_to_BP_B2(self):
        """GBP at alpha=1 should equal BP/B2."""
        gp = GeneralizedParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_2=0.5, alpha=1.0)
        bp = gp.as_beta_sector()
        x = np.linspace(1e-5, 20 * gp.theta, 100)
        f_gen = pdf_GBP(x, gp)
        f_orig = pdf_BP_B2(x, bp)
        rd = self._rel_diff(f_gen, f_orig)
        assert rd < 0.1, f"GBP/BP_B2 collapse: max_rel_diff={rd:.4f}"

    def test_GB1_collapses_to_B1(self):
        """GB1 at alpha=1 should equal B1."""
        gp = GeneralizedParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_1=0.5, alpha=1.0)
        bp = gp.as_beta_sector()
        upper = bp.beta_1
        x = np.linspace(1e-5, upper - 1e-5, 100)
        f_gen = pdf_GB1(x, gp)
        f_orig = pdf_B1(x, bp)
        rd = self._rel_diff(f_gen, f_orig)
        assert rd < 0.1, f"GB1/B1 collapse: max_rel_diff={rd:.4f}"

    def test_GGa_collapses_to_Ga(self):
        """GGa at alpha=1 should equal Ga."""
        gp = GeneralizedParams(gamma=2.0, theta=0.3, kappa=1.0, alpha=1.0)
        bp = gp.as_beta_sector()
        x = np.linspace(1e-5, 20 * gp.theta, 100)
        f_gen = pdf_GGa(x, gp)
        f_orig = pdf_Ga(x, bp)
        rd = self._rel_diff(f_gen, f_orig)
        assert rd < 0.1, f"GGa/Ga collapse: max_rel_diff={rd:.4f}"

    def test_mB_alpha_collapses_to_mB(self, p_mB):
        """mB_alpha at alpha=1 should equal mB."""
        x = np.linspace(1e-5, p_mB.beta_1 - 1e-5, 100)
        f_alpha = pdf_mB_alpha(x, p_mB, 1.0)
        f_orig = pdf_mB(x, p_mB)
        rd = self._rel_diff(f_alpha, f_orig)
        assert rd < 0.01, f"mB_alpha/mB collapse: max_rel_diff={rd:.4f}"

    def test_B1_alpha_collapses_to_B1(self, p_B1):
        """B1_alpha at alpha=1 should equal B1."""
        x = np.linspace(1e-5, p_B1.beta_1 - 1e-5, 100)
        f_alpha = pdf_B1_alpha(x, p_B1, 1.0)
        f_orig = pdf_B1(x, p_B1)
        rd = self._rel_diff(f_alpha, f_orig)
        assert rd < 0.01, f"B1_alpha/B1 collapse: max_rel_diff={rd:.4f}"

    def test_Ga_alpha_collapses_to_Ga(self, p_Ga):
        """Ga_alpha at alpha=1 should equal Ga."""
        x = np.linspace(1e-5, 20 * p_Ga.theta, 100)
        f_alpha = pdf_Ga_alpha(x, p_Ga, 1.0)
        f_orig = pdf_Ga(x, p_Ga)
        rd = self._rel_diff(f_alpha, f_orig)
        assert rd < 0.01, f"Ga_alpha/Ga collapse: max_rel_diff={rd:.4f}"


# ---------------------------------------------------------------------------
# Asymptotic slope tests
# ---------------------------------------------------------------------------

class TestAsymptoticSlopes:

    def test_mB_near_zero_slope(self, p_mB):
        """mB near-zero slope should be close to p-1."""
        expected = p_mB.p - 1
        result = estimate_asymptotic_slope(
            lambda x: pdf_mB(x, p_mB), x_lo=1e-7, x_hi=1e-3
        )
        assert abs(result["slope"] - expected) < 0.3, \
            f"mB slope: {result['slope']:.3f} vs expected {expected:.3f}"

    def test_B1_near_zero_slope(self, p_B1):
        """B1 near-zero slope should be close to p-1."""
        expected = p_B1.p - 1
        result = estimate_asymptotic_slope(
            lambda x: pdf_B1(x, p_B1), x_lo=1e-7, x_hi=1e-3
        )
        assert abs(result["slope"] - expected) < 0.3, \
            f"B1 slope: {result['slope']:.3f} vs expected {expected:.3f}"

    def test_distributional_route_slope(self, p_mB):
        """mB_alpha near-zero slope should be close to alpha*p-1."""
        alpha = 2.0
        expected = alpha * p_mB.p - 1
        upper = p_mB.beta_1 ** (1.0 / alpha)
        result = estimate_asymptotic_slope(
            lambda x: pdf_mB_alpha(x, p_mB, alpha), x_lo=1e-7, x_hi=upper * 0.01
        )
        assert abs(result["slope"] - expected) < 0.4, \
            f"mB_alpha slope: {result['slope']:.3f} vs expected {expected:.3f}"

    def test_dynamical_route_slope(self, gp_mGB):
        """mGB near-zero slope should be close to p+alpha-2."""
        expected = gp_mGB.p + gp_mGB.alpha - 2
        upper = gp_mGB.beta_1 ** (1.0 / gp_mGB.alpha)
        result = estimate_asymptotic_slope(
            lambda x: pdf_mGB(x, gp_mGB), x_lo=1e-7, x_hi=upper * 0.01
        )
        assert abs(result["slope"] - expected) < 0.5, \
            f"mGB slope: {result['slope']:.3f} vs expected {expected:.3f}"


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestSimulation:

    def test_B1_simulation_samples(self, p_B1):
        """B1 simulation should return correct number of samples in support."""
        samples = simulate_B1_sde(p_B1, n_samples=200, T=50, T_burn=10, dt=0.01, seed=1)
        assert len(samples) == 200
        assert np.all(samples > 0)
        assert np.all(samples < p_B1.beta_1)

    def test_Ga_simulation_samples(self, p_Ga):
        """Ga simulation should return positive samples."""
        samples = simulate_Ga_sde(p_Ga, n_samples=200, T=50, T_burn=10, dt=0.01, seed=2)
        assert len(samples) == 200
        assert np.all(samples > 0)

    def test_mB_simulation_mean(self, p_mB):
        """mB simulation mean should be close to theta."""
        samples = simulate_mB_sde(p_mB, n_samples=500, T=100, T_burn=20, dt=0.005, seed=42)
        result = check_mean_property(samples, p_mB.theta, tol=0.15)
        assert result["passed"], f"E[x]={result['empirical_mean']:.4f} vs theta={p_mB.theta}"

    def test_generalized_GGa_simulation(self):
        """GGa generalized SDE should run without errors."""
        gp = GeneralizedParams(gamma=2.0, theta=0.3, kappa=1.0, alpha=1.5)
        samples = simulate_generalized_sde(gp, branch="GGa", n_samples=200,
                                           T=50, T_burn=10, dt=0.01, seed=3)
        assert len(samples) == 200
        assert np.all(samples > 0)


# ---------------------------------------------------------------------------
# Validation utility tests
# ---------------------------------------------------------------------------

class TestValidationUtils:

    def test_check_normalization_Ga(self, p_Ga):
        """Normalization check utility should pass for Ga."""
        result = check_normalization(
            lambda x: pdf_Ga(x, p_Ga), 1e-8, 50 * p_Ga.theta, tol=0.05
        )
        assert result["passed"]

    def test_check_cdf_endpoints_B1(self, p_B1):
        """CDF endpoints check should pass for B1."""
        result = check_cdf_endpoints(lambda x: cdf_B1(x, p_B1), 0, p_B1.beta_1)
        assert result["passed"]

    def test_check_cdf_derivative_Ga(self, p_Ga):
        """dF/dx ≈ f should hold for Ga."""
        result = check_cdf_derivative(
            lambda x: pdf_Ga(x, p_Ga),
            lambda x: cdf_Ga(x, p_Ga),
            1e-5, 10 * p_Ga.theta,
        )
        assert result["passed"]

    def test_iad_same_density(self, p_mB):
        """IAD of a density with itself should be 0."""
        iad = compute_integrated_absolute_error(
            lambda x: pdf_mB(x, p_mB),
            lambda x: pdf_mB(x, p_mB),
            1e-6, p_mB.beta_1 - 1e-6,
        )
        assert iad < 1e-10

    def test_ks_statistic_range(self, p_B1):
        """KS statistic should be in [0, 1]."""
        samples = simulate_B1_sde(p_B1, n_samples=300, T=80, T_burn=15, dt=0.01, seed=99)
        result = compute_ks_statistic(samples, lambda x: cdf_B1(x, p_B1))
        assert 0 <= result["ks_stat"] <= 1

    def test_check_mean_property_pass(self):
        """Mean check should pass when within tolerance."""
        samples = np.random.default_rng(0).normal(1.0, 0.1, 1000)
        result = check_mean_property(samples, expected_mean=1.0, tol=0.1)
        assert result["passed"]

    def test_check_mean_property_fail(self):
        """Mean check should fail when far from expected."""
        samples = np.ones(100) * 5.0
        result = check_mean_property(samples, expected_mean=1.0, tol=0.1)
        assert not result["passed"]


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_mB_empty_input(self, p_mB):
        """PDF/CDF should handle empty arrays."""
        out = pdf_mB(np.array([]), p_mB)
        assert len(out) == 0

    def test_scalar_input(self, p_mB):
        """PDF should accept scalar input."""
        f = pdf_mB(0.1, p_mB)
        assert len(f) == 1
        assert f[0] >= 0

    def test_mB_boundary_values(self, p_mB):
        """PDF at x=0 and x=beta_1 should return 0."""
        assert pdf_mB(np.array([0.0]), p_mB)[0] == 0.0
        assert pdf_mB(np.array([p_mB.beta_1]), p_mB)[0] == 0.0

    def test_GBP_large_x(self):
        """GBP PDF should approach 0 for large x."""
        gp = GeneralizedParams(gamma=2.0, theta=0.3, kappa=1.0, kappa_2=0.5, alpha=1.0)
        f_large = pdf_GBP(np.array([1000.0]), gp)[0]
        assert f_large < 1e-5

    def test_Ga_at_zero(self, p_Ga):
        """Ga PDF at x=0 should be 0."""
        assert pdf_Ga(np.array([0.0]), p_Ga)[0] == 0.0

    def test_normalization_passes_tol(self, p_Ga):
        """Normalization error should be within tolerance."""
        r = check_normalization(
            lambda x: pdf_Ga(x, p_Ga), 1e-8, 100 * p_Ga.theta, tol=0.01
        )
        assert r["norm_error"] < 0.01
