# Repository: Beta-Sector Master SDE and Stationary Hierarchy

## Purpose
Reproduces and validates a family of stationary distributions for mean-reverting
SDEs forming the Beta-sector hierarchy and its power-transformation generalizations.

## Key Architecture
- `src/parameters.py` — `BetaSectorParams` and `GeneralizedParams` dataclasses
  with all derived quantities (beta_1, beta_2, p, q, beta, lam, u, v, w, delta)
- `src/densities.py` — all PDFs: mB, IGa1, BP_B2, B1, IGa, Ga (exp_1);
  mGB, GBP, GB1, GIGa1, GIGa, GGa (exp_2 dynamical); mB_alpha etc. (exp_3 distributional); GB (exp_4)
- `src/cdfs.py` — all CDFs; exp_3 CDFs use pushforward F_X(x)=F_Y(x^alpha)
- `src/simulation.py` — Euler-Maruyama with projection to bounded support
- `src/validation.py` — normalization/KS/CvM/slope/mean checks
- `exp/exp_[1-4].py` — experiment runners, each has a `main()` function
- `run_all.py` — top-level runner that calls all four main() functions
- `tests/test_hierarchy.py` — pytest suite covering params, PDFs, CDFs, simulations, alpha=1 collapse

## Critical Bugs Fixed
- `cdf_IGa`: CDF = `gammaincc(q+1, beta/x)` (NOT `1 - gammaincc`). The IGa CDF
  is the upper incomplete gamma ratio because the integral from 0 to x with
  substitution u=beta/t gives Gamma(q+1, beta/x)/Gamma(q+1).
- `cdf_GIGa`: Same logic — `gammaincc(shape, beta/(alpha*x^alpha))` without `1 -`.

## Parameter Conventions
| Symbol | Formula | Notes |
|---|---|---|
| beta_1 | 1/kappa_1^2 | inf when kappa_1=0 (unbounded support) |
| beta_2 | kappa^2/kappa_2^2 | inf when kappa=0 or kappa_2=0 |
| p | 2*gamma*theta/kappa^2 | NaN when kappa=0 |
| q (mB) | 2*gamma*(1-kappa_1^2*theta)/(kappa^2*kappa_1^2+kappa_2^2) | |
| q (BP_B2) | 2*gamma/kappa_2^2 - p | kappa_1=0 branch |
| beta | 2*gamma*theta/kappa_2^2 | IGa1/IGa branches |
| lam | kappa^2/(2*gamma) | Ga branch: lambda = theta/p |
| u | (p+alpha-1)/alpha | integrability condition: u>0 |
| v | q/alpha | mGB parameter |
| w | 1 + (p+q)/alpha | mGB parameter |

## Asymptotic Slopes
- Original: f(x) ~ x^(p-1) near zero for bounded branches (mB, B1, IGa1)
- Dynamical route: f(x) ~ x^(p+alpha-2) near zero (mGB, GBP, GB1)
- Distributional route: f(x) ~ x^(alpha*p-1) near zero (mB_alpha, B1_alpha)
- These differ for alpha != 1, which is the noncommutativity signature

## Branch Support
| Branch | Support | kappa conditions |
|---|---|---|
| mB | (0, beta_1) | kappa, kappa_1, kappa_2 > 0 |
| IGa1 | (0, beta_1) | kappa=0, kappa_1>0, kappa_2>0 |
| BP_B2 | (0, inf) | kappa>0, kappa_1=0, kappa_2>0 |
| B1 | (0, beta_1) | kappa>0, kappa_1>0, kappa_2=0 |
| IGa | (0, inf) | kappa=0, kappa_1=0, kappa_2>0 |
| Ga | (0, inf) | kappa>0, kappa_1=0, kappa_2=0 |
| mGB | (0, beta_1^(1/alpha)) | all three positive |
| GBP | (0, inf) | kappa>0, kappa_1=0, kappa_2>0 |
| GB1 | (0, beta_1^(1/alpha)) | kappa>0, kappa_1>0, kappa_2=0 |
| GGa | (0, inf) | kappa>0, kappa_1=0, kappa_2=0 |
| GIGa | (0, inf) | kappa=0, kappa_1=0, kappa_2>0 |

## No Market Data
All experiments use purely synthetic parameters. No Massive API or market
data of any kind is required. The `data/` directory exists for structural
compliance but remains empty.

## Running
```bash
pytest tests/ -v          # run all tests
python run_all.py         # run all experiments, save to results/
```
