# Beta-Sector Master SDE and Stationary Hierarchy

Quantitative research project reproducing and validating a family of mean-reverting
stochastic differential equations (SDEs) whose stationary distributions form a
generalized modified-Beta hierarchy.

## Overview

The project implements and validates four connected experiments:

| Experiment | Description |
|---|---|
| `exp_1` | Original Beta-sector: master SDE, mB law, and reduction tree (IGa1, BP/B2, B1, IGa, Ga) |
| `exp_2` | Generalized hierarchy — **dynamical route**: generalized SDE → stationary mGB, GBP, GB1, GIGa, GGa |
| `exp_3` | Generalized hierarchy — **distributional route**: pushforward transformation y=x^alpha applied to parent laws |
| `exp_4` | **Noncommutativity** test: dynamical vs distributional routes compared against traditional GB benchmark |

## Repository Structure

```
project/
├── src/                  # Core implementations
│   ├── parameters.py     # BetaSectorParams, GeneralizedParams (derived quantities)
│   ├── densities.py      # All stationary PDFs (mB, IGa1, BP/B2, B1, IGa, Ga, mGB, …)
│   ├── cdfs.py           # All stationary CDFs
│   ├── simulation.py     # Euler-Maruyama SDE simulators with boundary enforcement
│   └── validation.py     # Normalization, KS/CvM, asymptotic slope, mean checks
├── exp/                  # Experiment scripts
│   ├── exp_1.py          # Original Beta-sector hierarchy
│   ├── exp_2.py          # Generalized dynamical route
│   ├── exp_3.py          # Generalized distributional route
│   └── exp_4.py          # Noncommutativity comparison
├── tests/
│   └── test_hierarchy.py # pytest suite (90+ tests)
├── data/                 # Data directory (no market data needed — purely synthetic)
└── results/              # Experiment outputs (plots, CSVs, RESULTS.md)
```

## Master SDE

The original master mean-reverting SDE:

```
dx = -γ(x - θ) dt + √(κ²x(1 - x/β₁)(1 + x/β₂)) dW_t
```

where `β₁ = 1/κ₁²`, `β₂ = κ²/κ₂²`, `p = 2γθ/κ²`, `q = 2γ(1 - κ₁²θ)/(κ²κ₁² + κ₂²)`.

The stationary law is the **modified-Beta (mB)** distribution supported on `[0, β₁]`.

### Reduction Tree (exp_1)

```
mB (full)
├── κ = 0  → IGa1 (on [0, β₁])  → κ₁ = 0 → IGa (on (0, ∞))
├── κ₁ = 0 → BP/B2 (on [0, ∞)) → κ = 0  → IGa
│                                → κ₂ = 0 → Ga
└── κ₂ = 0 → B1  (on [0, β₁])  → κ₁ = 0 → Ga
```

### Generalized Routes (exp_2, exp_3)

Two non-commuting generalizations for `α > 0`:

| Route | SDE/Rule | Near-zero slope |
|---|---|---|
| Dynamical | Generalized SDE with x^(2-α) diffusion | x^(p+α-2) |
| Distributional | Pushforward Y = X^α from parent law | x^(αp-1) |

These coincide at `α = 1` (original Beta-sector) but differ for `α ≠ 1`.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run all experiments (saves results to results/)
python run_all.py
```

## Environment

Python ≥ 3.10. Key libraries: `numpy`, `scipy`, `matplotlib`, `pandas`, `pytest`.
No external market data API is required (all experiments use synthetic parameters).

## Reproducibility

SDE simulations use fixed seeds (default `seed=42`). Simulation parameters:
- `dt = 0.005` (step size, implementation choice)
- `T_burn = 50.0` (burn-in, implementation choice)
- `T = 200.0` (total horizon, implementation choice)
- Boundary enforcement: projection to `(ε, upper - ε)` with `ε = 1e-8`

See `results/RESULTS.md` for a summary of experiment findings.
