# First Principles of Uncertainty — Computational Validation Suite

**Agent-based simulations validating the axiomatic framework for risk, prediction, and systemic stability.**

[![Paper III](https://img.shields.io/badge/Paper_III-Applied_Probabilistic_Systems-C87941)](https://jasongething.substack.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Companion simulation package for **"The First Principles of Uncertainty: A Unified Axiomatic Framework for Risk, Prediction, and Systemic Stability"** — Paper III in the Applied Probabilistic Systems series.

The paper identifies five axiomatic properties of uncertainty — **Irreducibility** (Knight 1921), **Reflexivity** (Soros 1987), **Non-Stationarity** (Minsky 1992), **Emergent Coupling** (Mandelbrot 1963), and **Requisite Diversity** (Holling 1973) — and demonstrates that their unification into a single deductive system produces seven theorems with consequences no prior framework has derived.

This code validates the theorems computationally through agent-based simulation.

## Simulations

| Simulation | Validates | Key Result |
|---|---|---|
| **Precision-Fragility Paradox** | Theorem 6 (Axioms II + V) | Inverted-U: shared knowledge improves stability to a point, then destroys it |
| **Calibration Decay** | Theorem 3 (Axioms I + III) | Prediction error grows unpredictably. Decay rate is itself non-stationary |
| **Diversification Failure** | Theorem 4 (Axiom IV) | Portfolio diversification benefit collapses from 0.95 to near zero under stress |
| **Coupling Cascade** | Axiom IV | Non-linear phase transition: isolated failures become systemic cascades |
| **Endogenous Feedback** | Theorem 2 (Axiom II) | Risk estimates feeding back into reality amplify volatility ~86% |

Theorems 1, 5, and 7 are validated by logical argument in the paper rather than simulation.

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib scipy

# Run all simulations
python uncertainty_simulations.py

# Run a specific theorem
python uncertainty_simulations.py --theorem 6

# Export publication-quality figures
python uncertainty_simulations.py --export

# Custom parameters
python uncertainty_simulations.py --agents 1000 --trials 100 --seed 42
```

## Axiomatic Metrics

The code includes production-ready implementations of three core metrics defined in the paper:

- **DDR** (Distributional Drift Rate) — speed of distributional change
- **BHI** (Behavioural Homogeneity Index) — agent response similarity
- **CII** (Coupling Intensity Index) — cross-factor correlation intensity

## Reproducibility

All simulations use `numpy.random.default_rng` with explicit seed control. Default seed is 42. Results are deterministic for any given configuration.

## Paper Series

| Paper | Title | Date |
|---|---|---|
| I | *The Reflexive Stagnation Trap* | December 2025 |
| II | *The Homogeneity Threshold* | January 2026 |
| **III** | **The First Principles of Uncertainty** | **February 2026** |

All papers at [jasongething.substack.com](https://jasongething.substack.com)

## Citation

```
Gething, J. (2026). "The First Principles of Uncertainty: A Unified Axiomatic
Framework for Risk, Prediction, and Systemic Stability." FishIntel Global,
Applied Probabilistic Systems Working Paper Series, Paper III.
```

## Author

**Jason Gething** — Founder, [FishIntel Global](https://fishintelglobal.com)

## License

MIT — see [LICENSE](LICENSE)
