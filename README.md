# Gram Matrix Spectral Analysis — Single vs. Multi-Dirac Control

Numerical experiments comparing the spectral decay of the controllability/observability
Gramian $X_N$ for the 1-D Dirichlet Laplacian $A = -\partial_x^2$ on $(0,1)$, when the
control/observation operator $B$ is a **single Dirac mass** versus a **sum of $M$ Dirac
masses**. Eigenvalues of $X_N$ are compared against theoretical upper bounds derived from
an admissibility/Sobolev-type estimate involving the Riemann zeta function.

## Background

For $A = -\Delta$ with Dirichlet boundary conditions on $(0,1)$, the eigenpairs are

$$\lambda_k = (\pi k)^2, \qquad \varphi_k(x) = \sqrt{2}\,\sin(\pi k x), \qquad k = 1, 2, \dots$$

Given a control operator $B$ built from Dirac masses at points $x_m \in (0,1)$, the finite
section $X_N$ of the associated Gramian in the eigenbasis $\{\varphi_k\}$ satisfies

$$\mu_k(X_N) \;\le\; \tfrac{1}{2}\,\lambda_k^{-(1-\beta)}\, \big\| |A|^{-\beta/2} B \big\|_{S_\infty}^2,
\qquad \beta \in (1/2, 1),$$

where the right-hand side norm is estimated (for the single-Dirac case) via

$$\big\| |A|^{-\beta/2} B \big\|_{S_\infty}^2 \;\le\; \frac{2}{\pi^{2\beta}}\, \zeta(2\beta).$$

This repository numerically builds $X_N$ for both the single- and multi-Dirac cases,
computes its eigenvalues, and checks them against the bound above for several values of
$\beta$.

## Contents

| File | Description |
|---|---|
| `gram_dirac_analysis.py` | Main script: builds the Gramians, computes eigenvalues and bounds, produces plots and data tables. |
| `single_vs_multi_dirac_bounds.pdf` | Eigenvalues $\mu_k(X_N)$ vs. theoretical bounds, full index range (generated). |
| `single_vs_multi_dirac_bounds_zoom.pdf` | Same plot, zoomed to the first 20 indices (generated). |
| `figure1_single_vs_multi_dirac_data.csv` | Tabulated spectral data (generated). |
| `figure1_single_vs_multi_dirac_data.pkl` | Same data as a pickled pandas DataFrame (generated). |

The three generated-output files are written to the working directory when the script is
run; they are not tracked as static assets.

## Requirements

- Python 3.9+
- `numpy`
- `pandas`
- `matplotlib`
- `mpmath`

Install with:

```bash
pip install numpy pandas matplotlib mpmath
```

## Usage

```bash
python gram_dirac_analysis.py
```

This will:

1. Build the Gramian $X_N$ ($N = 100$) for a single Dirac control at $x_0 = 0.5$.
2. Build the Gramian $X_N$ for $M = 10$ equally-weighted Dirac controls placed on an evenly
   spaced grid in $(0.05, 0.95)$.
3. Compute and sort the eigenvalues $\mu_k(X_N)$ for both cases.
4. Compute zeta-based upper bounds for $\beta \in \{0.51, 0.75, 0.99\}$.
5. Save two comparison plots (full range and zoomed) as PDFs.
6. Export the spectral data table as CSV and pickle.
7. Print the numerical norm estimates to stdout.

## Configuration

Key parameters are set at the top of `gram_dirac_analysis.py`:

| Parameter | Meaning | Default |
|---|---|---|
| `N` | Truncation dimension of the Gramian | `100` |
| `X0` | Location of the single Dirac control | `0.5` |
| `M` | Number of Dirac controls (multi case) | `10` |
| `EPS` | Boundary margin for Dirac placement | `0.05` |
| `BETAS` | Admissibility exponents to test | `[0.51, 0.75, 0.99]` |

Edit these values directly in the script to explore other configurations.

## Notes / Caveats

- The multi-Dirac norm bound (`M * single-Dirac bound`) is a crude estimate that does not
  account for possible refinements from the specific placement or weighting of the Diracs;
  see the corresponding comment in the code.
- Eigenvalues are clipped to $[0, \infty)$ after diagonalization to suppress negligible
  negative numerical noise (the true spectrum is provably non-negative).

## This file corresponds to the numerical implementation of 
Grubišić, L., Lazar, M., & Muñoz, J. R. (2026). Schatten norm estimates for Lyapunov gramians in operator scales. Applied Mathematics Letters, 179, 109968. https://doi.org/10.1016/j.aml.2026.109968
