"""
gram_dirac_analysis.py

Numerical study of the observability/controllability Gramian X_N for the
1-D Dirichlet Laplacian A = -d^2/dx^2 on (0,1), controlled through Dirac
masses (point actuators/sensors).

Two configurations are compared:

    1. A single Dirac control located at x0.
    2. A sum of M equally-weighted Dirac controls at points {x_m} in (0,1).

For each configuration we:
    - build the discretized Gramian X_N in the sine eigenbasis of A,
    - compute its eigenvalues mu_k(X_N),
    - compare them against the theoretical upper bound

          mu_k(X_N) <= (1/2) * lambda_k^{-(1-beta)} * || |A|^{-beta/2} B ||_{S_inf}^2

      for several admissibility exponents beta in (1/2, 1), where the
      right-hand side norm is estimated via the Riemann zeta function.

Outputs:
    - single_vs_multi_dirac_bounds.pdf       (full-range comparison plot)
    - single_vs_multi_dirac_bounds_zoom.pdf  (zoom on the first 20 indices)
    - figure1_single_vs_multi_dirac_data.csv (tabulated spectral data)
    - figure1_single_vs_multi_dirac_data.pkl (same data, pickled DataFrame)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpmath import zeta

# ============================================================
# Parameters and discretization
# ============================================================

N = 100                    # Truncation dimension for the Gramian X_N
X0 = 0.5                   # Location x0 in (0,1) of the single Dirac control

M = 10                     # Number of Dirac controls in the multi-actuator case
EPS = 0.05                 # Margin kept away from the boundary when placing Diracs
X_POINTS = np.linspace(EPS, 1 - EPS, M)   # Dirac locations {x_m} in (0,1)
GAMMA = np.ones(M) / np.sqrt(M)           # Normalized weights for the multi-Dirac case

BETAS = [0.51, 0.75, 0.99]                # Admissibility exponents beta in (1/2, 1)


# ============================================================
# Discrete Gramians in the eigenbasis of A = -Delta
# ============================================================

def build_single_dirac_gramian(n: int, x0: float) -> np.ndarray:
    """
    Build the N x N Gramian matrix X_N associated with a single Dirac
    control delta_{x0}, expressed in the sine eigenbasis of the Dirichlet
    Laplacian:

        X_N[i, j] = (2 / pi^2) * sin(pi*i*x0) * sin(pi*j*x0) / (i^2 + j^2)

    Parameters
    ----------
    n : truncation dimension.
    x0 : location of the Dirac control in (0, 1).

    Returns
    -------
    (n, n) ndarray, symmetric positive semi-definite Gramian.
    """
    j = np.arange(1, n + 1, dtype=float)
    I, J = np.meshgrid(j, j, indexing="ij")
    return (2.0 / np.pi**2) * (np.sin(np.pi * I * x0) * np.sin(np.pi * J * x0)) / (I**2 + J**2)


def build_multi_dirac_gramian(n: int, x_points: np.ndarray) -> np.ndarray:
    """
    Build the N x N (unnormalized) Gramian matrix X_N associated with a sum
    of Dirac controls sum_m delta_{x_m}, expressed in the sine eigenbasis of
    the Dirichlet Laplacian. The normalization by 1/M^2 assumes `x_points`
    has length M.

    Parameters
    ----------
    n : truncation dimension.
    x_points : array of Dirac locations in (0, 1).

    Returns
    -------
    (n, n) ndarray, symmetric positive semi-definite Gramian.
    """
    m = len(x_points)
    j = np.arange(1, n + 1, dtype=float)
    I, J = np.meshgrid(j, j, indexing="ij")

    numerator = np.zeros((n, n))
    for x in x_points:
        numerator += np.sin(np.pi * I * x) * np.sin(np.pi * J * x)

    return (2.0 / (m * np.pi) ** 2) * numerator / (I**2 + J**2)


def sorted_nonnegative_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """
    Return the eigenvalues of a symmetric matrix, sorted in decreasing
    order and clipped to [0, inf) to correct for negligible negative
    numerical noise (the true spectrum is non-negative).
    """
    eigs = np.sort(np.linalg.eigvalsh(matrix))[::-1]
    return np.clip(eigs, 0.0, None)


def compute_norm_and_bound(betas, m, lambda_k):
    """
    For each beta in `betas`, estimate the Sinfinity-norm bound

        || |A|^{-beta/2} B ||_{S_inf}^2 <= (2 / pi^{2*beta}) * zeta(2*beta)

    for the single-Dirac case, and the (crude, unscaled) analogue
    M * (single-Dirac bound) for the M-Dirac case. From these, derive the
    corresponding spectral upper bounds on mu_k(X_N):

        mu_k(X_N) <= (1/2) * lambda_k^{-(1-beta)} * || |A|^{-beta/2} B ||_{S_inf}^2

    Returns
    -------
    norms_single, norms_multi, bounds_single, bounds_multi : dict[beta -> value/array]
    """
    norms_single, norms_multi = {}, {}
    bounds_single, bounds_multi = {}, {}

    for beta in betas:
        sinf_sq_single = (2.0 / np.pi ** (2 * beta)) * float(zeta(2 * beta))
        sinf_sq_multi = m * sinf_sq_single  # crude estimate, no M-scaling refinement

        norms_single[beta] = sinf_sq_single
        norms_multi[beta] = sinf_sq_multi

        bounds_single[beta] = 0.5 * lambda_k ** (-(1.0 - beta)) * sinf_sq_single
        bounds_multi[beta] = 0.5 * lambda_k ** (-(1.0 - beta)) * sinf_sq_multi

    return norms_single, norms_multi, bounds_single, bounds_multi


# ============================================================
# Plotting
# ============================================================

def plot_spectral_comparison(k, eigs_single, eigs_multi, bounds_single, betas,
                              xlim=None, ylim=None, title_suffix="",
                              output_path="single_vs_multi_dirac_bounds.pdf"):
    """
    Plot eigenvalues mu_k(X_N) for the single- and multi-Dirac Gramians
    together with the theoretical upper bounds for each beta, on a
    semilog-y scale. Saves the figure to `output_path` and displays it.
    """
    plt.figure(figsize=(10, 5.8))

    plt.semilogy(k, eigs_single, '-+', linewidth=3,
                 label=r'Single Dirac: $\mu_k(X_N)$')
    plt.semilogy(k, eigs_multi, '-*', linewidth=3,
                 label=rf'$M={M}$ Diracs: $\mu_k(X_N)$')

    colors = ['tab:red', 'tab:blue', 'tab:green']
    for beta, col in zip(betas, colors):
        plt.semilogy(k, bounds_single[beta], linestyle='--', color=col, linewidth=2,
                     label=rf'Upper bound, $\beta={beta}$')

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel(r'Index $k$')
    plt.ylabel(r'Magnitude (log scale)')
    plt.title(rf'Eigenvalues of $X_N$ and spectral bounds{title_suffix}')
    plt.grid(True, which='both', linestyle='--', alpha=0.35)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    # --- Build Gramians ---
    x_single = build_single_dirac_gramian(N, X0)
    x_raw = build_multi_dirac_gramian(N, X_POINTS)
    x_multi = np.sum(GAMMA**2) * x_raw

    # --- Eigenvalues mu_k(X_N) ---
    eigs_single = sorted_nonnegative_eigenvalues(x_single)
    eigs_multi = sorted_nonnegative_eigenvalues(x_multi)

    k = np.arange(1, N + 1)
    lambda_k = (np.pi * k) ** 2  # Eigenvalues lambda_k of -Delta

    # --- Theoretical upper bounds ---
    norms_single, norms_multi, bounds_single, bounds_multi = compute_norm_and_bound(
        BETAS, M, lambda_k
    )

    # --- Plots ---
    plot_spectral_comparison(
        k, eigs_single, eigs_multi, bounds_single, BETAS,
        output_path="single_vs_multi_dirac_bounds.pdf",
    )
    plot_spectral_comparison(
        k, eigs_single, eigs_multi, bounds_single, BETAS,
        xlim=(0, 20), ylim=(1e-6, 1e1),
        title_suffix=" (zoom to first 20 eigenvalues)",
        output_path="single_vs_multi_dirac_bounds_zoom.pdf",
    )

    # --- Print numerical norm estimates ---
    print("\nBounds for || |A|^{-beta/2} B ||_{S_inf}^2 (zeta-based):")
    for beta in BETAS:
        print(f"beta={beta:.2f} | single <= {norms_single[beta]:.6e} | "
              f"{M}-Dirac <= {norms_multi[beta]:.6e}")

    # --- Tabulate and export spectral data ---
    data = {
        "k": k,
        "lambda_k": lambda_k,
        "mu_k_single_dirac": eigs_single,
        "mu_k_multi_dirac_M10": eigs_multi,
    }
    for beta in BETAS:
        data[f"bound_single_beta_{beta}"] = bounds_single[beta]

    df = pd.DataFrame(data)
    df.to_csv("figure1_single_vs_multi_dirac_data.csv", index=False)
    df.to_pickle("figure1_single_vs_multi_dirac_data.pkl")

    print("\nSpectral data table exported:")
    print(df.head())


if __name__ == "__main__":
    main()
