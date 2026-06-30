"""
Peak OSEE vs N for fixed string lengths L.
 
For each L in L_list and each N in N_list, computes the peak (maximum over
all cuts L_A) of the averaged page curve S_bar(ell_A), and plots it
together with the theoretical upper bound Lln2.
 
This shows that the peak converges monotonically to Lln2 from below,
with convergence slower for larger L.
 
Output:
    peak_osee_vs_N.png
"""
 
import numpy as np
import matplotlib.pyplot as plt
from page_curve import (
    build_string_M, haar_random_orthogonal, evolve_M,
    restrict_M, osee_from_M
)
 
 
def compute_peak(N, L, n_samples, rng):
    """
    Compute mean and standard error of the peak OSEE over n_samples
    Haar-random rotations R in SO(N), for a string of length L.
 
    The peak is max_{L_A} S(L_A) for each sample, then averaged.
 
    Args:
        N: number of physical sites
        L: string length (must satisfy L <= N)
        n_samples: number of independent Haar-random R to average over
        rng: numpy.random.Generator
 
    Returns:
        mean_peak: mean of peak OSEE over samples
        sem_peak:  standard error of the mean (std / sqrt(n_samples))
    """
    assert L <= N, f"L={L} must be <= N={N}"
 
    # Build string covariance once (independent of R)
    M_string = build_string_M(N, range(1, L + 1))
 
    peaks = np.zeros(n_samples)
    for k in range(n_samples):
        R = haar_random_orthogonal(N, rng=rng)
        M_ev = evolve_M(M_string, R, N)
        # Compute S(L_A) for all cuts and take the maximum
        S_vals = np.array(
            [0.0] +
            [osee_from_M(restrict_M(M_ev, N, la)) for la in range(1, N + 1)]
        )
        peaks[k] = S_vals.max()
 
    return peaks.mean(), peaks.std() / np.sqrt(n_samples)
 
 
def plot_peak_vs_N(L_list, N_list, n_samples=80, seed=99):
    """
    For each L in L_list, compute and plot the peak OSEE as a function
    of N, together with the bound L*ln2 (dashed line, same color).
 
    Args:
        L_list:   list of string lengths to show
        N_list:   list of system sizes N to scan
        n_samples: number of Haar-random samples per (N, L) pair
        seed:     random seed for reproducibility
 
    Returns:
        fig, ax: matplotlib figure and axes
    """
    rng = np.random.default_rng(seed)
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(L_list)))
 
    fig, ax = plt.subplots(figsize=(8, 5))
 
    for L, color in zip(L_list, colors):
        valid_N = [N for N in N_list if N >= L]
        means, sems = [], []
        for N in valid_N:
            m, e = compute_peak(N, L, n_samples, rng)
            means.append(m)
            sems.append(e)
            print(f"  L={L}, N={N}: peak = {m:.4f} +/- {e:.4f}")
 
        # Data points with error bars
        ax.errorbar(valid_N, means, yerr=sems,
                    marker='o', markersize=4, color=color,
                    capsize=3, label=f'$L={L}$')
 
        # Theoretical bound Lln2 (dashed, same color)
        ax.axhline(L * np.log(2), color=color, linestyle='--',
                   linewidth=0.9, alpha=0.7)
 
    ax.set_xlabel(r'$N$', fontsize=12)
    ax.set_ylabel(r'$\max_{\ell_A}\,\bar{S}(\ell_A)$', fontsize=12)
    ax.set_title(
        f'Peak OSEE vs $N$, fixed $L$, averaged over {n_samples} random $R$\n'
        f'(dashed lines = bound $L\\ln 2$)',
        fontsize=11
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig, ax
 
 
# MAIN
if __name__ == "__main__":
 
    L_list   = [2, 4, 6, 8]
    N_list   = [8, 12, 16, 22, 30, 40, 60, 80]
    n_samples = 80
    seed      = 99
 
    print(f"Computing peak OSEE vs N for L in {L_list}, N in {N_list}")
    print(f"n_samples={n_samples}, seed={seed}\n")
 
    fig, ax = plot_peak_vs_N(L_list, N_list, n_samples=n_samples, seed=seed)
 
    outfile = "peak_osee_vs_N.png"
    fig.savefig(outfile, dpi=150)
    print(f"\nSaved {outfile}")