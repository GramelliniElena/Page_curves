"""
OSEE (Operator Space Entanglement Entropy) for Majorana strings
evolved under Gaussian unitaries.

Convention (from Ex5_Solution.pdf):
- Two species of Majorana: psi^a_j, psi^b_j, j=1,...,N; total of 2N Majoranas, where N is even (physical case)
- Single Clifford algebra: {psi^alpha_j, psi^beta_k} = 2 delta_jk delta_alphabeta
- Reference Choi state |I^ab>> defined as vacuum of c_j = psi^a_j - i*psi^b_j, j=1,..,N
- Covariance matrix: M^{alphabeta}_{jk} = -(i/2) <<O_U| [psi^alpha_j, psi^beta_k] |O_U>>
- Index layout of M (size 2N x 2N):
    rows/cols 0..N-1   -> species a
    rows/cols N..2N-1  -> species b

Gaussian unitary evolution:
    U_G(psi) psi_1...psi_L U_G^dagger(psi) |I>> = i U_G(psi) U_G(psibar) |nu>>
so a Gaussian unitary acting on the physical Majoranas rotates BOTH
species (a and b) by the SAME orthogonal matrix R in O(N). R is taken
Haar-random on O(N) (a "random Gaussian unitary"):
    psi^a_j -> sum_k R_jk psi^a_k ,   psi^b_j -> sum_k R_jk psi^b_k
Hence at the level of the covariance matrix M (size 2N x 2N):
    M -> O M O^T ,   O = blockdiag(R, R) .

Entropy formula: for a restricted covariance
matrix Gamma_A, bring it to canonical (Williamson) form
    Q Gamma_A Q^T = direct_sum_r [[0, nu_r], [-nu_r, 0]] ,  |nu_r| <= 1
and the entropy is
    S = sum_r h2( (1+nu_r)/2 ) ,   h2(p) = -p log p - (1-p) log(1-p).
The entropy formula only depends on the eigenvalues nu_r of the antisymmetric matrix up to an overall
sign flip nu_r -> -nu_r, which does not change S (h2 is symmetric under
nu -> -nu). So we can apply this canonical-form/entropy machinery
directly to M (or to -M), the OSEE is identical either way (in other notes there is a + sign in front of M definition,
it is the same as in this case).
"""

import numpy as np
from scipy.linalg import schur, block_diag


def build_M0(N):
    """
    Covariance matrix of the vacuum Choi state |I^ab>>.
    Size: (2N x 2N), block structure [[0, -I], [+I, 0]]; check explicitly this
    structure using the definition of M, vacuum |I^ab>>, and the Clifford algebra.

        <psi^a_j psi^a_k> = delta_jk  =>  M^aa = 0
        <psi^b_j psi^b_k> = delta_jk  =>  M^bb = 0
        <psi^a_j psi^b_k> = -i*delta_jk  =>  M^ab = -I_N
        <psi^b_j psi^a_k> = +i*delta_jk  =>  M^ba = +I_N
    """
    M = np.zeros((2*N, 2*N))
    M[:N, N:] = -np.eye(N)   # M^ab = -I
    M[N:, :N] = +np.eye(N)   # M^ba = +I
    return M


def check_M0(N):
    """
    Sanity checks for build_M0; quite trivial, but for simplicity with the updated cases:
    1. antisymmetry: verifies M^ab and M^ba have consistent signs
    2. M^T M = I: verifies the reference state is a pure Gaussian state
    Block structure checks omitted: trivially true by construction.
    """
    M = build_M0(N)
 
    print(f"check_M0  N={N}")
    print(f"  antisymmetry error  : {np.max(np.abs(M + M.T)):.2e}  (expect 0)")
    print(f"  M^T M = I error     : {np.max(np.abs(M.T @ M - np.eye(2*N))):.2e}  (expect 0)")
    print()


def update_one_majorana(M, m, N):
    """
    Update the covariance matrix after applying psi^a_{m+1} to the Choi state.

    From Clifford algebra (p = psi^a_{m+1}, p^2=1, {p,psi}=0 for other modes):
      p [psi^alpha_j, psi^beta_k] p = +[...] if neither index equals (m+1,a)
                                     = -[...] if exactly one index equals (m+1,a)
    This gives M^{(m+1)} = S_p M^{(m)} S_p,
    where S_p = diag(..., -1, ...) with -1 at index p=m (species-a block).
    """
    p = m
    S = np.ones(2*N)
    S[p] = -1
    M_new = S[:, None] * M * S[None, :]
    return M_new


def check_update_L1(N, i):
    M0 = build_M0(N)
    M1 = update_one_majorana(M0, i-1, N)
    diag_vals = -np.ones(N)
    diag_vals[i-1] = +1
    Mab_expected = np.diag(diag_vals)
    print(f"check_update_L1  N={N}, i={i}")
    print(f"  M^ab matches Example 1    : {np.allclose(M1[:N, N:], Mab_expected)}")
    print(f"  antisymmetry              : {np.allclose(M1 + M1.T, 0)}")
    print(f"  M^T M = I (pure Gaussian) : {np.allclose(M1.T @ M1, np.eye(2*N))}")
    print()

def check_update_L2(N, i, l):
    assert i != l
    M0 = build_M0(N)
    M1 = update_one_majorana(M0, i-1, N)
    M2 = update_one_majorana(M1, l-1, N)
    diag_vals = -np.ones(N)
    diag_vals[i-1] = +1
    diag_vals[l-1] = +1
    Mab_expected = np.diag(diag_vals)
    print(f"check_update_L2  N={N}, i={i}, l={l}")
    print(f"  M^ab got      : {np.diag(M2[:N, N:])}")
    print(f"  M^ab expected : {np.diag(Mab_expected)}")
    print(f"  matches       : {np.allclose(M2[:N, N:], Mab_expected)}")
    print(f"  antisymmetry  : {np.allclose(M2 + M2.T, 0)}")
    print(f"  M^T M = I     : {np.allclose(M2.T @ M2, np.eye(2*N))}")
    print()


# Gaussian unitary evolution
def haar_random_orthogonal(N, rng=None, det_plus_one=True):
    """
    Sample R uniformly (Haar measure) from O(N), via QR decomposition of a
    Ginibre (iid - independent and identically distributed - standard normal entries) random matrix Z = QR_, with the
    sign of each column fixed by the sign of the corresponding diagonal
    entry of R_ (this removes the bias in QR convention and
    makes Q Haar-distributed on O(N); see Mezzadri, "How to generate random
    matrices from the classical compact groups", for the standard proof).

    Args:
        N: dimension (number of physical Majorana sites per one species)
        rng: numpy.random.Generator
        det_plus_one: if True, flip the sign of one column when det(Q)=-1
            so that R in SO(N)
    Returns:
        R: (N x N) orthogonal matrix, Haar-random on SO(N)
    """
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.normal(size=(N, N))
    Q, Rmat = np.linalg.qr(Z)
    Q = Q * np.sign(np.diag(Rmat))[None, :]
    if det_plus_one and np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def evolve_M(M, R, N):
    """
    Evolve the full covariance matrix M (2N x 2N) under a Gaussian unitary
    that rotates the physical Majoranas by R (N x N, orthogonal).

    Both species (a and b) rotate by the SAME R, since the Choi state
    evolves as U_G(psi) U_G(psibar) |nu>>:
        O = blockdiag(R, R),   M -> O M O^T.
    """
    O = block_diag(R, R)
    return O @ M @ O.T


def check_evolution_preserves_faf(N, rng=None):
    """
    Sanity check: evolving vacuum state or any Majorana string state under a Gaussian
    unitary must preserve M^T M = I and antisymmetry.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    M0 = build_M0(N)
    R = haar_random_orthogonal(N, rng=rng)
    M_t = evolve_M(M0, R, N)
    print(f"check_evolution_preserves_faf  N={N}")
    print(f"  R orthogonal (R^T R = I) : {np.allclose(R.T @ R, np.eye(N))}")
    print(f"  M(t) antisymmetric       : {np.allclose(M_t + M_t.T, 0)}")
    print(f"  M(t)^T M(t) = I          : {np.allclose(M_t.T @ M_t, np.eye(2*N))}")
    print()


# OSEE from the restricted covariance matrix
def h2(p, eps=1e-12):
    """Binary entropy h2(p) = -p log p - (1-p) log(1-p), safe at p=0,1."""
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def restrict_M(M, N, L_A):
    """
    Restrict the full covariance matrix M (2N x 2N) to the spatial subregion
    A = first L_A physical sites. Each physical site carries both species
    (psi^a_j and psi^b_j), so the kept indices are
        {0, ..., L_A-1} (species a)  union  {N, ..., N+L_A-1} (species b).

    Returns:
        M_A: (2 L_A x 2 L_A) principal submatrix of M.
    """
    idx = np.concatenate([np.arange(L_A), np.arange(N, N + L_A)])
    return M[np.ix_(idx, idx)]


def canonical_nu(M_A):
    """
    Bring the real antisymmetric matrix M_A to Williamson block form
    via a real Schur decomposition, and extract the canonical covariance
    eigenvalues nu_r (one per 2x2 block).

    scipy.linalg.schur(M_A, output='real') returns an orthogonal Q and a
    block-upper-triangular T such that M_A = Q T Q^T; for a real antisymmetric
    matrix T is block-diagonal with 2x2 blocks [[0, nu_r], [-nu_r, 0]],
     which is precisely the canonical form needed.

    Returns:
        nu: 1D array of canonical eigenvalues nu_r, one per mode (length = dim/2)
    """
    T, Q = schur(M_A, output='real')
    dim = M_A.shape[0]
    nu = np.array([T[2*r, 2*r + 1] for r in range(dim // 2)])
    return nu


def osee_from_M(M_A):
    """
    Operator space entanglement entropy of the restricted covariance matrix
    M_A, using S = sum_r h2((1+nu_r)/2).

    Note: the entropy formula is invariant under nu_r -> -nu_r (h2 is
    symmetric about 1/2), so it does not matter whether M_A follows the
    Ex5 sign convention or the Bayona one.
    """
    nu = canonical_nu(M_A)
    nu = np.clip(nu, -1.0, 1.0)
    return np.sum(h2((1 + nu) / 2))


def osee_vs_LA(M, N):
    """
    Compute the OSEE as a function of the subsystem size L_A = 0, 1, ..., N.

    Returns:
        LAs: array [0, 1, ..., N]
        S:   OSEE for each L_A (S[0] = S[N] = 0 for a pure global state)
    """
    LAs = np.arange(0, N + 1)
    S = np.zeros_like(LAs, dtype=float)
    for k, L_A in enumerate(LAs):
        if L_A == 0:
            S[k] = 0.0
            continue
        M_A = restrict_M(M, N, L_A)
        S[k] = osee_from_M(M_A)
    return LAs, S


def check_osee_max_at_M0(N):
    """
    Sanity check: the reference Choi state |I^ab>> has zero OSEE for every cut L_A.

    Reason: M^ab = -I has all singular values equal to 1,
    so h2((1+1)/2) = h2(1) = 0 for every mode.
    """
    M0 = build_M0(N)
    LAs, S = osee_vs_LA(M0, N)
    print(f"check_osee_max_at_M0  N={N}")
    print(f"  S(L_A) for reference state |I^ab>>: {np.round(S, 6)}")
    print(f"  all zero (expect True)             : {np.allclose(S, 0)}")
    print()


# Page curves
def build_string_M(N, sites):
    """
    Build the covariance matrix of the Choi state of the Majorana string
    operator O = psi_{sites[0]} psi_{sites[1]} ... (all 1-based, distinct),
    starting from the reference vacuum M0 and applying update_one_majorana
    once per site in the string.
    """
    M = build_M0(N)
    for s in sites:
        M = update_one_majorana(M, s - 1, N)
    return M


def page_curve_data(N, string_lengths, rng=None):
    """
    Compute OSEE(L_A) for Majorana strings of different lengths L (each string
    is psi_1 psi_2 ... psi_L, the first L physical sites), all evolved under
    the SAME fixed Haar-random Gaussian unitary R. This reproduces the family
    of curves in the page-curve sketch: one curve per string length L, x-axis
    L_A = size of the bipartition.

    Args:
        N: number of physical sites
        string_lengths: iterable of string lengths L (1 <= L <= N)
        rng: numpy.random.Generator, optional (fixes the random unitary)

    Returns:
        LAs: array [0, ..., N]
        curves: dict {L: S(L_A) array} for each L in string_lengths
    """
    if rng is None:
        rng = np.random.default_rng(0)
    R = haar_random_orthogonal(N, rng=rng)

    curves = {}
    LAs = None
    for L in string_lengths:
        M_string = build_string_M(N, range(1, L + 1))
        M_evolved = evolve_M(M_string, R, N)
        LAs, S = osee_vs_LA(M_evolved, N)
        curves[L] = S
    return LAs, curves

def plot_page_curves(LAs, curves, title="OSEE page curve"):
    """Plot S(L_A) for each string length L in curves. Requires matplotlib."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for L, S in curves.items():
        ax.plot(LAs, S, marker='o', markersize=3, label=f"L={L}")
    ax.set_xlabel(r"$L_A$")
    ax.set_ylabel(r"$S_{\mathrm{OSEE}}$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def page_curve_data_averaged(N, string_lengths, n_samples=200, rng=None):
    """
    Same as page_curve_data, but averaged over n_samples independent
    Haar-random unitaries R, returning mean and standard deviation of
    S(L_A) for each string length L.

    Args:
        N: number of physical sites
        string_lengths: iterable of string lengths L (1 <= L <= N)
        n_samples: number of independent random R to average over
        rng: numpy.random.Generator, optional

    Returns:
        LAs: array [0, ..., N]
        mean_curves: dict {L: mean S(L_A) array}
        std_curves:  dict {L: std S(L_A) array} (population std over samples)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    LAs = np.arange(0, N + 1)
    samples = {L: np.zeros((n_samples, N + 1)) for L in string_lengths}

    # Build string covariance matrices once, outside the sampling loop
    # (M_string does not depend on R)
    M_strings = {L: build_string_M(N, range(1, L + 1)) for L in string_lengths}

    for n in range(n_samples):
        R = haar_random_orthogonal(N, rng=rng)
        for L in string_lengths:
            M_evolved = evolve_M(M_strings[L], R, N)
            _, S = osee_vs_LA(M_evolved, N)
            samples[L][n] = S

    mean_curves = {L: samples[L].mean(axis=0) for L in string_lengths}
    std_curves = {L: samples[L].std(axis=0) for L in string_lengths}
    return LAs, mean_curves, std_curves


def page_curve_single_N(N, L, n_samples=200, rng=None):
    """
    Averaged OSEE(L_A) for a single (N, L) pair, over n_samples independent
    Haar-random R.

    Returns:
        LAs: array [0, ..., N]
        mean_S: mean S(L_A) array
        std_S: std S(L_A) array
    """
    LAs, mean_curves, std_curves = page_curve_data_averaged(
        N, [L], n_samples=n_samples, rng=rng)
    return LAs, mean_curves[L], std_curves[L]


def page_curve_vs_N_fixed_L(N_list, L, n_samples=200, rng=None):
    """
    Compare page curves across different system sizes N, at FIXED absolute
    string length L (the same Majorana string psi_1...psi_L for every N).
    As N grows, the string becomes proportionally smaller, so the maximal
    entanglement (bounded by ~ L * log(2)) should saturate rather than grow
    with N.

    Returns:
        results: dict {N: (LAs, mean_S, std_S)}
    """
    if rng is None:
        rng = np.random.default_rng(0)
    results = {}
    for N in N_list:
        assert L <= N, f"L={L} must be <= N={N}"
        LAs, mean_S, std_S = page_curve_single_N(N, L, n_samples=n_samples, rng=rng)
        results[N] = (LAs, mean_S, std_S)
    return results


def page_curve_vs_N_fixed_ratio(N_list, L_fraction=0.5, n_samples=200, rng=None):
    """
    Compare page curves across different system sizes N, at FIXED ratio
    L/N (e.g. here the string always spans half the system). Here the string
    grows proportionally to N, so the (rescaled) curves are expected to
    look more self-similar than in the fixed-L case.

    Returns:
        results: dict {N: (LAs, mean_S, std_S, L)}
    """
    if rng is None:
        rng = np.random.default_rng(0)
    results = {}
    for N in N_list:
        L = max(1, round(L_fraction * N))
        LAs, mean_S, std_S = page_curve_single_N(N, L, n_samples=n_samples, rng=rng)
        results[N] = (LAs, mean_S, std_S, L)
    return results


def plot_vs_N_fixed_L(results, rescale_x=False, title="OSEE vs N (fixed L)",
                      marker='o'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for N, (LAs, mean_S, std_S) in results.items():
        x = LAs / N if rescale_x else LAs
        ax.plot(x, mean_S, marker=marker, markersize=3, label=f"N={N}")
    ax.set_xlabel(r"$L_A/N$" if rescale_x else r"$L_A$")
    ax.set_ylabel(r"$S_{\mathrm{OSEE}}$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_vs_N_fixed_ratio(results, rescale_x=False, title="OSEE vs N (fixed L/N)"):
    """Plot the fixed-ratio comparison, same options as plot_vs_N_fixed_L."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for N, (LAs, mean_S, std_S, L) in results.items():
        x = LAs / N if rescale_x else LAs
        ax.plot(x, mean_S, marker='o', markersize=3, label=f"N={N}, L={L}")
    ax.set_xlabel(r"$L_A/N$" if rescale_x else r"$L_A$")
    ax.set_ylabel(r"$S_{\mathrm{OSEE}}$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_vs_N_fixed_ratio_rescaled(results, title="OSEE vs N (fixed L/N), rescaled y"):
    """
    Two-panel version of plot_vs_N_fixed_ratio.
 
    Left panel  : raw S̄(L_A/N)       — raw
    Right panel : S̄(L_A/N) / N       — rescaled by system size
 
    With the rescaling the curves should collapse onto a single
    shape for large N.
 
    Args:
        results : dict {N: (LAs, mean_S, std_S, L)}
                  as returned by page_curve_vs_N_fixed_ratio
        title   : figure suptitle
 
    Returns:
        fig, (ax_left, ax_right)
    """
    import matplotlib.pyplot as plt
 
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
 
    for N, (LAs, mean_S, std_S, L) in results.items():
        x     = LAs / N
        label = f"$N={N}$, $L={L}$"
        ax_l.plot(x, mean_S,     marker='o', markersize=2, label=label)
        ax_r.plot(x, mean_S / N, marker='o', markersize=2, label=label)
 
    for ax, ylabel, subtitle in zip(
        [ax_l, ax_r],
        [r"$\bar{S}_{\rm OSEE}(\ell_A/N)$",
         r"$\bar{S}_{\rm OSEE}(\ell_A/N)\;/\;N$"],
        ["Raw OSEE",
         r"Rescaled $/N$"],
    ):
        ax.set_xlabel(r"$\ell_A / N$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(subtitle, fontsize=10)
        ax.legend(fontsize=7, ncol=2)
 
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig, (ax_l, ax_r)


def plot_page_curves_averaged(LAs, mean_curves, std_curves, n_samples,
                               title="OSEE page curve (averaged)"):
    """Plot mean S(L_A) +/- std error bars (std/sqrt(n_samples)) for each L."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for L, S in mean_curves.items():
        err = std_curves[L] / np.sqrt(n_samples)
        ax.errorbar(LAs, S, yerr=err, marker='o', markersize=3,
                     capsize=2, label=f"L={L}")
    ax.set_xlabel(r"$L_A$")
    ax.set_ylabel(r"$S_{\mathrm{OSEE}}$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

# MAIN
if __name__ == "__main__":

    print("=" * 55)
    print("=== Validating build_M0 ===")
    print("=" * 55 + "\n")
    for N in [2, 3, 4]:
        check_M0(N)

    print("=" * 55)
    print("=== Validating update_one_majorana (L=1) ===")
    print("=" * 55 + "\n")
    for N in [3, 4]:
        for i in [1, 2, N]:
            check_update_L1(N, i)

    print("=" * 55)
    print("=== Validating update_one_majorana twice (L=2) ===")
    print("=" * 55 + "\n")
    for N in [3, 4]:
        for i, l in [(1, 2), (1, 3), (2, 3)]:
            if l <= N:
                check_update_L2(N, i, l)

    print("=" * 55)
    print("=== Validating Gaussian unitary evolution ===")
    print("=" * 55 + "\n")
    for N in [4, 6]:
        check_evolution_preserves_faf(N)

    print("=" * 55)
    print("=== Validating OSEE on reference state ===")
    print("=" * 55 + "\n")
    for N in [4, 6]:
        check_osee_max_at_M0(N)

    print("=" * 55)
    print("=== Page curve example (single random R) ===")
    print("=" * 55 + "\n")
    N = 20
    string_lengths = [2, 4, 6, 8, 10]          # strings O = psi_1...psi_L for each L
    rng = np.random.default_rng(42)
    LAs, curves = page_curve_data(N, string_lengths, rng=rng)
    for L, S in curves.items():
        print(f"L={L} | S(L_A) = {np.round(S, 3)}")

    try:
        fig = plot_page_curves(LAs, curves, title=f"OSEE page curve, N={N}")
        fig.savefig("page_curve_example.png", dpi=150)
        print("\nSaved plot to page_curve_example.png")
    except ImportError:
        print("\nmatplotlib not available, skipping plot.")

    print("\n" + "=" * 55)
    print("=== Page curve example (averaged over many random R) ===")
    print("=" * 55 + "\n")
    n_samples = 200
    rng2 = np.random.default_rng(123)
    LAs, mean_curves, std_curves = page_curve_data_averaged(
        N, string_lengths, n_samples=n_samples, rng=rng2)
    for L in string_lengths:
        print(f"L={L} | mean S(L_A) = {np.round(mean_curves[L], 3)}")
        print(f"     | std  S(L_A) = {np.round(std_curves[L], 3)}")

    try:
        fig2 = plot_page_curves_averaged(
            LAs, mean_curves, std_curves, n_samples,
            title=f"OSEE page curve, N={N}, averaged over {n_samples} random R")
        fig2.savefig("page_curve_averaged.png", dpi=150)
        print(f"\nSaved plot to page_curve_averaged.png")
    except ImportError:
        print("\nmatplotlib not available, skipping plot.")

    print("\n" + "=" * 55)
    print("=== Varying N at FIXED string length L ===")
    print("=" * 55 + "\n")
    N_list = [4, 10, 16, 22, 28, 40, 60, 80, 100]
    L_fixed = 4
    n_samples_N = 100
    rng3 = np.random.default_rng(7)
    N_list_fixedL = [10, 16, 22, 28, 40, 60, 80, 100]
    results_fixedL = page_curve_vs_N_fixed_L(N_list_fixedL, L_fixed,  n_samples=n_samples_N, rng=rng3)
    for N, (LAs, mean_S, std_S) in results_fixedL.items():
        print(f"N={N:3d} | max mean S(L_A) = {mean_S.max():.4f}")

    try:
        fig3 = plot_vs_N_fixed_L(results_fixedL, rescale_x=False,
                          title=f"OSEE vs N, fixed L={L_fixed}",
                          marker=None)
        fig3.savefig("page_curve_vs_N_fixedL.png", dpi=150)
        print("Saved page_curve_vs_N_fixedL.png")
    except ImportError:
        print("matplotlib not available, skipping plot.")

    print("\n" + "=" * 55)
    print("=== Varying N at FIXED ratio L/N = 0.5 ===")
    print("=" * 55 + "\n")
    rng4 = np.random.default_rng(8)
    N_list_fixedratio = [10, 20, 30, 40, 60, 80]
    results_fixedratio = page_curve_vs_N_fixed_ratio(N_list_fixedratio, L_fraction=0.5, n_samples=n_samples_N, rng=rng4)
    for N, (LAs, mean_S, std_S, L) in results_fixedratio.items():
        print(f"N={N:3d}, L={L:3d} | max mean S(L_A) = {mean_S.max():.4f}")

    try:
        fig4 = plot_vs_N_fixed_ratio(results_fixedratio, rescale_x=True,
                                      title="OSEE vs N, fixed L/N=0.5 (rescaled x=L_A/N)")
        fig4.savefig("page_curve_vs_N_fixedratio_single.png", dpi=150)
        print("Saved page_curve_vs_N_fixedratio_single.png")
    except ImportError:
        print("matplotlib not available, skipping plot.")

    try:
        fig5, _ = plot_vs_N_fixed_ratio_rescaled(
            results_fixedratio,
            title="OSEE vs N, fixed L/N=0.5 — raw (left) and rescaled /N (right)"
        )
        fig5.savefig("page_curve_vs_N_fixedratio_rescaled.png", dpi=150)
        print("Saved page_curve_vs_N_fixedratio_rescaled.png")
    except ImportError:
        print("matplotlib not available, skipping plot.")
 
