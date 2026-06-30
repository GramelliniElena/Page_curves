"""
Comparison between the OSEE computed via the covariance-matrix pipeline
and ensemble B from Ex8_Solution.pdf.

Ex8 setup: N=8 physical Majorana modes,
A={1,2,3,4}, B={5,6,7,8}, initial string O = psi_1 psi_2 psi_3 psi_4 (L=4,
entirely in A), R Haar-random in O(8), 500 samples.
Reported value: mean OSEE (ensemble B) = 1.433.

Here we use N=8 (sites, each carrying species a,b in the Choi formalism,
corresponding to the 8 physical Majoranas of Ex8), string psi_1...psi_4,
L_A=4, averaged over 500 independent random R, and compare the mean against 1.433.
"""

import numpy as np
from page_curve import (
    build_string_M, haar_random_orthogonal, evolve_M, restrict_M, osee_from_M
)

N = 8
L = 4
L_A = 4
n_samples = 500

rng = np.random.default_rng(0)
M_string = build_string_M(N, range(1, L + 1))

values = np.zeros(n_samples)
for k in range(n_samples):
    R = haar_random_orthogonal(N, rng=rng)
    M_evolved = evolve_M(M_string, R, N)
    M_A = restrict_M(M_evolved, N, L_A)
    values[k] = osee_from_M(M_A)

print(f"N={N}, L={L}, L_A={L_A}, n_samples={n_samples}")
print(f"Mean OSEE  : {values.mean():.4f}   (Ex8 ensemble B: 1.433)")
print(f"Std  OSEE  : {values.std():.4f}")
print(f"Min  OSEE  : {values.min():.4f}")
print(f"Max  OSEE  : {values.max():.4f}")