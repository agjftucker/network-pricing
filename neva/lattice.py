import json
from itertools import combinations
import numpy as np

def load_coefficients():
    with open('pseudospectral.json') as f:
        coeff = json.load(f)
    return {k : np.array(v) for k, v in coeff}

GRID_LEN = 8
COEFF = load_coefficients()

def differentiate_boundary(B, v):
    """Calculate the first and second derivatives of v with respect to x.

    B is for bank labels, a tuple of integers between 0 and 23 sorted in ascending order.
    v is for valuation, defined on a grid with dimension matching the length of B.
    Find some notes on the use of einsum at https://github.com/agjftucker/neva/wiki/Pseudospectral-Solution#boundary-derivatives.
    """
    m, n = 24, 25
    Dx_v = {}
    for i in B:
        for s, j in enumerate(B):
            Dx_v[i, j] = (np.einsum(COEFF['first_deriv_edge'], (j,), v[B[:s] + B[s+1:]][i], B[:s] + B[s+1:])
                          + np.einsum(COEFF['first_deriv_interior'], (j, m), v[B][i], B[:s] + (m,) + B[s+1:]))
    Dxx_v = {}
    for i in B:
        for s, j in enumerate(B):
            Dxx_v[i, j, j] = (np.einsum(COEFF['second_deriv_edge'], (j,), v[B[:s] + B[s+1:]][i], B[:s] + B[s+1:])
                              + np.einsum(COEFF['second_deriv_interior'], (j, n), v[B][i], B[:s] + (n,) + B[s+1:]))
        for (s, j), (t, k) in combinations(enumerate(B), 2):
            Dxx_v[i, j, k] = (np.einsum(COEFF['cross_deriv_corner'], (j, k), v[B[:s] + B[s+1:t] + B[t+1:]][i], B[:s] + B[s+1:t] + B[t+1:])
                              + np.einsum(COEFF['cross_deriv_edge'], (j, k, m), v[B[:t] + B[t+1:]][i], B[:s] + (m,) + B[s+1:t] + B[t+1:])
                              + np.einsum(COEFF['cross_deriv_edge'], (k, j, n), v[B[:s] + B[s+1:]][i], B[:s] + B[s+1:t] + (n,) + B[t+1:])
                              + np.einsum(COEFF['cross_deriv_interior'], (j, k, m, n), v[B][i], B[:s] + (m,) + B[s+1:t] + (n,) + B[t+1:]))
    return Dx_v, Dxx_v
