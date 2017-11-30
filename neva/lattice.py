import json
from itertools import repeat, combinations, combinations_with_replacement
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab
from .gbm import BankGBM, BankingSystemGBM

def load_coefficients():
    with open('neva/pseudospectral.json') as f:
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
            Dx_v[i, j] = ((0.0 if i == j else np.einsum(COEFF['first_deriv_edge'], (j,), v[B[:s] + B[s+1:]][i], B[:s] + B[s+1:]))
                          + np.einsum(COEFF['first_deriv_interior'], (j, m), v[B][i], B[:s] + (m,) + B[s+1:]))
    Dxx_v = {}
    for i in B:
        for s, j in enumerate(B):
            Dxx_v[i, j, j] = ((0.0 if i == j else np.einsum(COEFF['second_deriv_edge'], (j,), v[B[:s] + B[s+1:]][i], B[:s] + B[s+1:]))
                              + np.einsum(COEFF['second_deriv_interior'], (j, m), v[B][i], B[:s] + (m,) + B[s+1:]))
        for (s, j), (t, k) in combinations(enumerate(B), 2):
            Dxx_v[i, j, k] = ((0.0 if i in (j, k) else np.einsum(COEFF['cross_deriv_corner'], (j, k), v[B[:s] + B[s+1:t] + B[t+1:]][i], B[:s] + B[s+1:t] + B[t+1:]))
                              + (0.0 if i == k else np.einsum(COEFF['cross_deriv_edge'], (j, k, m), v[B[:t] + B[t+1:]][i], B[:s] + (m,) + B[s+1:t] + B[t+1:]))
                              + (0.0 if i == j else np.einsum(COEFF['cross_deriv_edge'], (k, j, n), v[B[:s] + B[s+1:]][i], B[:s] + B[s+1:t] + (n,) + B[t+1:]))
                              + np.einsum(COEFF['cross_deriv_interior'], (j, k, m, n), v[B][i], B[:s] + (m,) + B[s+1:t] + (n,) + B[t+1:]))
    return Dx_v, Dxx_v

class BankingSystemCorrelatedGBM(BankingSystemGBM):
    """Banking system whose banks have external assets that follow a correlated
    Geometric Brownian Motion."""
    @classmethod
    def with_covar_asset(cls, bnksys, covar_asset):
        """Create a banking system whose banks have external assets that follow
        a Geometric Brownian Motion by providing the covariance matrix of
        external assets.
        
        Parameters:
            bnksys (`BankingSystemAdjust`): banking system object
            covar_asset (dictionary of float): covariance matrix of external assets
        """
        banks_gbm = [BankGBM(bnk, sigma_asset=np.sqrt(covar_asset[bnk.name, bnk.name])) for bnk in bnksys]
        new_sys = cls._from_bankingsystem(banks_gbm, bnksys)
        bidx = {bnk.name: idx for idx, bnk in new_sys.banksbyid.items()}
        new_sys.covar = {(bidx[bank_i], bidx[bank_j]) : S_ij for (bank_i, bank_j), S_ij in covar_asset.items()}
        return new_sys

def solve_pde(B, S, z, L, eta, v, boundary_derivs):
    n_banks = len(B)
    axis = {k : t for t, k in enumerate(B)}
    grid_shape = tuple(repeat(GRID_LEN, n_banks))
    Dx_y = np.empty((n_banks, n_banks) + grid_shape)
    Dxx_y = {}
    for t, k in enumerate(B):
        not_k = B[:t] + B[t+1:]
        Dx_v, Dxx_v = boundary_derivs[not_k]
        Dx_y[t, t] = eta[k]
        fail_k = z[k] - np.sum(v[j] * L[j, k] for j in not_k)
        pinched = grid_shape[:t] + (1,) + grid_shape[t+1:]
        for m in not_k:
            Dx_y[t, axis[m]] = (-np.sum(Dx_v[j, m] * L[j, k] for j in not_k) / fail_k).reshape(pinched)
        for m, n in combinations_with_replacement(not_k, 2):
            Dxx_y[k, m, n] = (-(fail_k * np.sum(Dxx_v[j, m, n] * L[j, k] for j in not_k)
                                + (np.sum(Dx_v[j, m] * L[j, k] for j in not_k)
                                   * np.sum(Dx_v[j, n] * L[j, k] for j in not_k))) / (fail_k * fail_k)).reshape(pinched)
    Dy_x_ = np.moveaxis(np.linalg.inv(np.moveaxis(Dx_y, (0, 1), (-2, -1))), (-2, -1), (0, 1))
    def Dy_x(j, k):
        return Dy_x_[axis[j], axis[k]]
    def Dyy_x(i, k, l):
        return -np.sum(
            Dy_x(i, j) * (np.sum(Dy_x(m, k) * Dy_x(m, l) * Dxx_y[j, m, m] for m in B[:s] + B[s+1:])
                          + np.sum((Dy_x(m, k) * Dy_x(n, l) + Dy_x(n, k) * Dy_x(m, l)) * Dxx_y[j, m, n]
                                   for m, n in combinations(B[:s] + B[s+1:], 2)))
            for s, j in enumerate(B))
    P = {(i, j) : (np.sum(S[k, k] * Dy_x(i, k) * Dy_x(j, k) for k in B)
                   + np.sum(S[k, l] * (Dy_x(i, k) * Dy_x(j, l) + Dy_x(j, k) * Dy_x(i, l)) for k, l in combinations(B, 2)))
         for i, j in combinations_with_replacement(B, 2)}
    q = {i : (np.sum(S[k, k] * (Dy_x(i, k) - Dyy_x(i, k, k)) for k in B)
              - 2.0 * np.sum(S[k, l] * Dyy_x(i, k, l) for k, l in combinations(B, 2))) for i in B}
    def a(u):
        m, n = 24, 25
        u.resize(grid_shape)
        return (0.5 * np.sum(q[i] * np.einsum(COEFF['first_deriv_interior'], (i, m), u, B[:s] + (m,) + B[s+1:])
                             - P[i, i] * np.einsum(COEFF['second_deriv_interior'], (i, m), u, B[:s] + (m,) + B[s+1:]) for s, i in enumerate(B))
                - np.sum(P[i, j] * np.einsum(COEFF['cross_deriv_interior'], (i, j, m, n), u, B[:s] + (m,) + B[s+1:t] + (n,) + B[t+1:])
                         for (s, i), (t, j) in combinations(enumerate(B), 2))
                + u).reshape(-1)
    def b(k):
        m, n = 24, 25
        return (1.0 - 0.5 * np.sum(q[i] * np.einsum(COEFF['first_deriv_edge'], (i,), v[B[:s] + B[s+1:]][k], B[:s] + B[s+1:])
                                   - P[i, i] * np.einsum(COEFF['second_deriv_edge'], (i,), v[B[:s] + B[s+1:]][k], B[:s] + B[s+1:])
                                   for s, i in enumerate(B) if k != i)
                + np.sum(P[i, j] *
                         ((0.0 if k in (i, j) else np.einsum(COEFF['cross_deriv_corner'], (i, j), v[B[:s] + B[s+1:t] + B[t+1:]][k], B[:s] + B[s+1:t] + B[t+1:]))
                          + (0.0 if k == j else np.einsum(COEFF['cross_deriv_edge'], (i, j, m), v[B[:t] + B[t+1:]][k], B[:s] + (m,) + B[s+1:t] + B[t+1:]))
                          + (0.0 if k == i else np.einsum(COEFF['cross_deriv_edge'], (j, i, n), v[B[:s] + B[s+1:]][k], B[:s] + B[s+1:t] + (n,) + B[t+1:])))
                         for (s, i), (t, j) in combinations(enumerate(B, 2)))).reshape(-1)
    N = np.prod(grid_shape)
    A = LinearOperator((N, N), matvec=a)
    return {k : bicgstab(A, b(k)) for k in B}
