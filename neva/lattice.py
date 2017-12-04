import json
from itertools import repeat, permutations, combinations, combinations_with_replacement
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab
from .gbm import BankGBM, BankingSystemGBM

def load_coefficients():
    """Loads up the coefficients used for the pseudospectral method."""
    with open('neva/pseudospectral.json') as f:
        coeff = json.load(f)
    return {k: np.array(v) for k, v in coeff}

GRID_LEN = 8
COEFF = load_coefficients()

def differentiate_boundary(B, v):
    """Calculates the first and second derivatives of v with respect to x.

    B is for bank labels, a tuple of integers between 0 and 23 sorted in ascending order.
    v is for valuation, defined on a grid with dimension matching the length of B.
    Find some notes on the use of einsum at https://github.com/agjftucker/neva/wiki/Pseudospectral-Solution#boundary-derivatives.
    """
    m, n = 24, 25
    Dx_v = {}
    for i in B:
        for s, j in enumerate(B):
            Dx_v[i, j] = ((0. if i == j else np.einsum(COEFF['first_deriv_edge'], (j,), v[B[:s] + B[s+1:]][i], B[:s] + B[s+1:]))
                          + np.einsum(COEFF['first_deriv_interior'], (j, m), v[B][i], B[:s] + (m,) + B[s+1:]))
    Dxx_v = {}
    for i in B:
        for s, j in enumerate(B):
            Dxx_v[i, j, j] = ((0. if i == j else np.einsum(COEFF['second_deriv_edge'], (j,), v[B[:s] + B[s+1:]][i], B[:s] + B[s+1:]))
                              + np.einsum(COEFF['second_deriv_interior'], (j, m), v[B][i], B[:s] + (m,) + B[s+1:]))
        for (s, j), (t, k) in combinations(enumerate(B), 2):
            Dxx_v[i, j, k] = ((0. if i in (j, k) else np.einsum(COEFF['cross_deriv_corner'], (j, k), v[B[:s] + B[s+1:t] + B[t+1:]][i], B[:s] + B[s+1:t] + B[t+1:]))
                              + (0. if i == k else np.einsum(COEFF['cross_deriv_edge'], (j, k, m), v[B[:t] + B[t+1:]][i], B[:s] + (m,) + B[s+1:t] + B[t+1:]))
                              + (0. if i == j else np.einsum(COEFF['cross_deriv_edge'], (k, j, n), v[B[:s] + B[s+1:]][i], B[:s] + B[s+1:t] + (n,) + B[t+1:]))
                              + np.einsum(COEFF['cross_deriv_interior'], (j, k, m, n), v[B][i], B[:s] + (m,) + B[s+1:t] + (n,) + B[t+1:]))
    return Dx_v, Dxx_v

def solve_pde(B, v, boundary_derivs, S, z, L):
    """Solves the set of n PDEs associated with a set of n live banks.

    B is for bank labels, a tuple of integers between 0 and 23 sorted in ascending order.
    v is the dictionary of solutions to the PDEs associated with smaller subsets.
    boundary_derivs contains the first and second derivatives of those solutions.
    S is the covariance matrix for external assets. It need only be populated for i ≤ j.
    z[i] is the total liability of bank i.
    L[i, j] is the debt owed by i to j.
    """
    n_banks = len(B)
    axis = {k: t for t, k in enumerate(B)}
    grid_shape = tuple(repeat(GRID_LEN, n_banks))
    Dx_y = np.empty((n_banks, n_banks) + grid_shape)
    Dxx_y = {}
    for t, k in enumerate(B):
        not_k = B[:t] + B[t+1:]
        Dx_v, Dxx_v = boundary_derivs[not_k]
        Dx_y[t, t] = 2. * np.sqrt(S[k, k])   # Needs work!
        fail_k = z[k] - np.sum(v[not_k][j] * L[j, k] for j in not_k)
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
    P = {(i, j): (np.sum(S[k, k] * Dy_x(i, k) * Dy_x(j, k) for k in B)
                  + np.sum(S[k, l] * (Dy_x(i, k) * Dy_x(j, l) + Dy_x(j, k) * Dy_x(i, l)) for k, l in combinations(B, 2)))
         for i, j in combinations_with_replacement(B, 2)}
    q = {i: (np.sum(S[k, k] * (Dy_x(i, k) - Dyy_x(i, k, k)) for k in B)
             - 2. * np.sum(S[k, l] * Dyy_x(i, k, l) for k, l in combinations(B, 2))) for i in B}
    N = np.prod(grid_shape)
    def a(u):
        m, n = 24, 25
        u = u.reshape(grid_shape)
        Au = (u + 0.5 * np.sum(q[i] * np.einsum(COEFF['first_deriv_interior'], (i, m), u, B[:s] + (m,) + B[s+1:])
                               - P[i, i] * np.einsum(COEFF['second_deriv_interior'], (i, m), u, B[:s] + (m,) + B[s+1:]) for s, i in enumerate(B))
              - np.sum(P[i, j] * np.einsum(COEFF['cross_deriv_interior'], (i, j, m, n), u, B[:s] + (m,) + B[s+1:t] + (n,) + B[t+1:])
                       for (s, i), (t, j) in combinations(enumerate(B), 2)))
        return Au.reshape(N)
    A = LinearOperator((N, N), matvec=a)
    def b(k):
        m, n = 24, 25
        bk = (np.ones(grid_shape) - 0.5 * np.sum(q[i] * np.einsum(COEFF['first_deriv_edge'], (i,), v[B[:s] + B[s+1:]][k], B[:s] + B[s+1:])
                                                 - P[i, i] * np.einsum(COEFF['second_deriv_edge'], (i,), v[B[:s] + B[s+1:]][k], B[:s] + B[s+1:])
                                                 for s, i in enumerate(B) if k != i)
              + np.sum(P[i, j] *
                       ((0. if k in (i, j) else np.einsum(COEFF['cross_deriv_corner'], (i, j), v[B[:s] + B[s+1:t] + B[t+1:]][k], B[:s] + B[s+1:t] + B[t+1:]))
                        + (0. if k == j else np.einsum(COEFF['cross_deriv_edge'], (i, j, m), v[B[:t] + B[t+1:]][k], B[:s] + (m,) + B[s+1:t] + B[t+1:]))
                        + (0. if k == i else np.einsum(COEFF['cross_deriv_edge'], (j, i, n), v[B[:s] + B[s+1:]][k], B[:s] + B[s+1:t] + (n,) + B[t+1:])))
                       for (s, i), (t, j) in combinations(enumerate(B), 2)))
        return bk.reshape(N)
    vB = {}
    for k in B:
        vk, err = bicgstab(A, b(k))
        print("B = " + str(B) + ", k = " + str(k) + ", grid_shape = " + str(grid_shape) + ", bicgstab return code = " + str(err))
        vB[k] = vk.reshape(grid_shape)
    return vB
    # return {k: (bicgstab(A, b(k))[0]).reshape(grid_shape) for k in B}

class BankingSystemCorrelatedGBM(BankingSystemGBM):
    """Banking system whose banks have external assets following a correlated
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

    def fixedpoint_equity(self):
        """Computes debt valuations (from which follow equity) using the lattice model.

        The calculation is made based on the lattice formed by the power set of banks.
        I shall explain this on the wiki.
        """
        banks = range(self.nbanks)
        ibasset_matrix = self.get_ibasset_matrix()
        sol = {(): {}}
        for m in range(self.nbanks):
            boundary_derivs = {subs: differentiate_boundary(subs, sol) for subs in combinations(banks, m)}
            for live_banks in combinations(banks, m + 1):
                sol[live_banks] = solve_pde(live_banks, sol, boundary_derivs,
                                            S={(i, j): self.covar[i, j] for i, j in combinations_with_replacement(live_banks, 2)},
                                            z={i: self[i].extliab + self[i].ibliabtot for i in live_banks},
                                            L={(i, j): ibasset_matrix[j][i] for i, j in permutations(live_banks, 2)})
        integrated_sol = {j: np.empty(np.repeat(1 + GRID_LEN, self.nbanks)) for j in banks}
        for live_banks, v in sol.items():
            face = tuple(slice(1, None) if i in live_banks else 0 for i in banks)
            for j in banks:
                integrated_sol[j][face] = v.get(j, 0.)   # if j is not in live_banks then its debt is worthless
        self.sol = integrated_sol
        return
