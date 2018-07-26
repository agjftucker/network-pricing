"""Define BankingSystemCorrelatedGBM exposing methods with_covar_asset() and fixedpoint_equity()."""

# pylint: disable=invalid-name

import itertools as it
from itertools import chain, combinations, combinations_with_replacement, permutations
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab
from .gbm import BankGBM, BankingSystemGBM

def weights(z, x, m):
    """Compute the differentiation matrices for pseudospectral collocation.

    Input are z, the location where the matrices are to be evaluated, and x[0..n], the set of n+1
    grid points. On output, c[0..n, 0..m] contains the weights at grid locations x[0..n] for
    derivatives of order 0..m. The element c[j, k] contains the weight to be applied to the function
    value at x[j] when the kth derivative is approximated by the set of n+1 collocation points x.
    Note that the elements of the zeroth derivative matrix are returned in c[0..n, 0]. These are
    just the values of the cardinal functions, i.e. the weights for interpolation."""
    # Press, Teukolsky, Vetterling and Flannery, Numerical Recipes Third Edition (2007), page 1092.
    n = len(x) - 1
    c1 = 1.
    c4 = x[0] - z
    c = np.zeros((n + 1, m + 1), dtype=float)
    c[0, 0] = 1.
    for i in range(n + 1):
        mn = min(i, m)
        c2 = 1.
        c5 = c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 = c2 * c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c

def calc_coefficients(n):
    """Compute first- and second-order derivatives with Chebyshev-Gauss-Radau quadrature."""
    # Shen, Tang and Wang, Spectral Methods: Algorithms, Analysis and Application (2011), page 108.
    x = -np.cos(2. * np.pi * np.arange(n + 1) / (2 * n + 1))
    w = [weights(xi, x, 2) for xi in x]
    d = np.array([c[:, 1] for c in w])
    dd = np.array([c[:, 2] for c in w])
    return x, d, dd

def differentiate_boundary(B, v, d, dd):
    """Calculate the first and second derivatives of v with respect to x.

    Arguments: B the set of indices of surviving banks, a tuple of integers between 0 and 23 sorted
    in ascending order; v a dictionary of valuation fns for subsets of surviving banks, each
    specified on a grid with dimension matching the set's size; d the differentiation matrix
    returned from calc_coefficients(n).  Find some notes on the use of einsum at
    https://github.com/agjftucker/neva/wiki/Pseudospectral-Solution#boundary-derivatives.
    """
    m, n = 24, 25
    Dx_v = {}
    for i in B:
        for s, j in enumerate(B):
            Dx_v[i, j] = ((0. if i == j else
                           np.einsum(d[1:, 0], (j,), v[B[:s] + B[s+1:]][i], B[:s] + B[s+1:]))
                          + np.einsum(d[1:, 1:], (j, m), v[B][i], B[:s] + (m,) + B[s+1:]))
    Dxx_v = {}
    for i in B:
        for s, j in enumerate(B):
            Dxx_v[i, j, j] = ((0. if i == j else
                               np.einsum(dd[1:, 0], (j,), v[B[:s] + B[s+1:]][i], B[:s] + B[s+1:]))
                              + np.einsum(dd[1:, 1:], (j, m), v[B][i], B[:s] + (m,) + B[s+1:]))
        for (s, j), (t, k) in combinations(enumerate(B), 2):
            Dxx_v[i, j, k] = ((0. if i in (j, k) else
                               np.einsum(d[1:, 0], (j,),
                                         np.einsum(d[1:, 0], (k,), v[B[:s] + B[s+1:t] + B[t+1:]][i], B[:s] + B[s+1:t] + B[t+1:]),
                                         B[:s] + B[s+1:]))
                              + (0. if i == k else
                                 np.einsum(d[1:, 0], (k,),
                                           np.einsum(d[1:, 1:], (j, m), v[B[:t] + B[t+1:]][i], B[:s] + (m,) + B[s+1:t] + B[t+1:]),
                                           B[:t] + B[t+1:]))
                              + (0. if i == j else
                                 np.einsum(d[1:, 0], (j,),
                                           np.einsum(d[1:, 1:], (k, n), v[B[:s] + B[s+1:]][i], B[:s] + B[s+1:t] + (n,) + B[t+1:]),
                                           B[:s] + B[s+1:]))
                              + np.einsum(d[1:, 1:], (j, m),
                                          np.einsum(d[1:, 1:], (k, n), v[B][i], B[:t] + (n,) + B[t+1:]),
                                          B[:s] + (m,) + B[s+1:]))
    return Dx_v, Dxx_v

def solve_eqn(B, grid_shape, d, dd, P, q, v, i):
    """Construct and solve the linear system of equations corresponding to a single PDE."""
    N = np.prod(grid_shape)
    m, n = 24, 25
    def a(u):
        u = u.reshape(grid_shape)
        Au = (u + 0.5 * np.sum(q[j] * np.einsum(d[1:, 1:], (j, m), u, B[:s] + (m,) + B[s+1:])
                               - P[j, j] * np.einsum(dd[1:, 1:], (j, m), u, B[:s] + (m,) + B[s+1:])
                               for s, j in enumerate(B))
              - np.sum(P[j, k] * np.einsum(d[1:, 1:], (j, m),
                                           np.einsum(d[1:, 1:], (k, n), u, B[:t] + (n,) + B[t+1:]),
                                           B[:s] + (m,) + B[s+1:])
                       for (s, j), (t, k) in combinations(enumerate(B), 2)))
        return Au.reshape(N)
    A = LinearOperator((N, N), matvec=a)
    b = (np.ones(grid_shape, dtype=float)
         - 0.5 * np.sum(q[j] * np.einsum(d[1:, 0], (j,), v[B[:s] + B[s+1:]], B[:s] + B[s+1:])
                        - P[j, j] * np.einsum(dd[1:, 0], (j,), v[B[:s] + B[s+1:]], B[:s] + B[s+1:])
                        for s, j in enumerate(B) if i != j)
         + np.sum(P[j, k] * ((0. if i in (j, k) else
                              np.einsum(d[1:, 0], (j,),
                                        np.einsum(d[1:, 0], (k,), v[B[:s] + B[s+1:t] + B[t+1:]], B[:s] + B[s+1:t] + B[t+1:]),
                                        B[:s] + B[s+1:]))
                             + (0. if i == j else
                                np.einsum(d[1:, 0], (j,),
                                          np.einsum(d[1:, 1:], (k, n), v[B[:s] + B[s+1:]], B[:s] + B[s+1:t] + (n,) + B[t+1:]),
                                          B[:s] + B[s+1:]))
                             + (0. if i == k else
                                np.einsum(d[1:, 0], (k,),
                                          np.einsum(d[1:, 1:], (j, m), v[B[:t] + B[t+1:]], B[:s] + (m,) + B[s+1:t] + B[t+1:]),
                                          B[:t] + B[t+1:])))
                  for (s, j), (t, k) in combinations(enumerate(B), 2)))
    b = b.reshape(N)
    u, err = bicgstab(A, b)
    assert err == 0
    return u.reshape(grid_shape)

def solve_pdes(pool, B, eta, v, boundary_derivs, coeff, S, z, L):
    """Solve the set of n PDEs associated with a set of n live banks.

    Arguments: B the set of indices of surviving banks, a tuple of integers between 0 and 23 sorted
    in ascending order; v the dictionary of solutions to PDEs associated with smaller subsets;
    boundary_derivs the first and second derivatives of those solutions; S the covariance matrix for
    external assets (entries required only for i ≤ j); z[i] the total liability of bank i; L[i, j]
    the debt owed by i to j.
    """
    x, d, dd = coeff
    n_banks = len(B)
    axis = {k: t for t, k in enumerate(B)}
    grid_shape = tuple(it.repeat(len(x) - 1, n_banks))
    Dx_G = np.empty((n_banks, n_banks) + grid_shape)
    Dxx_G = {}
    # Notes on this at https://github.com/agjftucker/neva/blob/master/algebraic-mapping.ipynb.
    for s, j in enumerate(B):
        not_j = B[:s] + B[s+1:]
        fail_j = z[j] - np.sum(v[not_j][i] * L[i, j] for i in not_j)
        Dx_v, Dxx_v = boundary_derivs[not_j]
        pinch = grid_shape[:s] + (1,) + grid_shape[s+1:]
        bulge = (grid_shape[s],) + tuple(it.repeat(1, n_banks - 1 - s))
        Dx_G[s, s] = (2. * eta[j] / (1. - x[1:]) ** 2).reshape(bulge)
        for m in not_j:
            Dx_G[s, axis[m]] = (-np.sum(Dx_v[i, m] * L[i, j] for i in not_j) / fail_j).reshape(pinch)
        Dxx_G[j, j, j] = (4. * eta[j] / (1. - x[1:]) ** 3).reshape(bulge)
        for m, n in combinations_with_replacement(not_j, 2):
            Dxx_G[j, m, n] = (-np.sum(Dxx_v[i, m, n] * L[i, j] for i in not_j) / fail_j
                              - (np.sum(Dx_v[i, m] * L[i, j] for i in not_j)
                                 * np.sum(Dx_v[i, n] * L[i, j] for i in not_j)) / fail_j ** 2).reshape(pinch)
    Dx_G_shift = np.moveaxis(Dx_G, (0, 1), (-2, -1))
    # I use NumPy's linalg rather than SciPy's because it supports broadcasting over Dx_G_shift. But
    # NumPy seems to contain no equivalent to cho_solve, so I just dispose of the factorisation and
    # call inv.
    try:
        np.linalg.cholesky(Dx_G_shift + Dx_G_shift.swapaxes(-2, -1))
    except np.linalg.LinAlgError:
        print("Dx_G possibly fails to be positive definite.")
        raise
    inv_Dx_G = np.moveaxis(np.linalg.inv(Dx_G_shift), (-2, -1), (0, 1))
    def Dy_x(i, k):
        return inv_Dx_G[axis[i], axis[k]]
    def Dyy_x(i, k, l):
        return -np.sum(
            inv_Dx_G[axis[i], s] * (np.sum(Dy_x(m, k) * Dy_x(m, l) * Dxx_G[j, m, m] for m in B)
                                    + np.sum((Dy_x(m, k) * Dy_x(n, l)
                                              + Dy_x(n, k) * Dy_x(m, l)) * Dxx_G[j, m, n]
                                             for m, n in combinations(B[:s] + B[s+1:], 2)))
            for s, j in enumerate(B))
    P = {(i, j): (np.sum(S[k, k] * Dy_x(i, k) * Dy_x(j, k) for k in B)
                  + np.sum(S[k, l] * (Dy_x(i, k) * Dy_x(j, l) + Dy_x(j, k) * Dy_x(i, l))
                           for k, l in combinations(B, 2)))
         for i, j in combinations_with_replacement(B, 2)}
    q = {i: (np.sum(S[k, k] * (Dy_x(i, k) - Dyy_x(i, k, k)) for k in B)
             - 2. * np.sum(S[k, l] * Dyy_x(i, k, l) for k, l in combinations(B, 2))) for i in B}
    boundary_set = {i: {} for i in B}
    for b in chain.from_iterable(combinations(B, m) for m in range(max(n_banks - 2, 0), n_banks)):
        for i in b:
            boundary_set[i][b] = v[b][i]
    return [(i, pool.submit(solve_eqn, B, grid_shape, d, dd, P, q, boundary_set[i], i)) for i in B]

def interpolate(h, u):
    N = tuple(range(len(u.shape)))
    for i in N:
        u = np.einsum(h[i], (i,), u, N[i:])
    return u.tolist()

def valuation(grid_points, eta, sol, z, L, a):
    """Find the debt valuation by solving for x given asset prices and then interpolating."""
    B = sorted(eta.keys())
    axis = {j: s for s, j in enumerate(B)}
    eta = np.array([eta[j] for j in B])
    y = np.log([a[j] for j in B])
    x = np.zeros(len(B))
    faces = {j: {i: sol[i][tuple(it.repeat(slice(None), s)) + (0,)] for i in B[:s] + B[s+1:]}
             for s, j in enumerate(B)}
    for _ in range(8):
        w = [weights(xi, grid_points, 1) for xi in x]
        h = {k: c[:, 0] for k, c in zip(B, w)}
        hp = {k: c[:, 1] for k, c in zip(B, w)}
        # Again refer to https://github.com/agjftucker/neva/blob/master/algebraic-mapping.ipynb.
        G = (1. + x) * eta / (1. - x) - y
        Dx_G = np.diag(2. * eta / (1. - x) ** 2)
        for s, j in enumerate(B):
            not_j = B[:s] + B[s+1:]
            face_not_j = faces[j]
            v_not_j = {i: interpolate([h[k] for k in not_j], face_not_j[i]) for i in not_j}
            fail_j = z[j] - sum(v_not_j[i] * L[i, j] for i in not_j)
            G[s] += np.log(fail_j)
            Dx_v_not_j = {(i, m): interpolate([hp[k] if k == m else h[k] for k in not_j],
                                              face_not_j[i])
                          for i, m in it.product(not_j, repeat=2)}
            for m in not_j:
                Dx_G[s, axis[m]] = -sum(Dx_v_not_j[i, m] * L[i, j] for i in not_j) / fail_j
        x -= np.linalg.solve(Dx_G, G)
    print("x = " + str({j: x[s] for s, j in enumerate(B)}))
    w = [weights(xi, grid_points, 0) for xi in x]
    h = [c[:, 0] for c in w]
    return {j: interpolate(h, sol[j]) for j in B}

class BankingSystemCorrelatedGBM(BankingSystemGBM):
    """Banking system whose banks' external assets follow a correlated Geometric Brownian Motion."""

    def solve_lattice(self, covar, grid_len, eta_mul):
        """Perform the valuation of banks' debt v using the lattice model described in the paper."""
        x, d, dd = calc_coefficients(grid_len)
        modelled_banks = sorted(i for i, b in self.banksbyid.items() if b.ibliabtot > 0.)
        S = {(i, j): covar[self[i].name, self[j].name]
             for i, j in combinations_with_replacement(modelled_banks, 2)}
        eta = {j: eta_mul * np.sqrt(S[j, j]) for j in modelled_banks}
        ibasset_matrix = self.get_ibasset_matrix()
        sol = {(): {}}
        with ProcessPoolExecutor() as pool:
            for m in range(len(modelled_banks)):
                boundary_derivs = {B: differentiate_boundary(B, sol, d, dd)
                                   for B in combinations(modelled_banks, m)}
                fut = [(B, solve_pdes(pool, B, eta, sol, boundary_derivs, (x, d, dd), S,
                                      z={j: self[j].extliab + self[j].ibliabtot for j in B},
                                      L={(i, j): ibasset_matrix[j][i]
                                         for i, j in permutations(B, 2)}))
                       for B in combinations(modelled_banks, m + 1)]
                sol.update((B, {i: f.result() for i, f in fsol}) for B, fsol in fut)
        cube = {j: np.empty(np.repeat(grid_len + 1, len(modelled_banks))) for j in modelled_banks}
        for live_banks, v in sol.items():
            face = tuple(slice(1, None) if i in live_banks else 0 for i in modelled_banks)
            for j in modelled_banks:
                cube[j][face] = v.get(j, 0.)
                # if j is not in live_banks then its debt is worthless.
        self.grid_points = x
        self.eta = eta
        self.sol = cube

    @classmethod
    def with_covar_asset(cls, bnksys, covar_asset, grid_len=8, eta_mul=1.):
        """Create an instance and compute its valuation fn using the supplied covariance matrix.

        Parameters:
            bnksys (`BankingSystem`): banking system object
            covar_asset (dictionary of float): covariance matrix of external assets
        """
        banks_gbm = [BankGBM(bnk, sigma_asset=np.sqrt(covar_asset[bnk.name, bnk.name]))
                     for bnk in bnksys]
        new_sys = cls._from_bankingsystem(banks_gbm, bnksys)
        new_sys.solve_lattice(covar_asset, grid_len, eta_mul)
        return new_sys

    def fixedpoint_equity(self):
        """Apply the valuation fn at the point corresponding to this instance's current state."""
        all_banks = self.banksbyid.keys()
        z = {j: self[j].extliab + self[j].ibliabtot for j in all_banks}
        ibasset_matrix = self.get_ibasset_matrix()
        L = {(i, j): ibasset_matrix[j][i] for i, j in permutations(all_banks, 2)}
        a = {j: self[j].extasset for j in all_banks}
        v = valuation(self.grid_points, self.eta, self.sol, z, L, a)
        for j in all_banks:
            self[j].equity = a[j] - z[j] + sum(v[i] * L[i, j] for i in v.keys() if i != j)
