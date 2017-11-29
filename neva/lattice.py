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
        banks_gbm = [BankGBM(bnksys.banksbyname[bank_i], sigma_asset=np.sqrt(covar_asset[bank_i, bank_i]))
                     for bank_i in bnksys.banksbyname]
        new_sys = cls._from_bankingsystem(banks_gbm, bnksys)
        # Copied from BankingSystemGBM. But I don't understand why this is not the usual instantiation... AT
        bidx = {bnk.name: idx for idx, bnk in new_sys.banksbyid.items()}
        new_sys.covar = {(bidx[bank_i], bidx[bank_j]) : S_ij for (bank_i, bank_j), S_ij in covar_asset.items()}
        return new_sys
