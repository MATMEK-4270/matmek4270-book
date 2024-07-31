import numpy as np
import sympy as sp
from scipy.interpolate import BarycentricInterpolator

x = sp.Symbol('x')

def Lagrangebasis(xj, x=x, sympy=True):
    """Construct Lagrange basis function for points in xj

    Parameters
    ----------
    xj : array
        Interpolation points
    x : Sympy Symbol

    Returns
    -------
    Lagrange basis functions
    """
    from sympy import Mul
    n = len(xj)
    ell = []
    numert = Mul(*[x - xj[i] for i in range(n)])

    for i in range(n):
        numer = numert/(x - xj[i])
        denom = Mul(*[(xj[i] - xj[j]) for j in range(n) if i != j])
        ell.append(numer/denom)
    if sympy:
        return ell
    return [np.polynomial.Polynomial(l.as_poly().all_coeffs()[::-1]) for l in ell]

def Lagrangefunction(u, basis):
    """Return Lagrange polynomial

    Parameters
    ----------
    u : array
        Mesh function values
    basis : tuple of Lagrange basis functions
        Output from Lagrangebasis
    """
    f = 0
    for j, uj in enumerate(u):
        f += basis[j]*uj
    return f

def Derivative(xj):
    N = len(xj)-1
    D = np.zeros((N+1, N+1))
    #for i in range(N+1):
    #    for j in range(N+1):
    #        if j != i:
    #            D[i, j] = w[j]/w[i]/(xj[i]-xj[j])
    #            D[i, i] -= D[i, j]
    # vectorized and faster version avoiding slow for loops
    w = BarycentricInterpolator(xj).wi
    W = w[None, :] / w[:, None]
    X = xj[:, None]-xj[None, :]
    np.fill_diagonal(X, 1)
    D[:] = W / X
    np.fill_diagonal(D, 0)
    np.fill_diagonal(D, -np.sum(D, axis=1))
    return D

def PolyDerivative(xj, m):
    N = len(xj)-1
    w = BarycentricInterpolator(xj).wi
    #D = Derivative(xj) # compute it directly below because W and X are needed
    D = np.zeros((N+1, N+1))
    W = w[None, :] / w[:, None]
    X = xj[:, None] - xj[None, :]
    np.fill_diagonal(X, 1)
    D[:] = W / X
    np.fill_diagonal(D, 0)
    np.fill_diagonal(D, -np.sum(D, axis=1))

    if m == 1:
        return D
    D2 = np.zeros_like(D)
    #for k in range(2, m+1):
    #    for i in range(N+1):
    #        for j in range(N+1):
    #            if j != i:
    #                D2[i, j] = k/(xj[i]-xj[j])*(w[j]/w[i]*D[i, i] - D[i, j])
    #                D2[i, i] -= D2[i, j]
    #    D[:] = D2
    # vectorized:
    for k in range(2, m+1):
        D2[:] = k / X * (W * D.diagonal()[:, None] - D)
        np.fill_diagonal(D2, 0)
        np.fill_diagonal(D2, -np.sum(D2, axis=1))
        D[:] = D2

    return D2
