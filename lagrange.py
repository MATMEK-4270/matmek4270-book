import sympy as sp 

x = sp.Symbol('x')

def Lagrangebasis(xj, x=x):
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
    return ell
 
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