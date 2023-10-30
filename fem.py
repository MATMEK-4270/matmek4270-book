import numpy as np
import sympy as sp
from scipy.integrate import quad
from lagrange import Lagrangebasis, Lagrangefunction

x = sp.Symbol('x')

def map_true_domain(xj, e, d=1, x=x): # return x(X)
    xL, xR = get_element_boundaries(xj, e, d=d)
    hj = get_element_length(xj, e, d=d)
    return (xL+xR)/2+hj*x/2

def map_reference_domain(xj, e, d=1, x=x): # return X(x)
    xL, xR = get_element_boundaries(xj, e, d=d)
    hj = get_element_length(xj, e, d=d)
    return (2*x-(xL+xR))/hj

def map_u_true_domain(u, xj, e, d=1, x=x): # return u(x(X))
    return u.subs(x, map_true_domain(xj, e, d=d, x=x))

def get_element_boundaries(xj, e, d=1):
    return xj[d*e], xj[d*(e+1)]

def get_element_length(xj, e, d=1):
    xL, xR = get_element_boundaries(xj, e, d=d)
    return xR-xL

def local_to_global_map(e, r=None, d=1): # q(e, r)
    if r is None:
        return slice(d*e, d*e+d+1)
    return d*e+r

def assemble_b(u, xj, d=1):
    l = Lagrangebasis(np.linspace(-1, 1, d+1), sympy=False)
    N = len(xj)-1
    Ne = N//d
    b = np.zeros(N+1)
    for elem in range(Ne):
        hj = get_element_length(xj, elem, d=d)
        us = sp.lambdify(x, map_u_true_domain(u, xj, elem, d=d))
        integ = lambda xj, r: us(xj)*l[r](xj)
        for r in range(d+1):
            b[d*elem+r] += hj/2*quad(integ, -1, 1, args=(r,))[0]
    return b

def fe_evaluate(uh, p, xj, d=1):
    l = Lagrangebasis(np.linspace(-1, 1, d+1), sympy=False)
    elem = np.max([np.nonzero(p <= xj[::d])[0][0]-1,
                   np.nonzero(p < xj[::d])[0][0]-1]) # find element containing p
    Xx = map_reference_domain(xj, elem, d=d, x=p)
    return Lagrangefunction(uh[d*elem:d*(elem+1)+1], l)(Xx)

def fe_evaluate_v(uh, pv, xj, d=1):
    l = Lagrangebasis(np.linspace(-1, 1, d+1), sympy=False)
    # Find points on element boundaries
    elem = np.max(np.vstack((np.argmax((pv <= xj[::d, None]), axis=0)-1,
                             np.argmax((pv < xj[::d, None]), axis=0)-1)), axis=0) # find elements containing pv
    xL = xj[:-1:d] # All left element boundaries
    xR = xj[d::d]  # All right element boundaries
    xm = (xL+xR)/2 # middle of all elements
    hj = (xR-xL)   # length of all elements
    Xx = 2*(pv-xm[elem])/hj[elem] # map pv to reference space all elements
    dofs = np.array([uh[e*d+np.arange(d+1)] for e in elem], dtype=float)
    V = np.array([lr(Xx) for lr in l], dtype=float) # All basis functions evaluated for all points
    return np.sum(dofs * V.T, axis=1)