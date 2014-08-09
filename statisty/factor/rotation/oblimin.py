"""
Factor rotation: oblimin
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from future.builtins import range
from future import standard_library
standard_library.install_hooks()
import numpy as np
import numpy.linalg as la

from statisty.utils import cross_product

import collections

def normalize_weight(X):
    return np.resize(np.sqrt(np.sum(X ** 2, axis = 1)), (2, 2)).T

from functools import partial,  wraps

def oblique(func):
    @wraps(func)
    def inner(X, normalize = False, eps = 1e-5, max_iter = 1000,
              **kwargs):
        if kwargs:
            return oblique_rotation(X, normalize, eps, max_iter,
                                    partial(func, **kwargs))
            
        else:
            return oblique_rotation(X, normalize, eps, max_iter,
                                    func)
            
    return inner

def orthogonal(func):
    @wraps(func)
    def inner(X, normalize = False, eps = 1e-5, max_iter = 1000,
              **kwargs):
        if kwargs:
            return orthogonal_rotation(X, normalize, eps, max_iter,
                                       partial(func, **kwargs))
        else:
            return orthogonal_rotation(X, normalize, eps, max_iter,
                                       func)
            
    return inner

@oblique
def oblimin(X, gamma = 0, normalize = False, eps = 1e-5, max_iter = 1000):
    """
    Oblimin rotation. (Oblique)
    """
    method = {0: 'Oblimin Quartimin',
              .5: 'Oblimin Biquartimin',
              1: 'Oblimin Covarimin'}
    n_col = X.shape[1]
    L = (X ** 2).dot(np.eye(n_col) <= np.zeros(n_col))

    if gamma:
        n_row = X.shape[0]
        R = np.empty((n_row, n_row))
        R.fill(gamma / n_row)
        L = (np.eye(n_row) - R).dot(L)

    try:
        method_name = method[gamma]

    except:
        method_name = 'Oblimin g = ' + gamma

    return X * L, np.sum(X ** 2 * L) / 4, method_name

@oblique
def quartimin(X):
    n_col = X.shape[1]
    L = (X ** 2).dot(np.eye(n_col) <= np.zeros(n_col))

    return X * L, np.sum(X ** 2 * L) / 4, 'Quartimin'

def target(X, target):
    mat = 2 * (X - target)
    X[np.is_nan(X)] = 0

    return X, np.nansum((X - target) ** 2), 'Target rotation'

@orthogonal
def targetT(X, target):
    return target(X, target)

@oblique
def targetQ(X, target):
    return target(X, target)

def pst(X, W, target = None):
    B = W * target

    return 2 * (W * X - B), np.sum((W * X - B) ** 2), \
        'Partially specified target'

@orthogonal
def pstT(X, W, target = None):
    return pst(X, W, target)

@oblique
def pstQ(X, W, target = None):
    return pst(X, W, target)

@oblique
def oblimax(X):
    return -(4 * X ** 3 / np.sum(X ** 4) - 4 * X / np.sum(X ** 2)), \
        -np.log(np.sum(X ** 4)) - 2 * np.log(np.sum(X ** 2)), \
        'oblimax'

def entropy(X, normalize = False, eps = 1e-5, max_iter = 1000):
    return oblique_rotation(X, normalize, eps, max_iter, entropy_inner)

@oblique
def entropy(X):
    return -(X * np.log(X ** 2 + (X ** 2 == 0)) + X), \
        -np.sum(X ** 2 * np.log(X ** 2 + (X ** 2 == 0))) / 2, \
        'Minimum entropy'

@orthogonal
def quartimax(X):
    return -X ** 3, -np.sum(np.diag(cross_product(X ** 2, X ** 2))) / 4, \
        'Quartimax'

@orthogonal
def varimax(X):
    """Varimax rotation. (Orthogonal)"""
    QL = (X ** 2) - np.mean(X ** 2, axis = 0)
    return -X * QL, -np.sqrt(np.sum(np.diag(cross_product(QL, QL)))) ** 2 / 4, \
        'Varimax'

@oblique
def simplimax(X, k = -1):
    if (k == -1):
        k = np.prod(X.shape) - 1
    Imat = np.sign(X ** 2 <= np.sort(X ** 2, axis = None)[k])
    return 2 * Imat * X, np.sum(Imat * X ** 2), 'Simplimax'

def bentler(X):
    X2 = X ** 2
    M = cross_product(X2, X2)
    D = np.diag(np.diag(M))

    return -L * (L2.dot(la.inv(M) - la.inv(D))), \
        -(np.log(la.det(M)) - np.log(la.det(D))) / 4, \
        'Bentler\'s criterion'

@oblique
def bentlerQ(X):
    return bentler(X)

@orthogonal
def bentlerT(X):
    return bentler(X)

@orthogonal
def tandemI(X):
    XX = X.dot(X)
    XX2 = XX ** 2
    Gq1 = 4 * X * (XX2.dot(X ** 2))
    Gp2 = 4 * (XX * (X ** 2).dot((X ** 2).T)).dot(X)
    Gq = -Gq1 - Gq2

    return Gq, -np.sum(np.diag(cross_product(X ** 2, XX2.dot(X ** 2)))), \
        'Tandem I'

@orthogonal
def tandemII(X):
    XX = X.dot(X.T)
    XX2 = XX ** 2
    f = np.sum(np.diag(cross_product(X ** 2, (1 - XX2).dot(X ** 2))))
    Gq1 = 4 * X * ((1 - XX2).dot(X ** 2))
    Gq2 = 4 * (XX * (X ** 2).dot((X ** 2).T)).dot(X)
    Gq = Gq1 - Gq2

    return Gq, f, 'Tandem II'

def geomin(X, delta = .01):
    row, col = X.shape
    X2 = X ** 2 + delta
    pro = np.exp(np.sum(np.log(X2), axis = 1) / col)

    return (2 / col) * (X / X2) * np.tile(pro, (col, 1)).T, \
        np.sum(pro), 'Geomin'

@orthogonal
def geominT(X, delta = .01):
    return geomin(X, delta)

@oblique
def geominQ(X, delta = .01):
    return geomin(X, delta)

def cf(X, kappa = 0):
    row, col = X.shape
    N = np.ones((col, col)) - np.eye(col)
    M = np.ones((row, row)) - np.eye(row)
    X2 = X ** 2
    f1 = (1 - kappa) * np.sum(np.diag(cross_product(X2, X2.dot(N))) / 4)
    f2 = kappa * np.sum(np.diag(cross_product(X2, M.dot(X2))) / 4)

    return (1 - kappa) * X * (X2.dot(N)) + kappa * X * (M.dot(X2)), \
        f1 + f2, 'Crawford-Ferguson kappa = ' + kappa

@orthogonal
def cfT(X, kappa = 0):
    cf(X, kappa)

@oblique
def cfQ(X, kappa = 0):
    cf(X, kappa)

def infomax(X):
    row, col = X.shape
    S = X ** 2
    s = np.sum(S)
    s1 = np.sum(S, axis = 1)
    s2 = np.sum(S, axis = 0)
    E = S / s
    e1 = s1 / s
    e2 = s2 / s
    Q0 = np.sum(-E * np.log(E))
    Q1 = np.sum(-e1 * np.log(e1))
    Q2 = np.sum(-e2 * np.log(e2))
    f = np.log(col) + Q0 - Q1 - Q2
    H = -(np.log(E) + 1)
    alpha = np.sum(S * H) / s ** 2
    G0 = H / s - alpha * np.ones((row, col))
    h1 = -(np.log(e1) + 1)
    alpha1 = s1.dot(h1) / s ** 2
    G1 = np.tile(h1, (col, 1)).T / s - alpha1.flatten() * np.ones((row, col))
    h2 = -(np.log(e2) + 1)
    alpha2 = h2.dot(s2) / s ** 2
    G2 = np.tile(h2, row).reshape((row, col)) / s - alpha2.flatten() \
         * np.ones((row, col))
    Gq = 2 * X * (G0 - G1 - G2)

    return Gq, f, 'Infomax'

@orthogonal
def mccammon(X):
    row, col = X.shape
    S = X ** 2
    M = np.ones((row, row))
    s2 = np.sum(S, axis = 0)
    P = S / np.tile(s2, row).reshape((row, col))
    Q1 = -np.sum(P * np.log(P))
    H = -(np.log(P) + 1)
    R = M.dot(S)
    G1 = H / R - M.dot(S * H / R ** 2)
    s = np.sum(S)
    p2 = s2 / s
    Q2 = -np.sum(p2 * np.log(p2))
    h = -(np.log(p2) + 1)
    alpha = h.dot(p2)
    G2 = np.ones(row).dot(h.T)/s - alpha.flatten() * np.ones((row, col))
    Gq = 2 * X * (G1 / Q1 - G2 / Q2)
    Q = np.log(Q1) - np.log(Q2)

    return Gq, Q, 'McCammon entropy'

def bifactor(X):
    def D(X):
        X2 = X ** 2
        X2N = X2.dot(np.ones(X.shape[1]) - np.eye(X.shape[1]))
        return 4 * X * X2N, np.sum(X2 * X2N)

    G, f = D(X[:, 1:])
    G = np.hstack((np.array(G[:, 0:1]), G))
    G[:, 0] = 0

    return G, f, 'bifactor'

@orthogonal
def bifactorT(X):
    return bifactor(X)

@oblique
def bifactorQ(X):
    return bifactor(X)

@orthogonal
def infomaxT(X):
    return infomax(X)

@oblique
def infomaxQ(X):
    return infomax(X)
    
def oblique_rotation(X, normalize = False, eps = 1e-5,
                     max_iter = 1000, method = None):
    """
	Parameters
    ----------
    X : 
        

    gamma : (default 0)
        

    normalize : (default False)
        

    eps : (default 1e-5)
        

    max_iter : (default 1000)
        
    """

    if normalize:
        W = normalize_weight(X)
        X /= W

    al = 1
    Tmat = np.eye(X.shape[1])
    Tinv = la.inv(Tmat)
    L = X.dot(Tinv.T)

    obli_mat, obli_sum, method_name = method(L)
    inter = -L.T.dot(obli_mat).dot(Tinv).T

    obli_mat2 = obli_mat.copy()
    obli_sum2 = obli_sum
    table = None
    row_ones = np.ones((1, inter.shape[0]))

    for i in range(max_iter + 1):
        inter2 = inter - Tmat.dot(np.diag(row_ones.dot(Tmat * inter)[0]))
        sqrt_inter = np.sqrt(np.sum(np.diag(inter2.T.dot(inter2))))

        record = np.array([i, obli_sum, np.log10(sqrt_inter), al])
        if table is None:
            table = record

        else:
            table = np.vstack([table, record])

        if sqrt_inter < eps:
            break

        al *= 2

        for j in range(11):
            A = Tmat - al * inter2
            V = 1 / np.sqrt(row_ones.dot(A ** 2))
            Tmat2 = A.dot(np.diag(V[0]))
            Tmat2Inv = la.inv(Tmat2)
            L = X.dot(Tmat2Inv.T)
            obli_mat2, obli_sum2, _ = method(L)
            improvement = obli_sum - obli_sum2

            if improvement > .5 * sqrt_inter ** 2 * al:
                break

            al /= 2

        Tmat = Tmat2
        obli_sum = obli_sum2
        inter = -L.T.dot(obli_mat2).dot(Tmat2Inv).T

    convergence = sqrt_inter < eps

    if i == max_iter and not convergence:
        print('Convergence not obtained in {}. {} iterations used.'
              .format(method_name, max_iter))

    if normalize:
        L *= W

    Result = collections.namedtuple('Result', ['loadings', 'Phi', 'Th',
                                               'table', 'method', 'orthogonal',
                                               'convergence', 'Gq'])

    return Result(loadings = L, Phi = Tmat.T.dot(Tmat), Th = Tmat,
                  table = table, method = method_name, orthogonal = False,
                  convergence = convergence, Gq = obli_mat2)

def orthogonal_rotation(X, normalize = False, eps = 1e-5,
                     max_iter = 1000, method = None):
    """
	Parameters
    ----------
    X : 
        

    gamma : (default 0)
        

    normalize : (default False)
        

    eps : (default 1e-5)
        

    max_iter : (default 1000)
        
    """

    if normalize:
        W = normalize_weight(X)
        X /= W

    al = 1
    Tmat = np.eye(X.shape[1])
    # diff
    L = X.dot(Tmat)
    # end

    obli_mat, obli_sum, method_name = method(L)
    # diff
    inter = cross_product(X, obli_mat)
    # end
    
    obli_mat2 = obli_mat.copy()
    obli_sum2 = obli_sum
    table = None
    row_ones = np.ones((1, inter.shape[0]))

    for i in range(max_iter + 1):
        # diff
        M = cross_product(Tmat, inter)
        inter2 = inter - Tmat.dot((M + M.T) / 2)
        sqrt_inter = np.sqrt(np.sum(np.diag(inter2.T.dot(inter2))))
        # end

        record = np.array([i, obli_sum, np.log10(sqrt_inter), al])
        if table is None:
            table = record

        else:
            table = np.vstack([table, record])

        if sqrt_inter < eps:
            break

        al *= 2

        for j in range(11):
            A = Tmat - al * inter2
            # diff
            U, D, Vt = la.svd(A)
            Tmat2 = U.dot(Vt)
            L = X.dot(Tmat2)
            # end
            
            obli_mat2, obli_sum2, _ = method(L)

            # diff
            if obli_sum2 < (obli_sum - .5 * sqrt_inter ** 2 * al):
                break
            # end

            al /= 2

        Tmat = Tmat2
        obli_sum = obli_sum2
        # diff
        inter = cross_product(X, obli_mat2)
        # end

    convergence = sqrt_inter < eps

    if i == max_iter and not convergence:
        print('Convergence not obtained in {}. {} iterations used.'
              .format(method_name, max_iter))

    if normalize:
        L *= W

    Result = collections.namedtuple('Result', ['loadings', 'Phi', 'Th',
                                               'table', 'method', 'orthogonal',
                                               'convergence', 'Gq'])

    return Result(loadings = L, Phi = None, Th = Tmat,
                  table = table, method = method_name, orthogonal = True,
                  convergence = convergence, Gq = obli_mat2)
