import numpy as np
import numpy.linalg as la

import collections

def normalize_weight(X):
    return np.resize(np.sqrt(np.sum(X ** 2, axis = 1)), (2, 2)).T

def oblimin_inner(X, gamma = 0):
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

def oblimin(X, gamma = 0, normalize = False, eps = 1e-5, max_iter = 1000):
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

    obli_mat, obli_sum, method_name = oblimin_inner(L, gamma)
    inter = -L.T.dot(obli_mat).dot(Tinv).T

    obli_mat2 = obli_mat.copy()
    obli_sum2 = obli_sum
    table = None
    row_ones = np.ones((1, inter.shape[0]))

    for i in range(max_iter + 1):
        inter2 = inter - Tmat.dot(np.diag(row_ones.dot(Tmat * inter)[0]))
        sqrt_inter = np.sqrt(np.sum(np.diag(inter2.T.dot(inter2))))

        record = np.array([i, obli_sum, np.log10(sqrt_inter), al])
        if table == None:
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
            obli_mat2, obli_sum2, _ = oblimin_inner(L, gamma)
            improvement = obli_sum - obli_sum2

            if improvement > .5 * sqrt_inter ** 2 * al:
                break

            al /= 2

        Tmat = Tmat2
        obli_sum = obli_sum2
        inter = -L.T.dot(obli_mat2).dot(Tmat2Inv).T

    convergence = sqrt_inter < eps

    if i == max_iter and not convergence:
        print('Convergence not obtained in Oblimin. {0} iterations used.'
              .format(max_iter))

    if normalize:
        L *= W

    Result = collections.namedtuple('Result', ['loadings', 'Phi', 'Th',
                                               'table', 'method', 'orthogonal',
                                               'convergence', 'Gq'])

    return Result(loadings = L, Phi = Tmat.T.dot(Tmat), Th = Tmat,
                  table = table, method = method_name, orthogonal = False,
                  convergence = convergence, Gq = obli_mat2)
