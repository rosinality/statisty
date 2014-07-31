import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.optimize

def pairwise_count(X):
    return (~np.isnan(X).astype(int)).T.dot(~np.isnan(X).astype(int))

def harmonic_mean(X, strip_nan = True):
    if len(X.shape) == 1:
        if strip_nan:
            return 1 / sp.stats.nanmean(1 / X)
            
        else:
            return 1 / np.mean(1 / X)
    else:
        if strip_nan:
            return 1 / np.apply_along_axis(sp.stats.nanmean, 0, 1 / X)
            
        else:
            return 1 / np.apply_along_axis(np.mean, 0, 1 / X)

def eigen_sorted(X):
    eig_val, eig_vec = la.eig(X)
    index = eig_val.argsort()[::-1]
    
    return eig_val[index], eig_vec[:, index]

def eigenh_sorted(X):
    eig_val, eig_vec = la.eigh(X)
    index = eig_val.argsort()[::-1]
    
    return eig_val[index], eig_vec[:, index]

def cov_to_cor(X):
    r, c = X.shape
    sd = 1 / np.sqrt(np.diag(X))
    X *= sd.reshape(r, 1)
    X *= sd.reshape(1, c)
    
    return X
        
def smooth_corrcoef(X):
    eig_val, eig_vec = eigen_sorted(X)
    eps = np.finfo(np.float64).eps
    
    if np.min(eig_val) < eps:
        eig_val[eig_val < eps] = 100 * eps
        var = X.shape[0]
        total = np.sum(eig_val)
        eig_val = eig_val * var / total
        X = eig_vec.dot(np.diag(eig_val)).dot(eig_vec.T)
        X = cov_to_cor(X)
        
    return X

def SMC(X):
    R = smooth_corrcoef(X)
    Rinv = la.inv(R)
    smc = 1 - 1 / np.diag(Rinv)
    return smc

def factor_to_cluster(X, cut = 0):
    var, n_factors = X.shape
    
    if n_factors == 1:
        m1 = np.ones((var, 1))
        
    else:
        m1 = np.apply_along_axis(np.argmax, 1, np.apply_along_axis(np.abs, 1, X)).reshape((var, 1))

    index = np.vstack((np.arange(0, var), m1[:, 0])).T
    
    cluster = np.zeros((var, n_factors))
    Xslice = X[index[:, 0], index[:, 1]]
    cluster[index[:, 0], index[:, 1]] = np.sign(Xslice) * ((np.abs(Xslice) > cut) + 0)
    n_items = np.sum(np.abs(cluster), axis = 0)
    
    for i in range(n_factors - 1, -1, -1):
        if n_items[i] < 1:
            cluster = sp.delete(cluster, i, 1)
            
    return cluster
