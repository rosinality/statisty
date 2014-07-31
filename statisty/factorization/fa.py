import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.stats
import scipy.optimize
import pandas as pd

import statisty as st
import statisty.utils
from statisty.utils import RoseTable
import statisty.factorization.rotation

import time
import multiprocessing as mp
import functools

from sklearn.decomposition import FastICA

def bootstrap_worker(parameters, no):
    resampled = parameters['fa_options'][0][np.random.randint(0, parameters['n_obs'],
                                                  parameters['n_obs']), :]
    fa = FA(*parameters['fa_options']).fit(calc_statistics = False)

    return fa.loadings

class FA(object):
    """
    """
    def __init__(self, X = None, cov = None, cor = None, n_obs = None, n_factors = 1,
                 rotate = None, conf_int = .1, method = 'ml'):
        self.iscov = False
        self.pairwise_obs = None
        self.variable_names = None
        
        try:
            self.X = X.as_matrix() # When X is pd.DataFrame
            self.pairwise_obs = st.utils.pairwise_count(self.X)
            self.n_obs = self.X.shape[0]
            self.cor = np.corrcoef(X, rowvar = 0)
            self.variable_names = X.columns.values.tolist()
            
        except AttributeError:
            if cov is not None:
                self.cor = cov
                self.iscov = True

                if n_obs is None:
                    raise ValueError('The number of observations should'
                                     ' specified when using covariance'
                                     ' matrix for analysis')
                self.n_obs = n_obs

            else:
                if cor is not None:
                    self.cor = cor
                    if n_obs is None:
                        raise ValueError('The number of observations should'
                                         ' specified when using covariance'
                                         ' matrix for analysis')
                        
                    self.n_obs = n_obs

                else:
                    if isinstance(X, np.ndarray):
                        self.X = X
                        self.pairwise_obs = st.utils.pairwise_count(self.X)
                        self.cor = np.corrcoef(X, rowvar = 0)
                        self.n_obs = self.X.shape[0]

                    else:
                        raise ValueError('Raw dataframe, or raw data matrix'
                                         ' or covariance/correlation matrix'
                                         ' should be given for analysis')
                        
        self.n_var = self.cor.shape[1]
        self.n_factors = n_factors
        self.conf_int = conf_int
        self.rotate = rotate
        
        self.lower = .001

        self.method = method
        self.method_fit = {'ml': self._fa_ml}
        self.method_rotate = {'oblimin': st.factorization.rotation.oblimin}
        
    def fit(self, calc_statistics = True):
        #self.start = (1 - 0.5 * self.n_factors / self.n_var) / \
        #             np.diag(la.inv(self.cor))
        self.start = np.diag(self.cor) - st.utils.SMC(self.cor)
        self.method_fit[self.method]()

        self._sort_loadings()
        self.loadings[self.loadings == 0] = 10e-15
        model = self.loadings.dot(self.loadings.T)

        try:
            rotated_result = self.method_rotate[self.rotate](self.loadings, 0, False, 1e-5, 1000)
            self.isrotated = True
            self.loadings = rotated_result.loadings
            self.Phi = rotated_result.Phi
            sign = self._sort_loadings()
            self.Phi = np.diag(sign).dot(self.Phi).dot(np.diag(sign))
            
        except KeyError:
            self.isrotated = False
            self.Phi = None

        ev_rotated = np.diag(self.loadings.T.dot(self.loadings))
        ev_order = np.argsort(ev_rotated)[::-1]
        self.loadings = self.loadings[:, ev_order]

        if self.isrotated:
            self.Phi = self.Phi[:, ev_order][ev_order, :]

        self.uniqueness = np.diag(self.cor - model)
        self.communality = np.diag(model)

        if calc_statistics:
            self._fa_stats()

        time_now = time.localtime()
        self.fitted_time = time.strftime('%H:%M:%S', time_now)
        self.fitted_date = time.strftime('%a, %d %b %Y', time_now)

        return self

    def bootstrap(self, niter = 100, njobs = -1):
        self.replicates = np.empty((niter, self.n_var, self.n_factors))
        self.replicate_rotations = None

        global parameter
        parameter = {'n_obs': self.n_obs,
                     'fa_options': [self.X, None, None, self.n_obs,
                                    self.n_factors, self.rotate, self.conf_int,
                                    self.method]}

        if njobs == -1:
            jobs = mp.cpu_count()

        else:
            jobs = njobs

        p = mp.Pool(jobs)
        
        start = time.time()
        prog_start = time.time()
        worker = functools.partial(bootstrap_worker, parameter)
        progress = st.utils.progress_bar(niter)

        for i, loadings in enumerate(p.imap_unordered(worker, range(niter))):
            progress.update(i)
            self.replicates[i] = loadings
        end = time.time()

    def _fa_stats(self):
        X = self.cor.copy()
        loadings = self.loadings.copy()
        if self.Phi is None:
            model = loadings.dot(loadings.T)
        else:
            Phi = self.Phi.copy()
            model = loadings.dot(Phi).dot(loadings.T)
        obs = self.n_obs
        pairwise_obs = self.pairwise_obs
        alpha = self.conf_int

        var = X.shape[1]
        n_factors = loadings.shape[1]

        residual = X - model
        self.residual = residual.copy()
        X2 = np.sum(X ** 2)
        Xstar2 = np.sum(residual ** 2)

        self.dof = var * (var - 1) / 2 - var * n_factors + (n_factors * (n_factors - 1) / 2)
        X2_off = X2 - np.trace(X)
        np.fill_diagonal(residual, 0)

        if self.pairwise_obs is None:
            Xstar_off = np.sum(residual ** 2)
            self.ENull = X2_off * obs
            self.chi = Xstar_off * obs
            self.rms = np.sqrt(Xstar_off / (var * (var - 1)))
            self.harmonic = obs

        else:
            Xstar_off = np.sum(residual ** 2 * pairwise_obs)
            X2_off = (X * X * pairwise_obs)
            X2_off = np.sum(X2_off) - np.trace(X2_off)
            self.chi = Xstar_off
            self.harmonic = st.utils.harmonic_mean(np.hstack(pairwise_obs.T))
            self.rms = np.sqrt(Xstar_off / (self.harmonic * var * (var - 1)))

            if self.dof > 0:
                self.EPVAL = sp.stats.chi2.sf(self.chi, self.dof)
                self.crms = np.sqrt(Xstar_off / (2 * self.harmonic * self.dof))
                self.EBIC = self.chi - self.dof * np.log(obs)
                self.ESABIC = self.chi - self.dof * np.log((self.harmonic + 2) / 24)

            else:
                self.EPVAL = None
                self.crms = None
                self.EBIC = None
                self.ESABIC = None

        self.fit_result = 1 - Xstar2 / X2
        self.fit_off = 1 - Xstar_off / X2_off
        self.sd = np.std(residual, ddof = 1)
        self.complexity = np.apply_along_axis(lambda x: np.sum(x ** 2), 1, loadings) ** 2 \
                               / np.apply_along_axis(lambda x: np.sum(x ** 4), 1, loadings)
        model[np.diag_indices_from(model)] = np.diag(X)
        model = st.utils.smooth_corrcoef(model)
        X = st.utils.smooth_corrcoef(X)
        model_inv = la.solve(model, X)
        self.objective = np.sum(np.diag(model_inv)) - np.log(la.det(model_inv)) - var
        chisq = self.objective * ((obs - 1) - (2 * var + 5) / 6 - (2 * n_factors) / 3)
        if chisq < 0:
            self.statistic = 0

        else:
            self.statistic = chisq

        if self.dof > 0:
            self.PVAL = sp.stats.chi2.sf(self.statistic, self.dof)
        else:
            self.PVAL = None

        F0 = np.sum(np.diag(X)) - np.log(la.det(X)) - var
        Fm = self.objective
        Mm = Fm / (var * (var - 1) / 2 - var * n_factors + (n_factors * (n_factors - 1) / 2))
        M0 = F0 * 2 / (var * (var - 1))
        nm = (obs - 1) - (2 * var + 5) / 6 - (2 * n_factors) / 3
        self.null_model = F0
        self.null_dof = var * (var - 1) / 2
        self.null_chi2 = F0 * ((obs - 1) - (2 * var + 5) / 6)
        self.TLI = (M0 - Mm) / (M0 - 1 / nm)
        if not np.isnan(self.TLI) and self.TLI > 1:
            self.F0 = 1

        if self.dof > 0 and not np.isnan(self.objective):
            RMSEA = np.sqrt(np.max(self.objective / self.dof - 1 / (obs - 1), 0))
            tail = alpha / 2
            chi_max = max(obs, chisq) + 2 * obs

            while chi_max > 1:
                opt_res = sp.optimize.minimize_scalar(
                    lambda x: (tail - sp.stats.ncx2.cdf(chisq, self.dof, x)) ** 2,
                    bracket = (0, chi_max))

                if np.sqrt(opt_res.fun) < tail / 100:
                    break

                chi_max = chi_max / 2

            if chi_max <= 1:
                lamU = np.nan
                chi_max = np.nan

            else:
                lamU = opt_res.x
                chi_max = lamU

            while (chi_max > 1):
                opt_res = sp.optimize.minimize_scalar(
                    lambda x: (1 - tail - sp.stats.ncx2.cdf(chisq, self.dof, x)) ** 2,
                    bracket = (0, chi_max))

                if np.sqrt(opt_res.fun) < tail / 100:
                    break

                chi_max = chi_max / 2

            if chi_max <= 1:
                lamL = np.nan

            else:
                lamL = opt_res.x

            RMSEA_U = np.sqrt(lamU / (obs * self.dof))
            RMSEA_L = min(np.sqrt(lamL / (obs * self.dof)), RMSEA)

            self.RMSEA = [RMSEA, RMSEA_L, RMSEA_U, alpha]
            self.BIC = chisq - self.dof * np.log(obs)
            self.SABIC = chisq - self.dof * np.log((obs + 2) / 24)

        if self.Phi is not None:
            loadings = loadings.dot(Phi)

        try:
            W = la.solve(X, loadings)

        except la.LinAlgError:
            print('Correlation matrix is singular; approximation used')
            
            eigval, eigvec = st.utils.eigenh_sorted(X)
            if np.sum(np.iscomplex()) == 0:
                print('Complex eigenvalues are detected. results are suspect.')

            else:
                eigval[eigval < np.finfo(np.float64).eps] = 100 * np.finfo(np.float64).eps
                X = eigvec.dot(np.diag(eigval)).dot(eigvec.T)
                np.fill_diagonal(X, 1)

                try:
                    W = la.solve(X, loadings)

                except la.LinAlgError:
                    print('Failed to calculate the beta weights for factor score estimates')
                    W = np.diag(np.ones((var, )))

        R2 = np.diag(W.T.dot(loadings))
        if np.prod(R2) < 0:
            print('Factor scoring weights matrix is probably singular;'
                  ' Factor score estimate results are likely to incorrect.'
                  ' Try a different factor extraction method.')
            R2[np.abs(R2) > 1] = np.nan
            R2[R2 <= 0] = np.nan

        if np.nanmax(R2) > (1 + np.finfo(np.float64).eps):
            print('The estimated weights for the factor scores are probably incorrect.'
                  'Try a different factor extraction method.')

        self.Rscores = st.utils.cov_to_cor(W.T.dot(X).dot(W))
        self.R2 = R2

        keys = st.utils.factor_to_cluster(loadings)
        covar = keys.T.dot(X).dot(keys)

        if n_factors > 1 and covar.shape[1] > 1:
            sdinv = np.diag(1 / np.sqrt(np.diag(covar)))
            cluster_corr = sdinv.dot(covar).dot(sdinv)
            valid = loadings.T.dot(keys).dot(sdinv)
            self.valid = np.diag(valid)
            self.score_corr = cluster_corr

        else:
            sdinv = 1 / np.sqrt(covar)
            if (sdinv.shape[0] == 1):
                sdinv = np.diag(sdinv)
                self.valid = loadings.T.dot(keys).dot(sdinv)

        self.weights = W

    def plot(self):
        pass

    def summary(self, loading_threshold = 0):
        var, n_factors = self.loadings.shape
        fac_header = ['']
        var_names = []
        loadings = self.loadings
        for i in range(1, n_factors + 1):
            fac_header.append('ML' + str(i))

        if self.variable_names is None:
            for i in range(1, var + 1):
                var_names.append('Var' + str(i))

        else:
            var_names = self.variable_names
            
        load_header = fac_header.copy()
        load_header.extend(['h2', 'uniq', 'com'])

        x = RoseTable()
        x.char_header = '-'
        x.char_center_top = '='
        x.title = 'Exploratory Factor Analysis Results'
        x.add_header([''] * 4, align = ['l', 'r', 'l', 'r'])
        x.add_row(['Number of factors:', n_factors, 'Estimation method:', 'Maximum Likelihood'])
        x.add_row(['Date:', self.fitted_date, 'Rotation method:', 'Oblimin'])
        x.add_row(['Time:', self.fitted_time, 'Mean item complexity', '{:10.1f}'.format(np.mean(self.complexity))])

        x.add_title('Test of the hypothesis that ' + str(n_factors) + ' factors are sufficient')
        x.add_header([''] * 6, align = ['l', 'r', 'l', 'r', 'l', 'r'])
        x.add_row(['dof(Null model):', self.null_dof,
                   'Objective function:', "{:10.2f}".format(self.null_model),
                   'Chi^2:', "{:10.2f}".format(self.null_chi2)])
        x.add_row(['dof(Model):', self.dof,
                   'Objective function:', "{:10.2f}".format(self.objective)])
        temp = ['RMSR:', "{:10.2f}".format(self.rms)]
        if self.crms is not None:
            temp.extend(['dof(corrected RMSR):', "{:10.2f}".format(self.crms)])
        x.add_row(temp)
        temp = ['Harmonic number of obs:', int(self.harmonic),
                'Emp. Chi^2:', "{:10.2f}".format(self.chi)]
        if self.EPVAL is not None:
            temp.extend(['Prob <', "{:10.1e}".format(self.EPVAL)])
        x.add_row(temp)
        temp = ['Total number of obs:', self.n_obs,
                'MLE Chi^2:', "{:10.2f}".format(self.statistic)]
        if self.PVAL is not None:
            temp.extend(['Prob <', "{:10.1e}".format(self.PVAL)])
        x.add_row(temp)
        temp = ['TLI:', "{:10.3f}".format(self.TLI)]
        try:
            temp.extend(['RMSEA index:', "{:10.3f}".format(self.RMSEA[0]),
                         '90% Conf. Int.', "{:1.3f} {:1.3f}".format(self.RMSEA[1], self.RMSEA[2])])
        except:
            pass
        x.add_row(temp)
        try:
            temp = ['BIC:', "{:10.2f}".format(self.BIC)]

        except:
            temp = []
        temp.extend(['Fit upon off diagonal:', '{:10.2f}'.format(self.fit_off)])
        x.add_row(temp)

        x.add_header(load_header, align = ['l'] + ['r'] * (n_factors + 3))

        if self.n_factors > 1:
            if self.isrotated:
                h2 = np.diag(loadings.dot(self.Phi).dot(loadings.T))

            else:
                h2 = np.sum(loadings ** 2, axis = 1)
            
        else:
            h2 = loadings ** 2
        
        var_total = np.sum(h2 + self.uniqueness)
        
        for row in zip(var_names, loadings, h2, self.uniqueness, self.complexity):
            temp_row = [row[0]]
            row[1][np.abs(row[1]) < loading_threshold] = 0
            thresholded = ['{:10.2f}'.format(i) if i != 0 else '' for i in row[1]]
            temp_row.extend(thresholded)
            temp_row.append("{:10.3f}".format(row[2]))
            temp_row.append("{:10.2f}".format(row[3]))
            temp_row.append("{:10.1f}".format(row[4]))
            x.add_row(temp_row)

        x.add_header(fac_header,
                     align = ['l'] + ['r'] * (n_factors))

        if self.Phi is None:
            if self.n_factors > 1:
                ss_load = np.sum(loadings ** 2, axis = 0)

            else:
                ss_load = np.sum(loadings ** 2)

        else:
            ss_load = np.diag(self.Phi.dot(loadings.T).dot(loadings))

        var_header = [['SS loadings', map(lambda x: "{:10.2f}".format(x), ss_load)],
                      ['Proportion Var', map(lambda x: "{:10.2f}".format(x), ss_load / var_total)],
                      ['Cumulative Var', map(lambda x: "{:10.2f}".format(x), np.cumsum(ss_load / var_total))],
                      ['Proportion Explained', map(lambda x: "{:10.2f}".format(x), ss_load / np.sum(ss_load))],
                      ['Cumulative Proportion', map(lambda x: "{:10.2f}".format(x),
                                                    np.cumsum(ss_load / np.sum(ss_load)))]
        ]

        for row in var_header:
            temp_row = [row[0]]
            temp_row.extend(row[1])
            x.add_row(temp_row)

        if self.isrotated:
            corr_header = fac_header.copy()
            if self.method == 'paf':
                corr_header[0] = 'Component correlations'

            else:
                corr_header[0] = 'Factor correlations'

            x.add_header(corr_header, align = ['l'] + ['r'] * (n_factors))
            for index, row in enumerate(corr_header[1:]):
                temp_row = [row]
                correlations = list(map(lambda x: "{:10.2f}".format(x), self.Phi[index, :]))
                temp_row.extend(correlations[:index + 1])
                x.add_row(temp_row)


        if self.method != 'paf':
            measure_row = [['Corr. of scores with factors', map(lambda x: "{:10.2f}".format(x), np.sqrt(self.R2))],
                           ['Multiple R^2 of scores with factors', map(lambda x: "{:10.2f}".format(x), self.R2)],
                           ['Min. Corr. of possible factor scores', map(lambda x: "{:10.2f}".format(x),
                                                                        2 * self.R2 - 1)],
            ]
            measure_header = fac_header.copy()
            measure_header[0] = 'Measures of factor score adequacy'
            x.add_header(measure_header,
                         align = ['l'] + ['r'] * (n_factors))
            for row in measure_row:
                temp_row = [row[0]]
                temp_row.extend(row[1])
                x.add_row(temp_row)

        print(x)

    def _sort_loadings(self):
        #    ssq = -np.sum(X ** 2, axis = 0)
        #    X = X[:, ssq.argsort()]
        #    neg = np.sum(X, axis = 0) < 0
        #    X[:, neg] = -X[:, neg]
        sign = np.sign(np.sum(self.loadings, axis = 0))
        sign[sign == 0] = 1

        self.loadings = self.loadings.dot(np.diag(sign))

        return sign

    def _fa_ml(self):
        def ml_out(Psi, S, q):
            sc = np.diag(1 / np.sqrt(Psi))
            Sstar = sc.dot(S).dot(sc)
            eig_val, eig_vec = st.utils.eigenh_sorted(Sstar)
            L = eig_vec[:, :q]
            load = L.dot(np.diag(np.sqrt(np.maximum(eig_val[:q] - 1, 0))))
        
            return np.diag(np.sqrt(Psi)).dot(load)

        def ml_function(Psi, S, q):
            sc = np.diag(1 / np.sqrt(Psi))
            Sstar = sc.dot(S).dot(sc)
            eig_val = la.eigvalsh(Sstar)
            eig_val.sort()
            eig_val = eig_val[::-1]
            e = eig_val[-(eig_val.shape[0] - q):]
            e = np.sum(np.log(e) - e) - q + S.shape[0]
        
            return -e

        def ml_gradient(Psi, S, q):
            sc = np.diag(1 / np.sqrt(Psi))
            Sstar = sc.dot(S).dot(sc)
            eig_val, eig_vec = st.utils.eigenh_sorted(Sstar)
            L = eig_vec[:, :q]
            load = L.dot(np.diag(np.sqrt(np.maximum(eig_val[:q] - 1, 0))))
            load = np.diag(np.sqrt(Psi)).dot(load)
            g = load.dot(load.T) + np.diag(Psi) - S
        
            return np.diag(g) / (Psi ** 2)
        
        result = sp.optimize.minimize(ml_function, self.start,
                                      args = (self.cor, self.n_factors),
                                      method = 'L-BFGS-B', jac = ml_gradient,
                                      bounds = [(self.lower, 1)] * self.n_var)
        
        self.loadings = ml_out(result.x, self.cor, self.n_factors)
        self.parameter = result.x
