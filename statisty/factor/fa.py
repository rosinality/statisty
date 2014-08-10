"""
This module implements factor analysis with continuous variables.
Almost all of the code are direct translation from psych package of R.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from future.builtins import zip
from future.builtins import int
from future.builtins import range
from future.builtins import map
from future.builtins import str
from future import standard_library
standard_library.install_hooks()
import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.stats
import scipy.optimize
import pandas as pd

import statisty as st
import statisty.utils as utils
from statisty.utils import RoseTable
import statisty.factor.rotation

import collections

import time
import multiprocessing as mp
import functools

ConfidenceInterval = collections.namedtuple('ConfidenceInterval',
                                            ['mean', 'std', 'lower', 'upper',
                                             'p'])

def bootstrap_worker(parameters, no):
    resampled = parameters['fa_options'][0][np.random.randint(0, parameters['n_obs'],
                                                  parameters['n_obs']), :]
    n_factors = parameters['fa_options'][4]
    new_parameters = parameters.copy()
    new_parameters['fa_options'][0] = resampled
    fa = FA(*new_parameters['fa_options']).fit(calc_statistics = False)

    if n_factors > 1:
        loadings, rotation, Phi = utils.target_rotation(
            fa.loadings, parameters['loadings'])
        if fa.Phi is not None:
            return loadings, np.tril(fa.Phi)

    return fa.loadings.copy(), None

class FA(object):
    """
    Factor Analysis with Continous Variables.

    This model dose most basic factor extractions with assumption of
    correlation matrix is calculated from continuous variables. If you want to
    do factor extraction with discrete variables, maybe using PolyFA
    (not implemented yet) will be better choice.

    Parameters
    ----------
    X : array-like, optional
        raw data array which can calculate correlation coefficients from.

    cov : array-like, optional
        covariance matrix. you can use this instead of raw data array (X).
        but if you use this instead of raw data array, you should specify
        the number of the observations (n_obs).

    cor : array-like, optional
        correlation matrix. you can use this instead of raw data array (X) or
        covariance matrix (cov). but if you use this instead of raw data array,
        you should specify the number of the observations (n_obs).

    n_obs : integer, optional
        number of the observations. you should specify this when you are using
        covariance or correlation matrix.

    n_factors : integer
        total number of the factors to be extracted.

    rotate : string, optional
        method for factor rotation. currently oblimin rotation supported.

    conf_int : float, optional
        confidence interval that can be used when calculate factor statistics.

    method : string, optional
        specify factor extraction methods. currently ml (maximum likelihood)
        supported.
    """
    def __init__(self, X = None, cov = None, cor = None, n_obs = None,
                 n_factors = 1, rotate = None, conf_int = .1, method = 'ml'):
        self.iscov = False
        self.pairwise_obs = None
        self.variable_names = None

        try:
            self.X = X.as_matrix() # When X is pd.DataFrame
            self.pairwise_obs = utils.pairwise_count(self.X)
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
                        self.pairwise_obs = utils.pairwise_count(self.X)
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
        self.method_rotate = {'quartimax': st.factor.rotation.quartimax,
                              'varimax': st.factor.rotation.varimax,
                              'bentlert': st.factor.rotation.bentlerT,
                              'geomint': st.factor.rotation.geominT,
                              'bifactort': st.factor.rotation.bifactorT,
                              'oblimin': st.factor.rotation.oblimin,
                              'simplimax': st.factor.rotation.simplimax,
                              'bentlerq': st.factor.rotation.bentlerQ,
                              'geominq': st.factor.rotation.geominQ,
                              'bifactorq': st.factor.rotation.bifactorQ}

        self.loadings = None

        self.bootstrapped = False

    def fit(self, calc_statistics = True):
        """
        Run factor analysis.

        Parameters
        ----------
        calc_statistics : boolean, optional
            If True (default), this function will calculate factor statistics.
            This option is mainly for the bootstrapping procedure.
        """
        # self.start = (1 - 0.5 * self.n_factors / self.n_var) / \
        #             np.diag(la.inv(self.cor))
        self.start = np.diag(self.cor) - utils.SMC(self.cor)

        if self.method == 'ml':
            self._fa_ml()

        elif self.method == 'paf':
            self._fa_paf()

        else:
            self._fa_residual(self.method)
        #self.method_fit[self.method]()

        self._sort_loadings()
        self.loadings[self.loadings == 0] = 10e-15
        model = self.loadings.dot(self.loadings.T)

        try:
            rotated_result = self.method_rotate[self.rotate.lower()](
                self.loadings, False, 1e-5, 1000)
            self.isrotated = True
            self.loadings = rotated_result.loadings
            self.Phi = rotated_result.Phi
            sign = self._sort_loadings()
            self.Phi = np.diag(sign).dot(self.Phi).dot(np.diag(sign))

        except (KeyError, TypeError, AttributeError) as e:
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

    def bootstrap(self, niter = 100, p = .05, njobs = 0):
        """
        Calculates confidence intervals of factor loadings and
        interfactor correlations by bootstrapping.

        Parameters
        ----------
        niter : integer, optional
            the number of the bootstrapping replicates.

        p : float, optional
            size of the 1 - X% confidence interval.
        
        njobs : integer, optional
            number of the cpu cores to be used in bootstrapping.
            if 0 is specified, all of the cpu cores are used for calculating.

        Notes
        -----
        Bootstrapping should be done after model fitting; if model fitting
        doesn't done, this method automatically calls fit() method.
        """
        if self.loadings is None:
            self.fit()
        
        replicates = np.empty((niter, self.n_var, self.n_factors))
        replicates_Phi = np.empty((niter, self.n_factors, self.n_factors))

        parameter = {'n_obs': self.n_obs,
                     'loadings': self.loadings,
                     'fa_options': [self.X, None, None, self.n_obs,
                                    self.n_factors, self.rotate, self.conf_int,
                                    self.method]}
        start = time.time()
        worker = functools.partial(bootstrap_worker, parameter)
        progress = utils.progress_bar(niter)
        
        if njobs == 0:
            jobs = mp.cpu_count()

        else:
            jobs = njobs

        pool = mp.Pool(jobs)

        for i, results in enumerate(pool.imap_unordered(worker, range(niter))):
            progress.update(i)
            replicates[i], replicates_Phi[i] = results

        pool.close()

        ppf_l = sp.stats.norm.ppf(p / 2)
        ppf_u = sp.stats.norm.ppf(1 - p / 2)

        ci_mean = np.mean(replicates, axis = 0)
        ci_std = np.std(replicates, axis = 0)
        tci = np.abs(ci_mean) / ci_std
        ptci = 1 - sp.stats.norm.cdf(tci)

        if replicates_Phi[0] is not None:
            rot_mean = np.mean(replicates_Phi, axis = 0)
            rot_std = np.std(replicates_Phi, axis = 0)
            rot_lower = rot_mean + ppf_l * rot_std
            rot_upper = rot_mean + ppf_u * rot_std
            rot_tci = np.abs(rot_mean) / rot_std
            rot_ptci = 1 - sp.stats.norm.cdf(rot_tci)

            self.ci_rotation = ConfidenceInterval(
                mean = rot_mean, std = rot_std,
                p = 2 * rot_tci,
                lower = rot_mean + ppf_l * rot_std,
                upper = rot_mean + ppf_u * rot_std)

        else:
            self.ci_rotation = None

        self.ci_loadings = ConfidenceInterval(
            mean = ci_mean, std = ci_std,
            p = 2 * ptci,
            lower = ci_mean + ppf_l * ci_std,
            upper = ci_mean + ppf_u * ci_std)

        self.bootstrapped = True
        
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

        self.dof = var * (var - 1) / 2 - var * n_factors + (
            n_factors * (n_factors - 1) / 2)
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
            self.harmonic = utils.harmonic_mean(np.hstack(pairwise_obs.T))
            self.rms = np.sqrt(Xstar_off / (self.harmonic * var * (var - 1)))

            if self.dof > 0:
                self.EPVAL = sp.stats.chi2.sf(self.chi, self.dof)
                self.crms = np.sqrt(Xstar_off / (2 * self.harmonic * self.dof))
                self.EBIC = self.chi - self.dof * np.log(obs)
                self.ESABIC = self.chi - self.dof * np.log(
                    (self.harmonic + 2) / 24)

            else:
                self.EPVAL = None
                self.crms = None
                self.EBIC = None
                self.ESABIC = None

        self.fit_result = 1 - Xstar2 / X2
        self.fit_off = 1 - Xstar_off / X2_off
        self.sd = np.std(residual, ddof = 1)
        self.complexity = np.apply_along_axis(
            lambda x: np.sum(x ** 2), 1, loadings) ** 2 \
            / np.apply_along_axis(lambda x: np.sum(x ** 4), 1, loadings)
        model[np.diag_indices_from(model)] = np.diag(X)
        model = utils.smooth_corrcoef(model)
        X = utils.smooth_corrcoef(X)
        model_inv = la.solve(model, X)
        self.objective = np.sum(np.diag(model_inv)) \
                         - np.log(la.det(model_inv)) - var
        chisq = self.objective * ((obs - 1) - (2 * var + 5) / 6 \
                                  - (2 * n_factors) / 3)
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
        Mm = Fm / (var * (var - 1) / 2 - var * n_factors + \
                   (n_factors * (n_factors - 1) / 2))
        M0 = F0 * 2 / (var * (var - 1))
        nm = (obs - 1) - (2 * var + 5) / 6 - (2 * n_factors) / 3
        self.null_model = F0
        self.null_dof = var * (var - 1) / 2
        self.null_chi2 = F0 * ((obs - 1) - (2 * var + 5) / 6)
        self.TLI = (M0 - Mm) / (M0 - 1 / nm)
        if not np.isnan(self.TLI) and self.TLI > 1:
            self.F0 = 1

        if self.dof > 0 and not np.isnan(self.objective):
            RMSEA = np.sqrt(
                np.max(self.objective / self.dof - 1 / (obs - 1), 0))
            tail = alpha / 2
            chi_max = max(obs, chisq) + 2 * obs

            while chi_max > 1:
                opt_res = sp.optimize.minimize_scalar(
                    lambda x: \
                    (tail - sp.stats.ncx2.cdf(chisq, self.dof, x)) ** 2,
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
                    lambda x: \
                    (1 - tail - sp.stats.ncx2.cdf(chisq, self.dof, x)) ** 2,
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

            eigval, eigvec = utils.eigenh_sorted(X)
            if np.sum(np.iscomplex()) == 0:
                print('Complex eigenvalues are detected. results are suspect.')

            else:
                eigval[eigval < np.finfo(np.float64).eps] = 100 * np.finfo(
                    np.float64).eps
                X = eigvec.dot(np.diag(eigval)).dot(eigvec.T)
                np.fill_diagonal(X, 1)

                try:
                    W = la.solve(X, loadings)

                except la.LinAlgError:
                    print('Failed to calculate the beta weights'
                          ' for factor score estimates')
                    W = np.diag(np.ones((var, )))

        R2 = np.diag(W.T.dot(loadings))
        if np.prod(R2) < 0:
            print('Factor scoring weights matrix is probably singular;'
                  ' Factor score estimate results are likely to incorrect.'
                  ' Try a different factor extraction method.')
            R2[np.abs(R2) > 1] = np.nan
            R2[R2 <= 0] = np.nan

        if np.nanmax(R2) > (1 + np.finfo(np.float64).eps):
            print('The estimated weights for the factor scores'
                  ' are probably incorrect.'
                  'Try a different factor extraction method.')

        self.Rscores = utils.cov_to_cor(W.T.dot(X).dot(W))
        self.R2 = R2

        keys = utils.factor_to_cluster(loadings)
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
        """
        Prints a summary of the factor analysis results.

        Parameters
        ----------
        loading_threshold : integer, optional
            threshold of loading filtering. if non-zero value spcified,
            loadings that < abs(loading_threshold) will be skipped and
            not printed in summary table.
        """
        var, n_factors = self.loadings.shape
        fac_header = ['']
        var_names = []
        loadings = self.loadings.copy()
        for i in range(1, n_factors + 1):
            fac_header.append('ML' + str(i))

        if self.variable_names is None:
            for i in range(1, var + 1):
                var_names.append('Var' + str(i))

        else:
            var_names = self.variable_names

        load_header = ['']
        if self.bootstrapped:
            for header in fac_header[1:]:
                load_header.append('[low')
                load_header.append(header)
                load_header.append('up]')
        else:
            load_header = fac_header.copy()
        load_header.extend(['h2', 'uniq', 'com'])

        x = RoseTable()
        x.char_header = '-'
        x.char_center_top = '='
        x.title = 'Exploratory Factor Analysis Results'
        x.add_header([''] * 4, align = ['l', 'r', 'l', 'r'])
        method_table = {'ml': 'Maximum Likelihood',
                        'paf': 'Principal Axis',
                        'minchi': 'Minimum Chi^2',
                        'gls': 'GLS',
                        'ols': 'OLS',
                        'wls': 'WLS'}
        x.add_row(['Number of factors:', n_factors,
                   'Estimation method:', method_table[self.method]])
        if self.rotate is None:
            method_name = 'None'
        else:
            method_name = self.rotate[0].upper() + self.rotate[1:]
        x.add_row(['Date:', self.fitted_date, 'Rotation method:', method_name])
        x.add_row(['Time:', self.fitted_time,
                   'Mean item complexity',
                   '{:10.1f}'.format(np.mean(self.complexity))])

        x.add_separator()
        x.add_title('Test of the hypothesis that '
                    + str(n_factors) +
                    ' factors are sufficient')
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
                         '90% Conf. Int.',
                         "{:1.3f} {:1.3f}".format(
                             self.RMSEA[1], self.RMSEA[2])])
        except:
            pass
        x.add_row(temp)
        try:
            temp = ['BIC:', "{:10.2f}".format(self.BIC)]

        except:
            temp = []
        temp.extend(['Fit upon off diagonal:',
                     '{:10.2f}'.format(self.fit_off)])
        x.add_row(temp)

        if self.bootstrapped:
            x.add_header(load_header,
                         align = ['l'] + ['r'] * (n_factors * 3 + 3))
            #x.add_header(load_header, align = ['l'] + ['l', 'r', 'r'] * n_factors + ['r'] * 3)
        else:
            x.add_header(load_header, align = ['l'] + ['r'] * (n_factors + 3))

        if self.n_factors > 1:
            if self.isrotated:
                h2 = np.diag(loadings.dot(self.Phi).dot(loadings.T))

            else:
                h2 = np.sum(loadings ** 2, axis = 1)

        else:
            h2 = loadings ** 2

        var_total = np.sum(h2 + self.uniqueness)

        if self.bootstrapped:
            zip_load = zip(var_names, loadings, h2,
                           self.uniqueness, self.complexity,
                           self.ci_loadings.lower.copy(),
                           self.ci_loadings.upper.copy())
        else:
            zip_load = zip(var_names, loadings, h2,
                           self.uniqueness, self.complexity)

        for row in zip_load:
            temp_row = [row[0]]
            row[1][np.abs(row[1]) < loading_threshold] = 0
            thresholded = [
                '{:3.2f}'.format(i) if i != 0 else '' for i in row[1]]
            if self.bootstrapped:
                row[5][np.abs(row[5]) < loading_threshold] = 0
                row[6][np.abs(row[6]) < loading_threshold] = 0
                for i, val in enumerate(thresholded):
                    temp_row.append('{:3.2f}'.format(
                        row[5][i]) if val != '' else '')
                    temp_row.append(val)
                    temp_row.append('{:3.2f}'.format(
                        row[6][i]) if val != '' else '')
            else:
                temp_row.extend(thresholded)
            temp_row.append("{:3.3f}".format(row[2]))
            temp_row.append("{:3.2f}".format(row[3]))
            temp_row.append("{:3.1f}".format(row[4]))
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

        var_header = [['SS loadings',
                       map(lambda x: "{:3.2f}".format(x),
                           ss_load)],
                      ['Proportion Var',
                       map(lambda x: "{:3.2f}".format(x),
                           ss_load / var_total)],
                      ['Cumulative Var',
                       map(lambda x: "{:3.2f}".format(x),
                           np.cumsum(ss_load / var_total))],
                      ['Proportion Explained',
                       map(lambda x: "{:3.2f}".format(x),
                           ss_load / np.sum(ss_load))],
                      ['Cumulative Proportion',
                       map(lambda x: "{:3.2f}".format(x),
                           np.cumsum(ss_load / np.sum(ss_load)))]
        ]

        for row in var_header:
            temp_row = [row[0]]
            temp_row.extend(row[1])
            x.add_row(temp_row)

        if self.isrotated:
            corr_header = fac_header.copy()
            

            corr_header = ['']
            if self.bootstrapped:
                for header in fac_header[1:]:
                    corr_header.append('[low')
                    corr_header.append(header)
                    corr_header.append('up]')
            else:
                corr_header = fac_header.copy()
            if self.method == 'paf':
                corr_header[0] = 'Component correlations'

            else:
                corr_header[0] = 'Factor correlations'

            if self.bootstrapped:
                x.add_header(corr_header,
                             align = ['l'] + ['r'] * (n_factors))

            else:
                x.add_header(corr_header,
                             align = ['l'] + ['r'] * (n_factors * 3))
                
            for index in range(n_factors):
                temp_row = [fac_header[index + 1]]
                correlations = list(map(lambda x: "{:4.2f}".format(x),
                                        self.Phi[index, :]))
                if self.bootstrapped and self.ci_rotation is not None:
                    for i, val in enumerate(correlations[:index + 1]):
                        temp_row.append('{:4.2f}'.format(
                            self.ci_rotation.lower[index, i]))
                        temp_row.append(val)
                        temp_row.append('{:4.2f}'.format(
                            self.ci_rotation.upper[index, i]))
                else:
                    temp_row.extend(correlations[:index + 1])
                x.add_row(temp_row)


        if self.method != 'paf':
            measure_row = [['Corr. of scores with factors',
                            map(lambda x: "{:3.2f}".format(x),
                                np.sqrt(self.R2))],
                           ['Multiple R^2 of scores with factors',
                            map(lambda x: "{:3.2f}".format(x),
                                self.R2)],
                           ['Min. Corr. of possible factor scores',
                            map(lambda x: "{:3.2f}".format(x),
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

    def _fa_paf(self, smc = True):
        cor = self.cor.copy()
        if smc:
            np.fill_diagonal(cor, utils.SMC(cor))
        eig_val1 = la.eigvalsh(self.cor)
        eig_val1.sort()
        comm = np.sum(np.diag(cor))
        err = comm
        comm_list = []
        i = 0
        while err > self.lower:
            eig_val, eig_vec = utils.eigenh_sorted(cor)
            if (self.n_factors > 1):
                loadings = eig_vec[:, :self.n_factors].dot(
                    np.diag(np.sqrt(eig_val[:self.n_factors])))
            else:
                loadings = eig_vec[:, 0] * np.sqrt(eig_val[0])

            model = loadings.dot(loadings.T)
            new = np.diag(model)
            comm1 = np.sum(new)
            np.fill_diagonal(cor, new)
            err = np.abs(comm - comm1)
            if np.iscomplex(err):
                print('Imaginary eigenvalue condition'
                      ' occurred!')
                break
            comm = comm1
            comm_list.append(comm1)
            i += 1
            if i > 1000:
                print('maximum iteration exceeded')
                err = 0

        self.loadings = loadings

    def _fa_residual(self, method = 'wls'):
        def residual_function(Psi, S, n_factors, Sinv, method, obs):
            S1 = S.copy()
            np.fill_diagonal(S1, 1 - Psi)
            if (Sinv is not None):
                Sdinv = np.diag(1 / np.diag(Sinv))
            eig_val, eig_vec = utils.eigen_sorted(S1)
            eig_val[eig_val < np.finfo(np.float64).eps] = \
                                            100 * np.finfo(
                                                np.float64).eps * 100
            if n_factors > 1:
                loadings = eig_vec[:, :n_factors].dot(
                    np.diag(np.sqrt(eig_val[:n_factors])))

            else:
                loadings = eig_vec[:, 0:1] * np.sqrt(eig_val[0])

            model = loadings.dot(loadings.T)

            if method == 'wls':
                residual = Sdinv.dot((S1 - model) ** 2).dot(Sdinv)

            elif method == 'gls':
                residual = (Sinv.dot(S1 - model)) ** 2
                
            else:
                residual = (S1 - model) ** 2

                if method == 'minres':
                    np.fill_diagonal(residual, 0)
                    
                else: # minchi
                    residual *= obs
                    np.fill_diagonal(residual, 0)

            return np.sum(residual)

        def residual_gradient(Psi, S, n_factors, Sinv, method, obs):
            sc = np.diag(1 / np.sqrt(Psi))
            Sstar = sc.dot(S).dot(sc)
            eig_val, eig_vec = utils.eigenh_sorted(Sstar)
            L = eig_vec[:, :n_factors]
            load = L.dot(np.diag(np.sqrt(np.fmax(eig_val[:n_factors] - 1, 0))))
            load = np.diag(np.sqrt(Psi)).dot(load)
            g = load.dot(load.T) + np.diag(Psi) - S

            return np.diag(g) / Psi ** 2

        def out_wls(Psi, S, q):
            S1 = S.copy()
            np.fill_diagonal(S1, 1 - Psi)
            eig_val, eig_vec = utils.eigen_sorted(S1)
            L = eig_vec[:, :q].dot(np.diag(np.sqrt(eig_val[:q])))

            return L

        if method == 'wls' or method == 'gls':
            Sinv = la.inv(self.cor)

        else:
            Sinv = None

        result = sp.optimize.minimize(residual_function, self.start,
                                      args = (self.cor, self.n_factors,
                                              Sinv, method, self.n_obs),
                                      method = 'L-BFGS-B',
                                      jac = residual_gradient,
                                      bounds = [(self.lower, 1)] * self.n_var)

        if method == 'wls' or method == 'gls':
            self.loadings = out_wls(result.x, self.cor, self.n_factors)
            
        else:
            self.loadings = self._fa_out(result.x, self.cor, self.n_factors)
            
        self.parameter = result.x

    def _fa_out(self, Psi, S, q):
        sc = np.diag(1 / np.sqrt(Psi))
        Sstar = sc.dot(S).dot(sc)
        eig_val, eig_vec = utils.eigenh_sorted(Sstar)
        L = eig_vec[:, :q]
        load = L.dot(np.diag(np.sqrt(np.maximum(eig_val[:q] - 1, 0))))

        return np.diag(np.sqrt(Psi)).dot(load)

    def _fa_ml(self):
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
            eig_val, eig_vec = utils.eigenh_sorted(Sstar)
            L = eig_vec[:, :q]
            load = L.dot(np.diag(np.sqrt(np.maximum(eig_val[:q] - 1, 0))))
            load = np.diag(np.sqrt(Psi)).dot(load)
            g = load.dot(load.T) + np.diag(Psi) - S

            return np.diag(g) / (Psi ** 2)

        result = sp.optimize.minimize(ml_function, self.start,
                                      args = (self.cor, self.n_factors),
                                      method = 'L-BFGS-B', jac = ml_gradient,
                                      bounds = [(self.lower, 1)] * self.n_var)

        self.loadings = self._fa_out(result.x, self.cor, self.n_factors)
        self.parameter = result.x
