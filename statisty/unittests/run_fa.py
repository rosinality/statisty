from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_hooks()
import pandas as pd
import sys

import multiprocessing as mp

from statisty.factor import FA

if __name__ == '__main__':
    df = pd.read_csv('statisty/tests/data/fa_wiscsem.csv')
    fa = FA(df, n_factors = 3, rotate = 'oblimin').fit()
    fa.summary(loading_threshold = .3)
    mp.freeze_support()
    fa.bootstrap(niter = 100, njobs = 2)
