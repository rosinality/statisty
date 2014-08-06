from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_hooks()
import numpy as np
import pandas as pd

from statisty.factorization import FA

df = pd.read_csv('data/fa.csv')

loading_truth = np.array(
    [[0.808338045830633, -0.385276494691433, 0.439506785649246],
     [0.75182855431359, -0.290357263314797, 0.499546033864576],
     [0.813498154087196, -0.22888801140731, -0.529959040059742],
     [0.729164573097509, -0.13913257238425, -0.474244935206943],
     [0.801741146467149, 0.520902245234982, 0.0397714394554125],
     [0.763903067202982, 0.636130332231566, 0.0825563590210063]])

oblimin_truth = np.array(
    [[0.984791202961035, -0.0402378809539094, 0.0630598457336418],
     [0.952602621513043, 0.0531641907875769, -0.0651997947556993],
     [0.0207502805504733, -0.0217109124450527, 0.99804210307702],
     [-0.0237055701316445, 0.054290039325664, 0.864591662450278],
     [0.035510029533889, 0.9025114984696, 0.0727857969253526],
     [-0.0173867486359187, 1.02308966359429, -0.0386048238273371]])

def test_ml():
    fa = FA(df, n_factors = 3, rotate = None, method = 'ml').fit()

    assert np.allclose(fa.loadings, loading_truth, rtol = 0, atol = 1e-01)
        
def test_oblimin():
    fa = FA(df, n_factors = 3, rotate = 'oblimin', method = 'ml').fit()
    print(fa.loadings)
    assert np.allclose(fa.loadings, oblimin_truth, rtol = 0, atol = 1e-02)
