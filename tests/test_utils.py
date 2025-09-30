import numpy as np
from src.utils import threshold_for_min_precision

def test_threshold_for_min_precision_basic():
    # y_true: 2 positivos e 3 negativos
    y = np.array([0, 1, 0, 1, 0])
    # scores bem separáveis
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.05])
    thr = threshold_for_min_precision(y, s, min_prec=0.8)
    # qualquer thr entre ~0.8-0.9 vai manter precisão alta
    assert 0.5 <= thr <= 0.95
