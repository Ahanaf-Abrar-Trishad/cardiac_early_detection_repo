import numpy as np

def robust_ef(ed, es):
    if ed < 20: return np.nan
    x = 100.0*(ed-es)/max(ed,1e-6)
    return x if -5 <= x <= 90 else np.nan

def test_ef_valid():
    assert abs(robust_ef(200, 100) - 50.0) < 1e-6

def test_ef_tiny_edv_is_nan():
    assert np.isnan(robust_ef(5, 3))

def test_ef_out_of_range_nan():
    assert np.isnan(robust_ef(100, 210))
    assert np.isnan(robust_ef(100, 0))  # we cap at 90 in this pipeline
