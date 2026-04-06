import pytest as pt
import numpy as np
import pandas as pd
import vec_analytics as va

# CREATE FIXTURES FOR TESTS.

# Create Quantile fixtures...
@pt.fixture
def wqts1():
    """
        Take a vector, 'xs', and a uniform weight vector, 'ws', along
        with a vector of quantiles, 'qs', return the quantile values from 'xs'.
    """
    np.random.seed(1)
    xs = np.linspace(0.0, 10.0, 2000)          # Data vector.
    ws = np.ones(2000)                         # Uniform weights (Don't need to be normalized).
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9]) # Quantiles.

    return va.wgt_quantiles(xs, ws, qs)

@pt.fixture
def wqts2():
    """
        Take a vector, 'xs', and NON uniform weight vector, 'ws', along
        with a vector of quantiles, 'qs', return the quantile values from 'xs'.
    """
    np.random.seed(1)
    xs = np.linspace(0.0, 10.0, 2000)           # Data vector.
    ws = np.random.rand(2000)                   # Random weights (Don't need to be normalized).
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])  # Quantiles.

    return va.wgt_quantiles(xs, ws, qs)

@pt.fixture
def wqts_tensor1():
    """
        Each row of the 10x200 matrix 'X' is a vector of data (length 200).
        We wish to form the quantiles for each of these vector using uniform weights.
        A quantile vector, 'qs', is given and we return 
        a new quantile (values values of 'X') matrix, <Q>, of dimension: 10x5 (5 -- the number of quantile values).
    """
    np.random.seed(1)
    X = np.linspace(0.0, 10.0, 2000)
    X = X.reshape(10, 200)                     # <X> is a "tensor" (a matrix), where each row is a vector of data.
    ws = np.ones(200)                          # Uniform weights (Don't need to be normalized).
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9]) # The quantiles.
    return va.wgt_quantiles_tensor(X, ws, qs)

@pt.fixture
def wqts_tensor2():
    """
        Each row of the 10x200 matrix 'X' is a vector of data (length 200).
        We wish to form the quantiles for each of these vector using uniform weights.
        A quantile vector, 'qs', is given and we return 
        a new quantile (values values of 'X') matrix, <Q>, of dimension: 10x5 (5 -- the number of quantile values).
    """
    np.random.seed(1)
    X = np.linspace(0.0, 10.0, 2000)
    X = X.reshape(10, 200)                     # <X> is a "tensor" (a matrix), where each row is a vector of data.
    ws = np.random.rand(200)                   # Random weights (Don't need to be normalized)
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9]) # The quantiles.
    return va.wgt_quantiles_tensor(X, ws, qs)

# Create Correlation fixtures...


@pt.fixture
def cov1():
    """
        Create synthetic returns for securities in our universe.
        Then compute their covariance matrix.
    """
    # Synthetic security returns.
    np.random.seed(1)
    X = np.random.rand(5, 10)

    # Find the correlations between each security.
    return va.corr_cov(X, corr=False)

@pt.fixture
def corr1():
    """
        Create synthetic returns for securities in our universe.
        Then compute their correlation matrix.
    """
    # Synthetic security returns.
    np.random.seed(1)
    X = np.random.rand(5, 10)

    # Find the correlations between each security.
    return va.corr_cov(X)

@pt.fixture
def most_corr1():
    """
        Create synthetic returns for securities in our universe.
        Find the most correlated securities to 'labs' from the universe, 'ulabs'.
        Use a uniform weights.
        Return a pandas dataframe.
    """
    # Synthetic security returns.
    np.random.seed(1)
    X = np.random.rand(5, 10)

    # Security universe.
    ulabs = np.array(["IBM", "PFE", "C", "BAC", "GS"])

    # Securities of interest within the universe.
    labs  = np.array(["PFE", "GS"])

    # Dictionary mapping security names to row indices in X.
    lab_dict = { 'IBM' : 0, 'PFE' : 1, 'C' : 2, 'BAC' : 3, 'GS' : 4 }

    # Find the most correlated security to each of the 
    # securities in <labs> within our universe, <ulabs>.
    return va.most_corr_vec(X, labs, ulabs, lab_dict)


@pt.fixture
def most_corr2():
    """
        Create synthetic returns for securities in our universe.
        Find the most correlated securities to 'labs' from the universe, 'ulabs'.
        Use a NON uniform weights.
        Return a pandas dataframe.
    """
    # Synthetic security returns.
    np.random.seed(1)
    X = np.random.rand(5, 10)

    # Security universe.
    ulabs = np.array(["IBM", "PFE", "C", "BAC", "GS"])

    # Securities of interest within the universe.
    labs  = np.array(["PFE", "GS"])

    # Dictionary mapping security names to row indices in X.
    lab_dict = { 'IBM' : 0, 'PFE' : 1, 'C' : 2, 'BAC' : 3, 'GS' : 4 }

    ws=np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])

    # Find the most correlated security to each of the 
    # securities in <labs> within our universe, <ulabs>.
    return va.most_corr_vec(X, labs, ulabs, lab_dict, ws=ws)


# TESTS BEGIN

# Quantile tests...
def test_wqts1(wqts1):
    gold_val = np.array([0.99049525, 2.49124562, 4.99249625, 7.49874937, 8.99949975])

    assert wqts1.shape == gold_val.shape
    assert np.allclose(wqts1, gold_val) 


def test_wqts2(wqts2):
    gold_val = np.array([1.03051526, 2.50625313, 5.09754877, 7.5937969 , 9.01950975])

    assert wqts2.shape == gold_val.shape
    assert np.allclose(wqts2, gold_val) 


def test_wqts_tensor1(wqts_tensor1):
    gold_val = np.array([ [0.09004502, 0.24012006, 0.49024512, 0.74037019, 0.89044522],
                          [1.09054527, 1.24062031, 1.49074537, 1.74087044, 1.89094547],
                          [2.09104552, 2.24112056, 2.49124562, 2.74137069, 2.89144572],
                          [3.09154577, 3.24162081, 3.49174587, 3.74187094, 3.89194597],
                          [4.09204602, 4.24212106, 4.49224612, 4.74237119, 4.89244622],
                          [5.09254627, 5.24262131, 5.49274637, 5.74287144, 5.89294647],
                          [6.09304652, 6.24312156, 6.49324662, 6.74337169, 6.89344672],
                          [7.09354677, 7.24362181, 7.49374687, 7.74387194, 7.89394697],
                          [8.09404702, 8.24412206, 8.49424712, 8.74437219, 8.89444722],
                          [9.09454727, 9.24462231, 9.49474737, 9.74487244, 9.89494747] ])

    assert wqts_tensor1.shape == gold_val.shape
    assert np.allclose(wqts_tensor1, gold_val)


def test_wqts_tensor2(wqts_tensor2):
    gold_val = np.array([ [0.11005503, 0.26013007, 0.49024512, 0.75037519, 0.90045023],
                          [1.11055528, 1.26063032, 1.49074537, 1.75087544, 1.90095048],
                          [2.11105553, 2.26113057, 2.49124562, 2.75137569, 2.90145073],
                          [3.11155578, 3.26163082, 3.49174587, 3.75187594, 3.90195098],
                          [4.11205603, 4.26213107, 4.49224612, 4.75237619, 4.90245123],
                          [5.11255628, 5.26263132, 5.49274637, 5.75287644, 5.90295148],
                          [6.11305653, 6.26313157, 6.49324662, 6.75337669, 6.90345173],
                          [7.11355678, 7.26363182, 7.49374687, 7.75387694, 7.90395198],
                          [8.11405703, 8.26413207, 8.49424712, 8.75437719, 8.90445223],
                          [9.11455728, 9.26463232, 9.49474737, 9.75487744, 9.90495248] ])

    assert np.allclose(wqts_tensor2, gold_val)


# Correlation tests...

def test_corr1(corr1):
    gold_val = np.array([ [1.       ,  0.2061907 ,  0.30361534, -0.33590049,  0.20879493],
                        [ 0.2061907 ,  1.        ,  0.15950074, -0.12891314,  0.60635175],
                        [ 0.30361534,  0.15950074,  1.        , -0.22498689,  0.04237877],
                        [-0.33590049, -0.12891314, -0.22498689,  1.        , -0.3822043 ],
                        [ 0.20879493,  0.60635175,  0.04237877, -0.3822043 ,  1.        ] ])
    assert corr1.shape == gold_val.shape
    assert np.allclose(corr1, gold_val)


def test_cov1(cov1):
    gold_val = np.array([ [ 0.04735338,  0.01242557,  0.02476038, -0.02287041,  0.01502366],
                          [ 0.01242557,  0.07669083,  0.0165536 , -0.01117009,  0.05552347],
                          [ 0.02476038,  0.0165536 ,  0.1404482 , -0.02638172,  0.00525154],
                          [-0.02287041, -0.01117009, -0.02638172,  0.09789838, -0.03954245],
                          [ 0.01502366,  0.05552347,  0.00525154, -0.03954245,  0.10933534] ])
    assert cov1.shape == gold_val.shape
    assert np.allclose(cov1, gold_val)


def test_most_corr1(most_corr1):
    """
        Test "most" correlated securities for 'lab' securities using securities from a a larger universe.
    """
    gold_val = pd.DataFrame({'lab' : ['PFE', 'GS'], 'best_correlate' : ['GS', 'PFE'], 'best_corr' : [0.606352, 0.606352], 'valid_cnt': [1, 1]})
    assert (gold_val['lab']             == most_corr1['lab']           ).all()
    assert (gold_val['best_correlate']  == most_corr1['best_correlate']).all()
    assert (gold_val['valid_cnt']       == most_corr1['valid_cnt']     ).all()
    assert np.allclose(gold_val['best_corr'], most_corr1['best_corr']) 
    
def test_most_corr2(most_corr2):
    """
        Test "most" correlated securities for 'lab' securities using securities from a a larger universe.
        Use a NON uniform set of weights.
    """
    gold_val = pd.DataFrame({'lab' : ['PFE', 'GS'], 'best_correlate' : ['GS', 'PFE'], 'best_corr' : [0.607007, 0.607007], 'valid_cnt': [1, 1]})
    assert (gold_val['lab']             == most_corr2['lab']           ).all()
    assert (gold_val['best_correlate']  == most_corr2['best_correlate']).all()
    assert (gold_val['valid_cnt']       == most_corr2['valid_cnt']     ).all()
    assert np.allclose(gold_val['best_corr'], most_corr2['best_corr']) 


# --- Helper function tests ---

def test_get_worst_corr():
    assert va.get_worst_corr(va.CorrType.MOST)  == -np.inf
    assert va.get_worst_corr(va.CorrType.LEAST) ==  np.inf
    assert va.get_worst_corr(va.CorrType.HIGH)  ==  0.0
    assert va.get_worst_corr(va.CorrType.LOW)   ==  np.inf


def test_get_best_corr_idx_most():
    corr = np.array([[ 0.0,  0.8, -0.3],
                     [ 0.8,  0.0,  0.5],
                     [-0.3,  0.5,  0.0]])
    ind = np.arange(3)
    # Set self-correlation to worst.
    corr[ind, ind] = -np.inf
    idx = va.get_best_corr_idx(corr, ind, va.CorrType.MOST)
    assert list(idx) == [1, 0, 1]  # row0->col1(0.8), row1->col0(0.8), row2->col1(0.5)


def test_get_best_corr_idx_least():
    corr = np.array([[ 0.0,  0.8, -0.3],
                     [ 0.8,  0.0,  0.5],
                     [-0.3,  0.5,  0.0]])
    ind = np.arange(3)
    corr[ind, ind] = np.inf
    idx = va.get_best_corr_idx(corr, ind, va.CorrType.LEAST)
    assert list(idx) == [2, 2, 0]  # row0->col2(-0.3), row1->col2(0.5), row2->col0(-0.3)


def test_get_best_corr_idx_high():
    corr = np.array([[ 0.0,  0.2, -0.9],
                     [ 0.2,  0.0,  0.5],
                     [-0.9,  0.5,  0.0]])
    ind = np.arange(3)
    corr[ind, ind] = 0.0
    idx = va.get_best_corr_idx(corr, ind, va.CorrType.HIGH)
    assert list(idx) == [2, 2, 0]  # highest |corr|: row0->col2(0.9), row1->col2(0.5), row2->col0(0.9)


def test_get_best_corr_idx_low():
    corr = np.array([[ 0.0,  0.8, -0.1],
                     [ 0.8,  0.0,  0.5],
                     [-0.1,  0.5,  0.0]])
    ind = np.arange(3)
    corr[ind, ind] = np.inf
    idx = va.get_best_corr_idx(corr, ind, va.CorrType.LOW)
    assert list(idx) == [2, 2, 0]  # lowest |corr|: row0->col2(0.1), row1->col2(0.5), row2->col0(0.1)


def test_get_best_corr_idxs_most():
    corr = np.array([[ 0.0,  0.8,  0.3, -0.5],
                     [ 0.8,  0.0,  0.5,  0.2],
                     [ 0.3,  0.5,  0.0,  0.9],
                     [-0.5,  0.2,  0.9,  0.0]])
    ind = np.arange(4)
    corr[ind, ind] = -np.inf
    idxs = va.get_best_corr_idxs(corr, ind, va.CorrType.MOST, 2)
    assert idxs.shape == (4, 2)
    assert list(idxs[0]) == [1, 2]  # row0 top-2: col1(0.8), col2(0.3)
    assert list(idxs[2]) == [3, 1]  # row2 top-2: col3(0.9), col1(0.5)


def test_get_best_corr_idxs_least():
    corr = np.array([[ 0.0,  0.8,  0.3, -0.5],
                     [ 0.8,  0.0,  0.5,  0.2],
                     [ 0.3,  0.5,  0.0,  0.9],
                     [-0.5,  0.2,  0.9,  0.0]])
    ind = np.arange(4)
    corr[ind, ind] = np.inf
    idxs = va.get_best_corr_idxs(corr, ind, va.CorrType.LEAST, 2)
    assert idxs.shape == (4, 2)
    assert list(idxs[0]) == [3, 2]  # row0 bottom-2: col3(-0.5), col2(0.3)


# --- most_corr_vec with different CorrTypes ---

@pt.fixture
def corr_data():
    np.random.seed(1)
    X = np.random.rand(5, 10)
    ulabs    = np.array(["IBM", "PFE", "C", "BAC", "GS"])
    labs     = np.array(["PFE", "GS"])
    lab_dict = {'IBM': 0, 'PFE': 1, 'C': 2, 'BAC': 3, 'GS': 4}
    return X, labs, ulabs, lab_dict


def test_most_corr_vec_least(corr_data):
    X, labs, ulabs, lab_dict = corr_data
    df = va.most_corr_vec(X, labs, ulabs, lab_dict, corr_type=va.CorrType.LEAST)
    assert len(df) == 2
    # LEAST should pick the smallest (most negative) correlation.
    for val in df['best_corr']:
        assert np.isfinite(val)


def test_most_corr_vec_high(corr_data):
    X, labs, ulabs, lab_dict = corr_data
    df = va.most_corr_vec(X, labs, ulabs, lab_dict, corr_type=va.CorrType.HIGH)
    assert len(df) == 2
    for val in df['best_corr']:
        assert np.isfinite(val)


def test_most_corr_vec_low(corr_data):
    X, labs, ulabs, lab_dict = corr_data
    df = va.most_corr_vec(X, labs, ulabs, lab_dict, corr_type=va.CorrType.LOW)
    assert len(df) == 2
    for val in df['best_corr']:
        assert np.isfinite(val)


# --- most_corr_vec with exclude_labs ---

def test_most_corr_vec_exclude(corr_data):
    X, labs, ulabs, lab_dict = corr_data
    # Exclude GS -- PFE can no longer pick GS as best.
    exclude = np.array(["GS"])
    df = va.most_corr_vec(X, labs, ulabs, lab_dict, exclude_labs=exclude)
    # PFE's best should NOT be GS.
    pfe_row = df[df['lab'] == 'PFE'].iloc[0]
    assert pfe_row['best_correlate'] != 'GS'


# --- most_corr_vecs (top-k) ---

def test_most_corr_vecs_uniform(corr_data):
    X, labs, ulabs, lab_dict = corr_data
    df = va.most_corr_vecs(X, labs, ulabs, lab_dict, k=2)
    assert len(df) == 2
    # Each row should have k=2 correlates.
    for correlates in df['best_correlates']:
        assert len(correlates) == 2
    for corrs in df['best_corrs']:
        assert len(corrs) == 2


def test_most_corr_vecs_weighted(corr_data):
    X, labs, ulabs, lab_dict = corr_data
    ws = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    df = va.most_corr_vecs(X, labs, ulabs, lab_dict, k=3, ws=ws)
    assert len(df) == 2
    for correlates in df['best_correlates']:
        assert len(correlates) == 3


def test_most_corr_vecs_exclude(corr_data):
    X, labs, ulabs, lab_dict = corr_data
    exclude = np.array(["IBM", "C"])
    df = va.most_corr_vecs(X, labs, ulabs, lab_dict, k=2, exclude_labs=exclude)
    assert len(df) == 2
    # Excluded labels should not appear in results.
    for correlates in df['best_correlates']:
        assert 'IBM' not in correlates
        assert 'C'   not in correlates


def test_most_corr_vecs_least(corr_data):
    X, labs, ulabs, lab_dict = corr_data
    df = va.most_corr_vecs(X, labs, ulabs, lab_dict, k=2, corr_type=va.CorrType.LEAST)
    assert len(df) == 2
    # Correlations should be in ascending order (least first).
    for corrs in df['best_corrs']:
        assert corrs[0] <= corrs[1]


# --- Input contract tests ---

def test_wgt_quantiles_contract_bad_vs():
    with pt.raises(ValueError):
        va.wgt_quantiles([1, 2, 3], np.ones(3), np.array([0.5]), chk_con=True)


def test_wgt_quantiles_contract_bad_qs():
    with pt.raises(ValueError):
        va.wgt_quantiles(np.array([1., 2., 3.]), np.ones(3), np.array([1.5]), chk_con=True)


def test_wgt_quantiles_contract_length_mismatch():
    with pt.raises(ValueError):
        va.wgt_quantiles(np.array([1., 2., 3.]), np.ones(4), np.array([0.5]), chk_con=True)


def test_wgt_quantiles_contract_negative_weights():
    with pt.raises(ValueError):
        va.wgt_quantiles(np.array([1., 2., 3.]), np.array([-1., 1., 1.]), np.array([0.5]), chk_con=True)


def test_wgt_quantiles_contract_zero_weight_sum():
    with pt.raises(ValueError):
        va.wgt_quantiles(np.array([1., 2., 3.]), np.zeros(3), np.array([0.5]), chk_con=True)


def test_wgt_quantiles_tensor_contract_bad_vs():
    with pt.raises(ValueError):
        va.wgt_quantiles_tensor(np.array([1., 2., 3.]), np.ones(3), np.array([0.5]), chk_con=True)


def test_corr_cov_contract_bad_X():
    with pt.raises(ValueError):
        va.corr_cov(np.array([1., 2., 3.]), chk_con=True)


def test_corr_cov_contract_bad_eps():
    np.random.seed(1)
    X = np.random.rand(3, 5)
    with pt.raises(ValueError):
        va.corr_cov(X, eps=-1.0, chk_con=True)


def test_corr_cov_contract_bad_ws():
    np.random.seed(1)
    X = np.random.rand(3, 5)
    with pt.raises(ValueError):
        va.corr_cov(X, ws=np.array([-1., 1., 1., 1., 1.]), chk_con=True)


def test_corr_cov_contract_ws_length_mismatch():
    np.random.seed(1)
    X = np.random.rand(3, 5)
    with pt.raises(ValueError):
        va.corr_cov(X, ws=np.ones(3), chk_con=True)


# --- Weighted correlation tests ---

def test_corr_weighted():
    np.random.seed(1)
    X  = np.random.rand(5, 10)
    ws = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    corr = va.corr_cov(X, ws=ws)
    assert corr.shape == (5, 5)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)


# Run the tests...
if __name__ == "__main__":
    pt.main()
