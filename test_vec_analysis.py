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


# Run the tests...
if __name__ == "__main__":
    pt.main()
