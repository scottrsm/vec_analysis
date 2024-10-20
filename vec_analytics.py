import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from enum import Enum
import input_contract as ic


class CorrType(Enum):
    LEAST = 1
    UN    = 2
    MOST  = 3

def get_worst_corr(corr_type):
    val = None
    if corr_type == CorrType.MOST:
        val = -np.inf
    elif corr_type == CorrType.LEAST:
        val = np.inf
    elif corr_type == CorrType.HIGH:
        val = 0.0
    elif corr_type == CorrType.LOW:
        val = np.inf
    else:
        raise ValueError(corr_type, "Unexpected CorrType.")

    return val

def get_best_corr_idx(corr, ind):
    # Get the top correlate position.
    idx = None
    if corr_type == CorrType.MOST:
        idx = np.argmax(corr[ind, :], axis=1) 
    elif corr_type == CorrType.LEAST:
        idx = np.argmin(corr[ind, :], axis=1) 
    elif corr_type == CorrType.HIGH:
        idx = np.argmax(np.abs(corr[ind, :]), axis=1) 
    elif corr_type == CorrType.LOW:
        idx = np.argmin(np.abs(corr[ind, :]), axis=1) 
    else:
        raise ValueError(corr_type, "Unexpected CorrType.")

    return idx

def get_best_corr_idxs(corr, ind):
    # Get the top correlate positions.
    idxs = None
    if corr_type == CorrType.MOST:
        idxs = np.argsort(corr[ind, :], axis=1)[:, -k:] 
    elif corr_type == CorrType.LEAST:
        idxs = np.argsort(corr[ind, :], axis=1)[:, :k] 
    elif corr_type == CorrType.HIGH:
        idxs = np.argsort(np.abs(corr[ind, :]), axis=1)[:, -k:] 
    elif corr_type == CorrType.LOW:
        idxs = np.argsort(np.abs(corr[ind, :]), axis=1)[:, :k] 
    else:
        raise ValueError(corr_type, "Unexpected CorrType.")

    return idxs

def wgt_quantiles(vs :np.ndarray, 
                  wts:np.ndarray, 
                  qs :np.ndarray ) -> np.ndarray:
    '''
    Get a numpy array consisting of an array of quantile weighted <vs> values.
    
    Parameters
    ----------
    vs    A numpy(np) (N) array of numeric values. 
    wts   A numpy(np) (N) array of numeric weights. (Weights need only be non-negative, they need not sum to 1.)
    qs    A numpy(np) (D) array of numeric values.  (Meant to be quantiles -- numbers in the range [0, 1]).

    Returns
    -------
    :A numpy array consisting of the quantile weighted values of <vs> using weights, <wts>, for each quantile in <qs>.
  
    Return-Type
    -----------
    A numpy(np) (D) array of weighted quantile <vs> values with the same length as <qs>.
  
    Packages
    --------
    numpy(np)

    Parameter Contract
    -----------------
    1. vs, wts, qs are all numpy arrays.
    2. qs in [0.0, 1.0]
    3. |vs| == |wts|
    4. all(wts) >= 0
    5. sum(wts) > 0
    
    Return
    ------
    A numpy array of length D.

    Throws
    ------
    ValueError

    Assumptions
    -----------
    1. <vs>, <wts>, and <qs> are all numeric arrays.
    '''

    # Check input parameter contract.
    ic.chk_wgt_quantiles_contract(vs, wts, qs)

    # Sort the vs array and the associated weights.
    # Turn the weights into proper weights and create a cumulative weight array.
    idx  = np.argsort(vs)
    ovs  = vs[idx]
    ows  = wts[idx]
    ows  = ows / np.sum(ows) # Normalize the weights.
    cws  = np.cumsum(ows)
  
    N    = np.size(cws)
    M    = np.size(qs)
  
    # Reshape to broadcast.
    cws.shape = (N, 1)
    qss = qs.copy()
    qss.shape  = (1, M)
  
    # Use broadcasting to get all comparisons of <cws> with each entry from <qs>.  
    # Form tensor (cws <= qss) * 1 and sandwich index of the value vectors with 0 and 1.
    A   = np.concatenate([np.ones(M).reshape(1,M), (cws <= qss) * 1, np.zeros(M).reshape(1,M)], axis=0)
  
    # Get the diff -- -1 will indicate the boundary where cws > qs.
    X   = np.diff(A, axis=0).astype(int)
  
    # Get the indices of the boundary.
    idx = np.maximum(0, np.where(X == -1)[0] - 1)
  
    # Return the weighted quantile value of <vs> against each <qs>.
    return(ovs[idx])


def wgt_quantiles_tensor(VS :np.ndarray, 
                         wts:np.ndarray, 
                         qs :np.ndarray ) -> np.ndarray:
    '''
    Compute a (D, M) numpy array consisting of the quantile weighted values of <VS> using weights, <wts>, for each quantile in <qs>.
    
    Parameters
    ----------
    VS    A numpy(np) (D, N) matrix of numeric values. 
    wgts  A numpy(np) (N) array of numeric weights. (Weights need only be non-negative, they need not sum to 1.)
    qs    A numpy(np) (M) array of numeric values.  (Meant to be quantiles -- numbers in the range [0, 1]).
  
    Returns
    -------
    A (D, M) numpy array of numeric values.
  
    Throws
    ------
    ValueError
  
    Packages
    --------
    numpy(np)
  
    Parameter Contract
    -----------------
    1. VS, and wts are numpy arrays.
    2. VS is a numpy matrix.
    3. qs in [0.0, 1.0]
    4. |VS[0]| == |wts|
    5. all(wts) >= 0
    6. sum(wts) > 0
    
    Assumptions
    -----------
    1. <VS>, <wts>, and <qs> are all numeric arrays.

    '''
  
    # Check input parameter contract.
    ic.chk_wgt_quantiles_tensor_contract(VS, wts, qs)

    # Normalize the weights.
    ws  = wts / np.sum(wts)
  
    D, N  = VS.shape
    M     = qs.size

    # Get the sorted index array for each of the value vectors in VS.
    idx = np.argsort(VS, axis=1)
  
    # Apply this index back to VS to get sorted values.
    OVS = np.take_along_axis(VS, idx, axis=1)
  
    # Apply the index to the weights, where, the dimension of ws (and cws) expands to: (D, N).
    ows = ws[idx]
    cws = np.cumsum(ows, axis=1)

    # Reshape to broadcast.
    cws.shape = (D, N, 1)
    qss = qs.copy()
    qss.shape  = (1, 1, M)

    # Use broadcasting to get all comparisons of <cws> with each entry from <qs>. 
    # Form tensor (cws <= qss) * 1 and sandwich index of the value vectors with 0 and 1.
    A = np.concatenate([np.ones(M*D).reshape(D,1,M), (cws <= qss) * 1, np.zeros(M*D).reshape(D,1,M)], axis=1)
  
    # Compute the index difference on the value vectors.
    Delta = np.diff(A, axis=1).astype(int)

    # Get the index of the values, this leaves, essentially, a (D, M) matrix. Reshape it as such.
    idx = np.maximum(0, np.where(Delta == -1)[1] - 1)
    idx = idx.reshape(D, M) 
  
    # Return the values in the value vectors that correspond to these indices -- the M quantiles for each of the D value vectors.
    # A (D, M) matrix.
    return(np.take_along_axis(OVS, idx, axis=1))



def corr(X           : np.ndarray                 , 
         eps         : float = 1.0e-6             ,
         chk_contract:bool = True                 , 
         ws          : Optional[np.ndarray] = None ) -> np.ndarray:
    """!
        Find the correlation between M vectors of length N, represented as the MxN matrix, X.

        Parameters
        ----------
        X           : A MxN numeric matrix representing M vectors of length N.
        eps         : A float value. The sum of the weights should be larger than this value.
        chk_contract: A boolean, defaults to True meaning, check the input parameter contract.
        ws          : (Optional) A N numeric vector of weights of non-negative values.

        Packages
        --------
        numpy(np)

        Return
        ------
        A MxM correlation matrix of the M vectors.

        Throws
        ------
        ValueError
    """

    if chk_contract:
        if len(X.shape) != 2:
            raise ValueError("corr: Parameter, X, is not a matrix.")
        if ws:
            if len(ws.shape) != 1:
                raise ValueError("corr: Parameter, ws, is not a 1-d numpy array.")

    # X is an MxN array -- meaning M vectors each of length N.
    M, N = X.shape

    # Check that ws and X are compatible.
    if ws:
        if N != len(ws):
            raise ValueError("corr: Parameter, X, and, ws, are not compatible.")

        if np.any(ws < 0):
            raise ValueError("corr: Parameter, ws, has some negative elements.")

        if np.sum(ws) < eps:
            raise ValueError("corr: Parameter, ws, has cumulative sum that is less than eps({eps}).")

    # If not given set ws to its default setting -- uniform weights.
    if not ws:
        ws = np.ones(N)

    # Reshape weights for computation.
    ws.shape = (1, 1, N)

    # Subtract off the mean of each row.
    # We need to "reshape" the mean, <mn>, to do this -- so that "broadcasting" works.
    mn = np.mean(X, axis=1)
    mn.shape=(M, 1)
    X = X - mn

    # Copy X and now do a reshape of each so that the "rules of broadcasting" will give us
    # all combinations of <X> * <Y>.
    Y = X.copy()
    X.shape = (1, M, N)
    Y.shape = (M, 1, N)

    # Now use aggregation to sum up the third index -- the values -- to get 
    # an MxM matrix of cross-correlations.
    return( np.sum(X * Y * ws, axis=2) / np.sqrt(np.sum(X * X * ws, axis=2) * np.sum(Y * Y * ws, axis=2)) )


def most_corr(X           : np.ndarray                 , 
              labs        : np.ndarray                 , 
              eps         : float = 1.0e-6             ,
              corr_type   : CorrType = CorrType.MOST   ,
              chk_contract: bool = True                , 
              ws          : Optional[np.ndarray] = None ) -> pd.DataFrame:
    """!
        Given, X, a collection of M vectors of length N, with labels, labs, find the most/least correlated vector for each.
        NOTE: Weights, <ws>, will be normalized by this function.

        Parameters
        ----------
        X           : The MxN matrix representing the M vectors values.
        labs        : The labels of each of the M vectors of <X>.
        eps         : A positive float used as a minimum cumulative weight threshold.
        corr_type   : One of four types of correlation to maximize: LEAST, MOST, LOW, HIGH.
                      For a given vector, we interpret what each of these possibilities 
                      mean when finding an associated vector that "maximizes" correlation.
                        If MOST : Finds the most correlated vector. Best correlation value  :  1.0 .
                        If LEAST: Finds the least correlated vector. Best correlation value : -1.0 .
                        If LOW  : Finds the most uncorrelated vector. Best correlation value:  0.0 .
                           Smallest in terms of absolute value. 
                        If HIGH : Finds the highest correlate regardless of sign. Best correlation values: [-1.0, 1.0]. 
                           Largest in terms of absolute value. 
        chk_contract: A boolean, defaults to True; meaning, check the input parameter contract.
        ws          : (Optional) A N numeric vector of weights of non-negative values.
                                 Defaults to uniform weights.

        Packages
        --------
        numpy(np)
        pandas(pd)

        Return
        ------
        A Pandas DataFrame with schema: lab(vector label), max_corr(vector label with max correlation), corr(their correlation) 

        Throws
        ------
        ValueError
    """
    # Check input contract?
    if chk_contract:
        ic.check_most_corr_input_contract(x, eps, ws)

    # Get M and N: M vectors of length N.
    M, N = X.shape

    # If not given set <ws> to its default setting -- uniform weights.
    if not ws:
        ws = np.ones(N)

    # Reshape weights for computation.
    ws.shape = (1, 1, N)

    # Subtract off row means.
    # Copy <x> to <y>, reshape both for computation.
    mn = np.mean(x, axis=1)
    mn.shape=(M, 1)
    X = X - mn
    Y = X.copy()
    X.shape = (1, M, N)
    Y.shape = (M, 1, N)

    # Compute correlation.
    corr = np.sum(X * Y * ws, axis=2) / ( np.sqrt(np.sum(X * X * ws, axis=2) * np.sum(Y * Y * ws, axis=2)) ) 
    ind = np.arange(M)

    # For each vector, find the associated vector with highest/lowest correlation.
    if anti:
        corr[ind, ind] = np.inf  # Set self correlation so as not to be selected.
        idx = np.argmin(np.abs(corr), axis=1) 
        val = corr[ind, idx]
    else:
        corr[ind, ind] = -np.inf # Set self correlation so as not to be selected.
        idx = np.argmax(corr, axis=1) 
        val = corr[ind, idx]

    # Return a DataFrame of vector labels; the most correlated vector -- by label; and their correlation.
    # Set correlation type label.
    ct_lab = "max_corr"
    if corr_type == CorrType.LEAST:
        ct_lab = "min_corr"
    elif corr_type == CorrType.UN:
        ct_lab = "un_corr"


    return(pd.DataFrame({'lab' : labs, ct_lab: labs[idx], 'corr' : val}))


def most_corr_vec(X           : np.ndarray                 ,
                  labs        : np.ndarray                 , 
                  ulabs       : np.ndarray                 , 
                  lab_dict    : Dict[Any, int]             , 
                  eps         : float = 1.0e-6             ,
                  ws          : Optional[np.ndarray] = None,
                  exclude_labs: Optional[np.ndarray] = None, 
                  chk_contract: bool = True                 ) -> pd.DataFrame:
    """!
        For each vector in a list, determine the most correlated vector from a larger universe.
        Correlation may be weighted; one may also exclude some vectors from the larger universe.
        NOTE: Weights, <ws>, will be normalized by this function.

        Parameters
        ---------
        X           : A MxN matrix of M vectors, each of length N.
        labs        : An H vector of the names of the vectors of interest.
        ulabs       : The names of the larger universe of vectors.
        lab_dict    : A dictionary mapping vector labels into the row index of <X>.
                      At a minimum, the keys must include <labs>.
        eps         : A positive float used as a minimum cumulative weight threshold.
        chk_contract: A boolean, defaults to True meaning, check the input parameter contract.
        ws          : (Optional) A N numeric weight vector of non-negative values.
                                 Defaults to uniform.
        exclude_labs: (Optional) A list of labels in the larger universe, <ulabs>, to exclude in the correlation analysis.

        Packages
        --------
        numpy(np)
        pandas(pd)

        Return
        ------
        A Pandas DataFrame with schema: lab(vector label), max_correlate(vector label), max_corr(their correlation)

        Throws
        ------
        ValueError
     """

    # Check input contract?
    if chk_contract:
        ic.check_most_corr_vec_input_contract(X, labs, ulabs, lab_dict, eps, ws, exclude_labs)

    M, N = X.shape

    # If not given set <ws> to its default setting -- uniform weights.
    if not ws:
        ws = np.ones(N)

    # Normalize the weights.
    ws /= np.sum(ws)
    ws.shape = (1, N)

    # Reshape weights for computation.
    ws.shape = (1, 1, N)

    # Subtract off row means.
    mn       = np.mean(X * ws, axis=1) # Vector means.
    mn.shape = (M, 1)                  # broadcasting
    X        = X - mn
    Y        = X.copy()

    # Get the index of each security of interest.
    idx = [lab_dict[lab] for lab in labs]

    # If <exclude_labs> is not None, find their row indices in the correlation matrix.
    eidx = None
    if exclude_labs:
        eidx = [lab_dict[lab] for lab in exclude_labs]

    # Get the correlation of our chosen vectors against the full universe 
    # -- giving an HxM matrix (H the length of labs).
    X.shape  = (M, 1, N)   # expand for broadcasting.
    X        = X[idx,:,:]  # Get only the H <labs> vectors.
    Y.shape  = (1, M, N)   # Expand for broadcasting and use the full set of vectors.
    ws.shape = (1, 1, N)   # Expand ws for broadcasting.
 
    # Find the "worst" correlation value.
    worst_corr_val = get_worst_corr(corr_type)

    # Compute correlation matrix of the <labs> vectors against the universe -- <ulabs> vectors.
    # Set self correlation to -infinity for only the <labs> rows of the correlation matrix.
    corr = np.sum(X * Y * ws, axis=2) / ( np.sqrt(np.sum(X * X * ws, axis=2) * np.sum(Y * Y * ws, axis=2)) )  # Aggregation of third index.
    corr[idx, idx] = worst_corr_val  # Array slicing -- fill "diagonal" with worst corr val
                                     # -- effectively eliminating themselves as "best" correlate.

    # If <exclude_labs> is not None, set correlations will all these <labs> 
    # vectors to worst correlation to exclude them from consideration.
    if eidx:
        corr[np.ix_(idx, eidx)] = worst_corr_val

    # Get the top correlate position and value.
    bidx = get_best_corr_idx(corr, idx)

    val = corr[idx, bidx]

    # Return a Dataframe of vector labels; the most correlated vector(its label); and their correlation.
    return(pd.DataFrame({'lab' : labs, 'best_correlate': ulabs[bidx], 'best_corr' : val}))


def most_corr_vecs(X           : np.ndarray                 ,
                   labs        : np.ndarray                 , 
                   ulabs       : np.ndarray                 , 
                   lab_dict    : Dict[Any, int]             , 
                   k           : int                        , 
                   eps         : float = 1.0e-6             ,
                   ws          : Optional[np.ndarray] = None,
                   exclude_labs: Optional[np.ndarray] = None,
                   chk_contract: bool = True                 ) -> pd.DataFrame:
    """!
        Determine the "most" correlated k vectors from a larger universe for each vector in a smaller subset.

        Parameters
        ---------
        X           : A MxN matrix of M vectors, each of length N.
        labs        : An H vector of the labels named of the vectors of interest.
        ulabs       : The larger universe of vectors.
        lab_dict    : A dictionary mapping vector labels into the row index of <X>.
                      At a minimum, the keys must include all values in <labs>.
        k           : Positive integer, the number of top correlates to retrieve.
        eps         : A positive float used as a minimum cumulative weight threshold.
        ws          : (Optional) A N numeric weight vector of non-negative values.
        exclude_labs: (Optional) A list of labels in the larger universe, ulabs, to exclude in the correlation analysis.
        chk_contract: A boolean, defaults to True meaning, check the input parameter contract.

        Packages
        --------
        numpy(np)
        pandas(pd)

        Return
        ------
        A Pandas Dataframe of length H with schema: lab(vector label), top_correlate(vector label), top_corr(their correlation)

        Throws
        ------
        ValueError

     """

    # Check input contract?
    if chk_contract:
        ic.check_most_corr_vecs_input_contract(X, labs, ulabs, lab_dict, k, eps, ws, exclude_labs)
        
    # Extract shape of <X> and of <labs>.
    M, N = X.shape
    H    = len(labs)

    # If not given set <ws> to its default setting -- uniform weights.
    if not ws:
        ws = np.ones(N)

    # Normalize the weights.
    ws /= np.sum(ws)
    ws.shape = (1, N)

    # Subtract off row means.
    mn       = np.mean(X * ws, axis=1) # Vector means.
    mn.shape = (M, 1)                  # Broadcasting.
    X        = X - mn
    Y        = X.copy()

    # Get the index of each security of interest.
    idx = [lab_dict[lab] for lab in labs]

    # If exclude_labs is not None, find their row indices in the correlation matrix.
    eidx = None
    if exclude_labs:
        eidx = [lab_dict[lab] for lab in exclude_labs]

    # Get the correlation of our chosen vectors against the full universe 
    # -- giving an HxM matrix (H the length of labs)
    X        = X[idx,:]  # Get only the <labs> vectors.
    X.shape  = (H, 1, N)   # Expand for broadcasting.
    Y.shape  = (1, M, N)   # Expand for broadcasting and use the full set of vectors.
    ws.shape = (1, 1, N)   # Expand ws for broadcasting.

    # Find the "worst" correlation value.
    worst_corr_val = get_worst_corr(corr_type)
    
    # Compute correlation matrix H(length of labs) x M(all vectors in the universe). 
    corr           = np.sum(X * Y * ws, axis=2) / ( np.sqrt(np.sum(X * X * ws, axis=2) * np.sum(Y * Y * ws, axis=2)) ) 
    corr[idx, idx] = worst_corr_val  # Array slicing -- fill "diagonal" with worst corr val.
                                     # -- effectively eliminating themselves as their "best" correlate.

    # If <exclude_labs> is not None, set correlations will all these <labs> 
    # vectors to "worst" correlation to exclude them from consideration.
    if eidx:
        corr[np.ix_(idx, eidx)] = worst_corr_val

    # Get the top <k> correlate positions and values.
    idxs = get_best_corr_idxs(corr, idx)
    vals = corr[idx, idxs]

    # Return a DataFrame of vector labels; the most correlated vector(its label); 
    # and their correlation.
    return(pd.DataFrame({'lab' : labs, 'best_correlates' : ulabs[idxs].tolist(), 'best_corrs': vals.tolist() }))


