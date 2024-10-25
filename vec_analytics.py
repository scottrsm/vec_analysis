import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from enum import Enum
import input_contract as ic


class CorrType(Enum):
    """
    Enumeration type determining what "best" correlation means.
    """
    LEAST = 1
    MOST  = 2
    LOW   = 3
    HIGH  = 4


def get_worst_corr(corr_type):
    """
    Return the "worst" value for a given correlation enumeration type.
    """
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


def get_best_corr_idx(corr, ind, corr_type):
    """
    Get the "top" correlation position (column index) for 
    a given row index of a correlation matrix.
    """
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


def get_best_corr_idxs(corr, ind, corr_type, k):
    """
    Get the "top" correlation positions (column indices) for 
    a given row index of a correlation matrix.
    """
    # Get the top correlates positions.
    idxs = None
    if corr_type == CorrType.MOST:
        idxs = np.flip(np.argsort(corr[ind, :], axis=1)[:, -k:], axis=1)
    elif corr_type == CorrType.LEAST:
        idxs = np.argsort(corr[ind, :], axis=1)[:, :k]
    elif corr_type == CorrType.HIGH:
        idxs = np.flip(np.argsort(np.abs(corr[ind, :]), axis=1)[:, -k:], axis=1)
    elif corr_type == CorrType.LOW:
        idxs = np.argsort(np.abs(corr[ind, :]), axis=1)[:, :k]
    else:
        raise ValueError(corr_type, "Unexpected CorrType.")

    return idxs


def wgt_quantiles(vs :np.ndarray, 
                  ws:np.ndarray, 
                  qs :np.ndarray ) -> np.ndarray:
    """
    Get a numpy array consisting of an array of quantile weighted <vs> values.
    
    Parameters
    ----------
    vs    A numpy(np) (N) array of numeric values. 
    ws    A numpy(np) (N) array of numeric weights. (Weights need only be non-negative, they need not sum to 1.)
    qs    A numpy(np) (D) array of numeric values.  (Meant to be quantiles -- numbers in the range [0, 1]).

    Returns
    -------
    A numpy array consisting of the quantile weighted values of <vs> using weights, <ws>, for each quantile in <qs>.
  
    Return-Type
    -----------
    A numpy(np) (D) array of weighted quantile <vs> values with the same length as <qs>.
  
    Packages
    --------
    numpy(np)

    Parameter Contract
    -----------------
    1. vs, ws, qs are all numpy arrays.
    2. qs in [0.0, 1.0]
    3. |vs| == |ws|
    4. all(ws) >= 0
    5. sum(ws) > 0
    
    Return
    ------
    A numpy array of length D.

    Throws
    ------
    ValueError

    Assumptions
    -----------
    1. <vs>, <ws>, and <qs> are all numeric arrays.
    """

    # Check input parameter contract.
    ic.chk_wgt_quantiles_contract(vs, ws, qs)

    # Sort the <vs> array and the associated weights.
    # Turn the weights into proper weights and create a cumulative weight array.
    idx  = np.argsort(vs)
    ovs  = vs[idx]
    ows  = ws[idx]
    ows  = ows / np.sum(ows) # Normalize the weights.
    cws  = np.cumsum(ows)
  
    N    = np.size(cws)
    M    = np.size(qs)
  
    # Reshape to broadcast.
    cws.shape = (N, 1)

    # Need to copy to broadcast quantiles (don't change inputs).
    qss = qs.copy()
    qss.shape  = (1, M)
  
    # Use broadcasting to get all comparisons of <cws> with each entry from <qs>.  
    # Form tensor (cws <= qss) * 1 and sandwich index of the value vectors with 0 and 1.
    A   = np.concatenate([np.ones(M).reshape(1,M), (cws <= qss) * 1, np.zeros(M).reshape(1,M)], axis=0)
  
    # Get the diff -- -1 will indicate the boundary where cws > qs.
    X   = np.diff(A, axis=0).astype(int)
  
    # Get the indices of the boundary.
    idx = np.maximum(0, np.where(X == -1)[0] - 1)
  
    # Return the weighted quantile value of <vs> against each quantile, <qs>.
    return ovs[idx]


def wgt_quantiles_tensor(VS :np.ndarray, 
                         ws:np.ndarray, 
                         qs :np.ndarray ) -> np.ndarray:
    """ 
    Compute a (D, M) numpy array consisting of the quantile weighted values of <VS> using weights, <ws>, for each quantile in <qs>.
    
    Parameters
    ----------
    VS    A numpy(np) (D, N) matrix of numeric values. 
    ws    A numpy(np) (N) array of numeric weights. (Weights need only be non-negative, they need not sum to 1.)
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
    1. VS, ws, and qs are numpy arrays.
    2. VS is a numpy matrix.
    3. qs in [0.0, 1.0]
    4. |VS[0]| == |ws|
    5. all(ws) >= 0
    6. sum(ws) > 0
    
    Assumptions
    -----------
    1. <VS>, <ws>, and <qs> are all numeric arrays.

    """
  
    # Check input parameter contract.
    ic.chk_wgt_quantiles_tensor_contract(VS, ws, qs)

    D, N  = VS.shape
    M     = qs.size

    # Get the sorted index array for each of the value vectors in <VS>.
    idx = np.argsort(VS, axis=1)
  
    # Apply this index back to <VS> to get sorted values.
    OVS = np.take_along_axis(VS, idx, axis=1)
  
    # Apply the index to the weights, where, the dimension of <ows> and <cws> expands to: (D, N).
    ows = ws[idx]
    wss = np.sum(ows, axis=1) # sorted weight row sums.
    ows /= wss[:, np.newaxis] # Normalize the sorted weights.
    cws = np.cumsum(ows, axis=1) # Get the cumulated ordered weight sum.

    # Reshape to broadcast. Copy <qs> so as not to modify an input.
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
  
    # Return the values in the value vectors that correspond to these indices,
    # the M quantiles for each of the D value vectors.
    # A (D, M) matrix.
    return np.take_along_axis(OVS, idx, axis=1)



def corr(X           : np.ndarray                 , 
         eps         : float = 1.0e-6             ,
         ws          : Optional[np.ndarray] = None, 
         chk_contract:bool = True                  ) -> np.ndarray:
    """!
        Find the correlation between M vectors of length N, represented as the MxN matrix, <X>.

        Parameters
        ----------
        X           : A MxN numeric matrix representing M vectors of length N.
        eps         : A float value. The sum of the weights should be larger than this value.
        ws          : (Optional) A N numeric vector of weights of non-negative values.
        chk_contract: A boolean, defaults to True meaning, check the input parameter contract.

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

    # Optionally check input contract for <X>.
    if chk_contract:
        if type(X) != np.ndarray:
            raise ValueError("corr: The parameter, X, is not a numpy array.")

        if len(X.shape) != 2:
            raise ValueError("corr: Parameter, X, is not a matrix.")

        if type(eps) == type(0.0) and eps <= 0.0:
            raise ValueError("corr: Parameter, eps, is not a positive number.")


    # Get shape of <X>.
    M, N = X.shape

    # Optionally check input contract for <ws>.
    if chk_contract:
        if type(ws) != type(None):
            if type(ws) != np.ndarray:
                raise ValueError("corr: The parameter, ws, is not a numpy array.")

            if len(ws.shape) != 1:
                raise ValueError("corr: Parameter, ws, is not a 1-d numpy array.")

            if N != len(ws):
                raise ValueError("corr: Parameter, X, and, ws, are not compatible.")

            if np.any(ws < 0):
                raise ValueError("corr: Parameter, ws, has some negative elements.")

            if np.sum(ws) < eps:
                raise ValueError("corr: Parameter, ws, has cumulative sum that is less than eps({eps}).")

    # If not given set ws to its default setting -- uniform weights.
    if type(ws) is type(None):
        ws = np.ones(N)

    # As we need to change the shape of thw weight vector, we make a copy first.
    wss = ws.copy()

    # Normalize the weights.
    wss /= np.sum(wss)

    # Subtract off the mean of each row.
    # Note: Need <wss> to be normalized in order to use np.sum to compute the mean.
    # We need to "reshape" the mean, <mn>, to do this -- so that "broadcasting" works.
    mn       = np.sum(X * wss, axis=1)
    mn.shape = (M, 1)
    X        = X - mn

    # Copy <X> and now do a reshape of each so that the "rules of broadcasting" will give us
    # all combinations of <X> * <Y>.
    Y = X.copy()
    X.shape   = (1, M, N)
    Y.shape   = (M, 1, N)
    wss.shape = (1, 1, N)

    # Now use aggregation to sum up the third index -- the values -- to get 
    # an MxM correlation matrix.
    corr = np.sum(X * Y * wss, axis=2) / np.sqrt( np.sum(X * X * wss, axis=2) * np.sum(Y * Y * wss, axis=2) )

    # Set NaNs to 0.
    corr[np.isnan(corr)] = 0.0

    # Return <X> to its original shape.
    X.shape = (M, N)

    # Return the (weighted) correlation matrix.
    return corr



def most_corr_vec(X           : np.ndarray                 ,
                  labs        : np.ndarray                 , 
                  ulabs       : np.ndarray                 , 
                  lab_dict    : Dict[Any, int]             , 
                  corr_type   : CorrType = CorrType.MOST   ,
                  eps         : float = 1.0e-6             ,
                  ws          : Optional[np.ndarray] = None,
                  exclude_labs: Optional[np.ndarray] = None, 
                  chk_contract: bool = True                 ) -> pd.DataFrame:
    """!
        For each vector in a list, <labs>, determine the "most" (weighted) correlated 
        vector from a larger universe of <M> names, <ulabs>, using the matrix of 
        series data, <X>, an MxN matrix. Correlation may be weighted; one may 
        also exclude some vectors from the larger universe.
        NOTE: Weights, <ws>, will be normalized by this function.

        Parameters
        ---------
        X           : A MxN matrix of M vectors, each of length N.
        labs        : An H vector of the names of the vectors of interest.
        ulabs       : The names of the larger universe of vectors.
        lab_dict    : A dictionary mapping vector labels into the row index of <X>.
                      At a minimum, the keys must include <labs>.
        corr_type   : An element from class CorrType, default is CorrType.MOST.
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
        A Pandas DataFrame with schema: lab(vector label), best_correlate(vector label), best_corr(their correlation)

        Throws
        ------
        ValueError
     """

    # Optionally check input contract.
    if chk_contract:
        ic.check_most_corr_vec_input_contract(X, labs, ulabs, lab_dict, eps, ws, exclude_labs)

    # We reshape <X> for later computations.
    M, N = X.shape
    H    = len(labs)

    # If not given, set <ws> to its default setting -- uniform weights.
    if type(ws) == type(None):
        ws = np.ones(N)

    # Since we are reshaping the weight vector, copy it.
    wss = ws.copy()

    # Normalize the weights.
    wss /= np.sum(wss)

    # Subtract off row means. 
    # Note: np.sum only works to compute mean if <wss> is normalized.
    mn       = np.sum(X * wss, axis=1) # Vector means.
    mn.shape = (M, 1)                  # Expand for broadcasting.
    X        = X - mn                  # Subtract off row means.

    # Get the index of each security of interest.
    idx = np.array([lab_dict[lab] for lab in labs])

    # If <exclude_labs> is not None, find their row indices in the correlation matrix.
    eidx = None
    if type(exclude_labs) != type(None):
        eidx = np.array([lab_dict[lab] for lab in exclude_labs])

    # Get the correlation of our chosen vectors against the full universe 
    # -- giving an HxM matrix (H the length of labs).
    Y         = X.copy()    # Make a copy of <X> -- full universe.
    X1        = X[idx,:]    # Get only the H <labs> vectors.
    X1.shape  = (H, 1, N)   # Expand <X1> for broadcasting, use only the <lab> vectors.
    Y.shape   = (1, M, N)   # Expand <Y> for broadcasting and use the full set of vectors.
    wss.shape = (1, 1, N)   # Expand <wss> for broadcasting.
 
    # Find the "worst" correlation value.
    worst_corr_val = get_worst_corr(corr_type)

    # Compute correlation matrix of the <labs> vectors against the universe -- <ulabs> vectors.
    # Set self correlation to -infinity for only the <labs> rows of the correlation matrix.
    # These operations aggregate the third index.
    corr = np.sum(X1 * Y * wss, axis=2) / np.sqrt(np.sum(X1 * X1 * wss, axis=2) * np.sum(Y * Y * wss, axis=2))

    # Set NaNs to 0.
    corr[np.isnan(corr)] = 0.0

    # Form HxM correlation matrix, <corr>.
    ind = np.arange(len(labs))
    corr[ind, idx] = worst_corr_val  # Array slicing -- fill "diagonal" with worst corr val
                                     # -- effectively eliminating themselves as "best" correlate.

    # If <exclude_labs> is not None, set correlations of these vector with <labs> 
    # vectors to worst correlation -- to exclude them from consideration.
    if type(eidx) != type(None):
        corr[np.ix_(ind, eidx)] = worst_corr_val

    # Get the top correlate position and value.
    bidx = get_best_corr_idx(corr, ind, corr_type)
    val = corr[ind, bidx]

    # Return a Pandas Dataframe consisting of <labs>; 
    # the most correlated vectors(their labels); and their correlation with <labs>.
    return pd.DataFrame({'lab' : labs, 'best_correlate': ulabs[bidx], 'best_corr' : val})



def most_corr_vecs(X           : np.ndarray                 ,
                   labs        : np.ndarray                 , 
                   ulabs       : np.ndarray                 , 
                   lab_dict    : Dict[Any, int]             , 
                   k           : int                        , 
                   corr_type   : CorrType = CorrType.MOST   ,
                   eps         : float = 1.0e-6             ,
                   ws          : Optional[np.ndarray] = None,
                   exclude_labs: Optional[np.ndarray] = None,
                   chk_contract: bool = True                 ) -> pd.DataFrame:
    """!
        Determine the "most" correlated k vectors from a larger universe for each vector in a smaller subset.

        Parameters
        ---------
        X           : A MxN np.ndarray matrix of M vectors, each of length N.
        labs        : An H np.ndarray vector of the labels named of the vectors of interest.
        ulabs       : The larger universe of vectors.
        lab_dict    : A dictionary mapping vector labels into the row index of <X>.
                      At a minimum, the keys must include all values in <labs>.
        k           : Positive integer, the number of top correlates to retrieve.
        corr_type   : An element from class CorrType, default is CorrType.MOST.
        eps         : A positive float used as a minimum cumulative weight threshold.
        ws          : (Optional) An np.ndarray umeric weight vector of length N of non-negative values.
        exclude_labs: (Optional) An np.ndarray of labels in the larger universe, ulabs, to exclude in the correlation analysis.
        chk_contract: A boolean, defaults to True, meaning; check the input parameter contract.

        Packages
        --------
        numpy(np)
        pandas(pd)

        Return
        ------
        A Pandas Dataframe of length H with schema: lab(vector label), top_correlates(vector label), top_corrs(their correlation)
        Note: The order of the top_correlates and top_corrs is from "best" to "worse" correlated where what is "best" is 
              determined by <corr_type>.

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

    # Copy weights as we will reshape them.
    wss = ws.copy()

    # Normalize the weights.
    wss /= np.sum(wss)

    # Subtract off row means.
    # Note: np.sum only works to compute mean if <wss> is normalized.
    mn       = np.sum(X * wss, axis=1) # Vector means.
    mn.shape = (M, 1)                 # Expand for broadcasting.
    X        = X - mn

    # Get the index of each security of interest.
    idx = np.array([lab_dict[lab] for lab in labs])

    # If exclude_labs is not None, find their row indices in the correlation matrix.
    eidx = None
    if type(exclude_labs) != type(None):
        eidx = np.array([lab_dict[lab] for lab in exclude_labs])

    # Get the correlation of our chosen vectors against the full universe 
    # -- giving an HxM matrix (H the length of labs)
    Y         = X.copy()    # Full universe.
    X1        = X[idx,:]    # Get only the <labs> vectors.
    X1.shape  = (H, 1, N)   # Expand <X1> for broadcasting.
    Y.shape   = (1, M, N)   # Expand <Y> for broadcasting and use the full set of vectors.
    wss.shape = (1, 1, N)   # Expand <wss> for broadcasting.

    # Find the "worst" correlation value.
    worst_corr_val = get_worst_corr(corr_type)
    
    # Compute correlation matrix H(length of labs) x M(all vectors in the universe). 
    ind = np.arange(len(labs))
    corr           = np.sum(X1 * Y * wss, axis=2) / np.sqrt(np.sum(X1 * X1 * wss, axis=2) * np.sum(Y * Y * wss, axis=2))  
    corr[ind, idx] = worst_corr_val  # Array slicing -- fill "diagonal" with worst corr val.
                                     # -- effectively eliminating themselves as their "best" correlate.
    # Set NaNs to 0.
    corr[np.isnan(corr)] = 0.0

    # If <exclude_labs> is not None, set correlations will all these <labs> 
    # vectors to "worst" correlation to exclude them from consideration.
    if type(eidx) != type(None):
        corr[np.ix_(ind, eidx)] = worst_corr_val

    # Get the top <k> correlate positions and values.
    idxs = get_best_corr_idxs(corr, ind, corr_type, k)
    vals = [corr[i, idxs[i]] for i in range(len(labs))] 

    # Return a DataFrame of vector labels; the most correlated vectors(their labels); 
    # and their correlations.
    return pd.DataFrame({'lab' : labs, 'best_correlates' : ulabs[idxs].tolist(), 'best_corrs': vals })


