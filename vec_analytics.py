import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from enum import Enum
import input_contract as ic


class CorrType(Enum):
    """
    Enumeration type determining what "best" correlation means.
    LEAST -- Means the smallest correlation.
    MOST  -- Means the largest correlation.
    LOW   -- Means the smallest correlation in absolute value.
    HIGH  -- Means the largest corelation in absolute value.
    """
    LEAST = 1
    MOST  = 2
    LOW   = 3
    HIGH  = 4


def get_worst_corr(corr_type:CorrType) -> float:
    """
    Return the "worst" possible value for a given correlation enumeration type.
    """
    match corr_type:
        case CorrType.MOST:  # Return -infinity
            val = -np.inf
        case CorrType.LEAST: # Return  infinity
            val = np.inf
        case CorrType.HIGH:  # Return  zero
            val = 0.0 
        case CorrType.LOW:   # Return  infinity
            val = np.inf
        case _:
            raise ValueError(corr_type, "<corr_type> is not of type CorrType.")

    return val


def get_best_corr_idx(corr:np.ndarray   , 
                      ind :int          , 
                      corr_type:CorrType) -> int:
    """
    Get the "most" correlated position (column index) for 
    a given row index of a correlation matrix.
    """
    match corr_type:
        case CorrType.MOST:
            idx = np.argmax(corr[ind, :], axis=1) 
        case CorrType.LEAST:
            idx = np.argmin(corr[ind, :], axis=1) 
        case CorrType.HIGH:
            idx = np.argmax(np.abs(corr[ind, :]), axis=1) 
        case CorrType.LEAST:
            idx = np.argmin(np.abs(corr[ind, :]), axis=1) 
        case _:
            raise ValueError(corr_type, "Value <corr_type> is not a CorrType.")

    return idx


def get_best_corr_idxs(corr:np.ndarray   , 
                       ind :int          , 
                       corr_type:CorrType, 
                       k:int              ) -> np.ndarray:
    """
    Get the "top" correlation positions (column indices) for 
    a given row index of a correlation matrix.
    """
    # Get the "most" correlated positions.
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


def wgt_quantiles(vs :np.ndarray    , 
                  ws :np.ndarray    , 
                  qs :np.ndarray    ,
                  chk_con:bool=False ) -> np.ndarray:
    """
    Get a numpy array consisting of an array of quantile weighted <vs> values.
    
    Arguments:
    ----------
    vs    A numpy(np) (N) array of numeric values. 
    ws    A numpy(np) (N) array of numeric weights. (Weights need only be non-negative, they need not sum to 1.)
    qs    A numpy(np) (D) array of numeric values.  (Meant to be quantiles -- numbers in the range [0, 1]).

    Keyword Arguments:
    chk_con (Optional) If True, check input contract -- see below.

    Return
    ------
    A numpy(np) (D) array of weighted quantile <vs> values with the same length as <qs>.
  
    Packages
    --------
    numpy(np)
    input_contract(ic)

    Input Contract:
    -----------------
    1. vs, ws, qs are all numpy arrays.
    2. qs in [0.0, 1.0]
    3. |vs| == |ws|
    4. all(ws) >= 0
    5. sum(ws) > 0
    
    Throws
    ------
    ValueError
    """

    # Check input contract.
    if chk_con:
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


def wgt_quantiles_tensor(VS     :np.ndarray , 
                         ws     :np.ndarray , 
                         qs     :np.ndarray ,
                         chk_con:bool=False  ) -> np.ndarray:
    """ 
    Compute a (D, M) numpy array consisting of the quantile weighted values of <VS> using weights, <ws>, for each quantile in <qs>.
    
    Arguments:
    ----------
    VS    A numpy(np) (D, N) matrix of numeric values. 
    ws    A numpy(np) (N) array of numeric weights. (Weights need only be non-negative, they need not sum to 1.)
    qs    A numpy(np) (M) array of numeric values.  (Meant to be quantiles -- numbers in the range [0, 1]).

    Keyword Arguments:
    chk_con  (Optional) If True check the input contract -- see below.
  
    Returns:
    -------
    A (D, M) numpy array of numeric values.
  
    Throws:
    ------
    ValueError
  
    Packages:
    --------
    numpy(np)
    input_contract(ic)
  
    Input Contract:
    -----------------
    1. VS, ws, and qs are numpy arrays.
    2. VS is a numpy matrix.
    3. qs in [0.0, 1.0]
    4. |VS[0]| == |ws|
    5. all(ws) >= 0
    6. sum(ws) > 0
    
    """
  
    # Check input contract.
    if chk_con:
        ic.chk_wgt_quantiles_tensor_contract(VS, ws, qs)

    D, N  = VS.shape
    M     = qs.size

    # Get the sorted index array for each of the value vectors in <VS>.
    idx = np.argsort(VS, axis=1)
  
    # Apply this index back to <VS> to get sorted values.
    OVS = np.take_along_axis(VS, idx, axis=1)
  
    # Apply the index to the weights, where, the dimension of <ows> and <cws> expands to: (D, N).
    ows = ws[idx]
    wss = np.sum(ows, axis=1)    # sorted weight row sums.
    ows /= wss[:, np.newaxis]    # Normalize the sorted weights.
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



def corr_cov(X      : np.ndarray                 , 
             eps    : float = 1.0e-6             ,
             ws     : Optional[np.ndarray] = None, 
             corr   : bool=True                  ,
             chk_con: bool = False                ) -> np.ndarray:
    """!
        Find the correlation or empirical covariance between M vectors of length N, represented as the MxN matrix, <X>.

        Arguments:
        ----------
        X      : A MxN numeric matrix representing M vectors of length N.


        Keyword Arguments:
        eps    : (Optional) A float value. The sum of the weights should be larger than this value.
        ws     : (Optional) A N numeric vector of weights of non-negative values.
        corr   : (Optional) If True, compute the correlation; otherwise, compute empirical covariance.
        chk_con: (Optional) A boolean, defaults to Fals; meaning, do NOT check the input contract -- see below.

        Return
        ------
        A MxM correlation, or emprical covariance matrix of the M vectors.

        Input Contract:
        1. X is a 2-D numpy array.
        2. eps > 0.0
        3. ws is 1-D numpy array.
        4. |ws| = |X[0:]| 
        5. all(ws) >= 0.0
        6. sum(ws) >= eps

        Packages
        --------
        numpy(np)

        Throws
        ------
        ValueError
    """

    # Optionally check input contract for <X>.
    nm = "corr_cov"
    if chk_con:
        if type(X) != np.ndarray:
            raise ValueError(f"{nm}: The parameter, X, is not a numpy array.")

        if len(X.shape) != 2:
            raise ValueError(f"{nm}: Parameter, X, is not a matrix.")

        if type(eps) == type(0.0) and eps <= 0.0:
            raise ValueError("{nm}: Parameter, eps, is not a positive number.")


    # Get shape of <X>.
    M, N = X.shape

    # Optionally check input contract for <ws>.
    if chk_con:
        if type(ws) != type(None):
            if type(ws) != np.ndarray:
                raise ValueError(f"{nm}: The parameter, ws, is not a numpy array.")

            if len(ws.shape) != 1:
                raise ValueError(f"{nm}: Parameter, ws, is not a 1-d numpy array.")

            if N != len(ws):
                raise ValueError(f"{nm}: Parameter, X, and, ws, are not compatible.")

            if np.any(ws < 0):
                raise ValueError(f"{nm}: Parameter, ws, has some negative elements.")

            if np.sum(ws) < eps:
                raise ValueError(f"{nm}: Parameter, ws, has cumulative sum that is less than eps({eps}).")

    # If not given, set ws to its default setting -- uniform weights.
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
    # an MxM correlation/covariance matrix.
    if corr:
        cmat = np.sum(X * Y * wss, axis=2) / np.sqrt( np.sum(X * X * wss, axis=2) * np.sum(Y * Y * wss, axis=2) )
    else:
        cmat = np.sum(X * Y * wss, axis=2) / ( 1.0 - np.sum(wss * wss) )

    # Set NaNs to 0.
    # The reasoning: This value represents "least correlated".
    cmat[np.isnan(cmat)] = 0.0

    # Return <X> to its original shape.
    X.shape = (M, N)

    # Return the (weighted) correlation/covariance matrix.
    return cmat



def most_corr_vec(X           : np.ndarray                 ,
                  labs        : np.ndarray                 , 
                  ulabs       : np.ndarray                 , 
                  lab_dict    : Dict[Any, int]             , 
                  corr_type   : CorrType = CorrType.MOST   ,
                  eps         : float = 1.0e-6             ,
                  ws          : Optional[np.ndarray] = None,
                  exclude_labs: Optional[np.ndarray] = None, 
                  chk_con     : bool = False                 ) -> pd.DataFrame:
    """!
        For each vector in a list, <labs>, determine the "most" (weighted) correlated 
        vector from a larger universe of <M> names, <ulabs>, using the matrix of 
        series data, <X>, an MxN matrix. Correlation may be weighted; one may 
        also exclude some vectors from the larger universe.
        NOTE: Weights, <ws>, will be normalized by this function.

        Arrguments:
        ---------
        X           : A MxN matrix of M vectors, each of length N.
        labs        : An H vector of the names of the vectors of interest.
        ulabs       : The names of the larger universe of vectors.
        lab_dict    : A dictionary mapping vector labels into the row index of <X>.
                      At a minimum, the keys must include <labs>.


        Keyword Arguments:
        corr_type   : (Optional) An element from class CorrType, default is CorrType.MOST.
        eps         : (Optional) A positive float used as a minimum cumulative weight threshold.
        ws          : (Optional) A N numeric weight vector of non-negative values.
                                 Defaults to uniform weigths.
        exclude_labs: (Optional) A list of labels in the larger universe, <ulabs>, 
                                 to exclude in the correlation analysis.
        chk_con     : (Optional) A boolean, defaults to False meaning, do NOT check the input contract -- see below.

        Input Contract:


        Packages
        --------
        numpy(np)
        pandas(pd)
        input_contract(ic)

        Return
        ------
        A Pandas DataFrame with schema: 
            lab(vector label)                             , 
            best_correlate(vector label)                  , 
            best_corr(their correlation)                  , 
            valid_cnt(1 if correlate is valid, 0 otherwise  )

        Throws
        ------
        ValueError
     """

    # Optionally check input contract.
    if chk_con:
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
    mn       = np.sum(X * wss, axis=1) # Vector of row means.
    mn.shape = (M, 1)                  # Expand for broadcasting.
    X        = X - mn                  # Subtract off row means.

    # Get the row index of each <lab> in <X>.
    idx = np.array([lab_dict[lab] for lab in labs])

    # If <exclude_labs> is not None, find their row indices in <X>.
    # These will be used as column indices in the new HxM 
    # correlation matrix, <corr>, below.
    eidx = None
    if type(exclude_labs) != type(None):
        eidx = np.array([lab_dict[lab] for lab in exclude_labs])

    # Get the correlation of our chosen vectors against the full universe 
    # -- giving an HxM matrix (H the length of labs).
    Y         = X.copy()    # Make a copy of <X> -- full universe.
    X1        = X[idx,:]    # Get only the H <labs> universe.
    X1.shape  = (H, 1, N)   # Expand <X1>  for broadcasting, use only the <lab> universe.
    Y.shape   = (1, M, N)   # Expand <Y>   for broadcasting and use the full universe.
    wss.shape = (1, 1, N)   # Expand <wss> for broadcasting.
 
    # Find the "worst" correlation value.
    worst_corr_val = get_worst_corr(corr_type)

    # Compute correlation matrix (HxM) of the <labs> vectors against 
    # the universe -- <ulabs> vectors. These operations aggregate the third index.
    corr = np.sum(X1 * Y * wss, axis=2) / np.sqrt(np.sum(X1 * X1 * wss, axis=2) * np.sum(Y * Y * wss, axis=2))

    # Set NaNs to worst correlation value.
    corr[np.isnan(corr)] = worst_corr_val 

    # Set self correlation to "worst" value.
    # <ind> and <idx> are respectively the row and column indices of <labs>
    # in <corr>.
    ind = np.arange(H)
    corr[ind, idx] = worst_corr_val  # Array slicing -- fill "diagonal" with worst corr val
                                     # -- effectively eliminating themselves as "best" correlate.

    # If <exclude_labs> is not None, set correlations with <labs> 
    # vectors to "worst" correlation so as to exclude them from consideration.
    if type(eidx) != type(None):
        corr[np.ix_(ind, eidx)] = worst_corr_val # Create index "mesh"

    # For each vector in <labs>, get the top correlate, its index, and if 
    # the correlation is valid.
    bidx = get_best_corr_idx(corr, ind, corr_type)
    val  = corr[ind, bidx]
    cnt  = [np.sum(val[i] != worst_corr_val) for i in range(H)]

    # Return a Pandas Dataframe consisting of <labs>; 
    # the most correlated vectors(their index values); their correlation with <labs>;
    # and if the best correlate is valid; 1 for yes, 0 for no.
    return pd.DataFrame({'lab'           : labs       , 
                         'best_correlate': ulabs[bidx], 
                         'best_corr'     : val        , 
                         'valid_cnt'     : cnt}        )



def most_corr_vecs(X           : np.ndarray                 ,
                   labs        : np.ndarray                 , 
                   ulabs       : np.ndarray                 , 
                   lab_dict    : Dict[Any, int]             , 
                   k           : int                        , 
                   corr_type   : CorrType = CorrType.MOST   ,
                   eps         : float = 1.0e-6             ,
                   ws          : Optional[np.ndarray] = None,
                   exclude_labs: Optional[np.ndarray] = None,
                   chk_con     : bool = False                 ) -> pd.DataFrame:
    """!
        Determine the "most" correlated k vectors from a larger universe for 
        each vector in a smaller subset.

        Arguments:
        ---------
        X           : A MxN np.ndarray matrix of M vectors, each of length N.
        labs        : An H np.ndarray vector of the labels named of the vectors of interest.
        ulabs       : The larger universe of vectors.
        lab_dict    : A dictionary mapping vector labels into the row index of <X>.
                      At a minimum, the keys must include all values in <labs>.
        k           : Positive integer, the number of top correlates to retrieve.

        Keyword Arguments:
        corr_type   : (Optional) An element from class CorrType, default is CorrType.MOST.
        eps         : (Optional) A positive float used as a minimum cumulative weight threshold.
        ws          : (Optional) An np.ndarray umeric weight vector of length N of non-negative values.
        exclude_labs: (Optional) An np.ndarray of labels in the larger universe, ulabs, to exclude in the correlation analysis.
        chk_contract: (Optional) A boolean, defaults to False, meaning; 
                                check the input contract -- see the documentation for the function: ic.check_most_corr_vecs_input_contract.

        Return:
        ------
        A Pandas Dataframe of length H with schema: 
            lab(vector label)                      , 
            best_correlates(vector label)          , 
            best_corrs(their correlation)          , 
            valid_cnt(number of valid correlations)
        Note: The order of the best_correlates and best_corrs is from 
              "best" to "worst" correlated where what is "best" is 
              determined by <corr_type>.


        Packages
        --------
        numpy(np)
        pandas(pd)
        input_contract(ic)

        Throws
        ------
        ValueError
     """

    # Check input contract?
    if chk_con:
        ic.check_most_corr_vecs_input_contract(X, labs, ulabs, lab_dict, k, eps, ws, exclude_labs)
        
    # Extract shape of <X> and of <labs>.
    M, N = X.shape
    H    = len(labs)

    # If not given, set <ws> to its default setting -- uniform weights.
    if not ws:
        ws = np.ones(N)

    # Copy weights as we will reshape them.
    wss = ws.copy()

    # Normalize the weights.
    wss /= np.sum(wss)

    # Subtract off row means.
    # Note: np.sum only works to compute mean if <wss> is normalized.
    mn       = np.sum(X * wss, axis=1) # Vector of row means.
    mn.shape = (M, 1)                  # Expand for broadcasting.
    X        = X - mn                  # Subtract off row means.

    # Get the row index of each <lab> in <X>.
    idx = np.array([lab_dict[lab] for lab in labs])

    # If <exclude_labs> is not None, find their row indices in <X>.
    # These will be used as column indices in the new HxM 
    # correlation matrix, <corr>, below.
    eidx = None
    if type(exclude_labs) != type(None):
        eidx = np.array([lab_dict[lab] for lab in exclude_labs]) 

    # Get the correlation of our chosen vectors against the full universe 
    # -- giving an HxM matrix (H the length of labs)
    Y         = X.copy()    # Full universe.
    X1        = X[idx,:]    # Select just the <labs> from the universe.
    X1.shape  = (H, 1, N)   # Expand <X1>  for broadcasting.
    Y.shape   = (1, M, N)   # Expand <Y>   for broadcasting and use the full set of vectors.
    wss.shape = (1, 1, N)   # Expand <wss> for broadcasting.

    # Find the "worst" correlation value.
    worst_corr_val = get_worst_corr(corr_type)
    
    # Compute correlation matrix H(length of labs) x M(all vectors in the universe). 
    corr           = np.sum(X1 * Y * wss, axis=2) / np.sqrt(np.sum(X1 * X1 * wss, axis=2) * np.sum(Y * Y * wss, axis=2))  

    # Set self correlation to "worst" value.
    # <ind> and <idx> are respectively the row and column indices of <labs>
    # in <corr>.
    ind            = np.arange(H)
    corr[ind, idx] = worst_corr_val  # Array slicing -- fill "diagonal" with worst corr val.
                                     # -- effectively eliminating themselves as their "best" correlate.

    # Set NaNs to worst correlation value.
    # It is possible that we have vectors with "close" to zero element values,
    # there correlation might be NaN.
    corr[np.isnan(corr)] = worst_corr_val

    # If <exclude_labs> is not None, set correlations with <labs> 
    # vectors to "worst" correlation so as to exclude them from consideration.
    if type(eidx) != type(None):
        corr[np.ix_(ind, eidx)] = worst_corr_val   # Create an index "mesh"

    # For each vector in <labs>, get the top <k> correlate indexes, correlations, 
    # and count of valid correlates.
    idxs = get_best_corr_idxs(corr, ind, corr_type, k)
    vals = [corr[i, idxs[i]] for i in range(H)] 
    cnts = [np.sum(vals[i] != worst_corr_val) for i in range(H)]

    # Return a DataFrame of vector labels; the most correlated vectors(their labels); 
    # their correlations; and the count of valid correlates.
    return pd.DataFrame({'lab'             : labs                , 
                         'best_correlates' : ulabs[idxs].tolist(), 
                         'best_corrs'      : vals                , 
                         'valid_cnt'       : cnts }               )


