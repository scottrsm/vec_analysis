import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
from enum import Enum

# The contract checks live in a sibling module. Support both the package form
# (`import vec_analysis`) and running the modules directly from this directory.
try:
    from . import input_contract as ic
except ImportError:                      # pragma: no cover -- run as a top-level module.
    import input_contract as ic


class CorrType(Enum):
    """
    Enumeration type determining what "best" correlation means.
    LEAST -- Means the smallest correlation.
    MOST  -- Means the largest correlation.
    LOW   -- Means the smallest correlation in absolute value.
    HIGH  -- Means the largest correlation in absolute value.
    """
    LEAST = 1
    MOST  = 2
    LOW   = 3
    HIGH  = 4


def get_worst_corr(corr_type:CorrType) -> float:
    """
    Return the "worst" possible value for a given correlation enumeration type.
    Correlations that must never be selected (a vector with itself, excluded
    vectors, undefined correlations) are set to this value before the "best"
    correlate is chosen.
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
            raise ValueError(corr_type, "Unexpected CorrType.")

    return val


def get_best_corr_idx(corr:np.ndarray   , 
                      ind :Any          , 
                      corr_type:CorrType) -> np.ndarray:
    """
    Get the "most" correlated position (column index) for 
    the given row index (or array of row indices) of a correlation matrix.
    """
    rows = corr[ind]
    match corr_type:
        case CorrType.MOST:
            idx = np.argmax(rows, axis=-1)
        case CorrType.LEAST:
            idx = np.argmin(rows, axis=-1)
        case CorrType.HIGH:
            idx = np.argmax(np.abs(rows), axis=-1)
        case CorrType.LOW:
            idx = np.argmin(np.abs(rows), axis=-1)
        case _:
            raise ValueError(corr_type, "Unexpected CorrType.")

    return idx


def get_best_corr_idxs(corr:np.ndarray   , 
                       ind :Any          , 
                       corr_type:CorrType, 
                       k:int              ) -> np.ndarray:
    """
    Get the "top" <k> correlation positions (column indices), best first, for 
    the given row index (or array of row indices) of a correlation matrix.

    Throws
    ------
    ValueError if <k> is not in the range [1, number of columns].
    """
    rows = corr[ind]
    ncols = rows.shape[-1]
    if not isinstance(k, (int, np.integer)) or k < 1 or k > ncols:
        raise ValueError(f"get_best_corr_idxs: k({k}) must be an integer in the range [1, {ncols}].")

    match corr_type:
        case CorrType.MOST:
            idxs = np.flip(np.argsort(rows, axis=-1)[..., -k:], axis=-1)
        case CorrType.LEAST:
            idxs = np.argsort(rows, axis=-1)[..., :k]
        case CorrType.HIGH:
            idxs = np.flip(np.argsort(np.abs(rows), axis=-1)[..., -k:], axis=-1)
        case CorrType.LOW:
            idxs = np.argsort(np.abs(rows), axis=-1)[..., :k]
        case _:
            raise ValueError(corr_type, "Unexpected CorrType.")

    return idxs


def _selection_key(corr: np.ndarray, invalid: np.ndarray, corr_type: CorrType) -> Tuple[np.ndarray, bool]:
    """
    The key on which "best" correlates are chosen, with invalid entries set to a
    value that can never win (-inf when the largest key wins, +inf when the smallest
    wins), and whether the largest key wins. Unlike the sentinel from
    <get_worst_corr>, this cannot tie with a genuine correlation (e.g. 0 for HIGH).
    """
    match corr_type:
        case CorrType.MOST:
            key, largest = corr.copy(), True
        case CorrType.LEAST:
            key, largest = corr.copy(), False
        case CorrType.HIGH:
            key, largest = np.abs(corr), True
        case CorrType.LOW:
            key, largest = np.abs(corr), False
        case _:
            raise ValueError(corr_type, "Unexpected CorrType.")
    key[invalid] = -np.inf if largest else np.inf
    return key, largest


def _best_positions(corr: np.ndarray, invalid: np.ndarray, corr_type: CorrType, k: Optional[int] = None) -> np.ndarray:
    """
    Column positions of the best (or best <k>, best first) correlates of every row,
    never choosing an invalid entry while a valid one exists.
    """
    key, largest = _selection_key(corr, invalid, corr_type)
    if k is None:
        return np.argmax(key, axis=1) if largest else np.argmin(key, axis=1)
    order = np.argsort(key, axis=1, kind="stable")
    return np.flip(order[:, -k:], axis=1) if largest else order[:, :k]


def _normalized_weights(ws: Optional[np.ndarray], N: int) -> np.ndarray:
    """
    Return a new float array of the weights <ws> normalized to sum to 1
    (uniform weights if <ws> is None). The input is never modified.
    """
    if ws is None:
        return np.full(N, 1.0 / N)
    wss = np.asarray(ws, dtype=float)
    return wss / np.sum(wss)


def _sorted_quantiles(qs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (sorted copy of <qs>, permutation that restores the original order).
    """
    qs   = np.asarray(qs, dtype=float)
    perm = np.argsort(qs, kind="stable")
    inv  = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    return qs[perm], inv


def _quantile_positions(cws: np.ndarray, qs_sorted: np.ndarray) -> np.ndarray:
    """
    For a 1-d cumulative (normalized) weight array <cws>, return, for each quantile,
    the index of the largest position whose cumulative weight is <= q; position 0
    if there is none. The last cumulative weight is taken to be exactly 1, so
    q = 1 selects the largest value regardless of rounding.
    """
    cws = cws.copy()
    cws[-1] = 1.0
    return np.maximum(0, np.searchsorted(cws, qs_sorted, side="right") - 1)


def wgt_quantiles(vs :np.ndarray    , 
                  ws :np.ndarray    , 
                  qs :np.ndarray    ,
                  chk_con:bool=False ) -> np.ndarray:
    """
    Get a numpy array consisting of quantile weighted <vs> values.
    By this we mean: For a given quantile value <q> in <qs>, find the 
    largest value in <vs> with the property that 
    the sum of the (normalized) weights of the values up to and including it is <= q.
    If no value satisfies this (q is below the weight of the smallest value), the smallest value is returned.
    Example: vs = [1,4,5,7] if 0.5 is in <qs>, and the weights are evenly distributed,
             ws =  [0.25, 0.25, 0.25, 0.25], then the weighted quantile (a weighted median)
             becomes 4. This is *NOT* what the usual definition of median is, it would be (4 + 5) / 2.
    NOTE: This is one rank below the usual "inverse CDF" quantile: with the weights above,
          q = 0.999 gives 5, not 7 (7 is returned for q = 1).

    Arguments:
    ----------
    vs    A numpy(np) (N) array of numeric values. 
    ws    A numpy(np) (N) array of numeric weights. (Weights need only be non-negative, they need not sum to 1.)
    qs    A numpy(np) (D) array of numeric values.  (Meant to be quantiles -- numbers in the range [0, 1].)
          They may be in any order; the result is in the same order as <qs>.

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
    1. vs, ws, qs are all 1-d numpy arrays.
    2. qs in [0.0, 1.0].
    3. |vs| == |ws| > 0
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
    idx  = np.argsort(vs, kind="stable")
    ovs  = vs[idx]
    ows  = _normalized_weights(ws, np.size(vs))[idx]
    cws  = np.cumsum(ows)

    # Work with sorted quantiles, then restore the caller's order.
    qs_sorted, inv = _sorted_quantiles(qs)
    pos = _quantile_positions(cws, qs_sorted)

    # Return the weighted quantile value of <vs> against each quantile, <qs>.
    return ovs[pos][inv]


def wgt_quantiles_tensor(VS     :np.ndarray , 
                         ws     :np.ndarray , 
                         qs     :np.ndarray ,
                         chk_con:bool=False  ) -> np.ndarray:
    """ 
    Compute a (D, M) numpy array consisting of the quantile weighted values of <VS> using weights, <ws>, for each quantile in <qs>.
    See the documentation for the function <wgt_quantiles>.
    
    Arguments:
    ----------
    VS    A numpy(np) (D, N) matrix of numeric values. 
    ws    A numpy(np) (N) array of numeric weights. (Weights need only be non-negative, they need not sum to 1.)
    qs    A numpy(np) (M) array of numeric values.  (Meant to be quantiles -- numbers in the range [0, 1].)
          They may be in any order; the columns of the result are in the same order as <qs>.

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
    1. VS is a 2-d numpy array(matrix).
    2. ws, qs are 1-d numpy arrays.
    3. qs in [0.0, 1.0].
    4. |VS[0]| == |ws| > 0
    5. all(ws) >= 0
    6. sum(ws) > 0
    
    """
  
    # Check input contract.
    if chk_con:
        ic.chk_wgt_quantiles_tensor_contract(VS, ws, qs)

    D, N  = VS.shape

    # Get the sorted index array for each of the value vectors in <VS>.
    idx = np.argsort(VS, axis=1, kind="stable")
  
    # Apply this index back to <VS> to get sorted values.
    OVS = np.take_along_axis(VS, idx, axis=1)
  
    # Apply the index to the (normalized) weights: (D, N) sorted weights and their cumulative sums.
    wss = _normalized_weights(ws, N)
    cws = np.cumsum(wss[idx], axis=1)

    # Work with sorted quantiles, then restore the caller's order.
    qs_sorted, inv = _sorted_quantiles(qs)
    M   = qs_sorted.size
    pos = np.empty((D, M), dtype=int)
    for d in range(D):
        pos[d] = _quantile_positions(cws[d], qs_sorted)
  
    # Return the values in the value vectors that correspond to these indices,
    # the M quantiles for each of the D value vectors: a (D, M) matrix.
    return np.take_along_axis(OVS, pos, axis=1)[:, inv]


def _weighted_corr(Xc: np.ndarray, Yc: np.ndarray, wss: np.ndarray) -> np.ndarray:
    """
    Weighted correlation of the rows of <Xc> (H x N) with the rows of <Yc> (M x N),
    both already centred with the normalized weights <wss> (N). Returns an (H, M)
    matrix computed with matrix products, so memory is O(HM), not O(HMN).
    Undefined entries (a zero-variance row) are NaN.
    """
    num  = (Xc * wss) @ Yc.T
    varx = np.sum(Xc * Xc * wss, axis=1)
    vary = np.sum(Yc * Yc * wss, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / np.sqrt(np.outer(varx, vary))


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
        chk_con: (Optional) A boolean, defaults to False; meaning, do NOT check the input contract -- see below.

        Return
        ------
        A MxM correlation, or empirical covariance matrix of the M vectors.
        Correlations that are undefined (a vector with zero variance) are set to 0.

        Input Contract:
        1. X is a 2-D numpy array.
        2. ws is 1-D numpy array.
        3. eps > 0.0
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

    # Optionally check input contract.
    if chk_con:
        ic.chk_corr_cov_contract(X, eps, ws)

    # Get shape of <X>.
    M, N = X.shape

    # Normalized weights (uniform if not given); the input is not modified.
    wss = _normalized_weights(ws, N)

    # Subtract off the (weighted) mean of each row.
    mn = np.sum(X * wss, axis=1)
    Xc = X - mn[:, np.newaxis]

    if corr:
        cmat = _weighted_corr(Xc, Xc, wss)
    else:
        # Unbiased weighted covariance (reliability weights).
        cmat = ((Xc * wss) @ Xc.T) / (1.0 - np.sum(wss * wss))

    # Set NaNs to 0.
    # The reasoning: This value represents "least correlated".
    cmat[np.isnan(cmat)] = 0.0

    # Return the (weighted) correlation/covariance matrix.
    return cmat


def _row_positions(labs: np.ndarray, ulabs: np.ndarray, lab_dict: Dict[Any, int], nm: str) -> np.ndarray:
    """
    Row indices in <X> of the labels <labs>, via <lab_dict>.
    <lab_dict> must agree with <ulabs> (lab_dict[ulabs[i]] == i), otherwise the
    labels reported for correlates would be wrong; this is always checked.
    """
    for i, lab in enumerate(ulabs):
        if lab_dict.get(lab, None) != i:
            raise ValueError(f"{nm}: lab_dict[{lab!r}] is {lab_dict.get(lab, None)}, but {lab!r} is row {i} of X (ulabs[{i}]).")
    try:
        return np.array([lab_dict[lab] for lab in labs], dtype=int)
    except KeyError as e:
        raise ValueError(f"{nm}: The label {e.args[0]!r} is not a key of lab_dict.") from None


def _best_corr_setup(X           : np.ndarray,
                     labs        : np.ndarray,
                     ulabs       : np.ndarray,
                     lab_dict    : Dict[Any, int],
                     corr_type   : CorrType,
                     ws          : Optional[np.ndarray],
                     exclude_labs: Optional[np.ndarray],
                     nm          : str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Shared work of <most_corr_vec> and <most_corr_vecs>: the (H x M) weighted correlation
    matrix of the <labs> vectors against the universe with self correlations, excluded
    vectors and undefined correlations set to the "worst" value, the boolean (H x M) mask
    of those invalid entries, the row indices of <labs>, and the worst value.
    """
    M, N = X.shape
    H    = len(labs)

    # Normalized weights (uniform if not given); the input is not modified.
    wss = _normalized_weights(ws, N)

    # Subtract off (weighted) row means.
    mn = np.sum(X * wss, axis=1)
    Xc = X - mn[:, np.newaxis]

    # Row indices of <labs> (and of <exclude_labs>) in <X>.
    idx  = _row_positions(labs, ulabs, lab_dict, nm)
    eidx = None if exclude_labs is None else _row_positions(exclude_labs, ulabs, lab_dict, nm)

    # Correlation of the <labs> vectors against the full universe: an (H x M) matrix.
    corr = _weighted_corr(Xc[idx], Xc, wss)

    # Entries that must never be chosen: undefined correlations, self correlations, excluded vectors.
    invalid = np.isnan(corr)
    ind = np.arange(H)
    invalid[ind, idx] = True
    if eidx is not None:
        invalid[np.ix_(ind, eidx)] = True

    worst_corr_val = get_worst_corr(corr_type)
    corr[invalid]  = worst_corr_val

    return corr, invalid, idx, worst_corr_val


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
        NOTE: Weights, <ws>, are normalized by this function (the input array is not modified).

        Arguments:
        ---------
        X           : A MxN matrix of M vectors, each of length N.
        labs        : An H vector of the names of the vectors of interest.
        ulabs       : The names of the larger universe of vectors: ulabs[i] is the name of row i of <X>.
        lab_dict    : A dictionary mapping vector labels into the row index of <X>.
                      It must agree with <ulabs>: lab_dict[ulabs[i]] == i (always checked).


        Keyword Arguments:
        corr_type   : (Optional) An element from class CorrType, default is CorrType.MOST.
        eps         : (Optional) A positive float used as a minimum cumulative weight threshold.
        ws          : (Optional) A N numeric weight vector of non-negative values.
                                 Defaults to uniform weights.
        exclude_labs: (Optional) A list of labels in the larger universe, <ulabs>, 
                                 to exclude in the correlation analysis.
        chk_con     : (Optional) A boolean, defaults to False meaning, do NOT check the input contract
                                 -- see the documentation for the function: ic.check_most_corr_vec_input_contract.

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
        When no valid correlate exists (everything excluded, or all correlations undefined),
        best_correlate is None, best_corr is NaN and valid_cnt is 0.

        Throws
        ------
        ValueError
     """

    # Optionally check input contract.
    if chk_con:
        ic.check_most_corr_vec_input_contract(X, labs, ulabs, lab_dict, eps, ws, exclude_labs)

    corr, invalid, _, _ = _best_corr_setup(X, labs, ulabs, lab_dict, corr_type, ws, exclude_labs, "most_corr_vec")
    H   = len(labs)
    ind = np.arange(H)

    # For each vector in <labs>, get the top correlate, its index, and if 
    # the correlation is valid.
    bidx  = _best_positions(corr, invalid, corr_type)
    valid = ~invalid[ind, bidx]
    val   = np.where(valid, corr[ind, bidx], np.nan)
    best  = [ulabs[b] if v else None for b, v in zip(bidx, valid)]

    # Return a Pandas Dataframe consisting of <labs>; 
    # the most correlated vectors(their labels); their correlation with <labs>;
    # and if the best correlate is valid; 1 for yes, 0 for no.
    return pd.DataFrame({'lab'           : labs             , 
                         'best_correlate': best             , 
                         'best_corr'     : val              , 
                         'valid_cnt'     : valid.astype(int)} )


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
        ulabs       : The larger universe of vectors: ulabs[i] is the name of row i of <X>.
        lab_dict    : A dictionary mapping vector labels into the row index of <X>.
                      It must agree with <ulabs>: lab_dict[ulabs[i]] == i (always checked).
        k           : Positive integer, the number of top correlates to retrieve (k <= M, always checked).

        Keyword Arguments:
        corr_type   : (Optional) An element from class CorrType, default is CorrType.MOST.
        eps         : (Optional) A positive float used as a minimum cumulative weight threshold.
        ws          : (Optional) An np.ndarray numeric weight vector of length N of non-negative values.
        exclude_labs: (Optional) An np.ndarray of labels in the larger universe, ulabs, to exclude in the correlation analysis.
        chk_con     : (Optional) A boolean, defaults to False, meaning do NOT
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
              determined by <corr_type>. Positions beyond the valid
              correlates hold None / NaN.


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

    corr, invalid, _, _ = _best_corr_setup(X, labs, ulabs, lab_dict, corr_type, ws, exclude_labs, "most_corr_vecs")
    H   = len(labs)
    ind = np.arange(H)

    # For each vector in <labs>, get the top <k> correlate indexes, correlations, 
    # and count of valid correlates.
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1 or k > corr.shape[1]:
        raise ValueError(f"most_corr_vecs: k({k}) must be an integer in the range [1, {corr.shape[1]}].")
    idxs  = _best_positions(corr, invalid, corr_type, k)
    valid = ~np.take_along_axis(invalid, idxs, axis=1)
    vals  = np.where(valid, np.take_along_axis(corr, idxs, axis=1), np.nan)
    best  = [[ulabs[b] if v else None for b, v in zip(row, vrow)] for row, vrow in zip(idxs, valid)]

    # Return a DataFrame of vector labels; the most correlated vectors(their labels); 
    # their correlations; and the count of valid correlates.
    return pd.DataFrame({'lab'             : labs                    , 
                         'best_correlates' : best                    , 
                         'best_corrs'      : list(vals)              , 
                         'valid_cnt'       : np.sum(valid, axis=1)   } )
