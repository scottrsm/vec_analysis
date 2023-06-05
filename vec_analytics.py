import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
import input_contract as ic



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
  

    ## Check input parameter contract.
    ic.chk_wgt_quantiles_contract(vs, wts, qs)

    ## Sort the vs array and the associated weights.
    ## Turn the weights into proper weights and create a cumulative weight array.
    idx  = np.argsort(vs)
    ovs  = vs[idx]
    ows  = wts[idx]
    ows  = ows / np.sum(ows) # Normalize the weights.
    cws  = np.cumsum(ows)
  
    N    = np.size(cws)
    M    = np.size(qs)
  
    ## Reshape to broadcast.
    cws.shape = (N, 1)
    qss = qs.copy()
    qss.shape  = (1, M)
  
    ## Use broadcasting to get all comparisons of <cws> with each entry from <qs>.  
    ## Form tensor (cws <= qss) * 1 and sandwich index of the value vectors with 0 and 1.
    A   = np.concatenate([np.ones(M).reshape(1,M), (cws <= qss) * 1, np.zeros(M).reshape(1,M)], axis=0)
  
    ## Get the diff -- -1 will indicate the boundary where cws > qs.
    X   = np.diff(A, axis=0).astype(int)
  
    ## Get the indices of the boundary.
    idx = np.maximum(0, np.where(X == -1)[0] - 1)
  
    ## Return the weighted quantile value of <vs> against each <qs>.
    return(ovs[idx])


def wgt_quantiles_tensor(vs :np.ndarray, 
                         wts:np.ndarray, 
                         qs :np.ndarray ) -> np.ndarray:
    '''
    Compute a (D, M) numpy array consisting of the quantile weighted values of <vs> using weights, <wts>, for each quantile in <qs>.
    
    Parameters
    ----------
    vs    A numpy(np) (D, N) matrix of numeric values. 
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
    1. vs, and wts are numpy arrays.
    2. vs is a numpy matrix.
    3. qs in [0.0, 1.0]
    4. |vs[0]| == |wts|
    5. all(wts) >= 0
    6. sum(wts) > 0
    
    Assumptions
    -----------
    1. <vs>, <wts>, and <qs> are all numeric arrays.

    '''
  
    ## Check input parameter contract.
    ic.chk_wgt_quantiles_tensor_contract(vs, wts, qs)

    ## Normalize the weights.
    ws  = wts / np.sum(wts)
  
    D, N  = vs.shape
    M     = qs.size

    ## Get the sorted index array for each of the value vectors in vs.
    idx = np.argsort(vs, axis=1)
  
    ## Apply this index back to vs to get sorted values.
    ovs = np.take_along_axis(vs, idx, axis=1)
  
    ## Apply the index to the weights, where, the dimension of ws (and cws) expands to: (D, N).
    ows = ws[idx]
    cws = np.cumsum(ows, axis=1)

    ## Reshape to broadcast.
    cws.shape = (D, N, 1)
    qss = qs.copy()
    qss.shape  = (1, 1, M)

    ## Use broadcasting to get all comparisons of <cws> with each entry from <qs>. 
    ## Form tensor (cws <= qss) * 1 and sandwich index of the value vectors with 0 and 1.
    A = np.concatenate([np.ones(M*D).reshape(D,1,M), (cws <= qss) * 1, np.zeros(M*D).reshape(D,1,M)], axis=1)
  
    ## Compute the index difference on the value vectors.
    Delta = np.diff(A, axis=1).astype(int)

    ## Get the index of the values, this leaves, essentially, a (D, M) matrix. Reshape it as such.
    idx = np.maximum(0, np.where(Delta == -1)[1] - 1)
    idx = idx.reshape(D, M) 
  
    ## Return the values in the value vectors that correspond to these indices -- the M quantiles for each of the D value vectors.
    ## A (D, M) matrix.
    return(np.take_along_axis(ovs, idx, axis=1))



def corr(x           : np.ndarray                 , 
         eps         : float = 1.0e-6             ,
         chk_contract:bool = True                 , 
         ws          : Optional[np.ndarray] = None ) -> np.ndarray:
    """!
        Find the correlations between the M vectors of length N, represented as the MxN matrix x.

        Parameters
        ----------
        x           : A MxN numeric matrix reprsenting M vectors of length N.
        eps         : A float value. The sum of the weights should be larger than this value.
        chk_contract: A boolean, defaults to True meaning, check the input parameter contract.
        ws          : (Opotional) A N numeric vector of weights of non-negative values.

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
        if len(x.shape) != 2:
            raise(ValueError("corr: Parameter, x, is not a matrix."))
        if ws:
            if len(ws.shape) != 1:
                raise(ValueError("corr: Parameter, ws, is not a 1-d numpy array."))

    ## x is an MxN array -- meaning M vectors each of length N.
    M, N = x.shape

    ## Check that ws and x are compatible.
    if ws:
        if N != len(ws):
            raise(ValueError("corr: Parameter, x, and, ws, are not compatible."))

        if np.any(ws < 0):
            raise(ValueError("corr: Parameter, ws, has some negative elements."))

        if np.sum(ws) < eps:
            raise(ValueError("corr: Parameter, ws, has cumulative sum that is less than eps({eps})."))

    ## If not given set ws to its default setting -- uniform weights.
    if not ws:
        ws = np.ones(N)

    ## Reshape weights for computation.
    ws.shape = (1, 1, N)

    ## Subtract off the mean of each row.
    ## We need to "reshape" the mean, <mn>, to do this -- so that "broadcasting" works.
    mn = np.mean(x, axis=1)
    mn.shape=(M, 1)
    x = x - mn

    ## Copy x and now do a reshape of each so that the "rules of broadcasting" will give us
    ## all combinations of <x> * <y>.
    y = x.copy()
    x.shape = (1, M, N)
    y.shape = (M, 1, N)

    ## Now use aggregation to sum up the third index -- the values -- to get an MxM matrix 
    ## of cross-correlations.
    return( np.sum(x * y * ws, axis=2) / np.sqrt(np.sum(x * x * ws, axis=2) * np.sum(y * y * ws, axis=2)) )


def most_corr(labs        : np.ndarray                 , 
              x           : np.ndarray                 , 
              eps         : float = 1.0e-6             ,
              anti        : bool = False               ,
              chk_contract: bool = True                , 
              ws          : Optional[np.ndarray] = None ) -> pd.DataFrame:
    """!
        For a collection of M vectors of length N, with labels, labs, find the most correlated vector for each.

        Parameters
        ----------
        labs        : The labels of each of the M vectors of x.
        x           : The MxN matrix representing the M vectors values.
        eps         : A positive float used as a minimum cumulative weight threshold.
        anti        : If True, then max correlation can mean most positive or 
                      most negative correlation, which ever is larger in absolute value.
        chk_contract: A boolean, defaults to True meaning, check the input parameter contract.
        ws          : (Optional) A N numeric vector of weights of non-negative values.

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
    ## Check input contract?
    if chk_contract:
        ic.check_most_corr_input_contract(x, eps, ws)

    ## Get M and N: M vectors of length N.
    M, N = x.shape

    ## If not given set <ws> to its default setting -- uniform weights.
    if not ws:
        ws = np.ones(N)

    ## Reshape weights for computation.
    ws.shape = (1, 1, N)

    ## Subtract off row means.
    ## Copy <x> to <y>, reshape both for computation.
    mn = np.mean(x, axis=1)
    mn.shape=(M, 1)
    x = x - mn
    y = x.copy()
    x.shape = (1, M, N)
    y.shape = (M, 1, N)

    ## Compute correlation.
    corr = np.sum(x * y * ws, axis=2) / ( np.sqrt(np.sum(x * x * ws, axis=2) * np.sum(y * y * ws, axis=2)) ) 
    ind = np.arange(M)

    ## For each vector, find associated vector with max correlation.
    if anti:
        corr[ind, ind] = 0.0 # Set self correlation to 0 to avoid being picked.
        idx = np.argmax(np.abs(corr), axis=1) 
        val = corr[ind, idx]
    else:
        corr[ind, ind] = -np.inf # Set self correlation to -infinity to avoid being picked.
        idx = np.argmax(corr, axis=1) 
        val = corr[ind, idx]

    ## Return a DataFrame of vector labels; the most correlated vector -- by label; and their correlation.
    return(pd.DataFrame({'lab' : labs, 'max_corr': labs[idx], 'corr' : val}))


def most_corr_vec(labs        : np.ndarray                 , 
                  ulabs       : np.ndarray                 , 
                  lab_dict    : Dict[Any, int]             , 
                  x           : np.ndarray                 ,
                  eps         : float = 1.0e-6             ,
                  chk_contract: bool = True                , 
                  ws          : Optional[np.ndarray] = None,
                  exclude_labs: Optional[np.ndarray] = None ) -> pd.DataFrame:
    """!
        For each vector in a list, determine the most correlated vector from a larger universe.
        Correlation may be weighted; one may also exclude some vectors from the larger universe.

        Parameters
        ---------
        labs        : The labels of the vectors of interest.
        ulabs       : The larger universe of vectors.
        lab_dict    : A dictionary mapping vector labels into the row index of <x>.
                      At a minimum, the keys must include <labs>.
        x           : A MxN matrix of M vectors, each of length N.
        eps         : A positive float used as a minimum cumulative weight threshold.
        chk_contract: A boolean, defaults to True meaning, check the input parameter contract.
        ws          : (Optional) A N numeric weight vector of non-negative values.
        exclude_labs: (Optional) A list of labels in the larger universe, ulabs, to exclude in the correlation analysis.

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

    ## Check input contract?
    if chk_contract:
        ic.check_most_corr_vec_input_contract(labs, ulabs, lab_dict, x, eps, ws, exclude_labs)

    M, N = x.shape

    ## If not given set <ws> to its default setting -- uniform weights.
    if not ws:
        ws = np.ones(N)

    ## Reshape weights for computation.
    ws.shape = (1, 1, N)

    ## Subtract off row means.
    mn       = np.mean(x, axis=1) # Vector means.
    mn.shape = (M, 1)             # broadcasting
    x        = x - mn
    y        = x.copy()

    ## Get the index of each security of interest.
    idx = [lab_dict[lab] for lab in labs]

    ## If exclude_labs is not None, find their row indices in the correlation matrix.
    eidx = None
    if exclude_labs:
        eidx = [lab_dict[lab] for lab in exclude_labs]

    ## Get the correlation of our chosen vectors against the full universe -- giving an HxM matrix (H the length of labs)
    x.shape = (M, 1, N)   # expand for broadcasting.
    x       = x[idx,:,:]  # Get only the <labs> vectors.
    y.shape = (1, M, N)   # expand for broadcasting and use the full set of vectors.

    ## Compute correlation matrix of the <labs> vectors against the universe -- <ulabs> vectors.
    ## Set self correlation to -infinity for only the <labs> rows of the correlation matrix.
    corr = np.sum(x * y * ws, axis=2) / ( np.sqrt(np.sum(x * x * ws, axis=2) * np.sum(y * y * ws, axis=2)) )  # Aggregation of third index.
    ind = np.arange(len(labs))
    corr[ind, idx] = -np.inf    # Array slicing -- fill "diagonal" with -infinity -- effectively eliminating themselves as their maximum correlate.

    ## If exclude_labs is not None, set correlations will all these <labs> vectors to -infinity to exclude them from consideration.
    if eidx:
        for id in ind:
            corr[id, eidx] = -np.inf

    ## Get the top correlate position and value.
    idx = np.argmax(corr[ind, :], axis=1) 
    val = corr[ind, idx]          

    ## Return a Dataframe of vector labels; the most correlated vector(its label); and their correlation.
    return(pd.DataFrame({'lab' : labs, 'max_correlate': ulabs[idx], 'max_corr' : val}))


def most_corr_vecs(labs        : np.ndarray                 , 
                   ulabs       : np.ndarray                 , 
                   lab_dict    : Dict[Any, int]             , 
                   k           : int                        , 
                   x           : np.ndarray                 ,
                   eps         : float = 1.0e-6             ,
                   chk_contract: bool = True                ,
                   ws          : Optional[np.ndarray] = None,
                   exclude_labs: Optional[np.ndarray] = None ) -> pd.DataFrame:
    """!
        Determine the most correlated k vectors from a large universe for each vector in a smaller subset.

        Parameters
        ---------
        labs        : The labels of the vectors of interest.
        ulabs       : The larger universe of vectors.
        lab_dict    : A dictionary mapping vector labels into the row index of <x>.
                      At a minimum, the keys must include <labs>.
        k           : Positive integer, the number of top correlates to retrieve.
        x           : A MxN matrix of M vectors, each of length N.
        eps         : A positive float used as a minimum cumulative weight threshold.
        chk_contract: A boolean, defaults to True meaning, check the input parameter contract.
        ws          : (Optional) A N numeric weight vector of non-negative values.
        exclude_labs: (Optional) A list of labels in the larger universe, ulabs, to exclude in the correlation analysis.

        Packages
        --------
        numpy(np)
        pandas(pd)

        Return
        ------
        A Pandas Dataframe with schema: lab(vector label), top_correlate(vector label), top_corr(their correlation)

        Throws
        ------
        ValueError

     """

    ## Check input contract?
    if chk_contract:
        ic.check_most_corr_vecs_input_contract(labs, ulabs, lab_dict, k, x, eps, ws, exclude_labs)
        
    ## Extract shape of <x>.
    M, N = x.shape

    ## Subtract off row means.
    mn       = np.mean(x, axis=1)
    mn.shape = (M, 1) # broadcasting
    x        = x - mn
    y        = x.copy()

    ## Get the index of each security of interest.
    idx = [lab_dict[lab] for lab in labs]

    ## If exclude_labs is not None, find their row indices in the correlation matrix.
    eidx = None
    if exclude_labs:
        eidx = [lab_dict[lab] for lab in exclude_labs]

    ## Get the correlation of our chosen vectors against the full universe -- giving an HxM matrix (H the length of labs)
    x.shape = (M, 1, N)   # expand for broadcasting.
    x       = x[idx,:,:]  # Get only the <labs> vectors.
    y.shape = (1, M, N)   # expand for broadcasting and use the full set of vectors.

    ## Compute correlation matrix H(length of labs) x M(all vectors in the universe). 
    corr           = np.sum(x * y, axis=2) / np.sqrt(np.sum(x * x, axis=2) * np.sum(y * y, axis=2))  
    ind            = np.arange(len(labs))
    corr[ind, idx] = -np.inf    # Array slicing -- fill "diagonal" with -infinity -- effectively eliminating themselves as their maximum correlate.

    ## If exclude_labs is not None, set correlations will all these lab vectors to -infinity to exclude them from consideration.
    if eidx:
        for id in ind:
            corr[id, eidx] = -np.inf

    ## Get the top <k> correlate positions and values.
    idx = np.argsort(corr, axis=1)[:, -k:] 
    val = np.sort(corr, axis=1)[:, -k:]    

    ## Return a DataFrame of vector labels; the most correlated vector(its label); and their correlation.
    return(pd.DataFrame({'lab' : labs, 'top_correlates' : ulabs[idx].tolist(), 'top_corrs': val.tolist() }))


