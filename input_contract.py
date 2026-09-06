import numpy as np
from typing import Any, Dict, Optional

def chk_wgt_quantiles_contract(vs :np.ndarray, 
                               wts:np.ndarray, 
                               qs :np.ndarray ) -> None:
    """
    Function which checks the input parameter contract for the function: wgt_quantiles.

    Parameter Contract
    -----------------
    1. vs, wts, qs are all numpy 1-d arrays.
    2. qs in [0.0, 1.0].
    3. |vs| == |wts| > 0
    4. all(wts) >= 0
    5. sum(wts) > 0

    Return
    ------
    None

    Throws
    ------
    ValueError

    """

    # The name of the function whose arguments we are checking.
    nm = "wgt_quantiles"

    # 1. Are vs, wts, and qs are numpy arrays?
    if not isinstance(vs, np.ndarray):
        raise ValueError(f"{nm}: <vs> : Not a numpy array."     )
    if len(vs.shape) != 1:
        raise ValueError(f"{nm}: <vs>: Not a 1-D numpy array."  )
    if not isinstance(wts, np.ndarray):
        raise ValueError(f"{nm}: <wts>: Not a numpy array."     )
    if len(wts.shape) != 1:
        raise ValueError(f"{nm}: <wts>: Not a 1-D numpy array." )
    if not isinstance(qs, np.ndarray):
        raise ValueError(f"{nm}: <qs> : Not a numpy array."     )
    if len(qs.shape) != 1:
        raise ValueError(f"{nm}: <qs>: Not a 1-D numpy array."  )
    
    # 2. All qs values in [0.0, 1.0]?
    if np.any((qs < 0.0) | (qs > 1.0)):
        raise ValueError(f"{nm}: <qs>: Not a proper quantiles array.")
  
    # 3. The length of vs and wts is the same (and positive)?
    if np.size(vs) != np.size(wts):
        raise ValueError(f"{nm}: <vs> and <wts> do not have the same length.")
    if np.size(vs) == 0:
        raise ValueError(f"{nm}: <vs> is empty.")

    # 4. all wts >= 0?
    if np.any(wts < 0.0):
        raise ValueError(f"{nm}: <wts> has one or more negative elements.")

    # 5. sum(wts) > 0?
    if np.sum(wts) <= 0:
        raise ValueError(f"{nm}: The sum of the elements of <wts> is not positive.")
      


def chk_wgt_quantiles_tensor_contract(VS :np.ndarray, 
                                      wts:np.ndarray, 
                                      qs :np.ndarray ) -> None:
    """
    This function checks the input parameter contract for the function wgt_quantiles_tensor.

    Parameter Contract
    -----------------
    1. VS, and wts are numpy arrays.
    2. VS is a numpy matrix.
    3. qs in [0.0, 1.0].
    4. |VS[0]| == |wts| > 0
    5. all(wts) >= 0
    6. sum(wts) > 0

    Return
    ------
    None

    Throws
    ------
    ValueError
    """
    
    # The name of the function whose arguments we are checking.
    nm = "wgt_quantiles_tensor"

    # 1. Are <VS> and <wts> are numpy arrays?
    if not isinstance(wts, np.ndarray):
        raise ValueError(f"{nm}: <wts>: Not a numpy array."  )
    if len(wts.shape) != 1:
        raise ValueError(f"{nm}: <wts>: Not a 1-D array."    )
    if not isinstance(qs, np.ndarray):
        raise ValueError(f"{nm}: <qs>: Not a numpy array."   ) 
    if len(qs.shape) != 1:
        raise ValueError(f"{nm}: <qs>: Not a 1-D array."     )

    # 2. Is <VS> is a numpy matrix?
    if not isinstance(VS, np.ndarray):
        raise ValueError(f"{nm}: <VS>: Not a numpy array."   )
    if len(VS.shape) != 2:
        raise ValueError(f"{nm}: <VS>: Not a numpy matrix."  )

    # 3. All <qs> values in [0.0, 1.0]?
    if np.any((qs < 0.0) | (qs > 1.0)):
        raise ValueError(f"{nm}: <qs>: Not a proper quantiles array.")
  
    # 4. The length of <VS> rows and the length of <wts> are the same (and positive)?
    if VS.shape[1] != np.size(wts):
        raise ValueError(f"{nm}: The rows of <VS> don't have the same length as <wts>.")
    if VS.shape[1] == 0:
        raise ValueError(f"{nm}: The rows of <VS> are empty.")

    # 5. Are all <wts> elements >= 0?
    if np.any(wts < 0.0):
        raise ValueError(f"{nm}: Weights array, <wts>, has one or more negative elements.")

    # 6. Is sum(wts) > 0?
    if np.sum(wts) <= 0:
        raise ValueError(f"{nm}: The sum of the elements of <wts> is not positive.")


def chk_corr_cov_contract(X  :np.ndarray,
                          eps:float     ,
                          ws :Optional[np.ndarray]) -> None:
    """
    Function which checks the input parameter contract for the function: corr_cov.

    Parameter Contract
    -----------------
    1. X is a 2-D numpy array.
    2. eps > 0.0
    3. If ws is not None: it is a 1-D numpy array, |ws| = |X[0]|, all(ws) >= 0, sum(ws) >= eps.

    Throws
    ------
    ValueError
    """
    nm = "corr_cov"
    if not isinstance(X, np.ndarray):
        raise ValueError(f"{nm}: The parameter, X, is not a numpy array.")
    if X.ndim != 2:
        raise ValueError(f"{nm}: Parameter, X, is not a matrix.")
    if eps <= 0.0:
        raise ValueError(f"{nm}: Parameter, eps, is not a positive number.")
    _check_weights(ws, X.shape[1], eps, nm)


def _check_weights(ws: Optional[np.ndarray], N: int, eps: float, nm: str) -> None:
    """
    Shared weight checks: if <ws> is not None it must be a finite, non-negative
    1-D numpy array of length <N> whose sum is at least <eps>.
    """
    if ws is None:
        return
    if not isinstance(ws, np.ndarray):
        raise ValueError(f"{nm}: The parameter, ws, is not a numpy array.")
    if ws.ndim != 1:
        raise ValueError(f"{nm}: The parameter, ws, is not a 1-d numpy array.")
    if N != len(ws):
        raise ValueError(f"{nm}: Parameter, X, and, ws, are not compatible.")
    if np.any(np.isnan(ws)):
        raise ValueError(f"{nm}: Parameter, ws, has at least one numpy.nan value.")
    if np.any(np.isinf(ws)):
        raise ValueError(f"{nm}: Parameter, ws, has at least one infinite value.")
    if np.any(ws < 0):
        raise ValueError(f"{nm}: Parameter, ws, has some negative elements.")
    if np.sum(ws) < eps:
        raise ValueError(f"{nm}: Parameter, ws, has cumulative sum that is less than eps({eps}).")


def _check_labels(X: np.ndarray, labs: np.ndarray, ulabs: np.ndarray, lab_dict: Dict[Any, int],
                  exclude_labs: Optional[np.ndarray], nm: str) -> None:
    """
    Shared label checks (items 5-9 of the most_corr_vec contract), including that
    <lab_dict> agrees with <ulabs>: lab_dict[ulabs[i]] == i.
    """
    M = X.shape[0]

    # 5. Are <ulabs> the same length as the number of vectors?
    if len(ulabs) != M:
        raise ValueError(f"{nm}: The length of ulabs({len(ulabs)}) does not match the first dimension of X({M}).")

    # 6. Are <ulabs> unique?
    if len(np.unique(ulabs)) != len(ulabs):
        raise ValueError(f"{nm}: The labels for the universe are not unique.")

    # 7. Is <labs> a subset of <ulabs>?
    if not np.all(np.isin(labs, ulabs)):
        raise ValueError(f"{nm}: Not all labels in, labs, are in the universe of labels, ulabs.")

    # 8. If non-null, is <exclude_labs> a subset of <ulabs>?
    if exclude_labs is not None:
        if not isinstance(exclude_labs, np.ndarray):
            raise ValueError(f"{nm}: The parameter, exclude_labs, is not a numpy array.")
        if not np.all(np.isin(exclude_labs, ulabs)):
            raise ValueError(f"{nm}: Not all labels in, exclude_labs, are in the universe of labels, ulabs.")

    # 9. Does <lab_dict> map every label of <ulabs> to its row?
    for i, lab in enumerate(ulabs):
        if lab not in lab_dict:
            raise ValueError(f"{nm}: The label {lab!r} in ulabs is not a key in lab_dict.")
        if lab_dict[lab] != i:
            raise ValueError(f"{nm}: lab_dict[{lab!r}] is {lab_dict[lab]}, but {lab!r} is row {i} of X (ulabs[{i}]).")
  

def check_most_corr_vec_input_contract(X           :np.ndarray          ,
                                       labs        :np.ndarray          , 
                                       ulabs       :np.ndarray          , 
                                       lab_dict    :Dict[Any, int]      , 
                                       eps         :float               , 
                                       ws          :Optional[np.ndarray],
                                       exclude_labs:Optional[np.ndarray] ) -> None:
    """
        This function checks the input contract for the function "most_corr_vec".
        This function returns nothing, but will raise a "ValueError" if the contract is not satisfied.

        Input Contract:
        ---------------
        1. Are labs and <ulabs> 1-d arrays?
        2. Is <X> a 2d array?
        3. Are the values of X valid? 
            a. Does it contain any -np.inf values?
            b. Does it contain any np.inf values?
            c. Does it contain any np.nan values?
        4. If ws is non-null:
            a. Is it a 1-d array?
            b. Are all elements non-negative?
            c. Is its cumulative sum >= eps?
            d. Does it contain any -np.inf values?
            e. Does it contain any np.inf values?
            f. Does it contain any np.nan values?
        5. Do the number of labels in our universe match the number of N-vectors: |<ulabs>| = <X>.shape[0] ?
        6. Are the elements of <ulabs> unique?
        7. Is <labs> a subset of <ulabs>?
        8. If non-null, is <exclude_labs> a subset of <ulabs>?
        9. Does <lab_dict> map every label in <ulabs> to its row in <X> (lab_dict[ulabs[i]] == i)?

        Returns
        -------
        None


        Throws
        ------
        ValueError

    """

    # The name of the function whose arguments we are checking.
    nm = "most_corr_vec"

    _check_arrays_and_values(X, labs, ulabs, eps, ws, nm)
    _check_labels(X, labs, ulabs, lab_dict, exclude_labs, nm)



def check_most_corr_vecs_input_contract(X           :np.ndarray          ,
                                        labs        :np.ndarray          , 
                                        ulabs       :np.ndarray          , 
                                        lab_dict    :Dict[Any, int]      , 
                                        k           :int                 ,
                                        eps         :float               , 
                                        ws          :Optional[np.ndarray],
                                        exclude_labs:Optional[np.ndarray] ) -> None:
    """
        This function checks the input contract for the function "most_corr_vecs".
        This function returns nothing, but will raise a "ValueError" if the contract is not satisfied.

        Input Contract:
        ---------------
        1. Are labs and <ulabs> 1-d arrays?
        2. Is <X> a 2d array? (M x N)
        3. Are the values of X valid?
            a. Does it contain any -np.inf values?
            b. Does it contain any np.inf values?
            c. Does it contain any np.nan values?
        4. If ws is non-null:
            a. Is it a 1-d array?
            b. Are all elements non-negative?
            c. Is its cumulative sum >= eps?
            d. Does it contain any -np.inf values?
            e. Does it contain any np.inf values?
            f. Does it contain any np.nan values?
        5. Do the number of labels in our universe match the number of M-vectors: |<ulabs>| = M ?
        6. Are the elements of <ulabs> unique?
        7. Is <labs> a subset of <ulabs>?
        8. If non-null, is <exclude_labs> a subset of <ulabs>?
        9. Is k an integer and is k > 0?
        10. Are there enough elements from the universe to form k correlates: (ulabs - exclude_labs) >= k?
        11. Does <lab_dict> map every label in <ulabs> to its row in <X> (lab_dict[ulabs[i]] == i)?

        Returns
        -------
        None

        Throws
        ------
        ValueError

    """
    
    # The name of the function whose arguments we are checking.
    nm = "most_corr_vecs"
    
    _check_arrays_and_values(X, labs, ulabs, eps, ws, nm)
    _check_labels(X, labs, ulabs, lab_dict, exclude_labs, nm)

    # 9a, Is <k> an integer?
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise ValueError(f"{nm}: The parameter, k({k}), is not an integer.")

    # 9b, Is <k> > 0?
    if k <= 0:
        raise ValueError(f"{nm}: The parameter, k({k}), is not positive.")

    # 10. Are there at least <k> labels in <ulabs> less <exclude_labs>?
    if exclude_labs is not None:
        if (len(ulabs) - len(np.unique(exclude_labs))) < k:
            raise ValueError(f"{nm}: The expression, (ulabs - exclude_labs), has less than k({k}) elements.")
    else:  
        if len(ulabs) < k:
            raise ValueError(f"{nm}: The parameter, ulabs, has less than k({k}) elements.")


def _check_arrays_and_values(X: np.ndarray, labs: np.ndarray, ulabs: np.ndarray, eps: float,
                             ws: Optional[np.ndarray], nm: str) -> None:
    """
    Shared array checks (items 1-4 of the most_corr_vec contract).
    """
    # 1a.
    if not isinstance(labs, np.ndarray):
        raise ValueError(f"{nm}: The parameter, labs, is not a numpy array.")
    if labs.ndim != 1:
        raise ValueError(f"{nm}: The parameter, labs, is not a 1-d numpy array.")

    # 1b.
    if not isinstance(ulabs, np.ndarray):
        raise ValueError(f"{nm}: The parameter, ulabs, is not a numpy array.")
    if ulabs.ndim != 1:
        raise ValueError(f"{nm}: The parameter, ulabs, is not a 1-d numpy array.")

    # 2.
    if not isinstance(X, np.ndarray):
        raise ValueError(f"{nm}: The parameter, X, is not a numpy array.")
    if X.ndim != 2:
        raise ValueError(f"{nm}: The parameter, X, is not a 2-d numpy array.")

    # 3.
    if np.any(np.isnan(X)):
        raise ValueError(f"{nm}: Parameter, X, has at least one numpy.nan value.")
    if np.any(np.isinf(X)):
        raise ValueError(f"{nm}: Parameter, X, has at least one infinite value.")

    # 4.
    _check_weights(ws, X.shape[1], eps, nm)
