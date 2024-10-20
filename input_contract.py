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
    2. qs in [0.0, 1.0]
    3. |vs| == |wts|
    4. all(wts) >= 0
    5. sum(wts) > 0

    Return
    ------
    None

    Throws
    ------
    ValueError

    """
    # 1. Are vs, wts, and qs are numpy arrays?
    if type(vs)  != np.ndarray:
        raise ValueError('wgt_quantiles: <vs> : Not a numpy array.'     )
    if len(vs.shape) != 1:
        raise ValueError('wgt_quantiles: <vs>: Not a 1-D numpy array.'  )
    if type(wts) != np.ndarray:
        raise ValueError('wgt_quantiles: <wts>: Not a numpy array.'     )
    if len(wts.shape) != 1:
        raise ValueError('wgt_quantiles: <wts>: Not a 1-D numpy array.' )
    if type(qs)  != np.ndarray:
        raise ValueError('wgt_quantiles: <qs> : Not a numpy array.'     )
    if len(qs.shape) != 1:
        raise ValueError('wgt_quantiles: <qs>: Not a 1-D numpy array.'  )
    
    # 2. All qs values in [0.0, 1.0]?
    if any((qs < 0.0) | (qs > 1.0)):
        raise ValueError('wgt_quantiles: <qs>: Not a proper quantiles array.')
  
    # 3. The length of vs and wts is the same?
    if np.size(vs) != np.size(wts):
        raise ValueError('wgt_quantiles: <vs> and <wts> do not have the same length.')

    # 4. all wts >= 0?
    if any(wts < 0.0):
        raise ValueError('wgt_quantiles: <wts> has one or more negative elements.')

    # 5. sum(wts) > 0?
    if sum(wts) <= 0:
        raise ValueError('wgt_quantiles: The sum of the elements of <wts> is not positive.')
      


def chk_wgt_quantiles_tensor_contract(VS :np.ndarray, 
                                      wts:np.ndarray, 
                                      qs :np.ndarray ) -> None:
    """
    This function checks the input parameter contract for the function wgt_quantiles_tensor.

    Parameter Contract
    -----------------
    1. VS, and wts are numpy arrays.
    2. VS is a numpy matrix.
    3. qs in [0.0, 1.0]
    4. |VS[0]| == |wts|
    5. all(wts) >= 0
    6. sum(wts) > 0

    Return
    ------
    None

    Throws
    ------
    ValueError

    """
    # 1. Are <VS> and <wts> are numpy arrays?
    if type(wts) != np.ndarray:
        raise ValueError('wgt_quantiles_tensor: <wts>: Not a numpy array.'  )
    if len(wts.shape) != 1:
        raise ValueError('wgt_quantiles_tensor: <wts>: Not a 1-D array.'    )
    if type(qs) != np.ndarray:
        raise ValueError('wgt_quantiles_tensor: <qs>: Not a numpy array.'   ) 
    if len(qs.shape) != 1:
        raise ValueError('wgt_quantiles_tensor: <qs>: Not a 1-D array.'     )

    # 2. Is <VS> is a numpy matrix?
    if type(VS)  != np.ndarray:
        raise ValueError('wgt_quantiles_tensor: <VS>: Not a numpy array.'   )
    if len(VS.shape) != 2:
        raise ValueError('wgt_quantiles_tensor: <VS>: Not a numpy matrix.'  )

    # 3. All <qs> values in [0.0, 1.0]?
    if any((qs < 0.0) | (qs > 1.0)):
        raise ValueError('wgt_quantiles_tensor: <qs>: Not a proper quantiles array.')
  
    # 4. The length of <VS> rows and the length of <wts> are the same?
    if np.size(VS[0]) != np.size(wts):
        raise ValueError("wgt_quantiles_tensor: The rows of <VS> don't have the same length as <wts>.")

    # 5. Are all <wts> elements >= 0?
    if any(wts < 0.0):
        raise ValueError('wgt_quantiles_tensor: Weights array, <wts>, has one or more negative elements.')

    # 6. Is sum(wts) > 0?
    if sum(wts) <= 0:
        raise ValueError('wgt_quantiles_tensor: The sum of the elements of <wts> is not positive.')
  

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
        9. Is <ulabs> a subset of the keys of <lab_dict>? 

        Returns
        -------
        None


        Throws
        ------
        ValueError

    """
    # 1a.
    if type(labs) != np.ndarray:
        raise ValueError("most_corr_vecs: The parameter, labs, is not a numpy array.")
    
    if len(labs.shape) != 1:
        raise ValueError("most_corr_vec: The parameter, labs, is not a 1-d numpy array.")

    # 1b.
    if type(ulabs) != np.ndarray:
        raise ValueError("most_corr_vec: The parameter, ulabs, is not a numpy array.")

    if len(ulabs.shape) != 1:
        raise ValueError("most_corr_vec: The parameter, ulabs, is not a 1-d numpy array.")

    # 2.
    if type(X) != np.ndarray:
        raise ValueError("most_corr_vec: The parameter, X, is not a numpy array.")

    if len(X.shape) != 2:
        raise ValueError("most_corr_vec: The parameter, X, is not a 2-d numpy array.")

    # 3.
    if np.any(X == -np.inf):
        raise ValueError("most_corr_vec: Parameter, X, has at least one -numpy.inf value.")

    if np.any(X == np.inf):
        raise ValueError("most_corr_vec: Parameter, X, has at least one numpy.inf value.")

    if np.any(np.isnan(X)):
        raise ValueError("most_corr_vec: Parameter, X, has at least one numpy.nan value.")

    # Need shape info for checking.
    M, N = X.shape
    
    # 4. Check <ws>.
    if ws:
        if type(ws) != np.ndarray:
            raise ValueError("most_corr_vec: The parameter, ws, is not a numpy array.")

        if N != len(ws):
            raise ValueError("most_corr_vec: Parameter, X, and, ws, are not compatible.")
            
        if np.any(ws < 0):
            raise ValueError("most_corr_vec: Parameter, ws, has some negative elements.")

        if np.sum(ws) < eps:
            raise ValueError(f"most_corr_vec: Parameter, ws, has cumulative sum that is less than eps({eps}).")

        if np.any(ws == -np.inf):
            raise ValueError("most_corr_vec: Parameter, ws, has at least one -numpy.inf value.")

        if np.any(ws == np.inf):
            raise ValueError("most_corr_vec: Parameter, ws, has at least one numpy.inf value.")

        if np.isnan(ws):
            raise ValueError("most_corr_vec: Parameter, ws, has at least one numpy.nan value.")

    # 5. Are <ulabs> the same length as the number of N-vectors?
    if len(ulabs) != M:
        raise ValueError(f"most_corr_vec: The length of ulabs({len(ulabs)}) does not match the first dimension of X({M}).")

    # 6. Are <ulabs> are unique?
    if len(np.unique(ulabs)) != len(ulabs):
        raise ValueError("most_corr_vec: The labels for the universe are not unique.")

    # 7. Is <labs> a subset of <ulabs>?
    if not np.all(np.isin(labs, ulabs)):
        raise ValueError("most_corr_vec: Not all labels in, labs, are in the universe of labels, ulabs.")

    # 8. If non-null, is <exclude_labs> a subset of the keys of <ulabs>?
    if exclude_labs:
        if not np.all(np.isin(exclude_labs, ulabs)):
            raise ValueError("most_corr_vec: Not all labs in, exlclude_labs, are in the universe of labels, ulabs.")

    # 9. Is <ulabs> a subset of the keys of <lab_dict>?
    if not np.all(np.isin(ulabs, np.array(list(lab_dict.keys())))):
        raise ValueError("most_corr_vec: Not all labels in, ulabs, are keys in lab_dict.")



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
        11. Is <labs> a subset of the keys of <lab_dict>? 

        Returns
        -------
        None

        Throws
        ------
        ValueError

    """
    # 1a.
    if type(labs) != np.ndarray:
        raise ValueError("most_corr_vecs: The parameter, labs, is not a numpy array.")

    if len(labs.shape) != 1:
        raise ValueError("most_corr_vecs: The parameter, labs, is not a 1-d numpy array.")

    # 1b.
    if type(ulabs) != np.ndarray:
        raise ValueError("most_corr_vecs: The parameter, ulabs, is not a numpy array.")

    if len(ulabs.shape) != 1:
        raise ValueError("most_corr_vecs: The parameter, ulabs, is not a 1-d numpy array.")

    # 2.
    if type(X) != np.ndarray:
        raise ValueError("most_corr_vecs: The parameter, ulabs, is not a numpy array.")

    if len(X.shape) != 2:
        raise ValueError("most_corr_vecs: The parameter, X, is not a 2-d numpy array.")

    # 3. Are the values of X valid?
    if np.any(X == -np.inf):
        raise ValueError("most_corr_vecs: Parameter, X, has at least one -numpy.inf value.")

    if np.any(X == np.inf):
        raise ValueError("most_corr_vecs: Parameter, X, has at least one numpy.inf value.")

    if np.any(np.isnan(X)):
        raise ValueError("most_corr_vecs: Parameter, X, has at least one numpy.nan value.")

    # Need shape info for checking.
    M, N = X.shape

    # 4. Check <ws>.
    if ws:
        if type(ws) != np.ndarray:
            raise ValueError("most_corr_vecs: The parameter, ws, is not a numpy array.")

        if len(ws.shape) != 1:
            raise ValueError("most_corr_vecs: The parameter, ws, is not a 1-d numpy array.")
        
        if N != len(ws):
            raise ValueError("most_corr_vecs: Parameter, X, and, ws, are not compatible.")
            
        if np.any(ws < 0):
            raise ValueError("most_corr_vecs: Parameter, ws, has some negative elements.")

        if np.sum(ws) < eps:
            raise ValueError(f"most_corr_vecs: Parameter, ws, has cumulative sum that is less than eps({eps}).")

        if np.any(ws == -np.inf):
            raise ValueError("most_corr_vecs: Parameter, ws, has at least one -numpy.inf value.")

        if np.any(ws == np.inf):
            raise ValueError("most_corr_vecs: Parameter, ws, has at least one numpy.inf value.")

        if np.isnan(ws):
            raise ValueError("most_corr_vecs: Parameter, ws, has at least one numpy.nan value.")

    # 5. Are <ulabs> the same length as the number of M-vectors?
    if len(ulabs) != M:
        raise ValueError(f"most_corr_vecs: The length of ulabs({len(ulabs)}) does not match the first dimension of x({M}).")

    # 6. Are <ulabs> unique?
    if len(np.unique(ulabs)) != len(ulabs):
        raise ValueError("most_corr_vecs: The labels for the universe are not unique.")

    # 7. Is <labs> a subset of <ulabs>?
    if not np.all(np.isin(labs, ulabs)):
        raise ValueError("most_corr_vecs: Not all labels in, labs, are in the universe of labels, ulabs.")

    # 8. Is <exclude_labs> a subset of <ulabs>?
    if exclude_labs:
        if type(exclude_labs) != np.ndarray:
            raise ValueError("most_corr_vecs: The parameter, exclude_labs, is not a numpy array.")

        if not np.all(np.isin(exclude_labs, ulabs)):
            raise ValueError("most_corr_vecs: Not all labels in, labs, are in the universe of labels, ulabs.")

    # 9a, Is <k> an integer?
    if type(k) != type(1):
        raise ValueError(f"most_corr_vecs: The parameter, k({k}), is not an integer.")

    # 9b, Is <k> > 0?
    if k <= 0:
        raise ValueError(f"most_corr_vecs: The parameter, k({k}), is not positive.")

    # 10. Are there at least <k> labels in <ulabs> less <exclude_labs>?
    if exclude_labs:
        if ( len(ulabs) - len(exclude_labs) ) < k:
            raise ValueError(f"most_corr_vecs: The expression, (ulabs - exclude_labs), has less than k({k}) elements.")
    else:  
        if len(ulabs) < k:
            raise ValueError(f"most_corr_vecs: The parameter, ulabs, has less than k({k}) elements.")

    # 11. Is <labs> a subset of the keys of <lab_dict>?
    if not np.all(np.isin(labs, np.array(list(lab_dict.keys())))):
        raise ValueError("most_corr_vecs: Not all labels in, labs, are keys in lab_dict.")



