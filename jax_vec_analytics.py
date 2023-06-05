import numpy as np
import jax.numpy as jnp


def wgt_quantiles(vs, wts, qs):
  '''
  Get a numpy array consisting of an array of quantile weighted <vs> values.
    
  Parameters
  ----------
  vs    A numpy(jax.numpy) (N) array of numeric values. 
  wts   A numpy(jax.numpy) (N) array of numeric weights. (Weights need only be non-negative, they need not sum to 1.)
  qs    A numpy(jax.numpy) (D) array of numeric values.  (Meant to be quantiles -- numbers in the range [0, 1]).

  Returns
  -------
  :A numpy array consisting of the quantile weighted values of <vs> using weights, <wts>, for each quantile in <qs>.
  
  Return-Type
  -----------
  A jax.numpy(jnp) (D) array of weighted quantile <vs> values with the same length as <qs>.
  
  Packages
  --------
  numpy(np)
  jax.numpy(jnp)

Parameter Contract
  -----------------
    1. vs, wts, qs are all numpy arrays.
    2. qs in [0.0, 1.0]
    3. |vs| == |wts|
    4. all(wts) >= 0
    5. sum(wts) > 0
    
  Assumptions
  -----------
    1. <vs>, <wts>, and <qs> are all numeric arrays.
  '''
  
  ## 1. Are vs, wts, and qs are numpy arrays?
  if type(vs)  != np.ndarray:
    raise(ValueError('wgt_quantiles: <vs> : Not a numpy array.'            ))
  if len(vs.shape) != 1:
    raise(ValueError('wgt_quantiles_tensor: <vs>: Not a 1-D numpy array.'  ))
  if type(wts) != np.ndarray:
    raise(ValueError('wgt_quantiles: <wts>: Not a numpy array.'            ))
  if len(wts.shape) != 1:
    raise(ValueError('wgt_quantiles_tensor: <wts>: Not a 1-D numpy array.' ))
  if type(qs)  != np.ndarray:
    raise(ValueError('wgt_quantiles: <qs> : Not a numpy array.'            ))
  if len(qs.shape) != 1:
    raise(ValueError('wgt_quantiles_tensor: <qs>: Not a 1-D numpy array.'  ))
    
  ## 2. All qs values in [0.0, 1.0]?
  if any((qs < 0.0) | (qs > 1.0)):
    raise(ValueError('wgt_quantiles: <qs>: Not a proper quantiles array.'  ))
  
  ## 3. The length of vs and wts is the same?
  if np.size(vs) != np.size(wts):
    raise(ValueError('wgt_quantiles: <vs> and <wts> do not have the same length.'))

  ## 4. all wts >= 0?
  if any(wts < 0.0):
    raise(ValueError('wgt_quantiles: <wts> has one or more negative elements.'))

  ## 5. sum(wts) > 0?
  if sum(wts) <= 0:
    raise(ValueError('wgt_quantiles: Sum of <wts> is not positive.'))
      
  ## Convert data to jax numpys.
  vs   = jnp.asarray(vs)
  qs   = jnp.asarray(qs)
  wts  = jnp.asarray(wts)

  ## Sort the vs array and the associated weights.
  ## Turn the weights into proper weights and create a cumulative weight array.
  idx  = jnp.argsort(vs)
  ovs  = vs[idx]
  ows  = wts[idx]
  ows  = ows / jnp.sum(ows) # Normalize the weights.
  cws  = jnp.cumsum(ows)
  
  N    = jnp.size(cws)
  M    = jnp.size(qs)
  
  ## Reshape to broadcast.
  cws = jnp.reshape(cws, (N, 1))
  qs  = jnp.reshape(qs, (1, M))
  
  ## Use broadcasting to get all comparisons of <cws> with each entry from <qs>.  
  ## Form tensor (cws <= qs) * 1 and sandwich index of the value vectors with 0 and 1.
  A   = jnp.concatenate([jnp.reshape(jnp.ones(M), (1,M)), (cws <= qs) * 1, jnp.reshape(jnp.zeros(M), (1,M))], axis=0)
  
  ## Get the diff -- -1 will indicate where the boundary is where cws > qs.
  X   = jnp.diff(A, axis=0).astype(int)
  
  ## Get the indices of the boundary and convert back to numpy array.
  idx = np.array(jnp.maximum(0, jnp.where(X == -1)[0] - 1))
  
  ## Return the weighted quantile value of <vs> against each <qs>.
  return(ovs[idx])


def wgt_quantiles_tensor(vs, wts, qs):
  '''
  Return a numpy matrix consisting of weighted quantile values.
  For each row of <vs>, compute all of the weighted quantiles, <qs> using weight vector, <wts>.
    
  Parameters
  ----------
  vs    A numpy(jax.numpy) (D, N) matrix of numeric values. 
  wgts  A numpy(jax.numpy) (N) array of numeric weights. (Weights need only be non-negative, they need not sum to 1.)
  qs    A numpy(jax.numpy) (M) array of numeric values.  (Meant to be quantiles -- numbers in the range [0, 1]).
  
  Returns
  -------
  A numpy array consisting of the quantile weighted values of <vs> using weights, <wts>, for each quantile in <qs>.
  
  Return-Type
  -----------
  A (D, M) numpy array of numeric values.
  
  Packages
  --------
  numpy(np)
  jax.numpy(jnp)
  
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
  
  ## 1. Are vs and wts are numpy arrays?
  if type(wts) != np.ndarray:
    raise(ValueError(  'wgt_quantiles_tensor: <wts>: Not a numpy array.'  ))
  if len(wts.shape) != 1:
    raise(ValueError('wgt_quantiles_tensor: <wts>: Not a 1-D array.'    ))
  if type(qs) != np.ndarray:
    raise(ValueError(  'wgt_quantiles_tensor: <qs>: Not a numpy array.'   )) 
  if len(qs.shape) != 1:
    raise(ValueError('wgt_quantiles_tensor: <qs>: Not a 1-D array.'     ))

  ## 2. Is vs is a numpy matrix?
  if type(vs)  != np.ndarray:
    raise(ValueError(   'wgt_quantiles_tensor: <vs>: Not a numpy array.'  ))
  if len(vs.shape) != 2:
    raise(ValueError('wgt_quantiles_tensor: <vs>: Not a numpy matrix.' ))
    
  ## 3. All qs values in [0.0, 1.0]?
  if any((qs < 0.0) | (qs > 1.0)):
    raise(ValueError('wgt_quantiles_tensor: <qs>: Not a proper quantiles array.'))
  
  ## 4. The length of vs rows and the length of wts are the same?
  if np.size(vs[0]) != np.size(wts):
    raise(ValueError("wgt_quantiles_tensor: The rows of <vs> don't have the same length as <wts>."))

  ## 5. Are all wts elements >= 0?
  if any(wts < 0.0):
    raise(ValueError('wgt_quantiles_tensor: Weights array, <wts>, has one or more negative elements.'))

  ## 6. Is sum(wts) > 0?
  if sum(wts) <= 0:
    raise(ValueError('wgt_quantiles_tensor: Sum of <wts> is not positive.'))
  
  ## Convert data to jax numpys.
  vs  = jnp.asarray(vs)
  qs  = jnp.asarray(qs)
  wts = jnp.asarray(wts)

  ## Normalize the weights.
  ws  = wts / jnp.sum(wts)
  
  D, N  = vs.shape
  M     = qs.size

  ## Get the sorted index array for each of the value vectors in vs.
  idx = jnp.argsort(vs, axis=1)
  
  ## Apply this index back to vs to get sorted values.
  ovs = jnp.take_along_axis(vs, idx, axis=1)
  
  ## Apply the index to the weights, where, the dimension of ws (and cws) expands to: (D, N).
  ows = ws[idx]
  cws = jnp.cumsum(ows, axis=1)

  ## Reshape to broadcast.
  cws = jnp.reshape(cws, (D, N, 1))
  qs  = jnp.reshape(qs, (1, 1, M))

  ## Use broadcasting to get all comparisons of <cws> with each entry from <qs>. 
  ## Form tensor (cws <= qs) * 1 and sandwich index of the value vectors with 0 and 1.
  A = jnp.concatenate([jnp.reshape(jnp.ones(M*D), (D,1,M)), (cws <= qs) * 1, jnp.reshape(jnp.zeros(M*D), (D,1,M))], axis=1)
  
  ## Compute the index difference on the value vectors.
  Delta = jnp.diff(A, axis=1).astype(int)

  ## Get the index of the values, this leaves, essentially, a (D, M) matrix. Reshape it as such.
  idx = jnp.maximum(0, jnp.where(Delta == -1)[1] - 1)
  idx = jnp.reshape(idx, (D, M)) 
  
  ## Return the values in the value vectors that correspond to these indices -- the M quantiles for each of the D value vectors.
  ## A (D, M) matrix.
  return(jnp.take_along_axis(ovs, idx, axis=1))

