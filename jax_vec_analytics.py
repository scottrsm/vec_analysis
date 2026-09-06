import numpy as np
import jax
import jax.numpy as jnp

try:
    from . import input_contract as ic
except ImportError:                      # pragma: no cover -- run as a top-level module.
    import input_contract as ic


def _as_numpy_for_checks(x):
    """Numpy view of a numpy or jax array, used only for the contract checks."""
    return np.asarray(x)


def _quantile_positions(cws, qs_sorted):
    """
    Index of the largest position whose cumulative weight is <= q (0 if none),
    for a 1-d cumulative weight array <cws>. The last cumulative weight is taken
    to be exactly 1, so q = 1 selects the largest value regardless of rounding.
    """
    cws = cws.at[-1].set(1.0)
    return jnp.maximum(0, jnp.searchsorted(cws, qs_sorted, side="right") - 1)


def wgt_quantiles(vs, wts, qs, chk_con=True):
    '''
    Get a jax array consisting of the quantile weighted <vs> values.
    Same definition as vec_analytics.wgt_quantiles: for each q, the largest value whose
    cumulative normalized weight is <= q (the smallest value if there is none).

    Parameters
    ----------
    vs    A numpy or jax (N) array of numeric values. 
    wts   A numpy or jax (N) array of numeric weights. (Weights need only be non-negative, they need not sum to 1.)
    qs    A numpy or jax (D) array of numeric values.  (Meant to be quantiles -- numbers in the range [0, 1], any order.)

    Keyword Arguments
    -----------------
    chk_con  (Optional, default True) If True, check the input contract (see input_contract.chk_wgt_quantiles_contract).

    Returns
    -------
    A jax.numpy (D) array of weighted quantile <vs> values, in the order of <qs>.

    Throws
    ------
    ValueError
    '''
    if chk_con:
        ic.chk_wgt_quantiles_contract(_as_numpy_for_checks(vs), _as_numpy_for_checks(wts), _as_numpy_for_checks(qs))

    # Convert data to jax arrays.
    vs  = jnp.asarray(vs)
    qs  = jnp.asarray(qs, dtype=float)
    wts = jnp.asarray(wts, dtype=float)

    # Sort the values and the associated weights; normalize the weights and accumulate.
    idx = jnp.argsort(vs, stable=True)
    ovs = vs[idx]
    ows = wts[idx] / jnp.sum(wts)
    cws = jnp.cumsum(ows)

    # Work with sorted quantiles, then restore the caller's order.
    perm = jnp.argsort(qs, stable=True)
    inv  = jnp.argsort(perm)
    pos  = _quantile_positions(cws, qs[perm])

    return ovs[pos][inv]


def wgt_quantiles_tensor(vs, wts, qs, chk_con=True):
    '''
    Return a (D, M) jax array of weighted quantile values: for each row of <vs>,
    all of the weighted quantiles <qs> using the weight vector <wts>.
    See wgt_quantiles for the definition.

    Parameters
    ----------
    vs    A numpy or jax (D, N) matrix of numeric values. 
    wts   A numpy or jax (N) array of numeric weights. (Weights need only be non-negative, they need not sum to 1.)
    qs    A numpy or jax (M) array of numeric values.  (Meant to be quantiles -- numbers in the range [0, 1], any order.)

    Keyword Arguments
    -----------------
    chk_con  (Optional, default True) If True, check the input contract (see input_contract.chk_wgt_quantiles_tensor_contract).

    Returns
    -------
    A (D, M) jax.numpy array, columns in the order of <qs>.

    Throws
    ------
    ValueError
    '''
    if chk_con:
        ic.chk_wgt_quantiles_tensor_contract(_as_numpy_for_checks(vs), _as_numpy_for_checks(wts), _as_numpy_for_checks(qs))

    # Convert data to jax arrays.
    vs  = jnp.asarray(vs)
    qs  = jnp.asarray(qs, dtype=float)
    wts = jnp.asarray(wts, dtype=float)

    # Normalize the weights.
    ws = wts / jnp.sum(wts)

    # Sort each row of <vs>, carrying the weights along, and accumulate the weights.
    idx = jnp.argsort(vs, axis=1, stable=True)
    ovs = jnp.take_along_axis(vs, idx, axis=1)
    cws = jnp.cumsum(ws[idx], axis=1)

    # Work with sorted quantiles, then restore the caller's order.
    perm = jnp.argsort(qs, stable=True)
    inv  = jnp.argsort(perm)
    pos  = jax.vmap(lambda c: _quantile_positions(c, qs[perm]))(cws)

    return jnp.take_along_axis(ovs, pos, axis=1)[:, inv]
