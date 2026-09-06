"""
vec_analysis: vectorized analytics (weighted quantiles, weighted correlation and
covariance, best correlates). The numpy implementation is in `vec_analytics`;
`jax_vec_analytics` holds the jax variants of the quantile functions and is
imported lazily so that jax is only required when it is used.
"""
from . import vec_analytics
from . import input_contract

__all__ = ["vec_analytics", "input_contract", "jax_vec_analytics"]


def __getattr__(name):
    if name == "jax_vec_analytics":
        from . import jax_vec_analytics
        return jax_vec_analytics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
