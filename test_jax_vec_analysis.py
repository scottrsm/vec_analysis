import numpy as np
import pytest as pt

jax = pt.importorskip("jax")
import jax.numpy as jnp
import vec_analytics as va
import jax_vec_analytics as jva


def test_jax_wgt_quantiles_matches_numpy():
    np.random.seed(1)
    xs = np.random.rand(500)
    ws = np.random.rand(500)
    qs = np.array([0.9, 0.1, 0.5, 0.0, 1.0])
    assert np.allclose(np.asarray(jva.wgt_quantiles(xs, ws, qs)), va.wgt_quantiles(xs, ws, qs))
    assert list(np.asarray(jva.wgt_quantiles(np.array([1., 4., 5., 7.]), np.ones(4), np.array([0.5, 0.999, 1.0])))) == [4., 5., 7.]


def test_jax_wgt_quantiles_tensor_matches_numpy():
    np.random.seed(2)
    X  = np.random.rand(6, 300)
    ws = np.random.rand(300)
    qs = np.array([0.1, 0.5, 0.9, 1.0])
    assert np.allclose(np.asarray(jva.wgt_quantiles_tensor(X, ws, qs)), va.wgt_quantiles_tensor(X, ws, qs))


def test_jax_accepts_jax_arrays_and_integer_weights():
    r = jva.wgt_quantiles(jnp.array([1., 2., 3.]), jnp.array([1, 1, 1]), jnp.array([0.5]))
    assert isinstance(r, jax.Array) and float(r[0]) == 1.0   # cumulative weights .33, .67, 1: largest <= 0.5 is 1
    r = jva.wgt_quantiles_tensor(jnp.array([[1., 2., 3.]]), np.array([1, 1, 1]), np.array([0.5, 1.0]))
    assert np.asarray(r).tolist() == [[1.0, 3.0]]


def test_jax_q_one_is_robust_to_rounding():
    # Weights whose normalized cumulative sum may not land exactly on 1.
    w = np.array([0.1, 0.2, 0.3, 0.4]) * 3.3
    assert float(jva.wgt_quantiles(np.array([1., 2., 3., 4.]), w, np.array([1.0]))[0]) == 4.0
    assert float(va.wgt_quantiles(np.array([1., 2., 3., 4.]), w, np.array([1.0]))[0]) == 4.0


def test_jax_contract():
    with pt.raises(ValueError):
        jva.wgt_quantiles(np.array([1., 2.]), np.ones(3), np.array([0.5]))
    with pt.raises(ValueError):
        jva.wgt_quantiles(np.array([1., 2.]), np.ones(2), np.array([1.5]))
    with pt.raises(ValueError):
        jva.wgt_quantiles_tensor(np.ones((2, 3)), np.array([-1., 1., 1.]), np.array([0.5]))
