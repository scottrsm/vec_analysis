## Project Intent
This project consists of analytics functions which have been vectorized in some way.
- In some cases this is done using broadcasting and array slicing from numpy.
  Examples:
    - Computing correlation/covariance matrices for securities.
    - Computing the "best" correlates of given securities from a larger universe.
    - Computing the "best k" correlates of given securities from a larger universe.
- In other cases the vectorization is done with the jax.numpy module.

version: 2.1.0

## Module Layout

### `vec_analytics.py` (numpy implementation)
- `wgt_quantiles(vs, ws, qs, chk_con=False) -> np.ndarray` -- weighted quantiles of a single 1-d value array.
- `wgt_quantiles_tensor(VS, ws, qs, chk_con=False) -> np.ndarray` -- weighted quantiles applied row-wise to a (D, N) matrix, returns a (D, M) matrix.
- `corr_cov(X, eps=1.0e-6, ws=None, corr=True, chk_con=False) -> np.ndarray` -- weighted correlation (or covariance if `corr=False`) matrix of the M rows (vectors) of X.
- `most_corr_vec(X, labs, ulabs, lab_dict, corr_type=CorrType.MOST, eps=1.0e-6, ws=None, exclude_labs=None, chk_con=False) -> pd.DataFrame` -- for each label in `labs`, the single "best" correlate from the universe `ulabs`.
- `most_corr_vecs(X, labs, ulabs, lab_dict, k, corr_type=CorrType.MOST, eps=1.0e-6, ws=None, exclude_labs=None, chk_con=False) -> pd.DataFrame` -- like `most_corr_vec`, but returns the top `k` correlates per label, best first.
- `CorrType` -- enum selecting what "best" means: `MOST` (largest), `LEAST` (smallest), `HIGH` (largest in absolute value), `LOW` (smallest in absolute value).
- `get_worst_corr(corr_type) -> float` -- sentinel value that can never be chosen as "best" for a given `CorrType`.
- `get_best_corr_idx(corr, ind, corr_type) -> np.ndarray` -- best correlate column index for the given row(s) of a correlation matrix.
- `get_best_corr_idxs(corr, ind, corr_type, k) -> np.ndarray` -- top `k` correlate column indices, best first, for the given row(s).

### `jax_vec_analytics.py` (jax variants of the quantile functions)
- `wgt_quantiles(vs, wts, qs, chk_con=True) -> jnp.ndarray` -- same definition as `vec_analytics.wgt_quantiles`, computed with `jax.numpy`; accepts numpy or jax input arrays.
- `wgt_quantiles_tensor(vs, wts, qs, chk_con=True) -> jnp.ndarray` -- jax version of `vec_analytics.wgt_quantiles_tensor`, vectorized over rows with `jax.vmap`.

### `input_contract.py` (parameter validation, used when `chk_con=True`)
- `chk_wgt_quantiles_contract(vs, wts, qs) -> None` -- contract for `wgt_quantiles`.
- `chk_wgt_quantiles_tensor_contract(VS, wts, qs) -> None` -- contract for `wgt_quantiles_tensor`.
- `chk_corr_cov_contract(X, eps, ws) -> None` -- contract for `corr_cov`.
- `check_most_corr_vec_input_contract(X, labs, ulabs, lab_dict, eps, ws, exclude_labs) -> None` -- contract for `most_corr_vec`.
- `check_most_corr_vecs_input_contract(X, labs, ulabs, lab_dict, k, eps, ws, exclude_labs) -> None` -- contract for `most_corr_vecs`.

All contract checks raise `ValueError` on failure and are opt-in (`chk_con=False` by default in `vec_analytics`, `chk_con=True` by default in `jax_vec_analytics`).

## Optional JAX Dependency
`__init__.py` imports `vec_analytics` and `input_contract` eagerly, but exposes `jax_vec_analytics` through a lazy `__getattr__` -- it is only imported (and `jax` only required) the first time code accesses `vec_analysis.jax_vec_analytics`. This means `import vec_analysis` and the entire numpy-based API (`vec_analytics`, `input_contract`) work with no `jax` installation at all; `jax` is needed only if you use the jax quantile functions.

## Dependencies
- `numpy` and `pandas` (required)
- `jax` (optional -- only needed for `jax_vec_analytics`)
- `pytest` (only needed to run the test suite)

## Usage Examples

Weighted quantiles:
```python
import numpy as np
import vec_analytics as va

vs = np.array([1.0, 4.0, 5.0, 7.0])
ws = np.array([0.25, 0.25, 0.25, 0.25])
qs = np.array([0.5])
va.wgt_quantiles(vs, ws, qs)   # -> array([4.])
```

Weighted correlation and "best correlate" lookup:
```python
import numpy as np
import vec_analytics as va

X = np.random.default_rng(0).random((5, 10))   # 5 securities, 10 observations each.
va.corr_cov(X)                                  # 5x5 weighted correlation matrix.

ulabs    = np.array(["IBM", "PFE", "C", "BAC", "GS"])
labs     = np.array(["PFE", "GS"])
lab_dict = {name: i for i, name in enumerate(ulabs)}
va.most_corr_vec(X, labs, ulabs, lab_dict)      # best correlate for PFE and GS.
```

## Tests
Run the test suite (51 tests covering `vec_analytics.py` and `input_contract.py`) with:
```
pytest test_vec_analysis.py
```

## Notebooks
- `Vec_Anaytics.ipynb` -- worked examples of weighted quantiles, weighted correlation/covariance, and finding the "most"/"least"/"high"/"low" correlated securities in a universe.
- `Find-dup-pics.ipynb` -- an unrelated utility notebook that finds and relocates duplicate JPEG images in a directory using image size and a simple averaged-pixel hash.
