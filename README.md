## Project Intent
This project consists of analytics functions which have been vectorized in some way.
- In some cases this is done using broadcasting and array slicing from numpy.
  Examples:
    - Computing correlation matrix for securities.
    - Computing the "best" correlates of given securities from a larger universe.
    - Computing the "best k" correlsates of given securities from a larger universe.
- In other cases the vectorization is done with the jax.numpy module.

version: 2.0.1


