# ecgi_tools
Some basics tools to work in ECGI (Electrocardiographic Imaging).

## Modules

1. **inverse.py** contains the following functions:
    * `explicit_zot`: compute Zero Order Tikhonov solutions from an array of hyperparameters based on explicit SVD computation.
    * `explicit_hot`: compute High Order Tikhonov solutions from an array of hyperparameters based on explicit GSVD computation (requiere LAPACK routine from https://github.com/ddrake/pygsvd).
    * `lambdas_zot`: compute a range of hyperparameters for ZOT case based on singular values of the matrix model. 
    * `residual_terms`: compute the residual terms for all hyperparameters.
    * `penalty_terms`: compute the penalty terms for all hyperparameters.
2. **tunning.py** contains functions to compute the L, U and CRESO curves taking residual and penalty terms as entries. Additionally, a `curvature` function to corner estimation of L-curve and a `first_local_max` function for CRESO.
3. **modelling.py** Only contains the `bem_model` function. This return the transfer matrix of the model using bempp library (http://bempp.com/). 
4. **utils.py** contains some functions as `laplace_operator` and `discrete_laplace_operator` to be used with HOT and `laplace_interpolation` to recovery bad channels. Finally, a `nearest_nodes` function to identify which electrodes are near to the mesh nodes (useful to validation).

## Installation
First setup your venv and activate it, then run
```
pip install git+https://github.com/sfcaracciolo/ecgi_tools.git
pip install git+https://github.com/ddrake/pygsvd.git
```
The package need scipy and pygsvd except for modelling module which requiere bempp and your dependeces (meshio & numba).

## Contact
Please, if you use this fragment, contact me at scaracciolo@conicet.gov.ar

## Notes
Laplace interpolation in this repo is coded with sparse matrices (scipy.sparse) so is better than the repo https://github.com/sfcaracciolo/laplace_interpolator which is based on dense numpy.ndarray.