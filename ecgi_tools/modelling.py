import bempp.api
import numpy as np
import scipy as sp
from scipy import sparse

def bem_geometry(nodes0, faces0, nodes1, faces1):
    """
    * Surfaces must have outpointing normales and be closed.
    * Surface 0 is outer Surface 1 is inner.
    * Dimensions: 3 x n0, 3 x m0, 3 x n1, 3 x m1, 
    """

    # greometry compound
    nodes = np.hstack((nodes0, nodes1))
    faces = np.hstack((faces0, faces1 + nodes0.shape[1]))
    n, m = nodes.shape[1], faces.shape[1]

    # domain ids: 0 outer, 1 inner
    ix = np.zeros(m, dtype=np.uint32)
    ix[faces0.shape[1]:] = 1

    # grid = bempp.api.shapes.sphere(h=0.1)
    grid = bempp.api.Grid(nodes, faces, domain_indices=ix)

    # bempp.api.PLOT_BACKEND = "gmsh"
    # grid.plot()
    return grid

def bem_model(grid):
    """Build the BEM matrix. Implemented from [1]
    [1] Stenroos*, Matti, and Jens Haueisen. “Boundary Element Computations in the Forward and Inverse Problems of Electrocardiography: Comparison of Collocation and Galerkin Weightings.” IEEE Transactions on Biomedical Engineering 55, no. 9 (September 2008): 2124–33. https://doi.org/10.1109/TBME.2008.923913.
    """

    # http://bempp.com/handbook/api/function_spaces.html
    sp0 = bempp.api.function_space(grid, "P", 1, segments=[0]) # known
    sp1 = bempp.api.function_space(grid, "P", 1, segments=[1]) # unknown

    # Operators. notations [field][source] = [dual][domain]
    # https://groups.google.com/g/bempp/c/0Yr5za_YKvI


    ident1 = bempp.api.operators.boundary.sparse.identity(sp1, sp1, sp1)
    dl1 = bempp.api.operators.boundary.laplace.double_layer(sp1, sp1, sp1)
    dl10 = bempp.api.operators.boundary.laplace.double_layer(sp0, sp1, sp1)
    sl1 = bempp.api.operators.boundary.laplace.single_layer(sp1, sp1, sp1)
    ident0 = bempp.api.operators.boundary.sparse.identity(sp0, sp0, sp0)
    dl0 = bempp.api.operators.boundary.laplace.double_layer(sp0, sp0, sp0) 
    dl01 = bempp.api.operators.boundary.laplace.double_layer(sp1, sp0, sp0)
    sl01 = bempp.api.operators.boundary.laplace.single_layer(sp1, sp0, sp0)

    # weak and strong return same results
    AD_B = (.5 * ident0 + dl0).weak_form().to_dense() 
    AD_H = (.5 * ident1 - dl1).weak_form().to_dense()
    G_HH = sl1.weak_form().to_dense()
    D_BH = dl01.weak_form().to_dense()
    D_HB = dl10.weak_form().to_dense()
    G_BH = sl01.weak_form().to_dense()

    G_HH[:] = np.linalg.inv(G_HH)
    aux = G_BH @ G_HH
    AD_B -= aux @ D_HB
    AD_B[:] = np.linalg.inv(AD_B)
    D_BH += aux @ AD_H
    L = AD_B @ D_BH

    return L
