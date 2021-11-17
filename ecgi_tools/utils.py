import numpy as np
import scipy as sp
from scipy import sparse

def neighbour_distance_matrix(nodes: np.ndarray, faces: np.ndarray) -> sp.sparse.coo_matrix:
    n, _ = nodes.shape
    m, _ = faces.shape

    rows = faces.flatten(order='F')     # f1, f2, f3
    cols = np.roll(rows, shift=m)       # f3, f1, f2
    data = np.linalg.norm(nodes[rows,:] - nodes[cols,:], ord=2, axis=1)
    # data = np.reciprocal(data) if reciprocal else data

    S = sp.sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    H = (S + S.T)/2.
        
    return H

def laplace_operator(nodes: np.ndarray, faces: np.ndarray) -> sp.sparse.coo_matrix:
    H = neighbour_distance_matrix(nodes, faces) # sparse
    b = 1./H.sum(axis=1) # ndarray
    # C = neighbour_distance_matrix(nodes, faces, reciprocal=True) # sparse
    np.reciprocal(H.data, out=H.data)
    # c = C.sum(axis=1).A1 # ndarray
    c = H.sum(axis=1).A1 # ndarray
    # L = C.multiply(np.broadcast_to(b, shape=(b.size, b.size))) # sparse
    L = H.multiply(np.broadcast_to(b, shape=(b.size, b.size))) # sparse
    L.setdiag(-c*b.A1, k=0) # sparse
    L.data *= 4.
    return L

def laplace_interpolation(nodes: np.ndarray, faces: np.ndarray, measured: np.ndarray, bad_channels: np.ndarray, operator: np.ndarray = None, in_place: bool = False) -> np.ndarray:
    """
    See Oostendorp TF, van Oosterom A, Huiskamp G. Interpolation on a triangulated 3D surface. Journal of Computational Physics. 1989 Feb 1;80(2):331–43. 
    """

    L = laplace_operator(nodes, faces).tocsc() if operator is None else operator

    channels = np.arange(L.shape[0], dtype=np.int32)
    good_channels = np.delete(channels, bad_channels)

    L11 = L[np.ix_(bad_channels, bad_channels)]
    L12 = L[np.ix_(bad_channels, good_channels)]
    L21 = L[np.ix_(good_channels, bad_channels)]
    L22 = L[np.ix_(good_channels, good_channels)]
    
    f2 = np.delete(measured, bad_channels, axis=0)
    Y = -sp.sparse.vstack((L12, L22)).dot(f2)
    A = sp.sparse.vstack((L11, L21))
    f1, _, _, _ = sp.linalg.lstsq(A.toarray(), Y)

    if not in_place:
        interpolated = measured.copy()
        interpolated[bad_channels] = f1
    else:
        measured[bad_channels] = f1
        interpolated = measured

    return interpolated

def nearest_nodes(nodes: np.ndarray, ref_nodes: np.ndarray) -> np.ndarray:
    n, m = nodes.shape[0], ref_nodes.shape[0]
    if n > m:
        raise ValueError('m should be grater than n')

    nodes = np.broadcast_to(nodes[:,:,np.newaxis], shape=(n, 3, m))
    ref_nodes = np.broadcast_to(ref_nodes[:,:,np.newaxis], shape=(m, 3, n)).swapaxes(0,2)
    dists = np.linalg.norm(nodes-ref_nodes, ord=2, axis=1) # n x m

    return np.argmin(dists, axis=1)

def discrete_laplace_operator(nodes: np.ndarray, faces: np.ndarray) -> sp.sparse.coo_matrix:
    H = neighbour_distance_matrix(nodes, faces) # sparse
    H.data.fill(1)
    H.setdiag(-H.sum(axis=1), k=0)
    return H