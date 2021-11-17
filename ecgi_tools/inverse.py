import numpy as np
import pygsvd
import scipy as sp
from scipy import sparse

# @profile
def explicit_hot(A: np.ndarray, B: np.ndarray, b: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """ High Order Tikhonov
    See book 'Matrix Computations' of Van Loan. Chapters 6.1.5 and 6.1.6 and https://github.com/ddrake/pygsvd
    """
    m1, n1 = A.shape
    m2, n2 = B.shape
    m3, n3 = b.shape

    if m2 != n1 or n2 != n1 or m3 != m1 :
        raise ValueError('Mismatch dimensions.')

    da, db, X, U, _ = pygsvd.gsvd(A, B, full_matrices=True, extras='uv', X1=True)
    Da = sp.sparse.dia_matrix((da, 0), shape=A.shape, dtype=np.float32)
    Db = sp.sparse.dia_matrix((db, 0), shape=B.shape, dtype=np.float32)
    Y = Da.T * ( U.T @ b ) # n1 x m1 . m1 x n3 = n1 x n3
    dDa = ( Da.T * Da ).diagonal()
    dDb = ( Db.T * Db ).diagonal()

    Z = np.empty((n1, lambdas.size), dtype=np.float32) # diagonals per column.
    Z[:] = np.broadcast_to(dDa[:, np.newaxis], shape=(n1, lambdas.size)) + np.outer(dDb, lambdas)  # broadcasting
    np.reciprocal(Z, out=Z)

    Yb = np.broadcast_to(Y[:,:,np.newaxis], shape=(n1, n3, lambdas.size))
    Zb = np.broadcast_to(Z[:,np.newaxis,:], shape=(n1, n3, lambdas.size))
    W = Zb * Yb
    m = np.tensordot(X, W, axes=[[1], [0]])

    return m

def explicit_zot(A: np.ndarray, b: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """ Zero Order Tikhonov
    """
    m1, n1 = A.shape
    m3, n3 = b.shape

    if m3 != m1 :
        raise ValueError('Mismatch dimensions.')

    U, s, Vt = sp.linalg.svd(A)
    S = sp.sparse.dia_matrix((s, 0), shape=A.shape, dtype=np.float32)
    Y = S.T * (U.T @ b) # n1xm1 * m1xm1 * m1xn3 = n1xn3
    ds = (S.T * S).diagonal()

    Z = np.empty((n1, lambdas.size), dtype=np.float32) # diagonals per column.
    Z[:] = ds[:, np.newaxis] + lambdas[np.newaxis, :] # broadcasting
    np.reciprocal(Z, out=Z)
    
    Yb = np.broadcast_to(Y[:,:,np.newaxis], shape=(n1, n3, lambdas.size))
    Zb = np.broadcast_to(Z[:,np.newaxis,:], shape=(n1, n3, lambdas.size))
    X = Zb * Yb
    m = np.tensordot(Vt.T, X, axes=[[1], [0]])
    return m

def lambdas_zot(A: np.ndarray, points: int, epsilon : float) -> np.ndarray:
    lambd_max = np.linalg.norm(A, ord=2)/epsilon
    lambd_min = np.linalg.norm(A, ord=-2)*epsilon
    return np.logspace(np.log10(lambd_min), np.log10(lambd_max), num=points, endpoint=True)[::-1]

def residual_terms(A: np.ndarray, X:np.ndarray, Y:np.ndarray) -> np.ndarray:
    r = np.tensordot(A, X, axes=[[1], [0]]) - Y[:, :, np.newaxis]
    return np.linalg.norm(r, ord='fro', axis=(0, 1))

def penalty_terms(X: np.ndarray)-> np.ndarray:
    return np.linalg.norm(X, ord='fro', axis=(0, 1))
