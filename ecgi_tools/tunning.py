from typing import Tuple
import numpy as np
import scipy as sp
from scipy import signal

# r is the residual term and p is the penalization term

def l_curve(r: np.ndarray, p: np.ndarray) -> Tuple:
    return np.log10(r), np.log10(p)

def u_curve(r: np.ndarray, p: np.ndarray, lambdas: np.ndarray) -> Tuple:
    y = r.copy()
    np.square(y, out=y)
    np.reciprocal(y, out=y)
    aux = np.square(p)
    np.reciprocal(aux, out=aux)
    y += aux
    return np.log10(lambdas), np.log10(y)

def creso(r: np.ndarray, p: np.ndarray, lambdas: np.ndarray) -> Tuple:
    y = np.square(p)
    y *= lambdas
    y[:] = np.gradient(y, lambdas)
    aux = np.square(r)
    aux[:] = np.gradient(aux, lambdas)
    y -= aux
    return np.log10(lambdas), y

def curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """https://www.johndcook.com/blog/2018/03/30/curvature-and-automatic-differentiation/"""
    c = np.empty_like(y)
    c[:] = np.gradient(y, x)
    aux = np.gradient(c, x)
    np.square(c, out=c)
    c += 1.
    np.power(c, -1.5, out=c)
    # np.abs(aux, out=aux)
    c *= aux
    return c

def first_local_max(y: np.ndarray) -> int:
    peaks, _ = sp.signal.find_peaks(y)
    if len(peaks) == 0:
        ix = np.argmax(y)
    else:
        ix = peaks[-1] # tomo el último max local (lambdas de mayor a menor)
    return ix