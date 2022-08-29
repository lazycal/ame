import numpy as np


def Z_FUNC(x):
    return max(x, 0)


def sorted_idx(coef):
    n = len(coef)
    return sorted(list(range(n)), key=lambda x: -coef[x])


def select(coef, thres, n, kff=True):
    if kff:
        w = [Z_FUNC(coef[i]) - Z_FUNC(coef[i+n]) for i in range(n)]
    else:
        w = coef[:n]
    return [i for i in sorted_idx(w) if w[i] >= thres]


def calc_fdr(w, t, kpls):
    a = (w >= t).sum()
    return ((1. if kpls else 0) + ((w <= -t).sum())) / a if a > 0 else 1e100


def calc_thres(w, q, kpls):
    w_abs = np.unique(sorted(map(abs, w)))
    if w_abs[0] == 0:
        w_abs = w_abs[1:]
    for t in w_abs:
        fdr = calc_fdr(w, t, kpls)
        if fdr <= q:
            return t, fdr
    return np.nan, np.nan


def knockoff_select(coef: np.ndarray, n, q=0, kpls=False):
    """Use knockoff to select with guaranteed false discovery rate `q`.

    Args:
        coef (np.ndarry): An array of length n*2+nbuc, storing the LASSO coefficients
        n (int): number of data points to select from
        q (float): desired false discovery rate
        kpls (bool): whether to use Knockoff+ or not. Knockoff+ is a more conservative
            procedure than Knockoff, but may not work well when there are too few data
            with high contribution. Consider switching to Knockoff in that case.

    Returns:
        list: ids of selected data points
    """

    assert coef.ndim == 1, coef.shape
    w = np.array([Z_FUNC(coef[i]) -
                  Z_FUNC(coef[i+n]) for i in range(n)])
    thres, fdr = calc_thres(w, q, kpls)
    sel = select(coef, thres, n)
    return sel
