import gc
import logging
import traceback
import numpy as np
import glmnet_python
from glmnet import glmnet
from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef
from glmnetCoef import glmnetCoef


def get_ones_pos(code):
    s = '{:b}'.format(code)
    res = [i for i, c in enumerate(s[::-1]) if c == '1']
    return res


def encode_subsets(S, n, ps, featurize_func):
    m = len(S)
    Xs = np.zeros((m, n), np.float64)
    for i, code in enumerate(S):
        Xs[i][get_ones_pos(code)] = 1
        Xs[i] = featurize_func(Xs[i], ps[i])
    return Xs


def regress_batch(subsets, n, ps, yss, featurize_func, use_knockoffs, use_buckets, lambdau, seed=1, nthreads=5):
    logger = logging.getLogger('AME')
    Xs = encode_subsets(subsets, n, ps, featurize_func)
    if use_knockoffs:
        Xs = knockoff(Xs, seed=seed, ps=ps, featurize_func=featurize_func)
    if use_buckets:
        pval = list(np.unique(ps))
        nbuc = len(pval)
        logger.info(f'Bucketing with NUM_BUC={nbuc}')
        p_feature = np.zeros([Xs.shape[0], nbuc])
        assert nbuc > 0, nbuc
        for i in range(len(ps)):
            p_ind = pval.index(ps[i])
            p_feature[i][p_ind] = 1
        Xs = np.append(Xs, p_feature, 1)
    else:
        nbuc = 0
    logger.debug(f"Xs[:10]={Xs[:10]}")
    logger.debug(f"Xs[:10]>0={(Xs[:10]>0).sum(axis=1)}")
    logger.debug(f"ps[:10]={ps[:10]}")
    logger.debug(f"yss[:10]={yss[:10, 0]}")
    gc.collect()

    bat_ames = []
    for i in range(yss.shape[1]):
        ames = regress(
            Xs, yss[:, i], nbuc, lambdau, nthreads=nthreads)
        bat_ames.append(ames)
    return np.stack(bat_ames)


def knockoff(Xs, ps, featurize_func, seed=None):
    if seed is not None:
        rg = np.random.default_rng(seed)
    assert Xs.ndim == 2, Xs.ndim
    assert len(ps) == Xs.shape[0]
    Xs1 = np.zeros_like(Xs)
    for i, p in enumerate(ps):
        Xs1[i] = rg.binomial(1, p, Xs.shape[1])
        Xs1[i] = featurize_func(Xs1[i], p)
    return np.concatenate([Xs, Xs1], axis=1)


def regress(Xs, ys, nbuc, lambdau, nfolds=20, nthreads=5):
    logger = logging.getLogger('AME')
    n = Xs.shape[1]
    pfac = np.ones(shape=[1, n])
    n_pen = n - nbuc
    logger.info(f'n_pen={n_pen}, n={n}, m={len(Xs)}, nbuc={nbuc}')
    if nbuc > 0:
        logger.info(f'not penalyzing last {nbuc} vars')
        pfac[0, -nbuc:] = 0
    assert lambdau in ['lambda_min', 'lambda_1se'], lambdau
    try:
        cvfit = cvglmnet(x=Xs,
                         y=ys.astype('float64').copy(),
                         ptype='mse',
                         nfolds=min(len(Xs), nfolds),
                         parallel=nthreads,
                         alpha=1,
                         penalty_factor=pfac,
                         intr=nbuc == 0,
                         )
        a = np.array(cvglmnetCoef(cvfit, s=lambdau))
        assert n == a.shape[0] - 1, (n, a.shape[0])
        ret = a[1:, -1]
        del cvfit
    except Exception:
        logger.info("LASSO failure:")
        logger.info("-"*60)
        traceback.print_exc()
        logger.info("-"*60)
        ret = np.zeros(n)

    return ret
