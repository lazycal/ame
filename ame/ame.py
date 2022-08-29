from abc import ABC, abstractmethod
from argparse import ArgumentParser
from glob import glob
import logging
import os
from pathlib import Path
import pickle
from typing import Any, Iterable, List, Tuple, Union
import numpy as np
from tqdm import tqdm
from . import regression
import math
from functools import reduce


def binseq2int(n, bin: Iterable):
    return reduce(lambda x, y: x | y, [
        (int(bin[j]) << j) for j in range(n)])


def int2binseq(n, i: int):
    bin = np.zeros(n).astype(np.bool8)
    bin[regression.get_ones_pos(i)] = True
    return bin


class ModelTrain(ABC):
    @abstractmethod
    def __call__(self, subset: np.ndarray, idx: int) -> Any:
        """Train a submodel on the specified subset and return the model. 
        It is recommended to train each submodel with the same number of iterations (i.e. gradient steps) rather than epochs, to avoid the loss from not yet being able to converge on small subsets due to insufficient gradient steps.

        Args:
            `subset` (np.ndarray): a 0-1 boolean array of size N. 
            `idx` (int): indicates which model currently is to be trained

        Returns:
            Any: Trained model that will be passed to a Query
        """
        pass


class Query(ABC):
    @abstractmethod
    def __call__(self, model: Any) -> Union[float, Iterable[float]]:
        """Quantify the behavior of the specified model"""
        pass


class AMEExplainer:
    def __init__(self, pval=[0.2, 0.4, 0.6, 0.8], lambdau='lambda_1se', verbose='INFO', nthreads=None, use_buckets=True, use_knockoffs=True) -> None:
        self.nthreads = nthreads or 5
        self.lambdau = lambdau
        self.pval = np.array(pval)
        self.logger = logging.getLogger('AME')
        self.logger.setLevel(getattr(logging, verbose))
        import sys
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d][%(name)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s',
            datefmt="%H:%M:%S")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.use_buckets = use_buckets
        self.use_knockoffs = use_knockoffs

    def sample_p(self, rng: np.random.Generator, size=None):
        return rng.choice(self.pval, size=size)

    def sample(self, n, m, seed=0, path=None) -> Tuple[List[int], np.ndarray]:
        """Sampling subsets"""
        rng = np.random.default_rng(seed)
        # The experiments in the paper use the following two lines instead
        # np.random.seed(seed)
        # rng = np.random
        if path is not None:
            ps_path = os.path.join(path, f'ps.npy')
            codes_path = os.path.join(path, f'subsets.pkl')

        codes, ps = [], []
        self.logger.info(f'design matrix size={m*n}')  # combs.nbytes)
        for i in tqdm(range(m)):
            code = 0
            p = self.sample_p(rng)
            code = binseq2int(n, rng.binomial(1, p, n).astype(np.bool8))
            codes.append(code)
            ps.append(p)
        ps = np.array(ps)
        if path is not None:
            self.logger.info(f'saving {ps_path}')
            if os.path.exists(ps_path):
                old_ps = np.load(ps_path)
                new_sz = min(len(old_ps), len(ps))
                assert np.all(old_ps[:new_sz] == ps[:new_sz]
                              ), 'Incremental sampling failed. The current sampling seems to be using a different seed than previous sampling'
                old_codes = pickle.load(open(codes_path, 'rb'))
                assert old_codes[:new_sz] == codes[:new_sz]
            np.save(ps_path, ps)
            self.logger.info(f'saving {codes_path}')
            pickle.dump(codes, open(codes_path, 'wb'))
        return codes, ps

    @staticmethod
    def train_and_eval_utility(n, m, sample_folder, model_folder, utility_folder, A, Q, njobs, jobid):
        """Train & save models and utilities. Only work on the `jobid`-th slice from evenly partitioned `njobs` slices. 
        When the current model already exists in `model_folder`, its training step will be skipped.
        When `model_folder=None`, the models will not be saved"""
        if model_folder is not None and not os.path.exists(model_folder):
            os.makedirs(model_folder)
        subsets = pickle.load(
            open(os.path.join(sample_folder, f'subsets.pkl'), 'rb'))[jobid:m:njobs]
        utility_folder = Path(utility_folder)
        utility_folder.mkdir(exist_ok=True, parents=True)
        yidx_exist = set()
        for path in utility_folder.glob('yidx*'):
            yidx_exist |= set(np.load(path))

        yval = []
        yidx = []
        for i, s in tqdm(enumerate(subsets), desc='Train & evaluate utilities', total=len(subsets)):
            idx = i * njobs + jobid
            if idx in yidx_exist:  # do not recompute
                continue
            model_path = os.path.join(
                model_folder, f'model-{idx}.pkl') if model_folder is not None else None
            if model_path is None or not os.path.exists(model_path):
                model = A(int2binseq(n, s), idx)
                if model_folder is not None:
                    pickle.dump(model, open(model_path, 'wb'))
            else:
                model = pickle.load(open(model_path, 'rb'))
            yidx.append(idx)
            yval.append(Q(model))
        if len(yidx) == 0:
            return
        yidx = np.stack(yidx)
        yval = np.stack(yval)
        if yval.ndim == 1:
            yval = yval[:, None]
        # incremental
        yval_path = utility_folder/f'yval-worker{jobid}.npy'
        yidx_path = utility_folder/f'yidx-worker{jobid}.npy'
        if yidx_path.exists():
            yidx = np.concatenate([yidx, np.load(yidx_path)], axis=0)
            yval = np.concatenate([yval, np.load(yval_path)], axis=0)
        np.save(yidx_path, yidx)
        np.save(yval_path, yval)

    def _regress(self, subsets, n, ps, yss: np.ndarray, knockoff_seed):
        if yss.ndim == 1:
            yss = yss[:, None]
        return regression.regress_batch(
            subsets, n, ps, yss, self.featurize_func, use_knockoffs=self.use_knockoffs, use_buckets=self.use_buckets, lambdau=self.lambdau, seed=knockoff_seed, nthreads=self.nthreads)

    def regress(self, n, m, sample_folder, utility_folder, knockoff_seed=1):
        subsets = pickle.load(
            open(os.path.join(sample_folder, 'subsets.pkl'), 'rb'))
        ps = np.load(os.path.join(sample_folder, 'ps.npy'))
        yss = None  # m x num_queries. type: np.ndarray
        utility_folder = Path(utility_folder)
        for path in utility_folder.glob('yidx*'):
            yidx = np.load(str(path))
            yval = np.load(path.with_name(path.name.replace('yidx', 'yval')))
            if yss is None:
                yss = np.empty(shape=(len(subsets), yval.shape[1]))
                yss[:] = np.nan
            yss[yidx] = yval
        yss = yss[:m]
        assert np.all(~np.isnan(yss)), 'NaN found in utilities at ' + \
            str(np.where(np.isnan(yss))[0])
        coef = self._regress(subsets[:m], n, ps[:m], yss, knockoff_seed)
        coef /= self.coef_scaling()
        score = coef[:, :n]
        return score, coef

    def explain(self, A: ModelTrain, Q: Query, n, m, seed=0, knockff_seed=None):
        """Executes all the steps above without parallelization/storing files. 
        Can be handy when the overall computation is lightweight."""
        if knockff_seed is None:
            knockff_seed = seed + 1
        subsets, ps = self.sample(n, m, seed)
        yss = []
        for i, s in enumerate(tqdm(subsets, desc='Evaluating utilities')):
            yss.append(Q(A(int2binseq(n, s), i)))
        yss = np.stack(yss)
        if yss.ndim == 1:
            yss = yss[:, None]
        coef = self._regress(subsets[:m], n, ps[:m], yss[:m], knockff_seed)
        coef /= self.coef_scaling()
        score = coef[:, :n]
        return score, coef

    def featurize_func(self, x, p):
        # We do slightly different from what is stated in the paper. The featurization
        # does not ensure unit variance, so we will rescale the coefficients to compensate.
        return 1 / (p + x.astype(np.float64) - 1)  # x==1: 1/p; x==0: -1/(1-p)

    def coef_scaling(self):
        return (1/self.pval/(1-self.pval)).mean()


class AMEExplainerPFTU(AMEExplainer):
    """Use p-featurization that chooses from p | 1-p, and truncated uniform(eps,1-eps).
    Mainly for being used as a Shapley value estimator."""

    def __init__(self, eps, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def sample_p(self, rng: np.random.Generator, size=None):
        return self.p_cdfinv(rng.uniform(0, 1, size=size))

    def cdf(self, p):
        return np.log(p)-np.log(1-p)

    def cdfinv(self, y):
        return 1/(np.exp(-y)+1)

    def p_cdfinv(self, y):
        eps = self.eps
        d = self.cdf(1-eps)-self.cdf(eps)
        t = d*y+self.cdf(eps)
        return self.cdfinv(t)

    def featurize_func(self, x, p):
        return x.astype(np.float64) - p  # x==1: 1-p; x==0: -p

    def coef_scaling(self):
        return 1


class AMEExplainerPFBeta(AMEExplainer):
    """Use p-featurization that chooses from p | 1-p, and beta(1+eps, 1+eps).
    Mainly for being used as a Shapley value estimator"""
    pass  # TODO
