from abc import abstractmethod
from glob import glob
import os
from pathlib import Path
import pickle
import random
from tempfile import TemporaryDirectory
import traceback
from ame.sv_estimator import GameBase, ShapleyEstimator
import numpy as np
import shap
from tqdm import tqdm
import math
from numpy import linalg as LA
import cvxpy

__all__ = [
    'KernelSHAPReg', 'MC', 'ComprSensing', 'GroupTesting', 'PairedSampling']


class CachedGame(GameBase):
    def __init__(self, game: GameBase):
        super().__init__(game.players)
        self.cache = {}
        self.freeze = False
        self.game = game

    def hash(self, s):
        return np.array(s).astype(bool).tobytes()

    def eval(self, s: np.ndarray):
        key = self.hash(s)
        if key in self.cache:
            ret = self.cache[key]
        else:
            assert not self.freeze, f'missing item \n{s}\nkey={key}'
            self.cache[key] = ret = self.game(s)
        return ret

    def dump_cache(self, path):
        pickle.dump(self.cache, open(path, 'wb'))

    def load_caches(self, paths):
        for p in paths:
            self.cache.update(pickle.load(open(p, 'rb')))
        print(f'{len(self.cache)} cache items loaded.')

    def reset(self):
        self.cache.clear()
        self.freeze = False


class _Adapter(ShapleyEstimator):
    """Adapting code written previously to the new interface.
    The older version uses CachedGame for parallelism.
    May not support incremental sampling."""

    def __init__(self, game: GameBase, path=None) -> None:
        super().__init__(game, path)
        self.cgame = CachedGame(game)

    @abstractmethod
    def run(self, game: CachedGame, m: int, st: float, ed: float, root: str, full_run: bool, seed=0):
        pass

    def sample(self, m, seed=0):
        return None

    def eval_utility(self, m, njobs=1, jobid=0, seed=0):
        st = jobid / njobs
        ed = (jobid + 1) / njobs
        self.run(self.cgame, m, st, ed, self.path, full_run=False, seed=seed)
        self.cgame.dump_cache(os.path.join(self.path, f"cache_{jobid}.pkl"))

    def estimate(self, m, seed=0):
        if self.path is None:
            with TemporaryDirectory(self.__class__.__name__) as tmp_dir:
                self.path = tmp_dir
                self.sample(m, seed)
                self.eval_utility(m, seed=seed)
                self.cgame.freeze = True
                res = self.run(self.cgame, m, 0, 1, self.path,
                               full_run=True, seed=seed)
            return res
        self.cgame.load_caches(
            list(glob(os.path.join(self.path, "cache_*.pkl"))))
        self.cgame.freeze = True
        return self.run(self.cgame, m, 0, 1, self.path, full_run=True, seed=seed)


class TracedDummyGame(GameBase):
    def __init__(self, players):
        super().__init__(players)
        self.players = players
        self.cache = []

    def eval(self, s):
        assert s.ndim == 1, s.ndim
        self.cache.append(s.astype(bool))
        return np.sum(s)


class KernelSHAPReg(_Adapter):

    def __init__(self, game, l1_reg=None, **kwargs) -> None:
        """l1_reg has the same semantic as that in shap.KernelExplainer.shap_values.
        When not specified, it uses 'auto' by default. Passing 0. means no regularization."""
        super().__init__(game, **kwargs)
        self.l1_reg = l1_reg
        self.game = game

    def __str__(self) -> str:
        return self.__class__.__name__ + f"l1_reg={self.l1_reg}"

    def sample(self, m, seed=0):
        save_path = os.path.join(self.path, 'trace.npy')
        d_g = TracedDummyGame(self.game.players)
        try:
            self.run(d_g, m, 0, 1, self.path, True, seed)
        except:
            traceback.print_exc()
        np.save(save_path, np.stack(d_g.cache))

    def run(self, game: GameBase, m: int, st: float, ed: float, root: str, full_run: bool, seed=0):
        if not full_run:  # utility evaluation
            trace = np.load(os.path.join(root, 'trace.npy'))
            mst, med = int(st * len(trace)), int(ed * len(trace))
            trace = trace[mst:med]
            for s in tqdm(trace, desc="KS ueval"):
                game(s)
            return
        assert st == 0 and ed == 1
        np.random.seed(seed)
        random.seed(seed)
        explainer = shap.KernelExplainer(game, np.zeros([1, game.players]))
        kwargs = {}
        if self.l1_reg is not None:
            kwargs['l1_reg'] = self.l1_reg
        ret = explainer.shap_values(
            np.ones([1, game.players]), nsamples=m, **kwargs)[0, :]
        return ret


class PairedSampling(_Adapter):
    def __init__(self, game, batch_size=128, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.batch_size = batch_size

    def sample(self, m, seed=0):
        save_path = os.path.join(self.path, 'trace.npy')
        d_g = TracedDummyGame(self.game.players)
        try:
            self.run(d_g, m, 0, 1, self.path, True, seed)
        except:
            traceback.print_exc()
        np.save(save_path, np.stack(d_g.cache))

    def run(self, game: GameBase, m: int, st: float, ed: float, root: str, full_run: bool, seed=0):
        if not full_run:  # utility evaluation
            trace = np.load(os.path.join(root, 'trace.npy'))
            mst, med = int(st * len(trace)), int(ed * len(trace))
            trace = trace[mst:med]
            for s in trace:
                game(s)
            return
        assert st == 0 and ed == 1
        np.random.seed(seed)
        random.seed(seed)
        batch_size = self.batch_size
        from shapreg.shapley import ShapleyRegression
        from shapreg.games import CooperativeGame

        class AdaptGame(CooperativeGame):
            def __init__(self, game: GameBase):
                self.game = game
                self.players = game.players
                pass

            def __call__(self, S):
                return game(S)
        explanation = ShapleyRegression(
            AdaptGame(game), batch_size, n_samples=m // 2, detect_convergence=False)
        return explanation.values


class MC(_Adapter):
    def __init__(self, game: GameBase, trunc_tol=None, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.trunc_tol = -1 if trunc_tol is None else trunc_tol

    def __str__(self) -> str:
        return super().__str__() + '_trunc=' + str(self.trunc_tol)

    def run_perm(self, game: CachedGame, p: np.ndarray, bar: tqdm, bound=None):
        n = game.players
        last_u = 0  # U(S)
        S = np.zeros(n)
        early_stop = False
        truncation_counter = 0
        u_full = game(np.ones(n))
        uevals = 0
        for j in p:
            S[j] = 1  # add j
            if not early_stop:
                u = game(S)  # U(S+{j})
                uevals += 1
                bar.update()
                if abs(u - u_full) <= self.trunc_tol * u_full:
                    truncation_counter += 1
                    if truncation_counter > 5:
                        early_stop = True
                        print(
                            f'Stop after {uevals} iterations (aka utility evals)')
                else:
                    truncation_counter = 0
            else:
                u = last_u
            self.res[j] += u - last_u  # U(S+{j}) - U(S)
            self.cnt[j] += 1
            last_u = u
            if bound is not None and uevals == bound:
                break

        self.stop_iters.append(uevals)
        return uevals

    def worker_run(self, game: CachedGame, m: int, seed, bound=None):
        rg = np.random.Generator(np.random.PCG64(seed))
        print('seed=', seed, 'm=', m)
        bar = tqdm(total=max(m, game.players))
        n = game.players
        i = 0
        while i < m:
            p = np.arange(n)
            rg.shuffle(p)
            print("i=", i, "p[0]=", p[0])
            i += self.run_perm(game, p, bar, bound=bound -
                               i if bound is not None else None)
        return i

    def run(self, game: CachedGame, m: int, st: float, ed: float, root: str, full_run: bool, seed=0):
        n = game.players
        assert game(np.zeros((n))) == 0, 'U(empty set) != 0'
        self.res = np.zeros(n)
        self.cnt = np.zeros(n)
        self.stop_iters = []
        tasks = []  # stores the task info (currently only mst is needed)
        if not full_run:
            mst = int(m * st)
            med = int(m * ed)
            path = os.path.join(root, f'worker_st{st}_ed{ed}.txt')
            assert not os.path.exists(
                path), f'{path} exists. Please start from a clean path'
            with open(path, 'w') as f:
                f.write(f'{mst} {med}')
            tasks = [(mst, med)]
        else:
            for i in glob(os.path.join(root, 'worker_st*_ed*.txt')):
                mst, med = map(int, open(i).read().split())
                tasks.append((mst, med))
            tasks = sorted(tasks, key=lambda x: x[0])
        for i, t in enumerate(tasks):
            print('----> running task', i)
            if full_run:  # HACK to avoid cache collision. The same subset might get different results due to randomness, making it unreproducible
                game.reset()
                game.load_caches([os.path.join(self.path, f'cache_{i}.pkl')])
                self.cgame.freeze = True
            m_t = min(m, med - mst)
            m -= self.worker_run(game, m_t,
                                 (t[0] + seed) % 2**32, bound=m if full_run else None)
            if m <= 0:
                break
        assert not full_run or m == 0, f'Need {m} more samples. Consider re-running eval_utility.'

        for i in range(n):
            if self.cnt[i] == 0:
                self.cnt[i] = 1
        self.res /= self.cnt
        print('Average truncation ratio:', np.mean(self.stop_iters) / n)
        return self.res


class GroupTesting(_Adapter):
    def __init__(self, game, eps=None, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.eps = eps

    def __str__(self) -> str:
        return 'GT' + str(self.eps)

    def run(self, game: GameBase, m: int, st: float, ed: float, root: str, full_run: bool, seed=0):
        n = game.players
        if self.eps is None:
            self.eps = 2 / n**0.5
        epsilon = self.eps / (2 * n**0.5)
        Z = 2*np.sum([1/i for i in range(1, n)])  # Z
        q = [1 / Z * (1 / k + 1 / (n - k)) for k in range(1, n)]  # q(k)
        # in paper theorem-3, the number of tests T should satisfies this number T_new
        # but here, use an input m
        # def h(x):
        #     y = (1+x)*np.log(1+x) - x
        #     return y
        # q_tot = q[0]
        # for j in range(1, N-1):
        #     k = j + 1
        #     q_tot += q[j] * (1 + 2 * k * (k - N) / (N*(N-1)))
        # T_new = 4/(1-q_tot**2)/h(2*epsilon/Z/r/(1-q_tot**2))*np.log(N*(N-1)/(2*self.delta))
        # m = T_new
        A = np.zeros((m, n))  # A.shape=TxN
        B_tst = np.zeros((m,))  # B_tst.shape=(T,)
        mst, med = int(m * st), int(m * ed)
        for t in tqdm(range(mst, med)):
            rg = np.random.Generator(np.random.PCG64((t + seed * m) % 2**32))
            num_active_users = rg.choice(np.arange(1, n), 1, False, q)
            active_users_ind = rg.choice(np.arange(n), num_active_users, False)
            A_t = np.zeros(n)
            A_t[active_users_ind] = 1
            A[t] = A_t
            # B_tst_t, w_tst_t = game(active_users_ind, self.regularizer, self.max_loss)
            S = A_t
            B_tst_t = game(S)
            B_tst[t] = B_tst_t
        S = np.ones(n)  # all x_trn, y_trn
        u_tot = game(S)
        if not full_run:
            return
        C = {}
        for i in tqdm(range(n), desc="Ci"):
            for j in range(i+1, n):
                C[(i, j)] = Z/m*(B_tst.dot(A[:, i]-A[:, j]))
        s = cvxpy.Variable(n)
        constraints = [cvxpy.sum(s) == u_tot]
        for i in tqdm(range(n), desc='appending'):
            for j in range(i+1, n):
                constraints.append(s[i]-s[j] <= epsilon + C[(i, j)])
                constraints.append(s[i] - s[j] >= -epsilon + C[(i, j)])

        prob = cvxpy.Problem(cvxpy.Minimize(0), constraints)
        print('Solving...')
        result = prob.solve(solver=cvxpy.SCS, verbose=True)
        print('Solved')
        s_opt = s.value

        return s_opt


class ComprSensing(_Adapter):
    def __init__(self, game, eps=None, base=0, M=None, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.epsilon = eps
        self.base = base
        self.M = M

    def run(self, game: GameBase, m: int, st: float, ed: float, root: str, full_run: bool, seed=0):
        n = game.players
        rng = np.random.Generator(np.random.PCG64(seed))
        rngM = np.random.Generator(np.random.PCG64(seed))
        if "CS_SEED" in os.environ:
            cs_seed = int(os.environ['CS_SEED'])
            print('[CS] Using CS_SEEEd=', cs_seed)
            rngM = np.random.Generator(np.random.PCG64(cs_seed))
        base = self.base
        epsilon = self.epsilon
        print('[CS] Epsilon=', epsilon)
        print('[CS] base=', base)
        if epsilon is None:
            epsilon = 2 / n**0.5
        T = int(math.ceil(m / n))
        # M = int(np.exp(T / (2*r**2) * epsilon**2) * delta / 4)
        M = self.M or 2048
        print('[CS] M=', M)
        # print(f'Compressive sensing: T={T} M={M}, time compilexity={M*T*n}')
        # Sample a Bernoulli matrix A, where Am,i∈{−1/sqrt.M, 1/sqrt.M} with equal probability;
        A = rngM.binomial(1, 0.5, (M, n))
        A = (A-0.5)*2*(1/math.sqrt(M))  # dimension: MxN
        y = []  # dimension: TxM
        tst, ted = int(T * st), int(T * ed)
        bar = tqdm(total=ted - tst)
        for t in range(T):
            # generate uniform random permutation
            permutation = rng.permutation(n)  # a random permutaion of index
            if t < tst:
                continue
            if t >= ted:
                break
            U_out = np.empty((n, 1))  # dimension: Nx1
            S = np.zeros(n)
            last_u = game(S)
            for i in range(len(permutation)):
                S[permutation[i]] = 1
                U_include = game(S)
                U_exclude = last_u
                last_u = U_include
                U_out[permutation[i]] = U_include - U_exclude
            bar.update()
            y_m_t = np.matmul(A, np.array(U_out))  # dimension: Mx1
            y.append(y_m_t)
        if not full_run:
            return
        y_m = np.mean(np.array(y), axis=0)  # dimension: Mx1
        s_bar = base
        # delta_s_out = argmin(delta_s) ||delta_s||_1, s.t. ||A*(s_bar+delta_s)-y_m||_2 <= epsilon

        delta_s = cvxpy.Variable(n)
        constraints = [cvxpy.norm2(cvxpy.matmul(
            A, s_bar + delta_s) - y_m[:, 0]) <= self.epsilon]

        prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.norm1(delta_s)), constraints)
        try:
            result = prob.solve(verbose=True)
        except:
            traceback.print_exc()
            return None
        delta_s_opt = delta_s.value

        s_hat = s_bar+delta_s_opt  # dimension: Nx1
        t = LA.norm(np.matmul(A, s_hat) - y_m[:, 0])
        if t > epsilon + 1e-4:
            print('[CS][WARNING] Solving failed.')
        return s_hat
