from abc import ABC, abstractmethod
import os
import numpy as np
from . import ame


class GameBase(ABC):
    def __init__(self, players) -> None:
        super().__init__()
        self.players = players

    @abstractmethod
    def eval(self, s: np.ndarray) -> float:
        pass

    def __call__(self, S: np.ndarray):
        batched = True
        if S.ndim == 1:
            batched = False
            S = S[None, :]
        ret = np.array([self.eval(s) for s in S])
        if not batched:
            ret = ret[0]
        return ret


class ShapleyEstimator(ABC):
    def __init__(self, game: GameBase, path=None) -> None:
        super().__init__()
        self.game = game
        self.path = path
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)

    def __str__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def sample(self, m, seed=0):
        pass

    @abstractmethod
    def eval_utility(self, m, njobs=1, jobid=0, seed=0):
        pass

    @abstractmethod
    def estimate(self, m, seed=0):
        pass


class AMESV(ShapleyEstimator):
    def __init__(self, game: GameBase, eps, path=None, dist='TU', nthreads=None) -> None:
        super().__init__(game, path)
        if dist == 'TU':
            self.ame_expl = ame.AMEExplainerPFTU(
                eps, use_knockoffs=False, use_buckets=False, lambdau='lambda_min', nthreads=nthreads)
        elif dist == 'Beta':
            self.ame_expl = ame.AMEExplainerPFBeta(
                eps, use_knockoffs=False, use_buckets=False, lambdau='lambda_min', nthreads=nthreads)
        else:
            raise ValueError(f"Unrecognized distribution {dist}")

        if path is not None:
            self.sample_folder = os.path.join(path, 'sample')
            self.utility_folder = os.path.join(path, 'utility')
            if not os.path.exists(self.sample_folder):
                os.makedirs(self.sample_folder)
                os.makedirs(self.utility_folder)

        class MyModelTrain(ame.ModelTrain):
            def __call__(self, subset: np.ndarray, idx: int):
                return game(subset)
        self.A = MyModelTrain()
        self.Q = lambda x: x

    def sample(self, m, seed):
        sample_folder = os.path.join(
            self.path, 'sample') if self.path is not None else None
        return self.ame_expl.sample(self.game.players, m, seed, sample_folder)

    def eval_utility(self, m, njobs=1, jobid=0, seed=0):
        self.ame_expl.train_and_eval_utility(
            self.game.players,
            m,
            self.sample_folder,
            None,  # don't save models to save disk
            self.utility_folder,
            self.A,
            self.Q,
            njobs,
            jobid)

    def estimate(self, m, seed=0):
        if self.path is not None:
            return self.ame_expl.regress(self.game.players, m, self.sample_folder, self.utility_folder)[0][0]
        else:
            return self.ame_expl.explain(self.A, self.Q, self.game.players, m, seed)[0][0]
