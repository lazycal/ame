import argparse
from ame.sv_estimator import GameBase, AMESV, ShapleyEstimator
import numpy as np


class ThresholdGame(GameBase):
    """Implement threshold utility"""

    def __init__(self, players=1000, k=3, threshold=2) -> None:
        super().__init__(players)
        self.k = k
        self.threshold = threshold

    def eval(self, S: np.ndarray):
        return (S[:self.k].sum() >= self.threshold).astype(np.float64)


def run(estimator: ShapleyEstimator):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '-m', help='Sample size', type=int, default=2**10)
    parser.add_argument('-o', '--output', default="output.npy")
    args = parser.parse_args()
    res = estimator.estimate(args.m, seed=args.seed)
    game = estimator.game
    gt = np.zeros(game.players)
    gt[:game.k] = 1/game.k
    print('l2 error=', np.sqrt(np.sum((res - gt)**2)))
    np.save(args.output, res)


if __name__ == '__main__':
    run(path="output.npy", estimator=AMESV(game=ThresholdGame(), eps=0.01))
