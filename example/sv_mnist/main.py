import os
from pathlib import Path
import pickle
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm
from ame.sv_estimator import GameBase, AMESV
import numpy as np
import argparse


class DummyClassifier:
    def __init__(self, label) -> None:
        self.label = label

    def predict(self, X):
        return np.array([self.label]*len(X))


class MNISTGame(GameBase):
    """Implement threshold utility"""

    def __init__(self, path) -> None:
        d_train = pickle.load(Path(path).joinpath('train.pkl').open('rb'))
        d_val = pickle.load(Path(path).joinpath('val.pkl').open('rb'))
        d_test = pickle.load(Path(path).joinpath('test.pkl').open('rb'))
        self.X_train, self.y_train = d_train
        self.X_val, self.y_val = d_val
        self.X_test, self.y_test = d_test

        players = len(self.X_train)
        super().__init__(players)

    def eval(self, s: np.ndarray):
        s = s.astype(bool)
        if np.all(s == False):
            return 0
        X_train, y_train = self.X_train[s], self.y_train[s]
        net = self.train(X_train, y_train, self.X_val, self.y_val)
        return self.get_score(net)

    def train(self, X_train, y_train, X_val, y_val):
        if np.all(y_train == y_train[0]):
            clf = DummyClassifier(y_train[0])
        else:
            clf = LogisticRegression(C=50.0 / len(X_train), penalty="l2", tol=0.1,
                                     multi_class='multinomial', solver='lbfgs', n_jobs=1, max_iter=200)
            clf.fit(X_train, y_train)
        return clf

    def get_score(self, net: LogisticRegression):
        y = net.predict(self.X_test)
        return np.mean(y == self.y_test)


def run(estimator=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--phase', choices=['sample', 'ueval', 'regress'], required=True)
    parser.add_argument('--njobs', type=int, default=1)
    parser.add_argument('--jobid', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--path')
    parser.add_argument(
        '-m', help='Sample size', type=int, default=2**10)
    parser.add_argument('--eps', type=float, default=0.001)
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    eps = args.eps
    njobs = args.njobs
    jobid = args.jobid
    phase = args.phase
    seed = args.seed
    m = args.m
    print('m=', m)
    path = args.path
    if path is None:
        path = f'logs-{eps}'
    if estimator is None:
        estimator = AMESV(MNISTGame('data'), eps=eps, path=path, nthreads=2)
    path = estimator.path
    output = args.output
    if output is None:
        output = os.path.join(path, "output.npy")

    if phase == 'sample':
        if jobid == 0:
            estimator.sample(m, seed=seed)
    elif phase == 'ueval':
        estimator.eval_utility(m, njobs=njobs, jobid=jobid, seed=seed)
    elif phase == 'regress':
        res = estimator.estimate(m, seed=seed)
        print('Estimated Shapley values=')
        print(res)
        np.save(output, res)


if __name__ == '__main__':
    run()
