import argparse
import pickle

import numpy as np
from ame.sv_estimator import AMESV
from main import MNISTGame
from ame.sv_baseline import *
import os


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument(
    '-m', help='Sample size', type=int, default=2**10)
parser.add_argument('--path', required=True)
parser.add_argument('--estimator', required=True)
parser.add_argument(
    '--phase', choices=["sample", "ueval", "regress"], required=True)
parser.add_argument('--njobs', type=int, default=1)
parser.add_argument('--jobid', type=int, default=0)
parser.add_argument('--gt')
args = parser.parse_args()

estimator = args.estimator
n = 1000
game = MNISTGame('data')

if estimator.startswith('TU'):
    est = AMESV(game, eps=float(
        estimator[2:]), dist='TU', path=args.path, nthreads=2)
elif estimator == 'MC':
    est = MC(game, path=args.path)
elif estimator == 'TMC':
    est = MC(game, trunc_tol=0.01, path=args.path)
elif estimator.startswith('GT'):
    eps = estimator[2:]
    if eps == 'dft':
        eps = 2 / n**0.5
    else:
        eps = float(eps)
    est = GroupTesting(game, eps=eps, path=args.path)
elif estimator.startswith('CS'):
    estimator, eps, M = estimator.split('_')
    M = 2**int(M)
    est = ComprSensing(game, eps=float(eps), path=args.path, M=M)
elif estimator == 'KSL1':
    est = KernelSHAPReg(game, path=args.path)
elif estimator == 'KS':
    est = KernelSHAPReg(game, 0., path=args.path)
elif estimator == 'PS':
    est = PairedSampling(game, path=args.path)
else:
    raise ValueError(f'Unrecognized estimator: `{estimator}`')

m = args.m
res = {}
if isinstance(est, KernelSHAPReg):
    # Cannot reuse sample since when `m` changes the sampling scheme seems to change too.
    path = est.path
    while m >= 2**10:
        est.cgame.reset()
        est.path = os.path.join(path, str(m))
        if not os.path.exists(est.path):
            os.makedirs(est.path)
        if args.phase == 'sample':
            est.sample(m, seed=args.seed)
        elif args.phase == 'ueval':
            est.eval_utility(m, seed=args.seed,
                             njobs=args.njobs, jobid=args.jobid)
        elif args.phase == 'regress':
            res[m] = est.estimate(m, seed=args.seed)
        else:
            raise ValueError("Unrecognized phase `{}`".format(args.phase))
        m //= 2
else:
    if args.phase == 'sample':
        est.sample(m, seed=args.seed)
    elif args.phase == 'ueval':
        est.eval_utility(m, seed=args.seed, njobs=args.njobs, jobid=args.jobid)
    elif args.phase == 'regress':
        while m >= 2**10:
            res[m] = est.estimate(m, seed=args.seed)
            m //= 2
    else:
        raise ValueError("Unrecognized phase `{}`".format(args.phase))
if args.phase == 'regress':
    pickle.dump(res, open(os.path.join(args.path, f"output.pkl"), "wb"))

if args.gt is not None and os.path.exists(args.gt):
    gt = pickle.load(open(args.gt, 'rb'))  # type: dict
    gt = max(gt.items(), key=lambda x: x[0])
    print('Using ground truth with m=', gt[0])
    gt = gt[1]
    for m, r in res.items():
        print('m=', m, 'l2 error=', np.sqrt(np.sum((r - gt)**2)))
