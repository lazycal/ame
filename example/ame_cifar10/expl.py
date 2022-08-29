from argparse import ArgumentParser
import math
from subprocess import check_call
from tempfile import TemporaryDirectory, TemporaryFile
from ame import ame
import os
import numpy as np
from ame.knockoff_sel import knockoff_select
import torch
from train import train_model, get_dataloader, Resnet9
import pandas as pd

poison_label = 1


class MyModelTrain(ame.ModelTrain):
    def __init__(self, k, train_data, num_workers=6) -> None:
        super().__init__()
        self.k = k
        self.train_data = train_data
        # self.num_workers = num_workers

    def __call__(self, subset, idx):
        with TemporaryDirectory(prefix="train") as folder:
            subset_f = os.path.join(folder, 'subset.npy')
            save_path = os.path.join(folder, 'model.pt')
            np.save(subset_f, subset)
            check_call(
                f"python train.py {self.k} --subset {subset_f} --seed {idx} --save_path {save_path} --verbose &> logs-20/model/train-{idx}.log",
                shell=True)
            return torch.load(save_path)
        # the following implementation has an underterministic deadlock issue in dataloader
        # train_subdata = [self.train_data[i] for i in np.where(subset)[0]]
        # return train_model(train_subdata, idx, num_workers=self.num_workers)


class MyQuery(ame.Query):
    def __init__(self, test_imgs) -> None:
        super().__init__()
        self.test_imgs = test_imgs

    def __call__(self, wts):
        model = Resnet9()
        model.init()
        model.load_state_dict(wts)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        with torch.no_grad():
            conf = torch.softmax(model.forward(self.test_imgs), dim=1)
            return conf[:, poison_label].cpu().numpy()


def print_selection(score, coef, kpls, k, n, c, path):
    prec, rec = [], []
    for qid in range(score.shape[0]):  # for each query
        print(f'-------> Selection results for query #{qid}:')
        print(f'Top {k} examples with largest positive scores:',
              np.argsort(score[qid])[::-1][:k])
        for q in [0, 0.1, 0.5]:
            selection = knockoff_select(coef[qid], n, q, kpls=kpls)
            print(
                f'Knockoff selection with desired precision {1-q}:', selection)
            if q == 0:
                num_correct = (np.array(selection) < k).sum()
                prec.append(num_correct / len(selection))
                rec.append(num_correct / k)
    pr = pd.DataFrame(
        {'qid': range(score.shape[0]), 'precision': prec, 'recall': rec})
    pr.to_pickle(path)
    print("Precision vs recall for desired precision=1:")
    print(pr)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--phase', choices=['sample', 'train_eval', 'regress', 'all'], default='all')
    parser.add_argument('--njobs', type=int, default=1)
    parser.add_argument('--jobid', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--path', default=None)
    parser.add_argument('--kpls', action='store_true')
    parser.add_argument(
        '-c', help='Will use m=clog2(n) samples', type=float, default=6)
    parser.add_argument('-k', choices=[20, 50], type=int, default=20,
                        help='which dataset to use')
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_threads', type=int, default=5,
                        help='Number of threads for lasso.')
    parser.add_argument('-v', type=str, default='INFO')
    args = parser.parse_args()

    k = args.k
    train_data = torch.load(f"data/train-{k}.pt")
    test_data = torch.load(f"data/test-poison-{k}.pt")
    if k == 50:
        to_expl_idx = [4409, 3340, 2170, 660, 609, 1612, 47, 2016, 3748,
                       4404, 1595, 1259, 1998, 2976, 2594, 3752, 578, 1397, 4191, 3524]
    else:
        to_expl_idx = [3928, 1651, 2063, 860, 1491, 4466, 1085, 2653, 2837,
                       1809, 3060, 2407, 3378, 2104, 2890, 3421, 2540, 1928, 1498, 4082]
    test_data = [test_data[i] for i in to_expl_idx]
    test_imgs, test_labels = iter(get_dataloader(test_data, False,
                                                 batch_size=len(test_data))).next()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_imgs = test_imgs.to(device)
    n = len(train_data)

    num_workers = args.num_workers
    num_threads = args.num_threads
    path = args.path
    if path is None:
        path = f'./logs-{k}'
    njobs = args.njobs
    jobid = args.jobid
    phase = args.phase
    seed = args.seed
    kpls = args.kpls
    c = args.c
    m = int(c * k * math.log2(n) + 0.5)  # assume k is known
    print('m=', m)
    sample_folder = os.path.join(path, 'sample')
    model_folder = os.path.join(path, 'model')
    utility_folder = os.path.join(path, 'utility')
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
        os.makedirs(model_folder)
        os.makedirs(utility_folder)
    ame_expl = ame.AMEExplainer(
        nthreads=num_threads, verbose=args.v)
    A = MyModelTrain(k, train_data, num_workers)
    Q = MyQuery(test_imgs)
    if phase == 'sample':
        if jobid == 0:
            ame_expl.sample(n, m, path=sample_folder, seed=seed)
    elif phase == 'train_eval':
        ame_expl.train_and_eval_utility(
            n, m, sample_folder, model_folder, utility_folder, A, Q, njobs=njobs, jobid=jobid)
    elif phase == 'regress':
        for _ in range(4):
            score, coef = ame_expl.regress(n, m, sample_folder, utility_folder)
            print_selection(score, coef, kpls, k, n, c,
                            os.path.join(path, f"pr-{c}.pkl"))
            c /= 2
            m = int(c * k * math.log2(n) + 0.5)
    else:  # run all steps end-to-end without parallelization
        for _ in range(4):
            score, coef = ame_expl.explain(A, Q, n, m, seed=seed)
            print_selection(score, coef, kpls, k, n, c,
                            os.path.join(path, f"pr-{c}.pkl"))
            c /= 2
            m = int(c * k * math.log2(n) + 0.5)


if __name__ == '__main__':
    main()
