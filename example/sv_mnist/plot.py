from glob import glob
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def l2_error(x, y):
    return np.sum((x-y)**2)**0.5


def load_data(patterns, gt=None):
    dfs = []
    for pattern in patterns:
        name, path_pattern = pattern
        paths = glob(path_pattern)
        assert len(paths) > 0, path_pattern
        for path in paths:
            profile = pickle.load(open(path, 'rb'))
            df = pd.DataFrame({'m': list(profile.keys()),
                              'result': list(profile.values())})
            df['name'] = name
            df['path'] = path
            dfs.append(df)

    df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    if gt is None:
        df1 = df[df.name == 'MC']
        gt = df1[df1.m == max(df1.m)]['result'].iloc[0]
    for i in df.index:
        df.loc[i, 'l2_err'] = l2_error(df.loc[i, 'result'], gt)

    df = df.sort_index()
    return df


def plot(df, figsize=(6, 4), ylim=[0, 0.5], use_order=True, names=None, exclude=['MC'], m_ed=2**19):
    plt.figure(figsize=figsize)
    df = df[['name', 'l2_err', 'm']]
    df = df[~df.name.isin(exclude)]
    data = df[(df.m >= 1024) & (df.m <= m_ed)].copy()
    sns.set_theme(font_scale=1.42, style='whitegrid')
    kwargs = {'markers': True}
    if use_order:
        for n in data.name:
            assert n in names, n
        kwargs.update({'style_order': names, 'hue_order': names})
    ax = sns.lineplot(data=data, x='m', y='l2_err',
                      hue='name',
                      style='name', markersize=13,
                      **kwargs
                      )
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = zip(
        *[(h, l) for h, l in zip(handles, labels) if l in data.name.tolist()])
    for h in handles:
        h.set_markersize(13)
        h.set_markeredgewidth(0.75)
        h.set_linewidth(1.5)
        h.set_markeredgecolor('w')
    ax.legend(handles[:], labels[:], handletextpad=0, columnspacing=0)
    ax.set_ylim(ylim)
    ax.set_xscale('log')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('L2-Norm Error')
    tm = ax.get_xlim()
    ax.set_xlim(tm)
    return ax, handles, labels


def CS_best_M(path, gt):
    path = Path(path)
    best_M, best_error = {}, {}
    for M in range(7, 17):
        error = {}
        for p in path.glob(f'CS_0.01_{M}/run-*/output.pkl'):
            for m, v in pickle.load(p.open('rb')).items():
                if m not in error:
                    error[m] = []
                error[m].append(l2_error(v, gt))
        for m in error:
            error[m] = np.mean(error[m])
        print(f"error for M={M}", error)
        for m, v in error.items():
            if m not in best_M or best_error[m] > v:
                best_M[m] = M
                best_error[m] = np.mean(v)
    print('best M=', best_M)
    print('best error=', best_error)

    seed = 0
    p = path/'CS_0.01_bestM'/f'run-{seed}'
    p.mkdir(parents=True, exist_ok=True)
    output = {}
    for m, M in best_M.items():
        ame = pickle.load(
            (path/f'CS_0.01_{M}/run-{seed}/output.pkl').open('rb'))[m]
        output[m] = ame
    pickle.dump(output, (p/'output.pkl').open('wb'))


root = 'logs'
n = 1000
k = 3
gt = pickle.load(open(f'{root}/MC/run-0/output.pkl', 'rb'))[2**22]
CS_best_M(root, gt)
path = [
    ('Compr. Sensing', f'{root}/CS_0.01_bestM/run-*/output.pkl'),
    ('KernelSHAP (L1)', f'{root}/KSL1/run-*/output.pkl'),
    ('KernelSHAP', f'{root}/KS/run-*/output.pkl'),
    (r'AME $\epsilon=0.001$', f'{root}/TU0.001/run-*/output.pkl'),
]
df = load_data(path, gt=gt)
save_fig = f'{root}/fig.png'
Path(save_fig).parent.mkdir(exist_ok=True)
names = [r'AME $\epsilon=0.01$', r'AME $\epsilon=0.001$',
         'Compr. Sensing', 'KernelSHAP (L1)', 'KernelSHAP', 'Paired Sampling']
ax, handles, labels = plot(df, figsize=(
    6, 4), names=names, ylim=[0, 0.25], m_ed=2**21)
ax.legend(handles[:], labels[:], handletextpad=0, columnspacing=0,
          ncol=1)
plt.tight_layout(pad=0)
if save_fig:
    ax.get_figure().savefig(save_fig)

path = [
    ('MC', f'{root}/MC/run-*/output.pkl'),
    ('Truncated MC', f'{root}/TMC/run-*/output.pkl'),
    (r'AME $\epsilon=0.01$', f'{root}/TU0.01/run-*/output.pkl'),
    (r'AME $\epsilon=0.001$', f'{root}/TU0.001/run-*/output.pkl'),
    ('Group Testing\n[Jia et al.]', f'{root}/GTdft/run-*/output.pkl'),
]
df = load_data(path)
ax, _, _ = plot(df, use_order=False, exclude=[],
                figsize=(7, 5), ylim=[0, 0.25], m_ed=2**21)
plt.tight_layout(pad=0)
ax.get_figure().savefig(f'{root}/fig_all.png')
