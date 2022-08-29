from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
root = sys.argv[1]
cs, precs, recalls = [], [], []
for path in Path(f"{root}").glob("pr-*.pkl"):
    c = float(path.stem.split('-')[1])
    df = pd.read_pickle(path)
    precs.append(np.nanmean(df.precision))
    recalls.append(np.nanmean(df.recall))
    cs.append(c)
idx = np.argsort(cs)
cs = [cs[i] for i in idx]
precs = [precs[i] for i in idx]
recalls = [recalls[i] for i in idx]
print(cs)
print(precs)
print(recalls)
plt.plot(cs, precs, label='Precision', marker='o', linestyle='solid')
plt.plot(cs, recalls, label='Recall', marker='o', linestyle='dashed')
plt.legend()
plt.xscale('log')
plt.minorticks_off()
plt.xticks(cs, map(str, cs))
plt.xlabel('$c$')
plt.ylabel('Precision (recall)')
plt.title(f"{root}")
plt.savefig(f"{root}/fig.png")
plt.show()
