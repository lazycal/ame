This folder contains example usage of the AME-based Shapley value estimator on MNIST as well as scripts for reproducing the Shapley value experiments on the MNIST dataset.

## Generate the dataset
Run `python gen_data.py` to subsample and poison the MNIST dataset.

## Run the example
This example estimates the Shapley values using the trunated uniform distribution Uniform(epsilon, 1-epsilon).
```bash
m=2097152 # the sample size
njobs=8 # how many cores to use
python main.py --phase sample -m $m
seq 0 $(($njobs-1)) | xargs -n 1 -P 0 -I{} bash -c "numactl --physcpubind=+{}-{} python main.py --phase ueval -m $m --jobid {} --njobs $njobs"
python main.py --phase regress -m $m
```
You can find the estimated Shapley values in `logs-0.1/output.npy`, where 0.1 denotes the epsilon used in the truncated uniform distribution. See `main.py` for how to use other epsilon and other options.

## Reproduce the experiment

Run `bash run.sh <num_cores> <sample_size>` and the result will be plotted as a curve of estimation error v.s. sample size, where the error is computed by treating the Monte Carlo result with `2*<sample_size>` as the true Shapley values. The plots will be saved to `./logs/fig.png` and `./logs/fig_all.png`. An example usage is `bash run.sh 48 2097152`, which will parallelize using 48 cores to run all the estimators up to sample size 2097152 (except Monte Carlo which will also run with `2*<sample_size>`). Note that this may take several days to finish, and some estimators may need around 250GB memory (irrelevant to how many cores are used).
