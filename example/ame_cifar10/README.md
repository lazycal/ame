This folder contains scripts for running the CIFAR10-20 and CIFAR10-50 experiments.

1.  Change to this directory:
    ```shell
    cd /path/to/the/repo/
    cd ./example/ame_cifar10
    ```
2. Generate poisoned dataset:
    ```shell
    python gen_data.py
    ```
3. (optional) Train models on the full dataset to verify the attack success rate and the training script.
    ```shell
    python train.py 50 # train on CIFAR10-50; attack success rate should be >99%
    python train.py 20 # train on CIFAR10-20; attack success rate should be >70%
    # Both of the above should have clean accuracy > 92%
    ```
4. Sample subsets
    ```shell
    python expl.py -k 20 --phase sample -c 8
    ```
    This runs the CIFAR10-20 experiment (pass `-k` with 50 to run CIFAR10-50).
    It will sample $ck\log_2(N)$ subsets where `c` is specified as 8 here.
5. Train submodels. 
    ```shell
    python expl.py -k 20 --phase train_eval -c 8
    ```
    Since this step requires training thousands of models (amounting to 2 to 3 V100-days as each model takes roughly 90s to train), one may consider parallelization. For instance, if you have 4 GPUs:
    ```shell
    njobs=4
    for i in $(seq 1 $njobs); do
      CUDA_VISIBLE_DEVICES=$i python expl.py -k 20 --phase train_eval -c 8 --njobs $njobs --jobid $((${i}-1)) &
    done
    wait
    ```
    Or if you are using SLURM, you may use the follow command to parllelize with 4 GPUs:
    ```shell
    njobs=4
    sbatch --array=0-$(($njobs-1)) -t1-00:00:00 --gres=gpu:1 -c 7 --mem=32GB --wrap "bash -c 'python expl.py -k 20 --phase train_eval -c 8 --njobs $njobs --jobid \$SLURM_ARRAY_TASK_ID'"
    ```
6. Run LASSO.
    ```shell
    python expl.py -k 20 --phase regress -c 8
    ```
    Note that this might need 250GB of memory. In case it runs out of memory, try pass `--num_threads` with a smaller number (the default is 5).
7. To increase the sample size, simply change `c` and rerun steps 4 to 6. It will reuse the previously trained models.

To plot the precision and recall vs sample size curve, run ``python plot.py logs-20``.