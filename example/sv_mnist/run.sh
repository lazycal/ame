#!/bin/bash -e

njobs=$1
m=$2
seed=0
mkdir -p logs
work()
{
  estimator=$1
  m=$2
  echo "--------> $estimator"
  log=logs/${estimator}/run-$seed
  mkdir logs/${estimator}
  mkdir $log
  python -u cmp_baseline.py -m $m --estimator ${estimator} --seed $seed --path $log --phase sample 2>&1 | tee ${log}/sample.log
  seq 0 $(($njobs-1)) | xargs -n 1 -P 0 -I{} bash -c "numactl --physcpubind=+{}-{} python -u cmp_baseline.py --phase ueval -m $m --njobs $njobs --jobid {} --estimator ${estimator} --seed $seed --path ${log} 2>&1 | tee ${log}/eval-{}.log"
  python -u cmp_baseline.py -m $m --estimator ${estimator} --seed $seed --path $log --phase regress --gt logs/MC/run-$seed/output.pkl 2>&1 | tee ${log}/result.log
}


work "MC" $(($m*2))

# to save time, one may consider uncommenting the next line to only run Compressive Sensing with their hyperparameter "M" (denoted as CS_M) =2^14, unlike stated in the paper that uses CS_M=2^7..2^16. However, empirically, this may cause a difference in the estimation error compared to those in the paper with an absolute error up to 2e-2 for CS_M>=2^13, especially for small sample size.
# for estimator in "TU0.001" "TU0.01" "TMC" "KSL1" "KS" "CS_0.01_14" "GTdft"; do
for estimator in "TU0.001" "TU0.01" "TMC" "KSL1" "KS" "CS_0.01_7" "CS_0.01_8" "CS_0.01_9" "CS_0.01_10" "CS_0.01_11" "CS_0.01_12" "CS_0.01_13" "CS_0.01_14" "CS_0.01_15" "CS_0.01_16" "GTdft"; do
  work $estimator $m
done
