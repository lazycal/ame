for i in $(seq 0 5); do
  # for estimator in "TU0.01" "TU0.1" "MC" "TMC" "KSL1" "KS" "PS" "CS_0.01_7" "CS_0.01_8" "CS_0.01_9" "CS_0.01_10" "CS_0.01_11" "CS_0.01_12" "CS_0.01_13" "CS_0.01_14" "CS_0.01_15" "CS_0.01_16"; do
  # to save time, only run Compressive Sensing with M=2^7, unlike stated in the paper that uses M=2^7..2^16. Empirically, the result is not sensitive to M (varies by 1e-3 only) in this case.
  for estimator in "TU0.01" "TU0.1" "MC" "TMC" "KSL1" "KS" "PS" "CS_0.01_7"; do
    python cmp_baseline.py -m 524288 --estimator ${estimator} --seed $i --path logs/${estimator}/run-$i
  done
done

# Run group testing for only once due to limited computation resources
i=0
estimator="GTdft"
python cmp_baseline.py -m 524288 --estimator ${estimator} --seed $i --path logs/${estimator}/run-$i

python plot.py