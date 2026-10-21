#!/bin/bash
# Overnight driver: extra seeds for the federated-recommendation comparison (pre-registered seeds 11, 42, 99, 123).
# Step sizes are the ones selected on seed 7 (pre-registered rule): yelp 0.03, amazon 0.1.
# Safe to restart: finished baseline runs and finished TA-MAIM runs are skipped.
cd "$(dirname "$0")" || exit 1
export PYTHONPATH=.:experiments CUDA_VISIBLE_DEVICES=""
mkdir -p results/fedrec logs

run_one() {   # dataset seed eta
  ds=$1; seed=$2; eta=$3
  echo "$(date +%T) start baselines $ds seed $seed" >> logs/driver.log
  python experiments/fedrec_baselines.py --dataset $ds --seed $seed --eta $eta \
      --out results/fedrec/${ds}_s${seed}.json >> logs/baselines_${ds}_s${seed}.log 2>&1
  for agg in mean normclip; do
    tag=_fedrec_${ds}_${agg}
    if [ -f results/fedrec_tamaim_${ds}_s${seed}${tag}.json ]; then continue; fi
    echo "$(date +%T) start TA-MAIM $ds seed $seed $agg" >> logs/driver.log
    FEDREC_ETA=$eta FEDREC_AGG=$agg TAMAIM_SEED=$seed TAMAIM_TAG=$tag TAMAIM_DATASET=$ds FEDREC_EPISODES=200 \
      python experiments/fedrec_tamaim_run.py > logs/tamaim_${ds}_s${seed}_${agg}.log 2>&1
  done
  echo "$(date +%T) finished $ds seed $seed" >> logs/driver.log
}

# two parallel streams (priority order: Amazon seeds first, then Yelp, then the last seeds if time remains)
( run_one amazon 11 0.1; run_one yelp 11 0.03; run_one yelp 99 0.03 ) &
( run_one amazon 42 0.1; run_one yelp 42 0.03; run_one yelp 123 0.03 ) &
wait
echo "$(date +%T) ALL DONE" >> logs/driver.log
