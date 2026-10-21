#!/bin/bash
# usage: bash run_fw.sh jobs_f1.txt [parallel]  (lines: "<variant> <attack> <seed>"); restart-safe
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=.
mkdir -p logs/fw results/fw
one() { v=$1; at=$2; s=$3
  [ -f results/fw/${v}_${at}_s${s}.json ] && return
  echo "$(date +%H:%M:%S) start $v $at $s" >> logs/fw/driver.log
  FW_VARIANT=$v TAMAIM_SEED=$s python experiments/fw_run.py --attack $at > logs/fw/${v}_${at}_s${s}.log 2>&1
  echo "$(date +%H:%M:%S) done  $v $at $s" >> logs/fw/driver.log; }
export -f one
cat "$1" | xargs -P ${2:-2} -L 1 bash -c 'one $0 $1 $2'
echo "$(date +%H:%M:%S) ALL DONE $1" >> logs/fw/driver.log
