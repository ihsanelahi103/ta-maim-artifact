#!/bin/bash
# usage: bash run_rs.sh jobs_e3.txt [parallel]  (lines: "<dataset> <variant> <seed>"); restart-safe
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=.:experiments
mkdir -p logs/rs results/rs
one() { d=$1; v=$2; s=$3
  [ -f results/rs/${d}_${v}_s${s}.json ] && return
  echo "$(date +%H:%M:%S) start $d $v $s" >> logs/rs/driver.log
  DATASET=$d RS_VARIANT=$v TAMAIM_SEED=$s python experiments/rs_run.py > logs/rs/${d}_${v}_s${s}.log 2>&1
  echo "$(date +%H:%M:%S) done  $d $v $s" >> logs/rs/driver.log; }
export -f one
cat "$1" | xargs -P ${2:-2} -L 1 bash -c 'one $0 $1 $2'
echo "$(date +%H:%M:%S) ALL DONE $1" >> logs/rs/driver.log
