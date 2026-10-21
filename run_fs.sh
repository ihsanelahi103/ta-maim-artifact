#!/bin/bash
# usage: bash run_fs.sh jobs.txt [parallel]   (jobs.txt lines: "<regime> <seed>"); restart-safe (skips finished runs)
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=.
mkdir -p logs/fs results/fs
one() { r=$1; s=$2
  [ -f results/fs/${r}_s${s}.json ] && return
  echo "$(date +%H:%M:%S) start $r $s" >> logs/fs/driver.log
  REGIME=$r TAMAIM_SEED=$s python experiments/fs_run.py > logs/fs/${r}_s${s}.log 2>&1
  echo "$(date +%H:%M:%S) done  $r $s" >> logs/fs/driver.log; }
export -f one
cat "$1" | xargs -P ${2:-2} -L 1 bash -c 'one $0 $1'
echo "$(date +%H:%M:%S) ALL DONE $1" >> logs/fs/driver.log
