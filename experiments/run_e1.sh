#!/bin/bash
# usage: run_e1.sh <scenario> <seed>
cd "$(dirname "$0")/.."
scn=$1; s=$2
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=. TAMAIM_VARIANTS=A5 TAMAIM_SEED=$s TAMAIM_TAG=_$scn
case $scn in
  static)   export TAMAIM_ATTACK=static ;;
  camou)    export TAMAIM_ATTACK=camouflage ;;
  collude)  export TAMAIM_ATTACK=collusion ;;
  qmatch)   export TAMAIM_ATTACK=collusion TAMAIM_QMATCH=1 ;;
  band1)    export TAMAIM_ATTACK=static TAMAIM_EVADE=0.30,0.50 ;;
  band2)    export TAMAIM_ATTACK=static TAMAIM_EVADE=0.40,0.60 ;;
  band3)    export TAMAIM_ATTACK=static TAMAIM_EVADE=0.50,0.70 ;;
  band4)    export TAMAIM_ATTACK=static TAMAIM_EVADE=0.60,0.80 ;;
  band5)    export TAMAIM_ATTACK=static TAMAIM_EVADE=0.65,1.00 ;;
esac
mkdir -p logs/e1
python experiments/qdump_run.py > logs/e1/${scn}_s${s}.log 2>&1
