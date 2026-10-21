export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 TAMAIM_ATTACK_EDGES=$1 TAMAIM_SEED=$2
python3 experiments/sybilrank_baseline.py --datasets yelp --attacks static > logs/sr_g$1_s$2.log 2>&1
