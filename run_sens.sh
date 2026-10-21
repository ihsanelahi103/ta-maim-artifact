export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
export TAMAIM_HONEST_DEG=$1 TAMAIM_REWIRE=$2 TAMAIM_SYBIL_DENSITY=$3 TAMAIM_ATTACK_EDGES=$4
python3 run.py --dataset yelp --attack $5 --seed $6 --scale medium --episodes 200 > logs/sens_d$1_r$2_p$3_a$4_$5_s$6.log 2>&1
