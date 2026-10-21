export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 TAMAIM_CATS_MIN_HIST=$1
python3 run.py --dataset yelp --attack collusion --seed $2 --scale medium --episodes 200 > logs/mh_h$1_s$2.log 2>&1
