import sys, os, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "experiments")
import robustness_sybil_ratio as R
atk, frac, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
r = R.run_single(frac, atk, "yelp", 200, seed)
json.dump(dict(attack=atk, frac=frac, seed=seed, **r), open("results/robx/%s_%d_s%d.json" % (atk, int(round(frac*100)), seed), "w"))
print("DONE", atk, frac, seed, r)
