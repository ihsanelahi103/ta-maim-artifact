"""Regression check: reference rows of utility_baselines4 must equal utility_baselines3 exactly."""
import json, sys, glob, os
bad = tot = 0
for ds in ["yelp", "amz"]:
    for seed in [7, 11, 42, 99, 123]:
        p3 = "results/ubase3/%s_s%d.json" % (ds, seed); p4 = "results/ubase4/%s_s%d.json" % (ds, seed)
        if not (os.path.exists(p3) and os.path.exists(p4)): continue
        r3 = {(r["poison"], r["strength"], r["ckpt"], r["method"]): r["ndcg"] for r in json.load(open(p3))}
        for r in json.load(open(p4)):
            k = (r["poison"], r["strength"], r["ckpt"], r["method"])
            if k in r3:
                tot += 1
                if abs(r3[k] - r["ndcg"]) > 1e-12: bad += 1; print("MISMATCH", ds, seed, k, r3[k], r["ndcg"])
print("compared %d reference rows, mismatches %d" % (tot, bad))
