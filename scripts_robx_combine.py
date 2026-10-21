import json, glob, math
def wilson(k, n, z=1.96):
    if n == 0: return (0, 1)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))
# earlier 5 seeds (miss_runs, fp_runs, n) from the verified per-seed table
# note: collusion_10 seed 11 had F1 0; its fp count was not separately verified, assumed 0
earlier = {"collusion_10": (1, 0, 5), "collusion_20": (0, 0, 5), "camouflage_20": (0, 1, 5)}
for case in ["collusion_10", "collusion_20", "camouflage_20"]:
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"results/robx/{case}_s*.json"))]
    n = len(rows)
    miss = [r["seed"] for r in rows if r["recall"] < 1]
    fpr = [r["seed"] for r in rows if r["fp"] > 0]
    f1s = [r["f1"] for r in rows]
    ts = [r["trust_sep"] for r in rows]
    em, ef, en = earlier[case]
    N = n + en; M = len(miss) + em; F = len(fpr) + ef
    lo, hi = wilson(M, N)
    print(f"\n{case}: new n={n}")
    print(f"  runs with recall<1: {len(miss)} seeds={miss}")
    print(f"  runs with FP>0:     {len(fpr)} seeds={fpr}")
    print(f"  mean F1 {sum(f1s)/n:.4f}  min F1 {min(f1s):.4f}")
    print(f"  trust_sep min {min(ts):.4f}  mean {sum(ts)/n:.4f}  runs trust_sep<0.1: {sum(t<0.1 for t in ts)}")
    print(f"  dp_eps values: {sorted(set(round(r['dp_eps'],3) for r in rows))}")
    print(f"  COMBINED n={N}: miss runs {M}/{N} (Wilson95 [{lo:.3f},{hi:.3f}]), FP runs {F}/{N}")
    for r in rows:
        if r["recall"] < 1 or r["fp"] > 0:
            print(f"   seed {r['seed']}: F1={r['f1']:.3f} rec={r['recall']:.3f} fp={r['fp']} trust_sep={r['trust_sep']:.4f} eps={r['dp_eps']:.3f}")
