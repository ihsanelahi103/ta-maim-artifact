"""Paired comparison TA-MAIM vs the new baselines (t, df=4, 95% CI), same convention as Tables 4/5."""
import json, glob, collections, numpy as np
TC = 2.776
NEW = ["fltrust", "flame"]; REF = ["fedavg", "median", "normclip"]; DIAG = ["fltrust_nosybil", "flame_nosybil"]
for ds in ["yelp", "amz"]:
    rows = []
    for p in sorted(glob.glob("results/ubase4/%s_s*.json" % ds)): rows += json.load(open(p))
    if not rows: continue
    seeds = sorted({r["seed"] for r in rows}); print("\n#### %s seeds %s" % (ds, seeds))
    D = collections.defaultdict(dict)
    for r in rows: D[(r["poison"], r["strength"], r["ckpt"], r["method"])][r["seed"]] = r["ndcg"]
    for ck in [200, 75]:
        for st in [1.0, 3.0, 6.0, 10.0]:
            for pk in ["identical", "diverse"]:
                cell = (pk, st, ck)
                ta = D.get(cell + ("TA-MAIM",));
                if not ta: continue
                out = ["%s c=%g ep%d  TA-MAIM %.4f" % (pk[:3], st, ck, np.mean(list(ta.values())))]
                for m in REF + NEW + DIAG:
                    x = D.get(cell + (m,))
                    if not x: continue
                    ss = sorted(set(x) & set(ta)); d = np.array([ta[s] - x[s] for s in ss])
                    if len(ss) < 2: continue
                    h = TC * d.std(ddof=1) / np.sqrt(len(ss)) if len(ss) == 5 else float("nan")
                    lo, hi = d.mean() - h, d.mean() + h
                    mark = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "tie")
                    out.append("%s %.4f(%s %+.4f[%+.4f,%+.4f])" % (m, np.mean(list(x.values())), mark, d.mean(), lo, hi))
                print(" | ".join(out))
    fl = [r for r in rows if r["method"] == "flame" and r["ckpt"] == 200]
    if fl: print("FLAME ep200: mean admitted %.1f of agents, mean Sybils admitted %.1f (of f=%.1f)" % (
        np.mean([r["admitted"] for r in fl]), np.mean([r["sybil_admitted"] for r in fl]), np.mean([r["f"] for r in fl])))
    ft = [r for r in rows if r["method"] == "fltrust" and r["ckpt"] == 200 and r.get("ts_sybil") is not None]
    if ft: print("FLTrust ep200: mean trust honest %.3f, Sybil %.3f" % (np.mean([r["ts_honest"] for r in ft]), np.mean([r["ts_sybil"] for r in ft])))
