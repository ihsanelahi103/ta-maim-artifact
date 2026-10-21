import json, glob, collections, sys
import numpy as np
rows = collections.defaultdict(list); meta = collections.defaultdict(list)
for p in sorted(glob.glob("results/rs/*_s*.json")):
    d = json.load(open(p)); k = (d["dataset"], d["variant"])
    meta[k].append(d)
    for r in d["rows"]: rows[k + (r["t"], r["method"])].append(r)
for k, ds in sorted(meta.items()):
    nh = 70
    print("\n=== %s %s: n=%d runs; targets %s; honest coverage %.2f ===" % (k[0], k[1], len(ds), [d["n_targets"] for d in ds], np.mean([d["cov_honest"] for d in ds])))
    print(" q honest %.3f | q sybil %.3f | AUC(-q) eps1-20 %.2f, 1-200 %.2f" % (np.mean([d["q_honest_mean"] for d in ds]), np.mean([d["q_sybil_mean"] for d in ds]),
          np.mean([d["sep_auc"]["1-20"] for d in ds]), np.mean([d["sep_auc"].get("1-200", np.nan) for d in ds])))
    sq = [sum(1 for a in d["quar_ep"] if int(a) >= nh) for d in ds]; hq = [sum(1 for a in d["quar_ep"] if int(a) < nh) for d in ds]
    print(" TA-MAIM quarantined Sybils/30 per run %s ; honest/70 %s" % (sq, hq))
    pc = collections.Counter(v[1] for d in ds for a, v in d.get("pathways", {}).items() if int(a) < nh)
    ps = collections.Counter(v[1] for d in ds for a, v in d.get("pathways", {}).items() if int(a) >= nh)
    if pc or ps: print(" first-quarantine pathway: honest %s | Sybil %s" % (dict(pc), dict(ps)))
    print(" %-4s %-10s %8s %9s %12s" % ("t", "method", "NDCG@10", "poison@10", "fab_in_pool"))
    for t in (0, 10, 25, 50, 100, 200):
        for m in ("none", "oracle", "TA-MAIM", "Q<0.5", "SybilRank"):
            rr = rows.get(k + (t, m))
            if rr: print(" %-4d %-10s %8.4f %9.3f %12.0f" % (t, m, np.mean([r["ndcg"] for r in rr]), np.mean([r["poison10"] for r in rr]), np.mean([r["fab_in_pool"] for r in rr])))
