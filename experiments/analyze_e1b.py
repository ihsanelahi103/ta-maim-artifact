"""EXPLORATORY follow-up to analyze_e1.py (NOT pre-registered; label as post hoc in the paper).
(a) per-Sybil timing: TA-MAIM quarantine episode vs D1@0.5 flag episode
(b) D4r = correlation-only on the RAW quality stream (CATS reads raw quality), D8 = D5 AND D4r
(c) D5 honest false positives per seed."""
import json, numpy as np
from analyze_e1 import load, d1, d4, d5, E, SEEDS   # analyze_e1 prints on import; ignore

def load_raw(scn, s):
    d = json.load(open("results/qdump_s%d_%s.json" % (s, scn)))
    n = 1 + max(r[2] for r in d["records"])
    R = np.full((n, E + 1), np.nan); cnt = np.zeros((n, E + 1))
    for ep, _, a, raw, pr in d["records"]:
        if ep <= E:
            R[a, ep] = raw if cnt[a, ep] == 0 else R[a, ep] + raw; cnt[a, ep] += 1
    with np.errstate(invalid="ignore"): R = R / np.where(cnt > 0, cnt, np.nan)
    return R

for scn in ["collude", "camou", "qmatch", "band1", "band2"]:
    early = same = late = never_d1 = both = 0; diffs = []
    tp4 = fp4 = tp8 = fp8 = 0; fp5 = []
    for s in SEEDS:
        q, P, nb = load(scn, s); syb = set(q["sybil_ids"]); qe = {int(k): v for k, v in q["quar_ep"].items()}
        f1 = d1(P, 0.5)
        for a in syb:
            if a in qe and a in f1:
                both += 1; diffs.append(f1[a] - qe[a])
                if f1[a] < qe[a]: early += 1
                elif f1[a] == qe[a]: same += 1
                else: late += 1
            elif a in qe: never_d1 += 1
        R = load_raw(scn, s); f4r = d4(R, nb); f5 = d5(P, nb)
        f8 = {a: f4r[a] for a in f4r if a in f5}
        tp4 += len(set(f4r) & syb); fp4 += len(set(f4r) - syb)
        tp8 += len(set(f8) & syb); fp8 += len(set(f8) - syb); fp5.append(len(set(f5) - syb))
    k = len(SEEDS)
    print("=== %s ===" % scn)
    print(" D1@.5 vs TA-MAIM (Sybils flagged by both, n=%d): D1 earlier %d, same %d, later %d; median (D1 - TA-MAIM) = %s ; TA-MAIM flagged but D1 never/truncated: %d" % (
        both, early, same, late, ("%.0f" % np.median(diffs)) if diffs else "-", never_d1))
    print(" D4r raw corr>0.3: TP %.1f/30  FP %.1f/70 | D8 = D5 AND D4r: TP %.1f/30  FP %.1f/70 | D5 FP per seed %s" % (
        tp4 / k, fp4 / k, tp8 / k, fp8 / k, fp5))
