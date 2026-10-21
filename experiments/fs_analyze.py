"""Replay every detector on the identical, uncensored shadow-mode stream (results/fs/<regime>_s<seed>.json).
Protocol: experiments/preregistration_review2 (project doc). Every detector is updated once per episode at the START of the
episode from data through the previous episode (the schedule of CATS), same warm-up (3 episodes), same stream.
Effective flag episode = the episode in which the decision applies (shadow TA-MAIM logs the same convention)."""
import json, glob, os, sys, collections, itertools
import numpy as np
sys.path.insert(0, os.getcwd())
from algorithms.collusion.collusion_detector import CollusionAwareTrustSignal as CATS

N = 100
QTAU = [0.3, 0.4, 0.5, 0.6, 0.7]
CGRID = [0.1, 0.2, 0.3, 0.4, 0.5]
SGRID = [0.3, 0.4, 0.5, 0.6, 0.7]
BUDGETS = [0, 1, 2, 5]


def load(path):
    d = json.load(open(path)); E = d["episodes"]
    rec = np.array(d["records"], dtype=float)
    g1 = {int(k): v for k, v in d["graph1"].items()}
    diff = {int(k): v for k, v in d["gdiff"].items()}
    return d, E, rec, g1, diff


def graph_at(g1, diff, ep):
    if ep not in diff: return {i: {j: 1.0 for j in nb} for i, nb in g1.items()}
    g = {i: set(nb) for i, nb in g1.items()}
    for i, j in diff[ep]:
        (g[i].discard(j) if j in g[i] else g[i].add(j))
    return {i: {j: 1.0 for j in nb} for i, nb in g.items()}


def episode_means(rec, E, col):
    S = np.zeros((N, E + 2)); C = np.zeros((N, E + 2))
    for r in rec:
        ep, a = int(r[0]), int(r[2])
        if ep <= E: S[a, ep] += r[col]; C[a, ep] += 1
    with np.errstate(invalid="ignore"):
        M = np.where(C > 0, S / np.maximum(C, 1), np.nan)
    return M, C


def cats_replay(rec, E, g1, diff, col):
    cd = CATS(num_agents=N)
    byep = collections.defaultdict(list)
    for r in rec: byep[int(r[0])].append(r)
    S, Cc, SUS = np.zeros((E + 2, N)), np.zeros((E + 2, N)), np.zeros((E + 2, N))
    for ep in range(1, E + 1):
        cd.update(graph_at(g1, diff, ep))
        S[ep], Cc[ep], SUS[ep] = cd.structural_score, cd.correlation_score, cd.suspicion
        for r in byep.get(ep, []): cd.record_quality(int(r[2]), r[col])
    return S, Cc, SUS


def first(cond, E, start=3):
    """cond: bool array [E+2, N]; first episode >= start where true (per agent) -> dict"""
    out = {}
    for a in range(N):
        idx = np.nonzero(cond[start:E + 1, a])[0]
        if len(idx): out[a] = int(idx[0]) + start
    return out


def q_detectors(M, E, tau, kind):
    flag = {}
    for a in range(N):
        v = M[a]
        al = be = 1.0; m = None; c = 0; hist = []
        for ep in range(1, E + 1):          # decision at start of ep uses data through ep-1
            if ep >= 3 and c >= 3:
                if kind == "Q" and np.mean(hist[-5:]) < tau: flag[a] = ep; break
                if kind == "EWMA" and m < tau: flag[a] = ep; break
                if kind == "Beta" and c >= 10 and al / (al + be) < 0.5: flag[a] = ep; break
            x = v[ep]
            if not np.isnan(x):
                hist.append(x); c += 1
                m = x if m is None else 0.9 * m + 0.1 * x
                al = 0.9 * al + x; be = 0.9 * be + (1 - x)
    return flag


COMM = set()


def score(flags, syb, calls, E, T=None):
    tp = [a for a in flags if a in syb]; fp = [a for a in flags if a not in syb]
    tot = sum(calls[a].sum() for a in syb)
    expo = 0.0
    for a in syb:
        fe = flags.get(a, E)
        expo += calls[a, 1:fe + 1].sum()
    lat = []
    for a in tp:
        t0 = 1 if T is None else int(T.get(a, 1))
        lat.append(flags[a] - t0)
    return dict(tp=len(tp), fp=len(fp), fpc=len([a for a in fp if a in COMM]), expo=expo / max(tot, 1), lat=lat)


def run_file(path):
    d, E, rec, g1, diff = load(path)
    COMM.clear(); COMM.update(d.get('comm') or [])
    syb = set(d["sybil_ids"] or range(70, 100))
    T = None
    if d.get("sybil_T") and d["sybil_T"].get("transitions"):
        T = {int(k): int(v) for k, v in d["sybil_T"]["transitions"].items()}
    P, Cn = episode_means(rec, E, 4); R, _ = episode_means(rec, E, 3)
    calls = np.zeros((N, E + 2))
    for r in rec:
        if int(r[0]) <= E: calls[int(r[2]), int(r[0])] += 1
    out = {}
    sh = {int(k): v for k, v in d["shadow_q"].items()}
    out["TA-MAIM(shadow)"] = score({a: v[0] for a, v in sh.items()}, syb, calls, E, T)
    out["GATEM-only(P1,P2)"] = score({a: v[0] for a, v in sh.items() if v[1] in ("P1", "P2")}, syb, calls, E, T)
    for nm, M in (("priv", P), ("raw", R)):
        for tau in QTAU: out["Q<%.1f [%s]" % (tau, nm)] = score(q_detectors(M, E, tau, "Q"), syb, calls, E, T)
        out["EWMA<.5 [%s]" % nm] = score(q_detectors(M, E, 0.5, "EWMA"), syb, calls, E, T)
        out["Beta<.5 [%s]" % nm] = score(q_detectors(M, E, 0.5, "Beta"), syb, calls, E, T)
    for nm, col in (("priv", 4), ("raw", 3)):
        S, Cc, SUS = cats_replay(rec, E, g1, diff, col)
        for c0 in CGRID: out["Corr>%.1f [%s]" % (c0, nm)] = score(first(Cc > c0, E), syb, calls, E, T)
        for s0 in SGRID: out["Struct>%.1f [%s]" % (s0, nm)] = score(first(S > s0, E), syb, calls, E, T)
        for s0, c0 in itertools.product(SGRID, CGRID):
            out["S>%.1f&C>%.1f [%s]" % (s0, c0, nm)] = score(first((S > s0) & (Cc > c0), E), syb, calls, E, T)
            out["CATS s%.1f c%.1f [%s]" % (s0, c0, nm)] = score(first((S > s0) & (Cc > c0) & (SUS > 0.55), E), syb, calls, E, T)
        # periodic audit (every 25 episodes) with the default rule
        cond = (S > 0.5) & (Cc > 0.3) & (SUS > 0.55)
        per = np.zeros_like(cond)
        for ep in range(25, E + 1, 25): per[ep] = cond[ep]
        out["Periodic-audit25 CATS [%s]" % nm] = score(first(per, E), syb, calls, E, T)
    return out


HEAD = ["TA-MAIM(shadow)", "GATEM-only(P1,P2)", "Q<0.5 [priv]", "Q<0.5 [raw]", "EWMA<.5 [priv]", "Beta<.5 [priv]",
        "Corr>0.3 [raw]", "Corr>0.3 [priv]", "Struct>0.5 [priv]",
        "S>0.5&C>0.3 [raw]", "S>0.5&C>0.3 [priv]", "CATS s0.5 c0.3 [raw]", "CATS s0.5 c0.3 [priv]",
        "Periodic-audit25 CATS [raw]"]


def summarize(regime, runs):
    keys = list(runs[0].keys()); n = len(runs)
    def agg(k):
        v = [r[k] for r in runs]
        lat = [x for r in v for x in r["lat"]]
        return (np.mean([r["tp"] for r in v]), np.mean([r["fp"] for r in v]), sum(1 for r in v if r["fp"] > 0),
                sum(1 for r in v if r["tp"] == 30), np.median(lat) if lat else float("nan"), np.mean([r["expo"] for r in v]))
    print("\n=== %s: n=%d runs (30 Sybils / 70 honest) ===" % (regime, n))
    print("%-30s %6s %6s %8s %7s %7s %7s %7s" % ("detector", "TP/30", "FP/70", "runsFP>0", "30/30", "medLat", "expo", "FPcomm"))
    for k in HEAD:
        if k in runs[0]:
            a = agg(k); print("%-30s %6.1f %6.1f %8d %7d %7.1f %7.3f %7.1f" % ((k,) + a + (np.mean([r[k]["fpc"] for r in runs]),)))
    print("matched-FP-budget (best mean TP on the grid, mean FP<=B):")
    fam = {"Q (priv)": [k for k in keys if k.startswith("Q<") and "[priv]" in k],
           "Corr (raw)": [k for k in keys if k.startswith("Corr>") and "[raw]" in k],
           "Struct": [k for k in keys if k.startswith("Struct>") and "[priv]" in k],
           "S&C (raw)": [k for k in keys if k.startswith("S>") and "[raw]" in k],
           "S&C (priv)": [k for k in keys if k.startswith("S>") and "[priv]" in k],
           "CATS (raw)": [k for k in keys if k.startswith("CATS") and "[raw]" in k],
           "CATS (priv)": [k for k in keys if k.startswith("CATS") and "[priv]" in k]}
    for f, ks in fam.items():
        row = []
        for B in BUDGETS:
            best = None
            for k in ks:
                a = agg(k)
                if a[1] <= B and (best is None or a[0] > best[0]): best = (a[0], k)
            row.append("B%d: %s" % (B, ("%.1f" % best[0]) if best else "-"))
        print("  %-12s %s" % (f, "  ".join(row)))


if __name__ == "__main__":
    regs = sys.argv[1:] or sorted({os.path.basename(p).rsplit("_s", 1)[0] for p in glob.glob("results/fs/*_s*.json")})
    allres = {}
    for rg in regs:
        files = sorted(glob.glob("results/fs/%s_s[0-9]*.json" % rg))
        runs = [run_file(p) for p in files]
        if runs: summarize(rg, runs); allres[rg] = runs
    json.dump(allres, open("results/fs_analysis.json", "w"))
