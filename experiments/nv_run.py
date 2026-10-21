"""NV-A / NV-B (pre-registered 17 Oct 2026: preregistration_novelty_experiments_2026-10-17 in the project docs).
NV-A: equally frequent simple ensembles (EN1 = S&C(raw) OR Beta<.5(priv); EN2 = EN1 OR Q<.5(priv)) vs TA-MAIM(shadow) on the shadow streams.
NV-B: adaptive audit scheduler (cheap PC1-share trigger + mandatory evaluation every 25 episodes) vs fixed audit periods.
Amendment A1 (17 Oct, after a smoke test on the seed-7 streams showed the original PC1-rise trigger never fires, because colluders are
correlated from episode 1 and there is no earlier baseline): the trigger is now a cheap correlation PRE-SCREEN (no graph computation) on the
last W=15 episodes of raw quality among agents not yet flagged: evaluate when >= 5 agents each have >= 5 partners with Pearson r > 0.5, min gap 2;
mandatory evaluation every Pmax=25 episodes. A1 parameters were set once and are NOT tuned further. Seed 7 is the development stream; the pre-registered
criteria are evaluated on the held-out seeds (all seeds != 7) when present.
usage: python experiments/nv_run.py        -> results/nv_results.json (+ printed summary and criteria)"""
import json, glob, os, re, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
from experiments import fs_analyze as FA

N = FA.N
W, GAP, PMAX, RTH, NPART, NAG = 15, 2, 25, 0.5, 5, 5
FIXED = [1, 5, 10, 25]


def union(*fl):
    out = {}
    for f in fl:
        for a, e in f.items(): out[a] = min(e, out.get(a, e))
    return out


def prescreen(R, ep, excl):
    """cheap correlation pre-screen: number of unflagged agents with >= NPART partners whose raw-quality correlation over the last W episodes exceeds RTH"""
    lo = max(1, ep - W); X = R[:, lo:ep]
    keep = [a for a in range(N) if a not in excl and (~np.isnan(X[a])).sum() >= 4]
    if len(keep) < 10 or X.shape[1] < 5: return 0
    Y = X[keep]; mu = np.nanmean(Y, axis=1, keepdims=True); Y = np.where(np.isnan(Y), mu, Y) - mu
    sd = np.sqrt((Y ** 2).sum(1)); sd[sd < 1e-9] = np.inf
    Cm = (Y @ Y.T) / np.outer(sd, sd); np.fill_diagonal(Cm, 0)
    return int(((Cm > RTH).sum(1) >= NPART).sum())


def adaptive_schedule(R, E, cond):
    """returns (evaluation episodes, per-episode decision array); flagged agents latch (as in the loop) and leave the pre-screen"""
    last = 0; evals = []; flagged = set(); per = np.zeros_like(cond)
    for ep in range(3, E + 1):
        trig = prescreen(R, ep, flagged) >= NAG
        if (trig and ep - last >= GAP) or ep - last >= PMAX:
            evals.append(ep); last = ep; per[ep] = cond[ep]
            flagged |= set(np.nonzero(cond[ep])[0].tolist())
    return evals, per


def one(path):
    d, E, rec, g1, diff = FA.load(path)
    FA.COMM.clear(); FA.COMM.update(d.get("comm") or [])
    syb = set(d["sybil_ids"] or range(70, 100))
    T = None
    if d.get("sybil_T") and d["sybil_T"].get("transitions"):
        T = {int(k): int(v) for k, v in d["sybil_T"]["transitions"].items()}
    calls = np.zeros((N, E + 2))
    for r in rec:
        if int(r[0]) <= E: calls[int(r[2]), int(r[0])] += 1
    P, _ = FA.episode_means(rec, E, 4); R, _ = FA.episode_means(rec, E, 3)
    S, C, SUS = FA.cats_replay(rec, E, g1, diff, 3)
    sc = FA.first((S > 0.5) & (C > 0.3), E)
    beta = FA.q_detectors(P, E, 0.5, "Beta"); q5 = FA.q_detectors(P, E, 0.5, "Q")
    sh = {int(k): v for k, v in d["shadow_q"].items()}
    out = {"A": {}, "B": {}}
    for nm, fl in (("TA-MAIM", {a: v[0] for a, v in sh.items()}), ("S&C", sc), ("EN1", union(sc, beta)), ("EN2", union(sc, beta, q5))):
        out["A"][nm] = FA.score(fl, syb, calls, E, T)
    cond = (S > 0.5) & (C > 0.3) & (SUS > 0.55)
    for Pn in FIXED:
        per = np.zeros_like(cond)
        for ep in range(Pn, E + 1, Pn): per[ep] = cond[ep]
        s = FA.score(FA.first(per, E), syb, calls, E, T); s["evals"] = len(range(Pn, E + 1, Pn)); out["B"]["P=%d" % Pn] = s
    ev, per = adaptive_schedule(R, E, cond)
    s = FA.score(FA.first(per, E), syb, calls, E, T); s["evals"] = len(ev); out["B"]["adaptive"] = s
    return out


def regime_of(p):
    return re.sub(r"_s\d+\.json$", "", os.path.basename(p))


def main():
    files = {}
    for p in sorted(glob.glob("results/fs/*_s[0-9]*.json")): files.setdefault(regime_of(p), []).append(p)
    res = {}
    t0 = time.time()
    seeds = {}
    for rg, ps in files.items():
        res[rg] = [one(p) for p in ps]; seeds[rg] = [int(re.search(r"_s(\d+)\.json$", p).group(1)) for p in ps]
        print("done %s (%d runs) %.0fs" % (rg, len(ps), time.time() - t0), flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump({"seeds": seeds, "res": res}, open("results/nv_results.json", "w"))
    allres = res
    held = {rg: [r for r, sd in zip(rr, seeds[rg]) if sd != 7] for rg, rr in res.items()}
    held = {rg: rr for rg, rr in held.items() if rr}
    use_held = len(held) > 0
    print("criteria use held-out seeds (!= 7): %s (%s)" % (use_held, {rg: len(rr) for rg, rr in (held if use_held else res).items()}))
    res = held if use_held else res
    mean = lambda rr, blk, k, f: float(np.mean([r[blk][k][f] for r in rr]))
    medlat = lambda rr, blk, k: float(np.median([x for r in rr for x in r[blk][k]["lat"]])) if any(r[blk][k]["lat"] for r in rr) else float("nan")
    print("\n=== NV-A: equally frequent simple ensembles (TP/30, FP/70, median latency) ===")
    print("%-10s " % "regime" + " ".join("%-22s" % k for k in ("TA-MAIM", "S&C", "EN1", "EN2")) + " match(EN1/EN2)")
    nmatch = {"EN1": 0, "EN2": 0}
    for rg, rr in res.items():
        cells, m = [], {}
        for k in ("TA-MAIM", "S&C", "EN1", "EN2"):
            cells.append("%5.1f %5.1f %6.1f     " % (mean(rr, "A", k, "tp"), mean(rr, "A", k, "fp"), medlat(rr, "A", k)))
        for k in ("EN1", "EN2"):
            ok = abs(mean(rr, "A", k, "tp") - mean(rr, "A", "TA-MAIM", "tp")) <= 1 and mean(rr, "A", k, "fp") <= mean(rr, "A", "TA-MAIM", "fp") + 0.5 \
                 and abs(medlat(rr, "A", k) - medlat(rr, "A", "TA-MAIM")) <= 2
            m[k] = ok; nmatch[k] += ok
        print("%-10s " % rg + " ".join(cells) + " %s/%s" % (m["EN1"], m["EN2"]))
    nreg = len(res)
    print("regimes where the ensemble matches TA-MAIM: EN1 %d/%d, EN2 %d/%d (criterion: >= 8 -> 'architecture not shown to be essential')" % (nmatch["EN1"], nreg, nmatch["EN2"], nreg))
    print("\n=== NV-B: adaptive scheduler vs fixed audit period (TP/30, FP/70, exposure, evals/200) ===")
    print("%-10s " % "regime" + " ".join("%-24s" % k for k in ["P=1", "P=5", "P=10", "P=25", "adaptive"]))
    strong = {"qmatch": None, "collude": None, "qm_ae10": None}; tp_ok = True
    for rg, rr in res.items():
        cells = ["%5.1f %5.1f %5.3f %4.0f     " % (mean(rr, "B", k, "tp"), mean(rr, "B", k, "fp"), mean(rr, "B", k, "expo"), mean(rr, "B", k, "evals")) for k in ["P=1", "P=5", "P=10", "P=25", "adaptive"]]
        print("%-10s " % rg + " ".join(cells))
        if mean(rr, "B", "adaptive", "tp") < mean(rr, "B", "P=5", "tp") - 1e-9: tp_ok = False
        if rg in strong: strong[rg] = (mean(rr, "B", "adaptive", "expo") <= mean(rr, "B", "P=1", "expo") + 0.010) and (mean(rr, "B", "adaptive", "evals") <= 50)
    got = {k: v for k, v in strong.items() if v is not None}
    print("criterion (1) strongly coordinated regimes within +0.010 of P=1 exposure with <= 50 evals: %s -> %s (need >= 2 of 3; available %d)" % (got, sum(got.values()) >= 2, len(got)))
    print("criterion (2) adaptive flags no fewer Sybils than P=5 in every regime: %s" % tp_ok)


if __name__ == "__main__":
    main()
