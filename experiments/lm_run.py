"""Learned merge-policy study (pre-registered: preregistration_learned_merge_2026-10-16.md).

Decision unit: agent j's contribution in episode t.  Action: merge (1) or drop (0).
Utility of merging a contribution of raw quality q:  +q if j is honest,  -LAM*q if j is a Sybil (fabricated data poisons the merge).
Dropping yields 0.  Oracle = drop Sybils, merge honest.  Gap closed G = (U - U_mergeall) / (U_oracle - U_mergeall).

Policies (all see ONLY label-free inputs at the start of episode t, i.e. data through t-1, like CATS):
  ALL   merge everything
  HARD  TA-MAIM CATS rule, latching:  S>.5 & C>.3 & suspicion>.55 (episode >= 3) -> dropped from then on
  THR   non-latching:  drop while suspicion > .55
  SOFT  merge weight = clip(1 - suspicion, 0, 1)
  FB    feedback-matched rule: drop j for good once a sparse validation check (prob PFB per merge) exposes poison
  HARD+FB  HARD or FB (rules given the same feedback the learners get)
  LIN   learned linear contextual bandit   (feedback-trained)
  MLP   learned 1-hidden-layer (16 tanh) contextual bandit (feedback-trained)
  LIN0 / MLP0  same learners but reward = observed quality only (NO validation feedback; label-free)
Learners get the merge reward only with probability PFB per merge event (sparse validation check), eps-greedy merging.
They are pre-trained on the streams of the OTHER regime families and evaluated on the held-out family (leave-one-family-out),
frozen ("-F") and continuing online on the test stream ("-O").

usage: python experiments/lm_run.py [--lam 1.0] [--pfb 0.1] [--seeds 5]   -> results/lm_results.json, prints summary
"""
import json, glob, os, sys, argparse, collections
import numpy as np
sys.path.insert(0, os.getcwd())
from experiments import fs_analyze as FA

N = FA.N
FAMILIES = {
    "quality-matching": ["qmatch", "qm_ae10", "qm_l050", "band1", "band2"],
    "collusion": ["collude"],
    "camouflage": ["camou"],
    "benign-community": ["bc10", "bc20"],
}
THR_S, THR_C, THR_SUS = 0.5, 0.3, 0.55
WIN = 15
EPS, ETA, HID, PASSES = 0.10, 0.01, 16, 3


def features(path):
    d, E, rec, g1, diff = FA.load(path)
    FA.COMM.clear(); FA.COMM.update(d.get("comm") or [])
    syb = set(d["sybil_ids"] or range(70, 100))
    S, C, SUS = FA.cats_replay(rec, E, g1, diff, 3)
    M, cnt = FA.episode_means(rec, E, 3)
    Q = np.nan_to_num(M, nan=0.0)
    rows = []   # (ep, agent, x, q, is_syb)
    for ep in range(3, E + 1):
        for a in range(N):
            if cnt[a, ep] == 0: continue
            lo = max(1, ep - WIN)
            h = M[a, lo:ep]; h = h[~np.isnan(h)]       # history through ep-1 only
            mq = h.mean() if len(h) else 0.5
            sd = h.std() if len(h) > 1 else 0.0
            x = np.array([1.0, S[ep, a], C[ep, a], SUS[ep, a], mq, sd, S[ep, a] * C[ep, a], ep / E])
            rows.append((ep, a, x, Q[a, ep], a in syb))
    return rows, (S, C, SUS), E


class Learner:
    def __init__(self, kind, rng, dim=8):
        self.kind, self.rng = kind, rng
        if kind == "lin":
            self.w = np.zeros(dim)
        else:
            self.W1 = rng.normal(0, 0.3, (HID, dim)); self.b1 = np.zeros(HID)
            self.w2 = rng.normal(0, 0.3, HID); self.b2 = 0.0

    def q(self, x):
        if self.kind == "lin": return float(self.w @ x)
        h = np.tanh(self.W1 @ x + self.b1)
        return float(self.w2 @ h + self.b2)

    def update(self, x, r):
        if self.kind == "lin":
            self.w += ETA * (r - self.w @ x) * x
        else:
            h = np.tanh(self.W1 @ x + self.b1); y = self.w2 @ h + self.b2; e = r - y
            gh = e * self.w2 * (1 - h * h)
            self.w2 += ETA * e * h; self.b2 += ETA * e
            self.W1 += ETA * np.outer(gh, x); self.b1 += ETA * gh

    def act(self, x, explore):
        return 1 if (self.q(x) > 0 or (explore and self.rng.random() < EPS)) else 0


def reward(q, syb, lam, feedback):
    return (-lam * q if syb else q) if feedback else q


def train(learner, rows_by_run, lam, pfb, feedback):
    rng = learner.rng
    for _ in range(PASSES):
        for rows in rows_by_run:
            for ep, a, x, q, syb in rows:
                if learner.act(x, True) and rng.random() < pfb:
                    learner.update(x, reward(q, syb, lam, feedback))


def evaluate(rows, E, S, C, SUS, lam, learners, pfb):
    """returns dict policy -> utility, plus honest-dropped fraction and Sybil-merged fraction"""
    out = collections.defaultdict(float); hd = collections.defaultdict(float); sm = collections.defaultdict(float)
    nh = ns = 0
    latched = set(); fbl = set(); hfl = set(); frng = np.random.default_rng(99)
    online = {k: (l if k.endswith("-O") else None) for k, l in learners.items()}
    frozen = {k: l for k, l in learners.items() if k.endswith("-F")}
    for ep, a, x, q, syb in rows:
        val = -lam * q if syb else q
        nh += (not syb); ns += syb
        cond = (S[ep, a] > THR_S) and (C[ep, a] > THR_C) and (SUS[ep, a] > THR_SUS)
        if cond: latched.add(a)
        # feedback-matched rules: same sparse validation check as the learners (probability pfb per merged contribution)
        fb_seen = frng.random() < pfb
        if a not in fbl and fb_seen and syb: fbl.add(a)
        if a not in hfl and ((fb_seen and syb) or cond): hfl.add(a)
        w = {"FB": 0.0 if a in fbl else 1.0, "HARD+FB": 0.0 if a in hfl else 1.0, "ALL": 1.0, "HARD": 0.0 if a in latched else 1.0, "THR": 0.0 if SUS[ep, a] > THR_SUS else 1.0,
             "SOFT": float(np.clip(1 - SUS[ep, a], 0, 1)), "ORACLE": 0.0 if syb else 1.0}
        for k, l in learners.items():
            if k.endswith("-F"): w[k] = float(l.q(x) > 0)
            else:                                           # online: act, then learn from sparse feedback
                m = l.act(x, True); w[k] = float(m)
                if m and l.rng.random() < pfb: l.update(x, reward(q, syb, lam, not k.startswith(("LIN0", "MLP0"))))
        for k, wk in w.items():
            out[k] += wk * val
            if not syb: hd[k] += 1 - wk
            else: sm[k] += wk
    return out, {k: hd[k] / max(nh, 1) for k in out}, {k: sm[k] / max(ns, 1) for k in out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--pfb", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5, help="learner initialisation seeds")
    a = ap.parse_args()
    data = {}   # family -> list of (regime, file, rows, stats, E)
    for fam, regs in FAMILIES.items():
        for rg in regs:
            for p in sorted(glob.glob("results/fs/%s_s[0-9]*.json" % rg)):
                rows, st, E = features(p)
                data.setdefault(fam, []).append((rg, os.path.basename(p), rows, st, E))
    print("streams:", {f: len(v) for f, v in data.items()})
    res = []
    for held in data:
        train_runs = [r[2] for f, v in data.items() if f != held for r in v]
        if not train_runs: continue
        for ls in range(a.seeds):
            rng = np.random.default_rng(1000 + ls)
            base = {}
            for name, kind, fb in [("LIN", "lin", True), ("MLP", "mlp", True), ("LIN0", "lin", False), ("MLP0", "mlp", False)]:
                L = Learner(kind, np.random.default_rng(rng.integers(1 << 30)))
                train(L, train_runs, a.lam, a.pfb, fb)
                base[name] = L
            for rg, fn, rows, (S, C, SUS), E in data[held]:
                import copy
                learners = {}
                for name, L in base.items():
                    learners[name + "-F"] = copy.deepcopy(L)
                    o = copy.deepcopy(L); o.rng = np.random.default_rng(rng.integers(1 << 30))
                    learners[name + "-O"] = o
                U, HD, SM = evaluate(rows, E, S, C, SUS, a.lam, learners, a.pfb)
                gap = U["ORACLE"] - U["ALL"]
                G = {k: (U[k] - U["ALL"]) / gap if gap > 1e-9 else float("nan") for k in U}
                res.append({"family": held, "regime": rg, "file": fn, "learner_seed": ls, "U": dict(U), "G": G,
                            "honest_dropped": HD, "sybil_merged": SM})
    os.makedirs("results", exist_ok=True)
    json.dump({"lam": a.lam, "pfb": a.pfb, "results": res}, open("results/lm_results.json", "w"))
    pols = ["ALL", "HARD", "THR", "SOFT", "FB", "HARD+FB", "LIN-F", "LIN-O", "MLP-F", "MLP-O", "LIN0-O", "MLP0-O", "ORACLE"]
    print("\nGap closed G (0 = merge-all, 1 = oracle); mean over streams x learner seeds; lam=%.2f pfb=%.2f" % (a.lam, a.pfb))
    print("%-18s " % "family/regime" + " ".join("%7s" % p for p in pols))
    for fam in data:
        for rg in sorted({r["regime"] for r in res if r["family"] == fam}):
            rr = [r for r in res if r["regime"] == rg]
            print("%-18s " % rg + " ".join("%7.3f" % np.nanmean([r["G"][p] for r in rr]) for p in pols))
        rr = [r for r in res if r["family"] == fam]
        if rr: print("%-18s " % ("[" + fam[:15] + "]") + " ".join("%7.3f" % np.nanmean([r["G"][p] for r in rr]) for p in pols))
    # macro average over families
    fams = [f for f in data if any(r["family"] == f for r in res)]
    print("%-18s " % "MACRO(families)" + " ".join(
        "%7.3f" % np.mean([np.nanmean([r["G"][p] for r in res if r["family"] == f]) for f in fams]) for p in pols))
    print("\nhonest merge-share dropped / Sybil merge-share kept (macro over families)")
    for p in pols[1:-1]:
        h = np.mean([np.mean([r["honest_dropped"][p] for r in res if r["family"] == f]) for f in fams])
        s = np.mean([np.mean([r["sybil_merged"][p] for r in res if r["family"] == f]) for f in fams])
        print("  %-7s honest dropped %.3f   Sybil kept %.3f" % (p, h, s))


if __name__ == "__main__":
    main()
