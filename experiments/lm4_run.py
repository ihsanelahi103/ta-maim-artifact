"""Model-derived verification policy (VOI index) vs the strongest fixed/adaptive rule on INDEPENDENT streams
(pre-registered: preregistration_voi_verification_2026-10-19.md).  No learning of the allocation: the index is the myopic value of
information of one more check under the SAME utility model, status rule and verifier that define the task (experiments/lm2_run.py,
lm3_run.py are reused unchanged).  Only the Sybil prior (a 7-feature logistic regression) is fitted, on pre-training streams of the
OTHER families (leave-one-family-out), exactly like TUNED-LIN.

usage:  python experiments/lm4_run.py --budgets 1 2        (writes results/lm4_results_B<b>.json)
        python experiments/lm4_run.py --smoke               (execution check only)
"""
import json, os, sys, argparse, math, hashlib
import numpy as np
sys.path.insert(0, os.getcwd())
from experiments import lm3_run as M
from experiments.lm2_run import (CFG, HELD_OUT, RULES, Rule, boot_ci, N)

TRAIN_SEEDS = [7, 11, 42, 99, 123]
TEST_SEEDS = [201, 202, 203, 204, 205]          # predetermined 19 Oct, generated for this study (jobs_bv2_test.txt), never examined
ndtr = lambda x: 0.5 * (1.0 + np.frompyfunc(math.erf, 1, 1)(np.asarray(x, float) / math.sqrt(2.0)).astype(float))
M.VIEW_KEYS = M.VIEW_KEYS + ("rate",)           # rule view additionally holds the PAST mining rate (episodes < ep only)


def add_rate(st):
    """rate[ep, j] = (number of episodes in 3..ep-1 where j mined + 1) / (ep - 3 + 1): uses the past only."""
    E = st["E"]; mined = st["mined"].astype(float); rate = np.ones((E + 2, N))
    cs = np.cumsum(mined, axis=0)
    for ep in range(3, E + 1):
        past = cs[ep - 1] - cs[2]
        rate[ep] = (past + 1.0) / (ep - 3 + 1.0)
    st["rate"] = rate
    return st


# ----------------------------------------------------------------------------- Sybil prior (logistic regression on 7 features)
def prior_rows(st):
    X, y = [], []
    for ep in range(3, st["E"] + 1):
        idx = np.nonzero(st["mined"][ep])[0]
        if len(idx) == 0: continue
        F = M.feats(st, ep, idx, np.zeros(N), np.zeros(N))[:, :7]
        X.append(F); y.append(st["syb"][idx].astype(float))
    return np.vstack(X), np.concatenate(y)


def fit_prior(train_streams, l2=1e-2, iters=15):
    Xs, ys = zip(*[prior_rows(s) for s in train_streams]); X = np.vstack(Xs); y = np.concatenate(ys)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30); p = 1 / (1 + np.exp(-z)); g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


class VoiRule(Rule):
    """Myopic value of one more check for each unresolved contributor, times the expected remaining contributions.
    Posterior Sybil logit = prior logit (fitted) + running LLR.  One check moves the LLR by -d*v:
    Sybil -> N(L + d^2/2, d^2), honest -> N(L - d^2/2, d^2).  A confirmation (|LLR| > A) is permanent.
    Default decision for an unresolved contributor is the shared rule: merge iff not CATS-latched.
      not latched: confirming a Sybil saves lam*q per future contribution;   latched: confirming an honest agent restores q.
    index = R * [ p * P(cross +A | Sybil) * lam * q * (not latched) + (1-p) * P(cross -A | honest) * q * latched ] - c,
    R = (E - ep) * past mining rate, q = window-15 mean quality.  No free parameters."""
    def __init__(self, w, seed): super().__init__("VOI", seed); self.w = w

    def choose(self, view, ep, idx, X, last, k):
        d, A, lam, c = CFG["dprime"], CFG["A"], CFG["lam"], CFG["c"]
        L = 3.0 * X[:, 8]
        p = 1.0 / (1.0 + np.exp(-np.clip(X[:, :7] @ self.w + L, -30, 30)))
        lat = X[:, 4] > 0.5
        q = np.clip(X[:, 5], 0.0, 1.0)
        R = (view["E"] - ep) * view["rate"][ep, idx]
        p_syb = 1.0 - ndtr((A - L - d * d / 2.0) / d)
        p_hon = ndtr((-A - L + d * d / 2.0) / d)
        gain = R * (p * p_syb * lam * q * (~lat) + (1.0 - p) * p_hon * q * lat) - c
        sc = gain + self.rng.random(len(idx)) * 1e-9
        return list(np.argsort(-sc)[:k])


# ----------------------------------------------------------------------------- evaluation
def verdict(rel, lo, hi, ok_hd, ok_sm, headroom):
    if headroom < 0.02: return "NO-HEADROOM"
    if rel.mean() >= 0.10 and lo > 0 and ok_hd and ok_sm: return "PASS-family"
    if lo <= 0 and hi >= 0.10: return "INCONCLUSIVE-family"
    return "FAIL-family"


def run_budget(B, train, test, held_list):
    CFG.update(B=B); out = {}
    for held in held_list:
        if held not in test or not test[held]: continue
        tr = [s for f, v in train.items() if f != held for s in v]; te = test[held]; ns = len(te)
        w = fit_prior(tr); wl = M.fit_tuned_lin(tr, B)
        names = RULES + ["UNC", "TUNED-LIN", "VOI", "HARD", "ALL", "ORACLE"]
        per = {k: [] for k in names}
        for si, st in enumerate(te):
            for k in RULES: per[k].append(M.simulate3(st, "verify", policy=Rule(k, 11 + si), B=B))
            per["UNC"].append(M.simulate3(st, "verify", policy=M.UncRule("UNC", 11 + si), B=B))
            per["TUNED-LIN"].append(M.simulate3(st, "verify", policy=M.LinRule(wl, 13 + si), B=B))
            per["VOI"].append(M.simulate3(st, "verify", policy=VoiRule(w, 17 + si), B=B))
            for k in ("HARD", "ALL", "ORACLE"): per[k].append(M.simulate3(st, k, B=B))
        F = ("U", "hd", "sm", "checks", "poison", "cost", "honest_value")
        res = {k: {f: np.array([r[f] for r in per[k]]) for f in F} for k in names}
        rules = RULES + ["UNC", "TUNED-LIN"]
        best = max(rules, key=lambda k: res[k]["U"].mean())
        reg = lambda k: res["ORACLE"]["U"] - res[k]["U"]
        headroom = float(reg(best).mean() / abs(res[best]["U"].mean()))
        o = dict(streams=ns, best_rule=best, headroom=headroom, prior_w=w.tolist(), means={k: {f: float(v[f].mean()) for f in F} for k, v in res.items()})
        print("\n=== B=%d  held-out family: %s (n=%d independent streams) ===" % (B, held, ns))
        print("%-10s %9s %9s %9s %9s %9s %9s" % ("policy", "net U", "regret", "poison", "cost", "honestDrop", "sybKept"))
        for k in ["ALL", "HARD"] + rules + ["VOI", "ORACLE"]:
            m = o["means"][k]; print("%-10s %9.1f %9.1f %9.1f %9.1f %9.3f %9.3f" % (k, m["U"], float(reg(k).mean()), m["poison"], m["cost"], m["hd"], m["sm"]))
        print("strongest rule: %s | its regret / U (headroom) %.3f" % (best, headroom))
        # primary: relative regret reduction vs the strongest rule; secondary: vs SUS and BND, and the old +5% net-utility criterion
        for ref in [best, "SUS", "BND"]:
            rr = reg(ref); rv = reg("VOI"); ok = rr > 1e-9
            rel = (rr[ok] - rv[ok]) / rr[ok]; lo, hi = boot_ci(rel)
            ok_hd = res["VOI"]["hd"].mean() - res[ref]["hd"].mean() <= 0.02; ok_sm = res["VOI"]["sm"].mean() - res[ref]["sm"].mean() <= 0.02
            vd = verdict(rel, lo, hi, ok_hd, ok_sm, headroom)
            tag = "PRIMARY" if ref == best else "secondary"
            print("  [%s] VOI vs %s: regret reduction %.3f CI [%.3f, %.3f] honest-ok %s sybil-ok %s -> %s" % (tag, ref, rel.mean(), lo, hi, ok_hd, ok_sm, vd))
            o["vs_" + ref] = dict(rel=float(rel.mean()), lo=lo, hi=hi, verdict=vd, primary=(ref == best))
        du = (res["VOI"]["U"] - res[best]["U"]) / np.abs(res[best]["U"])
        print("  [old criterion] net-utility change vs %s: %.4f (needs +0.05)" % (best, du.mean()))
        out[held] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--held", nargs="+", default=HELD_OUT)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print("lm4_run.py sha256:", hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16], "| lm3_run.py sha256:", hashlib.sha256(open(M.__file__, "rb").read()).hexdigest()[:16], flush=True)
    if a.smoke:
        M.FIXED.update(cem_iters=1, cem_pop=4, cem_streams=2)
        train = M.load_set([7], True); test = M.load_set([7], True); a.held = [h for h in a.held if h in test] or ["quality-matching"]
    else:
        train = M.load_set(TRAIN_SEEDS); test = M.load_set(TEST_SEEDS)
    for d in (train, test):
        for v in d.values():
            for s in v: add_rate(s)
    ref = next(iter(train.values()))[0]
    print("LEAK CHECK:", M.leak_check(ref), "| train streams", {f: len(v) for f, v in train.items()}, "| test streams", {f: len(v) for f, v in test.items()}, flush=True)
    res = {}
    for B in a.budgets:
        res[B] = run_budget(B, train, test, a.held)
        os.makedirs("results", exist_ok=True)
        json.dump(res[B], open("results/lm4_results_B%d%s.json" % (B, "_smoke" if a.smoke else ""), "w"))
        n_pass = sum(1 for h, o in res[B].items() if o["vs_" + o["best_rule"]]["verdict"] == "PASS-family")
        print("B=%d: VOI primary criterion passed in %d of %d families" % (B, n_pass, len(res[B])), flush=True)


if __name__ == "__main__":
    main()
