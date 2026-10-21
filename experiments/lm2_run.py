"""Budgeted verification allocation (pre-registered: preregistration_budgeted_verification_2026-10-16.md).

Replay of the uncensored shadow-mode streams results/fs/<regime>_s<seed>.json.  Every episode a shared budget of B checks may be
spent on contributors that mined in that episode and are still unresolved.  A check returns a noisy OBSERVABLE score
v = +/-d'/2 + z (never a label); a shared status rule turns accumulated log-likelihood ratios into confirmed-Sybil / confirmed-honest.
Policies only decide WHOM to check.  Learned policy: Plackett-Luce softmax over linear scores, REINFORCE on the discounted
pool-level reward R_t = U_t + N(0, sigR^2) (no per-contributor labels).  True labels are used only to score outcomes.
Leave-one-family-out: pre-train on the other families, test on the held-out family (frozen LRN-F, or continuing online LRN-O).

usage: python experiments/lm2_run.py [--seeds 5] [--dprime 2] [--budget 10] [--cost 0.1] [--quick]   -> results/lm2_results.json
"""
import json, glob, os, sys, argparse, copy, time, zlib
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
HELD_OUT = ["quality-matching", "camouflage", "benign-community"]     # frozen in the pre-registration
THR_S, THR_C, THR_SUS, WIN = 0.5, 0.3, 0.55, 15
RULES = ["RANDOM", "RR", "SUS", "FLAG", "BND"]
CFG = dict(B=10, c=0.10, dprime=2.0, A=float(np.log(9)), lam=1.0, sigR=2.0, gamma=0.95, lr=0.01, ema=0.05, passes=3, chunk=25, dim=10)
CACHE = "results/lm2_cache"


# ----------------------------------------------------------------------------- data
def load_stream(path):
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, os.path.basename(path).replace(".json", ".npz"))
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        return {k: z[k].item() if z[k].shape == () else z[k] for k in z.files}
    d, E, rec, g1, diff = FA.load(path)
    FA.COMM.clear(); FA.COMM.update(d.get("comm") or [])
    syb = np.zeros(N, bool)
    for a in (d["sybil_ids"] or range(70, 100)): syb[int(a)] = True
    S, C, SUS = FA.cats_replay(rec, E, g1, diff, 3)
    M, cnt = FA.episode_means(rec, E, 3)                                  # [N, E+2]
    mined = (cnt > 0).T                                                   # [E+2, N]
    q = np.nan_to_num(M, nan=0.0).T
    cond = (S > THR_S) & (C > THR_C) & (SUS > THR_SUS)
    cond[:3] = False
    latched = np.maximum.accumulate(cond, axis=0)
    mq = np.full((E + 2, N), 0.5); sd = np.zeros((E + 2, N))
    for ep in range(3, E + 1):
        h = M[:, max(1, ep - WIN):ep]
        with np.errstate(all="ignore"):
            m = np.nanmean(h, axis=1); s = np.nanstd(h, axis=1)
        mq[ep] = np.where(np.isnan(m), 0.5, m); sd[ep] = np.where(np.isnan(s), 0.0, s)
    zz = np.random.default_rng(zlib.crc32(os.path.basename(path).encode())).standard_normal((E + 2, N))
    out = dict(E=E, syb=syb, mined=mined, q=q, S=S, C=C, SUS=SUS, latched=latched, mq=mq, sd=sd, z=zz, name=os.path.basename(path))
    np.savez(cp, **out)
    return out


# ----------------------------------------------------------------------------- policies
def feats(st, ep, idx, LLR, nchk):
    E = st["E"]
    return np.column_stack([np.ones(len(idx)), st["S"][ep, idx], st["C"][ep, idx], st["SUS"][ep, idx], st["latched"][ep, idx].astype(float),
                            st["mq"][ep, idx], 5 * st["sd"][ep, idx], nchk[idx] / 5.0, np.clip(LLR[idx], -3, 3) / 3.0, np.full(len(idx), ep / E)])


class Rule:
    def __init__(self, kind, seed): self.kind, self.rng = kind, np.random.default_rng(seed)

    def choose(self, st, ep, idx, X, last, k):
        tie = self.rng.random(len(idx)) * 1e-6
        if self.kind == "RANDOM": sc = self.rng.random(len(idx))
        elif self.kind == "RR": sc = -last[idx] + tie
        elif self.kind == "SUS": sc = st["SUS"][ep, idx] + tie
        elif self.kind == "FLAG": sc = st["latched"][ep, idx] * 10.0 + st["SUS"][ep, idx] + tie
        elif self.kind == "BND": sc = -np.abs(st["SUS"][ep, idx] - THR_SUS) + tie
        return list(np.argsort(-sc)[:k])


class Learner:
    def __init__(self, seed):
        self.w = np.zeros(CFG["dim"]); self.rng = np.random.default_rng(seed); self.b = 0.0

    def choose(self, X, k, explore):
        z = X @ self.w
        if explore: z = z + self.rng.gumbel(size=len(z))
        return list(np.argsort(-z)[:k])

    def grad_episode(self, X, chosen):
        """Plackett-Luce score-function gradient of the ordered sample `chosen`."""
        z = X @ self.w; rem = np.ones(len(z), bool); g = np.zeros_like(self.w)
        for i in chosen:
            zr = np.where(rem, z, -np.inf); p = np.exp(zr - zr.max()); p /= p.sum()
            g += X[i] - p @ X
            rem[i] = False
        return g

    def update(self, buf):
        """buf: list of (X, chosen, reward). Discounted returns within the chunk, standardised advantages, clipped step."""
        if not buf: return
        R = np.array([r for _, _, r in buf]); G = np.zeros_like(R); run = 0.0
        for t in range(len(R) - 1, -1, -1): run = R[t] + CFG["gamma"] * run; G[t] = run
        self.b = G.mean() if self.b == 0.0 else (1 - CFG["ema"]) * self.b + CFG["ema"] * G.mean()
        adv = G - self.b; adv = adv / (adv.std() + 1e-8)
        g = np.zeros_like(self.w)
        for (X, ch, _), a in zip(buf, adv): g += a * self.grad_episode(X, ch)
        g /= len(buf); n = np.linalg.norm(g)
        if n > 5: g *= 5 / n
        self.w += CFG["lr"] * g


# ----------------------------------------------------------------------------- simulation
def simulate(st, mode, policy=None, learner=None, learn=False, B=None, c=None, d=None):
    """mode: 'ALL' | 'HARD' | 'ORACLE' | 'verify'."""
    B = CFG["B"] if B is None else B; c = CFG["c"] if c is None else c; d = CFG["dprime"] if d is None else d
    A, lam, sigR = CFG["A"], CFG["lam"], CFG["sigR"]
    E, syb = st["E"], st["syb"]
    LLR = np.zeros(N); nchk = np.zeros(N); last = np.full(N, -1e9); status = np.zeros(N, int)
    U = 0.0; checks = 0; hd = 0.0; nh = 0.0; sm = 0.0; ns = 0.0
    nrng = np.random.default_rng(zlib.crc32(st["name"].encode()) + 17)
    buf = []
    for ep in range(3, E + 1):
        mined = st["mined"][ep]; lat = st["latched"][ep]
        nchosen = 0; chosen_idx = []; X = None; elig = None
        if mode == "verify" and B > 0:
            elig = np.nonzero(mined & (status == 0))[0]
            if len(elig):
                k = min(B, len(elig)); X = feats(st, ep, elig, LLR, nchk)
                if isinstance(policy, Rule): ch = policy.choose(st, ep, elig, X, last, k)
                else: ch = learner.choose(X, k, explore=learn)
                chosen_idx = ch
                for i in ch:
                    j = elig[i]
                    v = (-d / 2 if syb[j] else d / 2) + st["z"][ep, j]
                    LLR[j] += -d * v; nchk[j] += 1; last[j] = ep
                    if LLR[j] > A: status[j] = 1
                    elif LLR[j] < -A: status[j] = -1
                nchosen = len(ch)
        if mode == "ALL": merged = mined
        elif mode == "ORACLE": merged = mined & ~syb
        else: merged = mined & ((status == -1) | ((status == 0) & ~lat))
        q = st["q"][ep]
        val = np.where(syb, -lam * q, q)
        Ut = float((val * merged).sum()) - c * nchosen
        U += Ut; checks += nchosen
        nh += (mined & ~syb).sum(); hd += (mined & ~syb & ~merged).sum()
        ns += (mined & syb).sum(); sm += (mined & syb & merged).sum()
        if learn and X is not None:
            buf.append((X, chosen_idx, Ut + sigR * nrng.standard_normal()))
            if len(buf) >= CFG["chunk"]: learner.update(buf); buf = []
    if learn and buf: learner.update(buf)
    return dict(U=U, checks=checks, hd=hd / max(nh, 1), sm=sm / max(ns, 1))


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dprime", type=float, default=2.0)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--cost", type=float, default=0.10)
    ap.add_argument("--passes", type=int, default=CFG["passes"])
    ap.add_argument("--quick", action="store_true", help="smoke test: 60 episodes of each stream, 1 pass")
    a = ap.parse_args()
    CFG.update(B=a.budget, c=a.cost, dprime=a.dprime, passes=a.passes)
    data = {}
    t0 = time.time()
    for fam, regs in FAMILIES.items():
        for rg in regs:
            for p in sorted(glob.glob("results/fs/%s_s[0-9]*.json" % rg)):
                st = load_stream(p)
                if a.quick:
                    E = 60
                    for kk in ("mined", "q", "S", "C", "SUS", "latched", "mq", "sd", "z"): st[kk] = st[kk][:E + 2]
                    st["E"] = E
                data.setdefault(fam, []).append(st)
    print("streams:", {f: len(v) for f, v in data.items()}, "loaded in %.0fs" % (time.time() - t0), "cfg", {k: CFG[k] for k in ("B", "c", "dprime")}, flush=True)
    allres = {}
    gate = {}
    for held in HELD_OUT:
        if held not in data: continue
        train_streams = [s for f, v in data.items() if f != held for s in v]
        test = data[held]
        per = {k: [] for k in RULES + ["HARD", "ALL", "ORACLE", "LRN-F", "LRN-O"]}
        gate_rows = []
        for ls in range(a.seeds):
            L = Learner(1000 + ls)
            rng = np.random.default_rng(2000 + ls)
            for _ in range(CFG["passes"]):
                for i in rng.permutation(len(train_streams)): simulate(train_streams[i], "verify", learner=L, learn=True)
            # sanity gate: pre-trained greedy learner vs RANDOM on the training streams
            gl = np.mean([simulate(s, "verify", learner=L, learn=False)["U"] for s in train_streams])
            gr = np.mean([simulate(s, "verify", policy=Rule("RANDOM", 5))["U"] for s in train_streams])
            gate_rows.append((gl, gr))
            for si, st in enumerate(test):
                if ls == 0:
                    for k in RULES: per[k].append([simulate(st, "verify", policy=Rule(k, 11 + si))])
                    for k in ("HARD", "ALL", "ORACLE"): per[k].append([simulate(st, k)])
                per["LRN-F"].append((si, simulate(st, "verify", learner=L, learn=False)))
                Lo = copy.deepcopy(L); Lo.rng = np.random.default_rng(3000 + ls * 100 + si)
                per["LRN-O"].append((si, simulate(st, "verify", learner=Lo, learn=True)))
        gate[held] = (float(np.mean([g[0] for g in gate_rows])), float(np.mean([g[1] for g in gate_rows])))
        # aggregate: per-stream metrics (learner seeds averaged)
        ns = len(test)
        def stream_metric(key, field):
            if key in ("LRN-F", "LRN-O"):
                return np.array([np.mean([r[field] for (si, r) in per[key] if si == s]) for s in range(ns)])
            return np.array([per[key][s][0][field] for s in range(ns)])
        res = {k: {f: stream_metric(k, f) for f in ("U", "hd", "sm", "checks")} for k in per}
        allres[held] = {k: {f: v.tolist() for f, v in d.items()} for k, d in res.items()}
        best = max(RULES, key=lambda k: res[k]["U"].mean())
        print("\n=== held-out family: %s (n=%d streams) ===" % (held, ns))
        print("%-8s %9s %7s %7s %7s" % ("policy", "net U", "honestDrop", "sybKept", "checks"))
        for k in ["ALL", "HARD"] + RULES + ["LRN-F", "LRN-O", "ORACLE"]:
            print("%-8s %9.1f %7.3f %7.3f %7.0f" % (k, res[k]["U"].mean(), res[k]["hd"].mean(), res[k]["sm"].mean(), res[k]["checks"].mean()))
        print("strongest rule on this family: %s ; sanity gate (train streams): LRN-F %.1f vs RANDOM %.1f" % (best, gate[held][0], gate[held][1]))
        for k in ("LRN-F", "LRN-O"):
            rel = (res[k]["U"] - res[best]["U"]) / np.abs(res[best]["U"])
            lo, hi = boot_ci(rel)
            ok_hd = res[k]["hd"].mean() - res[best]["hd"].mean() <= 0.02
            ok_sm = res[k]["sm"].mean() - res[best]["sm"].mean() <= 0.02
            verdict = "PASS-family" if (rel.mean() >= 0.05 and lo > 0 and ok_hd and ok_sm) else ("INCONCLUSIVE-family" if (lo <= 0 and hi >= 0.05) else "FAIL-family")
            print("  %s vs %s: relative improvement %.3f  95%% CI [%.3f, %.3f]  honest-ok %s sybil-ok %s -> %s" % (k, best, rel.mean(), lo, hi, ok_hd, ok_sm, verdict))
            allres[held]["verdict_" + k] = dict(rel=float(rel.mean()), lo=lo, hi=hi, verdict=verdict, best_rule=best)
    sane = all(g[0] > g[1] for g in gate.values())
    print("\nSANITY GATE (learner beats RANDOM on training streams in every fold): %s" % sane)
    for k in ("LRN-F", "LRN-O"):
        n_pass = sum(1 for h in HELD_OUT if h in allres and allres[h]["verdict_" + k]["verdict"] == "PASS-family")
        n_inc = sum(1 for h in HELD_OUT if h in allres and allres[h]["verdict_" + k]["verdict"] == "INCONCLUSIVE-family")
        overall = "INCONCLUSIVE (sanity gate failed)" if not sane else ("PASS" if n_pass >= 2 else ("INCONCLUSIVE" if n_inc else "FAIL"))
        print("OVERALL %s: %d of %d held-out families pass -> %s" % (k, n_pass, len(allres), overall))
    os.makedirs("results", exist_ok=True)
    json.dump(dict(cfg={k: CFG[k] for k in CFG}, gate=gate, results=allres), open("results/lm2_results.json", "w"))


if __name__ == "__main__":
    main()
