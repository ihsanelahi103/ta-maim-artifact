"""Binding-budget verification allocation (pre-registered: preregistration_binding_budget_2026-10-18.md).

Reuses experiments/lm2_run.py (streams, features, shared status rule, utility, rules, PL-REINFORCE learner LRN).
New here: (i) binding budgets B in {1,2,5,20} with a binding diagnostic, (ii) TA-DDQN-V = a Double-DQN item-scoring allocator
(the SAME learner family and hyper-parameters as the manuscript's TA-DDQN, NOT its trained policy), (iii) two more equally informed
rules (UNC, TUNED-LIN), (iv) INDEPENDENT test streams (seeds 101-105, generated for this study), pre-training on seeds 7,11,42,99,123 of the OTHER families.

usage:  python experiments/lm3_run.py --budgets 1 2 5 20          (full protocol, writes results/lm3_results_B<b>.json + lm3_summary.json)
        python experiments/lm3_run.py --smoke                      (execution check on whatever seed-7 streams exist; numbers are NOT results)
"""
import json, glob, os, sys, argparse, copy, time, zlib, hashlib
from collections import deque
import numpy as np
sys.path.insert(0, os.getcwd())
from experiments import lm2_run as L2
from experiments.lm2_run import (FAMILIES, HELD_OUT, RULES, CFG, Rule, Learner, load_stream, feats, boot_ci, N, THR_SUS)
import torch, torch.nn as nn

TRAIN_SEEDS = [7, 11, 42, 99, 123]
TEST_SEEDS = [101, 102, 103, 104, 105]       # predetermined, never generated or examined before this protocol (seeds 1-5 were used for the detector replication)
RULES_ALL = RULES + ["UNC", "TUNED-LIN"]
FIXED = dict(passes_lrn=10, passes_dqn=5, cem_iters=8, cem_pop=12, cem_streams=8, dqn_steps_per_episode=1)


# ----------------------------------------------------------------------------- extra rules
class UncRule(Rule):
    """Uncertainty sampling: check where the posterior log-odds (prior logit(suspicion) + running LLR) is closest to 0."""
    def choose(self, st, ep, idx, X, last, k):
        tie = self.rng.random(len(idx)) * 1e-6
        prior = np.log(np.clip(X[:, 3], .05, .95) / (1 - np.clip(X[:, 3], .05, .95)))
        sc = -np.abs(prior + 3.0 * X[:, 8]) + tie
        return list(np.argsort(-sc)[:k])


class LinRule(Rule):
    """Linear score on the same 10 features with weights fitted on training streams by CEM (uses the true net utility of the
    TRAINING streams: more informed than the learners, a conservative benchmark)."""
    def __init__(self, w, seed): super().__init__("TUNED-LIN", seed); self.w = w
    def choose(self, st, ep, idx, X, last, k):
        sc = X @ self.w + self.rng.random(len(idx)) * 1e-6
        return list(np.argsort(-sc)[:k])


def fit_tuned_lin(train_streams, B, seed=0):
    rng = np.random.default_rng(seed)
    sub = [train_streams[i] for i in rng.permutation(len(train_streams))[:FIXED["cem_streams"]]]
    mu, sd = np.zeros(CFG["dim"]), np.ones(CFG["dim"]) * 1.5
    def score(w): return np.mean([simulate3(s, "verify", policy=LinRule(w, 3), B=B)["U"] for s in sub])
    best_w, best_s = mu.copy(), score(mu)
    for _ in range(FIXED["cem_iters"]):
        pop = [mu + sd * rng.standard_normal(CFG["dim"]) for _ in range(FIXED["cem_pop"])]
        sc = np.array([score(w) for w in pop]); elite = np.argsort(-sc)[:max(2, FIXED["cem_pop"] // 4)]
        mu = np.mean([pop[i] for i in elite], axis=0); sd = np.std([pop[i] for i in elite], axis=0) + 0.05
        if sc[elite[0]] > best_s: best_s, best_w = sc[elite[0]], pop[elite[0]].copy()
    return best_w


# ----------------------------------------------------------------------------- TA-DDQN-V (Double DQN item scorer)
class DDQNAlloc:
    """Double DQN over 'which contributor to check'. Q(x_j) is a shared MLP (10 -> 128 -> 128 -> 1) on the contributor features.
    Hyper-parameters copied from the manuscript's TA-DDQN: Adam lr 1e-3, gamma .95, batch 64, buffer 5000, soft target tau .01,
    eps-greedy 1.0 -> .05 with decay .995 per episode. Reward = the pool-level noisy validation reward (EMA-centred and scaled),
    split equally among the checks of that episode. Next state = the candidate set of the next episode; target uses
    Double-DQN selection (online net argmax over candidates, target net value). One gradient step per episode."""
    def __init__(self, seed):
        torch.manual_seed(seed); torch.set_num_threads(1)
        self.rng = np.random.default_rng(seed)
        mk = lambda: nn.Sequential(nn.Linear(CFG["dim"], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.q, self.qt = mk(), mk(); self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=1e-3)
        self.buf = deque(maxlen=5000); self.eps = 1.0; self.pending = []
        self.rm, self.rv, self.rn = 0.0, 1.0, 0
        self.gamma, self.tau, self.bs = 0.95, 0.01, 64

    def _scores(self, X):
        with torch.no_grad(): return self.q(torch.as_tensor(X, dtype=torch.float32)).squeeze(1).numpy()

    def choose(self, X, k, explore):
        self._complete(X, False)
        if explore and self.rng.random() < self.eps: return list(self.rng.permutation(len(X))[:k])
        return list(np.argsort(-self._scores(X))[:k])

    def _complete(self, Xn, done):
        for (x, r) in self.pending: self.buf.append((x, r, Xn.astype(np.float32), done))
        self.pending = []

    def observe(self, X, chosen, R):
        self.rn += 1; a = 0.05 if self.rn > 1 else 1.0
        d = R - self.rm; self.rm += a * d; self.rv = (1 - a) * self.rv + a * d * d
        r = (R - self.rm) / (np.sqrt(self.rv) + 1e-6) / max(len(chosen), 1)
        self.pending = [(X[i].astype(np.float32), float(r)) for i in chosen]
        self.eps = max(0.05, self.eps * 0.995)
        for _ in range(FIXED["dqn_steps_per_episode"]): self._train()

    def finish(self):
        self._complete(np.zeros((1, CFG["dim"]), np.float32), True)

    def _train(self):
        if len(self.buf) < self.bs: return
        ix = self.rng.integers(0, len(self.buf), self.bs); b = [self.buf[i] for i in ix]
        x = torch.as_tensor(np.stack([t[0] for t in b])); r = torch.as_tensor([t[1] for t in b], dtype=torch.float32)
        dn = torch.as_tensor([float(t[3]) for t in b], dtype=torch.float32)
        L = max(len(t[2]) for t in b)
        Xn = torch.zeros(self.bs, L, CFG["dim"]); mask = torch.zeros(self.bs, L, dtype=torch.bool)
        for i, t in enumerate(b): Xn[i, :len(t[2])] = torch.as_tensor(t[2]); mask[i, :len(t[2])] = True
        with torch.no_grad():
            qo = self.q(Xn).squeeze(2).masked_fill(~mask, -1e9); a = qo.argmax(1)
            qn = self.qt(Xn).squeeze(2).gather(1, a[:, None]).squeeze(1)
            y = r + self.gamma * (1 - dn) * qn
        loss = nn.functional.smooth_l1_loss(self.q(x).squeeze(1), y)
        self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.q.parameters(), 5.0); self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.qt.parameters()): pt.mul_(1 - self.tau).add_(self.tau * p)


# ----------------------------------------------------------------------------- simulation (lm2.simulate + hooks + extra metrics)
# Information structure (offline replay of exogenous streams): the stream (graph, qualities, CATS signals, who mines when) is fixed and
# is NOT changed by any decision. Decisions change only the verification state (LLR, status, eligibility, number of checks) and the merge.
# Policies see: features computed from data through episode ep-1 (CATS S/C/suspicion, latch, window-15 mean/std of PAST raw quality),
# their own check history and running LLR, and the episode fraction. Learners additionally receive the pool-level noisy reward.
# Labels enter only (i) the modelled verifier's score v (observable, noisy), (ii) the utility that scores outcomes, (iii) the ORACLE reference
# (headroom only; never shown to a policy), (iv) the TRAINING-stream utility used to fit TUNED-LIN (a more informed benchmark).
VIEW_KEYS = ("E", "S", "C", "SUS", "latched", "mq", "sd")


def leak_check(st):
    """Structural audit: the features of a policy are unchanged if labels, the verifier noise and the raw qualities of episodes >= ep are replaced."""
    rng = np.random.default_rng(0); st2 = dict(st)
    st2["syb"] = ~st["syb"]; st2["z"] = rng.standard_normal(st["z"].shape)
    for ep in (5, 40, min(st["E"], 120)):
        q2 = st["q"].copy(); q2[ep:] = rng.random(q2[ep:].shape); st2["q"] = q2
        idx = np.arange(N)
        a = feats(st, ep, idx, np.zeros(N), np.zeros(N)); b = feats(st2, ep, idx, np.zeros(N), np.zeros(N))
        assert np.array_equal(a, b), "feature leak at episode %d" % ep
    return True
def simulate3(st, mode, policy=None, learner=None, learn=False, B=None, c=None, d=None):
    B = CFG["B"] if B is None else B; c = CFG["c"] if c is None else c; d = CFG["dprime"] if d is None else d
    A, lam, sigR = CFG["A"], CFG["lam"], CFG["sigR"]
    E, syb = st["E"], st["syb"]
    LLR = np.zeros(N); nchk = np.zeros(N); last = np.full(N, -1e9); status = np.zeros(N, int)
    U = 0.0; checks = 0; hd = nh = sm = ns = 0.0; hv = pz = 0.0
    nrng = np.random.default_rng(zlib.crc32(st["name"].encode()) + 17)
    buf = []
    view = {k: st[k] for k in VIEW_KEYS}          # what a RULE may touch: no labels, no raw quality of the current/future episodes, no verifier noise
    for ep in range(3, E + 1):
        mined = st["mined"][ep]; lat = st["latched"][ep]
        nchosen = 0; chosen_idx = []; X = None
        if mode == "verify" and B > 0:
            elig = np.nonzero(mined & (status == 0))[0]
            if len(elig):
                k = min(B, len(elig)); X = feats(st, ep, elig, LLR, nchk)
                if isinstance(policy, Rule): ch = policy.choose(view, ep, elig, X, last, k)
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
        hvt = float((q * (merged & ~syb)).sum()); pzt = float((lam * q * (merged & syb)).sum())
        Ut = hvt - pzt - c * nchosen
        U += Ut; checks += nchosen; hv += hvt; pz += pzt
        nh += (mined & ~syb).sum(); hd += (mined & ~syb & ~merged).sum()
        ns += (mined & syb).sum(); sm += (mined & syb & merged).sum()
        if learn and X is not None:
            Rn = Ut + sigR * nrng.standard_normal()
            if hasattr(learner, "observe"): learner.observe(X, chosen_idx, Rn)
            else:
                buf.append((X, chosen_idx, Rn))
                if len(buf) >= CFG["chunk"]: learner.update(buf); buf = []
    if learn and hasattr(learner, "finish"): learner.finish()
    if learn and buf: learner.update(buf)
    return dict(U=U, checks=checks, hd=hd / max(nh, 1), sm=sm / max(ns, 1), poison=pz, cost=c * checks, honest_value=hv)


def coverage(st, B, d=None):
    """Binding diagnostic under RANDOM allocation: eligible unresolved per episode, unresolved share at episodes 50/100/E, checks per contributor."""
    d = CFG["dprime"] if d is None else d
    A = CFG["A"]; syb, E = st["syb"], st["E"]; LLR = np.zeros(N); status = np.zeros(N, int); seen = np.zeros(N, bool)
    rule = Rule("RANDOM", 5); nel = []; unres = {}; tot = 0
    for ep in range(3, E + 1):
        mined = st["mined"][ep]; seen |= mined
        elig = np.nonzero(mined & (status == 0))[0]; nel.append(len(elig))
        if len(elig) and B > 0:
            for i in rule.choose(st, ep, elig, None, np.zeros(N), min(B, len(elig))):
                j = elig[i]; v = (-d / 2 if syb[j] else d / 2) + st["z"][ep, j]
                LLR[j] += -d * v; tot += 1
                status[j] = 1 if LLR[j] > A else (-1 if LLR[j] < -A else 0)
        if ep in (50, 100) or ep == E: unres[ep if ep != E else "E"] = float(((status == 0) & seen).sum() / max(seen.sum(), 1))
    for kk in (50, 100):
        if kk not in unres: unres[kk] = unres["E"]          # short (smoke) streams only
    return dict(mean_eligible=float(np.mean(nel)), unresolved_50=unres.get(50), unresolved_100=unres.get(100), unresolved_E=unres.get("E"),
                checks_per_contributor=tot / max(seen.sum(), 1))


# ----------------------------------------------------------------------------- data
def load_set(seeds, smoke=False):
    data = {}
    for fam, regs in FAMILIES.items():
        for rg in regs:
            for s in seeds:
                p = "results/fs/%s_s%d.json" % (rg, s)
                if not os.path.exists(p): continue
                st = load_stream(p)
                if smoke:
                    E = 60
                    for kk in ("mined", "q", "S", "C", "SUS", "latched", "mq", "sd", "z"): st[kk] = st[kk][:E + 2]
                    st["E"] = E
                data.setdefault(fam, []).append(st)
    return data


def verdict(rel, lo, hi, ok_hd, ok_sm, headroom):
    if headroom < 0.05: return "NO-HEADROOM"
    if rel.mean() >= 0.05 and lo > 0 and ok_hd and ok_sm: return "PASS-family"
    if lo <= 0 and hi >= 0.05: return "INCONCLUSIVE-family"
    return "FAIL-family"


def run_budget(B, train, test, a):
    CFG.update(B=B)
    out = {}; gates = {}
    for held in a.held:
        if held not in test or not test[held]: continue
        tr = [s for f, v in train.items() if f != held for s in v]
        te = test[held]; ns = len(te)
        t0 = time.time()
        per = {k: [None] * ns for k in RULES_ALL + ["HARD", "ALL", "ORACLE"]}
        for k in ("LRN-F", "LRN-O", "DDQN-F", "DDQN-O"): per[k] = [[] for _ in range(ns)]
        for si, st in enumerate(te):
            for k in RULES: per[k][si] = simulate3(st, "verify", policy=(UncRule if k == "UNC" else Rule)(k, 11 + si), B=B)
            per["UNC"][si] = simulate3(st, "verify", policy=UncRule("UNC", 11 + si), B=B)
            for k in ("HARD", "ALL", "ORACLE"): per[k][si] = simulate3(st, k, B=B)
        w = fit_tuned_lin(tr, B)
        for si, st in enumerate(te): per["TUNED-LIN"][si] = simulate3(st, "verify", policy=LinRule(w, 13 + si), B=B)
        gl = {"LRN": [], "DDQN": [], "RANDOM": []}
        for ls in range(a.seeds):
            rng = np.random.default_rng(2000 + ls)
            Lr = Learner(1000 + ls); Dq = DDQNAlloc(4000 + ls)
            for _ in range(a.passes_lrn):
                for i in rng.permutation(len(tr)): simulate3(tr[i], "verify", learner=Lr, learn=True, B=B)
            for _ in range(a.passes_dqn):
                for i in rng.permutation(len(tr)): simulate3(tr[i], "verify", learner=Dq, learn=True, B=B)
            gl["LRN"].append(np.mean([simulate3(s, "verify", learner=Lr, learn=False, B=B)["U"] for s in tr]))
            Dq.eps = 0.0
            gl["DDQN"].append(np.mean([simulate3(s, "verify", learner=Dq, learn=False, B=B)["U"] for s in tr]))
            gl["RANDOM"].append(np.mean([simulate3(s, "verify", policy=Rule("RANDOM", 5), B=B)["U"] for s in tr]))
            for si, st in enumerate(te):
                per["LRN-F"][si].append(simulate3(st, "verify", learner=Lr, learn=False, B=B))
                per["DDQN-F"][si].append(simulate3(st, "verify", learner=Dq, learn=False, B=B))
                Lo = copy.deepcopy(Lr); Lo.rng = np.random.default_rng(3000 + ls * 100 + si)
                per["LRN-O"][si].append(simulate3(st, "verify", learner=Lo, learn=True, B=B))
                Do = copy.deepcopy(Dq); Do.eps = 0.05
                per["DDQN-O"][si].append(simulate3(st, "verify", learner=Do, learn=True, B=B))
        gates[held] = {k: float(np.mean(v)) for k, v in gl.items()}
        fields = ("U", "hd", "sm", "checks", "poison", "cost", "honest_value")
        res = {}
        for k, lst in per.items():
            res[k] = {f: np.array([np.mean([r[f] for r in x]) if isinstance(x, list) else x[f] for x in lst]) for f in fields}
        best = max(RULES_ALL, key=lambda k: res[k]["U"].mean())
        headroom = float((res["ORACLE"]["U"].mean() - res[best]["U"].mean()) / abs(res[best]["U"].mean()))
        rnd_gap = float((res[best]["U"].mean() - res["RANDOM"]["U"].mean()) / abs(res["RANDOM"]["U"].mean()))
        o = dict(streams=ns, best_rule=best, headroom_vs_best_rule=headroom, best_rule_vs_random=rnd_gap, seconds=time.time() - t0)
        o["means"] = {k: {f: float(v[f].mean()) for f in fields} for k, v in res.items()}
        print("\n=== B=%d  held-out family: %s (n=%d fresh streams) ===" % (B, held, ns))
        print("%-10s %9s %9s %9s %9s %9s" % ("policy", "net U", "poison", "cost", "honestDrop", "sybKept"))
        for k in ["ALL", "HARD"] + RULES_ALL + ["LRN-F", "LRN-O", "DDQN-F", "DDQN-O", "ORACLE"]:
            m = o["means"][k]; print("%-10s %9.1f %9.1f %9.1f %9.3f %9.3f" % (k, m["U"], m["poison"], m["cost"], m["hd"], m["sm"]))
        print("strongest rule: %s | headroom (ORACLE vs it) %.3f | best rule vs RANDOM %.3f | gate train: LRN %.1f DDQN %.1f RANDOM %.1f" %
              (best, headroom, rnd_gap, gates[held]["LRN"], gates[held]["DDQN"], gates[held]["RANDOM"]))
        for k in ("DDQN-F", "LRN-F", "DDQN-O", "LRN-O"):
            rel = (res[k]["U"] - res[best]["U"]) / np.abs(res[best]["U"]); lo, hi = boot_ci(rel)
            ok_hd = res[k]["hd"].mean() - res[best]["hd"].mean() <= 0.02; ok_sm = res[k]["sm"].mean() - res[best]["sm"].mean() <= 0.02
            vd = verdict(rel, lo, hi, ok_hd, ok_sm, headroom)
            o["verdict_" + k] = dict(rel=float(rel.mean()), lo=lo, hi=hi, verdict=vd)
            print("  %s vs %s: rel %.4f CI [%.4f, %.4f] honest-ok %s sybil-ok %s -> %s" % (k, best, rel.mean(), lo, hi, ok_hd, ok_sm, vd))
        out[held] = o
    return out, gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 5, 20])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--passes_lrn", type=int, default=FIXED["passes_lrn"])
    ap.add_argument("--passes_dqn", type=int, default=FIXED["passes_dqn"])
    ap.add_argument("--held", nargs="+", default=HELD_OUT, help="held-out families to run (default: all three)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print("lm3_run.py sha256:", hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16], "FIXED", FIXED, flush=True)
    if a.smoke:
        a.seeds, a.passes_lrn, a.passes_dqn = 1, 1, 1; FIXED.update(cem_iters=1, cem_pop=4, cem_streams=2)
        train = load_set([7], True); test = load_set([7], True)
        HELD = [h for h in HELD_OUT if h in test]
        if not HELD: HELD = ["quality-matching"]
        if "camouflage" not in train: pass
    else:
        train = load_set(TRAIN_SEEDS); test = load_set(TEST_SEEDS)
    ref = next(iter(train.values()))[0]
    print("LEAK CHECK (features invariant to labels, verifier noise, current/future raw quality):", leak_check(ref), "| rule view keys:", VIEW_KEYS, flush=True)
    print("train streams", {f: len(v) for f, v in train.items()}, "| test streams", {f: len(v) for f, v in test.items()}, flush=True)
    summary = {}
    for B in a.budgets:
        # binding diagnostic on ALL streams (train+test) before any learner is trained
        cov = [coverage(s, B) for v in list(train.values()) + list(test.values()) for s in v]
        cd = {k: float(np.mean([c[k] for c in cov])) for k in cov[0]}
        print("\n##### B=%d binding diagnostic (RANDOM): %s" % (B, {k: round(v, 3) for k, v in cd.items()}), flush=True)
        res, gates = run_budget(B, train, test, a)
        sane = all(g["DDQN"] > g["RANDOM"] for g in gates.values())
        n_pass = sum(1 for h, o in res.items() if o["verdict_DDQN-F"]["verdict"] == "PASS-family")
        binding = cd["unresolved_100"] is not None and cd["unresolved_100"] >= 0.25
        summary[B] = dict(binding=binding, coverage=cd, sanity_gate_ddqn=sane, n_pass_DDQN_F=n_pass, results=res, gates=gates)
        print("B=%d: binding=%s | sanity gate (DDQN-V beats RANDOM on training streams in every fold)=%s | DDQN-F families passed: %d" % (B, binding, sane, n_pass), flush=True)
        os.makedirs("results", exist_ok=True)
        json.dump(summary[B], open("results/lm3_results_B%d_%s%s.json" % (B, "+".join(h[:4] for h in a.held), "_smoke" if a.smoke else ""), "w"))
    json.dump(summary, open("results/lm3_summary_B%s_%s%s.json" % ("-".join(map(str, a.budgets)), "+".join(h[:4] for h in a.held), "_smoke" if a.smoke else ""), "w"))


if __name__ == "__main__":
    main()
