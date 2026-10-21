"""Information audit of the VOI index (companion to the frozen lm4_run.py; changes nothing in the evaluation).
Checks that the index at episode ep depends only on: the 10 shared features (computed from data through ep-1), the policy's own revealed
verifier outcomes (LLR, check counts), the past mining rate, E and ep, and the fitted prior weights.  It must NOT depend on: Sybil labels,
the verifier noise z, raw qualities of episodes >= ep, mining/CATS/latch/window rows of episodes > ep.
usage: python experiments/audit_voi.py [--smoke]"""
import os, sys, inspect, copy, argparse
import numpy as np
sys.path.insert(0, os.getcwd())
from experiments import lm3_run as M
from experiments import lm4_run as L4
from experiments.lm2_run import N, CFG

ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
seeds = [7] if a.smoke else L4.TRAIN_SEEDS
data = M.load_set(seeds, a.smoke)
for v in data.values():
    for s in v: L4.add_rate(s)
streams = [s for v in data.values() for s in v]
w = L4.fit_prior(streams[:3]) if len(streams) > 3 else L4.fit_prior(streams)
rng = np.random.default_rng(0); rule = L4.VoiRule(w, 17); n_checked = 0
M.VIEW_KEYS = M.VIEW_KEYS if "rate" in M.VIEW_KEYS else M.VIEW_KEYS + ("rate",)
for st in streams[:6]:
    E = st["E"]
    for ep in sorted({5, 20, min(60, E - 1), min(150, E - 1)}):
        idx = np.nonzero(st["mined"][ep])[0]
        if len(idx) < 5: continue
        LLR = rng.uniform(-2.0, 2.0, N); nchk = rng.integers(0, 4, N).astype(float)
        view = {k: st[k] for k in M.VIEW_KEYS}
        assert "syb" not in view and "z" not in view and "q" not in view and "mined" not in view
        X = M.feats(st, ep, idx, LLR, nchk); ch = L4.VoiRule(w, 17).choose(view, ep, idx, X, np.zeros(N), 5)
        # perturb everything the index must not see
        s2 = dict(st); s2["syb"] = ~st["syb"]; s2["z"] = rng.standard_normal(st["z"].shape)
        for k in ("S", "C", "SUS", "mq", "sd", "rate"):
            arr = st[k].copy(); arr[ep + 1:] = rng.random(arr[ep + 1:].shape); s2[k] = arr
        lat = st["latched"].copy(); lat[ep + 1:] = rng.random(lat[ep + 1:].shape) > 0.5; s2["latched"] = lat
        q2 = st["q"].copy(); q2[ep:] = rng.random(q2[ep:].shape); s2["q"] = q2
        m2 = st["mined"].copy(); m2[ep + 1:] = rng.random(m2[ep + 1:].shape) > 0.5; s2["mined"] = m2
        view2 = {k: s2[k] for k in M.VIEW_KEYS}
        X2 = M.feats(s2, ep, idx, LLR, nchk); ch2 = L4.VoiRule(w, 17).choose(view2, ep, idx, X2, np.zeros(N), 5)
        assert np.array_equal(X, X2), "feature leak"
        assert ch == ch2, "VOI choice depends on labels/noise/future data"
        n_checked += 1
src = inspect.getsource(L4.VoiRule)
bad = [t for t in ('"syb"', "'syb'", '"z"', '"mined"', '["q"]', "ORACLE", "status") if t in src]
print("AUDIT PASSED: %d (stream, episode) cases; VoiRule source mentions of forbidden names: %s" % (n_checked, bad if bad else "none"))
assert not bad and n_checked > 0
