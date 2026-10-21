"""E3: quality derived from real review data feeds the UNCHANGED TA-MAIM loop (in-loop quarantine); the same poisoned
submissions are pooled for the ranking evaluated end to end. usage: DATASET=yelp RS_VARIANT=F5x3 TAMAIM_SEED=7 python experiments/rs_run.py"""
import os, sys, json, time
os.environ["TAMAIM_NO_SYBIL_PRIOR"] = "1"
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
import numpy as np
DS = os.environ.get("DATASET", "yelp"); SEED = int(os.environ.get("TAMAIM_SEED", "7")); VAR = os.environ.get("RS_VARIANT", "F5x3")
VARNAME = VAR                                   # reported name, e.g. F5x3e = attack F5x3 with signal B; trailing n = NV-C self-calibrating input
NORM = VAR.endswith("n")
if NORM: VAR = VAR[:-1]
if VAR.endswith("e"): VAR = VAR[:-1]; os.environ["RS_SIGNAL"] = "ext"
else: os.environ["RS_SIGNAL"] = "dev"
VARS = {"F5x3": dict(fab_per_ep=3, fab_stars=5.0), "F4x3": dict(fab_per_ep=3, fab_stars=4.0), "F5x1": dict(fab_per_ep=1, fab_stars=5.0)}
EPIS = int(os.environ.get("RS_EPISODES", "200"))
import rs_data as RD
D = RD.load(DS, seed=SEED, episodes=EPIS, **VARS[VAR])
Q, cov = RD.quality_table(D)
if NORM:   # pre-registered NV-C (preregistration_novelty_experiments_2026-10-17): label-free per-episode robust z-score onto the
    #        simulator's honest quality distribution U(0.65,1.0): mean 0.825, SD 0.101. Trust rule, thresholds and CATS unchanged.
    for t in range(1, Q.shape[1]):
        col = Q[:, t]; m = np.median(col); sc = max(1.4826 * np.median(np.abs(col - m)), 0.02)
        Q[:, t] = np.clip(0.825 + 0.101 * (col - m) / sc, 0.0, 1.0)
NH, NS = D["n_h"], D["n_s"]
os.environ["TAMAIM_ATTACK"] = "collusion"

from environment.edge_mining_env import EdgeMiningEnvironment as Env
from environment import sybil_attack_model as sam
CTR, CUR = [0], [None]
_reset, _mine, _unif = Env.reset, Env._execute_mine_data, np.random.uniform
def reset(self, *a, **k): CTR[0] += 1; return _reset(self, *a, **k)
def mine(self, agent_id, is_sybil): CUR[0] = agent_id; r = _mine(self, agent_id, is_sybil); CUR[0] = None; return r
def uniform(lo=0.0, hi=1.0, size=None):
    if size is None and lo == 0.65 and hi == 1.0 and CUR[0] is not None and CUR[0] < NH:
        return float(Q[CUR[0], min(CTR[0], EPIS)])
    return _unif(lo, hi, size)
Env.reset, Env._execute_mine_data = reset, mine; np.random.uniform = uniform
sam.ColludingSybilAttack.get_quality = lambda self, si, ep: float(Q[NH + si, min(CTR[0], EPIS)])

# pathway logging (diagnosis only; behaviour unchanged): which RMP pathway first quarantines each agent
from algorithms.brmp.reputation_manager import ReputationManager as RMP
PATH = {}
_sq = RMP.should_quarantine
def _pathway(self, a):
    f, rep, u = self.sybil_flag_counts[a], self.reputation_scores[a], len(self.unique_reporters[a])
    if f >= self.flag_threshold and rep < self.rep_threshold and u >= self.min_unique: return "P1"
    if hasattr(self, "_trust_scores") and self._trust_scores[a] < 0.08 and rep < 0.75: return "P2"
    return "P3"
def _sqlog(self, agent_id):
    r = _sq(self, agent_id)
    if r and agent_id not in PATH: PATH[agent_id] = [CTR[0], _pathway(self, agent_id)]
    return r
RMP.should_quarantine = _sqlog

tag = "_rs_%s_%s" % (DS, VARNAME); os.environ["TAMAIM_TAG"] = tag
from experiments.ablation_study import run_variant
t0 = time.time()
res = run_variant("A5", dataset="yelp", episodes=EPIS, attack_scenario="collusion")
qj = json.load(open("results/quar_A5_s%d%s.json" % (SEED, tag)))
import collections
print("pathways honest:", collections.Counter(v[1] for k, v in PATH.items() if k < NH), "sybil:", collections.Counter(v[1] for k, v in PATH.items() if k >= NH))
qe = {int(k): int(v) for k, v in qj["quar_ep"].items()}

# ---------- end-to-end ranking ----------
import utility_eval2 as V
cands = RD.ranking_setup(D, SEED)
targets = D["targets"]
n_items = D["n_items"]
hist = np.zeros((NH, n_items)); histC = np.zeros((NH, n_items))
for a in range(NH):
    for r in D["shards"][a]["hist"]:
        k = D["idx"][r[D["ik"]]]; hist[a, k] += float(r["stars"]); histC[a, k] += 1
cumS, cumC = {}, {}
def cum(a, t):
    if (a, t) not in cumS: cumS[(a, t)], cumC[(a, t)] = RD.agent_cum(D, a, t)
    return cumS[(a, t)], cumC[(a, t)]
def q_flags():
    f = {}
    for a in range(NH + NS):
        hist_q = []
        for t in range(1, EPIS + 1):
            if t >= 3 and len(hist_q) >= 3 and np.mean(hist_q[-5:]) < 0.5: f[a] = t; break
            hist_q.append(Q[a, t])
    return f
qf = q_flags()
sr = None
try:
    import utility_eval as U
    sr = U.sybilrank_flags(NH + NS, NH, NS, SEED)
except Exception as e:
    print("sybilrank skipped", e)
def pool(t, fl, whole=()):
    S = np.zeros(n_items); C = np.zeros(n_items); fab = 0
    for a in range(NH + NS):
        if a in whole: continue
        up = min(t, fl.get(a, t))
        if a < NH: S += hist[a]; C += histC[a]
        if up > 0:
            s_, c_ = cum(a, up); S += s_; C += c_
        if a >= NH: fab += sum(len(D["fab"][a - NH][u]) for u in range(1, up + 1))
    return S, C, fab
rows = []
CKPT = [t for t in (0, 10, 25, 50, 100, 200) if t <= EPIS]
for t in CKPT:
    meths = {"none": ({}, ()), "oracle": ({a: 0 for a in range(NH, NH + NS)}, ()), "TA-MAIM": (qe, ()), "Q<0.5": (qf, ())}
    if sr is not None: meths["SybilRank"] = ({}, tuple(sr))
    for m, (fl, whole) in meths.items():
        S, C, fab = pool(t, fl, whole) if t > 0 else (hist.sum(0), histC.sum(0), 0)
        nd, pk = V.rank_eval(V.score_from(S, C), cands, targets, SEED)
        flagged = set(whole) if whole else (set(fl) if m != "oracle" else set())
        rows.append(dict(t=t, method=m, ndcg=nd, poison10=pk, fab_in_pool=fab,
                         flagged_syb=sum(1 for a in flagged if a >= NH), flagged_hon=sum(1 for a in flagged if a < NH)))
# separation of benign vs malicious quality (AUC of -Q), early window and full
def auc(x_pos, x_neg):
    x = np.concatenate([x_pos, x_neg]); r = x.argsort().argsort() + 1.0
    return float((r[:len(x_pos)].sum() - len(x_pos) * (len(x_pos) + 1) / 2) / (len(x_pos) * len(x_neg)))
sep = {}
for lo, hi in ((1, 20), (1, EPIS)):
    ms = Q[NH:, lo:hi + 1].mean(1); mh = Q[:NH, lo:hi + 1].mean(1)
    sep["%d-%d" % (lo, hi)] = auc(-ms, -mh)
os.makedirs("results/rs", exist_ok=True)
json.dump(dict(dataset=DS, variant=VARNAME, signal=os.environ["RS_SIGNAL"], pathways={str(k): v for k, v in PATH.items()}, seed=SEED, episodes=EPIS, quar_ep={str(k): v for k, v in qe.items()}, qflags={str(k): v for k, v in qf.items()},
               sybranks=sorted(int(x) for x in sr) if sr is not None else None, rows=rows, sep_auc=sep,
               q_honest_mean=float(Q[:NH, 1:].mean()), q_sybil_mean=float(Q[NH:, 1:].mean()), cov_honest=float(cov[:NH, 1:].mean()),
               n_targets=int(len(targets)), res=res, secs=time.time() - t0),
          open("results/rs/%s_%s_s%d.json" % (DS, VARNAME, SEED), "w"))
print("rs done", DS, VAR, SEED, "sep", sep, "syb q %d hon q %d" % (len([a for a in qe if a >= NH]), len([a for a in qe if a < NH])))
