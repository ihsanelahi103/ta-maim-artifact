"""Common-stream / shadow-mode runner (review-2 experiments E1, E2).
Runs experiments/ablation_study.py::run_variant('A5') UNCHANGED except for these hooks:
  * SHADOW: RMP quarantine decisions are logged (first episode + pathway) but never applied, so every agent
    produces a full, uncensored 200-episode stream that all detectors can be replayed on.
  * every MineData call is logged (episode, step, agent, raw q, privatized q); per-episode graphs and the online
    CATS scores (structural, correlation, suspicion) are logged.
  * regime hooks (REGIME env): quality-matched / decorrelated / split-latent colluders, benign correlated communities.
Default is the ground-truth-free initial trust (TAMAIM_NO_SYBIL_PRIOR=1) unless TAMAIM_PRIOR=1.
usage: TAMAIM_SEED=7 REGIME=qmatch python experiments/fs_run.py [--episodes 200] [--shadow 1]
"""
import os, sys, json, time, argparse
if os.environ.get("TAMAIM_PRIOR") == "1":
    os.environ.pop("TAMAIM_NO_SYBIL_PRIOR", None)
else:
    os.environ["TAMAIM_NO_SYBIL_PRIOR"] = "1"
sys.path.insert(0, os.getcwd())
import numpy as np

REGIME = os.environ.get("REGIME", "qmatch")
SEED = int(os.environ.get("TAMAIM_SEED", "42"))
SHADOW = os.environ.get("SHADOW", "1") == "1"

# ---- regime table -> (attack, evade band, lambda, groups, comm_k, comm_amp, topology env)
R = {
 "static":   dict(attack="static"),
 "collude":  dict(attack="collusion"),
 "qmatch":   dict(attack="collusion", qm=True),
 "band1":    dict(attack="static", evade="0.30,0.50"),
 "band2":    dict(attack="static", evade="0.40,0.60"),
 "band3":    dict(attack="static", evade="0.50,0.70"),
 "band4":    dict(attack="static", evade="0.60,0.80"),
 "camou":    dict(attack="camouflage"),
 # adaptive attackers (quality-matched collusion with weakened coordination)
 "qm_l050":  dict(attack="collusion", qm=True, lam=0.5),
 "qm_l025":  dict(attack="collusion", qm=True, lam=0.25),
 "qm_l000":  dict(attack="collusion", qm=True, lam=0.0),
 "qm_split5": dict(attack="collusion", qm=True, groups=5),
 "qm_d070":  dict(attack="collusion", qm=True, topo={"TAMAIM_SYBIL_DENSITY": "0.7"}),
 "qm_d050":  dict(attack="collusion", qm=True, topo={"TAMAIM_SYBIL_DENSITY": "0.5"}),
 "qm_ae10":  dict(attack="collusion", qm=True, topo={"TAMAIM_ATTACK_EDGES": "10"}),
 "qm_hd16":  dict(attack="collusion", qm=True, topo={"TAMAIM_HONEST_DEG": "16"}),
 # benign correlated communities (honest cliques with shared latent); with quality-matched Sybils present
 "bc10":     dict(attack="collusion", qm=True, comm_k=10, comm_amp=1.0),
 "bc20":     dict(attack="collusion", qm=True, comm_k=20, comm_amp=1.0),
 "bc10h":    dict(attack="collusion", qm=True, comm_k=10, comm_amp=0.5),
 "bc20h":    dict(attack="collusion", qm=True, comm_k=20, comm_amp=0.5),
 "bc10only": dict(attack="static", evade="0.65,1.00", comm_k=10, comm_amp=1.0),   # Sybils honest-range & uncorrelated: community FP only
}[REGIME]
os.environ["TAMAIM_ATTACK"] = R["attack"]
if R.get("evade"): os.environ["TAMAIM_EVADE"] = R["evade"]
for k, v in R.get("topo", {}).items(): os.environ[k] = v
LAM = R.get("lam", 1.0); GROUPS = R.get("groups", 1)
COMM_K = R.get("comm_k", 0); COMM_AMP = R.get("comm_amp", 1.0)

from environment.edge_mining_env import EdgeMiningEnvironment as Env
from algorithms.brmp.reputation_manager import ReputationManager as RMP
from environment import sybil_attack_model as sam
import utils.seed as useed

REC, CTR = [], [0]
NEIGH, GRAPH1, GDIFF = {}, [None], {}
TRAJ = {"S": [], "C": [], "sus": []}
SH = {}
COMM = set()
CUR = [None]
COMM_LAT = {"c": {0: 0.0}, "rng": None}

_orig_reset, _orig_mine, _orig_topo = Env.reset, Env._execute_mine_data, Env._apply_topology
_orig_sq = RMP.should_quarantine
_orig_uniform = np.random.uniform


def _adj(g): return {int(i): sorted(int(j) for j in nb) for i, nb in g.items()}


def reset(self, *a, **k):
    CTR[0] += 1
    out = _orig_reset(self, *a, **k)
    cd = self.collusion_detector
    TRAJ["S"].append([round(float(x), 4) for x in cd.structural_score])
    TRAJ["C"].append([round(float(x), 4) for x in cd.correlation_score])
    TRAJ["sus"].append([round(float(x), 4) for x in cd.suspicion])
    g = _adj(self.graph)
    if GRAPH1[0] is None: GRAPH1[0] = g
    else:
        d = [(i, j) for i, nb in g.items() for j in nb if j not in GRAPH1[0].get(i, [])]
        d += [(i, j) for i, nb in GRAPH1[0].items() for j in nb if j not in g.get(i, [])]
        if d: GDIFF[CTR[0]] = d
    return out


def mine(self, agent_id, is_sybil):
    CUR[0] = agent_id
    r = _orig_mine(self, agent_id, is_sybil)
    CUR[0] = None
    if not NEIGH:
        for i in range(self.num_agents): NEIGH[i] = [int(x) for x in self._get_neighbours(i)]
    REC.append([CTR[0], int(self.current_step), int(agent_id),
                round(float(self._quality_buffer[agent_id][-1]), 4),
                round(float(self.mined_knowledge[agent_id]["quality"]), 4)])
    return r


def comm_latent(ep):
    c = COMM_LAT["c"]
    if ep not in c:
        prev = c[max(e for e in c if e < ep)]
        c[ep] = float(np.clip(0.85 * prev + COMM_LAT["rng"].normal(0, 0.025), -0.07, 0.07))
    return c[ep]


def uniform(lo=0.0, hi=1.0, size=None):
    x = _orig_uniform(lo, hi, size)
    if size is None and lo == 0.65 and hi == 1.0 and CUR[0] in COMM:
        return float(np.clip(x + COMM_AMP * comm_latent(CTR[0]), 0.0, 1.0))
    return x


def topo(self):
    _orig_topo(self)
    if COMM_K:
        rr = np.random.RandomState(SEED + 4242)
        COMM.update(int(x) for x in rr.choice(self.num_honest, COMM_K, replace=False))
        COMM_LAT["rng"] = np.random.RandomState(SEED + 31)
        dl = self.data_loader
        for gr in [dl.graph] + list(dl.episode_graphs):
            for i in COMM:
                for j in COMM:
                    if i != j and j not in gr[i]:
                        gr[i][j] = 0.25
                        gr[j][i] = 0.25


def pathway(self, a):
    f, rep, u = self.sybil_flag_counts[a], self.reputation_scores[a], len(self.unique_reporters[a])
    if f >= self.flag_threshold and rep < self.rep_threshold and u >= self.min_unique: return "P1"
    if hasattr(self, "_trust_scores") and self._trust_scores[a] < 0.08 and rep < 0.75: return "P2"
    return "P3"


def sq(self, agent_id):
    r = _orig_sq(self, agent_id)
    if r and agent_id not in SH:
        SH[agent_id] = [CTR[0], pathway(self, agent_id)]
    return False if SHADOW else r


Env.reset, Env._execute_mine_data, Env._apply_topology = reset, mine, topo
RMP.should_quarantine = sq
np.random.uniform = uniform

if R.get("qm"):
    def qm(self, sybil_idx, episode):
        g = sybil_idx // max(1, int(round(self.num_sybil / GROUPS)))
        if g == 0:
            lat = self._shared_latent(episode)
        else:
            if not hasattr(self, "_glat"): self._glat = {}
            d = self._glat.setdefault(g, ({0: 0.0}, np.random.RandomState(useed.CURRENT_SEED + 7 + 100 * g)))
            c, rng = d
            if episode not in c:
                prev = c[max(e for e in c if e < episode)]
                c[episode] = float(np.clip(0.85 * prev + rng.normal(0, 0.025), -0.07, 0.07))
            lat = c[episode]
        return float(np.clip(_orig_uniform(0.65, 1.0) + LAM * lat, 0, 1))
    sam.ColludingSybilAttack.get_quality = qm

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=200)
ap.add_argument("--dataset", default="yelp")
a = ap.parse_args()

NAME = REGIME + ("" if SHADOW else "-inloop")
tag = "_fs_%s" % NAME
os.environ["TAMAIM_TAG"] = tag
from experiments.ablation_study import run_variant
t0 = time.time()
res = run_variant("A5", dataset=a.dataset, episodes=a.episodes, attack_scenario=R["attack"])
os.makedirs("results/fs", exist_ok=True)
qpath = "results/quar_A5_s%d%s.json" % (SEED, tag)
qj = json.load(open(qpath)) if os.path.exists(qpath) else {}
json.dump({"regime": NAME, "seed": SEED, "shadow": SHADOW, "prior": os.environ.get("TAMAIM_PRIOR") == "1",
           "episodes": a.episodes, "records": REC, "neigh": NEIGH, "graph1": GRAPH1[0], "gdiff": {str(k): v for k, v in GDIFF.items()},
           "traj": TRAJ, "shadow_q": {str(k): v for k, v in SH.items()}, "comm": sorted(COMM),
           "sybil_ids": qj.get("sybil_ids"), "sybil_T": qj.get("sybil_T"), "quar_ep": qj.get("quar_ep"),
           "res": res, "secs": time.time() - t0},
          open("results/fs/%s_s%d.json" % (NAME, SEED), "w"))
print("fs done", REGIME, SEED, "shadow" if SHADOW else "inloop", "%.0fs" % (time.time() - t0), "flagged", len(SH), res.get("sybil_f1"))
