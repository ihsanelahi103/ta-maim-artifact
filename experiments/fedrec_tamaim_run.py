"""TA-MAIM in the federated-recommendation experiment: runs experiments/ablation_study.py (variant A5,
unchanged TA-MAIM: TPP, RMP, CATS, DAPP) while the QUALITY SIGNAL of every MineData call is replaced by the
peer-validation quality of the federated recommender (pre-registered). One FL round per episode.
Quarantined agents are dropped from aggregation from the next round.
env: FEDREC_ETA (required, step size from fedrec_baselines.select_eta), FEDREC_AGG=mean|normclip,
     TAMAIM_SEED, TAMAIM_TAG, TAMAIM_DATASET (yelp|amazon)
Writes results/fedrec_tamaim_<ds>_s<seed><TAG>.json and the usual results/quar_A5_s<seed><TAG>.json."""
import os, sys, json, runpy
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
import numpy as np
import fedrec_core as F
import fedrec_baselines as B
from environment.edge_mining_env import EdgeMiningEnvironment as Env

SEED = int(os.environ.get("TAMAIM_SEED", "42")); TAG = os.environ.get("TAMAIM_TAG", "")
DS = os.environ.get("TAMAIM_DATASET", "yelp"); ETA = float(os.environ["FEDREC_ETA"])
AGG = os.environ.get("FEDREC_AGG", "mean")
os.environ["TAMAIM_VARIANTS"] = "A5"; os.environ.setdefault("TAMAIM_ATTACK", "static")

S, C, cands = B.load(DS, SEED)
SIM = F.FedSim(S, C, cands, SEED, attack=True)
LOG = {"auc": [], "q_honest_mean": [], "q_sybil_mean": [], "active_sybil": [], "active_honest": []}
_orig_reset = Env.reset


def auc(pos, neg):
    s = sum((neg < p).sum() + 0.5 * (neg == p).sum() for p in pos)
    return float(s / (len(pos) * len(neg)))


def fed_round(env):
    n, n_h = SIM.n, SIM.n_h
    G, sy, _ = SIM.collect()
    quar = np.array([bool(env.agents_ref[i].is_quarantined) for i in range(n)])
    idx = np.nonzero(~quar)[0]
    valid = [i for i in range(n_h) if not quar[i]]
    r = np.zeros(n)
    if len(valid) >= 4:
        r[idx] = SIM.peer_scores(G, valid, ETA, ids=idx)[idx]
    med = np.median(r[idx]); mad = 1.4826 * np.median(np.abs(r[idx] - med)) + 1e-9
    q = np.clip(0.65 + 0.12 * (r - med) / mad, 0.0, 1.0)
    env._fq = q
    hs, ss = idx[idx < n_h], idx[idx >= n_h]
    LOG["q_honest_mean"].append(float(q[hs].mean()) if len(hs) else None)
    LOG["q_sybil_mean"].append(float(q[ss].mean()) if len(ss) else None)
    LOG["active_sybil"].append(int(len(ss))); LOG["active_honest"].append(int(len(hs)))
    if sy.any() and len(ss) and len(hs):
        LOG["auc"].append(auc(-r[ss], -r[hs]))
    agg = F.agg_normclip(G, idx) if AGG == "normclip" else F.agg_mean(G, idx)
    SIM.apply(agg, ETA)


def reset(self, *a, **k):
    fed_round(self)
    return _orig_reset(self, *a, **k)


def mine(self, agent_id, is_sybil):
    """Same as EdgeMiningEnvironment._execute_mine_data, but quality = federated peer-validation quality."""
    quality = float(self._fq[agent_id]); data_authentic = not is_sybil
    dp_noise = np.random.laplace(0, 0.1)
    noised_quality = np.clip(quality + dp_noise, 0.0, 1.0)
    tx = {"tx_id": self.blockchain.record_mining_tx(agent_id, 0.5, "action")}
    self.mined_knowledge[agent_id] = {"quality": noised_quality, "authentic": data_authentic, "step": self.current_step}
    self.collusion_detector.record_quality(agent_id, quality)
    self.tpp.update_from_mining_quality(agent_id=agent_id, quality=noised_quality)
    base_reward = noised_quality if not is_sybil else -noised_quality
    if not is_sybil:
        current_eps = float(self.dapp.epsilon_values[agent_id])
        reward = base_reward - (1.0 - 1.0 / (1.0 + current_eps)) * 0.3
    else:
        reward = base_reward
    return reward, tx


Env.reset, Env._execute_mine_data = reset, mine
out = "results/fedrec_tamaim_%s_s%d%s.json" % (DS, SEED, TAG)
sys.argv = ["ablation_study.py", "--dataset", DS, "--episodes", os.environ.get("FEDREC_EPISODES", "200")]
try:
    runpy.run_path("experiments/ablation_study.py", run_name="__main__")
finally:
    os.makedirs("results", exist_ok=True)
    res = SIM.evaluate(); res.update(agg=AGG, eta=ETA, seed=SEED, dataset=DS, rounds=SIM.round, log=LOG)
    json.dump(res, open(out, "w"))
    print("fedrec result", {k: v for k, v in res.items() if k != "log"})
