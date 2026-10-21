"""Run experiments/ablation_study.py unchanged, but record (episode, agent, raw q, privatized q)
for every MineData call and the initial neighbour lists. Optional TAMAIM_QMATCH=1 makes
ColludingSybilAttack draw honest-range quality (U(0.65,1.0)) + the same shared AR(1) latent.
Nothing else in the simulator is changed."""
import os, sys, json, runpy
import numpy as np
sys.path.insert(0, os.getcwd())
from environment.edge_mining_env import EdgeMiningEnvironment as Env

REC, NEIGH, CTR = [], {}, [0]
_orig_mine, _orig_reset = Env._execute_mine_data, Env.reset

def reset(self, *a, **k):
    CTR[0] += 1
    return _orig_reset(self, *a, **k)

def mine(self, agent_id, is_sybil):
    r = _orig_mine(self, agent_id, is_sybil)
    if not NEIGH:
        for i in range(self.num_agents):
            NEIGH[i] = [int(x) for x in self._get_neighbours(i)]
    REC.append([CTR[0], int(getattr(self, "current_episode", -1)), int(agent_id),
                float(self._quality_buffer[agent_id][-1]),
                float(self.mined_knowledge[agent_id]["quality"])])
    return r

Env.reset, Env._execute_mine_data = reset, mine

if os.environ.get("TAMAIM_QMATCH") == "1":
    from environment import sybil_attack_model as sam
    def qm(self, sybil_idx, episode):
        return float(np.clip(np.random.uniform(0.65, 1.0) + self._shared_latent(episode), 0, 1))
    sam.ColludingSybilAttack.get_quality = qm

out = "results/qdump_s%s%s.json" % (os.environ.get("TAMAIM_SEED", "42"), os.environ.get("TAMAIM_TAG", ""))
sys.argv = ["ablation_study.py"]
try:
    runpy.run_path("experiments/ablation_study.py", run_name="__main__")
finally:
    json.dump({"records": REC, "neigh": NEIGH}, open(out, "w"))
    print("qdump written", out, len(REC), "records")
