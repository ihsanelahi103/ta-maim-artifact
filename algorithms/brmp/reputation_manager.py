"""
RMP: Reputation Management Protocol — C3 contribution.

Formerly BRMP (Blockchain-based Reputation Management Protocol).
Now uses local tamper-evident ledger instead of blockchain.

Key functions unchanged:
- Trust-weighted Sybil flagging
- Multi-reporter consensus requirement (≥3 unique reporters)
- Reputation score management
- Quarantine enforcement (local persistent)

For WWW 2027 submission — blockchain dependency removed.
Accountability maintained via LocalLedger hash chain.
"""

import numpy as np
from config.config import AGENT, BRMP


class ReputationManager:
    """
    RMP — Reputation Management Protocol.

    Manages agent reputation scores and enforces
    Sybil quarantine based on trust-weighted flags
    from multiple independent reporters.

    Quarantine threshold:
    - ≥ BRMP['flag_threshold'] trust-weighted flags
    - Reputation < BRMP['quarantine_rep_threshold']
    - ≥ BRMP['min_unique_reporters'] unique reporters

    This ensures false positives are avoided — honest agents
    with high trust scores cannot be wrongly quarantined by
    a single malicious reporter.
    """

    def __init__(self, ledger=None):
        self.num_agents       = AGENT["num_agents"]
        self.num_honest       = AGENT["num_honest"]
        self.ledger           = ledger  # LocalLedger (optional)

        # Reputation scores — start at 1.0 for all
        self.reputation_scores    = np.ones(self.num_agents)
        # Trust-weighted flag accumulator per agent
        self.sybil_flag_counts    = np.zeros(self.num_agents)
        # Unique reporter tracking (prevents single-source attacks)
        self.unique_reporters     = {i: set()
                                     for i in range(self.num_agents)}
        # Quarantine set
        self.quarantined_agents   = set()

        # Config
        self.flag_threshold       = BRMP.get("flag_threshold", 8)
        self.rep_threshold        = BRMP.get("quarantine_rep_threshold", 0.40)
        self.min_unique           = BRMP.get("min_unique_reporters", 3)
        self.rep_decay            = BRMP.get("reputation_decay", 0.95)
        self.rep_boost            = BRMP.get("reputation_boost", 0.05)
        # Reputation decay/boost previously compounded on EVERY
        # environment step (update() is called from the per-step loop
        # in run.py/main.py, 50x/episode) using constants that read as
        # intended for per-episode application — e.g. 0.99 decay
        # compounds to 0.99^50=0.605 per episode. This caused
        # reputation to collapse to near-zero within ~5-6 episodes for
        # ANY agent never crossing the quality>0.55 boost threshold,
        # regardless of attack behaviour. Now throttled to the
        # existing (previously unread) "reputation_update_interval"
        # config key so decay/boost apply at the intended cadence.
        # Quarantine CHECKS still run every step so detection remains
        # prompt once thresholds are actually met.
        self.rep_update_interval  = BRMP.get("reputation_update_interval", 5)
        self._step_counter        = 0

        # Pathway 3 (Collusion-Aware Trust Signal) — off by default so
        # existing single-agent pathways are unaffected unless synced.
        # Ablation A7 ("w/o collusion-aware detection") sets this False.
        self.use_collusion_pathway = BRMP.get("use_collusion_pathway", True)
        self._collusion_suspicion  = None
        # Guard: require trust not clearly high before honouring
        # Pathway 3, to reduce false-positive risk on densely-connected
        # honest sub-communities (e.g. a tight-knit city cluster) that
        # may also show elevated structural suspicion.
        self.collusion_trust_guard = BRMP.get("collusion_trust_guard", 0.85)
        self.collusion_rep_guard   = BRMP.get("collusion_rep_guard", 0.90)
        # Structural/correlation floors replace the trust/rep guards
        # above for Pathway 3's actual gating condition (see
        # should_quarantine): colluding Sybils are specifically
        # designed to keep individual trust/reputation high, so
        # gating on those blocks genuine detections -- confirmed
        # empirically (trust=0.99, rep=1.00 throughout, while CATS
        # correctly flagged is_suspicious=True). Structural/
        # correlation independently crossing their own floors is the
        # correct signal for "genuinely coordinated" vs. "honest
        # dense community."
        self.collusion_structural_floor   = BRMP.get("collusion_structural_floor", 0.5)
        self.collusion_correlation_floor  = BRMP.get("collusion_correlation_floor", 0.3)

    def update(self, agents, step_info=None):
        """Update reputation scores based on agent behaviour.

        Score decay/boost is throttled to rep_update_interval (see
        __init__) since this method is called once per environment
        step. Quarantine conditions are still checked every call so
        detection remains prompt once thresholds are actually met.
        """
        self._step_counter += 1
        do_score_update = (self._step_counter % self.rep_update_interval == 0)

        mined = step_info.get("mined_knowledge", {}) if step_info else {}

        # Get trust scores if synced (from TPP)
        trust = getattr(self, '_trust_scores', None)

        if do_score_update:
            for i in range(self.num_agents):
                if i in self.quarantined_agents:
                    continue

                if i in mined:
                    # Use actual mining quality signal
                    quality = float(mined[i].get("quality", 0.5))
                elif trust is not None:
                    # Fallback: use trust score as proxy quality signal
                    quality = float(trust[i])
                else:
                    quality = 0.5  # Unknown — no update

                # RMP observes quality signal only — not ground truth label
                # Camouflage effective: high quality fake = good reputation
                if quality > 0.55:
                    # Good quality → boost reputation
                    self.reputation_scores[i] = min(
                        1.0,
                        self.reputation_scores[i] + self.rep_boost * quality
                    )
                elif quality < 0.25:
                    # Clearly bad quality → decay reputation
                    self.reputation_scores[i] = max(
                        0.0,
                        self.reputation_scores[i] * self.rep_decay
                    )
                # Ambiguous (0.25-0.55): slow decay to reflect uncertainty
                else:
                    self.reputation_scores[i] = max(
                        0.0,
                        self.reputation_scores[i] * 0.99
                    )

        # Check quarantine conditions and notify agent objects
        # Note: quarantine is ONLY committed if agent object exists
        # This prevents premature quarantine during empty-agent calls
        for i in range(self.num_agents):
            if i not in self.quarantined_agents:
                if self.should_quarantine(i):
                    # Find the agent object
                    agent_obj = None
                    for agent in agents:
                        if agent.agent_id == i:
                            agent_obj = agent
                            break
                    # Only quarantine if we have the agent object
                    # (prevents quarantine during test calls with agents=[])
                    if agent_obj is not None:
                        self.quarantined_agents.add(i)
                        if self.ledger:
                            self.ledger.quarantine_agent(i, "rmp_threshold")
                        agent_obj.is_quarantined = True

    def record_sybil_flag(self, reporter_id, target_id,
                          reporter_trust, evidence=None):
        """
        Record a trust-weighted Sybil flag.

        Multiple unique reporters required to quarantine —
        prevents false positive attacks by single malicious reporter.
        """
        if target_id in self.quarantined_agents:
            return

        # Trust-weighted flag increment
        self.sybil_flag_counts[target_id] += reporter_trust
        self.unique_reporters[target_id].add(reporter_id)

        # Record in ledger
        if self.ledger:
            self.ledger.record_flag(
                reporter_id, target_id,
                reporter_trust, evidence or {}
            )

    def should_quarantine(self, agent_id):
        """
        Check quarantine conditions — two pathways:

        Pathway 1 (flag-based): Standard multi-reporter consensus
          - Enough trust-weighted flags from multiple reporters
          - Reputation below threshold

        Pathway 2 (trust-based): Direct trust collapse detection
          - Trust score collapses to near-zero (strong signal)
          - Used when agent camouflages then attacks
        """
        if agent_id in self.quarantined_agents:
            return False

        flags  = self.sybil_flag_counts[agent_id]
        rep    = self.reputation_scores[agent_id]
        unique = len(self.unique_reporters[agent_id])

        # Pathway 1: Flag-based quarantine (standard)
        if (flags >= self.flag_threshold and
                rep < self.rep_threshold and
                unique >= self.min_unique):
            return True

        # Pathway 2: Trust collapse quarantine
        # When trust drops to near-zero AND reputation decays significantly
        # This catches camouflaged Sybil that built reputation then attacked
        if hasattr(self, '_trust_scores'):
            trust = self._trust_scores[agent_id]
            if trust < 0.08 and rep < 0.75:
                return True

        # Pathway 3: Collusion-Aware Trust Signal (C5)
        # Catches coordinated Sybil clusters that individual/pairwise
        # trust (Pathways 1 & 2) cannot see — high structural
        # suspicion (dense mutual clique in the interaction graph)
        # combined with correlated behavioral history among direct
        # graph neighbours. Guarded by trust/reputation ceilings so a
        # densely-connected honest community isn't quarantined on
        # structural similarity alone.
        if (self.use_collusion_pathway and
                self._collusion_suspicion is not None):
            # Gate on structural density and behavioral correlation
            # independently crossing their own floors -- NOT on
            # individual trust/reputation, which collusion is
            # specifically designed to keep high (confirmed
            # empirically: trust~0.99, rep~1.00 throughout a real
            # colluding agent's episodes, while structural/
            # correlation both climbed as expected). This is what
            # actually distinguishes a genuinely coordinated cluster
            # from a structurally dense but behaviorally independent
            # honest community (e.g. a tight-knit city cluster): the
            # latter shows high structural score but near-zero
            # correlation, since honest neighbours mining unrelated
            # data shouldn't move in lockstep.
            structural_ok = (
                self._collusion_suspicion.structural_score[agent_id]
                > self.collusion_structural_floor
            )
            correlation_ok = (
                self._collusion_suspicion.correlation_score[agent_id]
                > self.collusion_correlation_floor
            )
            if (structural_ok and correlation_ok and
                    self._collusion_suspicion.is_suspicious(agent_id)):
                return True

        return False

    def sync_collusion_suspicion(self, collusion_detector):
        """Sync external CollusionAwareTrustSignal instance for
        Pathway 3 detection. Called once per step/episode alongside
        sync_trust_scores()."""
        self._collusion_suspicion = collusion_detector

    def sync_trust_scores(self, trust_scores):
        """Sync external trust scores for pathway 2 detection."""
        self._trust_scores = trust_scores

    def get_reputation(self, agent_id):
        return float(self.reputation_scores[agent_id])

    def get_stats(self):
        quarantined = len(self.quarantined_agents)
        sybil_q     = sum(
            1 for i in self.quarantined_agents
            if i >= self.num_honest
        )
        honest_q    = quarantined - sybil_q
        return {
            "total_quarantined": quarantined,
            "sybil_quarantined": sybil_q,
            "honest_quarantined":honest_q,
            "avg_reputation":    float(np.mean(self.reputation_scores)),
        }


# Backward compatibility alias
BlockchainReputationManager = ReputationManager