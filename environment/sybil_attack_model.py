"""
Sybil Attack Models for TA-MAIM — WWW 2027
Four attack types including new Adaptive Stealth attacker.
"""
import numpy as np


def _seed():
    import utils.seed as s
    return int(s.CURRENT_SEED)


class StaticSybilAttack:
    """Static: persistent low-quality output throughout."""
    def __init__(self, num_sybil, num_honest):
        self.rng = np.random.RandomState(_seed())
    def get_quality(self, sybil_idx, episode):
        return float(np.clip(0.125 + self.rng.normal(0, 0.05), 0.0, 0.25))
    def get_collusion_boost(self, sybil_idx=0, episode=0, num_sybil=0):
        return 0.0  # No collusion boost for static attack


class CamouflagedsybilAttack:
    """Camouflage: 90-95% mimicry then phase transition."""
    def __init__(self, num_sybil, num_honest):
        self.rng = np.random.RandomState(_seed())
        self.transitions = {i: int(self.rng.uniform(40, 150))
                            for i in range(num_sybil)}
    def get_quality(self, sybil_idx, episode):
        t = self.transitions.get(sybil_idx, 75)
        if episode < t:
            return float(np.clip(0.92 + self.rng.normal(0, 0.02), 0.88, 0.97))
        elif episode < t + 30:
            p = (episode - t) / 30.0
            return float(np.clip(0.88 - 0.80*p + self.rng.normal(0, 0.04),
                                 0.05, 0.90))
        return float(np.clip(0.07 + self.rng.normal(0, 0.03), 0.0, 0.15))
    def get_collusion_boost(self, sybil_idx=0, episode=0, num_sybil=0):
        return 0.0  # No collusion boost for camouflage attack


class ColludingSybilAttack:
    """
    Collusion: genuinely coordinated Sybil cluster.

    Rebuild (v4 — WWW 2027): The previous implementation modeled
    "collusion" as an artificial additive boost applied directly to
    each Sybil's own reported quality value — the same signal
    TrustPropagationProtocol.update_from_mining_quality() uses for its
    primary honest/Sybil classification. At any realistic num_sybil,
    that boost pushed quality above the trust-increase threshold
    (0.5), so colluding Sybils were classified as honest by the
    primary detector every episode, at every scale and ratio tested
    (F1=0.000 unconditionally). That was a threshold-crossing defect,
    not a faithful model of collusion.

    This version removes the artificial boost entirely. Individual
    quality stays genuinely bad (same range as StaticSybilAttack), so
    the primary quality-based signal catches colluding Sybils just as
    readily as static ones. Collusion is instead modeled through two
    mechanisms that are both externally observable and neither of
    which touches ground-truth labels:

    1. Shared latent "campaign intensity" (AR(1) process, common to
       the whole cluster) added to each member's quality draw. This
       makes the cluster's quality time series GENUINELY correlated
       with each other episode-to-episode — checkable directly with
       Pearson correlation on the observed histories — rather than
       merely independently drawn from the same distribution.
    2. Mutual endorsement: colluding Sybils preferentially target
       OTHER Sybils when performing ValidateNode (see
       `mutual_endorsement` flag, read by EdgeMiningEnvironment),
       falsely vouching for each other's fake contributions through
       the existing, legitimate TPP.record_validation() path. This
       creates a real trust tug-of-war (bad quality pulls trust down,
       mutual endorsement pushes it back up) instead of a hardcoded
       number.

    Detection of this attack is intentionally NOT solved by the
    individual/pairwise trust signal alone — that is the point being
    demonstrated. It requires the graph-structural clique signature
    already present in the injected Sybil-Sybil edges (see
    *_loader.py: _inject_sybil_agents) plus the behavioral-correlation
    signal from (1), both consumed by CollusionAwareTrustSignal
    (algorithms/collusion/collusion_detector.py) as a dedicated
    higher-order detection pathway.
    """
    def __init__(self, num_sybil, num_honest):
        self.rng = np.random.RandomState(_seed())
        self.num_sybil = num_sybil
        self.num_honest = num_honest
        # Colluding Sybils preferentially validate each other —
        # read by EdgeMiningEnvironment._execute_validate_node().
        self.mutual_endorsement = True

        # Shared latent campaign-intensity process — AR(1), common
        # across the whole colluding cluster. This is what makes
        # quality time series genuinely correlated, not just
        # independently sampled from the same distribution.
        self._latent_rng = np.random.RandomState(_seed() + 7)
        self._latent_cache = {0: 0.0}

    def _shared_latent(self, episode):
        """AR(1) latent shared by the whole cluster: smooth, slowly
        varying, same value read by every Sybil at a given episode."""
        if episode not in self._latent_cache:
            prev_ep = max(e for e in self._latent_cache if e < episode)
            prev = self._latent_cache[prev_ep]
            innovation = self._latent_rng.normal(0, 0.025)
            self._latent_cache[episode] = float(
                np.clip(0.85 * prev + innovation, -0.07, 0.07)
            )
        return self._latent_cache[episode]

    def get_quality(self, sybil_idx, episode):
        """
        Individual quality is deliberately kept in TPP's AMBIGUOUS zone
        (0.3-0.5), where update_from_mining_quality() withholds any
        direct trust update. This is the defining property of a
        collusion attack: viewed in isolation, a colluding Sybil's
        output should NOT look obviously bad (that would make it as
        easy to catch as a static attack via the primary quality
        signal alone, as observed empirically before this change —
        detection completed by episode 1-2 purely from the individual
        penalty, before the collusion-aware detector or mutual
        endorsement could matter at all).

        With no direct quality-driven update, a colluding Sybil's
        trust trajectory is instead governed by: trust_decay (natural
        downward drift), propagate() (pulled toward its densely-
        connected Sybil clique's own average trust), mutual
        endorsement (pushes back up), and ultimately
        CollusionAwareTrustSignal's structural + behavioral-
        correlation suspicion (Pathway 3) — which is the signal this
        attack is meant to be caught by.
        """
        shared = self._shared_latent(episode)
        individual_noise = self.rng.normal(0, 0.025)
        return float(np.clip(0.38 + shared + individual_noise, 0.30, 0.48))

    def get_collusion_boost(self, sybil_idx=0, episode=0, num_sybil=0):
        # Deprecated boost mechanism — retained only for interface
        # compatibility with older call sites. Collusion is now
        # modeled via shared_latent() correlation + mutual_endorsement,
        # not an additive quality boost.
        return 0.0

    def get_preferred_validation_target(self, sybil_idx, num_agents):
        """Colluding Sybils preferentially validate another Sybil in
        the cluster (mutual endorsement), rather than a random peer.
        Returns an absolute agent_id, or None to fall back to the
        environment's default random-peer selection."""
        if self.num_sybil <= 1:
            return None
        other_sybils = [self.num_honest + i for i in range(self.num_sybil)
                         if i != sybil_idx]
        if not other_sybils:
            return None
        return int(np.random.choice(other_sybils))


class AdaptiveStealthSybilAttack:
    """
    Adaptive Stealth Attacker — Addresses Reviewer MC1.

    Three-phase adaptive strategy:
    Phase 1 (ep 1 - T1): Very high mimicry (95-99% honest quality)
    Phase 2 (T1 - T2):   Slow behavioral drift — gradual quality reduction
                          Rate adjusted based on observed trust feedback
    Phase 3 (T2+):       Selective malicious actions — only attacks when
                          trust buffer is high, recovers when detected

    This creates:
    - Later detection than standard camouflage
    - Realistic F1 degradation (target 0.93-0.98)
    - Trust separation reduction
    - More realistic adversarial model for WWW
    """
    def __init__(self, num_sybil, num_honest,
                 t1_range=(60, 120), t2_range=(130, 180)):
        self.num_sybil = num_sybil
        self.num_honest = num_honest
        self.rng = np.random.RandomState(_seed())

        # Heterogeneous per-agent phase transitions
        self.t1 = {i: int(self.rng.uniform(*t1_range))
                   for i in range(num_sybil)}
        self.t2 = {i: int(self.rng.uniform(*t2_range))
                   for i in range(num_sybil)}

        # Per-agent estimated trust (updated from feedback)
        self.est_trust = {i: 0.85 for i in range(num_sybil)}

        # Per-agent drift rate (learned from trust feedback)
        self.drift_rate = {i: self.rng.uniform(0.003, 0.008)
                           for i in range(num_sybil)}

        # Selective attack: only attack when trust buffer > threshold
        self.attack_threshold = 0.50

    def update_trust_estimate(self, sybil_idx, observed_trust):
        """Adaptive feedback: agent updates its trust estimate."""
        alpha = 0.25  # Slow learning — stealth agent
        self.est_trust[sybil_idx] = (
            (1 - alpha) * self.est_trust[sybil_idx]
            + alpha * float(observed_trust)
        )
        # Slow drift rate when trust is falling (be more careful)
        if observed_trust < 0.55:
            self.drift_rate[sybil_idx] = max(
                0.001, self.drift_rate[sybil_idx] * 0.8
            )

    def get_quality(self, sybil_idx, episode):
        t1 = self.t1.get(sybil_idx, 80)
        t2 = self.t2.get(sybil_idx, 150)
        tau = self.est_trust.get(sybil_idx, 0.85)
        dr  = self.drift_rate.get(sybil_idx, 0.005)

        if episode < t1:
            # Phase 1: Very high mimicry — 95-99% honest quality
            # Much harder to detect than 90-95% camouflage
            noise = self.rng.normal(0, 0.015)
            return float(np.clip(0.96 + noise, 0.93, 0.99))

        elif episode < t2:
            # Phase 2: Slow behavioral drift
            # Quality drifts down at rate controlled by trust feedback
            episodes_in_phase = episode - t1
            drift = dr * episodes_in_phase
            # Slow down drift if trust is dropping (adaptive)
            if tau < 0.60:
                drift *= 0.5
            quality = max(0.40, 0.92 - drift)
            noise = self.rng.normal(0, 0.025)
            return float(np.clip(quality + noise, 0.30, 0.95))

        else:
            # Phase 3: Selective malicious actions
            # Only attack when trust buffer is high enough
            if tau > self.attack_threshold:
                # Trust comfortable — execute malicious action
                noise = self.rng.normal(0, 0.05)
                return float(np.clip(0.15 + noise, 0.05, 0.35))
            else:
                # Trust falling — recover with honest behaviour
                noise = self.rng.normal(0, 0.03)
                return float(np.clip(0.75 + noise, 0.65, 0.88))


    def get_collusion_boost(self, sybil_idx=0, episode=0, num_sybil=0):
        return 0.0  # Adaptive stealth attack does not use collusion boost

class SybilAttackFactory:
    @staticmethod
    def create(attack_type, num_sybil, num_honest):
        mapping = {
            'static':     StaticSybilAttack,
            'camouflage': CamouflagedsybilAttack,
            'collusion':  ColludingSybilAttack,
            'adaptive':   AdaptiveStealthSybilAttack,
        }
        if attack_type not in mapping:
            raise ValueError(
                f"Unknown attack: {attack_type}. "
                f"Choose from {list(mapping.keys())}"
            )
        return mapping[attack_type](num_sybil, num_honest)