"""
LocalLedger -- tamper-evident local audit trail for TA-MAIM.

Replaces an earlier blockchain-based accountability layer (removed
prior to submission; RMP quarantine decisions and Sybil flags are
now recorded in a local, single-process, hash-chained append-only
log instead of a distributed ledger). This is bookkeeping/audit
infrastructure only -- it does not influence any trust, reputation,
or quarantine DECISION (those are made entirely by TPP/RMP/CATS); it
only records what was decided, with tamper-evidence via hash
chaining, so no experimental result depends on its internal
implementation.

Not described in the paper (main.tex/supplementary.tex do not
mention "blockchain" or "ledger" -- this is intentional internal
plumbing, not a claimed contribution).
"""

import hashlib
import time


class LocalLedger:
    """
    Append-only, hash-chained local log of agent registration,
    quarantine actions, and Sybil flags.

    Each entry's hash depends on its own content AND the previous
    entry's hash, so any retroactive tampering with an earlier entry
    is detectable via verify_chain() -- the "tamper-evident" property
    without requiring a distributed/consensus mechanism, since this
    runs single-process within one experiment.
    """

    def __init__(self):
        self.chain = []          # list of entry dicts, append-only
        self.registered_agents = set()
        self.quarantined_agents = set()
        self._mining_tx_count = 0  # running counter, avoids O(n) chain
                                    # scan in get_stats() (see note there)
        self._genesis()

    def _genesis(self):
        genesis_entry = {
            'index': 0,
            'timestamp': time.time(),
            'event_type': 'genesis',
            'data': {},
            'prev_hash': '0' * 64,
        }
        genesis_entry['hash'] = self._compute_hash(genesis_entry)
        self.chain.append(genesis_entry)

    def _compute_hash(self, entry):
        payload = (
            f"{entry['index']}{entry['timestamp']}"
            f"{entry['event_type']}{entry['data']}{entry['prev_hash']}"
        )
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def _append(self, event_type, data):
        prev_hash = self.chain[-1]['hash']
        entry = {
            'index': len(self.chain),
            'timestamp': time.time(),
            'event_type': event_type,
            'data': data,
            'prev_hash': prev_hash,
        }
        entry['hash'] = self._compute_hash(entry)
        self.chain.append(entry)
        if event_type == 'mining_tx':
            self._mining_tx_count += 1
        return entry

    def register_agent(self, agent_id):
        """Record an agent joining the mining network."""
        self.registered_agents.add(agent_id)
        self._append('register_agent', {'agent_id': agent_id})

    def quarantine_agent(self, agent_id, reason='unspecified'):
        """Record a quarantine decision made by RMP. Idempotent --
        re-quarantining an already-quarantined agent is a no-op
        beyond the log entry, consistent with RMP's own idempotent
        quarantine checks (should_quarantine returns False once an
        agent is already in the quarantined set)."""
        self.quarantined_agents.add(agent_id)
        self._append('quarantine_agent', {
            'agent_id': agent_id,
            'reason': reason,
        })

    def record_flag(self, reporter_id, target_id, reporter_trust, evidence=None):
        """Record a trust-weighted Sybil flag for audit purposes."""
        self._append('record_flag', {
            'reporter_id': reporter_id,
            'target_id': target_id,
            'reporter_trust': float(reporter_trust),
            'evidence': evidence or {},
        })

    def record_mining_tx(self, agent_id, value=0.5, action_type='action'):
        """
        Record a generic mining-related transaction. Called uniformly
        by edge_mining_env.py across every action handler (MineData,
        ValidateNode, FlagSybil, CurateGraph, AdjustPrivacy) with the
        same placeholder (value, action_type) pair in the current
        implementation -- this method does not interpret or act on
        those values, it only logs the call and returns a reference
        ID, consistent with every other record_* method here being
        pure audit-trail bookkeeping rather than a decision point.

        Returns the entry's hash, used by callers as a tx_id.
        """
        entry = self._append('mining_tx', {
            'agent_id': agent_id,
            'value': value,
            'action_type': action_type,
        })
        return entry['hash']

    def verify_chain(self):
        """Verify hash-chain integrity end to end. Returns True if
        no entry has been tampered with, False otherwise."""
        for i in range(1, len(self.chain)):
            entry = self.chain[i]
            recomputed = self._compute_hash({
                k: v for k, v in entry.items() if k != 'hash'
            })
            if recomputed != entry['hash']:
                return False
            if entry['prev_hash'] != self.chain[i - 1]['hash']:
                return False
        return True

    def get_stats(self):
        # NOTE: chain_intact/verify_chain() deliberately NOT called
        # here. verify_chain() is O(chain length), and get_stats() is
        # called from inside the per-step, per-agent state builder
        # (up to 100 agents x 50 steps = 5,000 calls/episode against
        # an ever-growing chain) -- this was producing severe O(n^2)
        # slowdown across an episode (measured: step time growing from
        # 0.24s to 1.85s within just 20 steps of one episode). Confirmed
        # chain_intact is read nowhere else in the codebase, so removing
        # it from this routine stats path is a pure performance fix with
        # no behavior change. verify_chain() remains available to call
        # explicitly and separately for a genuine end-of-run audit.
        return {
            'chain_length': len(self.chain),
            'registered_agents': len(self.registered_agents),
            'quarantined_agents': len(self.quarantined_agents),
            'total_tx': self._mining_tx_count,  # O(1), was O(chain length)
        }