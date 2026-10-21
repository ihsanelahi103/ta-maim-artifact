"""
TA-MAIM: Trust-Aware Autonomous Multi-Agent Intelligence
Configuration file — all hyperparameters and system settings in one place.
"""

# ─────────────────────────────────────────────
# SYSTEM IDENTITY
# ─────────────────────────────────────────────
SYSTEM_NAME = "TA-MAIM"
PAPER_TITLE = (
    "Trust-Aware Autonomous AI Agents for "
    "Trust-Aware Collaborative Edge Data Mining"
)
VENUE = "WWW 2027"

# ─────────────────────────────────────────────
# SCALABILITY EXPERIMENT CONFIGURATIONS
# Switch by setting SCALE = "small" | "medium" | "large"
# ─────────────────────────────────────────────
SCALE = "medium"

SCALE_CONFIGS = {
    "small": {                          # Small deployment
        "num_agents": 50,
        "num_honest": 35,
        "num_sybil":  15,
        "label": "50 agents (35H+15S)",
    },
    "medium": {                         # Main evaluation — WWW primary scale
        "num_agents": 100,
        "num_honest": 70,
        "num_sybil":  30,
        "label": "100 agents (70H+30S)",
    },
    "large": {                          # Scalability proof — WWW main result
        "num_agents": 200,
        "num_honest": 140,
        "num_sybil":  60,
        "label": "200 agents (140H+60S)",
    },
}

# ─────────────────────────────────────────────
# AGENT CONFIGURATION — reads from active SCALE
# ─────────────────────────────────────────────
AGENT = {
    **SCALE_CONFIGS[SCALE],
    "sybil_fraction": 0.30,            # Always 30% Sybil across all scales

    # Agent state space (20-dim — same across all scales)
    # Local metrics (6) + Trust metrics (5) + Graph metrics (5) + System (4)
    "state_dim": 20,

    # Action space
    # MineData, ValidateNode, FlagSybil, CurateGraph, AdjustPrivacy
    "action_dim": 5,
    "actions": [
        "MineData",        # Mine and contribute data to shared knowledge base
        "ValidateNode",    # Validate a peer agent's contribution
        "FlagSybil",       # Flag a peer as Sybil — triggers on-chain report
        "CurateGraph",     # Curate local graph — remove low-trust edges
        "AdjustPrivacy",   # Adjust local DP epsilon based on threat level
    ],
}

# ─────────────────────────────────────────────
# STATE SPACE DEFINITION (20 dimensions)
# ─────────────────────────────────────────────
STATE_SPACE = {
    # --- Local agent metrics (indices 0-5) ---
    0:  "local_mining_quality",        # Quality score of own mined data [0,1]
    1:  "delta_mining_quality",        # Change in mining quality vs last step
    2:  "cpu_utilization",             # CPU usage [0,1]
    3:  "memory_utilization",          # Memory usage [0,1]
    4:  "contribution_count",          # Normalised count of contributions made
    5:  "validation_success_rate",     # % of own validations confirmed correct

    # --- Trust and reputation metrics (indices 6-10) ---
    6:  "own_trust_score",             # Own on-chain trust score [0,1]
    7:  "peer_trust_variance",         # Variance of neighbour trust scores
    8:  "sybil_suspicion_score",       # Own Sybil suspicion score from peers
    9:  "global_threat_level",         # System-wide Sybil threat Θ [0,1]
    10: "reputation_trend",            # Rising (+) or falling (-) reputation

    # --- Graph structural metrics (indices 11-15) ---
    11: "degree_centrality",           # Node degree / max degree in graph
    12: "clustering_coefficient",      # Local clustering coefficient [0,1]
    13: "neighbour_avg_trust",         # Average trust of direct neighbours
    14: "sybil_neighbour_ratio",       # Fraction of neighbours flagged Sybil
    15: "graph_density_local",         # Local subgraph density

    # --- System / blockchain metrics (indices 16-19) ---
    16: "current_dp_epsilon",          # Current DP epsilon value
    17: "on_chain_latency",            # Normalised blockchain latency
    18: "gas_cost_normalised",         # Normalised gas cost of last TX
    19: "episode_progress",            # Normalised episode number [0,1]
}

# ─────────────────────────────────────────────
# GRAPH-AWARE BT-MO-DQN CONFIGURATION (C2)
# Extension of EdgeChain BT-MO-DQN:
# - 20-dim graph-aware state (vs EdgeChain 15-dim)
# - Graph structural features: centrality, clustering, Sybil suspicion
# - Trust-aware action semantics for collaborative web data mining
# ─────────────────────────────────────────────
BT_MO_DQN = {
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "gamma": 0.95,                     # Discount factor
    "tau": 0.01,                       # Soft target update rate
    "replay_buffer_size": 5000,
    "batch_size": 64,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.995,
    "update_frequency": 4,             # Update every N steps
    "target_update_frequency": 100,

    # Multi-objective reward weights (adaptive per threat level)
    # [mining_quality, trust_security, privacy, efficiency]
    "reward_weights": {
        "low":      [0.35, 0.25, 0.20, 0.20],   # Θ < 0.3
        "medium":   [0.30, 0.35, 0.20, 0.15],   # 0.3 ≤ Θ < 0.6
        "high":     [0.25, 0.40, 0.20, 0.15],   # Θ ≥ 0.6
    },

    # Byzantine-aware exploration boost
    "exploration_boost": {
        "low":    0.00,
        "medium": 0.05,
        "high":   0.15,
    },
}

# ─────────────────────────────────────────────
# TRUST PROPAGATION PROTOCOL (TPP) — C1/C3
# ─────────────────────────────────────────────
TPP = {
    "initial_trust": 0.5,              # All agents start at neutral trust
    "min_trust": 0.0,
    "max_trust": 1.0,
    "trust_decay": 0.99,               # Trust decays slightly each episode
    "positive_reinforcement": 0.05,    # Trust gain for good contribution
    "negative_reinforcement": 0.10,    # Trust loss for bad contribution
    "sybil_penalty": 0.20,             # Trust loss when flagged as Sybil
    "propagation_depth": 2,            # How many hops trust propagates
    "propagation_damping": 0.5,        # Damping factor per hop
    "quarantine_threshold": 0.15,      # Trust below this → quarantine
    "rehabilitation_threshold": 0.40,  # Trust above this → restore access
}

# ─────────────────────────────────────────────
# BLOCKCHAIN-BASED REPUTATION MANAGEMENT (BRMP) — C3
# ─────────────────────────────────────────────
BRMP = RMP = {
    "reputation_update_interval": 5,   # Episodes between reputation updates
    "min_reputation": 0.0,
    "max_reputation": 1.0,
    "initial_reputation": 0.5,
    "contribution_weight": 0.4,        # Weight of mining contribution in rep
    "validation_weight": 0.3,          # Weight of validation accuracy in rep
    "consistency_weight": 0.3,         # Weight of interaction consistency
    "sybil_flag_threshold": 8,         # Reports before on-chain quarantine
    "reputation_history_window": 10,   # Episodes to consider for reputation
}

# ─────────────────────────────────────────────
# DISTRIBUTION-AWARE ADAPTIVE PRIVACY (DAPP) — C4
# ─────────────────────────────────────────────
DAPP = {
    "epsilon_init_mean": 1.0,
    "epsilon_init_std": 0.25,
    "epsilon_min": 0.30,               # Hard floor (same as EdgeChain)
    "epsilon_max": 3.0,
    "delta": 1e-5,

    # Adaptation multipliers per threat regime
    "alpha": {
        "low":    1.01,                # Slightly relax privacy when safe
        "medium": 0.95,                # Tighten under moderate threat
        "high":   0.92,                # Aggressively tighten under attack
    },

    # Non-IID awareness
    "distribution_skew_threshold": 0.3,   # KL divergence threshold for Non-IID
    "noniid_epsilon_penalty": 0.05,        # Extra tightening for skewed data
    "sensitivity": 1.0,                    # L2 sensitivity for Laplace mechanism
}

# ─────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────
TRAINING = {
    "num_episodes": 200,
    "max_steps_per_episode": 50,
    "warmup_episodes": 10,             # Random policy for initial exploration
    "eval_frequency": 10,              # Evaluate every N episodes
    "checkpoint_frequency": 20,        # Save checkpoint every N episodes
    "early_stopping_patience": 30,
    "seed": 42,
}

# ─────────────────────────────────────────────
# DATASET CONFIGURATION
# ─────────────────────────────────────────────
DATASET = {
    "yelp": {
        "raw_path": "data/yelp/",
        "processed_path": "data/processed/yelp/",
        "review_file": "yelp_academic_dataset_review.json",
        "user_file": "yelp_academic_dataset_user.json",
        "business_file": "yelp_academic_dataset_business.json",
        "min_reviews_per_user": 5,
        "min_reviews_per_business": 10,
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "sybil_injection_rate": 0.30,  # Synthetic Sybil fraction
    },
    "amazon": {
        "raw_path": "data/amazon/",
        "processed_path": "data/processed/amazon/",
        "review_file": "Electronics.jsonl",       # Amazon Electronics reviews
        "meta_file": "meta_Electronics.jsonl",    # Amazon Electronics metadata
        "min_reviews_per_user": 5,
        "min_reviews_per_item": 10,
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "sybil_injection_rate": 0.30,
    },
}

# ─────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────
EVALUATION = {
    "ndcg_k": [5, 10, 20],
    "hr_k": [5, 10, 20],
    "sybil_detection_threshold": 0.5,  # Trust score below this = Sybil pred
    "privacy_epsilon_target": 0.30,
    "baselines": [
        "FedAvg", "FedProx", "InFEDge",
        "SybilRank", "RGCN",
        "MADDPG", "MAPPO",
    ],
}

# ─────────────────────────────────────────────
# BLOCKCHAIN SIMULATION CONFIG
# ─────────────────────────────────────────────
BLOCKCHAIN = {
    "consensus": "PBFT",
    "num_nodes": 4,
    "fault_tolerance": 1,
    "avg_latency_ms": 444,             # Based on EdgeChain real measurements
    "gas_per_tx": {
        "register_agent": 42199,
        "mine_data": 17382,
        "validate_node": 11714,
        "flag_sybil": 20319,
        "update_reputation": 15000,
        "adjust_privacy": 8000,
    },
    "simulate_blockchain": True,       # True = simulate; False = real FISCO
}