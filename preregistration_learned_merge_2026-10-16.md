# Pre-registration: does a learned merge policy add anything beyond the trust rules? (16 Oct 2026, written before any learner result was seen)

## Question
TA-MAIM's learned policy (TA-DDQN) was shown to ignore trust/graph inputs. Reviewers may expect "autonomous multi-agent intelligence" to include learned decision making.
Can a learned policy that decides, per contribution, whether to merge it, add measurable value over the trust rules (HARD = CATS latch, THR, SOFT) and over merge-all?

## Task (offline replay on the existing uncensored shadow-mode streams results/fs/<regime>_s<seed>.json; no new simulation)
- Unit: agent j's contribution in episode t (episodes >= 3 where j mined). Action: merge or drop.
- Utility of merging raw quality q: +q if j honest, -LAM*q if j Sybil (fabricated data poisons the merge). Dropping = 0. LAM = 1 primary; 0.5 and 2 sensitivity.
- Oracle: drop Sybils, merge honest. Gap closed G = (U - U_all)/(U_oracle - U_all).
- All policies see only label-free inputs available at the start of episode t (data through t-1): CATS S, C, suspicion, window-15 mean/std of j's raw quality, S*C, episode fraction, bias.

## Policies (fixed now)
ALL; HARD (CATS S>.5 & C>.3 & suspicion>.55, latching; note: only the P3 pathway is in the shadow streams); THR (drop while suspicion>.55); SOFT (weight 1-suspicion).
Learned: LIN (linear contextual bandit) and MLP (16 tanh hidden units), eps-greedy merging eps=.10, SGD lr=.01, 3 passes, learner seeds 5.
Feedback: reward observed only for a merged contribution and only with probability PFB=.10 (a sparse validation check). This is an assumption the rules do not use; it is the analogue of the periodic audit.
Controls LIN0/MLP0: same learners with reward = observed quality only (no validation feedback; label-free).
Variants: -F frozen after pre-training; -O continues online on the test stream (same PFB).

## Protocol
Leave-one-FAMILY-out: families = quality-matching {qmatch, qm_ae10, qm_l050, band1, band2}, collusion {collude}, camouflage {camou}, benign-community {bc10, bc20}. Learner is pre-trained on all streams of the other families (all seeds) and tested on the held-out family. No hyperparameter is tuned after seeing results; any change is reported as post-hoc.

## Decision rule (macro-average of G over the four families, 5 stream seeds x 5 learner seeds; bootstrap over streams)
- Learned policy "adds value" iff best of {LIN-O, MLP-O, LIN-F, MLP-F} beats the best of {HARD, THR, SOFT} by >= 0.05 G (macro), the paired bootstrap 95% CI excludes 0, AND it is not worse than the best rule by more than 0.05 on any family.
- If it adds value only with feedback and LIN0/MLP0 do not, the claim is restricted to "with sparse validation feedback".
- Otherwise the paper states: trust rules suffice; the learned layer adds no measured value. This is an acceptable outcome and is reported as such.

## Where it goes in the paper
Positive: one table + paragraph (main if page budget allows, else appendix), abstract wording changed only after 21 Oct title gate.
Null/negative: appendix table + one sentence in Sec. 5.6 and Limitations.

## Known limits (stated in advance)
Offline replay: merge decisions do not feed back into the stream. HARD lacks P1/P2 pathways. Poison utility is a modelling choice (LAM sensitivity). Feedback assumption not used by rules.

## Amendment A1 (16 Oct, before the 5-seed run; AFTER a 4-stream container smoke test that showed online learners with feedback beating HARD on quality-matching streams)
The smoke test showed that learners receiving sparse validation feedback can look better than label-free rules simply because the feedback exposes Sybils. To make the comparison fair, two feedback-matched rules are added and join the rule set in the decision rule:
- FB: drop j for good once a validation check (prob PFB per merged contribution) exposes poison.
- HARD+FB: HARD or FB.
The learned policy now "adds value" only if it beats the best of {HARD, THR, SOFT, FB, HARD+FB} by the same margin. Smoke-test learner numbers (container, 4 streams, seed 7) are exploratory and not part of the evidence.
