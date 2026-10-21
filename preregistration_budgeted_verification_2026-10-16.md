# Pre-registration: budgeted verification allocation (written 16 Oct 2026 ~21:00 local, BEFORE any code for this study was run)
Status: EXPLORATORY side study. The WWW 2027 manuscript (16d) stays unchanged unless every criterion below is met AND an in-loop confirmation is later run. Not a MARL claim.

## Question
When validation is scarce, can a learned policy that chooses WHOM to verify beat the best fixed verification rule? (Earlier null: learned merge policy vs feedback-matched rules, preregistration_learned_merge.)

## Information separation (fixed)
- Decision information (policies may use): CATS S, C, suspicion; CATS latch flag (S>.5 & C>.3 & suspicion>.55, latching); window-15 mean/std of the contributor's raw quality; number of checks so far; running verification log-likelihood ratio (LLR); episode fraction.
- Training feedback (learner may use): (a) verification outcomes v for contributors it chose to verify; (b) per-episode pool-level validation reward R_t = U_t + N(0, 2^2) (aggregate held-out value of the merged pool minus verification cost; no per-contributor labels).
- Evaluation only: true Sybil identities, used to compute realised utility, honest-dropped share and Sybil-merged share. No policy sees them.

## Verification (observable, noisy; same for every method)
Verifying contributor j in episode t returns v = +d'/2 + z if j honest, -d'/2 + z if Sybil, z~N(0,1), d'=2 (single-check AUC ~0.92). z is pre-drawn per (j,t) per stream (common random numbers: paired across methods). LLR_j += -d'*v. Shared status rule: LLR_j > ln9 -> j confirmed Sybil (dropped for good); LLR_j < -ln9 -> confirmed honest (merged for good, overrides the CATS latch); otherwise default: merge iff not CATS-latched. Budget: at most B=10 checks per episode, system-wide (shared resource), cost c=0.10 each (utility units; an honest contribution is worth its raw quality ~0.8). Only contributors that mined in the episode and are unresolved are eligible.

## Utility (evaluation oracle)
Per episode: U_t = sum over merged contributions of (+q if honest, -LAM*q if Sybil) - c*#checks, LAM=1, q = raw quality of the contribution. Primary metric: cumulative net utility over episodes 3..E per stream.

## Methods (all get the same B, c, d', feedback type, streams, noise)
Rules (allocation only; status rule shared): RANDOM; RR (least recently checked); SUS (highest CATS suspicion first); FLAG (latched-flagged first, then SUS); BND (suspicion closest to .55). No-verification references: HARD (CATS latch, B=0), ALL (merge-all), ORACLE (true labels, no cost; ceiling).
Learned: LRN, a softmax (Plackett-Luce) policy with linear scores over the 10 decision features, trained by REINFORCE on discounted pool-level reward (gamma=.95, lr=.01, EMA baseline .05), top-B sampling during training, greedy top-B at test. LRN-F frozen after pre-training on the OTHER families (3 passes); LRN-O continues the same updates online on the test stream. 5 learner seeds. Hyperparameters fixed here; none tuned on test families.
Correct label: this is a shared-budget sequential verification / contextual-bandit problem with pool-level credit assignment, not multi-agent reinforcement learning.

## Held-out families and regimes (frozen; selected from rule-only baselines before any learner run)
Families: quality-matching {qmatch, qm_ae10, qm_l050, band1, band2}, camouflage {camou}, benign-community {bc10, bc20}; collusion {collude} is training-only (HARD already closes .97 of its gap, no headroom). Leave-one-family-out: pre-train on all other families (including collusion), test on the held-out one. 5 stream seeds each.

## Sanity gate (before interpreting anything)
On the training families the pre-trained learner must beat RANDOM on mean net utility; otherwise report "learner did not learn" and the study is INCONCLUSIVE for learning (not evidence against).

## Success criteria (all required)
1. Primary: mean relative improvement (U_LRN - U_R)/|U_R| >= 0.05 against the STRONGEST rule R on the held-out family itself (best of RANDOM, RR, SUS, FLAG, BND by family-mean utility; deliberately conservative), in >= 2 of the 3 held-out families.
2. The paired bootstrap 95% CI (resampling streams, learner seeds averaged) excludes 0 in each of those families.
3. Honest-dropped share and Sybil-merged share not higher than the rule's by more than 0.02 absolute.
4. No hidden-label training signal (guaranteed by the information separation above).
5. Matched budget and feedback for all methods.
Outcome classes: PASS (all met); INCONCLUSIVE (CI includes both 0 and 5% of the rule's utility, or sanity gate failed); FAIL otherwise (report as null).
Secondary (reported, not decisive): comparison with the best rule selected on training families; LRN-F vs LRN-O; sensitivity to d' in {1,3}, B in {5,20}, c in {.05,.20}.

## If PASS
Exploratory finding only. The manuscript changes only after an in-loop evaluation (actions affecting trust, quarantine and mining) confirms it, and only if time and the 8+4 page budget allow. The abstract registered on 18 Oct stays as is. If FAIL/INCONCLUSIVE: appendix paragraph at most; manuscript otherwise unchanged.

## Known limits (stated in advance)
Offline replay: actions do not influence the stream (Sybil behaviour, graph, qualities). The verifier is a modelled noisy oracle with fixed d'. CATS evidence is computed from the recorded stream. Poison utility (LAM) and cost c are modelling choices (sensitivity reported).

## Timeline
17 Oct protocol/code gate; 18-19 Oct run on user machine; 20 Oct decision gate.

## Implementation notes (added 16 Oct ~21:40, after the code was written and smoke-tested only for execution on 4 container streams x 60 episodes with 1 learner seed; no learner/rule numbers from that test enter the study and no setting was changed because of them)
- Decision features (10): bias, S, C, suspicion, latch flag, window-15 mean quality, 5 x window-15 std, checks/5, clip(LLR,-3,3)/3, episode fraction.
- REINFORCE details: updates every 25 episodes within a stream (chunk); discounted returns inside the chunk (gamma .95); advantages standardised by chunk std; EMA baseline .05; gradient norm clipped at 5; Plackett-Luce score-function gradient.
- Rule tie-breaks: random with a fixed per-stream seed. RR = least recently checked. FLAG = latched first, then suspicion.
- Verdict logic per family: PASS-family = mean relative improvement >= .05 vs the family's strongest rule AND bootstrap CI lower bound > 0 AND honest-dropped and Sybil-merged shares each <= rule + .02; INCONCLUSIVE-family = CI contains both 0 and .05; else FAIL-family. Overall: sanity gate failed -> INCONCLUSIVE; >= 2 PASS-families -> PASS; else INCONCLUSIVE if any family is inconclusive; else FAIL. Evaluated separately for LRN-F and LRN-O.
- Stream seeds are whatever results/fs/*_s*.json exist (5 per regime on the user's machine). Bootstrap resamples streams (2000 draws).
