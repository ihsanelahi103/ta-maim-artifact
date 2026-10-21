# Pre-registration, written 16 Oct 2026 15:55 local, BEFORE any run of the experiments below (code smoke tests use 10 episodes only and are not analysed).
Motivation: external reviewer assessment of main_2026-10-16 (novelty = audit frequency; real-signal failure).

## P1. Audit-period sweep (replay; no new simulation)
Streams: results/fs/{qmatch,collude,qm_ae10,qm_l050,camou,bc10,bc20}_s{7,11,42,99,123}.json (existing shadow-mode streams).
Rule: the CATS rule (S>0.5 & C>0.3 & suspicion>0.55, raw quality, warm-up 3) evaluated only at episodes P,2P,... with P in {1,2,5,10,25,50}. P=1 is the in-loop schedule.
Metrics per regime: Sybils flagged /30, honest flagged /70, runs with 30/30, median first-flag episode, exposure (fraction of Sybil MineData calls before the flag), number of evaluations per 200 episodes, measured seconds per evaluation (replay incl. history update).
Reporting: whole table goes to the appendix; main text states exposure at P=1,5,10,25 for qmatch and what an evaluation costs. No claim beyond what the table shows. Honest-FP rows reported even if worse at P=1.
Interpretation fixed in advance: if exposure at P<=2 is within 0.01 of P=1 and at P>=10 is >=1.5x P=1 for qmatch, the paper says "the benefit is the audit period; the in-loop loop is the cheapest way to get P=1" and the cost of P=1 is stated. If not, we report the actual curve.

## P2. Second quality signal ("signal B", extremity)
Definition (fixed): quality of a contributor's batch in an episode = 1 - (fraction of the batch's reviews with 5 stars); a batch with no reviews carries the previous value (initial 0.75). No reference means, no history, no per-dataset parameters, no rescaling, no thresholds changed. Everything else (TA-MAIM loop, parameters, attacks F5x3 = 3 fake 5-star/episode, F5x1, F4x3, targets, seeds 7,11,42,99,123, chronological split, ranking evaluation, baselines none/oracle/Q<0.5/SybilRank) identical to E3.
Runs: Yelp F5x3, F5x1, F4x3 and Amazon F5x3, 5 seeds each (20 runs); variants are named with suffix 'e' (F5x3e ...).
Expected by design (stated before the runs): F4x3e cannot be separated (fabricated reviews are 4-star, not extreme); honest agents with many 5-star reviews are at risk of false positives.
Decision rule: the paper reports the 4 configs x metrics (Sybils quarantined /30, honest FP /70, AUC(-q), NDCG@10 none/oracle/TA-MAIM/Q<0.5) next to signal A, whatever the outcome, as one extra row group in the real-signal table. The paper's statement that TA-MAIM fails with review-derived signals is relaxed ONLY IF signal B quarantines >=25/30 Sybils at <=2 honest FP in >=4 of 5 seeds for Yelp F5x3e; otherwise it states that two independently motivated signals both failed. No variant of signal B, mapping or threshold will be tried after seeing results.

## P3. Pathway diagnosis of E3 honest false positives (logging only; behaviour unchanged)
Rerun the 20 original E3 runs (signal A) with pathway logging. Check: quarantine sets must reproduce e3_summary.txt exactly (Sybils/30 and honest/70 per run); if they do not reproduce, report the discrepancy. Report for each dataset which pathway (P1 flag consensus, P2 trust collapse, P3 CATS) first quarantined honest agents and Sybils. Explanation only; no retuning.
