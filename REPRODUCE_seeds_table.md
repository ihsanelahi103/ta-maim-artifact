# Seeds, job files and scripts per study (for the anonymous repository README)
Main seeds M = {7, 11, 42, 99, 123}. Fresh seeds F = {1, 2, 3, 4, 5} (replication only). Seed 7 was the development stream for the NV-B scheduler (its criteria use M minus 7).
All shadow streams are written to results/fs/<regime>_s<seed>.json; all analysis scripts read those files; restart-safe drivers skip finished runs.

| Study (paper location) | Seeds | Job file / driver | Analysis script | Output |
|---|---|---|---|---|
| Detector comparison, Table 3 (static, collusion, Qmatch, bands 1-4, camouflage) | M | jobs_e1_s1.txt via `bash run_fs.sh` | experiments/verify_fs_table.py (also fs_analyze.py) | results/fs/*_s{7,11,42,99,123}.json |
| Detector comparison, BC10/BC20 and Qmatch weakening, benign communities | M | jobs_e2.txt via `bash run_fs.sh` | verify_fs_table.py, fs_analyze.py | results/fs/ |
| Audit period (app. Audit period) | M | same streams | experiments/fs_audit_sweep.py | results/fs_audit_sweep.json |
| Ensembles NV-A and adaptive audit NV-B (app. Ensembles) | M minus 7 (criteria); 7 = development | same streams | experiments/nv_run.py | results/nv_results.json |
| Replication on fresh seeds (app. Replication) | F | jobs_rep.txt via `bash run_fs.sh` | experiments/rep_summarize.py | results/rep_results.json |
| Controls / ablations (Table 4 etc.) | M | jobs_f1.txt via `bash run_fw.sh` | see experiments/fw_run.py header | results/fw/ |
| Real-signal tests A and B (Table 5) | M | jobs_e3.txt (signal A), jobs_e3b.txt (signal B) via `bash run_rs.sh` | experiments/rs_summarize.py | results/rs/*_s*.json |
| Self-calibration NV-C (Sec. 5.6) | M | jobs_nv.txt via `bash run_rs.sh` | experiments/rs_summarize.py (variants F5x3n, F5x3en) | results/rs/ |
| Learned merge / verification allocation (app.) | see headers of experiments/lm_run.py, lm2_run.py | - | same scripts | results/lm_results.json, lm2_results*.json |
| Binding-budget verification allocation (not in paper) | 101-105 | jobs_bv_test.txt via `bash run_fs.sh` | experiments/lm3_run.py | results/lm3_results_*.json |
| Model-derived verification policy VOI (not in paper) | 201-205 | jobs_bv2_test.txt via `bash run_fs.sh` | experiments/lm4_run.py, audit_voi.py, lm4_report.py | results/lm4_results_B*.json, results/lm4_per_stream_B*.csv |
Pre-registrations (frozen before the runs; dates in file names) are in the project docs; NV-C and the replication were registered on 17 Oct 2026 before any run.
