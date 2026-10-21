# TA-MAIM (anonymous artifact for WWW 2027 submission)
Code and scripts for the experiments in "TA-MAIM: Trust-Aware Autonomous Multi-Agent Intelligence for Collaborative Edge Web Mining".
All attacks are simulated (generated agent graph and quality); Yelp and Amazon review shards are used only for the ranking task and for the review-derived quality signals. Raw datasets (the public Yelp and Amazon review releases) are not included.

## Name map (class and folder names kept from earlier versions)
| Paper | Code |
|---|---|
| GATEM: trust propagation (TPP) + quarantine (RMP) | `algorithms/tpp`, `algorithms/brmp` (the `brmp` folder name is historical; there is no blockchain in the experiments) |
| CATS (group pathway) | `algorithms/collusion/collusion_detector.py` |
| DAPP (windowed Laplace privacy) | `algorithms/dapp` |
| TA-DDQN (policy layer; class `BT-MO-DQN` in older code) | `algorithms/bt_mo_dqn` |

## Reproducing
`REPRODUCE_seeds_table.md` lists, for every study, the seeds, job file, analysis script and output files.
Quick checks (need the shadow streams in `results/fs/`):
```
python experiments/verify_fs_table.py      # recomputes the detector-comparison table on seeds 7,11,42,99,123
python experiments/rep_summarize.py        # fresh-seed replication (seeds 1-5)
python experiments/nv_run.py               # ensemble and adaptive-audit comparison
python experiments/rs_summarize.py         # review-derived signals and self-calibration
```
Shadow runs: `bash run_fs.sh jobs_e1_s1.txt 2`; framework controls: `bash run_fw.sh jobs_f1.txt 2`; review-derived signals: `bash run_rs.sh jobs_e3.txt 2`. All drivers are restart-safe.
Pre-registrations were written before the corresponding runs; dates are in the file names of the supplementary protocol documents.

## Data and streams
The shadow-mode event streams (`results/fs/*.json`, about 20-25 MB each) and the stream caches are not included. They are regenerated from the job files with the drivers above (restart-safe); the analysis scripts read them from `results/fs/`. Summary files that are small are included under `results/`.

## Additional studies not reported in the paper
Two further studies were run under pre-registered protocols and gave negative or inconclusive results; they are included for completeness: binding-budget verification allocation (`experiments/lm3_run.py`, `README_bv.md`, `jobs_bv_test.txt`, seeds 101-105) and a model-derived verification policy with an information audit and a per-stream report (`experiments/lm4_run.py`, `experiments/audit_voi.py`, `experiments/lm4_report.py`, `README_lm4.md`, `jobs_bv2_test.txt`, seeds 201-205).
