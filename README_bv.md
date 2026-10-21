# Binding-budget verification package (pre-registration: preregistration_binding_budget_2026-10-18.md, amended before any run)
Run from the repo root (the folder that contains experiments/, results/, run_fs.sh):
```
unzip -o bv_pkg.zip
sha256sum experiments/lm3_run.py        # must be a8dead711d6b62da7a9bf4f5e7c22fbb8c714275a044de2383dcf178f58d281c
bash run_fs.sh jobs_bv_test.txt 2       # 40 new streams, seeds 101-105 (about 10 min each with 2 in parallel, about 3-4 h); restart-safe
ls results/fs/*_s10[1-5].json | wc -l   # 40 when finished
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=""
nohup python experiments/lm3_run.py --budgets 1 2  > lm3_B1_2.out  2>&1 &
nohup python experiments/lm3_run.py --budgets 5 20 > lm3_B5_20.out 2>&1 &
```
Start lm3 when all 40 streams exist, or at the fixed deadline (Mon 19 Oct 22:00 local), whichever comes first; at the deadline only regimes with all five seeds are used (declared in the pre-registration). Do not edit lm3_run.py.
The first lines of each output must show "LEAK CHECK ... True". When both finish, send me: cat lm3_B1_2.out lm3_B5_20.out
