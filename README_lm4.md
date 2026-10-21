# VOI verification package (pre-registration: preregistration_voi_verification_2026-10-19.md)
Run from the repo root (lm2_run.py and lm3_run.py must already be there; lm3_run.py hash must stay a8dead711d6b62da...).
```
unzip -o lm4_pkg.zip
sha256sum experiments/lm4_run.py experiments/lm3_run.py     # compare with the pre-registration
nohup bash run_fs.sh jobs_bv2_test.txt 2 > bv2_fs.out 2>&1 &     # 40 new streams, seeds 201-205, about 3-4 h
ls results/fs/*_s20[1-5].json | wc -l                             # 40 when finished
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=""
python experiments/lm4_run.py --budgets 1 2 2>&1 | tee lm4.out    # a few minutes; first run builds 40 stream caches (about 15 min)
```
Do not edit lm4_run.py. Send: cat lm4.out
