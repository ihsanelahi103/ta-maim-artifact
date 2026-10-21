"""Print every finished federated-recommendation result in one paste (run from the repo root).
usage: python experiments/summarize_fedrec.py
Baselines: results/fedrec/<dataset>_s<seed>.json ; TA-MAIM: results/fedrec_tamaim_<dataset>_s<seed><tag>.json"""
import glob, json, os, re
import numpy as np

def tam_files(ds):
    out = []
    for f in glob.glob('results/fedrec_tamaim_%s_s*.json' % ds):
        m = re.match(r'results/fedrec_tamaim_%s_s(\d+)(.*)\.json$' % ds, f)
        if not m: continue
        seed, tag = int(m.group(1)), m.group(2)
        agg = 'normclip' if tag.endswith(('_nc', '_normclip')) else 'mean'
        out.append((seed, agg, f))
    return out

def ms(v):
    v = [x for x in v if x is not None]
    if not v: return '-'
    return '%.3f' % np.mean(v) + (' ±%.3f' % np.std(v, ddof=1) if len(v) > 1 else '')

for ds in ('yelp', 'amazon'):
    base = {}
    for f in sorted(glob.glob('results/fedrec/%s_s*.json' % ds)):
        seed = int(re.search(r'_s(\d+)\.json', f).group(1))
        rows = json.load(open(f))
        for r in rows:
            base.setdefault(r['method'], {}).setdefault(bool(r['attack']), {})[seed] = r
    if not base: continue
    seeds = sorted({s for m in base.values() for a in m.values() for s in a})
    print('\n==== %s: baseline seeds present: %s' % (ds.upper(), seeds))
    print('%-13s | n | NDCG clean | NDCG attack | target HR@10 attack | clean HR@10' % 'method')
    for m, d in base.items():
        both = sorted(set(d.get(False, {})) & set(d.get(True, {})))
        print('%-13s | %d | %s | %s | %s | %s' % (m, len(both),
              ms([d[False][s]['ndcg'] for s in both]), ms([d[True][s]['ndcg'] for s in both]),
              ms([d[True][s]['hr10'] for s in both]), ms([d[False][s]['hr10'] for s in both])))
    tam = {}
    for seed, agg, f in tam_files(ds):
        d = json.load(open(f)); L = d['log']
        tam.setdefault(agg, {})[seed] = (d, L)
    for agg, per in tam.items():
        sd = sorted(per)
        print('TA-MAIM+%-8s seeds %s | NDCG %s | target HR@10 %s' % (agg, sd,
              ms([per[s][0]['ndcg'] for s in sd]), ms([per[s][0]['hr10'] for s in sd])))
        for s in sd:
            L = per[s][1]
            print('   seed %d: active sybils/honest round 49: %s/%s ; round 200: %s/%s' % (
                s, L['active_sybil'][48], L['active_honest'][48], L['active_sybil'][-1], L['active_honest'][-1]))
    # extra diagnostics from baselines
    for m in ('recess', 'conda', 'flame'):
        for s, r in sorted(base.get(m, {}).get(True, {}).items()):
            ex = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k.startswith(('recess_', 'conda_', 'flame_'))}
            print('   %s seed %d: %s' % (m, s, ex))
