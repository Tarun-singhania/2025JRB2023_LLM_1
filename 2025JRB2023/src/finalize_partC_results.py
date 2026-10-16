#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('--base',default='results/detail/partC_attacks_base.json')
ap.add_argument('--raw',default='results/detail/partC_attacks_raw.json')
ap.add_argument('--san',default='results/detail/partC_attacks_san.json')
ap.add_argument('--utility',default='results/partC_utility.json')
ap.add_argument('--failure_traces',default='results/detail/partC_failure_traces.json')
ap.add_argument('--output',default='results/partC_attacks.json')
ap.add_argument('--plot',default='results/partC_exposure.png')
a=ap.parse_args()
load=lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
base,raw,san,utility=load(a.base),load(a.raw),load(a.san),load(a.utility)
combined={
 'utility': {'v_base_ppl':utility['v_base']['ppl'],'v_raw_ppl':utility['v_raw']['ppl'],'v_san_ppl':utility['v_san']['ppl']},
 'models': {'v_base':base,'v_raw':raw,'v_san':san},
 'notes': {'pii_index':'Detector-derived regex+Presidio index because partC_pii_index.jsonl was absent.', 'canary_estimator_ceiling_bits':13.287712379549449},
}
if Path(a.failure_traces).exists(): combined['c3_failure_traces']=load(a.failure_traces)
# Compatibility with the assignment's example single-victim top-level schema.
combined.update({k:raw[k] for k in ['victim','heldout_ppl','discoverable','untargeted','canary','targeted'] if k in raw})
if 'mean_edit_similarity' in combined['discoverable']:
    combined['discoverable'].setdefault('mean_edit_sim',combined['discoverable']['mean_edit_similarity'])
Path(a.output).write_text(json.dumps(combined,indent=2),encoding='utf-8')

xr=sorted(int(k) for k in raw['canary']['by_n_reps']); xs=sorted(int(k) for k in san['canary']['by_n_reps'])
yr=[raw['canary']['by_n_reps'][str(k)]['mean_exposure'] for k in xr]
ys=[san['canary']['by_n_reps'][str(k)]['mean_exposure'] for k in xs]
plt.figure(figsize=(6.4,4.2)); plt.plot(xr,yr,marker='o',label='V-raw'); plt.plot(xs,ys,marker='o',label='V-san')
plt.xlabel('Canary repetitions'); plt.ylabel('Mean exposure (bits)'); plt.xscale('log',base=2); plt.legend(); plt.tight_layout(); plt.savefig(a.plot,dpi=200)
print('Saved',a.output,'and',a.plot)
