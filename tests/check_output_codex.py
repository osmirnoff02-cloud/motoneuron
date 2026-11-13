"""Codex 2026-11-13: compare a real run with the packaged numerical reference."""
from pathlib import Path
import argparse, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
a=p.parse_args()
s=json.loads((a.output/'summary_codex.json').read_text(encoding='utf-8'))
assert s['states']==4425 and s['all_finite'] and s['no_reset']
assert all(np.isfinite(v) and abs(v)<3e-5 for v in s['checks'].values())
assert s['human_calibrated'] is False
assert (a.output/'figure_codex.png').is_file() and (a.output/'checkpoint_codex.json').is_file()
report={'protocol':s['protocol'],'state_count':s['states'],'balances_passed':True}
if s['protocol']=='smoke' and s['profile']=='Burley' and s['candidate']=='C08':
    with np.load(a.output/'trace_codex.npz',allow_pickle=False) as got, np.load(ROOT/'results/reference_smoke_codex.npz',allow_pickle=False) as ref:
        # Materialize compressed arrays once; repeated NPZ indexing decompresses each time.
        got={key:got[key] for key in got.files}
        ref={key:ref[key] for key in ref.files}
        assert np.array_equal(got['baseline'],ref['baseline']), 'Baseline changed'
        assert np.allclose(got['time_s'][[0,-1]],ref['time_s'][[0,-1]],rtol=0,atol=1e-12)
        # Solver-internal sampling can differ between supported library builds.
        expected=np.column_stack([np.interp(got['time_s'],ref['time_s'],ref['state'][:,i]) for i in range(4425)])
        exact_grid=np.array_equal(got['time_s'],ref['time_s'])
        if exact_grid:
            assert np.allclose(got['state'],ref['state'],rtol=2e-7,atol=1e-8), 'Full-state regression failed'
        else:
            # Endpoints are directly comparable; interpolated paths are diagnostic only.
            assert np.allclose(got['state'][[0,-1]],ref['state'][[0,-1]],rtol=2e-7,atol=1e-8), 'Endpoint regression failed'
        report.update(exact_time_grid=exact_grid,all_state_final_max_difference=float(np.max(abs(got['state'][-1]-ref['state'][-1]))),interpolated_path_max_difference=float(np.max(abs(got['state']-expected))))
    if 'restart' in s:assert s['restart']['exact_initial_state'] and s['restart']['within_tolerance']
elif s['protocol']=='pulse' and s['profile']=='Burley' and s['candidate']=='C08' and s.get('current_pA',25)==25 and s.get('start_time_s',0)==0:
    config=json.loads((a.output/'config_codex.json').read_text(encoding='utf-8'))
    # Only the documented 25-pA fresh-start demo is covered by this golden row.
    rows=json.loads((ROOT/'data/2026-11-13-joint-kcnq-pump-v2/electrical_summary_codex.json').read_text())['rows']
    target=next(x for x in rows if x['candidate']=='C08' and x['profile']=='Burley' and x['pA']==25 and x['start_s']==0)
    assert s['features']['spikes']==target['features']['spikes']==4, 'Documented 25-pA pulse count changed'
    assert abs(s['features']['peak_mV']-target['features']['peak_mV'])<.25, 'Pulse peak differs from historical reference'
    report['spikes_during_step']=s['features']['spikes']
else:
    report['scope']='Numerical checks only; no golden scientific target for this configuration'
print(json.dumps(report,indent=2))
