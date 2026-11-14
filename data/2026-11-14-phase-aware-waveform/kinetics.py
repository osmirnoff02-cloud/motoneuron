"""Codex. Opt-in phase-tested effective gates; no change to legacy defaults."""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
B=Path(__file__).resolve().parent
PARAMETER_INDICES=(8,9,10,11,13,14,15)
PARAMETER_NAMES=('tm','thmin','threc','tn','ta','tmfast','tndeact')
PHASE_KINETIC_BOUNDS={'tm':(.5,1.5),'thmin':(.6,1.4),'threc':(.6,1.4),'tn':(.25,2.),'ta':(.6,1.4),'tmfast':(.6,1.4),'tndeact':(.6,1.4)}
def phase_kinetic_reference():
    path=B/'accepted.json'
    if not path.exists():raise RuntimeError('No completed full-model acceptance is available')
    return json.loads(path.read_text(encoding='utf8'))
def apply_phase_kinetics(options):
    reference=phase_kinetic_reference();profile=options.get('profile')
    if profile not in reference['profiles']:raise ValueError('Unknown phase-test profile')
    if any(k in options.get('control_context',{}) for k in ('phase14','waveform14','waveform_kinetics')):
        raise ValueError('A kinetic scenario was already applied')
    row=reference['profiles'][profile];p=np.asarray(options.get('profile_p'),float)
    if p.shape!=(16,) or not np.array_equal(p,np.array(row['original_parameters'])):
        raise ValueError('Phase kinetics require original profile; do not apply after another kinetics helper')
    x=np.asarray(row['multipliers'],float)
    if x.shape!=(7,) or not np.isfinite(x).all():raise ValueError('Seven finite multipliers required')
    for value,name in zip(x,PARAMETER_NAMES):
        lo,hi=PHASE_KINETIC_BOUNDS[name]
        if not lo<=value<=hi:raise ValueError('Multiplier outside declared scenario bounds')
    q=p.copy();q[list(PARAMETER_INDICES)]*=x
    if not np.allclose(q,row['parameters'],rtol=0,atol=1e-14):raise ValueError('Inconsistent selected parameter record')
    out=dict(options);out['profile_p']=q
    out['control_context']=deepcopy(options.get('control_context',{}))
    out['control_context']['phase14']=dict(indices=list(PARAMETER_INDICES),multipliers=x.tolist(),human_calibrated=False)
    return out
