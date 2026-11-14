"""[codex] Explicit effective-gate kinetic scenarios; no silent default change."""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np

B=Path(__file__).resolve().parent
PARAMETER_INDICES=(9,10,13,14,15)
PARAMETER_NAMES=('thmin','threc','ta','tmfast','tndeact')
WAVEFORM_MULTIPLIER_RANGES={name:[.6,1.,1.4] for name in PARAMETER_NAMES}

def waveform_kinetic_reference():
    """Scenario ranges and selected profiles, never a human normal interval."""
    path=B/'selected.json'
    if not path.exists():raise RuntimeError('No completed waveform profile selection is available')
    return json.loads(path.read_text(encoding='utf8'))

def apply_waveform_kinetics(options,*,multipliers=None):
    """Return independent inputs; old dictionaries/arrays remain intact.

    This changes effective gate time functions only. It cannot be interpreted
    as a measured Kv7, SK or BK rate. Full tests cover the declared C08+acid
    conditions, not arbitrary mixtures of options from other profiles.
    """
    profile=options.get('profile')
    if profile not in ('Burley','Selvaraj-Con1','Selvaraj-Con2'):raise ValueError('Unknown waveform profile')
    p=np.asarray(options.get('profile_p'),dtype=float)
    if p.shape!=(16,) or not np.isfinite(p).all():raise ValueError('Expected finite sixteen-parameter HH profile')
    if multipliers is None:
        reference=waveform_kinetic_reference()['profiles'][profile]
        if not np.array_equal(p,np.array(reference['original_parameters'])):
            raise ValueError('Selected kinetics require the declared original HH profile; do not apply twice')
        multipliers=reference['multipliers']
    x=np.asarray(multipliers,dtype=float)
    if x.shape!=(5,) or not np.isfinite(x).all() or np.any(x<.6) or np.any(x>1.4):
        raise ValueError('Five finite kinetic multipliers within declared scenario bounds .6..1.4 required')
    out=dict(options);q=p.copy();q[list(PARAMETER_INDICES)]*=x;out['profile_p']=q
    if 'control_context' in options:
        out['control_context']=deepcopy(options['control_context'])
        out['control_context']['waveform_kinetics']=dict(profile=profile,multipliers=x.tolist(),
            parameter_names=PARAMETER_NAMES,original_parameters=p.tolist(),human_calibrated=False,
            scope='Conditional effective-gate scenario; tests apply only to declared bath, temperature, geometry and C08 acid inputs')
    return out
