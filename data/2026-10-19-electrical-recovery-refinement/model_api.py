"""Opt-in verified electrical refinement; preserved earlier API remains accessible.

No fitted calcium-to-spike, calpain, TDP-43, or muscle-contact model is introduced.
"""
from pathlib import Path
import importlib.util,sys,json
import numpy as np
B=Path(__file__).resolve().parent
prior=B.parent/'2026-10-19-recovery-protocols'
sys.path.insert(0,str(prior))
spec=importlib.util.spec_from_file_location('_pre_refinement_api',prior/'model_api.py')
_old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=_old;spec.loader.exec_module(_old)
from recovery_model import nr,h,recovery_features
for _name in ['preserved_electrical','bursch_healthy_profile','bursch_calcium_scenario','compare_bursch_ampa_amplitude','observe_calibrated_fura2','observe_spikes','nmda_fura2_response','hess_dye_calibration','ERScenario','er_exchange']:
    globals()[_name]=getattr(_old,_name)
sys.path.insert(0,str(B.parent/'2026-10-16-selvaraj-waveform'))
from waveform_fit import features,FEATURE_NAMES

def refined_profile(profile):
    doc=json.loads((B/'profiles-accepted.json').read_text(encoding='utf-8'))
    if profile not in doc['profiles']:raise ValueError('No accepted refinement for '+profile)
    r=doc['profiles'][profile]
    return np.array(r['parameters']),np.array(r['context'])

def refined_electrical(profile='Burley',current_pA=25.,dt_ms=.005):
    """Source-specific protocols, same counts; shape/latency are conditional refinements."""
    if not np.isfinite(current_pA) or not np.isfinite(dt_ms) or dt_ms<=0:raise ValueError('Finite current and positive dt required')
    if profile=='Kiskinis':return preserved_electrical(profile,current_pA)
    p,c=refined_profile(profile);bur=profile=='Burley';duration=500. if bur else 1000.
    h.celsius=23
    out=nr.run(p,c,current_pA,duration=duration,dt=dt_ms,pre=200,post=1800,rest=not bur,init=c[5] if bur else c[2])
    if bur:out['count']=int(out['features']['zero_crossings'])
    else:
        out['features']=dict(zip(FEATURE_NAMES,features(out['v'],dt_ms,200.,duration,c[2]).tolist()))
        out['count']=int(out['features']['count'])
    out['profile']=profile;out['version']='2026-10-19 conditional electrical refinement'
    out['limitations']='Partial source-conditioned electrical model. Full waveform criteria and calcium/TDP/NMJ calibration are unmet; recovery gains depend on protocol.'
    return out

def run_recovery(profile,*,current_pA,duration_ms,dt_ms=.005,pre_ms=1000.,post_ms=3000.,version='refined'):
    if version=='preserved':return _old.run_recovery(profile,current_pA=current_pA,duration_ms=duration_ms,dt_ms=dt_ms,pre_ms=pre_ms,post_ms=post_ms)
    if version!='refined':raise ValueError('version must be refined or preserved')
    if not np.isfinite([current_pA,duration_ms,dt_ms,pre_ms,post_ms]).all() or current_pA<=0 or duration_ms<=0 or dt_ms<=0 or pre_ms<100 or post_ms<100:raise ValueError('Invalid recovery protocol')
    p,c=refined_profile(profile);bur=profile=='Burley';h.celsius=23
    out=nr.run(p,c,current_pA,duration=duration_ms,dt=dt_ms,pre=pre_ms,post=post_ms,rest=not bur,init=c[5] if bur else c[2])
    out['recovery']=recovery_features(out['t'],out['v'],onset_ms=pre_ms,offset_ms=pre_ms+duration_ms)
    out['protocol']=dict(profile=profile,version=version,current_pA=current_pA,duration_ms=duration_ms,dt_ms=dt_ms,pre_ms=pre_ms,post_ms=post_ms,temperature_C=23,holding='held -70mV' if bur else 'free rest')
    return out
