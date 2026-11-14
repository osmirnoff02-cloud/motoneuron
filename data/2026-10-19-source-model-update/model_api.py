"""Codex 2026-10-19: source-aware observation layer, inherited physiology unchanged.

No automatic transfer of disease fluorescence / fish / rat / reporter constants
to healthy human motoneurons. See README and source_training_data().
"""
from pathlib import Path
import importlib.util
import sys
import json
import numpy as np

B=Path(__file__).resolve().parent
prior=B.parent/'2026-10-19-equal-calcium-load'
spec=importlib.util.spec_from_file_location('_equal_api_before_sources_20261019',prior/'model_api.py')
_old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=_old;spec.loader.exec_module(_old)
_INHERITED=('equal_entry_protocol','calcium_equal_entry','inward_current_to_ca_flux',
    'observe_cleavage_reporter','observe_pulse_chase','enzyme_driven_chase','simulate_equal_load',
    'CalpainTDPParameters','baseline_equilibrium','simulate_calpain_tdp','bursch_calpain_scenario',
    'refined_profile','refined_electrical','run_recovery','preserved_electrical','bursch_healthy_profile',
    'bursch_calcium_scenario','compare_bursch_ampa_amplitude','observe_calibrated_fura2','observe_spikes',
    'nmda_fura2_response','hess_dye_calibration','ERScenario','er_exchange')
for name in _INHERITED:globals()[name]=getattr(_old,name)

def source_training_data():
    """Return measurements with their source/assay limits, not pooled fit targets."""
    names=('boivin_fitted_observation.json','controls_Hirunagi_healthy_summary.json',
           'source_benchmarks.json','gerencser_benchmarks.json')
    return {n:json.loads((B/n).read_text(encoding='utf-8')) for n in names}

def apply_boivin_drift(time_s, fluorescence, *, plate, well, line, timepoint, run, field, channel, sequence_sha256):
    """Apply only the fitted pilot field's nuisance correction, with explicit provenance.

    Input is already background-subtracted fluorescence; last axis is time.
    Cannot extrapolate, estimate [Ca], or apply this disease fit to 39b-cor.
    """
    doc=json.loads((B/'boivin_fitted_observation.json').read_text(encoding='utf-8'))
    keys=('plate','well','line','timepoint','run','field','channel','sequence_sha256')
    found=[p for p in doc['profiles'] if tuple(p[k] for k in keys)==(plate,well,line,timepoint,run,field,channel,sequence_sha256)]
    if len(found)!=1:raise ValueError('No fitted correction for this exact source/field/line/timepoint')
    p=found[0];t=np.asarray(time_s,float);f=np.asarray(fluorescence,float)
    if (t.ndim!=1 or len(t)<2 or f.ndim<1 or f.shape[-1]!=len(t)
        or not np.isfinite(t).all() or not np.isfinite(f).all() or np.any(np.diff(t)<=0)
        or t[0]<0 or t[-1]>44):raise ValueError('Finite 0..44s paired fluorescence required, no extrapolation')
    factor=np.interp(t,p['correction_time_s'],p['correction_factor'])
    return dict(time_s=t,fluorescence_corrected=f/factor,
        status='Within-field optical nuisance correction only; no absolute Ca or physiological rate',
        profile=p,independent_biological_validation=False)

# Assay helpers are imported by explicit file path so old same-named API modules
# on sys.path cannot silently substitute an earlier version.
spec=importlib.util.spec_from_file_location('_source_assay_observers_20261019',B/'assay_observers.py')
assay_observers=importlib.util.module_from_spec(spec);sys.modules[spec.name]=assay_observers;spec.loader.exec_module(assay_observers)
for name in ('simulate_substrate','simulate_fluorescent_intact','robinson_fret',
             'epitope_linear_mixture','gerencser_corrected_fret'):
    globals()[name]=getattr(assay_observers,name)

def observe_biochemistry_substrate(biochemistry, *, observation_time_s, **assay_parameters):
    """Feed inherited effective calpain into an explicitly specified assay.

    Calpain is interpolated to interval starts and held constant per interval.
    A chosen observation grid is a numerical approximation, not a new measured
    calpain trace. Caspase/synthesis/removal/kinetics stay explicit caller inputs.
    """
    t=np.asarray(biochemistry['time_s'],float)
    activity=np.asarray(biochemistry['effective_activity'],float)
    grid=np.asarray(observation_time_s,float)
    if (t.ndim!=1 or activity.shape!=t.shape or len(t)<2 or not np.isfinite(t).all()
        or not np.isfinite(activity).all() or np.any(np.diff(t)<=0) or np.any(activity<0)
        or grid.ndim!=1 or len(grid)<2 or not np.isfinite(grid).all() or np.any(np.diff(grid)<=0)
        or grid[0]<t[0] or grid[-1]>t[-1]):raise ValueError('Paired calpain/grid within original time range required')
    if 'calpain_activity' in assay_parameters:raise ValueError('Calpain must come from biochemistry, not an override')
    sampled=np.interp(grid[:-1],t,activity)
    result=simulate_substrate(grid,calpain_activity=sampled,**assay_parameters)
    result['driving_effective_calpain']=sampled
    result['coupling_status']='Inherited conditional calpain -> separate assay; no TDP or electrical feedback'
    return result
