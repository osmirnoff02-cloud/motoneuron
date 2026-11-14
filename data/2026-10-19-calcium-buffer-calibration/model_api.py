"""Codex 2026-10-19. Extend the calcium model without rewriting electrical sources."""
from pathlib import Path
import importlib.util,sys,json
from calcium_calibration import (DyeCalibration,decode_adu,ratio_to_ca,ca_to_ratio,
    indicator_capacity,fit_decay,fit_added_buffer,buffered_response)
B=Path(__file__).resolve().parent
_prior=B.parent/'2026-10-16-calcium-data-integration'
sys.path.insert(0,str(_prior))
spec=importlib.util.spec_from_file_location('_calcium_api_20261016',_prior/'model_api.py')
_old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=_old;spec.loader.exec_module(_old)
preserved_electrical=_old.preserved_electrical
observe_spikes=_old.observe_spikes
nmda_fura2_response=_old.nmda_fura2_response
Fura2Calibration=_old.Fura2Calibration
observe_fura2=_old.observe_fura2


def hess_dye_calibration(cell_id):
    """Calibration for this mouse experiment's optics, never a human default."""
    rows=json.loads((B/'hess-calibrations.json').read_text(encoding='utf-8'))
    row=next((r for r in rows if r['cell']==cell_id),None)
    if row is None:raise ValueError('Unknown Hess cell')
    return DyeCalibration(**{k:row[k] for k in DyeCalibration.__dataclass_fields__})


def observe_calibrated_fura2(time_s,ca_uM,cycle_starts_s,*,calibration,
                            exposure340_s,exposure380_s,channel_gap_s):
    """Equilibrium optical scenario with explicit unequal sequential exposures.

    Outputs integrated channel counts in arbitrary scale and the exposure-
    normalized ratio expected by the calibration. Finite binding kinetics and
    unidentified within-cycle timing must not be inferred from this function.
    """
    import numpy as np
    from scipy.integrate import cumulative_trapezoid
    t=np.asarray(time_s,float);c=np.asarray(ca_uM,float);starts=np.asarray(cycle_starts_s,float);p=calibration
    if not isinstance(p,DyeCalibration):raise ValueError('Explicit DyeCalibration required')
    if t.ndim!=1 or len(t)<2 or c.shape!=t.shape or not np.isfinite(t).all() or np.any(np.diff(t)<=0) or not np.isfinite(c).all() or np.any(c<0):raise ValueError('Finite ordered Ca series required')
    if starts.ndim!=1 or len(starts)<1 or not np.isfinite(starts).all() or np.any(np.diff(starts)<=0):raise ValueError('Ordered acquisition cycles required')
    if not np.isfinite([exposure340_s,exposure380_s,channel_gap_s]).all() or min(exposure340_s,exposure380_s)<=0 or channel_gap_s<0:raise ValueError('Invalid exposure or gap')
    s380=starts+exposure340_s+channel_gap_s
    if starts[0]<t[0] or s380[-1]+exposure380_s>t[-1]+1e-10:raise ValueError('Exposure extends outside trace')
    occ=c/(p.kd_uM+c);i380=p.beta*(1-occ)+occ;i340=p.rmin*p.beta*(1-occ)+p.rmax*occ
    def integrate(y,s,d):
        a=cumulative_trapezoid(y,t,initial=0);return np.interp(s+d,t,a)-np.interp(s,t,a)
    q340=integrate(i340,starts,exposure340_s);q380=integrate(i380,s380,exposure380_s)
    return dict(cycle_start_s=starts,F340_integrated_au=q340,F380_integrated_au=q380,
                ratio=(q340/exposure340_s)/(q380/exposure380_s),
                provenance=p.provenance,status='explicit equilibrium optical scenario; timing and binding kinetics not calibrated')


def bursch_healthy_profile(*,external_mg_mM,external_ca_mM,age_days):
    """Explicit source gate: do not install a zero-Mg mean in every human profile."""
    if external_mg_mM!=0 or external_ca_mM!=2 or not 40<=age_days<=44:
        raise ValueError('Bursch calcium anchor supports only 0 mM Mg, 2 mM Ca, days 40-44')
    return json.loads((B/'bursch-healthy-profile.json').read_text(encoding='utf-8'))


def bursch_calcium_scenario(time_s,influx_uM_per_s,*,external_mg_mM,external_ca_mM,
                           age_days,endogenous_kappa,gamma_per_s,kinetic_provenance,
                           dye_uM=0.,dye_kd_uM=None):
    """Use the measured baseline; kinetics remain explicitly caller-provided."""
    p=bursch_healthy_profile(external_mg_mM=external_mg_mM,external_ca_mM=external_ca_mM,age_days=age_days)
    out=buffered_response(time_s,influx_uM_per_s,basal_ca_uM=p['basal_ca_uM'],
        endogenous_kappa=endogenous_kappa,gamma_per_s=gamma_per_s,dye_uM=dye_uM,
        dye_kd_uM=dye_kd_uM,provenance=kinetic_provenance)
    out['baseline_source']=p['source_doi'];out['status']='Bursch baseline anchored; influx/buffer/removal are explicit scenarios, no voltage feedback'
    return out


def compare_bursch_ampa_amplitude(delta_ca_uM,*,external_mg_mM,external_ca_mM,age_days,agonist_uM):
    p=bursch_healthy_profile(external_mg_mM=external_mg_mM,external_ca_mM=external_ca_mM,age_days=age_days)
    import numpy as np
    if agonist_uM!=100 or not np.isfinite(delta_ca_uM) or delta_ca_uM<0:raise ValueError('Nonnegative amplitude at 100 uM AMPA required')
    return dict(target_delta_uM=p['ampa_delta_ca_uM'],error_uM=float(delta_ca_uM-p['ampa_delta_ca_uM']),
        error_in_reported_sem=float((delta_ca_uM-p['ampa_delta_ca_uM'])/p['ampa_delta_sem_uM']),
        status='descriptive source-summary comparison; not an independent validation or population z-score')
