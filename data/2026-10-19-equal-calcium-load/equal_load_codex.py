"""[codex] 2026-10-19: equal entry dose and assay-aware observation layers.

No molecular AMPA pore, membrane feedback or new human parameter fit.
Source-specific apparent decay and enzyme-driven lysate assays stay separate.
"""
from pathlib import Path
import sys,math
from dataclasses import dataclass,asdict
import numpy as np
from scipy.linalg import expm
from scipy.integrate import cumulative_trapezoid

B=Path(__file__).resolve().parent
sys.path.append(str(B.parent/'2026-10-19-calpain-tdp-scenarios'))
from calpain_tdp_codex import CalpainTDPParameters,baseline_equilibrium,simulate_calpain_tdp
sys.path.append(str(B.parent/'2026-10-19-calcium-buffer-calibration'))
from calcium_calibration_codex import buffered_response

FARADAY_C_PER_MOL=96485.33212
KINETICS='Hypothetical human-compatible scenario; Bursch baseline only. Not calibrated AMPA, calpain or TDP kinetics.'

def _time(t):
    t=np.asarray(t,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Increasing finite time in seconds required')
    return t

def equal_entry_protocol(kind,peak_factor=1.,*,dose_uM=91.8,onset_s=60.):
    """Fixed volume: dose_uM is total delivered concentration before buffering.

    Short/long/repeated templates have 3/60/(6*3) s total ON at factor1.
    Peak factor changes amplitude and inverse pulse duration, NOT delivered dose.
    Repeated onsets remain 30s apart; overlap is rejected.
    """
    if not np.isfinite([peak_factor,dose_uM,onset_s]).all() or min(peak_factor,dose_uM)<=0 or onset_s<0:
        raise ValueError('Positive finite peak/dose, nonnegative onset required')
    if kind not in ('short','long','repeated'):raise ValueError('Unknown protocol')
    n=6 if kind=='repeated' else 1
    width=(60. if kind=='long' else 3.)/peak_factor
    if kind=='repeated' and width>30.:raise ValueError('Repeated pulses would overlap')
    windows=[(onset_s+30*k,onset_s+30*k+width) for k in range(n)]
    rate=dose_uM/(n*width)
    return dict(kind=kind,peak_factor=float(peak_factor),dose_uM=float(dose_uM),
                windows_s=windows,peak_influx_uM_per_s=rate,total_on_s=n*width,
                last_offset_s=windows[-1][1],source='prescribed nonnegative Ca entry; not AMPA pore kinetics')

def calcium_equal_entry(kind,peak_factor=1.,removal_factor=1.,*,dose_uM=91.8,
                        fine_dt_s=.1,end_s=21600.):
    if not np.isfinite([removal_factor,fine_dt_s,end_s]).all() or min(removal_factor,fine_dt_s)<=0:
        raise ValueError('Finite positive removal factor and sampling step required')
    p=equal_entry_protocol(kind,peak_factor,dose_uM=dose_uM)
    if end_s<900 or p['last_offset_s']>=900:raise ValueError('Protocol must finish before 900s; end >=900s')
    edges=np.array(p['windows_s']).ravel()
    t=np.unique(np.r_[np.arange(0.,900.,fine_dt_s),np.arange(900.,end_s,10.),end_s,edges])
    j=np.zeros_like(t)
    for lo,hi in p['windows_s']:j[(t>=lo)&(t<hi)]=p['peak_influx_uM_per_s']
    gamma=5.1*removal_factor;capacity=51.
    r=buffered_response(t,j,basal_ca_uM=.0804,endogenous_kappa=50.,gamma_per_s=gamma,provenance=KINETICS)
    dt=np.diff(t);x=r['ca_uM']-.0804;tau=capacity/gamma
    # Exact interval integrals of the exponential solution; do not trapezoid step input.
    area=float(np.sum(j[:-1]*dt/gamma+(x[:-1]-j[:-1]/gamma)*tau*(-np.expm1(-dt/tau))))
    delivered=float(np.dot(j[:-1],dt))
    balance=delivered-gamma*area-capacity*(x[-1]-x[0])
    r.update(influx_uM_per_s=j,protocol=p,delivered_uM=delivered,
             free_ca_excess_auc_uM_s=area,gamma_per_s=gamma,removal_factor=removal_factor,
             calcium_balance_error_uM=float(balance),remaining_excess_uM=float(x[-1]))
    return r

def inward_current_to_ca_flux(current_pA,*,calcium_current_fraction,volume_um3,
                              receptor_state,provenance):
    """Faraday conversion only, inward current positive, fixed well-mixed volume.

    Fraction means calcium fraction of inward electrical CHARGE, not P_Ca/P_Na.
    It is caller-supplied at the recording voltage/solutions. Q/R alone does not
    determine it. This adapter neither creates nor fits an AMPA conductance.
    """
    states=('AMPA_GluA2_Q_unedited','AMPA_GluA2_lacking','AMPA_GluA2_R_edited','other_measured_Ca_source')
    a=np.asarray(current_pA,float)
    if not np.isfinite(a).all() or np.any(a<0):raise ValueError('Inward current must be nonnegative pA')
    if (not np.isfinite([calcium_current_fraction,volume_um3]).all() or
        not 0<=calcium_current_fraction<=1 or volume_um3<=0 or
        receptor_state not in states or not provenance.strip()):raise ValueError('Explicit valid current fraction, volume, source state and provenance required')
    j=a*calcium_current_fraction*1e9/(2*FARADAY_C_PER_MOL*volume_um3)
    return dict(influx_uM_per_s=j,calcium_current_pA=a*calcium_current_fraction,
                source_state=receptor_state,volume_um3=volume_um3,provenance=provenance,
                pore_model=False,electrical_feedback=False)

def observe_cleavage_reporter(time_s,effective_activity,*,cleavage_per_s,
                              nonspecific_rate_per_s=0.,provenance):
    """Irreversible substrate cleavage, NOT instantaneous enzyme activity/FRET ratio.

    Polster2022 cA-TAT and Gerencser2009 require controls for specificity/optics.
    No reporter replenishment/clearance or detector calibration is assumed here.
    """
    t=_time(time_s);e=np.asarray(effective_activity,float)
    bg=np.broadcast_to(np.asarray(nonspecific_rate_per_s,float),t.shape)
    if e.shape!=t.shape or not np.isfinite(e).all() or np.any(e<0) or not np.isfinite(bg).all() or np.any(bg<0):raise ValueError('Nonnegative matching activity/background rate required')
    if not np.isfinite(cleavage_per_s) or cleavage_per_s<0 or not provenance.strip():raise ValueError('Nonnegative finite reporter rate and provenance required')
    H=cumulative_trapezoid(cleavage_per_s*e+bg,t,initial=0.)
    uncut=np.exp(-H)
    return dict(time_s=t,uncut_fraction=uncut,cleaved_fraction=-np.expm1(-H),
                cumulative_hazard=H,provenance=provenance,fluorescence_ratio=None,
                status='Ideal substrate observation. Accumulated product is not enzyme activity.')

def observe_pulse_chase(time_s,effective_activity,*,parameters,substrate,provenance,
                        photobleach_per_s=0.):
    """Track molecules labelled at t0; new synthesis is unlabelled, excluded.

    Barmada's population and single-cell apparent half-lives are separate assays.
    Pre-existing large product can disappear by secondary cleavage OR removal.
    """
    t=_time(time_s);e=np.asarray(effective_activity,float);parameters.vector()
    if e.shape!=t.shape or not np.isfinite(e).all() or np.any(e<0):raise ValueError('Nonnegative paired activity required')
    if not np.isfinite(photobleach_per_s) or photobleach_per_s<0 or not provenance.strip():raise ValueError('Valid bleach rate and provenance required')
    if substrate=='intact_TDP':h=parameters.intact_half_life_s;k=parameters.cleavage1_per_s
    elif substrate=='large_N_fragment':h=parameters.large_half_life_s;k=parameters.cleavage2_per_s
    else:raise ValueError('Specify intact_TDP or large_N_fragment, not generic TDP damage')
    decay=cumulative_trapezoid(math.log(2)/h+k*e,t,initial=0.)
    remaining=np.exp(-decay)
    return dict(time_s=t,labelled_substrate_fraction=remaining,
                signal_fraction=remaining*np.exp(-photobleach_per_s*(t-t[0])),
                substrate=substrate,provenance=provenance,
                newly_synthesized_label=False,status='Label cohort, not total steady-state protein')

def enzyme_driven_chase(time_s,relative_enzyme,*,k1_per_s,k2_per_s,initial=(1.,0.,0.),provenance):
    """P->large N->smaller N, exact linear lysate-assay propagation.

    Explicit relative enzyme input can change at knots (S13 adds enzyme at15min).
    Rates per relative input are supplied assumptions, not inferred cellular rates.
    No synthesis/removal in this short assay. Both are essential in the cell model.
    """
    t=_time(time_s);e=np.asarray(relative_enzyme,float);y=np.asarray(initial,float)
    if (e.shape!=t.shape or not np.isfinite(e).all() or np.any(e<0) or y.shape!=(3,) or
        not np.isfinite(y).all() or np.any(y<0) or not np.isfinite([k1_per_s,k2_per_s]).all() or
        min(k1_per_s,k2_per_s)<0 or not provenance.strip()):raise ValueError('Valid enzyme schedule, nonnegative rates/state and provenance required')
    ys=[y.copy()]
    for i,dt in enumerate(np.diff(t)):
        a,b=k1_per_s*e[i],k2_per_s*e[i]
        M=np.array([[-a,0.,0.],[a,-b,0.],[0.,b,0.]])
        y=expm(M*dt)@y;ys.append(y)
    return dict(time_s=t,relative_enzyme=e,states=np.array(ys),provenance=provenance,
                status='Assay-specific N-lineage structural scenario, no human kinetic fit')

def simulate_equal_load(kind,peak_factor=1.,removal_factor=1.,*,parameters,
                         fine_dt_s=.1,reporter_cleavage_per_s=.001):
    ca=calcium_equal_entry(kind,peak_factor,removal_factor,fine_dt_s=fine_dt_s)
    base=baseline_equilibrium(.0804,parameters)
    bio=simulate_calpain_tdp(ca['t_s'],ca['ca_uM'],parameters=parameters,
        ca_provenance=KINETICS,compartment='hypothetical fixed-volume well-mixed compartment',baseline=base)
    obs=observe_cleavage_reporter(bio['time_s'],bio['effective_activity'],
        cleavage_per_s=reporter_cleavage_per_s,provenance='Hypothetical reporter kinetics; optical calibration absent')
    return dict(calcium=ca,biochemistry=bio,reporter=obs,
                status='Equal entry dose, hypothetical kinetics; no inferred AMPA pore or survival')
