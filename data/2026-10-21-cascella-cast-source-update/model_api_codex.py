"""[codex] 2026-10-21. Cascella clearance and Kamei substrate susceptibility.

Optional modules over the existing neuron. These animal/cell-line experiments
do not calibrate human neuronal clearance, calcium or p35 rates.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util, sys, json
import numpy as np
from scipy.integrate import solve_ivp

_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_before_cascella20261021',_HERE.parent/'2026-10-21-protease-memory-clearance/model_api_codex.py')
_previous=importlib.util.module_from_spec(_spec); sys.modules[_spec.name]=_previous
_spec.loader.exec_module(_previous)
for _name in dir(_previous):
    if not _name.startswith('_') and _name not in globals(): globals()[_name]=getattr(_previous,_name)


def _provenance(value):
    if not isinstance(value,str) or not value.strip(): raise ValueError('Explicit source or assumption required')


def _times(time_s):
    t=np.asarray(time_s,dtype=float)
    if t.ndim!=1 or len(t)<1 or not np.isfinite(t).all() or t[0]<0 or np.any(np.diff(t)<=0):
        raise ValueError('Nonnegative, finite, strictly increasing seconds required')
    return t


def source_update_evidence():
    return json.loads((_HERE/'source-parameters_codex.json').read_text(encoding='utf-8'))


def source_training_data():
    out=_previous.source_training_data()
    out['protease_processing_and_assay_domains']=protease_memory_evidence()
    out['cascella_kamei_source_update']=source_update_evidence()
    return out


def protease_memory_evidence():
    out=_previous.protease_memory_evidence()
    for name,current in [('Cascella2017_provisional','Cascella2017'),('Ladewig2003','Ladewig2003')]:
        out[name]['historical_access_status_only']=True
        out[name]['superseded_by_current_source_record']=current
        out[name]['current_record']=source_update_evidence()[current]
    return out


def cascella_reference(time_s,*,condition,system):
    """Published fitted fluorescence, not reconstructed raw cell observations.

    A and plateau remain independent as reported; F(0) need not equal 100.
    No extrapolation beyond the measured 0--24 h window is allowed here.
    """
    if system!='NSC34_exogenous_TDP43_inclusion_bodies':
        raise ValueError('Cascella numeric fits apply to the specified NSC34 assay only')
    t=_times(time_s)
    if t[-1]>86400: raise ValueError('Source reference restricted to 0--24 h')
    source=source_update_evidence()['Cascella2017']
    p=source['fits'][condition]
    f=p['plateau_percent']+p['amplitude_percent']*np.exp(-p['k_per_s']*t)
    return dict(time_s=t,fluorescence_percent=f,fit=p,
                status='published_fit_reconstruction_not_raw_data',system=system)


@dataclass(frozen=True)
class AggregateRouteParameters:
    ups_per_s: float
    alp_per_s: float
    resistant_per_s: float
    ups_remaining_activity: float
    alp_remaining_activity: float
    provenance: str


def aggregate_route_chase(time_s,*,initial_pools,parameters):
    """Three operational aggregate classes with explicit mass accounting.

    Pools: UPS-susceptible, ALP-susceptible, resistant over the observation
    window. No synthesis/input or interconversion. Rates are phenomenological;
    UPS may include disassembly. An assumption must be supplied for the split.
    Resistant clearance above zero is an uncalibrated sensitivity scenario.
    """
    t=_times(time_s); p=parameters; _provenance(p.provenance)
    y0=np.asarray(initial_pools,dtype=float)
    v=np.array([p.ups_per_s,p.alp_per_s,p.resistant_per_s,p.ups_remaining_activity,p.alp_remaining_activity])
    if y0.shape!=(3,) or not np.isfinite(y0).all() or np.any(y0<0): raise ValueError('Three nonnegative finite pools required')
    if not np.isfinite(v).all() or np.any(v<0) or np.any(v[3:]>1): raise ValueError('Nonnegative rates; residual activities in [0,1] required')
    rates=np.array([v[0]*v[3],v[1]*v[4],v[2]])
    pools=y0[None,:]*np.exp(-t[:,None]*rates[None,:])
    removed=y0[None,:]-pools
    return dict(time_s=t,pools=pools,remaining=pools.sum(axis=1),removed_by_route=removed,
                removed=removed.sum(axis=1),parameters=asdict(p),initial_pools=y0,
                status='conditional_clearance_not_human_calibration')


@dataclass(frozen=True)
class P35SusceptibilityParameters:
    cleavage_per_s: float
    thr138_phosphorylation_per_s: float
    thr138_dephosphorylation_per_s: float
    phosphorylated_relative_cleavage: float
    p35_clearance_per_s: float
    p25_clearance_per_s: float
    synthesis_per_s: float
    provenance: str


def simulate_p35_susceptibility(time_s,effective_calpain_activity,*,parameters,
                                initial_pools,activity_provenance,thr138_available=True):
    """Conditional p35/pThr138-p35/p25 lineage, driven by the enzyme API.

    Kamei supports the direction of the susceptibility effect, not these rates.
    Ser8, membrane binding, Cdk5 feedback, p10 and Cdk5 toxicity are unresolved.
    Units are p35-lineage equivalents, not total peptide mass; a cleavage event
    keeps one tracked C-terminal lineage. p25 phosphorylation is not resolved.
    """
    t=_times(time_s); a=np.asarray(effective_calpain_activity,float); p=parameters
    _provenance(activity_provenance); _provenance(p.provenance)
    if len(t)<2 or a.shape!=t.shape or not np.isfinite(a).all() or np.any(a<0): raise ValueError('Matching nonnegative activity series required')
    y0=np.asarray(initial_pools,float)
    if y0.shape!=(3,) or not np.isfinite(y0).all() or np.any(y0<0): raise ValueError('Initial p35, pThr138-p35, p25 required')
    v=np.array([x for k,x in asdict(p).items() if k!='provenance'],float)
    if not np.isfinite(v).all() or np.any(v<0) or p.phosphorylated_relative_cleavage>1: raise ValueError('Nonnegative rates and relative susceptibility in [0,1] required')
    if not thr138_available and y0[1]!=0: raise ValueError('T138A cannot start with phosphorylated Thr138')
    phosphorylation=p.thr138_phosphorylation_per_s if thr138_available else 0.
    def rhs(s,y):
        ac=float(np.interp(s,t,a))
        exchange=phosphorylation*y[0]-p.thr138_dephosphorylation_per_s*y[1]
        cu=p.cleavage_per_s*ac*y[0]
        cp=p.cleavage_per_s*ac*p.phosphorylated_relative_cleavage*y[1]
        d35=p.p35_clearance_per_s*(y[0]+y[1]); d25=p.p25_clearance_per_s*y[2]
        return [p.synthesis_per_s-exchange-cu-p.p35_clearance_per_s*y[0],
                exchange-cp-p.p35_clearance_per_s*y[1],cu+cp-d25,d35+d25,p.synthesis_per_s,cu+cp]
    # Limiting max_step prevents skipping short pulses between observed samples.
    sol=solve_ivp(rhs,(t[0],t[-1]),np.r_[y0,0.,0.,0.],t_eval=t,rtol=1e-9,atol=1e-11,max_step=float(np.min(np.diff(t))))
    if not sol.success: raise RuntimeError(sol.message)
    y=sol.y.T
    if np.min(y)<-1e-8: raise RuntimeError('Nonphysical integration result')
    balance=y[:,:3].sum(axis=1)+y[:,3]-y[:,4]-y0.sum()
    return dict(time_s=t,p35=y[:,0],phospho_p35=y[:,1],p25=y[:,2],removed=y[:,3],synthesized=y[:,4],
                cleavage_events=y[:,5],lineage_balance_error=balance,parameters=asdict(p),
                activity_provenance=activity_provenance,thr138_available=thr138_available,
                status='qualitative_substrate_extension_unknown_human_rates')
