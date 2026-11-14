"""[codex] 2026-10-21. Modular, conditional TDP -> STMN2 -> regrowth support.

Chan 2026 supplies experimental endpoints, not a calibrated human kinetic chain.
The axonal endpoint observer and slow dynamics have deliberately separate scopes.
Inherited electrical, calcium, cleavage and localization APIs are unchanged.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util
import sys
import json
import numpy as np
from scipy.integrate import solve_ivp

_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_before_stmn2_20261021',_HERE.parent/'2026-10-21-tdp-transport-clearance/model_api.py')
_previous=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=_previous
_spec.loader.exec_module(_previous)
for _name in dir(_previous):
    if not _name.startswith('_') and _name not in globals(): globals()[_name]=getattr(_previous,_name)


def _source(s):
    if not isinstance(s,str) or not s.strip(): raise ValueError('Explicit provenance or assumption required')


def _positive(x,name):
    if not np.isfinite(x) or x<=0: raise ValueError(name+' must be finite and positive')


def _array(x,t,name,upper=None):
    a=np.full(t.shape,float(x)) if np.ndim(x)==0 else np.asarray(x,float)
    if a.shape!=t.shape or not np.isfinite(a).all() or np.any(a<0) or (upper is not None and np.any(a>upper)):
        raise ValueError(name+' outside admissible range or wrong shape')
    return a


def normalized_hill(x,half,hill):
    """Positive saturating transfer, normalized to f(1)=1; no CE fraction implied."""
    _positive(half,'half'); _positive(hill,'hill')
    a=np.asarray(x,float)
    if not np.isfinite(a).all() or np.any(a<0): raise ValueError('Nonnegative finite input required')
    z=a**hill; k=half**hill
    return z*(1+k)/(z+k)


@dataclass(frozen=True)
class STMN2Parameters:
    tdp_half_relative: float
    tdp_hill: float
    mrna_tau_s: float
    somatodendritic_protein_half_life_s: float
    axonal_availability_tau_s: float
    regrowth_response_tau_s: float
    axonal_gain: float
    regrowth_half_relative: float
    regrowth_hill: float
    provenance: str

    def __post_init__(self):
        _source(self.provenance)
        for k,v in asdict(self).items():
            if k!='provenance': _positive(v,k)


def chan_stmn2_evidence():
    return json.loads((_HERE/'calibration.json').read_text(encoding='utf-8'))


def chan_regeneration_endpoint(axonal_stmn2_relative, *, half_relative, hill,
                               provenance, assay='Chan_Fig8_DIV14_injury_DIV17_readout'):
    """Empirical group-level Fig8 mapping in mm/mm2, NOT a growth speed.

    WT and DeltaNLS controls constrain shape; ASO10 is a held-out comparison.
    Input must be the same axonal spots/20um readout relative to WT. It is NOT
    somatic staining, STMN2 mRNA, mass, or the model's nuclear TDP fraction.
    No row pairing between 8b/8c or generalization to other protocols is assumed.
    """
    _source(provenance)
    if assay!='Chan_Fig8_DIV14_injury_DIV17_readout':
        raise ValueError('No calibrated endpoint observer for this assay')
    ref=chan_stmn2_evidence()['axonal_endpoint']
    relative=normalized_hill(axonal_stmn2_relative,half_relative,hill)
    return dict(relative_to_WT=relative,
                neurite_length_per_cell_body_area_mm_per_mm2=ref['WT_mean_mm_per_mm2']*relative,
                observer_scope=assay,provenance=provenance,
                human_kinetic_fit=False,individual_cell_fit=False,
                uncertainty='Unknown response shape; source SD is not a donor confidence interval')


def simulate_stmn2_regrowth(time_s,nuclear_tdp_relative,*,functional_fraction,
        splice_rescue_fraction,stmn2_rna_retention,parameters,initial_state,
        drive_provenance,rtol=1e-8,atol=1e-10):
    """Four normalized readouts, driven by explicitly assumed functional nuclear TDP.

    M'= (target-M)/tauM; B'=ln2*(M-B)/HLbody;
    A'= (gain*M-A)/tauA; R'=(Hill(A)-R)/tauR.
    target = retention * [Hill(N*functional_fraction)+rescue*(1-Hill(...))].
    Rescue raises deficient splicing to WT (no reduction when above WT).

    B is a soma/dendrite protein proxy; A an independently lagged axonal
    availability proxy. A is not transported B and these are not conserved mass
    compartments. R is a dimensionless support index, NOT length, survival or NMJ.
    No human time constants were measured by this fit. No feedback to Vm.
    Applying the 72h RNA endpoint transfer as an equilibrium target is an extra
    assumption; fitting those endpoints does not validate this temporal response.
    Inputs are left-constant; all changes must appear as time grid points.
    Initial state [M,B,A,R] must be provided. No seconds-to-days extrapolation.
    """
    _source(drive_provenance)
    if not isinstance(parameters,STMN2Parameters): raise TypeError('STMN2Parameters required')
    t=np.asarray(time_s,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Strictly increasing finite time in seconds required')
    n=_array(nuclear_tdp_relative,t,'Relative nuclear TDP')
    f=_array(functional_fraction,t,'Functional fraction',1.)
    s=_array(splice_rescue_fraction,t,'Splice rescue',1.)
    retain=_array(stmn2_rna_retention,t,'RNA retention',1.)
    y=np.asarray(initial_state,float)
    if y.shape!=(4,) or not np.isfinite(y).all() or np.any(y<0): raise ValueError('Four nonnegative initial readouts required')
    p=parameters
    basal=normalized_hill(n*f,p.tdp_half_relative,p.tdp_hill)
    target=retain*(basal+s*np.maximum(1-basal,0))
    out=np.empty((len(t),4)); out[0]=y
    # Combine intervals with the same input target; never interpolate an intervention backwards.
    edges=np.r_[0,np.flatnonzero(np.diff(target[:-1])!=0)+1,len(t)-1]
    for a,b in zip(edges[:-1],edges[1:]):
        q=float(target[a])
        def rhs(_,v):
            m,body,axon,reg=v
            return [(q-m)/p.mrna_tau_s,np.log(2)*(m-body)/p.somatodendritic_protein_half_life_s,
                    (p.axonal_gain*m-axon)/p.axonal_availability_tau_s,
                    (float(normalized_hill(max(axon,0),p.regrowth_half_relative,p.regrowth_hill))-reg)/p.regrowth_response_tau_s]
        sol=solve_ivp(rhs,(t[a],t[b]),out[a],t_eval=t[a+1:b+1],method='DOP853',rtol=rtol,atol=atol)
        if not sol.success: raise RuntimeError(sol.message)
        out[a+1:b+1]=sol.y.T
    if np.min(out)<-1e-8: raise ArithmeticError('Numerical positivity failure')
    return dict(time_s=t.copy(),functional_nuclear_tdp=n*f,full_length_mrna=out[:,0],
        somatodendritic_protein_proxy=out[:,1],axonal_stmn2_proxy=out[:,2],regrowth_support_index=out[:,3],
        full_length_mrna_target=target,parameters=asdict(p),drive_provenance=drive_provenance,
        human_kinetic_fit=False,feedback_to_electrical_model=False,
        cryptic_exon_fraction=None,axon_length_um=None,NMJ_function=None,
        limitation='Conditional dimensionless dynamics; protein readouts not mass compartments; not a patient prediction')


def localized_tdp_to_stmn2(localization,*,healthy_nuclear_reference,
        functional_fraction,splice_rescue_fraction,stmn2_rna_retention,
        parameters,initial_state,drive_provenance):
    """One-way adapter on the original time axis. Does not infer RNA activity from total TDP."""
    _positive(healthy_nuclear_reference,'Healthy nuclear reference')
    return simulate_stmn2_regrowth(localization['time_s'],
        np.asarray(localization['nuclear_intact'])/healthy_nuclear_reference,
        functional_fraction=functional_fraction,splice_rescue_fraction=splice_rescue_fraction,
        stmn2_rna_retention=stmn2_rna_retention,parameters=parameters,
        initial_state=initial_state,drive_provenance=drive_provenance)


def source_training_data():
    out=_previous.source_training_data()
    out['Chan_STMN2_regrowth']=chan_stmn2_evidence()
    return out
