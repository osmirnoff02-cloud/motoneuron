"""[codex] 2026-10-21. Sequential spectrin cleavage and TDP epitope observation.

Adds conditional biochemical states to the existing Ca/calpain neuron traces.
Glantz 2007 constrains topology/epitopes, not human rate constants. Accessibility
is supplied explicitly: it is neither free calcium nor fitted CaM occupancy.
Hasegawa 2008 constrains the observation, not a phosphorylation kinetic model.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util
import sys
import json
import numpy as np
from numba import njit

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    '_before_sequential20261021', _HERE.parent/'2026-10-20-substrate-model-update/model_api_codex.py')
_previous = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _previous
_spec.loader.exec_module(_previous)
for _name in dir(_previous):
    if not _name.startswith('_') and _name not in globals():
        globals()[_name] = getattr(_previous, _name)

SPECTRIN_LINEAGES = {
    'alphaII_C': dict(first='C-terminal BDP150 (1177 onward)',
                     second='C-terminal BDP145 (1231 onward)',
                     probe='alpha-bdp2: GMMPR neo-N terminus; lost at second cut',
                     secondary_requires_accessibility=True),
    'betaII_N': dict(first='N-terminal BDP166',
                    second='pooled N-terminal BDP163/159',
                    probe='beta-bdp1: first-cut neo-C terminus; lost at second cut',
                    secondary_requires_accessibility=False),
}


@dataclass(frozen=True)
class SequentialSpectrinParameters:
    first_basal_per_activity_s: float
    first_accessible_per_activity_s: float
    second_per_activity_s: float
    intact_half_life_s: float
    first_half_life_s: float
    second_half_life_s: float
    provenance: str

    def vector(self, lineage):
        if lineage not in SPECTRIN_LINEAGES:
            raise ValueError('Choose an explicitly defined spectrin lineage')
        v = np.array([self.first_basal_per_activity_s, self.first_accessible_per_activity_s,
                      self.second_per_activity_s, self.intact_half_life_s,
                      self.first_half_life_s, self.second_half_life_s], float)
        if (not np.isfinite(v).all() or np.any(v[:3]<0) or np.any(v[3:]<=0)
                or not isinstance(self.provenance,str) or not self.provenance.strip()):
            raise ValueError('Explicit nonnegative rates, positive half-lives and provenance required')
        if lineage=='betaII_N' and v[0]!=0:
            raise ValueError('This CaM-sensitive betaII branch excludes the distinct BDP220/215 route')
        return np.r_[v[:3],np.log(2)/v[3:],float(lineage=='alphaII_C')]


def _series(value,t,label):
    a = np.full_like(t,float(value)) if np.ndim(value)==0 else np.asarray(value,float)
    if a.shape!=t.shape or not np.isfinite(a).all() or np.any((a<0)|(a>1)):
        raise ValueError(f'{label} must be finite [0,1], scalar or matching time')
    return a


@njit
def _rates(a,m,p):
    return a*(p[0]+m*p[1]), a*p[2]*(m if p[6]>0 else 1.)


def spectrin_equilibrium(basal_activity, basal_accessibility, *, lineage, parameters):
    """One retained lineage, U=1. Shared unblocked baseline for interventions."""
    a,m=float(basal_activity),float(basal_accessibility)
    if not np.isfinite([a,m]).all() or not 0<=a<=1 or not 0<=m<=1:
        raise ValueError('Basal activity/accessibility must be finite [0,1]')
    p=parameters.vector(lineage)
    k1,k2=_rates(a,m,p)
    f=k1/(k2+p[4]); g=k2*f/p[5]
    return dict(initial_state=np.array([1.,f,g,0.,0.,0.,0.]), synthesis_per_s=p[3]+k1)


@njit
def _rhs(y,a,m,p,synthesis):
    k1,k2=_rates(a,m,p)
    q1,q2=k1*y[0],k2*y[1]
    d0,d1,d2=p[3]*y[0],p[4]*y[1],p[5]*y[2]
    return np.array([synthesis-d0-q1,q1-d1-q2,q2-d2,d0+d1+d2,
                     synthesis,q1,q2])


@njit
def _integrate(t,a,m,mult,p,y0,synthesis,max_step):
    y=np.empty((len(t),7));y[0]=y0
    safe=min(max_step,.1/max(p[0]+p[1]+p[3],p[2]+p[4],p[5]))
    for i in range(len(t)-1):
        n=max(1,int(np.ceil((t[i+1]-t[i])/safe)))
        h=(t[i+1]-t[i])/n;v=y[i].copy()
        for j in range(n):
            # Activity is continuous; accessibility and drug commands are left-constant.
            a0=(a[i]+(a[i+1]-a[i])*j/n)*mult[i]
            am=(a[i]+(a[i+1]-a[i])*(j+.5)/n)*mult[i]
            a1=(a[i]+(a[i+1]-a[i])*(j+1)/n)*mult[i]
            k1=_rhs(v,a0,m[i],p,synthesis)
            k2=_rhs(v+h*k1/2,am,m[i],p,synthesis)
            k3=_rhs(v+h*k2/2,am,m[i],p,synthesis)
            k4=_rhs(v+h*k3,a1,m[i],p,synthesis)
            v+=h*(k1+2*k2+2*k3+k4)/6
        y[i+1]=v
    return y


def simulate_sequential_spectrin(time_s, activity, *, accessibility, lineage,
                                parameters, baseline, activity_multiplier=1., max_step_s=.01):
    """U -> first fragment -> further-cut fragment; not full molecular mass balance.

    Each cut retains one tracked end; complementary products are excluded.
    The betaII branch omits CaM-independent 220/215-kDa fragments. AlphaII_C
    does NOT stand for the N-terminal SNTF pool in the previous API.
    Accessibility is a supplied scenario, NOT computed CaM binding/dynamics.
    Put abrupt inhibitor/accessibility switches in time_s (left-constant).
    """
    t=np.asarray(time_s,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Increasing finite seconds required')
    if not isinstance(parameters,SequentialSpectrinParameters):
        raise ValueError('Explicit SequentialSpectrinParameters required')
    a=_series(activity,t,'activity');m=_series(accessibility,t,'accessibility')
    mult=_series(activity_multiplier,t,'activity multiplier')
    p=parameters.vector(lineage)
    y0=np.asarray(baseline['initial_state'],float);s=float(baseline['synthesis_per_s'])
    if (y0.shape!=(7,) or not np.isfinite(y0).all() or np.any(y0<0)
            or not np.isfinite(s) or s<0 or not np.isfinite(max_step_s) or max_step_s<=0):
        raise ValueError('Invalid baseline or integration step')
    y=_integrate(t,a,m,mult,p,y0,s,max_step_s)
    balance=y[:,:4].sum(axis=1)-y[:,4]
    error=float(np.max(abs(balance-balance[0])))
    if not np.isfinite(y).all() or y.min() < -1e-10 or error>1e-8*max(1.,np.max(y[:,:4].sum(axis=1))):
        raise ArithmeticError('Invalid sequential trajectory or lineage balance')
    return dict(time_s=t,intact=y[:,0],first_fragment=y[:,1],second_fragment=y[:,2],
                removed=y[:,3],synthesized=y[:,4],first_cut_events=y[:,5],
                second_cut_events=y[:,6],first_cut_neoepitope_amount=y[:,1].copy(),
                retained_fragment_total=y[:,1]+y[:,2],lineage=lineage,
                definition=SPECTRIN_LINEAGES[lineage],parameters=asdict(parameters),
                lineage_balance_error=error,human_kinetic_fit=False,
                predicts_viability=False,absolute_assay_signal=None)


def coupled_trace_to_stateful_reporters(trace, *, parameters, compartment, basal_ca_uM,
                                      spectrin_parameters, accessibility,
                                      basal_accessibility, inhibition=0., max_step_s=.01):
    """Latest neuron -> Ca -> calpain/TDP plus separately tracked spectrin states."""
    if (not spectrin_parameters or set(spectrin_parameters)-set(SPECTRIN_LINEAGES)
            or set(accessibility)!=set(spectrin_parameters)
            or set(basal_accessibility)!=set(spectrin_parameters)):
        raise ValueError('Explicit matching parameter and accessibility maps required')
    tdp=_previous.coupled_trace_to_tdp(trace,parameters=parameters,compartment=compartment,
                                     basal_ca_uM=basal_ca_uM,inhibition=inhibition,max_step_s=max_step_s)
    initial=_previous.baseline_equilibrium(basal_ca_uM,parameters)
    basal=initial['initial_state'][0]/(1+parameters.cast_over_ki)
    out={}
    for name,p in spectrin_parameters.items():
        out[name]=simulate_sequential_spectrin(
            tdp['time_s'],tdp['active_fraction']/(1+parameters.cast_over_ki),
            accessibility=accessibility[name],lineage=name,parameters=p,
            baseline=spectrin_equilibrium(basal,basal_accessibility[name],lineage=name,parameters=p),
            activity_multiplier=1-tdp['inhibition'],max_step_s=max_step_s)
    return dict(TDP=tdp,sequential_spectrin=out,
                pS409_410_signal=None,
                pTDP_missing='Phosphorylation, extraction and complementary C-terminal species are not determined',
                nuclear_TDP_function=None,neuronal_viability=None,NMJ_function=None,
                biochemical_feedback_to_voltage=False,human_kinetic_fit=False)


def observe_tdp_species(amount, *, residue_start, residue_end, epitope,
                        phosphorylated_fraction, recovered_fraction, gain, provenance):
    """Contribution from ONE declared TDP species, never whole-assay prediction.

    Hasegawa: pS409/410 is phosphorylation dependent; total C-terminal 405-414
    is a different antibody. Explicit recovery represents the assayed fraction,
    not an inferred solubility. Gain is hypothetical unless calibrated externally.
    No causal enzyme is inferred from a band. A missing epitope cannot fluoresce.
    """
    epitopes={'pS409/410':(409,410),'total_C405-414':(405,414)}
    if epitope not in epitopes:
        raise ValueError('Use a defined sequence-specific epitope')
    if (isinstance(residue_start,bool) or isinstance(residue_end,bool)
            or int(residue_start)!=residue_start or int(residue_end)!=residue_end
            or not 1<=residue_start<=residue_end<=414):
        raise ValueError('A declared human TDP-43 residue interval is required')
    x=np.asarray(amount,float)
    if not np.isfinite(x).all() or np.any(x<0) or not isinstance(provenance,str) or not provenance.strip():
        raise ValueError('Finite nonnegative amount and provenance required')
    recovery=np.asarray(recovered_fraction,float)
    if not np.isfinite(recovery).all() or np.any((recovery<0)|(recovery>1)):
        raise ValueError('Explicit assayed fraction in [0,1] required')
    gain=float(gain)
    if not np.isfinite(gain) or gain<0: raise ValueError('Finite nonnegative gain required')
    lo,hi=epitopes[epitope];present=residue_start<=lo and residue_end>=hi
    if epitope=='pS409/410':
        if present and phosphorylated_fraction is None:
            raise ValueError('Phosphorylated fraction is unknown; cannot predict pTDP from total TDP')
        frac=np.asarray(0. if phosphorylated_fraction is None else phosphorylated_fraction,float)
        if not np.isfinite(frac).all() or np.any((frac<0)|(frac>1)) or (not present and np.any(frac!=0)):
            raise ValueError('Invalid site phosphorylation or absent phosphorylatable residues')
    else:
        if phosphorylated_fraction is not None:
            raise ValueError('Total antibody is phosphorylation independent; pass None')
        frac=1.
    signal=x*recovery*gain*frac*float(present)
    if signal.shape!=x.shape: raise ValueError('Observation factors must broadcast to the amount shape')
    return dict(signal_contribution=signal,epitope_present=present,epitope=epitope,
                whole_assay_prediction=False,calpain_specific=False,
                neuronal_damage=None,provenance=provenance)


def stateful_source_evidence():
    return json.loads((_HERE/'evidence_codex.json').read_text(encoding='utf-8'))


def source_training_data():
    result=_previous.source_training_data()
    result['sequential_spectrin_and_TDP_epitopes']=stateful_source_evidence()
    return result
