"""[codex] 2026-10-28: three stages, optional Ca-adaptive endocytic capacity.

Rat hippocampal reference constraints are separate from human assumptions.
Acidification is a lumped first-order state, not a proton/vesicular pH model.
Additional priming time, energetic fractions, Ca scale and memory are unknown.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from recycling_energy_codex import isolated_recycling_rhs

@dataclass(frozen=True)
class AdaptiveRecyclingParameters:
    enabled: bool=False
    calcium_capacity_enabled: bool=False
    acidification_tau_s: float=2.3
    priming_tau_s: float=10.
    acidification_cost_fraction: float=.5
    capacity_max_multiplier: float=2.
    capacity_half_Ca_excess_uM: float=.5
    capacity_memory_s: float=10.
    provenance: str='Balaji2008 rat CNS: retrieval is not acidification/readiness; Ca changes capacity, not single-vesicle tau. Human times/cost/Ca scale/memory unknown. 2.3s is a 36C data-not-shown source reference.'
    def vector(self):
        a=np.array([float(self.enabled),float(self.calcium_capacity_enabled),self.acidification_tau_s,self.priming_tau_s,self.acidification_cost_fraction,self.capacity_max_multiplier,self.capacity_half_Ca_excess_uM,self.capacity_memory_s])
        if not isinstance(self.enabled,bool) or not isinstance(self.calcium_capacity_enabled,bool) or not np.isfinite(a).all() or min(a[2],a[3],a[6],a[7])<=0 or not 0<=a[4]<=1 or a[5]<1 or not self.provenance:raise ValueError('Invalid adaptive recycling parameters')
        if self.calcium_capacity_enabled and not self.enabled:raise ValueError('Capacity coupling requires explicit staged recycling')
        return a

ADAPTIVE_RANGES=dict(acidification_tau_s=[2.3,4.,10.],priming_tau_s=[1.,10.,60.],acidification_cost_fraction=[0.,.5,1.],capacity_max_multiplier=[1.,2.,4.],capacity_half_Ca_excess_uM=[.1,.5,2.],capacity_memory_s=[1.,10.,60.])
ADAPTIVE_NAMES=('terminal_acidified_unready_quanta','terminal_capacity_Ca_memory_uM','terminal_acidified_quanta','terminal_acidification_ATP_molecules','terminal_priming_ATP_molecules')

@njit
def capacity(memory,rc,nv,ad):
    factor=1.
    if ad[1]:factor+=(ad[5]-1)*max(memory,0.)/(ad[6]+max(memory,0.))
    return min(nv[5],rc[2]*factor)

@njit
def staged_local(z,old,tv,nv,en,rc,ad):
    """33 terminal states; replace the two-stage derivative, no extra ATP cost."""
    out=np.zeros(33);out[:28]=old
    if not ad[0] or not tv[0]:return out
    A=max(z[7],0.);fn=(A/(nv[7]+A))/(en[0]/(nv[7]+en[0]))
    ce=capacity(z[29],rc,nv,ad)
    endo=min(max(z[21],0.),ce)/rc[1]*fn
    acid=max(z[22],0.)/ad[2]*fn
    prim=max(z[28],0.)/ad[3]*fn
    cost_e=endo*nv[15]*rc[4]
    cost_a=acid*nv[15]*(1-rc[4])*ad[4]
    cost_p=prim*nv[15]*(1-rc[4])*(1-ad[4])
    delta=(cost_e+cost_a+cost_p-old[6])/(602.214076*tv[1])
    out[1]=prim/nv[5];out[5]=prim;out[6]=cost_e+cost_a+cost_p
    out[7]-=delta;out[8]+=delta
    out[21]=-endo;out[22]=endo-acid;out[23]=endo
    out[24]=cost_e;out[25]=cost_a+cost_p
    out[28]=acid-prim
    out[29]=(max(z[0],0.)-z[29])/ad[7] if ad[1] else 0.
    out[30]=acid;out[31]=cost_a;out[32]=cost_p
    return out

@njit
def adaptive_extension(z,old,tv,nv,en,rc,ad):
    out=np.zeros(308);out[:275]=old[:275]
    out[275:]=staged_local(z[275:],old[275:303],tv,nv,en,rc,ad)
    return out

@njit
def isolated_adaptive_rhs(z,tv,nv,en,av,rc,paths,support,gs,os,ad):
    old=isolated_recycling_rhs(z[:28],tv,nv,en,av,rc,paths,support,gs,os)
    return staged_local(z,old,tv,nv,en,rc,ad)

def phluorin_signal(surface,alkaline,*,surface_visible=True):
    """Baseline-subtracted reporter equivalents F=S+A. Acid quench: F=A.
    Not absolute intensity, EPP, ACh content or calibrated vesicle pH.
    """
    s,a=np.broadcast_arrays(np.asarray(surface,float),np.asarray(alkaline,float))
    if not np.isfinite(s+a).all() or min(s.min(),a.min()) < -1e-6:raise ValueError('Invalid reporter pools')
    return (s if surface_visible else 0.)+a

def poststimulus_fluorescence(t,tau_end_s,tau_acid_s):
    """Balaji Appendix Eq7, unit surface pulse, nonsaturated, constant ATP."""
    t=np.asarray(t,float)
    if np.any(t<0) or min(tau_end_s,tau_acid_s)<=0:raise ValueError('Invalid times')
    if abs(tau_end_s-tau_acid_s)<1e-10:return (1+t/tau_end_s)*np.exp(-t/tau_end_s)
    return (tau_end_s*np.exp(-t/tau_end_s)-tau_acid_s*np.exp(-t/tau_acid_s))/(tau_end_s-tau_acid_s)

def takeuchi_assay_gate(extension_aa,linker_aa,*,proteasome_active=True,ubiquitin_activation=True):
    """Qualitative fixture for Rpn10-GFP constructs in S1/S3, NOT TDP law.
    Ubiquitin-independent direct Rpn10 docking; cannot infer neuronal bypass.
    """
    if linker_aa not in (8,21) or extension_aa not in (0,15,22,27,32,37):raise ValueError('Outside tested construct classes')
    return bool(proteasome_active and extension_aa>=22)
