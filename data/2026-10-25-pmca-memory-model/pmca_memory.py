"""Codex 2026-10-25. Conditional PMCA2 memory, not a human MN calibration.

Activation anchor: Caride2001 Fig3 (human PMCA2b in Sf9 membranes).
Forgetting anchor: Fig5D (0.021/s at 50 nM). The one-state reduction is OUR
phenomenology, not the paper's three-state kinetic scheme. Total CaM fixed.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.integrate import solve_ivp

B=Path(__file__).resolve().parent

@dataclass(frozen=True)
class PMCAMemoryParameters:
    memory_gain: float=0.
    forget_at_50nM_per_s: float=.021
    activation_max_per_M_s: float=5.04e5
    free_CaM_uM: float=.117
    activation_half_uM: float=.64
    activation_hill: float=4.6
    off_half_uM: float=.11
    off_floor_per_s: float=.0005
    pmca2_ratio: float=1.
    pmca2_share: float=.5
    membrane_support: float=1.
    saturation_weight: float=0.
    saturation_half_uM: float=.43
    saturation_hill: float=2.
    provenance: str='Hypothesis: Caride2001 PMCA2b/CaM source anchors, not measured human MN. Saturation optional: Constantin SI model borrowed from human T cells.'
    def vector(self):
        d=asdict(self);v=np.array([v for k,v in d.items() if k!='provenance'],float)
        if not np.isfinite(v).all() or min(v)<0 or not self.provenance.strip():raise ValueError('Finite nonnegative parameters/provenance required')
        if not 0<=self.pmca2_share<=1 or not 0<=self.membrane_support<=1 or not 0<=self.saturation_weight<=1:raise ValueError('Fractions outside [0,1]')
        if min(self.activation_half_uM,self.activation_hill,self.off_half_uM,self.saturation_half_uM,self.saturation_hill,self.forget_at_50nM_per_s)<=0 or self.forget_at_50nM_per_s<self.off_floor_per_s:raise ValueError('Invalid rates or half concentrations')
        return v

PMCA_MEMORY_RANGES=dict(memory_gain=[0.,.5,1.],forget_at_50nM_per_s=[.007,.021,.063],free_CaM_uM=[.03,.117,.3],
                       pmca2_share=[0.,.5,1.],pmca2_ratio=[1.,2.066863],membrane_support=[.25,1.],saturation_weight=[0.,1.])

def memory_rates(c,p):
    v=p.vector() if isinstance(p,PMCAMemoryParameters) else p
    c=np.maximum(c,0.)
    on=v[2]*v[3]*1e-6*c**v[5]/(c**v[5]+v[4]**v[5])
    off=v[7]+(v[1]-v[7])*(1+(.05/v[6])**4)/(1+(c/v[6])**4)
    return on,off

def memory_equilibrium(c,p):
    a,b=memory_rates(c,p);return a/(a+b)

def pmca2_flux(c,activation,c0,gamma,pmca_fraction,atp_factor,p):
    v=p.vector() if isinstance(p,PMCAMemoryParameters) else p
    c=max(float(c),0.)
    base=memory_equilibrium(c0,v)
    memory=(1+v[0]*activation)/(1+v[0]*base)
    hill=c**v[13]/(v[12]**v[13]+c**v[13])
    hill0=c0**v[13]/(v[12]**v[13]+c0**v[13])
    shaped=(1-v[11])*c+v[11]*c0*hill/hill0
    return pmca_fraction*gamma*atp_factor*((1-v[9])*c+v[9]*v[8]*v[10]*shaped*memory)

def simulate_pmca_memory(segments,parameters=None,checkpoint=None,output_step_s=.2):
    """Prescribed Ca segments [start,end,Ca_uM]; no Ca/ATP consumption simulated."""
    p=PMCAMemoryParameters(memory_gain=1.) if parameters is None else parameters
    v=p.vector();sg=np.asarray(segments,float)
    if sg.ndim!=2 or sg.shape[1]!=3 or not len(sg) or not np.isfinite(sg).all() or np.any(sg<0) or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[1:,0]!=sg[:-1,1]):raise ValueError('Invalid Ca protocol')
    if not np.isfinite(output_step_s) or output_step_s<=0:raise ValueError('Positive output step required')
    fp=hashlib.sha256(json.dumps(asdict(p),sort_keys=True).encode()).hexdigest()
    a=0.
    if checkpoint is not None:
        if checkpoint['fingerprint']!=fp or checkpoint['time_s']!=sg[0,0] or not 0<=checkpoint['activation']<=1:raise ValueError('Incompatible checkpoint')
        a=checkpoint['activation']
    ts=[sg[0,0]];aa=[a];cc=[sg[0,2]]
    for start,end,c in sg:
        on,off=memory_rates(c,v);ss=on/(on+off)
        t=np.unique(np.r_[np.arange(start+output_step_s,end,output_step_s),end]);x=ss+(a-ss)*np.exp(-(on+off)*(t-start))
        ts.extend(t);aa.extend(x);cc.extend(np.full(len(t),c));a=float(x[-1])
    return dict(time_s=np.array(ts),activation=np.array(aa),calcium_uM=np.array(cc),parameters=asdict(p),
                checkpoint=dict(time_s=float(sg[-1,1]),activation=a,fingerprint=fp),human_kinetic_fit=False)

def pmca_source_reference():
    return json.loads((B/'source_constraints.json').read_text(encoding='utf-8'))

def pmca_density_capacity(*,density_per_um2,area_um2,accessible_volume_um3,cycles_per_s):
    """Stoichiometric upper capacity, not actual flux, gamma, or human fit."""
    if not np.isfinite([density_per_um2,area_um2,accessible_volume_um3,cycles_per_s]).all() or min(density_per_um2,cycles_per_s)<0 or min(area_um2,accessible_volume_um3)<=0:raise ValueError('Invalid pump geometry/cycle parameters')
    pumps=density_per_um2*area_um2
    capacity=pumps*cycles_per_s/(602.214076*accessible_volume_um3)
    return dict(pump_count=pumps,max_Ca_flux_uM_s=capacity,max_ATP_cost_uM_s=capacity,
                limitation='All specified pumps assumed active and saturated. Use actual local Ca/ATP/support and spatial geometry to obtain flux; not an extrusion rate constant.')

def run_pmca_memory_probe(*,pmca_parameters,**kwargs):
    import sys
    sys.path.insert(0,str(B))
    from electrical_pmca import run_pmca_coupled
    return run_pmca_coupled(pmca_parameters=pmca_parameters,**kwargs)

def run_pmca_memory_network(segments,*,pmca_parameters,**kwargs):
    import sys
    sys.path.insert(0,str(B))
    from long_pmca import run_pmca_network
    return run_pmca_network(segments,pmca_parameters=pmca_parameters,**kwargs)
