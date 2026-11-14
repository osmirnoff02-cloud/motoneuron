"""Explicit uncalibrated cytosol/ER scenario motivated by Dafinca/Jahn.

ER is an amount divided by cytosolic volume (uM-equivalent), NOT ER free [Ca].
All kinetic inputs are caller-supplied. No default species or AP/SK coupling.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

@dataclass(frozen=True)
class ERScenario:
    basal_ca_uM:float
    initial_er_uM_equivalent:float
    cytosolic_kappa:float
    extrusion_per_s:float
    er_leak_per_s:float
    provenance:str
    def __post_init__(self):
        x=[self.basal_ca_uM,self.initial_er_uM_equivalent,self.cytosolic_kappa,self.extrusion_per_s,self.er_leak_per_s]
        if not np.isfinite(x).all() or self.basal_ca_uM<=0 or min(x[1:])<0 or not self.provenance.strip():raise ValueError('Explicit nonnegative scenario parameters and provenance required')

def er_exchange(time_s,*,scenario,influx_uM_s,ryr_open_per_s,serca_fraction,basal_entry_fraction):
    """Zero-order-held commands at time_s; exact endpoint splitting of stimuli.

    J_SERCA = leak * initial_store * Ca/basal_Ca * serca_fraction.
    J_release = (leak + ryr_open) * store. Linear pump is a local scenario.
    Basal entry fraction=1 balances extrusion at rest, 0 removes that entry.
    Separate extrusion and store transfer; neither fitted from fluorescence tau.
    """
    t=np.asarray(time_s,float);p=scenario
    if not isinstance(p,ERScenario) or t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Explicit scenario and increasing times required')
    commands=[np.asarray(x,float) for x in [influx_uM_s,ryr_open_per_s,serca_fraction,basal_entry_fraction]]
    if any(x.shape!=t.shape or not np.isfinite(x).all() or np.any(x<0) for x in commands):raise ValueError('Finite nonnegative paired command arrays required')
    if np.any(commands[2]>1) or np.any(commands[3]>1):raise ValueError('Fractions must lie in [0,1]')
    cs=np.vstack(commands);cuts=np.r_[0,np.where(np.any(cs[:,1:]!=cs[:,:-1],axis=0))[0]+1,len(t)-1]
    cuts=np.unique(cuts);y=np.empty((3,len(t)));y[:,0]=[p.basal_ca_uM,p.initial_er_uM_equivalent,0.]
    pump_gain=p.er_leak_per_s*p.initial_er_uM_equivalent/p.basal_ca_uM
    for a,b in zip(cuts[:-1],cuts[1:]):
        entry,opening,serca,background=cs[:,a]
        def rhs(_,state):
            c,q,_=state
            net=entry+background*p.extrusion_per_s*p.basal_ca_uM-p.extrusion_per_s*c
            transfer=(p.er_leak_per_s+opening)*q-serca*pump_gain*c
            return [(net+transfer)/(1+p.cytosolic_kappa),-transfer,net]
        sol=solve_ivp(rhs,(t[a],t[b]),y[:,a],t_eval=t[a:b+1],rtol=1e-9,atol=1e-11,method='LSODA')
        if not sol.success:raise RuntimeError(sol.message)
        y[:,a:b+1]=sol.y
    total=(1+p.cytosolic_kappa)*y[0]+y[1]
    residual=total-total[0]-y[2]
    if min(y[0].min(),y[1].min()) < -1e-8:raise RuntimeError('Nonphysical negative concentration/store')
    return dict(time_s=t,ca_uM=y[0],er_uM_equivalent=y[1],net_membrane_accumulation_uM_equivalent=y[2],
        conservation_error_uM_equivalent=float(np.max(abs(residual))),provenance=p.provenance,
        status='uncalibrated finite-store scenario; no ER free-Ca measurement, no human rates, no SK feedback')
