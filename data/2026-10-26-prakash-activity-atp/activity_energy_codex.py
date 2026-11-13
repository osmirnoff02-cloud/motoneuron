"""[codex] 2026-10-26. Optional activity-driven terminal ATP synthesis.

Rangaraju2014 supports coupling, not these human gains or time constants.
State 294 is a filtered AP frequency (Hz), 295 is cumulative additional ATP
production (uM). ATP is converted from ADP, never added to adenylate mass.
Default gain=0 exactly preserves the prior kinetic model.
"""
from dataclasses import dataclass,asdict
import numpy as np
from numba import njit

@dataclass(frozen=True)
class ActivityEnergyParameters:
    gain:float=0.
    decay_s:float=10.
    half_Hz:float=5.
    provenance:str='Conditional activity-coupled production; gain/tau/half-frequency are unmeasured in human motor terminals. Rangaraju2014 rat hippocampus is a separate benchmark.'
    def vector(self):
        v=np.array([self.gain,self.decay_s,self.half_Hz])
        if not np.isfinite(v).all() or v[0]<0 or min(v[1:])<=0 or not self.provenance:raise ValueError('Invalid activity energy parameters')
        return v

ACTIVITY_ENERGY_RANGES=dict(gain=[0.,30.,60.,100.],decay_s=[1.,10.,60.],half_Hz=[1.,5.,10.])
ACTIVITY_NAMES=('terminal_filtered_AP_Hz','terminal_activity_produced_ATP_uM')

@njit
def activity_extension(z,old,tv,av):
    out=np.zeros(296);out[:294]=old
    if tv[0]:
        hz=max(0.,z[294]);extra=old[288]*av[0]*hz/(av[2]+hz)
        out[282]+=extra;out[283]-=extra;out[288]+=extra
        out[294]=-hz/av[1];out[295]=extra
    return out

def activity_jump(z,tv,av):
    y=z.copy()
    if tv[0]:y[294]+=1./av[1]
    return y

@njit
def isolated_terminal_rhs(z,tv,npv,en,av,supply):
    """21 terminal states; same budget as API0.17, isolated from the process.

    Passive endplate/gACh states are excluded from this *energy* benchmark,
    not reused as a claim about muscle transmission. Exchange must be zero.
    """
    out=np.zeros(21)
    c,a=z[0],z[1];A,D,M=z[7:10]
    f=max(A,0.)/(npv[7]+max(A,0.));f0=en[0]/(npv[7]+en[0]);fn=f/f0
    repl=max(0.,1-a)/npv[6]*f*npv[5]
    caout=max(c,0.)/npv[1]*(1+tv[5])*fn
    other=tv[3]*fn
    prod=tv[3]*(max(D,0.)/(tv[4]+max(D,0.)))/(en[1]/(tv[4]+en[1]))*supply
    extra=prod*av[0]*max(z[19],0.)/(av[2]+max(z[19],0.))
    prod+=extra
    recycle=repl*npv[15]/(602.214076*tv[1])
    ak=en[11]*(D*D-A*M/(en[0]*en[2]/en[1]**2))
    cost=other+recycle+caout
    out[0]=-caout/(1+tv[5]);out[1]=repl/npv[5]
    out[5]=repl;out[6]=repl*npv[15]
    out[7:10]=np.array([prod-cost+ak,cost-prod-2*ak,ak])
    out[13:19]=np.array([prod,other,caout,ak,0.,caout])
    out[19]=-z[19]/av[1];out[20]=extra
    return out
