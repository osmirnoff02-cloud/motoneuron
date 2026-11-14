"""[codex] 2026-10-24. Explicit effective cytosolic ATP budget; uM, seconds.

Pool reference is Connolly2016 MODEL, not measured human iPSC-MN ATP.
ATP saturation, demand, reserve and pathway split are sensitivity hypotheses.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit

@dataclass(frozen=True)
class EnergyParameters:
    ATP0_uM: float = 2700.
    ADP0_uM: float = 200.
    AMP0_uM: float = 24.
    other_demand_uM_s: float = 6.
    pump_half_ATP_uM: float = 300.
    other_half_ATP_uM: float = 300.
    ox_half_ADP_uM: float = 200.
    gly_half_ADP_uM: float = 200.
    glycolytic_fraction: float = .3
    pmca_fraction: float = .5
    ncx_ATP_equivalent: float = 1.
    adenylate_kinase_per_uM_s: float = 1e-5
    pump_feedback: bool = True
    provenance: str = 'Connolly2016 reference pools; stoichiometry PMCA1/SERCA2 Ca per ATP. Other parameters are hypotheses, not a human kinetic fit.'
    def validate(self):
        d=asdict(self);v=[x for k,x in d.items() if k not in ('provenance','pump_feedback')]
        if not np.isfinite(v).all() or min(v)<0 or not self.provenance.strip():raise ValueError('Finite nonnegative energy parameters and provenance required')
        if min(self.ATP0_uM,self.ADP0_uM,self.AMP0_uM,self.pump_half_ATP_uM,self.other_half_ATP_uM,self.ox_half_ADP_uM,self.gly_half_ADP_uM)<=0:raise ValueError('Positive initial pools and half-saturation constants required')
        if not 0<=self.glycolytic_fraction<=1 or not 0<=self.pmca_fraction<=1:raise ValueError('Energy fractions outside [0,1]')
        if not isinstance(self.pump_feedback,bool):raise ValueError('pump_feedback must be bool')
    def vector(self):
        self.validate()
        return np.array([self.ATP0_uM,self.ADP0_uM,self.AMP0_uM,self.other_demand_uM_s,self.pump_half_ATP_uM,self.other_half_ATP_uM,self.ox_half_ADP_uM,self.gly_half_ADP_uM,self.glycolytic_fraction,self.pmca_fraction,self.ncx_ATP_equivalent,self.adenylate_kinase_per_uM_s,float(self.pump_feedback)])

ENERGY_RANGES=dict(ATP0_uM=[1000.,2700.,5000.],other_demand_uM_s=[1.,6.,30.],
    pump_half_ATP_uM=[50.,300.,1000.],other_half_ATP_uM=[50.,300.,1000.],
    ox_half_ADP_uM=[50.,200.,1000.],gly_half_ADP_uM=[50.,200.,1000.],
    glycolytic_fraction=[.1,.3,.7],pmca_fraction=[0.,.5,1.],
    adenylate_kinase_per_uM_s=[0.,1e-5,1e-4],supply_scale=[.25,.5,1.])

@njit
def normalized_saturation(x,k,x0):
    xx=max(x,0.)
    return xx/(k+xx)*(k+x0)/x0

@njit
def energy_fluxes(a,d,m,c,serca_unscaled,gamma,c0,serca0,en,drive):
    f=normalized_saturation(a,en[4],en[0]) if en[12]>0 else 1.
    pmca=en[9]*gamma*max(c,0.)*f
    ncx=(1-en[9])*gamma*max(c,0.)*f
    serca=serca_unscaled*f
    other=en[3]*drive[2]*normalized_saturation(a,en[5],en[0])
    basal=en[3]+gamma*c0*(en[9]+(1-en[9])*en[10])+serca0/2
    ox=basal*(1-en[8])*drive[0]*normalized_saturation(d,en[6],en[1])
    gly=basal*en[8]*drive[1]*normalized_saturation(d,en[7],en[1])
    keq=en[0]*en[2]/(en[1]*en[1])
    ak=en[11]*(d*d-a*m/keq)
    return np.array([pmca,ncx,serca,other,ox,gly,ak])

def validate_schedule(schedule,segments):
    sg=np.asarray(segments,float)
    e=np.c_[sg[:,:2],np.ones((len(sg),3))] if schedule is None else np.asarray(schedule,float)
    if e.shape!=(len(sg),5) or not np.isfinite(e).all() or np.any(e<0) or not np.array_equal(e[:,:2],sg[:,:2]):raise ValueError('Energy schedule must match every stimulus segment: start,end,OXPHOS support,glycolysis support,other demand')
    return e
