"""[codex] 2026-10-26. Optional delayed vesicle work and split ATP supply.

All human rates remain assumptions. Two conserved queues replace the old
instantaneous refill pathway: released -> surface -> internal -> ready.
The internal stage lumps reacidification/loading/priming, not a pH equation.
Production is an effective ADP->ATP budget, not resolved glycolytic enzymes.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from activity_energy_codex import isolated_terminal_rhs

@dataclass(frozen=True)
class RecyclingParameters:
    enabled:bool=False
    endocytosis_tau_s:float=6.
    endocytosis_capacity_vesicles:float=28.
    maturation_tau_s:float=10.
    endocytosis_cost_fraction:float=.5
    provenance:str='Effective two-stage queue. Balaji2008 ~6s/~28 vesicles at36C/30Hz is a rat CNS reference, not human NMJ kinetics. Maturation/cost split unknown.'
    def vector(self):
        v=np.array([float(self.enabled),self.endocytosis_tau_s,self.endocytosis_capacity_vesicles,self.maturation_tau_s,self.endocytosis_cost_fraction])
        if not isinstance(self.enabled,bool) or not np.isfinite(v).all() or min(v[1:4])<=0 or not 0<=v[4]<=1 or not self.provenance:raise ValueError('Invalid recycling parameters')
        return v

@dataclass(frozen=True)
class TerminalPathwayParameters:
    enabled:bool=False
    glycolysis_fraction:float=.5
    glycolysis_half_ADP_uM:float=200.
    oxidative_half_ADP_uM:float=200.
    glycolysis_activity_weight:float=1.
    oxidative_activity_weight:float=1.
    provenance:str='Hypothetical allocation of existing terminal ADP-dependent ATP supply; fractions/affinities not fitted in human motor neurons. No terminal glucose/O2/proton balance.'
    def vector(self):
        v=np.array([float(self.enabled),self.glycolysis_fraction,self.glycolysis_half_ADP_uM,self.oxidative_half_ADP_uM,self.glycolysis_activity_weight,self.oxidative_activity_weight])
        if not isinstance(self.enabled,bool) or not np.isfinite(v).all() or not 0<=v[1]<=1 or min(v[2:4])<=0 or min(v[4:])<0 or not self.provenance:raise ValueError('Invalid terminal pathways')
        return v

RECYCLING_RANGES=dict(endocytosis_tau_s=[6.,20.,60.],endocytosis_capacity_vesicles=[5.,28.,100.],maturation_tau_s=[1.,10.,60.],endocytosis_cost_fraction=[0.,.5,1.],ATP_per_recycled_vesicle=[100.,1000.,10000.])
PATHWAY_RANGES=dict(glycolysis_fraction=[.2,.5,.8],glycolysis_half_ADP_uM=[50.,200.,1000.],oxidative_half_ADP_uM=[50.,200.,1000.],glycolysis_activity_weight=[0.,1.,2.],oxidative_activity_weight=[0.,1.,2.],glycolysis_support=[0.,.5,1.],oxidative_support=[0.,.5,1.])
RECYCLING_NAMES=('terminal_surface_queue_quanta','terminal_maturation_queue_quanta','terminal_endocytosed_quanta','terminal_endocytosis_ATP_molecules','terminal_maturation_ATP_molecules','terminal_glycolytic_ATP_uM','terminal_oxidative_ATP_uM')

@njit
def local_extension(z,old,tv,nv,en,av,rc,paths,support,gly_support,ox_support):
    """28 local states, corresponding to global275:303. Replace, never add twice."""
    out=np.zeros(28);out[:21]=old
    if not tv[0]:return out
    A=max(z[7],0.);D=max(z[8],0.)
    fn=(A/(nv[7]+A))/(en[0]/(nv[7]+en[0]))
    if rc[0]:
        endo=min(max(z[21],0.),rc[2])/rc[1]*fn
        mat=max(z[22],0.)/rc[3]*fn
        ce=endo*nv[15]*rc[4];cm=mat*nv[15]*(1-rc[4])
        delta_cost=(ce+cm-old[6])/(602.214076*tv[1])
        out[1]=mat/nv[5];out[5]=mat;out[6]=ce+cm
        out[7]-=delta_cost;out[8]+=delta_cost
        out[21]=-endo;out[22]=endo-mat;out[23]=endo;out[24]=ce;out[25]=cm
    if paths[0]:
        h=max(z[19],0.);signal=av[0]*h/(av[2]+h)
        g0=tv[3]*paths[1]*(D/(paths[2]+D))/(en[1]/(paths[2]+en[1]))*support*gly_support
        o0=tv[3]*(1-paths[1])*(D/(paths[3]+D))/(en[1]/(paths[3]+en[1]))*support*ox_support
        gg=g0*(1+signal*paths[4]);oo=o0*(1+signal*paths[5])
        delta_prod=gg+oo-old[13]
        out[7]+=delta_prod;out[8]-=delta_prod;out[13]=gg+oo
        out[20]=signal*(g0*paths[4]+o0*paths[5]);out[26]=gg;out[27]=oo
    return out

@njit
def recycling_extension(z,old,tv,nv,en,av,rc,paths,support,gly_support,ox_support):
    out=np.zeros(303);out[:275]=old[:275]
    out[275:]=local_extension(z[275:],old[275:296],tv,nv,en,av,rc,paths,support,gly_support,ox_support)
    return out

@njit
def isolated_recycling_rhs(z,tv,nv,en,av,rc,paths,support,gly_support,ox_support):
    return local_extension(z,isolated_terminal_rhs(z,tv,nv,en,av,support),tv,nv,en,av,rc,paths,support,gly_support,ox_support)

def recycling_jump(z,released,tv,rc):
    y=z.copy()
    if tv[0] and rc[0]:y[296]+=released
    return y

def takeuchi_constraints(association,initiation):
    """Qualitative source constraint, not a fitted GFP/TDP turnover rate."""
    if not 0<=association<=1 or not 0<=initiation<=1:raise ValueError('Fractions required')
    return association*initiation
