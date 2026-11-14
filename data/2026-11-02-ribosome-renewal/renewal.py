"""[codex]2026-11-02. Bounded production/delivery control with basal maintenance.

Phenomenological homeostasis, not a measured human feedback pathway.
No particle, amino acid or ATP is created by this controller. It changes the
capacities of the existing resource-limited synthesis and transport fluxes.
47S remains coupled: unequal40/60 losses can accumulate precursor surplus.
"""
from dataclasses import dataclass,asdict
import numpy as np
from numba import njit

RENEWAL_RANGES=dict(response_s=[1800.,7200.,28800.],production_gain=[0.,2.,8.],
 delivery_gain=[0.,2.,8.],RP_pool_gain=[0.,2.,8.],maximum_capacity_factor=[1.,2.,4.],
 production_adaptation=[0.,1.],delivery_adaptation=[0.,1.],RP_adaptation=[0.,1.])

@dataclass(frozen=True)
class RibosomeRenewalParameters:
 response_s:float=7200.
 production_gain:float=2.
 delivery_gain:float=2.
 RP_pool_gain:float=2.
 maximum_capacity_factor:float=4.
 def vector(self):
  v=np.array(list(asdict(self).values()),float)
  if not np.isfinite(v).all() or v[0]<=0 or min(v[1:4])<0 or v[4]<1:raise ValueError('Positive response time, nonnegative gains, capacity limit>=1')
  return v

NAMES=('renewal_production_capacity_factor','renewal_delivery40_capacity_factor','renewal_delivery60_capacity_factor')

def initialize(base,v,rm,lt,lr,b,ratio,KATP):
 z=np.r_[base,np.ones(3)];ep=z[136]/(KATP+z[136]);scale=b[0]*ratio
 free=z[564:566];rp=rm[6]
 if rp<=0 or min(free)<=0:raise ValueError('Renewal needs positive reference RPS26 and local free pools')
 retirement=np.log(2)/(rm[2]*3600)*ep
 repair=ep/rm[1]*rp/(rm[7]+rp)
 bad=rm[0]*free/np.array([repair+retirement*rm[3],retirement*rm[3]])
 loss=retirement*(free+rm[3]*bad)
 extra=rm[0]*free+retirement*free;extra[0]-=repair*bad[0]
 birth=np.log(2)/(b[1]*86400);send=z[390:392]/lr[0]
 prod=1+max(loss)/birth;delivery=1+extra/send
 gg=z[576]/(lt[14]+z[576]);qg=z[575]/(lt[13]+z[575])
 required=repair*bad[0]*scale+np.log(2)/(rm[9]*3600)*rp*ep
 # Slots count all elongating chains; reserve enough initiation capacity.
 expression=required*(lt[2]/gg+115/(lt[1]*gg*qg))/max(rm[8],1e-30)
 ref=np.r_[base[330:332],free,rp,prod,delivery,expression,bad,loss]
 # Start with old pools; adaptations and maintenance run from t=0, no warm reset.
 return z,ref

def controls(sg,supplied=None):
 a=np.c_[sg[:,:2],np.ones((len(sg),3))] if supplied is None else np.asarray(supplied,float)
 if a.shape!=(len(sg),5) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>1):raise ValueError('Renewal:start,end,production adaptation,delivery adaptation,RP adaptation in0..1')
 return a.copy()

@njit
def modulate(z,v,ref,ctl,bc,sc,rc):
 b=bc.copy();s=sc.copy();r=rc.copy()
 factor=1+ctl[0]*(z[675]-1)
 b[0]*=factor;b[1]*=factor;s[6]*=factor
 rp=max(z[643],0.)
 expr=ref[8]*(1+v[3]*(1-rp/ref[4]))
 # Adaptation=0 uses fixed demand-derived basal expression, never zero supply.
 expr=ref[8]+ctl[2]*(expr-ref[8])
 r[5]*=min(1.,max(0.,expr))
 return b,s,r

@njit
def delivery(z,old,v,ctl,lr,lc,C,ratio,ATP0,KATP):
 out=old.copy();a=max(z[120],0.);energy=min(1.,a/(KATP+a)*(KATP+ATP0)/ATP0)
 delta=np.maximum(z[390:392],0.)/lr[0]*energy*lc[0]*ctl[1]*(z[676:678]-1)
 out[390:392]-=delta;out[562:564]+=delta;out[566:568]+=delta
 cost=delta.sum()*C*lr[4];out[572]+=cost
 out[120]-=cost;out[121]+=cost;out[125]+=cost
 return out

@njit
def finish(z,old,v,ref):
 out=np.zeros(678);out[:675]=old
 starget=ref[5]+v[1]*(1-min(max(z[330],0)/ref[0],max(z[331],0)/ref[1]))
 targets=np.empty(3);targets[0]=starget
 targets[1:3]=ref[6:8]+v[2]*(1-np.maximum(z[564:566],0)/ref[2:4])
 for i in range(3):out[675+i]=(min(v[4],max(0.,targets[i]))-z[675+i])/v[0]
 return out

def observe(z,base,ref):
 return dict(capacity_factors=z[:,675:678],local_free_relative=z[:,564:566]/base[564:566],
  soma_free_relative=z[:,330:332]/base[330:332],spare_RPS26_relative=z[:,643]/base[643],
  precursor_RNA_Ceq=z[:,384:388],reference=ref.tolist(),
  reference_RP_capacity_sufficient=bool(ref[8]<=1),human_calibrated=False,
  scope='Continuous selected maintenance with delayed bounded capacities; old inventories start a transient. No sensor molecule calibration, no rRNA salvage, no complete balanced whole proteome; unequal40/60 loss may build47S-derived surplus. Finite substrates and energy gates remain in original fluxes.')
