"""[codex] 2026-11-02. Conditional GAP43-dependent recovery readout.

This is a normalized virtual-lesion assay, NOT anatomical length or a repaired
NMJ. Animal GAP43 localization/UTR experiments motivate the direction only.
Fallini HuD/IMP1 rescue is not a GAP43-only dose response. No human rate fit.
The readout consumes existing GAP43 and ATP values without consuming protein;
it adds no unmeasured tissue mass or guessed repair-energy debit to the neuron.
"""
from dataclasses import dataclass,asdict
import numpy as np
from numba import njit

REGROWTH_RANGES=dict(repair_time_h=[6.,24.,72.],gap43_dependence=[0.,.5,1.],
 gap43_half_relative=[.25,1.,4.],gap43_hill=[1.,2.,4.],ATP_half_uM=[100.,500.,2000.],
 initial_deficit=[0.,.25,.75],injury_per_h=[0.,.25,1.],
 permissive_support=[0.,.25,1.],GAP43_function_support=[0.,.25,1.])

@dataclass(frozen=True)
class GAP43RegrowthParameters:
 repair_time_h:float=24.
 gap43_dependence:float=.5
 gap43_half_relative:float=1.
 gap43_hill:float=2.
 ATP_half_uM:float=500.
 initial_deficit:float=0.
 def vector(self):
  v=np.array(list(asdict(self).values()),float)
  if not np.isfinite(v).all() or min(v[[0,2,3,4]])<=0 or not 0<=v[1]<=1 or not 0<=v[5]<=1:
   raise ValueError('Positive recovery time, response scale/Hill and ATP half; dependence/deficit in [0,1]')
  return v

NAMES=('GAP43_assay_integrity_fraction','GAP43_assay_injury_cumulative','GAP43_assay_recovery_cumulative')

def initialize(base,v):
 if len(base)!=617 or base[583]<=0 or base[136]<=0:raise ValueError('Positive reference GAP43/ATP and617-state RNA branch required')
 return np.r_[base,1-v[5],0.,0.],np.array([base[583],base[136]])

def controls(segments,supplied=None):
 sg=np.asarray(segments,float)
 a=np.c_[sg[:,:2],np.zeros(len(sg)),np.ones((len(sg),2))] if supplied is None else np.asarray(supplied,float)
 if a.shape!=(len(sg),5) or not np.isfinite(a).all() or not np.array_equal(a[:,:2],sg[:,:2]) or np.any(a[:,2:]<0) or np.any(a[:,3:]>2):
  raise ValueError('Regrowth schedule:matching start/end,nonnegative injury/h,permissive and GAP43 function supports0..2')
 return a.copy()

@njit
def factors(z,v,ctl,ref):
 g=max(z[583],0.)/ref[0]*ctl[2];k=v[2];n=v[3]
 signal=(1+k**n)*g**n/(k**n+g**n)
 # dependence=0 is an explicit null hypothesis, not an error.
 gap=(1-v[1])+v[1]*signal
 ATP=max(z[136],0.);energy=min(1.,ATP/(v[4]+ATP)*(v[4]+ref[1])/ref[1])
 return gap,energy

@njit
def extension(z,old,v,ctl,ref):
 out=np.zeros(620);out[:617]=old
 x=min(1.,max(0.,z[617]));gap,energy=factors(z,v,ctl,ref)
 injury=ctl[0]/3600*x
 repair=(1-x)/(v[0]*3600)*gap*energy*ctl[1]
 out[617:620]=np.array([repair-injury,injury,repair])
 return out

def checks(z,base):
 d=z-base
 return dict(GAP43_recovery_ledger=float(np.max(abs(d[:,617]+d[:,618]-d[:,619]))),
  GAP43_recovery_bounds=float(max(0.,-z[:,617:620].min(),z[:,617].max()-1)))

def observe(time_s,z,segments,v,ref):
 ix=np.clip(np.searchsorted(segments[:,0],time_s,side='right')-1,0,len(segments)-1)
 f=np.array([factors(q,v,segments[j,2:],ref) for q,j in zip(z,ix)])
 return dict(integrity_fraction=z[:,617],unrecovered_fraction=1-z[:,617],
  cumulative_injury=z[:,618],cumulative_recovery=z[:,619],GAP43_factor=f[:,0],ATP_factor=f[:,1],
  scope='Conditional normalized virtual-lesion readout, not length, membrane regrowth or NMJ function',
  human_kinetic_fit=False,feeds_back_to_geometry_or_electrical=False,
  energy_scope='Uses existing ATP as permissive signal; no new physical tissue or repair ATP budget represented')
