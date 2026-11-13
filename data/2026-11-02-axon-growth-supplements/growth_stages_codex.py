"""[codex]2026-11-02. Alternative downstream assay: initiation vs progression.

Not an anatomical branch simulator. GAP43 roles in rat DRG elongation and adult
mouse climbing-fibre sprouting are separate evidence, not human kinetic fits.
No extra ATP debit: the assay makes no physical new tissue. Core remains620D.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp

GROWTH_STAGE_RANGES=dict(initiation_time_h=[6.,24.,72.],progression_time_h=[6.,24.,72.],
 gap43_dependence=[0.,.5,1.],gap43_half_relative=[.25,1.,4.],gap43_hill=[1.,2.,4.],
 ATP_half_uM=[100.,500.,2000.],initial_deficit=[0.,.25,.75])

@dataclass(frozen=True)
class GrowthStageParameters:
 initiation_time_h:float=24.
 progression_time_h:float=24.
 gap43_dependence:float=.5
 gap43_half_relative:float=1.
 gap43_hill:float=2.
 ATP_half_uM:float=500.
 initial_deficit:float=0.
 gap43_stage:str='initiation'
 def validate(self):
  d=asdict(self);mode=d.pop('gap43_stage');v=np.array(list(d.values()),float)
  if not np.isfinite(v).all() or min(v[[0,1,3,4,5]])<=0 or not 0<=v[2]<=1 or not 0<=v[6]<=1 or mode not in ('initiation','progression','both'):
   raise ValueError('Positive finite times/scales;dependence/deficit0..1;stage initiation/progression/both')
  return self

def evidence():
 return json.loads((Path(__file__).resolve().parent/'observations_codex.json').read_text(encoding='utf8'))

def run_growth_stages(time_s,gap43_relative,ATP_uM,schedule,*,parameters=None,
                      reference_ATP_uM=None,initial_state=None,step_s=120.,rtol=1e-9,atol=1e-11):
 """U=uninitiated deficit,S=initiated deficit,R=recovered/initially intact.

 U' = d*(S+R)-h*U; S'=h*U-(k+d)*S; R'=k*S-d*R.
 Repeated lesions retain all state; expected fractions sum to1, not cell counts.
 schedule columns:start,end,injury/h,permissive,GAP43 function support.
 Full solver inputs must cover all schedule times; no extrapolation permitted.
 A continuation requires the original parameters and reference ATP explicitly,
 so a new input window cannot silently renormalize the energy response.
 """
 p=(parameters or GrowthStageParameters()).validate()
 t=np.asarray(time_s,float);g=np.asarray(gap43_relative,float);a=np.asarray(ATP_uM,float);sg=np.asarray(schedule,float)
 if t.ndim!=1 or len(t)<2 or g.shape!=t.shape or a.shape!=t.shape or not np.isfinite(np.r_[t,g,a]).all() or np.any(np.diff(t)<=0) or min(g.min(),a.min())<0:
  raise ValueError('Finite ordered time series and nonnegative protein/ATP required')
 if sg.ndim!=2 or sg.shape[1]!=5 or not len(sg) or not np.isfinite(sg).all() or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[:,2:]<0) or np.any(sg[:,3:]>2) or not np.array_equal(sg[1:,0],sg[:-1,1]) or sg[0,0]<t[0] or sg[-1,1]>t[-1]:
  raise ValueError('Contiguous five-column schedule within the recorded time range required')
 if initial_state is not None and (reference_ATP_uM is None or parameters is None):
  raise ValueError('Continuation requires original parameters and reference_ATP_uM')
 ref=float(a[0] if reference_ATP_uM is None else reference_ATP_uM)
 if not np.isfinite([ref,step_s,rtol,atol]).all() or min(ref,step_s,rtol,atol)<=0:raise ValueError('Positive referenceATP and solver settings required')
 y=np.array([p.initial_deficit,0.,1-p.initial_deficit,0.,0.,0.]) if initial_state is None else np.array(initial_state,float,copy=True)
 if y.shape!=(6,) or not np.isfinite(y).all() or min(y)<-1e-8 or abs(y[:3].sum()-1)>1e-8:raise ValueError('Six valid checkpoint fractions/ledgers required')
 origin=y.copy();times=[float(sg[0,0])];states=[y.copy()]
 for row in sg:
  lo,hi,inj,permissive,function=row
  def rhs(local_time,q):
   ti=lo+local_time;gv=np.interp(ti,t,g)*function;av=np.interp(ti,t,a)
   signal=(1+p.gap43_half_relative**p.gap43_hill)*gv**p.gap43_hill/(p.gap43_half_relative**p.gap43_hill+gv**p.gap43_hill)
   factor=1-p.gap43_dependence+p.gap43_dependence*signal
   en=min(1.,av/(p.ATP_half_uM+av)*(p.ATP_half_uM+ref)/ref)*permissive
   h=en/(3600*p.initiation_time_h)*(factor if p.gap43_stage in ('initiation','both') else 1.)
   k=en/(3600*p.progression_time_h)*(factor if p.gap43_stage in ('progression','both') else 1.)
   start=h*q[0];finish=k*q[1];damage=inj/3600*(q[1]+q[2])
   return [damage-start,start-finish-inj/3600*q[1],finish-inj/3600*q[2],start,finish,damage]
  grid=np.r_[np.arange(step_s,hi-lo,step_s),hi-lo]
  sol=solve_ivp(rhs,(0.,hi-lo),y,t_eval=grid,method='LSODA',rtol=rtol,atol=atol,max_step=step_s)
  if not sol.success:raise RuntimeError(sol.message)
  y=sol.y[:,-1];times.extend(lo+sol.t);states.extend(sol.y.T)
 z=np.asarray(states);d=z-origin
 checks=dict(fraction_sum_error=float(np.max(abs(z[:,:3].sum(1)-1))),
  initiation_ledger_error=float(np.max(abs(d[:,0]-d[:,5]+d[:,3]))),
  nonnegative_error=float(max(0.,-z.min())))
 return dict(time_s=np.array(times),state=z,columns=('uninitiated','initiated_unrecovered','intact_or_recovered','initiation_cumulative','progression_cumulative','injury_cumulative'),
  checks=checks,parameters=asdict(p),reference_ATP_uM=ref,checkpoint=y.copy(),
  human_calibrated=False,physical_geometry=False,NMJ_function=False,upstream_feedback=False,
  scope='Conditional fractions in an alternative virtual-lesion assay; not observed sprouting frequencies, lengths or synapses')
