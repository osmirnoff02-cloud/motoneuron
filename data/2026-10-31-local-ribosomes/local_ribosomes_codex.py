"""[codex]2026-10-31. Conserved soma->neurite precursor delivery/maturation.

Pools use the existing selected-ribosome reference C (soma-volume equivalent),
not local concentrations. No independent copy of a somatic particle is made.
Local mature particles are an idle reserve; return to soma precedes reuse or
retirement. Local translation and RP exchange are NOT implemented here.
"""
from dataclasses import dataclass,asdict
import numpy as np
from numba import njit

LOCAL_RIBOSOME_RANGES={
 'delivery_time_s':[1800.,7200.,28800.],
 'local40_maturation_s':[600.,1800.,7200.],
 'local60_maturation_s':[600.,3600.,14400.],
 'return_time_s':[3600.,21600.,86400.],
 'ATP_per_delivery':[100.,1000.,10000.],
 'ATP_per_return':[100.,1000.,10000.],
 'transport_support':[0.,.25,1.],
 'local40_support':[0.,.25,1.],
 'local60_support':[0.,.25,1.],
}

@dataclass(frozen=True)
class LocalRibosomeParameters:
 delivery_time_s:float=7200.
 local40_maturation_s:float=1800.
 local60_maturation_s:float=3600.
 return_time_s:float=21600.
 ATP_per_delivery:float=1000.
 ATP_per_return:float=1000.
 def vector(self):
  v=np.array(list(asdict(self).values()),float)
  if not np.isfinite(v).all() or np.any(v[:4]<=0) or np.any(v[4:]<0):raise ValueError('Positive times and nonnegative finite ATP costs required')
  return v

NAMES=tuple('local_'+x for x in (
 'pre40_Ceq','pre60_Ceq','mature40_Ceq','mature60_Ceq',
 'delivered40_Ceq','delivered60_Ceq','matured40_Ceq','matured60_Ceq',
 'returned40_Ceq','returned60_Ceq','delivery_ATP_soma_uM','return_ATP_process_uM'))

def controls(segments,supplied=None):
 sg=np.asarray(segments,float)
 a=np.c_[sg[:,:2],np.ones((len(sg),3))] if supplied is None else np.asarray(supplied,float)
 if a.shape!=(len(sg),5) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>1):raise ValueError('Local controls: start,end,delivery_support,late40_support,late60_support; supports0..1, same segment boundaries')
 return a.copy()

@njit
def extension(z,old,v,ctl,C,volume_ratio,ATP0s,ATP0p,KATP):
 out=np.zeros(574);out[:562]=old
 # Normalized ATP availability gates delivery at source and maturation/return
 # at destination. This is a scenario law, not a measured motor dose response.
 a=max(z[120],0.);b=max(z[136],0.)
 es=min(1.,a/(KATP+a)*(KATP+ATP0s)/ATP0s)
 ep=min(1.,b/(KATP+b)*(KATP+ATP0p)/ATP0p)
 send=np.maximum(z[390:392],0.)/v[0]*es*ctl[0]
 mature=np.maximum(z[562:564],0.)/v[1:3]*ep*ctl[1:3]
 back=np.maximum(z[564:566],0.)/v[3]*ep
 out[390:392]-=send
 out[330:332]+=back
 out[396:398]+=mature # global mature-production counters, soma + process
 out[562:564]=send-mature
 out[564:566]=mature-back
 out[566:568]=send;out[568:570]=mature;out[570:572]=back
 # C is uM in soma-equivalent amounts; convert process debit with Vs/Vp.
 cs=send.sum()*C*v[4];cp=back.sum()*C*volume_ratio*v[5]
 out[572]=cs;out[573]=cp
 out[120]-=cs;out[121]+=cs;out[125]+=cs
 out[136]-=cp;out[137]+=cp;out[150]+=cp
 return out

def adjusted(z,base):
 zz=z.copy();bb=base.copy()
 zz[:,390:392]+=zz[:,562:564];bb[390:392]+=bb[562:564]
 zz[:,330:332]+=zz[:,564:566];bb[330:332]+=bb[564:566]
 return zz,bb

def checks(z,base,v,C,volume_ratio):
 d=z-base;err=lambda x:float(np.max(abs(x)))
 out={'local_ribosomes_nonnegative':float(max(0.,-z[:,562:574].min()))}
 for j in range(2):
  out[f'local_pre{40+20*j}_balance']=err(d[:,562+j]-d[:,566+j]+d[:,568+j])
  out[f'local_mature{40+20*j}_balance']=err(d[:,564+j]-d[:,568+j]+d[:,570+j])
 out['local_delivery_ATP']=err(d[:,572]-d[:,566:568].sum(1)*C*v[4])
 out['local_return_ATP']=err(d[:,573]-d[:,570:572].sum(1)*C*volume_ratio*v[5])
 return out

def observe(z,C,volume_ratio):
 a=np.asarray(z,float)
 if a.ndim!=2 or a.shape[1]!=574:raise ValueError('Requires local-ribosome branch574')
 return dict(precursor_uM=a[:,562:564]*C*volume_ratio,
  mature_uM=a[:,564:566]*C*volume_ratio,
  paired_idle_capacity_uM=np.minimum(a[:,564],a[:,565])*C*volume_ratio,
  scope='Capacity observer only; no new local peptide production or inferred RP exchange')
