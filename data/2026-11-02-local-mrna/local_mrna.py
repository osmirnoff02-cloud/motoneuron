"""[codex] 2026-11-02. GAP43 cargo, conserved mRNA transport and local translation.

Human canonical GAP43 sequence P17677:238aa. Mechanistic cargo support from
mouse motor neurons is not a human kinetic calibration. Free local mRNA is
bound at initiation and released at completion; one ribosome/message, no
polysomes, decay of bound messages or somatic synthesis of this cargo.
"""
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import sys
import numpy as np
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'2026-11-02-local-translation'))
from local_translation import LocalTranslationParameters,extension as translate

MRNA_RANGES=dict(free_local_reference_uM=[.00001,.0001,.001],departure_s=[600.,3600.,21600.],
 arrival_s=[300.,1800.,7200.],soma_half_h=[1.,6.,24.],transit_half_h=[1.,6.,24.],local_half_h=[1.,6.,24.],
 transcription_ATP_equiv_per_message=[1000.,3000.,10000.],departure_ATP_per_message=[100.,1000.,10000.],
 arrival_ATP_per_message=[100.,1000.,10000.],production_support=[0.,.25,1.],departure_support=[0.,.25,1.],
 arrival_support=[0.,.25,1.],local_decay_support=[.5,1.,2.])

@dataclass(frozen=True)
class LocalMRNAParameters:
 free_local_reference_uM:float=.0001
 departure_s:float=3600.
 arrival_s:float=1800.
 soma_half_h:float=6.
 transit_half_h:float=6.
 local_half_h:float=6.
 transcription_ATP_equiv_per_message:float=3000.
 departure_ATP_per_message:float=1000.
 arrival_ATP_per_message:float=1000.
 def vector(self):
  a=np.array(list(asdict(self).values()),float)
  if not np.isfinite(a).all() or min(a[:6])<=0 or min(a[6:])<0:raise ValueError('mRNA requires positive pools/times and finite nonnegative costs')
  return a

NAMES=tuple('GAP43_mRNA_'+s for s in ('soma_uM','transit_soma_uM','free_local_uM','produced_soma_uM',
 'decayed_soma_uM','departed_soma_uM','decayed_transit_soma_uM','arrived_soma_uM','decayed_local_uM',
 'transcription_ATP_soma_uM','departure_ATP_soma_uM','arrival_ATP_local_uM'))

def cargo_parameters():return replace(LocalTranslationParameters(),length_aa=238.,initial_local_fraction=0.)

def initialize(base,v,rv,b,w,ratio,KATP):
 """Stationary selected molecular pools; replaces empty-start initialization.

No derivative subtraction or state freezing. NDPK capacity is derived from
basal demand, like the pre-existing somatic substrate module; it is not fitted.
ATP supply is not retuned here: full neuron electrical initial transients remain.
"""
 if v[0]!=238. or v[5]!=0:raise ValueError('GAP43 branch requires238aa and stationary, not redistributed, initial pools')
 z=np.r_[base,np.zeros(12)];v=v.copy();scale=b[0]*ratio
 # Mature return restores the portion of somatic maturation diverted locally.
 birth=np.log(2)/(b[1]*86400)
 z[390:392]=birth/(1/b[4:6]+1/rv[0])
 send=z[390:392]/rv[0]
 z[562:564]=send*rv[1:3];z[564:566]=send*rv[3]
 A,Q,G,D=z[497:501];ATP=z[136];ADP=z[137];energy=ATP/(KATP+ATP)
 z[574]=A;z[576:578]=[G,D]
 load=min(min(z[564:566]),w[0]/scale)/v[2]*G/(v[14]+G)
 inc=v[0]*load*scale
 capacity=v[11]*A/(v[12]+A)*energy
 if inc>=capacity:raise ValueError('Basal charging capacity cannot sustain specified free mRNA; reduce message pool or raise charging capacity')
 Q=v[7]*(1-inc/capacity);z[575]=Q
 step=4*v[1]/v[0]*Q/(v[13]+Q)*G/(v[14]+G)
 z[578:582]=load/step;z[582]=load*v[3]/energy
 z[583]=load*scale/(np.log(2)/(v[21]*3600)*energy)
 z[584]=inc*v[22]
 force=ATP*D-ADP*G/v[18]
 if force<=0:raise ValueError('Basal nucleotide ratios cannot drive NDPK')
 v[17]=(2*inc+v[19]*load*scale)/force
 kd=np.log(2)/(w[3:6]*3600)
 z[607]=w[0];z[606]=kd[2]*w[0]/ratio*w[2]
 z[605]=z[606]*(1/w[2]+kd[1])*w[1]
 production=z[605]*(1/w[1]+kd[0])
 ref=np.array([production,z[120],z[136]])
 return z,v,ref

def controls(sg,supplied=None):
 a=np.c_[sg[:,:2],np.ones((len(sg),4))] if supplied is None else np.asarray(supplied,float)
 if a.shape!=(len(sg),6) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>2):raise ValueError('mRNA schedule:start,end,production,departure,arrival,local decay supports0..2')
 return a.copy()

@njit
def extension(z,old,v,ctl,w,mc,ref,C,ratio,KATP):
 scale=C*ratio;vv=v.copy();tc=ctl.copy()
 # Slots are now free+bound messages; ctl[0] gates free message engagement.
 vv[4]=max(z[607],0.)*ctl[0]+np.maximum(z[578:582],0.).sum()*scale;tc[0]=1.
 d=translate(z,old,vv,tc,C,ratio,KATP);out=np.zeros(617);out[:605]=d
 s,t,l=np.maximum(z[605:608],0.);kd=np.log(2)/(w[3:6]*3600)
 a=max(z[120],0.);p=max(z[136],0.)
 es=min(1.,a/(KATP+a)*(KATP+ref[1])/ref[1]);ep=min(1.,p/(KATP+p)*(KATP+ref[2])/ref[2])
 prod=ref[0]*mc[0]*es;depart=s/w[1]*mc[1]*es;arrive=t/w[2]*mc[2]*ep
 ds=kd[0]*s;dt=kd[1]*t;dl=kd[2]*l*mc[3]
 out[605:608]=np.array([prod-ds-depart,depart-dt-arrive,arrive*ratio-dl+(d[586]-d[585])*scale])
 out[608:614]=np.array([prod,ds,depart,dt,arrive,dl])
 out[614:617]=np.array([prod*w[6],depart*w[7],arrive*ratio*w[8]])
 sc=out[614]+out[615];pc=out[616]
 out[120]-=sc;out[121]+=sc;out[125]+=sc
 out[136]-=pc;out[137]+=pc;out[150]+=pc
 return out

def checks(z,base,w,C,ratio):
 d=z-base;err=lambda a:float(np.max(abs(a)))
 bound=d[:,578:582].sum(1)*C*ratio
 return dict(mRNA_nonnegative=float(max(0.,-z[:,605:].min())),
  mRNA_soma_balance=err(d[:,605]-d[:,608]+d[:,609]+d[:,610]),
  mRNA_transit_balance=err(d[:,606]-d[:,610]+d[:,611]+d[:,612]),
  mRNA_local_balance=err(d[:,607]+bound-ratio*d[:,612]+d[:,613]),
  mRNA_global_balance=err((d[:,605]+d[:,606]-d[:,608]+d[:,609]+d[:,611])*ratio+d[:,607]+bound+d[:,613]),
  mRNA_transcription_cost=err(d[:,614]-w[6]*d[:,608]),
  mRNA_departure_cost=err(d[:,615]-w[7]*d[:,610]),
  mRNA_arrival_cost=err(d[:,616]-w[8]*ratio*d[:,612]))

def observe(z,C,ratio):
 return dict(gene='GAP43',protein_accession='P17677',length_aa=238,soma_mRNA_uM=z[:,605],
  transit_mRNA_soma_uM=z[:,606],free_local_mRNA_uM=z[:,607],bound_local_mRNA_uM=z[:,578:582].sum(1)*C*ratio,
  protein_uM=z[:,583],scope='One locally translated cargo; no calibrated GAP43-to-regrowth/NMJ/disease effect')
