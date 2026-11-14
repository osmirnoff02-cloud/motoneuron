"""[codex] 2026-11-03. API0.41: precursor state != queue length.

Healthy pools stay in legacy productive states384:392. Waiting/excess are a
non-overlapping observation of those same pools, not extra material. Damage
is an independently imposed susceptibility/input, NEVER a function of queue
age, assembly delay or ATP shortage. Defective pools cannot mature. Recognition
and degradation are separate finite steps. No repair, nucleotide salvage or
specific HUWE1/exosome kinetics are asserted.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit

KINDS=('47S','small_rRNA','large_rRNA','5S','RP40','RP60','pre40','pre60')
QUALITY_RANGES=dict(damage_per_h=[.001,.03,.3],recognition_s=[300.,3600.,21600.],
 RNA_clearance_s=[300.,7200.,86400.],protein_clearance_s=[300.,7200.,86400.],
 RNA_recognition_ATP_per_equivalent=[0.,10.,100.],protein_recognition_ATP_per_aa=[0.,.01,.1],
 imposed_RNA_damage_support=[0.,1.],imposed_RP_damage_support=[0.,1.],imposed_RNP_damage_support=[0.,1.],
 recognition_support=[0.,1.],RNA_clearance_support=[0.,1.],protein_clearance_support=[0.,1.],orphan_clearance_support=[0.,1.])

@dataclass(frozen=True)
class PrecursorQualityParameters:
 damage_per_h:float=.03
 recognition_s:float=3600.
 RNA_clearance_s:float=7200.
 protein_clearance_s:float=7200.
 RNA_recognition_ATP_per_equivalent:float=10.
 protein_recognition_ATP_per_aa:float=.01
 def vector(self):
  v=np.array(list(asdict(self).values()),float)
  if not np.isfinite(v).all() or min(v)<0 or min(v[1:4])<=0:raise ValueError('Invalid precursor quality parameters')
  return v

NAMES=tuple('precursor_'+stage+'_'+k+'_Ceq' for stage in ('defective_unrecognized','defective_recognized','damage_cumulative','defective_disposed') for k in KINDS)+('precursor_recognition_RNA_ATP_uM','precursor_recognition_protein_ATP_uM')

def controls(sg,supplied=None):
 # start,end, imposed RNA/RP/RNP damage, recognition,RNA disposal,protein disposal,orphan disposal
 a=np.c_[sg[:,:2],np.tile([0.,0.,0.,1.,1.,1.,1.],(len(sg),1))] if supplied is None else np.asarray(supplied,float)
 if a.shape!=(len(sg),9) or not np.isfinite(a).all() or not np.array_equal(a[:,:2],sg[:,:2]) or np.any(a[:,2:]<0) or np.any(a[:,2:]>1):raise ValueError('Quality schedule: matching start/end plus seven supports 0..1')
 return a.copy()

@njit
def fluxes(z,base,v,ctl,sv,sctl,recipe,C,KATP,proteasome_support):
 energy=max(z[120],0.)/(KATP+max(z[120],0.))
 H=np.maximum(z[384:392],0.);D=np.maximum(z[692:700],0.);Q=np.maximum(z[700:708],0.)
 injury=np.empty(8);injury[:4]=ctl[0];injury[4:6]=ctl[1];injury[6:]=ctl[2]
 damage=H*v[0]/3600*injury
 recognize=D/v[1]*ctl[3]*energy
 disposal=np.zeros(8)
 disposal[:4]=Q[:4]/v[2]*ctl[4]*energy
 disposal[4:6]=Q[4:6]/v[3]*ctl[5]*energy*proteasome_support
 # Effective disassembly of a mixed RNP requires both handling branches.
 disposal[6:]=Q[6:]/max(v[2],v[3])*min(ctl[4],ctl[5])*energy*proteasome_support
 orphan=np.zeros(8)
 # Only excess unassembled protein is an ERISQ-like substrate. Healthy RNA/RNP
 # is never discarded for waiting or abundance alone.
 orphan[4:6]=np.maximum(H[4:6]-sv[0]*base[388:390],0.)/sv[1]*sctl[0]*ctl[6]*energy*proteasome_support
 prot_mark=C*(recipe[0,1]*(recognize[4]+recognize[6])+recipe[1,1]*(recognize[5]+recognize[7]))
 rna_mark=C*(recognize[:4].sum()+recognize[6:].sum())
 return damage,recognize,disposal,orphan,np.array([v[4]*rna_mark,v[5]*prot_mark])

@njit
def finish(z,old,base,v,ctl,sv,sctl,recipe,C,KATP,proteasome_support):
 # old is the API0.40 derivative with legacy excess cleanup disabled, but
 # fusion coproduct processing and production-feedback remain active.
 out=np.zeros(726);out[:692]=old
 damage,recognize,disposal,orphan,cost=fluxes(z,base,v,ctl,sv,sctl,recipe,C,KATP,proteasome_support)
 out[384:392]-=damage+orphan
 out[692:700]=damage-recognize;out[700:708]=recognize-disposal
 out[708:716]=damage;out[716:724]=disposal
 total=disposal+orphan
 out[678:686]+=total
 prot=C*np.array([recipe[0,1]*(total[4]+total[6]),recipe[1,1]*(total[5]+total[7])])
 out[513]+=prot.sum();out[516]+=prot.sum();out[526:528]+=prot
 cr=C*sv[4]*(total[:4].sum()+total[6:].sum());cp=sv[3]*prot.sum()
 out[690]+=cr;out[691]+=cp;out[724:726]=cost
 paid=cr+cp+cost.sum();out[120]-=paid;out[121]+=paid;out[125]+=paid
 return out

def adjusted(z,base):
 zz=z.copy();bb=base.copy()
 zz[:,384:392]+=z[:,692:700]+z[:,700:708]
 bb[384:392]+=base[692:700]+base[700:708]
 return zz,bb

def checks(z,base,v,C,recipe):
 d=z-base;err=lambda x:float(np.max(np.abs(x)))
 orphan=d[:,678:686]-d[:,716:724]
 mark=d[:,700:708]+d[:,716:724]
 pr=C*(recipe[0,1]*(mark[:,4]+mark[:,6])+recipe[1,1]*(mark[:,5]+mark[:,7]))
 rr=C*(mark[:,:4].sum(1)+mark[:,6:].sum(1))
 return dict(precursor_quality_nonnegative=max(0.,float(-z[:,692:].min())),
  precursor_defect_balance=err(d[:,692:700]+d[:,700:708]+d[:,716:724]-d[:,708:716]),
  healthy_RNA_RNP_not_disposed_for_excess=err(orphan[:,[0,1,2,3,6,7]]),
  orphan_disposal_nonnegative=max(0.,float(-orphan.min())),
  precursor_RNA_recognition_energy=err(d[:,724]-v[4]*rr),
  precursor_protein_recognition_energy=err(d[:,725]-v[5]*pr))

def observe(z,base,sv):
 healthy=z[:,384:392];cap=sv[0]*base[384:392]
 return dict(kinds=KINDS,healthy_waiting_Ceq=np.minimum(healthy,cap),healthy_excess_Ceq=np.maximum(healthy-cap,0.),
  defective_unrecognized_Ceq=z[:,692:700],defective_recognized_Ceq=z[:,700:708],
  damaged_cumulative_Ceq=z[:,708:716],defective_disposed_Ceq=z[:,716:724],
  healthy_RP_disposed_Ceq=z[:,678:686]-z[:,716:724],recognition_ATP_uM=z[:,724:726],
  human_kinetics_calibrated=False,scope='Waiting/excess partition is observational, not a measured damage classifier. Imposed injury independent of delay. Defective material is nonproductive and irreversible in this module. No specific enzyme identity, rRNA repair or nucleotide salvage.')
