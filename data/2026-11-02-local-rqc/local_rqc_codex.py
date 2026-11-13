"""[codex]2026-11-02. Conservative effective local ribosome rescue and disposal.

ASCC/PELO are distinct mechanisms, not interchangeable measured rates. This
module resolves material stages, not collisions, ubiquitin chains or tRNA
repair. Default rescue GTP=0 is ATP-dependent ASCC-like bookkeeping; a GTP
cost is an alternative effective scenario, not an additional default debit.
No CAT tails, no inference that ordinary slowing is an irreversible stall.
"""
from dataclasses import dataclass,asdict
import numpy as np
from numba import njit

def growth_source_evidence():
 from pathlib import Path
 import json
 return json.loads((Path(__file__).resolve().parent/'growth_source_constraints_codex.json').read_text(encoding='utf8'))

LOCAL_RQC_RANGES=dict(stall_per_s=[0.,.0001,.001,.01],split_s=[10.,300.,1800.],
 release_s=[30.,300.,3600.],proteolysis_s=[60.,1800.,7200.],
 split_ATP_per_chain=[2.,10.,100.],release_ATP_equiv_per_chain=[2.,10.,100.],
 proteolysis_ATP_per_aa=[.05,.2,2.],rescue_GTP_per_chain=[0.,1.],
 reusable_message_fraction=[0.,.5,1.],stall_support=[0.,1.],
 split_support=[0.,.1,1.],release_support=[0.,.1,1.],proteolysis_support=[0.,.1,1.])

@dataclass(frozen=True)
class LocalRQCParameters:
 stall_per_s:float=.001
 split_s:float=300.
 release_s:float=300.
 proteolysis_s:float=1800.
 split_ATP_per_chain:float=10.
 release_ATP_equiv_per_chain:float=10.
 proteolysis_ATP_per_aa:float=.2
 rescue_GTP_per_chain:float=0.
 reusable_message_fraction:float=1.
 def vector(self):
  a=np.array(list(asdict(self).values()),float)
  if not np.isfinite(a).all() or min(a)<0 or min(a[1:4])<=0 or a[8]>1:raise ValueError('Finite nonnegative RQC values;positive times;message reuse0..1')
  return a

NAMES=tuple('local_RQC_'+s for s in (
 'stalled0_Ceq','stalled1_Ceq','stalled2_Ceq','stalled3_Ceq',
 'bound60S0_Ceq','bound60S1_Ceq','bound60S2_Ceq','bound60S3_Ceq',
 'released_chain_aa_uM','stalled_Ceq','split_Ceq','released_Ceq',
 'released_aa_uM','proteolyzed_aa_uM','split_ATP_uM','rescue_GTP_uM',
 'release_ATP_equiv_uM','proteolysis_ATP_uM','stalled_aa_uM','destroyed_mRNA_uM'))

def controls(sg,supplied=None):
 a=np.c_[sg[:,:2],np.zeros(len(sg)),np.ones((len(sg),3))] if supplied is None else np.asarray(supplied,float)
 if a.shape!=(len(sg),6) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or min(a[:,2:].ravel())<0 or max(a[:,2:].ravel())>1:raise ValueError('RQC schedule:start,end,stall,split,release,proteolysis supports0..1')
 return a.copy()

@njit
def extension(z,old,v,ctl,L,C,ratio,KATP,KGTP):
 out=np.zeros(640);out[:620]=old
 scale=C*ratio;a=max(z[136],0.);g=max(z[576],0.)
 en=a/(KATP+a);gg=g/(KGTP+g) if v[7]>0 else 1.
 stalled=np.maximum(z[620:624],0.);bound=np.maximum(z[624:628],0.)
 enter=v[0]*ctl[0]*np.maximum(z[578:582],0.);enter[0]=0. # unresolved first quarter excluded
 split=stalled/v[1]*en*gg*ctl[1]
 release=bound/v[2]*en*ctl[2]
 weights=np.arange(4)*L/4*scale
 aa=(weights*release).sum();lost=(weights*enter).sum()
 degrade=max(z[628],0.)/v[3]*en*ctl[3]
 ns=split.sum();nr=release.sum();ne=enter.sum()
 ca=v[4]*ns*scale;cr=v[5]*nr*scale;cd=v[6]*degrade;cg=v[7]*ns*scale
 out[578:582]-=enter
 out[620:624]=enter-split;out[624:628]=split-release;out[628]=aa-degrade
 out[629:640]=np.array([ne,ns,nr,aa,degrade,ca,cg,cr,cd,lost,(1-v[8])*ns*scale])
 out[564]+=ns;out[565]+=nr #40S first,60S after nascent-chain release
 out[607]+=v[8]*ns*scale # only explicitly reusable messages return
 out[584]+=degrade # existing peptide hydrolysis returns AA later
 out[576]-=cg;out[577]+=cg # GTP replenishment paid only by existing NDPK
 cost=ca+cr+cd
 out[136]-=cost;out[137]+=cost;out[150]+=cost
 return out

def adjusted_particles(z,base):
 zz=z.copy();bb=base.copy()
 zz[:,564]+=z[:,620:624].sum(1);bb[564]+=base[620:624].sum()
 zz[:,565]+=z[:,620:628].sum(1);bb[565]+=base[620:628].sum()
 return zz,bb

def revise_checks(ch,z,base,v,L,C,ratio):
 out=dict(ch);d=z-base;err=lambda a:float(np.max(abs(a)));scale=C*ratio
 s=d[:,620:624].sum(1);r=d[:,624:628].sum(1)
 nas=(d[:,578:582]+d[:,620:624]+d[:,624:628])@np.arange(4)*L/4*scale
 active_nas=d[:,578:582]@np.arange(4)*L/4*scale
 A=d[:,599]-d[:,600];G=d[:,601]-d[:,602]
 out.update(local_busy_balance=err(d[:,578:582].sum(1)+s-d[:,585]+d[:,586]+d[:,630]),
  local_nascent_balance=err(nas+L*scale*d[:,586]+d[:,632]-d[:,588]),
  local_peptide_balance=err(d[:,584]-L*d[:,596]-d[:,633]+d[:,597]),
  local_AA_balance=err(d[:,574]+d[:,575]+nas+L*d[:,583]+d[:,584]+d[:,628]-A),
  local_GTP_balance=err(d[:,576]-d[:,592]+d[:,593]+d[:,590]+d[:,591]+d[:,635]-G),
  local_energy_reserve_identity=err(d[:,594]+d[:,592]-d[:,593]-4*d[:,588]-d[:,591]-d[:,635]-2*d[:,575]-d[:,576]+G),
  mRNA_local_balance=err(d[:,607]+(d[:,578:582].sum(1)+s)*scale-ratio*d[:,612]+d[:,613]+d[:,639]),
  mRNA_global_balance=err((d[:,605]+d[:,606]-d[:,608]+d[:,609]+d[:,611])*ratio+d[:,607]+(d[:,578:582].sum(1)+s)*scale+d[:,613]+d[:,639]),
  local_RQC_stalled_balance=err(s-d[:,629]+d[:,630]),
  local_RQC_60S_balance=err(r-d[:,630]+d[:,631]),
  local_RQC_released_balance=err(d[:,628]-d[:,632]+d[:,633]),
  local_RQC_abort_mass=err(active_nas+L*scale*d[:,586]+d[:,638]-d[:,588]),
  local_RQC_split_cost=err(d[:,634]-v[4]*scale*d[:,630]),
  local_RQC_release_cost=err(d[:,636]-v[5]*scale*d[:,631]),
  local_RQC_GTP_cost=err(d[:,635]-v[7]*scale*d[:,630]),
  local_RQC_proteolysis_cost=err(d[:,637]-v[6]*d[:,633]),
  local_RQC_message_fate=err(d[:,639]-(1-v[8])*scale*d[:,630]),
  local_RQC_nonnegative=float(max(0.,-z[:,620:640].min())))
 return out

def observe(z,C,ratio):
 scale=C*ratio
 return dict(stalled80S_uM=z[:,620:624].sum(1)*scale,bound60S_uM=z[:,624:628].sum(1)*scale,
 released_chain_aa_uM=z[:,628],proteolyzed_aa_uM=z[:,633],
 direct_ATP_equiv_uM=z[:,634]+z[:,636]+z[:,637],rescue_GTP_uM=z[:,635],
 bound_message_uM=(z[:,578:582]+z[:,620:624]).sum(1)*scale,
 destroyed_message_uM=z[:,639],human_calibrated=False,
 scope='Effective eligible-stall scenarios;no collision detector/CAT tails/tRNA repair/full NGD. Direct energy excludes GTP recharge already counted by NDPK.')
