"""[codex]2026-11-02. Resource-owned excess-precursor disposal, API0.40.
Capacity feedback and human kinetic constants are hypotheses. No RNA->AA,
no ATP refund, no forced reset. Ordinary throughput buffers may stay elevated.
"""
from pathlib import Path
from dataclasses import dataclass,asdict
import json
import numpy as np
from numba import njit
B=Path(__file__).resolve().parent
RECIPE=json.loads((B/'human_RP_recipe.json').read_text())
RECIPE_VECTOR=np.array([[RECIPE['recipe'][s][k] for k in ('translated_aa','bound_aa','coproduct_aa','chains')] for s in ('40S','60S')],float)
SURPLUS_RANGES=dict(buffer_multiple=[1.1,2.,4.],clearance_s=[600.,7200.,86400.],feedback_gain=[0.,1.,4.],
 protein_disassembly_ATP_per_aa=[0.,.2,2.],RNA_disposal_ATP_per_equivalent=[0.,100.,1000.],
 coproduct_half_hours=[1.,24.,72.],cleanup_support=[0.,1.],feedback_support=[0.,1.])
LABEL_OBSERVER_RANGES=dict(arrival_h=[.1,2.,24.],precursor_loss_h=[.1,4.,24.],bound_loss_h=[1.,24.,168.],heavy_input_per_h=[.001,.01,.1])

@dataclass(frozen=True)
class RibosomeSurplusParameters:
 buffer_multiple:float=2.
 clearance_s:float=7200.
 feedback_gain:float=1.
 protein_disassembly_ATP_per_aa:float=.2
 RNA_disposal_ATP_per_equivalent:float=100.
 coproduct_half_hours:float=24.
 def vector(self):
  v=np.array(list(asdict(self).values()),float)
  if not np.isfinite(v).all() or v[0]<1 or min(v[[1,5]])<=0 or min(v[2:5])<0:raise ValueError('Invalid surplus parameters')
  return v

NAMES=tuple('surplus_disposed_'+s+'_Ceq' for s in ('47S','small_rRNA','large_rRNA','5S','RP40','RP60','pre40','pre60'))+('fusion_coproduct40_residues_uM','fusion_coproduct60_residues_uM','fusion_coproduct40_to_peptide_uM','fusion_coproduct60_to_peptide_uM','surplus_RNA_disposal_ATP_uM','surplus_protein_disposal_ATP_uM')

def controls(sg,supplied=None):
 a=np.c_[sg[:,:2],np.ones((len(sg),2))] if supplied is None else np.asarray(supplied,float)
 if a.shape!=(len(sg),4) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>1):raise ValueError('Surplus schedule:start,end,cleanup,feedback; supports0..1')
 return a.copy()

@njit
def modulate(z,base,v,ctl,bc,sc):
 b=bc.copy();s=sc.copy();ex=np.maximum(z[384:392]/base[384:392]-v[0],0.)
 # Joint47S precursor can form both subunits. Do not invent separate47S sources.
 b[0]/=1+ctl[1]*v[2]*max(ex[0],ex[1],ex[2],ex[6],ex[7])
 s[6]/=1+ctl[1]*v[2]*max(ex[3],ex[7])
 s[0]/=1+ctl[1]*v[2]*max(ex[4],ex[6])
 s[1]/=1+ctl[1]*v[2]*max(ex[5],ex[7])
 return b,s

@njit
def finish(z,old,base,v,ctl,recipe,C,KATP,proteasome_support):
 out=np.zeros(692);out[:678]=old
 energy=max(z[120],0.)/(KATP+max(z[120],0.))
 loss=np.maximum(z[384:392]-v[0]*base[384:392],0.)/v[1]*ctl[0]*energy
 # Protein-bearing precursors require available proteolysis, RNA-only do not.
 loss[4:]*=proteasome_support
 out[384:392]-=loss;out[678:686]=loss
 prot=C*np.array([recipe[0,1]*(loss[4]+loss[6]),recipe[1,1]*(loss[5]+loss[7])])
 copro=C*np.maximum(old[394:396],0.)*recipe[:,2]
 release=np.maximum(z[686:688],0.)*np.log(2)/(v[5]*3600)*energy*proteasome_support
 out[686:688]=copro-release;out[688:690]=release
 out[513]+=prot.sum()+release.sum();out[516]+=prot.sum()+release.sum()
 out[526:528]+=prot+release
 # RNA counters are precursor equivalents, not nucleotide/ATP regeneration.
 cr=C*v[4]*(loss[:4].sum()+loss[6:].sum());cp=v[3]*(prot.sum()+release.sum())
 out[690:692]=np.array([cr,cp]);cost=cr+cp
 out[120]-=cost;out[121]+=cost;out[125]+=cost
 return out

def adjusted(z,base):
 zz=z.copy();bb=base.copy()
 zz[:,384:392]+=z[:,678:686];bb[384:392]+=base[678:686]
 return zz,bb

def checks(z,base,v,C):
 d=z-base;err=lambda x:float(np.max(abs(x)));cop=C*d[:,394:396]*RECIPE_VECTOR[:,2]
 prot=C*np.c_[d[:,682]+d[:,684],d[:,683]+d[:,685]]*RECIPE_VECTOR[:,1]
 # Conservation of fusion mass remains separate from complete bound RP mass.
 return dict(surplus_nonnegative=max(0.,float(-z[:,678:].min())),
  fusion40_residue_balance=err(d[:,686]+d[:,688]-cop[:,0]),
  fusion60_residue_balance=err(d[:,687]+d[:,689]-cop[:,1]),
  surplus_protein_energy=err(d[:,691]-v[3]*(prot.sum(1)+d[:,688:690].sum(1))),
  surplus_RNA_energy=err(d[:,690]-C*v[4]*(d[:,678:682].sum(1)+d[:,684:686].sum(1))))

def observe(z,base):
 return dict(precursor_relative=z[:,384:392]/base[384:392],disposed_Ceq=z[:,678:686],
  coproduct_residues_uM=z[:,686:688],direct_ATP_uM=z[:,690:692].sum(1),recipe=RECIPE['recipe'],human_kinetics_calibrated=False,
  scope='Finite selected precursor buffers; effective excess-triggered surveillance, not identification of defective particles. Ub/FUBI coproduct mass is separate, not a ubiquitin-signalling model. No RNA nucleotide salvage; full transcription/assembly-factor costs unresolved.')

def heavy_fraction(heavy,light):
 h,l=np.broadcast_arrays(np.asarray(heavy,float),np.asarray(light,float))
 if not np.isfinite(h+l).all() or np.any(h<0) or np.any(l<0) or np.any(h+l<=0):raise ValueError('Nonnegative signals with positive total required')
 return h/(h+l)

def pulse_chase_observer(time_h,*,pulse_h=1.,arrival_h=2.,precursor_loss_h=4.,bound_loss_h=24.,heavy_input_per_h=.01):
 """Separate source-system tracer observer; not a fit of human turnover.
 Labels are normalized residue/protein equivalents. Bound total set1;
 precursor label can keep entering after heavy medium is removed.
 """
 from scipy.integrate import solve_ivp
 t=np.asarray(time_h,float)
 if t.ndim!=1 or len(t)<2 or t[0]!=0 or np.any(np.diff(t)<=0) or min(pulse_h,arrival_h,precursor_loss_h,bound_loss_h,heavy_input_per_h)<=0:raise ValueError('Increasing times from0 and positive parameters')
 state=np.zeros(2);states=np.zeros((len(t),2))
 for a,b in zip(sorted(set([0.,min(pulse_h,t[-1]),t[-1]]))[:-1],sorted(set([0.,min(pulse_h,t[-1]),t[-1]]))[1:]):
  def rhs(_,y):return [heavy_input_per_h*(a<pulse_h)-y[0]/arrival_h-y[0]/precursor_loss_h,y[0]/arrival_h-y[1]/bound_loss_h]
  r=solve_ivp(rhs,(a,b),state,dense_output=True,rtol=1e-10,atol=1e-12);mask=(t>=a)&(t<=b);states[mask]=r.sol(t[mask]).T;state=r.y[:,-1]
 if states[:,1].max()>1:raise ValueError('Tracer input exceeds normalized bound total')
 return dict(precursor_heavy=states[:,0],bound_heavy_fraction=states[:,1],human_fit=False)
