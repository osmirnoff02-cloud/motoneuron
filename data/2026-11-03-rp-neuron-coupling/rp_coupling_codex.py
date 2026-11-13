"""[codex]2026-11-03. Conservative per-gene resolution of existing RP kit pools.

80 free, 80 defective and 80 recognized RP inventories. Old kit states are
mass-weighted views of the SAME material, not additional proteins. The inherited
two translation classes own nascent chains, AA/ATP/GTP and completed kit supply.
This is not 80 independent mRNAs/codon queues or an all-RP bound-exchange model.
Frankenfield whole-cell turnover anchors only a labelled FREE-loss sensitivity
scenario; it is not a measurement of the free pool's degradation rate.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from numba import njit

ROOT=Path(__file__).resolve().parent.parent
DATA=json.loads((ROOT/'2026-11-03-individual-ribosomal-proteins/protein_inputs_codex.json').read_text())
P=DATA['proteins'];GENES=tuple(q['gene'] for q in P)
SUB=np.array([0 if q['subunit']=='40S' else 1 for q in P],np.int64)
COPIES=np.array([q['copies'] for q in P],float)
LEN=np.array([sum(q['aa_bound']) for q in P],float)
HALF=np.array([q['motor_days'] if q['use_as_total_turnover_anchor'] else np.nan for q in P])
MASS=np.array([sum(COPIES[SUB==s]*LEN[SUB==s]) for s in range(2)])
F=slice(1600,1680);D=slice(1680,1760);Q=slice(1760,1840)
MADE=slice(1840,1920);ASSEMBLED=slice(1920,2000);REMOVED=slice(2000,2080)
EXTRA=slice(2080,2160);N=2160
NAMES=tuple('individual_RP_'+kind+'_'+g for kind in
 ('free_uM','defective_uM','recognized_uM','made_uM','assembled_uM','removed_uM','extra_free_removed_uM') for g in GENES)
RANGES=dict(free_loss_from_total_multiplier=[0.,1.,3.,10.],missing_half_days=[2.,8.,32.],
 selective_loss_multiplier=[0.,1.,3.,10.])

@dataclass(frozen=True)
class IndividualRPParameters:
 free_loss_from_total_multiplier:float=1.
 missing_half_days:float=8.
 selective_loss_gene:str='RPS26'
 selective_loss_multiplier:float=1.
 def rates(self):
  vals=[self.free_loss_from_total_multiplier,self.missing_half_days,self.selective_loss_multiplier]
  if not np.isfinite(vals).all() or min(vals)<0 or self.missing_half_days<=0 or self.selective_loss_gene not in GENES:
   raise ValueError('Invalid individual-RP scenario')
  k=np.log(2)/(86400*np.where(np.isfinite(HALF),HALF,self.missing_half_days))*self.free_loss_from_total_multiplier
  k[GENES.index(self.selective_loss_gene)]*=self.selective_loss_multiplier
  return k

def initialize(base,C):
 if len(base)!=1600:raise ValueError('Individual RP requires coupled API0.44 topology')
 z=np.r_[base,np.zeros(N-1600)]
 for block,offset in ((F,388),(D,696),(Q,704)):
  z[block]=C*COPIES*base[offset+SUB]
 return z

@njit
def assembly_view(z,C):
 """Only the biogenesis assembly reaction reads the limiting component."""
 zz=z.copy()
 for s in range(2):
  limit=1e100
  for i in range(80):
   if SUB[i]==s:limit=min(limit,max(z[F][i],0.)/(C*COPIES[i]))
  zz[388+s]=limit
 return zz

@njit
def finish(z,old,C,rates,assembly,damage,recognition,disposal,orphan,energy,proteasome_support,cost_per_aa):
 out=np.zeros(N);out[:1600]=old
 f=np.maximum(z[F],0.);d=np.maximum(z[D],0.);q=np.maximum(z[Q],0.)
 made=C*COPIES*np.maximum(old[394:396],0.)[SUB]
 built=C*COPIES*np.maximum(assembly,0.)[SUB]
 df=np.zeros(80);dd=np.zeros(80);dq=np.zeros(80);removed=np.zeros(80)
 extra=rates*f*energy*proteasome_support
 for s in range(2):
  fm=0.;dm=0.;qm=0.
  for i in range(80):
   if SUB[i]==s:fm+=f[i]*LEN[i];dm+=d[i]*LEN[i];qm+=q[i]*LEN[i]
  for i in range(80):
   if SUB[i]!=s:continue
   hurt=damage[4+s]*C*MASS[s]*f[i]/fm if fm>0 else 0.
   mark=recognition[4+s]*C*MASS[s]*d[i]/dm if dm>0 else 0.
   clear=disposal[4+s]*C*MASS[s]*q[i]/qm if qm>0 else 0.
   excess=orphan[4+s]*C*MASS[s]*f[i]/fm if fm>0 else 0.
   df[i]=made[i]-built[i]-hurt-excess-extra[i]
   dd[i]=hurt-mark;dq[i]=mark-clear;removed[i]=clear+excess+extra[i]
  dmF=0.;dmD=0.;dmQ=0.;additional=0.
  for i in range(80):
   if SUB[i]==s:
    dmF+=df[i]*LEN[i];dmD+=dd[i]*LEN[i];dmQ+=dq[i]*LEN[i];additional+=extra[i]*LEN[i]
  out[388+s]=dmF/(C*MASS[s]);out[696+s]=dmD/(C*MASS[s]);out[704+s]=dmQ/(C*MASS[s])
  # Existing disposal, peptide and energy ledgers are the unique destination.
  out[682+s]+=additional/(C*MASS[s]);out[526+s]+=additional
  out[513]+=additional;out[516]+=additional
  debit=additional*cost_per_aa
  out[691]+=debit;out[120]-=debit;out[121]+=debit;out[125]+=debit
 out[F]=df;out[D]=dd;out[Q]=dq
 out[MADE]=made;out[ASSEMBLED]=built;out[REMOVED]=removed;out[EXTRA]=extra
 return out

def checks(z,base,C):
 delta=z-base;err=lambda a:float(np.max(np.abs(a)))
 result=dict(individual_RP_nonnegative=max(0.,float(-z[:,1600:].min())),
  individual_RP_each_lineage=err(delta[:,F]+delta[:,D]+delta[:,Q]+delta[:,ASSEMBLED]+delta[:,REMOVED]-delta[:,MADE]))
 for sl,off in ((F,388),(D,696),(Q,704)):
  for s in range(2):
   result[f'individual_RP_projection_{off+s}']=err(z[:,off+s]-z[:,sl][:,SUB==s]@LEN[SUB==s]/(C*MASS[s]))
 for s in range(2):
  result[f'individual_RP_paid_synthesis_{s}']=err(delta[:,MADE][:,SUB==s]-C*delta[:,394+s,None]*COPIES[SUB==s])
 return result

def observe(z,C,parameters):
 limiting=np.column_stack([
  np.min(z[:,F][:,SUB==s]/COPIES[SUB==s],axis=1)/C for s in range(2)])
 return dict(genes=GENES,free_uM=z[:,F],defective_uM=z[:,D],recognized_uM=z[:,Q],
  produced_uM=z[:,MADE],assembled_cumulative_uM=z[:,ASSEMBLED],removed_uM=z[:,REMOVED],
  extra_free_removed_uM=z[:,EXTRA],limiting_kits_Ceq=limiting,measured_total_half_days=HALF,
  parameters=asdict(parameters),human_free_pool_calibrated=False,
  scope='80 unassembled RP inventories share original 2 translation classes and common soma AA/tRNA/GTP/ATP; limiting component gates assembly. Old pools are mass projections, not extra material. Whole-cell human dSILAC anchors a free-loss sensitivity assumption only. Bound turnover remains inherited particle retirement/local RPS26 branch; no 80 separate mRNAs, bound-exchange rates or codon queues. No new ATP supply or duplicate synthesis debit.')
