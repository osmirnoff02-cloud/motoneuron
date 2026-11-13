"""[codex] 2026-11-03. API0.42: selected rRNA material reuse with owned energy.

All material pools are micromolar nucleotide equivalents per soma volume.
Four bases are lumped: this is NOT a complete nucleotide metabolic network.
RNA-reserved activated precursors do not replenish ATP or translation GTP.
Nuclear products -> NMP; lysosomal products -> nucleosides -> NMP. Effective
net routes, not a claim that all nucleases or purines use the same chemistry.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,numpy as np
from numba import njit
RECIPE=json.loads((Path(__file__).parent/'rna_recipe_codex.json').read_text())
LENGTHS=np.array(RECIPE['lengths_nt'],float)
SPACERS=float(RECIPE['spacers_nt'])
SALVAGE_RANGES=dict(nuclear_hydrolysis_s=[300.,3600.,21600.],lysosomal_hydrolysis_s=[600.,7200.,86400.],
 export_s=[300.,3600.,21600.],nucleoside_activation_s=[300.,3600.,21600.],reserve_response_s=[300.,3600.,21600.],
 NMP0_uM_nt=[10.,100.,1000.],NTP_reserve0_uM_nt=[10.,100.,1000.],material_recovery_fraction=[0.,.5,1.],
 nucleoside_to_NMP_ATP_equiv=[1.,2.,4.],NMP_to_NTP_ATP_equiv=[2.],
 external_supply_support=[0.,.25,1.],nuclear_hydrolysis_support=[0.,1.],lysosomal_hydrolysis_support=[0.,1.],export_support=[0.,1.],nucleoside_activation_support=[0.,1.],NTP_activation_support=[0.,1.])

@dataclass(frozen=True)
class RNASalvageParameters:
 nuclear_hydrolysis_s:float=3600.
 lysosomal_hydrolysis_s:float=7200.
 export_s:float=3600.
 nucleoside_activation_s:float=3600.
 reserve_response_s:float=3600.
 NMP0_uM_nt:float=100.
 NTP_reserve0_uM_nt:float=100.
 material_recovery_fraction:float=.5
 nucleoside_to_NMP_ATP_equiv:float=1.
 NMP_to_NTP_ATP_equiv:float=2.
 def vector(self):
  v=np.array(list(asdict(self).values()),float)
  if not np.isfinite(v).all() or min(v[:7])<=0 or not 0<=v[7]<=1 or v[8]<1 or v[9]!=2:raise ValueError('Invalid RNA salvage parameters; two phosphorylation equivalents NMP->NTP required')
  return v

NAMES=tuple('rna_salvage_'+x for x in ('nuclear_debris_nt_uM','lysosomal_debris_nt_uM','lysosomal_nucleosides_nt_uM','cytosolic_nucleosides_nt_uM','NMP_nt_uM','RNA_reserved_NTP_nt_uM','nuclear_input_nt_uM','lysosomal_input_nt_uM','nuclear_hydrolyzed_nt_uM','lysosomal_hydrolyzed_nt_uM','exported_nucleosides_nt_uM','activated_nucleosides_nt_uM','activated_NMP_nt_uM','external_NMP_import_nt_uM','NMP_export_nt_uM','RNA_incorporated_nt_uM','ATP_spent_uM','legacy_RNA_ATP_replaced_uM','catabolized_nt_uM'))
assert len(NAMES)==19
def initial(v):
 q=np.zeros(19);q[4:6]=v[5:7];return q
def controls(sg,supplied=None):
 a=np.c_[sg[:,:2],np.ones((len(sg),6))] if supplied is None else np.asarray(supplied,float)
 if a.shape!=(len(sg),8) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>1):raise ValueError('RNA schedule: matching start,end and six supports0..1')
 return a.copy()

@njit
def gate(z,v):return min(1.,max(z[731],0.)/v[6])

@njit
def source_rates(old,b):
 # Actual derivatives of cumulative production/disposal, never history as stock.
 # tr47 - d(healthy47S) - damage47S = actual processing into mature branches.
 process=old[392]-old[384]-old[708]
 nuclear=b[0]*(max(process,0.)*SPACERS+np.dot(np.maximum(old[678:686],0.),LENGTHS))
 # Local particle counters are in soma-C equivalents; no second volume factor.
 lysosomal=b[0]*(LENGTHS[6]*(old[398]+old[669])+LENGTHS[7]*(old[399]+old[670]))
 used=b[0]*(LENGTHS[0]*old[392]+LENGTHS[3]*old[393])
 replaced=b[0]*(b[11]*old[392]+b[12]*old[393])
 return nuclear,lysosomal,used,replaced

@njit
def finish(z,old,v,ctl,b,KATP,ATP0,lysosome_support):
 out=np.zeros(745);out[:726]=old
 q=np.maximum(z[726:732],0.);nuc,lys,used,replaced=source_rates(old,b)
 energy=max(z[120],0.)/(KATP+max(z[120],0.))*(KATP+ATP0)/ATP0
 hydro_n=q[0]/v[0]*ctl[1]
 hydro_l=q[1]/v[1]*ctl[2]*min(1.,max(lysosome_support,0.))
 # Net export closure; no claim of one-way ENT3 or measured transport constants.
 export=q[2]/v[2]*ctl[3]
 phosphorylate=q[3]/v[3]*energy*ctl[4]
 j0=b[0]*np.log(2)/(b[1]*86400)*(LENGTHS[0]+LENGTHS[3])
 activate=(j0+max(v[6]-q[5],0.)/v[4])*min(1.,q[4]/v[5])*max(0.,2-q[5]/v[6])*energy*ctl[5]
 # External NMP boundary: no free de-novo synthesis asserted. Labelled import.
 incoming=max(0.,j0+(v[5]-q[4])/v[4])*ctl[0]
 outgoing=max(q[4]-v[5]-j0*v[4],0.)/v[4]
 recovered=v[7];lost=(1-recovered)*(hydro_n+hydro_l)
 out[726:732]=np.array([nuc-hydro_n,lys-hydro_l,recovered*hydro_l-export,export-phosphorylate,
  recovered*hydro_n+phosphorylate+incoming-outgoing-activate,activate-used])
 cost=v[8]*phosphorylate+v[9]*activate
 out[732:745]=np.array([nuc,lys,hydro_n,hydro_l,export,phosphorylate,activate,incoming,outgoing,used,cost,replaced,lost])
 # Replace legacy per-RNA lumped debit exactly once. No adjustment of historical
 # biogenesis counter400: it is explicitly cancelled by counter743 here.
 delta=cost-replaced;out[120]-=delta;out[121]+=delta;out[125]+=delta
 return out

def checks(z,base,v,b):
 d=z-base;q=d[:,726:745];err=lambda a:float(np.max(abs(a)))
 process=d[:,392]-d[:,384]-d[:,708]
 nuclear=b[0]*(process*SPACERS+d[:,678:686]@LENGTHS)
 lyso=b[0]*(LENGTHS[6]*(d[:,398]+d[:,669])+LENGTHS[7]*(d[:,399]+d[:,670]))
 used=b[0]*(LENGTHS[0]*d[:,392]+LENGTHS[3]*d[:,393])
 return dict(RNA_salvage_nonnegative=max(0.,float(-z[:,726:745].min())),
  RNA_nuclear_debris=err(q[:,0]-q[:,6]+q[:,8]),RNA_lyso_debris=err(q[:,1]-q[:,7]+q[:,9]),
  RNA_lyso_nucleosides=err(q[:,2]-v[7]*q[:,9]+q[:,10]),RNA_cytosol_nucleosides=err(q[:,3]-q[:,10]+q[:,11]),
  RNA_NMP=err(q[:,4]-v[7]*q[:,8]-q[:,11]-q[:,13]+q[:,14]+q[:,12]),
  RNA_NTP=err(q[:,5]-q[:,12]+q[:,15]),RNA_source_nuclear=err(q[:,6]-nuclear),RNA_source_lyso=err(q[:,7]-lyso),
  RNA_incorporation=err(q[:,15]-used),RNA_energy=err(q[:,16]-v[8]*q[:,11]-v[9]*q[:,12]),
  RNA_legacy_debit_removed=err(q[:,17]-b[0]*(b[11]*d[:,392]+b[12]*d[:,393])),
  RNA_catabolism=err(q[:,18]-(1-v[7])*(q[:,8]+q[:,9])),
  RNA_open_material_balance=err(q[:,:6].sum(1)+q[:,15]+q[:,14]+q[:,18]-q[:,13]-q[:,6]-q[:,7]))

def observe(z):
 return dict(pools_nt_uM=z[:,726:732],counters_nt_or_ATP_uM=z[:,732:745],pool_names=NAMES[:6],counter_names=NAMES[6:],
  human_calibrated=False,scope='Selected rRNA only; four bases pooled in nucleotide equivalents per soma volume; RefSeq inventory proxy. RNA reserve is separate from ATP/translation GTP. Nuclear NMP and lysosomal nucleoside routes effective, not full purine/modified-nucleotide chemistry. External NMP boundary is imported material, not de-novo synthesis. No functional benefit or K-drift correction implied.')
