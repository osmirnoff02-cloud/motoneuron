"""[codex]2026-11-02. Selected local RP exchange and particle retirement.

RPS26/eS26 canonical precursor115aa, UniProtP62854. Functional impairment,
damage selectivity and all human rates are hypotheses, not Fusco fits.
Only free subunits are selected; busy GAP43 particles first finish/RQC.
rRNA disposal is recorded in subunit equivalents, never credited as AA/GTP.
"""
from dataclasses import dataclass,asdict
import numpy as np
from numba import njit

MAINTENANCE_RANGES=dict(damage_per_s=[0.,1e-5,1e-4],exchange_s=[300.,3600.,14400.],
 retirement_half_h=[6.,48.,192.],impaired_retirement_gain=[1.,4.,10.],
 hydrolysis_s=[300.,3600.,14400.],export_s=[60.,900.,7200.],
 RPS26_initial_uM=[0.,.001,.01],RPS26_half_uM=[.0001,.001,.01],
 RPS26_message_uM=[.00001,.0001,.001],RPS26_orphan_half_h=[1.,6.,24.],
 exchange_ATP_per_RP=[0.,10.,100.],sorting_ATP_per_subunit=[0.,100.,1000.],
 hydrolysis_ATP_equiv_per_aa=[0.,.02,.2],export_ATP_equiv_per_aa=[0.,.05,.5],
 damage_support=[0.,1.],exchange_support=[0.,.25,1.],retirement_support=[0.,.25,1.],
 hydrolysis_support=[0.,.25,1.],export_support=[0.,.25,1.],
 RP_synthesis_support=[0.,.25,1.],orphan_clearance_support=[0.,1.])

@dataclass(frozen=True)
class RibosomeMaintenanceParameters:
 damage_per_s:float=1e-5
 exchange_s:float=3600.
 retirement_half_h:float=48.
 impaired_retirement_gain:float=4.
 hydrolysis_s:float=3600.
 export_s:float=900.
 RPS26_initial_uM:float=.001
 RPS26_half_uM:float=.001
 RPS26_message_uM:float=.0001
 RPS26_orphan_half_h:float=6.
 exchange_ATP_per_RP:float=10.
 sorting_ATP_per_subunit:float=100.
 hydrolysis_ATP_equiv_per_aa:float=.02
 export_ATP_equiv_per_aa:float=.05
 def vector(self):
  a=np.array(list(asdict(self).values()),float)
  if not np.isfinite(a).all() or min(a)<0 or min(a[[1,2,3,4,5,7,9]])<=0:raise ValueError('Finite nonnegative maintenance values;positive times/half constants')
  return a

NAMES=tuple('maintenance_'+s for s in (
 'impaired40_Ceq','lysosomal40_Ceq','lysosomal60_Ceq','free_RPS26_uM',
 'RPS26_bin0_Ceq','RPS26_bin1_Ceq','RPS26_bin2_Ceq','RPS26_bin3_Ceq','RPS26_post_Ceq','lysosomal_AA_uM',
 'damaged40_Ceq','exchanged_RPS26_Ceq','retired40_Ceq','retired60_Ceq','digested40_Ceq','digested60_Ceq','AA_exported_uM',
 'RP_initiated_Ceq','RP_completed_Ceq','RP_recycled_Ceq','RP_incorporated_aa_uM','RP_init_GTP_uM','RP_elong_GTP_uM',
 'RP_recycling_ATP_uM','exchange_ATP_uM','sorting_ATP_uM','hydrolysis_ATP_equiv_uM','export_ATP_equiv_uM',
 'replaced_RP_aa_uM','rRNA40_disposed_Ceq','rRNA60_disposed_Ceq','orphan_RP_degraded_uM','orphan_ATP_uM',
 'impaired60_Ceq','damaged60_Ceq'))
assert len(NAMES)==35

def initialize(base,v):
 z=np.r_[base,np.zeros(35)];z[643]=v[6];return z

def controls(sg,supplied=None):
 # No inferred basal local damage/extra retirement/synthesis; explicit challenge.
 a=np.c_[sg[:,:2],np.tile([0.,1.,0.,1.,1.,0.,0.],(len(sg),1))] if supplied is None else np.asarray(supplied,float)
 if a.shape!=(len(sg),9) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or min(a[:,2:].ravel())<0 or max(a[:,2:].ravel())>1:raise ValueError('Maintenance:start,end,damage,exchange,retirement,hydrolysis,export,RP_synthesis,orphan supports0..1')
 return a.copy()

@njit
def extension(z,old,v,ctl,tv,C,ratio,KATP,rp40aa,rp60aa):
 out=np.zeros(675);out[:640]=old;scale=C*ratio;L=115.
 energy=max(z[136],0.)/(KATP+max(z[136],0.))
 gg=max(z[576],0.)/(tv[14]+max(z[576],0.));qg=max(z[575],0.)/(tv[13]+max(z[575],0.))
 free=np.maximum(z[564:566],0.);bad=np.maximum(np.array([z[640],z[673]]),0.)
 damage=v[0]*ctl[0]*free
 rp=max(z[643],0.);exchange=bad[0]/v[1]*rp/(v[7]+rp)*energy*ctl[1]
 ret=np.log(2)/(v[2]*3600)*energy*ctl[2]
 goodret=ret*free;badret=ret*v[3]*bad;retired=goodret+badret
 hydro=np.maximum(z[641:643],0.)/v[4]*energy*ctl[3]
 exported=max(z[649],0.)/v[5]*energy*ctl[4]
 bins=np.maximum(z[644:648],0.);post=max(z[648],0.)
 slots=max(v[8]*ctl[5]/scale-bins.sum(),0.)
 load=min(min(free),slots)/tv[2]*gg
 steps=4*tv[1]/L*gg*qg*bins
 recycle=post/tv[3]*energy;inc=L/4*steps.sum()*scale
 orphan=np.log(2)/(v[9]*3600)*rp*energy*ctl[6]
 gi=tv[19]*load*scale;ge=2*inc;ca=tv[20]*recycle*scale
 ce=v[10]*exchange*scale;cs=v[11]*retired.sum()*scale
 residues=(hydro[0]*rp40aa+hydro[1]*rp60aa)*scale
 ch=v[12]*residues;cx=v[13]*exported;co=tv[23]*L*orphan
 out[564:566]+=recycle-load-damage-goodret;out[564]+=exchange
 out[640]=damage[0]-exchange-badret[0];out[673]=damage[1]-badret[1]
 out[641:643]=retired-hydro;out[643]=steps[3]*scale-exchange*scale-orphan
 out[644]=load-steps[0];out[645:648]=steps[:3]-steps[1:];out[648]=steps[3]-recycle
 out[649]=residues-exported
 out[650:673]=np.array([damage[0],exchange,retired[0],retired[1],hydro[0],hydro[1],exported,
  load,steps[3],recycle,inc,gi,ge,ca,ce,cs,ch,cx,L*exchange*scale,hydro[0],hydro[1],orphan,co])
 out[674]=damage[1]
 out[575]-=inc;out[576]-=gi+ge;out[577]+=gi+ge
 out[584]+=L*(exchange*scale+orphan) # existing peptide hydrolysis; no instant AA
 out[574]+=exported # only after lysosomal hydrolysis AND export
 cost=ca+ce+cs+ch+cx+co
 out[136]-=cost;out[137]+=cost;out[150]+=cost
 return out

def adjusted_particles(z,base):
 zz=z.copy();bb=base.copy()
 for j in range(2):
  # Destroyed particles are history for lineage only, never actual free capacity.
  extra=z[:,644:649].sum(1)+z[:,640 if j==0 else 673]+z[:,641+j]+z[:,654+j]
  e0=base[644:649].sum()+base[640 if j==0 else 673]+base[641+j]+base[654+j]
  zz[:,564+j]+=extra;bb[564+j]+=e0
 return zz,bb

def revise_checks(ch,z,base,v,tv,C,ratio,rp40aa,rp60aa):
 out=dict(ch);d=z-base;scale=C*ratio;L=115.;err=lambda x:float(np.max(abs(x)))
 nasRP=d[:,644:648]@np.arange(4)*L/4*scale
 nas=(d[:,578:582]+d[:,620:624]+d[:,624:628])@np.arange(4)*tv[0]/4*scale
 A=d[:,599]-d[:,600];G=d[:,601]-d[:,602]
 residues=scale*(rp40aa*d[:,654]+rp60aa*d[:,655])
 busy=d[:,578:583].sum(1)+d[:,644:649].sum(1)
 particles40=d[:,562]+d[:,564]+busy+d[:,620:624].sum(1)+d[:,640]+d[:,641]
 particles60=d[:,563]+d[:,565]+busy+d[:,620:628].sum(1)+d[:,673]+d[:,642]
 backbone=scale*(rp40aa*particles40+rp60aa*particles60)
 delivered_residues=scale*(rp40aa*(d[:,566]-d[:,570])+rp60aa*(d[:,567]-d[:,571]))
 out.update(maintenance_nonnegative=float(max(0.,-z[:,640:].min())),
  local_tRNA_balance=err(d[:,575]-d[:,589]+d[:,588]+d[:,660]),
  local_GTP_balance=err(d[:,576]-d[:,592]+d[:,593]+d[:,590]+d[:,591]+d[:,635]+d[:,661]+d[:,662]-G),
  local_energy_reserve_identity=err(d[:,594]+d[:,592]-d[:,593]-4*(d[:,588]+d[:,660])-d[:,591]-d[:,635]-d[:,661]-2*d[:,575]-d[:,576]+G),
  local_peptide_balance=err(d[:,584]-tv[0]*d[:,596]-d[:,633]+d[:,597]-d[:,668]-L*d[:,671]),
  local_AA_balance=err(d[:,574]+d[:,575]+nas+tv[0]*d[:,583]+d[:,584]+d[:,628]+nasRP+L*d[:,643]-A-d[:,656]),
  maintenance_total_local_residues=err(d[:,574]+d[:,575]+nas+tv[0]*d[:,583]+d[:,584]+d[:,628]+nasRP+L*d[:,643]+d[:,649]+backbone-A-delivered_residues),
  maintenance_actual_rRNA40=err(particles40+d[:,669]-d[:,566]+d[:,570]),
  maintenance_actual_rRNA60=err(particles60+d[:,670]-d[:,567]+d[:,571]),
  maintenance_RP_nascent=err(nasRP+L*scale*d[:,658]-d[:,660]),
  maintenance_RP_pool=err(d[:,643]-scale*d[:,658]+scale*d[:,651]+d[:,671]),
  maintenance_RP_busy=err(d[:,644:648].sum(1)-d[:,657]+d[:,658]),
  maintenance_RP_post=err(d[:,648]-d[:,658]+d[:,659]),
  maintenance_lyso_AA=err(d[:,649]-residues+d[:,656]),
  maintenance_replaced_RP=err(d[:,668]-L*scale*d[:,651]),
  maintenance_RNA40=err(d[:,669]-d[:,654]),maintenance_RNA60=err(d[:,670]-d[:,655]),
  maintenance_initGTP=err(d[:,661]-tv[19]*scale*d[:,657]),
  maintenance_elongGTP=err(d[:,662]-2*d[:,660]),
  maintenance_recycleATP=err(d[:,663]-tv[20]*scale*d[:,659]),
  maintenance_exchangeATP=err(d[:,664]-v[10]*scale*d[:,651]),
  maintenance_sortATP=err(d[:,665]-v[11]*scale*(d[:,652]+d[:,653])),
  maintenance_hydroATP=err(d[:,666]-v[12]*residues),
  maintenance_exportATP=err(d[:,667]-v[13]*d[:,656]),
  maintenance_orphanATP=err(d[:,672]-tv[23]*L*d[:,671]))
 for j in range(2):out['maintenance_lyso'+str(j)]=err(d[:,641+j]-d[:,652+j]+d[:,654+j])
 return out

def observe(z,C,ratio):
 return dict(impaired40_uM=z[:,640]*C*ratio,impaired60_uM=z[:,673]*C*ratio,
  RPS26_uM=z[:,643],exchanges_uM=z[:,651]*C*ratio,
  retired_subunits_uM=z[:,652:654]*C*ratio,lysosomal_AA_uM=z[:,649],returned_AA_uM=z[:,656],
  rRNA_disposed_subunit_equivalents=z[:,669:671],
  direct_ATP_equiv_uM=z[:,663:668].sum(1)+z[:,672],human_calibrated=False,
  scope='One selected RPS26 exchange hypothesis, finite reserve and inducible local synthesis. Generic selected-subunit retirement, not a calibrated NUFIP1 response. RP kit masses inherited effective inventory. rRNA not converted to amino acids or charged nucleotides. No translation fidelity/collision detector or all-protein replacement.')
