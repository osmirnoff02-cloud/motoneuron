"""[codex]2026-11-02. One representative local cargo, no assigned gene function.

Particle occupancy is soma-equivalent C; substrates/protein are local uM.
AA/GTP/GDP exchange transfers amounts, not duplicated sources. Charging keeps
the inherited fast AMP-recycling approximation (2ATP-equivalents/aa); NDPK
transfers ATP+GDP<->ADP+GTP. Existing protein fluxes/costs are not replaced.
"""
from dataclasses import dataclass,asdict
import numpy as np
from numba import njit

LOCAL_TRANSLATION_RANGES={
 'length_aa':[100.,250.,500.], 'elongation_aa_s':[1.,5.,10.],
 'initiation_s':[10.,30.,300.], 'recycling_s':[1.,5.,30.],
 'message_slots_uM':[.0001,.001,.01], 'initial_local_fraction':[0.,.05,.2],
 'AA0_uM':[100.,1000.,5000.], 'tRNA_total_uM':[10.,50.,200.],
 'charged_fraction':[.2,.5,.8], 'GTP0_uM':[100.,400.,1000.], 'GDP0_uM':[50.,100.,400.],
 'charging_capacity_uM_s':[.01,.1,1.], 'charging_half_AA_uM':[10.,100.,1000.],
 'elongation_half_charged_uM':[1.,10.,50.], 'elongation_half_GTP_uM':[5.,50.,200.],
 'AA_exchange_s':[300.,3600.,36000.], 'guanylate_exchange_s':[30.,300.,3000.],
 'NDPK_rate_per_uM_s':[1e-7,1e-6,1e-5], 'NDPK_equilibrium':[.5,1.,2.],
 'initiation_GTP_per_chain':[0.,2.,4.], 'recycling_ATP_per_chain':[0.,2.,10.],
 'protein_half_h':[6.,24.,72.], 'peptide_hydrolysis_s':[300.,1800.,14400.],
 'degradation_ATP_per_aa':[0.,.2,2.],
 'message_support':[0.,.25,1.], 'elongation_support':[0.,.25,1.],
 'AA_exchange_support':[0.,.25,1.], 'guanylate_exchange_support':[0.,.25,1.],
 'charging_support':[0.,.25,1.], 'NDPK_support':[0.,.25,1.],
}

@dataclass(frozen=True)
class LocalTranslationParameters:
 length_aa:float=250.
 elongation_aa_s:float=5.
 initiation_s:float=30.
 recycling_s:float=5.
 message_slots_uM:float=.001
 initial_local_fraction:float=0.
 AA0_uM:float=1000.
 tRNA_total_uM:float=50.
 charged_fraction:float=.5
 GTP0_uM:float=400.
 GDP0_uM:float=100.
 charging_capacity_uM_s:float=.1
 charging_half_AA_uM:float=100.
 elongation_half_charged_uM:float=10.
 elongation_half_GTP_uM:float=50.
 AA_exchange_s:float=3600.
 guanylate_exchange_s:float=300.
 NDPK_rate_per_uM_s:float=1e-6
 NDPK_equilibrium:float=1.
 initiation_GTP_per_chain:float=2.
 recycling_ATP_per_chain:float=2.
 protein_half_h:float=24.
 peptide_hydrolysis_s:float=1800.
 degradation_ATP_per_aa:float=.2
 def vector(self):
  a=np.array(list(asdict(self).values()),float)
  if not np.isfinite(a).all() or min(a)<0 or min(a[[0,1,2,3,4,6,7,9,10,12,13,14,15,16,18,21,22]])<=0 or not 0<=a[5]<=.5 or not 0<a[8]<1:raise ValueError('Invalid local translation parameter; positive times/pools, fraction0..0.5 and charged fraction0..1 required')
  return a

NAMES=tuple('local_translation_'+s for s in (
 'AA_uM','charged_tRNA_uM','GTP_uM','GDP_uM',
 'bin0_Ceq','bin1_Ceq','bin2_Ceq','bin3_Ceq','posttermination_Ceq',
 'representative_protein_uM','peptides_aa_uM',
 'initiated_Ceq','completed_Ceq','recycled_Ceq','incorporated_aa_uM',
 'charged_aa_uM','elongation_GTP_uM','initiation_GTP_uM','NDPK_forward_uM','NDPK_reverse_uM',
 'charging_ATP_equiv_uM','recycling_ATP_uM','protein_degraded_uM','hydrolyzed_aa_uM','degradation_ATP_uM',
 'AA_from_soma_uM','AA_to_soma_uM','GTP_from_soma_uM','GTP_to_soma_uM','GDP_from_soma_uM','GDP_to_soma_uM'))
assert len(NAMES)==31

def initialize(base,v):
 b=np.r_[base,np.zeros(31)]
 # Optional redistribution of existing free subunits, not new ribosome creation.
 moved=base[330:332]*v[5];b[330:332]-=moved;b[564:566]+=moved
 b[574:578]=[v[6],v[7]*v[8],v[9],v[10]]
 return b

def controls(segments,supplied=None):
 sg=np.asarray(segments,float)
 a=np.c_[sg[:,:2],np.ones((len(sg),6))] if supplied is None else np.asarray(supplied,float)
 if a.shape!=(len(sg),8) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>1):raise ValueError('Local translation: start,end,message,elongation,AAexchange,Gexchange,charging,NDPK; supports0..1')
 return a.copy()

@njit
def extension(z,old,v,ctl,C,ratio,KATP):
 out=np.zeros(605);out[:574]=old
 q=np.maximum(z[574:605],0.);dx=np.zeros(31);scale=C*ratio;L=v[0]
 ATP=max(z[136],0.);ADP=max(z[137],0.);energy=ATP/(KATP+ATP)
 A,Q,G,D=q[:4];bins=q[4:8];post=q[8]
 gateQ=Q/(v[13]+Q);gateG=G/(v[14]+G)
 slots=v[4]*ctl[0]/scale;available=max(slots-bins.sum(),0.)
 freepair=min(max(z[564],0.),max(z[565],0.))
 load=min(freepair,available)/v[2]*gateG
 steps=4*v[1]/L*gateQ*gateG*ctl[1]*bins
 complete=steps[3];recycle=post/v[3]*energy
 inc=L/4*steps.sum()*scale
 charge=v[11]*A/(v[12]+A)*max(v[7]-Q,0.)/v[7]*energy*ctl[4]
 forward=v[17]*ATP*D*ctl[5];reverse=v[17]*ADP*G/v[18]*ctl[5]
 initG=v[19]*load*scale;elongG=2*inc
 deg=np.log(2)/(v[21]*3600)*q[9]*energy
 hydro=q[10]/v[22] # effective peptide hydrolysis; transporters/lysosomal pH unresolved
 AAexchange=(max(z[497],0.)-A)/v[15]*ctl[2]
 Gexchange=(max(z[499],0.)-G)/v[16]*ctl[3]
 Dexchange=(max(z[500],0.)-D)/v[16]*ctl[3]
 dx[:4]=[AAexchange-charge+hydro,charge-inc,forward-reverse-elongG-initG+Gexchange,elongG+initG-forward+reverse+Dexchange]
 dx[4]=load-steps[0];dx[5:8]=steps[:3]-steps[1:]
 dx[8]=complete-recycle;dx[9]=complete*scale-deg;dx[10]=L*deg-hydro
 dx[11:25]=np.array([load,complete,recycle,inc,charge,elongG,initG,forward,reverse,2*charge,v[20]*recycle*scale,deg,hydro,v[23]*L*deg])
 dx[25:31]=np.array([max(AAexchange,0.),max(-AAexchange,0.),max(Gexchange,0.),max(-Gexchange,0.),max(Dexchange,0.),max(-Dexchange,0.)])
 out[564:566]+=recycle-load
 out[497]-=AAexchange/ratio;out[499]-=Gexchange/ratio;out[500]-=Dexchange/ratio
 # Charging and NDPK are distinct actual expenditures; no extra4ATP/aa debit.
 cost=dx[20]+dx[21]+dx[24]+forward-reverse
 out[136]-=cost;out[137]+=cost;out[150]+=cost
 out[574:]=dx
 return out

def adjusted_particles(z,base):
 zz=z.copy();bb=base.copy();busy=zz[:,578:583].sum(1);busy0=bb[578:583].sum()
 zz[:,564:566]+=busy[:,None];bb[564:566]+=busy0
 return zz,bb

def revise_somatic_checks(ch,z,base,ratio):
 out=dict(ch);d=z-base;err=lambda a:float(np.max(abs(a)))
 A=(d[:,599]-d[:,600])/ratio;G=(d[:,601]-d[:,602])/ratio;D=(d[:,603]-d[:,604])/ratio
 out['AA_open_balance']=err(d[:,497]+d[:,498]+d[:,505]+d[:,502]-d[:,501]-d[:,503]-d[:,521]+A)
 out['guanylate_balance']=err(d[:,499]+d[:,500]+G+D)
 out['GTP_balance']=err(d[:,499]-d[:,507]+d[:,508]+d[:,506]+G)
 out['translation_reserve_energy']=err(d[:,509]+d[:,510]-d[:,511]-d[:,512]-2*d[:,498]-d[:,499]-G)
 return out

def checks(z,base,v,C,ratio):
 q=z[:,574:605];d=q-base[574:605];err=lambda x:float(np.max(abs(x)))
 scale=C*ratio;L=v[0];nas=(d[:,5]+2*d[:,6]+3*d[:,7])*L/4*scale
 A=d[:,25]-d[:,26];G=d[:,27]-d[:,28];D=d[:,29]-d[:,30]
 return dict(local_translation_nonnegative=float(max(0.,-q.min())),local_tRNA_bound=float(max(0.,(q[:,1]-v[7]).max())),
  local_busy_balance=err(d[:,4:8].sum(1)-d[:,11]+d[:,12]),
  local_posttermination_balance=err(d[:,8]-d[:,12]+d[:,13]),
  local_nascent_balance=err(nas+L*scale*d[:,12]-d[:,14]),
  local_protein_balance=err(d[:,9]-scale*d[:,12]+d[:,22]),
  local_peptide_balance=err(d[:,10]-L*d[:,22]+d[:,23]),
  local_AA_balance=err(d[:,0]+d[:,1]+nas+L*d[:,9]+d[:,10]-A),
  local_tRNA_balance=err(d[:,1]-d[:,15]+d[:,14]),
  local_guanylate_balance=err(d[:,2]+d[:,3]-G-D),
  local_GTP_balance=err(d[:,2]-d[:,18]+d[:,19]+d[:,16]+d[:,17]-G),
  local_charging_energy=err(d[:,20]-2*d[:,15]),
  local_elongation_GTP=err(d[:,16]-2*d[:,14]),
  local_initiation_GTP=err(d[:,17]-v[19]*scale*d[:,11]),
  local_recycling_ATP=err(d[:,21]-v[20]*scale*d[:,13]),
  local_degradation_ATP=err(d[:,24]-v[23]*L*d[:,22]),
  local_energy_reserve_identity=err(d[:,20]+d[:,18]-d[:,19]-4*d[:,14]-d[:,17]-2*d[:,1]-d[:,2]+G),
  global_guanylate_exchange=err((z[:,499]+z[:,500]-base[499]-base[500])*ratio+d[:,2]+d[:,3]))

def observe(z,C,ratio):
 a=np.asarray(z,float)
 if a.ndim!=2 or a.shape[1]!=605:raise ValueError('Expected local translation605 states')
 return dict(free_AA_uM=a[:,574],charged_tRNA_uM=a[:,575],GTP_uM=a[:,576],GDP_uM=a[:,577],
  busy_ribosomes_uM=a[:,578:582].sum(1)*C*ratio,
  posttermination_ribosomes_uM=a[:,582]*C*ratio,
  total_occupied_ribosomes_uM=a[:,578:583].sum(1)*C*ratio,
  protein_uM=a[:,583],peptide_aa_uM=a[:,584],
  completed_protein_uM=a[:,586]*C*ratio,ATP_equivalent_net_uM=a[:,594]+a[:,595]+a[:,598]+a[:,592]-a[:,593],
  scope='One representative local cargo, no assigned gene/function or whole-proteome calibration')
