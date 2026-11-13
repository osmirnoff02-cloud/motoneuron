"""[codex] 2026-11-03. Finite, conservative alpha/beta/CAS/NTF2/Ran transport.

All amounts are uM per cytoplasmic reference volume, including nuclear pools.
This is a scenario model, not a fit to human MN kinetics or the Riddick model.
TDP pools partition existing intact U+P; synthesis/loss and nucleotide recharge
remain owned by the host. Bound and free nuclear intact TDP retain the host's
functional equivalence assumption. RNA binding/retention is unresolved.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit

SPECIES=('TDP_c','TDP_n','KPNA_c','KPNA_n','KPNB1_c','KPNB1_n',
 'KPNA_KPNB1_c','KPNA_KPNB1_n','TDP_KPNA_KPNB1_c','TDP_KPNA_KPNB1_n',
 'KPNB1_RanGTP_c','KPNB1_RanGTP_n','CAS_c','CAS_n','CAS_KPNA_RanGTP_c',
 'CAS_KPNA_RanGTP_n','RanGDP_c','RanGDP_n','RanGTP_c','RanGTP_n',
 'NTF2_c','NTF2_n','NTF2_RanGDP_c','NTF2_RanGDP_n')
COUNTERS=('RCC1_forward','RCC1_reverse','RanGTP_hydrolyzed','carrier_TDP_in',
 'carrier_TDP_out','passive_TDP_in','passive_TDP_out','TDP_loss','TDP_supply')
NAMES=tuple('nct_'+s+'_uM_cyt_equiv' for s in SPECIES+COUNTERS)
SIZE=len(NAMES)
GROUPS={'TDP':[0,1,8,9], 'KPNA':[2,3,6,7,8,9,14,15],
 'KPNB1':[4,5,6,7,8,9,10,11], 'CAS':[12,13,14,15],
 'Ran':[10,11,14,15,16,17,18,19,22,23], 'NTF2':[20,21,22,23]}
GBOUND=np.array([10,11,14,15,18,19])
CONTROL_NAMES=('import_binding','facilitated_pore','RCC1','RanGAP_RanBP','NTF2_binding','CAS_binding','passive_pore')

@dataclass(frozen=True)
class NuclearTransportParameters:
    nuclear_to_cytoplasmic_volume:float=.25
    KPNA_total_uM:float=.3
    KPNB1_total_uM:float=.5
    CAS_total_uM:float=.3
    Ran_total_uM:float=2.
    NTF2_total_uM:float=.2
    initial_nuclear_RanGTP_fraction:float=.2
    association_uM_inv_s:float=.05
    dissociation_s_inv:float=.005
    Ran_release_uM_inv_s:float=.2
    CAS_assembly_uM_inv2_s:float=.2
    NTF2_association_uM_inv_s:float=1.
    NTF2_release_s_inv:float=.1
    RCC1_exchange_uM_inv_s:float=.005
    RCC1_reverse_ratio:float=1.
    RanGAP_s_inv:float=.1
    facilitated_permeability_s_inv:float=.05
    passive_TDP_permeability_s_inv:float=1e-5
    passive_Ran_permeability_s_inv:float=.001
    pore_load_half_uM:float=10.
    provenance:str='U40: all MN kinetic/pool/volume values are assumptions; architecture Riddick2005/2007, Gorlich2003 and Pinarbasi2018. No imported HeLa rates or fitted MN kinetics.'
    def vector(self):
        v=np.array([getattr(self,k) for k in self.__dataclass_fields__ if k!='provenance'],float)
        if not np.isfinite(v).all() or np.any(v<0) or v[0]<=0 or v[19]<=0 or not 0<=v[6]<=1 or not self.provenance.strip():
            raise ValueError('Finite nonnegative transport parameters; positive volume and pore half-load; Ran fraction in [0,1]')
        return v

# Sensitivity grids, NOT measured ranges or biological bounds.
RANGES=dict(nuclear_to_cytoplasmic_volume=[.1,.25,.5],KPNA_total_uM=[0.,.1,.3,1.],
 KPNB1_total_uM=[0.,.1,.5,2.],CAS_total_uM=[0.,.1,.3,1.],Ran_total_uM=[0.,.5,2.,5.],
 NTF2_total_uM=[0.,.05,.2,1.],initial_nuclear_RanGTP_fraction=[0.,.2,.8],
 association_uM_inv_s=[.005,.05,.5],dissociation_s_inv=[.0005,.005,.05],
 Ran_release_uM_inv_s=[.02,.2,2.],CAS_assembly_uM_inv2_s=[.02,.2,2.],
 NTF2_association_uM_inv_s=[.1,1.,10.],NTF2_release_s_inv=[.01,.1,1.],
 RCC1_exchange_uM_inv_s=[.0005,.005,.05],RCC1_reverse_ratio=[.1,1.,10.],
 RanGAP_s_inv=[.01,.1,1.],facilitated_permeability_s_inv=[.005,.05,.5],
 passive_TDP_permeability_s_inv=[1e-6,1e-5,1e-4],passive_Ran_permeability_s_inv=[.0001,.001,.01],
 pore_load_half_uM=[1.,10.,100.],support=[0.,.25,1.,2.])

def controls(sg,supplied=None):
    a=np.c_[sg[:,:2],np.ones((len(sg),7))] if supplied is None else np.asarray(supplied,float)
    if a.shape!=(len(sg),9) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>2):
        raise ValueError('NCT schedule: matching start/end and seven supports in [0,2]')
    return a

def initialize(base,C,v):
    if len(base) not in (1600,2160):raise ValueError('Detailed transport requires coupled translation or individual RP topology')
    x=np.zeros(SIZE);T=C*(base[6]+base[7]);x[0]=T*(1-base[320]);x[1]=T*base[320]
    for i,total in [(2,v[1]),(4,v[2]),(12,v[3]),(20,v[5])]:
        x[i]=total/(1+v[0]);x[i+1]=total*v[0]/(1+v[0])
    x[16]=v[4]*(1-v[6]);x[19]=v[4]*v[6]
    return np.r_[base,x]

@njit
def reaction(d,rate,reactants,products):
    for i in reactants:d[i]-=rate
    for i in products:d[i]+=rate

@njit
def fluxes(x,GTP,GDP,v,c):
    x=np.maximum(x,0.);d=np.zeros(SIZE);r=v[0]
    # Shared saturable access; permeability is reciprocal, not an NPC motor.
    load=0.
    for i in (4,6,8,10,12,14,20,22):load+=x[i]+x[i+1]/r
    gate=c[1]/(1+load/v[19])
    for j in range(2):
        vol=1. if j==0 else r
        a,b,ab,t,abt,rg,br=2+j,4+j,6+j,j,8+j,18+j,10+j
        reaction(d,v[7]*c[0]*x[a]*x[b]/vol,(a,b),(ab,))
        reaction(d,v[8]*x[ab],(ab,),(a,b))
        reaction(d,v[7]*c[0]*x[ab]*x[t]/vol,(ab,t),(abt,))
        reaction(d,v[8]*x[abt],(abt,),(ab,t))
        reaction(d,v[9]*x[rg]*x[abt]/vol,(rg,abt),(br,a,t))
        reaction(d,v[9]*x[rg]*x[ab]/vol,(rg,ab),(br,a))
        reaction(d,v[9]*x[rg]*x[b]/vol,(rg,b),(br,))
        cas,car=12+j,14+j
        reaction(d,v[10]*c[5]*x[cas]*x[a]*x[rg]/vol**2,(cas,a,rg),(car,))
        reaction(d,v[8]*x[car],(car,),(cas,a,rg))
        f,fd,rd=20+j,22+j,16+j
        reaction(d,v[11]*c[4]*x[f]*x[rd]/vol,(f,rd),(fd,))
        reaction(d,v[12]*x[fd],(fd,),(f,rd))
    for i in (4,6,8,10,12,14,20,22):
        inside=v[16]*gate*x[i];outside=v[16]*gate*x[i+1]/r
        reaction(d,inside,(i,),(i+1,));reaction(d,outside,(i+1,),(i,))
        if i==8:d[27]+=inside;d[28]+=outside
    # Passive diffusion depends on concentrations, in BOTH directions.
    for i in (0,16,18):
        k=v[17] if i==0 else v[18]
        inside=k*c[6]*x[i];outside=k*c[6]*x[i+1]/r
        reaction(d,inside,(i,),(i+1,));reaction(d,outside,(i+1,),(i,))
        if i==0:d[29]+=inside;d[30]+=outside
    # Nuclear RCC1 exchanges bound nucleotide; no direct ATP hydrolysis.
    forward=v[13]*c[2]*x[17]*max(GTP,0.)
    reverse=v[13]*v[14]*c[2]*x[19]*max(GDP,0.)
    reaction(d,forward,(17,),(19,));reaction(d,reverse,(19,),(17,))
    d[24]=forward;d[25]=reverse
    # Cytoplasmic GAP + lumped RanBP-mediated receptor disassembly.
    for i in (18,10,14):
        h=v[15]*c[3]*x[i];d[26]+=h
        if i==18:reaction(d,h,(18,),(16,))
        elif i==10:reaction(d,h,(10,),(4,16))
        else:reaction(d,h,(14,),(12,2,16))
    return d

@njit
def finish(z,old,C,v,ctl,offset):
    x=np.maximum(z[offset:offset+SIZE],0.)
    d=fluxes(x,z[499],z[500],v,ctl)
    # Host gross production plus any net return from other full-length pools.
    total=C*max(z[6]+z[7],0.);dt=C*(old[6]+old[7])
    supply=max(C*old[18],dt,0.);loss=supply-dt
    amount=x[0]+x[1]+x[8]+x[9]
    for i in (0,1,8,9):
        q=loss*x[i]/amount if amount>0 else 0.
        d[i]-=q
        if i==8:d[6]+=q
        if i==9:d[7]+=q
    d[0]+=supply;d[31]=loss;d[32]=supply
    out=np.zeros(offset+SIZE);out[:offset]=old
    out[offset:]=d
    exchange=d[24]-d[25];out[499]-=exchange;out[500]+=exchange
    # Only replacement of old fraction ODE: quotient rule, no second TDP pool.
    f=z[320]
    out[320]=(d[1]+d[9]-f*dt)/total if total>1e-20 else 0.
    return out

def checks(z,base,C,offset):
    x=z[:,offset:offset+SIZE];b=base[offset:offset+SIZE];delta=x-b
    err=lambda a:float(np.max(np.abs(a)))
    result={'nct_nonnegative':float(max(0.,-x.min())),
      'nct_TDP_projection':err(x[:,GROUPS['TDP']].sum(1)-C*(z[:,6]+z[:,7])),
      'nct_nuclear_projection':err(x[:,1]+x[:,9]-z[:,320]*C*(z[:,6]+z[:,7])),
      'nct_TDP_lineage':err(delta[:,GROUPS['TDP']].sum(1)+delta[:,31]-delta[:,32]),
      'nct_bound_GTP':err(delta[:,GBOUND].sum(1)-delta[:,24]+delta[:,25]+delta[:,26]),
      'nct_nuclear_fraction_bounds':float(max(0.,-z[:,320].min(),z[:,320].max()-1))}
    for name,ids in GROUPS.items():
        if name!='TDP':result['nct_conserved_'+name]=err(delta[:,ids].sum(1))
    return result

def revise_somatic_checks(ch,z,base,offset):
    # Existing ledger already includes local translation/ISR exchanges. Add the
    # new exchange debit with sign BEFORE taking an absolute error, via a view.
    zz=z.copy();bb=base.copy()
    zz[:,499]+=z[:,offset+24]-z[:,offset+25]
    zz[:,500]-=z[:,offset+24]-z[:,offset+25]
    bb[499]+=base[offset+24]-base[offset+25]
    bb[500]-=base[offset+24]-base[offset+25]
    return zz,bb

def observe(z,v,offset,parameters):
    x=z[:,offset:offset+SIZE];r=v[0]
    def ratio(n,c):return np.divide(n/r,c,out=np.full_like(n,np.nan),where=c>0)
    load=sum(x[:,i]+x[:,i+1]/r for i in (4,6,8,10,12,14,20,22))
    return dict(species=SPECIES,pools_cyt_equivalent_uM=x[:,:24],counters=x[:,24:],
      counter_names=COUNTERS,nuclear_TDP_fraction=z[:,320],
      TDP_concentration_N_over_C=ratio(x[:,1]+x[:,9],x[:,0]+x[:,8]),
      free_RanGTP_N_over_C=ratio(x[:,19],x[:,18]),
      pore_occupancy_fraction=load/(v[19]+load),parameters=asdict(parameters),
      scope='Finite KPNA/KPNB1/CAS/NTF2 and Ran pools; reversible pore passage with shared algebraic saturation. RCC1 exchanges shared GTP/GDP; cytoplasmic GAP/RanBP hydrolyzes bound GTP; existing NDPK is the sole recharge/ATP owner. TDP partitions existing U+P; uniform loss and equal nuclear functionality retained. No XPO1-TDP export, RNA retention, individual NUP turnover, RCC1 sequestration, all cargo competition or human MN calibration.')
