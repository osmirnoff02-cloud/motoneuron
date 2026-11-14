"""[codex] 2026-10-30. Finite selected somatic AA/charged-tRNA/GTP pools.
Charging uses inherited fast AMP-recycling approximation:2 ATP->2ADP/aa.
NDPK explicitly transfers ATP+GDP <-> ADP+GTP; not a new energy source.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit

NAMES=('translation_free_AA_uM','translation_charged_tRNA_uM','translation_GTP_uM','translation_GDP_uM',
 'AA_imported_uM','AA_exported_uM','AA_orphan_recovered_uM','tRNA_charged_uM','AA_incorporated_uM',
 'elongation_GTP_hydrolyzed_uM','NDPK_forward_uM','NDPK_reverse_uM','charging_ATP_equiv_uM',
 'NDPK_ATP_spent_uM','NDPK_ATP_returned_uM','replaced_translation_equiv_uM')

@dataclass(frozen=True)
class TranslationSubstrateParameters:
    AA0_uM:float=1000.
    tRNA_total_uM:float=50.
    charged_fraction:float=.5
    GTP0_uM:float=400.
    GDP0_uM:float=100.
    charging_half_AA_uM:float=100.
    elongation_half_charged_uM:float=10.
    elongation_half_GTP_uM:float=50.
    AA_exchange_s:float=3600.
    NDPK_equilibrium:float=1.
    orphan_recovery_fraction:float=.8
    provenance:str='Selected aggregate AA/tRNA and guanylate pools, not20specific amino acids or allGTP users. Human MN concentrations/kinetics unknown. Charging2ATP-equivalents and elongation2GTP/aa repartition prior4-equivalent cost; fast AMP recycling remains lumped. NDPK reversible; enzyme capacity derived from basal flux, not measured. Only orphan-peptide AA recovery explicit; mature turnover/export belongs to open AA balance.'
    def vector(self):
        a=np.array([getattr(self,k) for k in list(self.__dataclass_fields__)[:-1]])
        if not np.isfinite(a).all() or min(a[:2])<=0 or not 0<a[2]<1 or min(a[3:10])<=0 or not 0<=a[10]<=1 or not self.provenance.strip():raise ValueError('Invalid substrate parameters')
        return a

SUBSTRATE_RANGES=dict(AA0_uM=[100.,1000.,5000.],tRNA_total_uM=[10.,50.,200.],charged_fraction=[.2,.5,.8],
 GTP0_uM=[100.,400.,1000.],GDP0_uM=[50.,100.,400.],charging_half_AA_uM=[10.,100.,1000.],
 elongation_half_charged_uM=[1.,10.,50.],elongation_half_GTP_uM=[5.,50.,200.],AA_exchange_s=[300.,3600.,36000.],
 NDPK_equilibrium=[.5,1.,2.],orphan_recovery_fraction=[0.,.8,1.],AA_supply_support=[0.,.25,1.],charging_support=[0.,.25,1.],NDPK_support=[0.,.25,1.])

def initial(v,b,nuc,shref,ATP0,ADP0):
    # basal residue incorporation from all8 classes, uM residues/s
    j=b[0]*(np.sum(nuc[6:8])*nuc[5]+np.sum(shref[:6]*shref[6:12]))
    force=ATP0*v[4]-ADP0*v[3]/v[9]
    if force<=0:raise ValueError('Chosen guanylate/adenylate ratios cannot sustain forward basal NDPK')
    q=np.zeros(16);q[:4]=[v[0],v[1]*v[2],v[3],v[4]]
    return q,np.array([j,2*j/force,ATP0,ADP0])

def controls(sg,supplied=None):
    a=np.c_[sg[:,:2],np.ones((len(sg),3))] if supplied is None else np.asarray(supplied,float)
    if a.shape!=(len(sg),5) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>2):raise ValueError('Substrate schedule start,end,AA_supply,charging,NDPK supports[0,2]')
    return a

@njit
def gate(z,v):
    Q=max(z[498],0.);G=max(z[499],0.);Q0=v[1]*v[2]
    q=min(1.,Q/(v[6]+Q)*(v[6]+Q0)/Q0)
    g=min(1.,G/(v[7]+G)*(v[7]+v[3])/v[3])
    return q*g

@njit
def fluxes(z,old,v,ref,b,nuc,control):
    A=max(z[497],0.);Q=max(z[498],0.);G=max(z[499],0.);D=max(z[500],0.)
    ATP=max(z[120],0.);ADP=max(z[121],0.);Q0=v[1]*v[2];j0=ref[0]
    charge=j0*A/(v[5]+A)*(v[5]+v[0])/v[0]*max(v[1]-Q,0)/(v[1]-Q0)*ATP/(nuc[8]+ATP)*(nuc[8]+ref[2])/ref[2]*control[1]
    forward=ref[1]*ATP*D*control[2];reverse=ref[1]*ADP*G/v[9]*control[2]
    inc=(old[254]+old[328]+old[493])/4
    recovered=b[0]*v[10]*(old[382]+old[383]+np.sum(old[403:493].reshape((6,15))[:,13]))
    incoming=(j0+max(v[0]-A,0)/v[8])*control[0];export=max(A-v[0],0)/v[8]
    q=np.zeros(16);q[:4]=[incoming+recovered-export-charge,charge-inc,forward-reverse-2*inc,2*inc-forward+reverse]
    q[4:]=np.array([incoming,export,recovered,charge,inc,2*inc,forward,reverse,2*charge,forward,reverse,4*inc])
    return q

@njit
def append_substrates(z,old,v,ref,b,nuc,control):
    q=fluxes(z,old,v,ref,b,nuc,control);out=np.zeros(513);out[:497]=old;out[497:]=q
    delta=q[12]+q[13]-q[14]-q[15]
    out[120]-=delta;out[121]+=delta;out[125]+=delta
    return out

def checks(z,base,v):
    q=z[:,497:513];q0=base[497:513];d=q-q0;err=lambda a:float(np.max(abs(a)))
    return dict(substrate_nonnegative=float(max(0.,-q.min())),charged_tRNA_bound=float(max(0.,(q[:,1]-v[1]).max())),
      AA_open_balance=err(d[:,0]+d[:,1]+d[:,8]+d[:,5]-d[:,4]-d[:,6]),
      charged_tRNA_balance=err(d[:,1]-d[:,7]+d[:,8]),guanylate_balance=err(d[:,2]+d[:,3]),
      GTP_balance=err(d[:,2]-d[:,10]+d[:,11]+d[:,9]),charging_energy=err(d[:,12]-2*d[:,7]),
      NDPK_forward_energy=err(d[:,13]-d[:,10]),NDPK_reverse_energy=err(d[:,14]-d[:,11]),
      elongation_two_GTP=err(d[:,9]-2*d[:,8]),reference_translation_removal=err(d[:,15]-4*d[:,8]),
      translation_reserve_energy=err(d[:,12]+d[:,13]-d[:,14]-d[:,15]-2*d[:,1]-d[:,2]))
