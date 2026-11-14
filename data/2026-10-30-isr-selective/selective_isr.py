"""[codex] 2026-10-30. Opt-in PERK/GCN2 and resource-owned ATF4/GADD34.

Ten selected translation classes share existing 40S/60S, AA, tRNA and GTP.
uORF selection is an effective delayed-reinitiation law, not codon simulation.
No human-MN kinetic calibration. Old API0.30 is retained unchanged.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit

LENGTHS=np.array([351.,674.]) # human canonical P18848/O75807, UniProt 2026-10-30
NAMES=('PERK_activity_proxy','GADD34_message_relative','ATF4_protein_uM','GADD34_protein_uM')
NAMES+=tuple(c+'_'+s for c in ('ATF4','GADD34') for s in ('init40','elong1','elong2','elong3','elong4','post80'))
NAMES+=tuple(c+'_'+s for s in ('incorporated_uM_aa','completed_uM','removed_uM_aa') for c in ('ATF4','GADD34'))
NAMES+=('ISR_reference_4eq_uM','ISR_recycling_ATP_uM','ISR_replaced_background_ATP_uM')
assert len(NAMES)==25

@dataclass(frozen=True)
class SelectiveISRParameters:
    PERK_response_s:float=300.
    PERK_phosphorylation_s:float=60.
    ATF4_baseline_uM:float=.001
    GADD34_baseline_uM:float=.001
    ATF4_half_life_s:float=1800.
    GADD34_half_life_s:float=3600.
    reinitiation_window:float=6.
    reinitiation_window_ratio:float=2.
    GADD34_message_response_s:float=1800.
    GADD34_message_gain:float=20.
    GADD34_ATF4_half_excess:float=4.
    GADD34_dephosphorylation_gain:float=12.
    provenance:str='Novoa2003 mouse ES/MEF and Lu2004 CHO/mouse: qualitative PERK/eIF2/ATF4/GADD34 signs. Human canonical protein lengths from UniProt; all concentrations, effective rates and gains are hypotheses. GADD34 mRNA is relative drive, not an absolute nucleotide pool. ISR nascent chains pause when resources run out; their separate abort/RQC route and uORF peptide cost remain unresolved.'
    def vector(self):
        a=np.array([getattr(self,k) for k in list(self.__dataclass_fields__)[:-1]])
        if not np.isfinite(a).all() or min(a[:9])<=0 or a[7]<=1 or min(a[9:])<0 or a[10]<=0 or not self.provenance.strip():raise ValueError('Invalid selective ISR parameters')
        return a

ISR_RANGES=dict(PERK_response_s=[30.,300.,1800.],PERK_phosphorylation_s=[30.,60.,300.],
 ATF4_baseline_uM=[.0001,.001,.01],GADD34_baseline_uM=[.0001,.001,.01],
 ATF4_half_life_s=[300.,1800.,7200.],GADD34_half_life_s=[900.,3600.,14400.],
 reinitiation_window=[3.,6.,9.],reinitiation_window_ratio=[1.5,2.,3.],
 GADD34_message_response_s=[300.,1800.,7200.],GADD34_message_gain=[5.,20.,50.],
 GADD34_ATF4_half_excess=[1.,4.,10.],GADD34_dephosphorylation_gain=[0.,12.,30.],
 ER_stress_drive=[0.,.25,1.],PERK_support=[0.,1.],GADD34_function_support=[0.,1.])

def controls(sg,supplied=None):
    a=np.c_[sg[:,:2],np.zeros(len(sg)),np.ones((len(sg),2))] if supplied is None else np.asarray(supplied,float)
    if a.shape!=(len(sg),5) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>1):raise ValueError('ISR schedule: start,end,ERstress,PERKsupport,GADD34functional_support in[0,1]')
    return a

def initialize(base,iv,b,nuc,rv,cybase,sub,subref,nrv,nrref):
    z=np.r_[base,np.zeros(25)];z[538]=1.;z[539:541]=iv[2:4]
    J=np.log(2)/iv[4:6]*iv[2:4] # protein uM/s
    x=z[541:553].reshape(2,6)
    for k,L in enumerate(LENGTHS):
        j=J[k]/b[0];x[k]=np.r_[j*rv[0],np.full(4,j*L/(4*nuc[5])),j*rv[1]]
    occupied=np.array([x.sum(),x[:,1:].sum()]);z[330:332]-=occupied
    cy=cybase.copy();cy[:2]-=occupied
    if min(cy[:2])<=0:raise ValueError('ISR demand exceeds available ribosomes')
    inc=float(J@LENGTHS);sref=subref.copy();sref[1]*=(sref[0]+inc)/sref[0];sref[0]+=inc
    # All newly represented ISR mature protein goes through existing cyto queue.
    z[513]+=inc*nrv[2];nref=nrref.copy();nref[0]+=nrv[1]*inc;nref[1]+=inc
    z[534:536]=.5
    ref=np.r_[J,inc,4*inc+J.sum()*rv[5],cy[:2],base[120]]
    return z,cy,sref,nref,ref

@njit
def selective_factor(tc,a,ratio):
    tc=min(1.,max(tc,0.));return tc*(np.exp(-a*tc)-np.exp(-a*ratio*tc))/(np.exp(-a)-np.exp(-a*ratio))

@njit
def general_control(nc,z,nrv):
    out=nc.copy();tc=1-min(1.,max(z[533],0.))
    denominator=nrv[12]+(1-nrv[12])*tc
    out[1]*=tc/denominator if denominator>0 else 0.
    return out

@njit
def fluxes(z,iv,ref,b,nuc,rv,sub,nrv,rawnc,cc,ctl,rctl,substrate_gate):
    q=np.zeros(25);x=np.maximum(z[541:553].reshape((2,6)),0.);dx=np.zeros((2,6))
    energy=max(z[120],0)/(nuc[8]+max(z[120],0))*(nuc[8]+ref[6])/ref[6]
    tc=1-min(1.,max(z[533],0.));mt=min(1.,max(z[536],0.))
    choice=np.array([selective_factor(tc,iv[6],iv[7]),max(z[538],0)*tc])
    inc=np.zeros(2);complete=np.zeros(2);cycle=0.;free40=0.;free60=0.
    for k in range(2):
        load=ref[k]/b[0]*choice[k]*mt*energy*substrate_gate*rawnc[1]*max(z[330],0)/ref[4]
        join=x[k,0]/rv[0]*max(z[331],0)/ref[5]*energy*cc[0]
        step=x[k,1:5]*(4*nuc[5]/LENGTHS[k])*energy*rawnc[2]*substrate_gate
        release=x[k,5]/rv[1]*energy*cc[1]
        dx[k,0]=load-join;dx[k,1]=join-step[0]
        dx[k,2:5]=step[:3]-step[1:];dx[k,5]=step[3]-release
        free40+=release-load;free60+=release-join
        inc[k]=step.sum()*LENGTHS[k]/4*b[0];complete[k]=step[3]*b[0];cycle+=release*b[0]*rv[5]
    protein=np.maximum(z[539:541],0.);loss=np.log(2)/iv[4:6]*protein
    q[0]=(ctl[0]*ctl[1]*min(1.,energy)-min(1.,max(z[537],0)))/iv[0]
    excess=max(protein[0]/iv[2]-1,0)
    q[1]=((1+iv[9]*excess/(iv[10]+excess))*min(1.,energy)-max(z[538],0))/iv[8]
    q[2:4]=complete-loss;q[4:16]=dx.ravel();q[16:18]=inc;q[18:20]=complete;q[20:22]=loss*LENGTHS
    q[22]=4*inc.sum();q[23]=cycle
    kinase=max(z[532],0)/nrv[6]+max(z[537],0)/iv[1]
    # Constitutive phosphatase is the separate1 below. All GADD34 protein
    # contributes, not only the excess over its arbitrary reference abundance.
    feedback=iv[11]*protein[1]/iv[3]*ctl[2]*rctl[4]
    pe=1-tc;phospho=kinase*tc*min(1.,energy)-pe*(1+feedback)/nrv[7]
    return q,np.array([free40,free60]),phospho

@njit
def before_substrates(old,q,free,ref,underlying,other0):
    out=old.copy();out[330:332]+=free
    replaced=ref[3]*max(underlying,0)/other0 if other0>0 else 0.
    delta=q[22]+q[23]-replaced
    out[120]-=delta;out[121]+=delta;out[125]+=delta
    # Temporary reference debit solely to call the existing substrate allocator.
    # Caller restores493 after allocation; actual ISR reference stays559.
    out[493]+=q[22]
    return out,replaced

@njit
def finish(z,old,q,phospho,replaced,iv):
    out=np.zeros(562);out[:537]=old;out[537:]=q;out[561]=replaced
    loss=q[20:22].sum();out[513]+=loss;out[516]+=loss
    out[533]=phospho
    for k in range(2):
        # Compatibility observer, not a second protein pool.
        out[534+k]=iv[2+k]/(iv[2+k]+max(z[539+k],0))**2*q[2+k]
    return out

def adjusted_ribosomes(z,base):
    zz=z.copy();bb=base.copy()
    x=zz[:,541:553].reshape(-1,2,6);xb=bb[541:553].reshape(2,6)
    zz[:,330]+=x.sum((1,2));zz[:,331]+=x[:,:,1:].sum((1,2))
    bb[330]+=xb.sum();bb[331]+=xb[:,1:].sum()
    return zz,bb

def revise_checks(out,z,base,iv,b):
    out=dict(out);d=z-base;err=lambda x:float(np.max(abs(x)))
    extra=d[:,557:559].sum(1)
    out['source_partition']=err(d[:,516]+d[:,517]-d[:,524:532].sum(1)-extra)
    out['recycled_material_balance']=err(d[:,513:516].sum(1)+d[:,521]+d[:,522]-d[:,524:532].sum(1)-extra)
    out['ISR_nonnegative']=float(max(0.,-z[:,537:].min()))
    out['PERK_bound']=float(max(0.,z[:,537].max()-1))
    x=z[:,541:553].reshape(-1,2,6);xb=base[541:553].reshape(2,6)
    for k,L in enumerate(LENGTHS):
        nas=(x[:,k,2]+2*x[:,k,3]+3*x[:,k,4])*L/4*b[0]
        nas0=(xb[k,2]+2*xb[k,3]+3*xb[k,4])*L/4*b[0]
        out[('ATF4','GADD34')[k]+'_nascent_balance']=err(nas-nas0+L*d[:,555+k]-d[:,553+k])
        out[('ATF4','GADD34')[k]+'_mature_balance']=err(L*d[:,539+k]-L*d[:,555+k]+d[:,557+k])
        out[('ATF4','GADD34')[k]+'_proxy_consistency']=err(z[:,534+k]-z[:,539+k]/(iv[2+k]+z[:,539+k]))
    out['ISR_reference_energy']=err(d[:,559]-4*d[:,553:555].sum(1))
    return out
