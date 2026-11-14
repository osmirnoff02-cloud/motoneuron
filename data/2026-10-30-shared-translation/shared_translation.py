"""[codex] 2026-10-30. API0.28: selected shared translation, not whole proteome.
Six new message classes share existing40S/60S and ATP with TARDBP/STMN2.
RP classes are representative chains/kit equivalents, not fused giant proteins.
"""
from dataclasses import dataclass
from pathlib import Path
import sys,numpy as np
from numba import njit
sys.path.insert(0,str(Path(__file__).parent.parent/'2026-10-30-ribosome-biogenesis'))
from biogenesis import checks as bio_checks

CLASSES=('RP40','RP60','KCNQ_WT','KCNQ_delta','PKN_full','PKN207')
FIELDS=('init40','elong0','elong1','elong2','elong3','stalled80','stalled_aa','post80','RQC60','RQC_aa','orphan_aa','incorporated_aa','completed_chains','cleared_aa','aborted_chains')
NAMES=tuple(c+'_'+f for c in CLASSES for f in FIELDS)+('shared_translation_ATP_uM','shared_recycling_ATP_uM','shared_RQC_ATP_uM','shared_replaced_background_ATP_uM')

@dataclass(frozen=True)
class SharedTranslationParameters:
    KCNQ_unit_uM:float=.01
    PKN_unit_uM:float=.01
    RP40_representative_length_aa:float=250.
    RP60_representative_length_aa:float=250.
    KCNQ_WT_length_aa:float=872.
    KCNQ_delta_length_aa:float=872.
    PKN_full_length_aa:float=942.
    PKN207_length_aa:float=207.
    provenance:str='KCNQ2 O43526=872aa and PKN1 Q16512=942aa verified UniProt; PKN207=207aa verified Yang2026 full PDF p9. KCNQ delta length remains a cost proxy until exact isoform sequence matched. Concentrations/RP representative lengths unknown. Slomnicki2016: rat S6/S14 andL4 effects differ with cell age, not a uniform human knockdown calibration. RP kits use prior effective synthesis costs/4 as assumed residue equivalents.'
    def vector(self):
        a=np.array([getattr(self,k) for k in list(self.__dataclass_fields__)[:-1]])
        if not np.isfinite(a).all() or min(a)<=0 or not self.provenance.strip():raise ValueError('Invalid shared translation parameters')
        return a

SHARED_RANGES=dict(KCNQ_unit_uM=[.001,.01,.1],PKN_unit_uM=[.001,.01,.1],RP40_representative_length_aa=[100.,250.,500.],RP60_representative_length_aa=[100.,250.,500.],KCNQ_delta_length_aa=[600.,872.],message_support=[0.,.25,1.,10.],fiveS_support=[0.,.25,1.,2.])

def controls(sg,supplied=None):
    a=np.c_[sg[:,:2],np.ones((len(sg),7))] if supplied is None else np.asarray(supplied,float)
    if a.shape!=(len(sg),9) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:8]>100) or np.any(a[:,8]>2):raise ValueError('Shared schedule: start,end,6message supports[0,100],5S[0,2]')
    return a

def setup(cyclebase,v,rv,b,nv,kp,kbase,sv):
    if min(b[13:15])<=0:raise ValueError('RP kit residue-equivalents must be positive')
    C=b[0];birth=np.log(2)/(b[1]*86400);L=sv[2:];kit=b[13:15]/4
    j=np.r_[birth*kit/L[:2],np.log(2)/kp[5]*kbase[:2]*sv[0]/C,np.log(2)/nv[3]*sv[1]/C,0.]
    x=np.zeros((6,15));x[:,0]=j*rv[0];x[:,1:5]=(j*L/v[5]/4)[:,None];x[:,7]=j*rv[1]
    cb=cyclebase.copy();cb[0]-=x[:,:6].sum()+x[:,7].sum();cb[1]-=x[:,1:6].sum()+x[:,7].sum()
    if min(cb[:2])<=0:raise ValueError('Shared basal demand exceeds selected ribosome capacity')
    # RP translation baseline is already allocated by prior biogenesis debit.
    extra0=4*C*np.sum(j[2:]*L[2:])+C*np.sum(j)*rv[5]
    return np.r_[x.ravel(),np.zeros(4)],cb,np.r_[j,L,kit,extra0]

@njit
def bound_counts(x):
    # Total40 includes initiating complexes; total60 also includes bound RQC.
    return np.array([np.sum(x[:,:6])+np.sum(x[:,7]),np.sum(x[:,1:6])+np.sum(x[:,7])+np.sum(x[:,8])])

@njit
def fluxes(z,v,rv,b,sv,ref,kp,nv,atp0,nc,cc,bc,sc,cybase,old):
    x=np.maximum(z[403:493].reshape((6,15)),0.);L=ref[6:12];C=b[0];kit=ref[12:14]
    energy=max(z[120],0)/(v[8]+max(z[120],0))*(v[8]+atp0)/atp0
    birth=np.log(2)/(b[1]*86400)
    demand=np.empty(6)
    demand[:2]=birth*kit/L[:2]*bc[1]
    demand[2:4]=np.log(2)/kp[5]*np.maximum(z[342:344],0)*sv[0]/C
    demand[4]=max(old[75],0)*sv[1]/C;demand[5]=max(old[77],0)*sv[1]/C
    load=demand*sc[:6]*max(z[330],0)/cybase[0]*energy*nc[1]
    join=x[:,0]/rv[0]*max(z[331],0)/cybase[1]*energy*cc[0]
    step=x[:,1:5]*(4*v[5]/L)[:,None]*energy*nc[2]
    stall=rv[3]*cc[3]*x[:,1:5]
    rescue=x[:,5]/rv[2]*energy*cc[2];resc_aa=x[:,6]/rv[2]*energy*cc[2]
    rec=x[:,7]/rv[1]*energy*cc[1]
    release=x[:,8]/b[6]*energy*bc[5];releaseaa=x[:,9]/b[6]*energy*bc[5]
    clear=x[:,10]/b[8]*energy*bc[4]
    dx=np.zeros((6,15));dx[:,0]=load-join
    dx[:,1]=join-step[:,0]-stall[:,0];dx[:,2:5]=step[:,:3]-step[:,1:]-stall[:,1:]
    dx[:,5]=np.sum(stall,axis=1)-rescue
    dx[:,6]=L/4*(stall[:,1]+2*stall[:,2]+3*stall[:,3])-resc_aa
    dx[:,7]=step[:,3]-rec;dx[:,8]=(1-b[7])*rescue-release
    dx[:,9]=(1-b[7])*resc_aa-releaseaa;dx[:,10]=b[7]*resc_aa+releaseaa-clear
    dx[:,11]=L/4*np.sum(step,axis=1);dx[:,12]=step[:,3];dx[:,13]=clear;dx[:,14]=rescue
    free=np.array([np.sum(rescue+rec-load),np.sum(b[7]*rescue+rec+release-join)])
    cost=np.array([4*C*np.sum(dx[:,11]),C*rv[5]*np.sum(rescue+rec),C*(b[9]*np.sum(clear)+b[10]*np.sum(release)),ref[14]])
    return dx,free,step[:,3],cost

@njit
def replace_shared(z,old,nucdx,cycdx,biodx,p,v,rv,b,sv,ref,kp,nv,atp0,nc,cc,bc,sc,cybase):
    # old includes inherited PKN synthesis and all other biological losses.
    dx,free,complete,cost=fluxes(z,v,rv,b,sv,ref,kp,nv,atp0,nc,cc,bc,sc,cybase,old)
    out=old.copy();cy=cycdx.copy();dq=biodx.copy();cy[:2]+=free
    C=b[0];kit=ref[12:14];L=ref[6:12]
    energy=max(z[120],0)/(v[8]+max(z[120],0))*(v[8]+atp0)/atp0
    birth=np.log(2)/(b[1]*86400);rpold=birth*energy*bc[1]
    rpnew=complete[:2]*L[:2]/kit
    dq[24:26]+=rpnew-rpold;dq[30:32]=rpnew
    dq[36]-=C*rpold*(b[13]+b[14]) # shared nascent debit replaces externalRP kit debit
    five=birth*energy*sc[6];oldfive=birth*energy*bc[0]
    dq[23]+=five-oldfive;dq[29]=five;dq[36]+=C*(five-oldfive)*b[12]
    pkn=complete[4:6]*C/sv[1]
    out[63]+=pkn[0]-old[75];out[75]=pkn[0]
    out[65]+=pkn[1]-old[77];out[77]=pkn[1]
    return out,cy,dq,dx,complete,cost

@njit
def append_shared(z,old,dx,complete,cost,sv,b,kp,underlying,other0):
    out=np.zeros(497);out[:403]=old;out[403:493]=dx.ravel()
    # Existing KCNQ traffic/clearance remain; replace only source terms.
    for i in range(2):
        syn=complete[2+i]*b[0]/sv[0];prior=np.log(2)/kp[5]*max(z[342+i],0)
        out[344+6*i]+=syn-prior;out[358+2*i]=syn
    out[493:496]=cost[:3]
    replaced=cost[3]*max(underlying,0)/other0 if other0>0 else 0.
    out[496]=replaced;delta=np.sum(cost[:3])-replaced
    out[120]-=delta;out[121]+=delta;out[125]+=delta
    return out

def checks(z,base,v,b,sv,ref):
    x=z[:,403:493].reshape((-1,6,15));xb=base[403:493].reshape((6,15))
    adjusted=z[:,:403].copy();initial=base[:403].copy()
    for j,a in enumerate(x):adjusted[j,330:332]+=bound_counts(a)
    initial[330:332]+=bound_counts(xb)
    out=bio_checks(adjusted,initial,v,b);err=lambda q:float(np.max(abs(q)))
    out['shared_nonnegative']=float(max(0.,-x.min()))
    for k,c in enumerate(CLASSES):
        a=x[:,k];a0=xb[k];L=ref[6+k]
        mass=L/4*(a[:,2]+2*a[:,3]+3*a[:,4])+a[:,6]+a[:,9]+a[:,10]+a[:,12]*L+a[:,13]-a[:,11]
        mass0=L/4*(a0[2]+2*a0[3]+3*a0[4])+a0[6]+a0[9]+a0[10]+a0[12]*L+a0[13]-a0[11]
        out[c+'_nascent_mass']=err(mass-mass0)
        out[c+'_stalled_bound']=float(max(0.,(a[:,6]-.75*L*a[:,5]).max(),(a[:,9]-.75*L*a[:,8]).max()))
    out['shared_translation_energy']=err(z[:,493]-base[493]-4*b[0]*(x[:,:,11].sum(1)-xb[:,11].sum()))
    for k in range(2):
        out[CLASSES[k]+'_synthesis']=err(z[:,394+k]-base[394+k]-(x[:,k,12]-xb[k,12])*ref[6+k]/ref[12+k])
        out['KCNQ_source_'+str(k)]=err((z[:,358+2*k]-base[358+2*k])*sv[0]-(x[:,2+k,12]-xb[2+k,12])*b[0])
        out['PKN_source_'+str(k)]=err((z[:,75+2*k]-base[75+2*k])*sv[1]-(x[:,4+k,12]-xb[4+k,12])*b[0])
    return out
