"""[codex] 2026-11-03. Finite RNA/RP/pre-ribosome transport sharing NCT48.

All physical amounts: uM per somatic cytoplasmic reference volume. Existing
RNA, RP inventory and premature-subunit states are projections, not extra mass.
Mechanistic topology is sourced; human MN rates and absolute pools are scenarios.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from numba import njit

ROOT=Path(__file__).resolve().parent.parent
PROTEINS=json.loads((ROOT/'2026-11-03-individual-ribosomal-proteins/protein_inputs_codex.json').read_text())['proteins']
RP_GENES=tuple(p['gene'] for p in PROTEINS)
BASE_GENES=('TARDBP','STMN2','KCNQ2_WT','KCNQ2_delta','GAP43')
CYT=np.array([50,10,342,343,605],dtype=np.int64)
RNA_FIELDS=('nuclear_free','UAP56_bound','NXF_nuclear','NXF_cytoplasmic','cytoplasmic_free','produced','removed_or_delivered_downstream','export_released')
RP_FIELDS=('cytoplasmic_free','KPNB1_cytoplasmic','KPNB1_nuclear','nuclear_free')
RIB_FIELDS=('nuclear_free','XPO1_RanGTP_nuclear','XPO1_RanGTP_cytoplasmic','cytoplasmic_free')
CARRIER_NAMES=('NXF1_NXT1_cytoplasmic','NXF1_NXT1_nuclear','UAP56_free_nuclear','XPO1_cytoplasmic','XPO1_nuclear')
COUNTER_NAMES=('UAP56_ATP','DDX19_ATP','new_RP_RNA_synthesis_ATP','ribosome_RanGTP_hydrolyzed','RP_import_released','pre40_export_released','pre60_export_released')
CONTROL_NAMES=('UAP56_loading','TREX2_NXF_handoff','RNA_pore','DDX19_release','XPO1_loading','ribosome_pore','RP_binding','RP_pore','new_RP_transcription')

@dataclass(frozen=True)
class RNAExportParameters:
    normalized_RNA_reference_uM:float=.001
    RP_RNA_reference_uM:float=.001
    NXF1_NXT1_total_uM:float=.2
    UAP56_total_uM:float=.1
    DDX19_effective_total_uM:float=.05
    XPO1_total_uM:float=.3
    UAP56_loading_uM_inv_s:float=.05
    UAP56_release_s_inv:float=.005
    TREX2_handoff_uM_inv_s:float=.1
    RNA_pore_s_inv:float=.05
    DDX19_kcat_s_inv:float=.1
    DDX19_RNA_half_uM:float=.01
    ATP_half_uM:float=500.
    UAP56_ATP_per_loading:float=1.
    DDX19_ATP_per_release:float=1.
    XPO1_loading_uM_inv2_s:float=.1
    XPO1_unbinding_s_inv:float=.001
    ribosome_pore_s_inv:float=.02
    XPO1_return_s_inv:float=.05
    XPO1_Ran_units_per_particle:float=1.
    RP_binding_uM_inv_s:float=.05
    RP_unbinding_s_inv:float=.005
    RP_pore_s_inv:float=.05
    RP_Ran_release_uM_inv_s:float=.2
    RP_RNA_half_life_h:float=6.
    nuclear_RNA_decay_s_inv:float=0.
    RP_RNA_synthesis_ATP_equiv:float=3000.
    RNA_pore_load_weight:float=1.
    ribosome_pore_load_weight:float=10.
    provenance:str='U41 scenario parameters; Lin2018/Hohmann2026 UAP56/NXF/DDX19; Trotta2003/Zemp2009/Junod2023 XPO1/Ran and cytoplasmic maturation. KPNB1 is one selected effective RP route, not proven common specificity for80 genes. No TPM-to-rate conversion.'
    def vector(self):
        v=np.array([getattr(self,k) for k in self.__dataclass_fields__ if k!='provenance'],float)
        if not np.isfinite(v).all() or np.any(v<0) or min(v[[0,1,11,12,24]])<=0 or v[19]<1 or not self.provenance.strip():
            raise ValueError('Finite nonnegative RNA parameters; positive scales/half-times, >=1 XPO1/Ran units')
        return v

RANGES=dict(normalized_RNA_reference_uM=[.0001,.001,.01],RP_RNA_reference_uM=[.0001,.001,.01],
 NXF1_NXT1_total_uM=[.02,.2,2.],UAP56_total_uM=[.01,.1,1.],DDX19_effective_total_uM=[.005,.05,.5],XPO1_total_uM=[.03,.3,3.],
 UAP56_loading_uM_inv_s=[.005,.05,.5],UAP56_release_s_inv=[.0005,.005,.05],TREX2_handoff_uM_inv_s=[.01,.1,1.],
 RNA_pore_s_inv=[.005,.05,.5],DDX19_kcat_s_inv=[.01,.1,1.],DDX19_RNA_half_uM=[.001,.01,.1],ATP_half_uM=[100.,500.,2000.],
 UAP56_ATP_per_loading=[1.,2.,10.],DDX19_ATP_per_release=[1.,2.,10.],XPO1_loading_uM_inv2_s=[.01,.1,1.],
 XPO1_unbinding_s_inv=[.0001,.001,.01],ribosome_pore_s_inv=[.002,.02,.2],XPO1_return_s_inv=[.005,.05,.5],
 XPO1_Ran_units_per_particle=[1.,2.,4.],RP_binding_uM_inv_s=[.005,.05,.5],RP_unbinding_s_inv=[.0005,.005,.05],
 RP_pore_s_inv=[.005,.05,.5],RP_Ran_release_uM_inv_s=[.02,.2,2.],RP_RNA_half_life_h=[1.,6.,24.],
 nuclear_RNA_decay_s_inv=[0.,1e-5,1e-4],RP_RNA_synthesis_ATP_equiv=[1000.,3000.,10000.],
 RNA_pore_load_weight=[1.,5.,10.],ribosome_pore_load_weight=[1.,10.,50.],support=[0.,.25,1.,2.])

def layout(with_rp):
    nr=85 if with_rp else 5; np_=80 if with_rp else 0
    rp=8*nr; rib=rp+4*np_;car=rib+8;count=car+5
    return np.array([nr,np_,rp,rib,car,count,count+7],dtype=np.int64)

def names(shape):
    nr,np_,rp,rib,car,count,size=shape
    genes=BASE_GENES+(RP_GENES if np_ else ())
    return tuple('nrt_'+g+'_'+f+'_uM' for g in genes for f in RNA_FIELDS)+tuple('nrt_'+g+'_'+f+'_uM' for g in RP_GENES[:np_] for f in RP_FIELDS)+tuple('nrt_pre'+s+'_'+f+'_uM' for s in ('40S','60S') for f in RIB_FIELDS)+tuple('nrt_'+s+'_uM' for s in CARRIER_NAMES+COUNTER_NAMES)

def controls(sg,supplied=None):
    a=np.c_[sg[:,:2],np.ones((len(sg),9))] if supplied is None else np.asarray(supplied,float)
    if a.shape!=(len(sg),11) or not np.isfinite(a).all() or not np.array_equal(a[:,:2],sg[:,:2]) or np.any(a[:,2:]<0) or np.any(a[:,2:]>2):
        raise ValueError('RNA export schedule: matching times and nine supports in [0,2]')
    return a.copy()

def initialize(base,v,nctv,C,shape,rppv):
    x=np.zeros(shape[6]);nr,np_,rp,rib,car,count,size=shape;rna=x[:rp].reshape(nr,8)
    scales=np.array([v[0]]*4+[1.])
    rna[:5,4]=np.maximum(base[CYT],0.)*scales
    rna[0,0]=base[322]*scales[0];rna[1,0]=base[324]*scales[1]
    if np_:
        rna[5:,4]=v[1]*rppv[0]
        x[rp:rib].reshape(np_,4)[:,3]=base[1600:1680]
    # Inherited mature-ready precursor pools begin cytoplasmic; new assembly
    # enters the nucleus. No claim of initial equilibrium of the new network.
    x[rib:car].reshape(2,4)[:,3]=C*base[390:392]
    r=nctv[0]
    x[car:car+5]=[v[2]/(1+r),v[2]*r/(1+r),v[3],v[5]/(1+r),v[5]*r/(1+r)]
    return np.r_[base,x]

@njit
def extra_load(x,v,r,shape):
    nr,np_,rp,rib,car,count,size=shape
    total=x[car]+x[car+1]/r+x[car+3]+x[car+4]/r
    for i in range(nr):total+=v[27]*(x[8*i+2]/r+x[8*i+3])
    for i in range(np_):total+=x[rp+4*i+1]+x[rp+4*i+2]/r
    for i in range(2):total+=v[28]*(x[rib+4*i+1]/r+x[rib+4*i+2])
    return total

@njit
def common_pore(x,nct,v,nv,shape):
    own=0.
    for i in (4,6,8,10,12,14,20,22):own+=max(nct[i],0.)+max(nct[i+1],0.)/nv[0]
    extra=extra_load(np.maximum(x,0.),v,nv[0],shape)
    gate=1/(1+(own+extra)/nv[19])
    return gate,gate*(1+own/nv[19])

@njit
def host_views(z,v,shape,off,C):
    """Nuclear assembly and cytoplasmic maturation use different pools."""
    nr,np_,rp,rib,car,count,size=shape;x=np.maximum(z[off:off+size],0.)
    a=z.copy();cy=z.copy()
    for s in range(2):a[390+s]=x[rib+4*s+3]/C;cy[390+s]=a[390+s]
    if np_:
        for i in range(np_):a[1600+i]=x[rp+4*i+3]
    return a,cy

@njit
def message_controls(z,pv,control,v,shape,off):
    a=pv.copy();c=control.copy()
    for i in range(shape[1]):a[0,i]=1.;c[i]=max(z[off+8*(5+i)+4],0.)/v[1]
    return a,c

@njit
def referenced_message_controls(z,pv,control,v,shape,off,reference_cyt_uM):
    """Q3 conditional gene-specific loading efficiencies at basal cytoplasmic RNA.

    Absolute RNA abundances do not identify absolute protein-production rates.
    Each reference is fixed at initialization; perturbations are never renormalized.
    """
    a=pv.copy();c=control.copy()
    for i in range(shape[1]):
        a[0,i]=1.
        c[i]=control[i]*max(z[off+8*(5+i)+4],0.)/reference_cyt_uM[i]
    return a,c

@njit
def chemistry(x,nct,ATP,v,nv,ctl,nctl,shape):
    """Transport only; returns RNA/RP/particle rates and SAME Ran-pool deltas."""
    x=np.maximum(x,0.);n=np.maximum(nct,0.);d=np.zeros(shape[6]);dn=np.zeros(33)
    nr,np_,rp,rib,car,count,size=shape;r=nv[0];e=max(ATP,0.)/(v[12]+max(ATP,0.))
    gate,_=common_pore(x,n,v,nv,shape);gate*=nctl[1]
    bound=0.
    for i in range(nr):bound+=x[8*i+3]
    for i in range(nr):
        k=8*i
        load=v[6]*ctl[0]*e*x[k]*x[car+2]/r
        unbind=v[7]*x[k+1]
        hand=v[8]*ctl[1]*x[k+1]*x[car+1]/r
        passage=v[9]*ctl[2]*gate*(x[k+2]/r-x[k+3])
        release=v[10]*v[4]*ctl[3]*e*x[k+3]/(v[11]+bound)
        d[k]+=-load+unbind;d[k+1]+=load-unbind-hand
        d[k+2]+=hand-passage;d[k+3]+=passage-release;d[k+4]+=release;d[k+7]+=release
        d[car+2]+=-load+unbind+hand;d[car+1]-=hand;d[car]+=release
        d[count]+=v[13]*load;d[count+1]+=v[14]*release
    ret=v[9]*gate*(x[car]-x[car+1]/r)
    d[car]-=ret;d[car+1]+=ret
    # Explicit RP-KPNB1 complexes borrow existing KPNB1; Ran unloads them.
    for i in range(np_):
        k=rp+4*i
        bc=v[20]*ctl[6]*x[k]*n[4];bn=v[20]*ctl[6]*x[k+3]*n[5]/r
        uc=v[21]*x[k+1];un=v[21]*x[k+2]
        move=v[22]*ctl[7]*gate*(x[k+1]-x[k+2]/r)
        rc=v[23]*x[k+1]*n[18];rn=v[23]*x[k+2]*n[19]/r
        d[k]+=-bc+uc+rc;d[k+1]+=bc-uc-move-rc
        d[k+2]+=bn-un+move-rn;d[k+3]+=-bn+un+rn
        dn[4]+=-bc+uc;dn[5]+=-bn+un;dn[18]-=rc;dn[19]-=rn;dn[10]+=rc;dn[11]+=rn
        d[count+4]+=rn
    # Coarse stoichiometry n>=1 permits multi-XPO1 sensitivity; not a measured
    # receptor count. A bounded loading law vanishes when any reactant is zero.
    units=v[19]
    for s in range(2):
        k=rib+4*s
        load=v[15]*ctl[4]*x[k]*(x[car+4]/units/r)*(n[19]/units/r)
        unbind=v[16]*x[k+1]
        move=v[17]*ctl[5]*gate*(x[k+1]/r-x[k+2])
        hydro=nv[15]*nctl[3]*x[k+2]
        d[k]+=-load+unbind;d[k+1]+=load-unbind-move;d[k+2]+=move-hydro;d[k+3]+=hydro
        d[car+4]+=units*(-load+unbind);d[car+3]+=units*hydro
        dn[19]+=units*(-load+unbind);dn[16]+=units*hydro;dn[26]+=units*hydro
        d[count+3]+=units*hydro;d[count+5+s]+=hydro
    ret=v[18]*gate*(x[car+3]-x[car+4]/r)
    d[car+3]-=ret;d[car+4]+=ret
    return d,dn

@njit
def finish(z,old,v,nv,ctl,nctl,shape,off,noff,C,nucv,kp,assembly,rppv,rppcontrol):
    x=np.maximum(z[off:off+shape[6]],0.);nr,np_,rp,rib,car,count,size=shape
    d,dn=chemistry(x,z[noff:noff+33],z[120],v,nv,ctl,nctl,shape)
    out=np.zeros(off+size);out[:off]=old
    scales=np.array([v[0],v[0],v[0],v[0],1.]);src=np.zeros(nr);sink=np.zeros(nr)
    # Preserve inherited processing/splicing, transcript-specific cytoplasmic
    # degradation and GAP43 axonal departure. Only replace transport sources.
    oldexport0=max(z[322],0.)/nucv[2];oldexport1=max(z[324],0.)/nucv[2]
    src[0]=(old[322]+oldexport0)*v[0];src[1]=(old[324]+oldexport1)*v[0]
    sink[0]=(oldexport0-old[50])*v[0];sink[1]=(oldexport1-old[10])*v[0]
    kr=np.log(2)/kp[4]
    for j in (2,3):src[j]=(old[342+j-2]+kr*max(z[342+j-2],0.))*v[0];sink[j]=kr*x[8*j+4]
    src[4]=max(old[608],0.);sink[4]=src[4]-old[605]
    kd=np.log(2)/(v[24]*3600)
    if np_:
        for i in range(np_):src[5+i]=kd*v[1]*rppv[0,i]*rppcontrol[i]*ctl[8]*max(z[120],0.)/(v[12]+max(z[120],0.));sink[5+i]=kd*x[8*(5+i)+4]
        d[count+2]=np.sum(src[5:])*v[26]
    for i in range(nr):
        k=8*i;source=max(src[i],0.);loss=max(sink[i],0.)
        d[k]+=source;d[k+4]-=loss;d[k+5]+=source;d[k+6]+=loss
        for j in range(4):
            # Nuclear decay excludes receptor-bound cytoplasmic export complex.
            if j==3:continue
            decay=v[25]*x[k+j];d[k+j]-=decay;d[k+6]+=decay
            if j==1:d[car+2]+=decay
            if j==2:d[car+1]+=decay
    for i in range(5):out[CYT[i]]=d[8*i+4]/scales[i]
    out[322]=d[0]/v[0];out[324]=d[8]/v[0]
    out[608]=d[8*4+7] # delivered to soma; original transcription debit retained
    # New RP compartment pools partition the old per-gene unassembled stock.
    for i in range(np_):
        k=rp+4*i;made=max(old[1840+i],0.);built=max(old[1920+i],0.)
        loss=max(made-built-old[1600+i],0.);total=np.sum(x[k:k+4])
        d[k]+=made;d[k+3]-=built
        for j in range(4):
            rem=loss*x[k+j]/total if total>0 else 0.;d[k+j]-=rem
            if j==1:dn[4]+=rem
            if j==2:dn[5]+=rem
    for s in range(2):
        k=rib+4*s;source=C*max(assembly[s],0.);total=np.sum(x[k:k+4])
        # Host total already includes actual maturation/local departure and
        # irreversible precursor damage. These sinks may only take free cyt.
        damage=C*max(old[714+s],0.)
        sink=max(source-C*old[390+s]-damage,0.)
        d[k+3]-=sink
        for j in range(4):
            loss=damage*x[k+j]/total if total>0 else 0.;d[k+j]-=loss
            if j==1:d[car+4]+=v[19]*loss;dn[19]+=v[19]*loss
            if j==2:d[car+3]+=v[19]*loss;dn[18]+=v[19]*loss
        d[k]+=source
    debit=d[count]+d[count+1]+d[count+2]
    out[120]-=debit;out[121]+=debit;out[125]+=debit
    out[noff:noff+33]+=dn;out[off:]=d
    return out

def nct_check_view(z,base,v,shape,off,noff):
    zz=z.copy();bb=base.copy();nr,np_,rp,rib,car,count,size=shape
    for rows in (zz,bb[None,:]):
        x=rows[:,off:off+size]
        if np_:
            q=x[:,rp:rib].reshape(len(x),np_,4);rows[:,noff+4]+=q[:,:,1].sum(1);rows[:,noff+5]+=q[:,:,2].sum(1)
        q=x[:,rib:car].reshape(-1,2,4)
        rows[:,noff+19]+=v[19]*q[:,:,1].sum(1);rows[:,noff+18]+=v[19]*q[:,:,2].sum(1)
    return zz,bb

def checks(z,base,v,shape,off,noff,C,lmrv):
    nr,np_,rp,rib,car,count,size=shape;x=z[:,off:off+size];b=base[off:off+size];dx=x-b;rna=x[:,:rp].reshape(-1,nr,8);dr=dx[:,:rp].reshape(-1,nr,8)
    err=lambda a:float(np.max(np.abs(a)))
    out={'nrt_nonnegative':max(0.,float(-x.min())), 'nrt_each_RNA_lineage':err(dr[:,:,:5].sum(2)+dr[:,:,6]-dr[:,:,5]),
         'nrt_NXF_conserved':err(dx[:,car]+dx[:,car+1]+dr[:,:,2:4].sum((1,2))),
         'nrt_UAP56_conserved':err(dx[:,car+2]+dr[:,:,1].sum(1)),
         'nrt_XPO1_conserved':err(dx[:,car+3]+dx[:,car+4]+v[19]*dx[:,rib:car].reshape(-1,2,4)[:,:,1:3].sum((1,2))),
         'nrt_RNA_cyt_projections':err(z[:,CYT]*np.array([v[0]]*4+[1.])-rna[:,:5,4]),
         'nrt_RNA_nuc_projections':err(z[:,[322,324]]*v[0]-rna[:,:2,0]),
         'nrt_pre_ribosome_projections':err(C*z[:,390:392]-x[:,rib:car].reshape(-1,2,4).sum(2)),
         'nrt_GAP43_transcription_cost':err(z[:,614]-base[614]-lmrv[6]*dr[:,4,5]),
         'nrt_RP_RNA_transcription_cost':err(dx[:,count+2]-v[26]*dr[:,5:,5].sum(1))}
    if np_:out['nrt_each_RP_projection']=err(z[:,1600:1680]-x[:,rp:rib].reshape(len(x),np_,4).sum(2))
    return out

def revise_host_checks(ch,z,base,v,shape,off,lmrv):
    ch=dict(ch);d=z-base;nr,np_,rp,rib,car,count,size=shape
    q=d[:,off:off+rp].reshape(-1,nr,8)
    # The historical KCNQ ledger counted only cytoplasm; RNA source/decay
    # counters remain historical. Added nuclear RNA is included explicitly.
    ch['KCNQ_RNA_balance']=float(np.max(np.abs(d[:,342:344].sum(1)+q[:,2:4,:4].sum((1,2))/v[0]-d[:,356]+d[:,357]+q[:,2:4,6].sum(1)/v[0]-d[:,357])))
    ch['mRNA_transcription_cost']=float(np.max(np.abs(d[:,614]-lmrv[6]*q[:,4,5])))
    return ch

def observe(z,v,shape,off,noff,parameters):
    nr,np_,rp,rib,car,count,size=shape;x=z[:,off:off+size]
    return dict(genes=BASE_GENES+(RP_GENES if np_ else ()),RNA_fields=RNA_FIELDS,RNA_pools_counters=x[:,:rp].reshape(-1,nr,8),
      RP_genes=RP_GENES[:np_],RP_pools=x[:,rp:rib].reshape(len(x),np_,4),pre_ribosome_fields=RIB_FIELDS,pre_ribosome_pools=x[:,rib:car].reshape(-1,2,4),
      carriers=x[:,car:count],carrier_names=CARRIER_NAMES,counters=x[:,count:],counter_names=COUNTER_NAMES,
      parameters=asdict(parameters),human_MN_calibrated=False,
      scope='NXF1/NXT1-UAP56-DDX19 exports five selected RNA classes plus optional80 RP mRNAs without direct Ran consumption. RP import borrows finite KPNB1; pre40/60 XPO1 complexes borrow the SAME RanGTP, recycle via RanGAP/NTF2/RCC1 and the existing NDPK. Cytoplasmic precursor availability gates maturation/local delivery. Unknown human rates; no other RNA classes, all karyopherin paralogs, explicit adaptor recycling or full transcriptome.')
