"""[codex] 2026-10-30. Selected mature residue recycling and nutrient response.
Existing proteins own their mass. This module receives only identified removal
fluxes; cleavage, generic cargo and fragment-lineage counts are NOT AA sources.
All pools use somatic-volume equivalents; lysosomal lumen concentration differs.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
_data=Path(__file__).resolve().parent.parent
for _folder in ('2026-10-30-translation-substrates','2026-10-26-proteasome-energy'):
    sys.path.insert(0,str(_data/_folder))
import numpy as np
from numba import njit
from substrates import checks as substrate_checks, gate as substrate_gate
from proteostasis_parameters import proteasome_factor

SOURCES=('intact_TDP','STMN2','RP40','RP60','somatic_KCNQ_WT','somatic_KCNQ_delta','PKN_full','PKN207')
NAMES=('cytosolic_recycling_peptides_uM_aa','lysosomal_recycling_peptides_uM_aa','lysosomal_free_AA_soma_equivalent_uM',
 'mature_cytosolic_input_uM_aa','mature_lysosomal_input_uM_aa','cytosolic_peptides_hydrolyzed_uM_aa',
 'lysosomal_peptides_hydrolyzed_uM_aa','lysosomal_AA_exported_uM','mature_AA_returned_uM','mature_AA_unrecovered_uM','replaced_external_AA_uM')
NAMES+=tuple(s+'_removed_uM_aa' for s in SOURCES)
NAMES+=('GCN2_activity_proxy','eIF2alpha_phosphorylated_fraction_proxy','ATF4_nuclear_response_proxy','GADD34_response_proxy','mTORC1_nutrient_activity_proxy')
assert len(NAMES)==24

@dataclass(frozen=True)
class NuclearRecyclingParameters:
    lysosomal_route_fraction:float=.5
    recoverable_fraction:float=.8
    cytosolic_peptidase_s:float=300.
    lysosomal_hydrolysis_s:float=1800.
    lysosomal_export_s:float=900.
    GCN2_response_s:float=300.
    eIF2_phosphorylation_s:float=300.
    eIF2_dephosphorylation_s:float=900.
    ATF4_response_s:float=1800.
    GADD34_response_s:float=3600.
    GADD34_feedback_gain:float=4.
    ATF4_half_pEIF2:float=.2
    residual_initiation:float=.1
    mTOR_response_s:float=300.
    mTOR_half_AA_uM:float=200.
    shortage_stall_gain:float=1.
    nutrient_response_enabled:bool=True
    provenance:str='Wyant2017/AbuRemaileh2017 HEK293T separate lysosomal hydrolysis and export; Nofal2022 mouse pancreatic cancer and Darnell2018 human non-neuronal lines distinguish initiation, elongation and output. GADD34 feedback qualitative Novoa. All human MN rates, routes and response gains hypothetical. Activity proxies are not additional absolute protein pools. Existing clearance/background ATP budget retained, no ATP returned by hydrolysis and no duplicate clearance charge.'
    def vector(self):
        a=np.array([getattr(self,n) for n in list(self.__dataclass_fields__)[:-1]])
        if not isinstance(self.nutrient_response_enabled,bool) or not np.isfinite(a).all() or min(a)<0 or max(a[:2])>1 or min(a[2:10])<=0 or a[11]<=0 or a[12]>1 or min(a[13:15])<=0 or not self.provenance.strip():raise ValueError('Invalid recycling/response parameters')
        return a

RECYCLING_RANGES=dict(lysosomal_route_fraction=[0.,.5,1.],recoverable_fraction=[0.,.5,.8,1.],cytosolic_peptidase_s=[30.,300.,3000.],lysosomal_hydrolysis_s=[300.,1800.,14400.],lysosomal_export_s=[60.,900.,7200.],GCN2_response_s=[30.,300.,1800.],eIF2_phosphorylation_s=[30.,300.,1800.],eIF2_dephosphorylation_s=[300.,900.,7200.],ATF4_response_s=[300.,1800.,7200.],GADD34_response_s=[900.,3600.,14400.],GADD34_feedback_gain=[0.,4.,10.],ATF4_half_pEIF2=[.05,.2,.5],residual_initiation=[0.,.1,.5],mTOR_response_s=[30.,300.,1800.],mTOR_half_AA_uM=[20.,200.,2000.],shortage_stall_gain=[0.,1.,5.],lysosomal_hydrolysis_support=[0.,.25,1.],lysosomal_export_support=[0.,.25,1.],cytosolic_hydrolysis_support=[0.,.25,1.],GCN2_support=[0.,1.],GADD34_support=[0.,1.],mTOR_support=[0.,1.])

RECYCLING_SWITCH_RANGES=dict(nutrient_response_enabled=[False,True])

def controls(sg,supplied=None):
    a=np.c_[sg[:,:2],np.ones((len(sg),6))] if supplied is None else np.asarray(supplied,float)
    if a.shape!=(len(sg),8) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>1):raise ValueError('Recycling schedule: start,end,cyto_hydro,lyso_hydro,lyso_export,GCN2,GADD34,mTOR supports[0,1]')
    return a

@njit
def removal_sources(z,old,tdp_intact_residue_flux,b,v,nv,sv,kp):
    """Gross existing loss, not net disappearance or protease cleavage."""
    q=np.zeros(8);q[0]=tdp_intact_residue_flux
    q[1]=np.log(2)/v[11]*max(z[11],0.)*v[9]*v[10]
    q[2]=max(old[398],0.)*b[0]*b[13]/4
    q[3]=max(old[399],0.)*b[0]*b[14]/4
    for i in range(2):
        # Only ER/Golgi/somatic membrane; process and transit cargo excluded.
        off=344+6*i
        q[4+i]=np.log(2)/kp[5]*np.maximum(z[off:off+3],0.).sum()*sv[0]*sv[4+i]
    q[6]=np.log(2)/nv[3]*max(z[63],0.)*sv[1]*sv[6]
    q[7]=np.log(2)/nv[4]*max(z[65],0.)*sv[1]*sv[7]
    return q

@njit
def intact_TDP_loss(z,p,nv,prv,wv,ATP0,unit,row,support,folding,efficiency,ATPase_support):
    # Same removal law as _network_rhs + work_loss, only U/P full-length.
    ups=1.
    if prv[0]>0:
        atp=ATP0 if wv[0]>0 and wv[6]==0 else z[120]
        work=efficiency*ATPase_support if wv[0]>0 else 1.
        ups=proteasome_factor(atp,ATP0,prv,support*work,folding)
    lyso=max(z[72],0.)*((1-nv[21])+nv[21]*max(z[68],0.))*row[12]
    normal=(1-nv[19])*row[10]+nv[19]*lyso if nv[25]>0 else 1.
    effective=normal+(1-nv[19])*row[10]*(ups-1.)
    return max(p[14]*(z[6]+z[7])*effective,0.)*unit*414.

def initial(source0,v,subref):
    j=float(np.sum(source0));returned=v[1]*j
    if returned>subref[0]+1e-12:raise ValueError('Mature AA return exceeds basal incorporation: revise source ownership, initial turnover or external supply')
    q=np.zeros(24);q[:3]=[j*(1-v[0])*v[2],j*v[0]*v[3],j*v[0]*v[1]*v[4]];q[23]=1.
    return q,np.array([returned,j])

@njit
def modulate(z,v,sub,rv,nc,cc,bc,sc):
    nc=nc.copy();cc=cc.copy();bc=bc.copy();sc=sc.copy()
    if v[16]==0:return nc,cc,bc,sc
    pe=min(1.,max(z[533],0.));mt=min(1.,max(z[536],0.))
    nc[1]*=mt*(v[12]+(1-v[12])*(1-pe))
    # Nuclear PolI/III and TOP-like RP response are effective, not fitted.
    bc[0]*=mt;sc[6]*=mt;sc[:2]*=mt
    deficiency=max(0.,1-max(z[498],0.)/(sub[1]*sub[2]))
    bound=max(z[326]+z[327],0.)+max(z[403:493].reshape((6,15))[:,1:5].sum(),0.)
    cc[3]+=v[15]*deficiency**2*min(bound,1.)
    return nc,cc,bc,sc

@njit
def append_recycling(z,old,source,v,ref,sub,nuc,ATP0,ctl,AA_support,lyso_capacity):
    q=np.zeros(24);P=max(z[513],0.);L=max(z[514],0.);A=max(z[515],0.)
    cy=(1-v[0])*source.sum();ly=v[0]*source.sum()
    energy=max(z[120],0.)/(nuc[8]+max(z[120],0.))*(nuc[8]+ATP0)/ATP0
    hc=P/v[2]*ctl[0];hl=L/v[3]*ctl[1]*max(lyso_capacity,0.)
    ex=A/v[4]*ctl[2]*min(1.,energy)
    returned=v[1]*hc+ex;lost=(1-v[1])*(hc+hl);credit=ref[0]*AA_support
    q[:3]=[cy-hc,ly-hl,v[1]*hl-ex]
    q[3:11]=np.array([cy,ly,hc,hl,ex,returned,lost,credit]);q[11:19]=source
    x=np.minimum(np.maximum(z[532:537],0.),1.)
    lack=max(0.,1-max(z[498],0.)/(sub[1]*sub[2]))
    q[19]=(lack*ctl[3]*min(1.,energy)-x[0])/v[5]
    ph=x[0]*(1-x[1])*min(1.,energy)/v[6];deph=x[1]*(1+v[10]*x[3])/v[7]
    q[20]=ph-deph
    # Effective response needs substrate availability; no free protein synthesis.
    competence=substrate_gate(z,sub)*min(1.,energy)
    q[21]=(x[1]/(v[11]+x[1])*competence-x[2])/v[8]
    q[22]=(x[2]*competence*ctl[4]-x[3])/v[9]
    a=max(z[497],0.);target=min(1.,a/(v[14]+a)*(v[14]+sub[0])/sub[0])*ctl[5]
    q[23]=(target-x[4])/v[13]
    out=np.zeros(537);out[:513]=old;out[513:]=q
    out[497]+=returned-credit;out[501]-=credit
    return out

def checks(z,base,v,sub):
    q=z[:,513:537];d=q-base[513:537];err=lambda x:float(np.max(abs(x)))
    out=substrate_checks(z,base,sub)
    s=z[:,497:513]-base[497:513]
    out['AA_open_balance']=err(s[:,0]+s[:,1]+s[:,8]+s[:,5]-s[:,4]-s[:,6]-d[:,8])
    out.update(recycling_nonnegative=float(max(0.,-q.min())),response_bounds=float(max(0.,q[:,19:].max()-1)),
      recycled_material_balance=err(d[:,:3].sum(1)+d[:,8]+d[:,9]-d[:,11:19].sum(1)),
      source_partition=err(d[:,3]+d[:,4]-d[:,11:19].sum(1)),
      cytosolic_peptide_balance=err(d[:,0]-d[:,3]+d[:,5]),
      lysosomal_peptide_balance=err(d[:,1]-d[:,4]+d[:,6]),
      lysosomal_free_AA_balance=err(d[:,2]-v[1]*d[:,6]+d[:,7]),
      mature_AA_return=err(d[:,8]-v[1]*d[:,5]-d[:,7]),
      unrecoverable_AA=err(d[:,9]-(1-v[1])*(d[:,5]+d[:,6])))
    return out
