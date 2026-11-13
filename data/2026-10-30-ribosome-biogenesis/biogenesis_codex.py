"""[codex] 2026-10-30. API0.27, selected somatic pool; no human kinetic fit.
Four elongation bins charge residues before completion; rescue may retain60S.
Lumped nucleolar supply is external to the selected TARDBP/STMN2 pool.
"""
from dataclasses import dataclass
from pathlib import Path
import sys,numpy as np
from numba import njit
sys.path.insert(0,str(Path(__file__).parent.parent/'2026-10-30-nuclear-targets-cycle'))
from nuclear_cycle_codex import initial_cycle,replace_cycle_targets

NAMES=tuple('nascent_'+t+'_bin'+str(k) for t in ('TARDBP','STMN2') for k in (1,2,3))
NAMES+=tuple(q+'_'+t for q in ('stalled_aa','RQC_bound_60S','RQC_bound_aa','orphan_aa','incorporated_aa','completed_aa','cleared_aa') for t in ('TARDBP','STMN2'))
NAMES+=('pre47S_equivalents','small_rRNA_equivalents','large_rRNA_equivalents','free5S_equivalents','RP40_kit_equivalents','RP60_kit_equivalents','premature40S','premature60S')
NAMES+=('pre47S_produced','5S_produced','RP40_kits_produced','RP60_kits_produced','mature40S_produced','mature60S_produced','mature40S_retired','mature60S_retired','biogenesis_ATP_uM','RQC_orphan_ATP_uM','biogenesis_replaced_background_ATP_uM')
assert len(NAMES)==39

@dataclass(frozen=True)
class BiogenesisParameters:
    selected_mature_pool_uM:float=.1
    mature_turnover_half_days:float=7.
    precursor_processing_s:float=600.
    assembly_s:float=1800.
    maturation40_s:float=1800.
    maturation60_s:float=3600.
    RQC_release_s:float=300.
    rescue_direct60_fraction:float=0.
    orphan_clearance_s:float=1800.
    orphan_ATP_per_aa:float=.2
    RQC_ATP_per_complex:float=2.
    pre47S_ATP_equivalents:float=27000.
    fiveS_ATP_equivalents:float=240.
    RP40_kit_ATP_equivalents:float=30000.
    RP60_kit_ATP_equivalents:float=50000.
    assembly_ATP_per_subunit:float=500.
    provenance:str='Pisareva2011 rescue substrate restrictions and long-chain caveat; Popov2013 early pre-rRNA processing and Slomnicki2016 rat neuronal biogenesis are separate references. All human rates/capacities and effective bio/RQC costs are assumptions. Selected somatic pool, four-bin elongation, upstream RP production not self-replicating.'
    def vector(self):
        names=list(self.__dataclass_fields__)[:-1];a=np.array([getattr(self,n) for n in names])
        if not np.isfinite(a).all() or min(a)<0 or min(a[[0,1,2,3,4,5,6,8]])<=0 or a[7]>1 or not self.provenance.strip():raise ValueError('Invalid biogenesis parameters')
        return a

BIOGENESIS_RANGES=dict(selected_mature_pool_uM=[.03,.1,.3],mature_turnover_half_days=[1.,7.,21.],
 precursor_processing_s=[300.,600.,3600.],assembly_s=[600.,1800.,7200.],maturation40_s=[600.,1800.,7200.],maturation60_s=[600.,3600.,14400.],
 RQC_release_s=[30.,300.,3000.],rescue_direct60_fraction=[0.,.5,1.],orphan_clearance_s=[300.,1800.,7200.],orphan_ATP_per_aa=[0.,.2,2.],
 RQC_ATP_per_complex=[0.,2.,10.],pre47S_ATP_equivalents=[13000.,27000.,54000.],fiveS_ATP_equivalents=[120.,240.,480.],
 RP40_kit_ATP_equivalents=[15000.,30000.,60000.],RP60_kit_ATP_equivalents=[25000.,50000.,100000.],assembly_ATP_per_subunit=[0.,500.,5000.],
 rRNA_support=[0.,.25,1.,2.],RP_supply_support=[0.,.25,1.,2.],assembly_support=[0.,.25,1.],maturation_support=[0.,.25,1.],orphan_support=[0.,.25,1.],RQC_support=[0.,.25,1.],retirement_support=[0.,1.,2.])

def prepare(v,rv,p,budget,b):
    if budget[9]<=0 or budget[10]!=414 or budget[11]!=4:raise ValueError('Biogenesis requires positive TDP concentration, canonical414aa and4 high-energy equivalents/aa')
    v=v.copy();rv=rv.copy();rv[4]=b[0]
    # One concentration scale for both message classes, derived from gross flux.
    k=v[5]/np.array([414.,v[10]])
    flux=np.array([p[20]*budget[9],np.log(2)/v[11]*v[9]])
    v[6:8]=flux/(k*b[0])
    initial_cycle(v,rv) # rejects insufficient capacity rather than renormalizing
    return v,rv

def initial(v,rv,b):
    q=np.zeros(39);q[:6]=np.repeat(v[6:8]/4,3)
    birth=np.log(2)/(b[1]*86400)
    q[20:28]=birth*np.array([b[2],b[3],b[3],b[3],b[3],b[3],b[4],b[5]])
    return q

def controls(segments,supplied=None):
    a=np.c_[segments[:,:2],np.ones((len(segments),7))] if supplied is None else np.asarray(supplied,float)
    if a.shape!=(len(segments),9) or not np.array_equal(a[:,:2],segments[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>2):raise ValueError('Bio schedule: start/end,rRNA,RP,assembly,maturation,orphan,RQC,retirement supports [0,2]')
    return a

@njit
def mature_counts(z):
    x=z[330:342];e=z[326:328]
    return np.array([x[0]+sum(x[2:8])+sum(e),x[1]+sum(x[4:8])+sum(e)+sum(z[372:374])])

@njit
def bio_fluxes(z,v,rv,b,ATP0,nc,cc,bc,cybase,biobase):
    C=b[0];q=np.maximum(z[364:403],0.);x=np.maximum(z[330:342],0.);elong=np.maximum(z[326:328],0.)
    energy=max(z[120],0.)/(v[8]+max(z[120],0.))*(v[8]+ATP0)/ATP0
    k=v[5]/np.array([414.,v[10]]);length=np.array([414.,v[10]]);c0=k*v[6:8]
    load=c0*np.maximum(np.array([z[50],z[10]]),0.)*x[0]/cybase[0]*energy*nc[1]
    join=x[2:4]/rv[0]*x[1]/cybase[1]*energy*cc[0]
    stall_rate=rv[3]*cc[3];rescue=x[4:6]/rv[2]*energy*cc[2];recycle=x[6:8]/rv[1]*energy*cc[1]
    dq=np.zeros(39);cy=np.zeros(12);complete=np.zeros(2);incorporated=np.zeros(2);stalled_mass=np.zeros(2)
    for t in range(2):
        off=3*t;bins=np.array([max(elong[t]-sum(q[off:off+3]),0.),q[off],q[off+1],q[off+2]])
        step=4*k[t]*energy*nc[2]*bins;complete[t]=step[3]
        dq[off:off+3]=step[:3]-step[1:]-stall_rate*bins[1:]
        incorporated[t]=length[t]/4*sum(step)
        stalled_mass[t]=stall_rate*length[t]/4*(bins[1]+2*bins[2]+3*bins[3])
    stall=stall_rate*elong
    rescued_mass=q[6:8]/rv[2]*energy*cc[2]
    rqc=q[8:10]/b[6]*energy*bc[5];rqc_mass=q[10:12]/b[6]*energy*bc[5]
    cleared=q[12:14]/b[8]*energy*bc[4]
    cy[0]=sum(rescue+recycle-load);cy[1]=sum(b[7]*rescue+recycle+rqc-join)
    cy[2:4]=load-join;cy[4:6]=stall-rescue;cy[6:8]=complete-recycle;cy[8:10]=rescue
    cy[10]=sum(rescue+recycle)*C*rv[5];cy[11]=sum(c0)*C*rv[5]
    dq[6:8]=stalled_mass-rescued_mass
    dq[8:10]=(1-b[7])*rescue-rqc;dq[10:12]=(1-b[7])*rescued_mass-rqc_mass
    dq[12:14]=b[7]*rescued_mass+rqc_mass-cleared
    dq[14:16]=incorporated;dq[16:18]=complete*length;dq[18:20]=cleared
    birth=np.log(2)/(b[1]*86400)
    tr47=birth*energy*bc[0];tr5=tr47;rp=birth*energy*bc[1]
    process=q[20]/b[2]*energy*bc[2]
    a40=q[21]/b[3]*q[24]/biobase[24]*energy*bc[2]
    a60=q[22]/b[3]*q[23]/biobase[23]*q[25]/biobase[25]*energy*bc[2]
    m40=q[26]/b[4]*energy*bc[3];m60=q[27]/b[5]*energy*bc[3]
    loss40=birth*x[0]/cybase[0]*bc[6];loss60=birth*x[1]/cybase[1]*bc[6]
    dq[20:28]=np.array([tr47-process,process-a40,process-a60,tr5-a60,rp-a40,rp-a60,a40-m40,a60-m60])
    dq[28:36]=np.array([tr47,tr5,rp,rp,m40,m60,loss40,loss60])
    cy[0]+=m40-loss40;cy[1]+=m60-loss60
    dq[36]=C*(tr47*b[11]+tr5*b[12]+rp*(b[13]+b[14])+(a40+a60)*b[15])
    dq[37]=C*(sum(cleared)*b[9]+sum(rqc)*b[10])
    dq[38]=C*birth*(sum(b[11:15])+2*b[15])
    return dq,cy,join-complete-stall,complete/c0,incorporated*C*4

@njit
def replace_targets(z,old,p,hv,nv,v,rv,b,f0,ATP0,nc,cc,bc,cybase,biobase,row):
    out,nx,_=replace_cycle_targets(z,old,p,hv,nv,v,rv,f0,ATP0,nc,cc,cybase,row)
    dq,cy,de,fac,aa_cost=bio_fluxes(z,v,rv,b,ATP0,nc,cc,bc,cybase,biobase)
    syn=p[20]*fac[0];delta=syn-out[18];out[6]+=delta;out[18]+=delta
    out[11]=np.log(2)/v[11]*(fac[1]-z[11]);nx[6:8]=de
    energy=max(z[120],0.)/(v[8]+max(z[120],0.))*(v[8]+ATP0)/ATP0
    f=min(1.,max(0.,z[320]));T=max(z[6]+z[7],1e-20)
    nx[0]=f0/(1-f0)*(v[0]+p[20])*nc[0]*energy*(1-f)-v[0]*f-f*syn/T
    nx[8]=np.log(2)/v[11]*v[9]*v[10]*4*fac[1] # replaced by nascent cost at final extension
    return out,nx,cy,dq,aa_cost

@njit
def append_biogenesis(z,old,dq,aa_cost,underlying_other,other0):
    out=np.zeros(403);out[:364]=old;out[364:]=dq
    # Replace completed-protein translation debit; never charge it twice.
    delta=sum(aa_cost)-old[254]-old[328]
    out[254]=aa_cost[0];out[328]=aa_cost[1]
    replaced=dq[38]*max(underlying_other,0.)/other0 if other0>0 else 0.
    out[402]=replaced;delta+=dq[36]+dq[37]-replaced
    out[120]-=delta;out[121]+=delta;out[125]+=delta
    return out

def checks(z,base,v,b):
    q=z[:,364:403];qb=base[364:403];x=z[:,330:342]
    m40=x[:,0]+x[:,2:8].sum(1)+z[:,326:328].sum(1)
    m60=x[:,1]+x[:,4:8].sum(1)+z[:,326:328].sum(1)+q[:,8:10].sum(1)
    b40=base[330]+base[332:338].sum()+base[326:328].sum()
    b60=base[331]+base[334:338].sum()+base[326:328].sum()+qb[8:10].sum()
    err=lambda a:float(np.max(abs(a)))
    out=dict(biogenesis_nonnegative=float(max(0.,-q.min())),
      mature40_balance=err(m40+q[:,34]-q[:,32]-b40),mature60_balance=err(m60+q[:,35]-q[:,33]-b60),
      rRNA40_lineage=err(q[:,20]+q[:,21]+q[:,26]+m40+q[:,34]-q[:,28]-sum(qb[[20,21,26]])-b40),
      rRNA60_lineage=err(q[:,20]+q[:,22]+q[:,27]+m60+q[:,35]-q[:,28]-sum(qb[[20,22,27]])-b60),
      rRNA5S_lineage=err(q[:,23]+q[:,27]+m60+q[:,35]-q[:,29]-sum(qb[[23,27]])-b60),
      RP40_lineage=err(q[:,24]+q[:,26]+m40+q[:,34]-q[:,30]-sum(qb[[24,26]])-b40),
      RP60_lineage=err(q[:,25]+q[:,27]+m60+q[:,35]-q[:,31]-sum(qb[[25,27]])-b60))
    for t,L in enumerate((414.,v[10])):
        aa=L/4*(q[:,3*t]+2*q[:,3*t+1]+3*q[:,3*t+2]);aa0=L/4*(qb[3*t]+2*qb[3*t+1]+3*qb[3*t+2])
        out['nascent_aa_lineage_'+str(t)]=err(aa+q[:,6+t]+q[:,10+t]+q[:,12+t]+q[:,16+t]+q[:,18+t]-q[:,14+t]-aa0-qb[6+t]-qb[10+t]-qb[12+t])
        out['nascent_bins_bounded_'+str(t)]=float(max(0.,(q[:,3*t:3*t+3].sum(1)-z[:,326+t]).max()))
        out['stalled_mass_bounded_'+str(t)]=float(max(0.,(q[:,6+t]-.75*L*z[:,334+t]).max()))
        out['RQC_mass_bounded_'+str(t)]=float(max(0.,(q[:,10+t]-.75*L*q[:,8+t]).max()))
        out['translation_energy_'+str(t)]=err(z[:,254 if t==0 else 328]-base[254 if t==0 else 328]-4*b[0]*(q[:,14+t]-qb[14+t]))
    return out
