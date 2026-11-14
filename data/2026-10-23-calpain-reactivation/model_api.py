"""[codex] 2026-10-23. Finite calcium gating and recoverable calpain capacity.

New opt-in API; old electrical, acute and long-memory APIs are inherited unchanged.
Rates are hypotheses, not human fits. Native/processed open enzyme is distinguished
from closed, CAST-bound and destructively inactivated enzyme. Autolysis is one-way.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util, sys, json, hashlib
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_before_reactivation20261023',_HERE.parent/'2026-10-23-long-calcium-memory/model_api.py')
_prior=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_prior;_spec.loader.exec_module(_prior)
for _name in dir(_prior):
    if not _name.startswith('_') and _name not in globals():globals()[_name]=getattr(_prior,_name)

REACTIVATION_NAMES=MEMORY_NAMES+('native_open','processed_open','closed_processed_loss_cumulative')

@dataclass(frozen=True)
class CalpainReactivationParameters:
    native_gating_tau_s: float
    processed_gating_tau_s: float
    closed_processed_loss_per_s: float
    provenance: str

    def vector(self):
        v=np.array([self.native_gating_tau_s,self.processed_gating_tau_s,self.closed_processed_loss_per_s])
        if not isinstance(self.provenance,str) or not self.provenance.strip() or not np.isfinite(v).all() or min(v[:2])<=0 or v[2]<0:
            raise ValueError('Explicit provenance, positive finite gate times and nonnegative loss required')
        return v

@njit
def _hill(c,k,n):
    if c<=0:return 0.
    z=n*(np.log(c)-np.log(k))
    if z>=0:return 1/(1+np.exp(-z))
    q=np.exp(z);return q/(1+q)

@njit
def _enzyme_rhs(e,c,ev,iv,kv):
    # Explicit non-overlapping order Nc,No,Pc,Po,NC,PC,I.
    # CAST is a maintained reservoir, no double application of inhibition.
    gn=_hill(c,ev[0],ev[2]);gp=_hill(c,ev[1],ev[2])
    nn=(gn*e[0]-(1-gn)*e[1])/kv[0]
    pp=(gp*e[2]-(1-gp)*e[3])/kv[1]
    aut=ev[3]*e[1];bn=ev[4]*e[1]-ev[5]*e[4];bp=ev[4]*e[3]-ev[5]*e[5]
    dn=iv[0]*e[1];dp=iv[1]*e[3];silent=kv[2]*e[2];d=ev[6]
    return np.array([ev[7]-nn-d*e[0],nn-aut-bn-dn-d*e[1],
        -pp-silent-d*e[2],pp+aut-bp-dp-d*e[3],bn-d*e[4],bp-d*e[5],
        dn+dp+silent-d*e[6]])

def _from_state(z):
    return np.array([z[1]-z[28],z[28],z[2]-z[29],z[29],z[3],z[4],z[5]])

def calpain_reactivation_equilibrium(c,ep,ip,kp):
    ev=ep.vector();iv=ip.vector();kv=kp.vector()
    if not np.isfinite(c) or c<0 or min(ev[6:])<=0:raise ValueError('Nonnegative calcium and positive synthesis/turnover required')
    source=_enzyme_rhs(np.zeros(7),c,ev,iv,kv)
    M=np.column_stack([_enzyme_rhs(np.eye(7)[i],c,ev,iv,kv)-source for i in range(7)])
    e=np.linalg.solve(M,-source)
    if e.min()<-1e-12 or np.max(abs(_enzyme_rhs(e,c,ev,iv,kv)))>1e-10:raise ArithmeticError('Invalid basal equilibrium')
    return e

@njit
def _reactivation_rhs(y,j,p,kv):
    c=p[0]+y[0];ev=p[2:10];iv=p[10:12]
    e=np.array([y[1]-y[28],y[28],y[2]-y[29],y[29],y[3],y[4],y[5]])
    de=_enzyme_rhs(e,c,ev,iv,kv)
    act=(y[28]+y[29])*(1-p[30])
    qU=p[12]*act*y[6];qP=p[12]*act*p[19]*y[7];qN=p[13]*act*y[8]
    du=p[14]*y[6];dp=p[14]*y[7];dn=p[15]*y[8];ds=p[16]*y[9]
    ph=p[17]*y[6]-p[18]*y[7]
    n=max(0.,y[6]+y[7]);nh=n**p[22];kh=p[21]**p[22];target=nh*(1+kh)/(nh+kh)
    ax=max(0.,y[12]);ah=ax**p[29];ak=p[28]**p[29];rg=ah*(1+ak)/(ah+ak)
    out=np.zeros(31);out[0]=j-y[0]/p[1]
    out[1]=de[0]+de[1];out[2]=de[2]+de[3];out[3]=de[4];out[4]=de[5];out[5]=de[6]
    out[28]=de[1];out[29]=de[3];out[30]=kv[2]*e[2]
    out[6]=p[20]-du-qU-ph;out[7]=ph-dp-qP;out[8]=qU+qP-qN-dn;out[9]=qN-ds
    out[10]=(target-y[10])/p[23];out[11]=np.log(2)*(y[10]-y[11])/p[24]
    out[12]=(p[27]*y[10]-y[12])/p[25];out[13]=(rg-y[13])/p[26]
    out[14]=p[9];out[15]=p[8]*np.sum(y[1:6]);out[16]=p[5]*y[28]
    out[17]=p[10]*y[28]+p[11]*y[29]+out[30]
    out[18]=p[20];out[19]=du+dp+dn+ds;out[20]=qU+qP;out[21]=qN
    out[22]=j;out[23]=y[0]/p[1];out[24]=y[0];out[25]=act;out[26]=qU;out[27]=qP
    return out

def calcium_reactivation_chain(segments,*,basal_ca_uM,calcium_tau_s,enzyme_parameters,tdp_parameters,
        localization_parameters,stmn2_parameters,inactivation_parameters,phosphorylation_parameters,
        reactivation_parameters,provenance,checkpoint=None,output_step_s=30.,rtol=1e-8,atol=1e-10,
        max_step_s=600.,calpain_block_fraction=0.):
    """Contiguous calcium flux segments with all state preserved through pauses.

    Closed <-> open transition rates are g(C)/tau and (1-g(C))/tau. They have
    the former Hill equilibrium ONLY in the isolated gating limit. Synthesis
    creates closed native enzyme; autolysis carries open native into open
    processed enzyme; CAST release produces open enzyme which can close again.
    Optional loss of closed processed enzyme is a separate sensitivity branch,
    inspired by purified-protein stability experiments, NOT a neuronal rate.
    Constant native/processed gate taus are phenomenological, not Ca binding rates.
    Legacy TDP-parameter Ca/activation/deactivation/CAST fields are not used:
    TDP cleavage receives the explicit open-enzyme activity without double gating.
    """
    sg=np.asarray(segments,float)
    if sg.ndim!=2 or sg.shape[1]!=3 or len(sg)==0 or not np.isfinite(sg).all() or np.any(sg[:,0]<0) or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[:,2]<0) or np.any(sg[1:,0]!=sg[:-1,1]):raise ValueError('Contiguous nonnegative flux segments required')
    scales=[basal_ca_uM,calcium_tau_s,output_step_s,rtol,atol,max_step_s]
    if not np.isfinite(scales).all() or min(scales)<=0 or not np.isfinite(calpain_block_fraction) or not 0<=calpain_block_fraction<=1:raise ValueError('Positive scales and fractional block required')
    if not isinstance(provenance,str) or not provenance.strip():raise ValueError('Input provenance required')
    ep=enzyme_parameters;tp=tdp_parameters;sp=stmn2_parameters;lp=localization_parameters;kp=reactivation_parameters
    ev=ep.vector();iv=inactivation_parameters.vector();tv=tp.vector();pv=phosphorylation_parameters.vector();kv=kp.vector();lp.validate()
    e=calpain_reactivation_equilibrium(basal_ca_uM,ep,inactivation_parameters,kp)
    # Initialization is the untreated equilibrium even in a catalytic-block arm.
    activity0=float(e[1]+e[3]);pi=phospho_tdp_basal_state(activity0,tp,phosphorylation_parameters)
    ini=healthy_chain_state(basal_ca_uM=basal_ca_uM,enzyme_parameters=ep,tdp_parameters=tp,
        localization_parameters=lp,stmn2_parameters=sp,inactivation_parameters=inactivation_parameters)
    ybase=np.r_[0.,e[0]+e[1],e[2]+e[3],e[4:7],pi['pools'],ini['STMN2'],np.zeros(14),e[1],e[3],0.]
    config=dict(topology='reactivation_v1_31states',basal_ca_uM=basal_ca_uM,calcium_tau_s=calcium_tau_s,
        enzyme=asdict(ep),TDP=asdict(tp),STMN2=asdict(sp),localization=asdict(lp),
        inactivation=asdict(inactivation_parameters),phosphorylation=asdict(phosphorylation_parameters),
        reactivation=asdict(kp),calpain_block_fraction=calpain_block_fraction)
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();syn=pi['synthesis_per_s']
    if checkpoint is None:y=ybase.copy();origin=float(sg[0,0])
    else:
        if checkpoint['fingerprint']!=fingerprint or checkpoint['time_s']!=sg[0,0]:raise ValueError('Exact time/parameter continuation required')
        y=np.asarray(checkpoint['state'],float).copy();origin=checkpoint['origin_time_s']
        if y.shape!=(31,) or not np.isfinite(y).all() or y.min()<-1e-7 or _from_state(y).min()<-1e-7:raise ValueError('Invalid checkpoint')
        if checkpoint['synthesis_per_s']!=syn or not np.array_equal(checkpoint['baseline_state'],ybase):raise ValueError('Changed baseline')
    st=np.array([sp.tdp_half_relative,sp.tdp_hill,sp.mrna_tau_s,sp.somatodendritic_protein_half_life_s,
        sp.axonal_availability_tau_s,sp.regrowth_response_tau_s,sp.axonal_gain,sp.regrowth_half_relative,sp.regrowth_hill])
    p=np.r_[basal_ca_uM,calcium_tau_s,ev,iv,tv[5:10],pv,syn,st,calpain_block_fraction]
    times=[np.array([sg[0,0]])];states=[y[None,:]];nfev=0
    for start,end,j in sg:
        near=start+np.geomspace(1e-4,min(end-start,3600.),100)
        grid=np.unique(np.r_[np.arange(start+output_step_s,end,output_step_s),near[near<end],end])
        sol=solve_ivp(lambda t,z:_reactivation_rhs(z,j,p,kv),(start,end),y,t_eval=grid,method='LSODA',rtol=rtol,atol=atol,max_step=max_step_s)
        if not sol.success:raise RuntimeError(sol.message)
        y=sol.y[:,-1].copy();times.append(sol.t);states.append(sol.y.T);nfev+=sol.nfev
    t=np.concatenate(times);z=np.concatenate(states);c=basal_ca_uM+z[:,0]
    checks=dict(enzyme_balance=float(abs(z[:,1:6].sum(1)+z[:,15]-z[:,14]-ybase[1:6].sum()).max()),
        TDP_lineage_balance=float(abs(z[:,6:10].sum(1)+z[:,19]-z[:,18]-ybase[6:10].sum()).max()),
        calcium_balance_uM=float(abs(z[:,0]+z[:,23]-z[:,22]).max()),
        calcium_AUC_identity=float(abs(z[:,24]-calcium_tau_s*z[:,23]).max()),
        minimum_state=float(min(z.min(),(z[:,1]-z[:,28]).min(),(z[:,2]-z[:,29]).min())))
    if not np.isfinite(z).all() or checks['minimum_state']<-1e-7 or max(v for k,v in checks.items() if k!='minimum_state')>1e-5:raise ArithmeticError(str(checks))
    cp=dict(time_s=float(t[-1]),state=y.tolist(),baseline_state=ybase.tolist(),origin_time_s=origin,synthesis_per_s=syn,fingerprint=fingerprint)
    return dict(time_s=t,state=z,columns=REACTIVATION_NAMES,calcium_uM=c,activity=(z[:,28]+z[:,29])*(1-calpain_block_fraction),
        baseline_state=ybase,baseline_activity=activity0,checkpoint=cp,config=config,checks=checks,nfev=nfev,segments=sg,
        intact_TDP=z[:,6]+z[:,7],nuclear_TDP_relative=z[:,6]+z[:,7],nuclear_TDP=lp.nuclear_fraction*(z[:,6]+z[:,7]),
        free_recoverable_pool=z[:,1]+z[:,2],total_nondestroyed_pool=z[:,1:5].sum(1),provenance=provenance,
        human_kinetic_fit=False,electrical_feedback=False,
        limitations=['Finite gating and closed-processed loss rates are not measured human rates',
            'Free and total nondestroyed enzyme are pool sizes, not measured saturating-Ca assay activity',
            'Maintained CAST reservoir, one effective calpain population, bulk/local Ca not resolved',
            'Conditional equal-partition TDP and STMN2 proxies; no NMJ or disease prediction',
            'Old acute/electrical API retained separately; this opt-in module receives hypothetical flux'])

def calpain_rechallenge_probe(state,*,basal_ca_uM,calcium_tau_s,entry_uM_per_s,duration_s,
        enzyme_parameters,inactivation_parameters,reactivation_parameters,rtol=1e-9,atol=1e-11):
    """Non-mutating virtual Ca challenge of a saved enzyme state.

    Identical Ca initial condition/entry permits history vs fresh-state comparison.
    Includes finite gating, CAST and damage, but no TDP substrate limitation. This
    is a simulated finite challenge, NOT a saturating-Ca experimental enzyme assay.
    """
    y=np.asarray(state,float)
    if y.shape!=(31,) or not np.isfinite(y).all() or y.min()<-1e-7 or _from_state(y).min()<-1e-7:raise ValueError('31-state vector required')
    if not np.isfinite([basal_ca_uM,calcium_tau_s,entry_uM_per_s,duration_s]).all() or min(basal_ca_uM,calcium_tau_s,duration_s)<=0 or entry_uM_per_s<0:raise ValueError('Invalid challenge scales')
    ev=enzyme_parameters.vector();iv=inactivation_parameters.vector();kv=reactivation_parameters.vector()
    def rhs(t,z):return np.r_[entry_uM_per_s-z[0]/calcium_tau_s,_enzyme_rhs(z[1:8],basal_ca_uM+z[0],ev,iv,kv),z[2]+z[4]]
    sol=solve_ivp(rhs,(0.,duration_s),np.r_[y[0],_from_state(y),0.],method='LSODA',rtol=rtol,atol=atol,max_step=10.)
    if not sol.success or sol.y.min()<-1e-7:raise ArithmeticError('Probe failed')
    return dict(activity_AUC_s=float(sol.y[-1,-1]),final_activity=float(sol.y[2,-1]+sol.y[4,-1]),
        duration_s=duration_s,mutates_input=False,human_assay_fit=False)

def calpain_reactivation_evidence():
    return json.loads((_HERE.parent/'2026-10-23-calpain-off-kinetics/evidence.json').read_text(encoding='utf-8'))

def source_training_data():
    r=_prior.source_training_data();r['calpain_off_and_reactivation_context']=calpain_reactivation_evidence();return r
