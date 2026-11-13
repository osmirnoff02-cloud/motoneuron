"""[codex] 2026-10-23. Calpain structural stability sensitivity extension.

Crawford 1987 motivates a fragile but still associated active intermediate.
Li 2004 motivates an optional Ca-independent stability loss of processed enzyme.
Neither pathway has measured rates in human motor neurons. Zero added rates
recover the previous model. Prior electrical APIs and their parameters are frozen.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util,sys,json,hashlib
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_before_stability20261023',_HERE.parent/'2026-10-23-calpain-reactivation/model_api_codex.py')
_prior=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_prior;_spec.loader.exec_module(_prior)
for _name in dir(_prior):
    if not _name.startswith('_') and _name not in globals():globals()[_name]=getattr(_prior,_name)
STABILITY_NAMES=REACTIVATION_NAMES+('fragile_free','fragile_open','fragile_CAST',
    'fragile_formation_cumulative','fragile_inactivation_cumulative','processed_background_loss_cumulative')

@dataclass(frozen=True)
class CalpainStabilityParameters:
    further_cleavage_per_s: float
    fragile_closed_loss_per_s: float
    processed_background_loss_per_s: float
    provenance: str
    def vector(self):
        v=np.array([self.further_cleavage_per_s,self.fragile_closed_loss_per_s,self.processed_background_loss_per_s])
        if not isinstance(self.provenance,str) or not self.provenance.strip() or not np.isfinite(v).all() or v.min()<0:
            raise ValueError('Nonnegative finite rates and explicit provenance required')
        return v

_old_enzyme_rhs=_prior._enzyme_rhs
_old_rhs=_prior._reactivation_rhs
_hill=_prior._hill

@njit
def _stability_enzyme_rhs(e,c,ev,iv,kv,sv):
    # Nc,No,Pc,Po,NC,PC,I,Xc,Xo,XC: all disjoint amounts of enzyme equivalents.
    out=np.zeros(10);out[:7]=_old_enzyme_rhs(e[:7],c,ev,iv,kv)
    fragment=sv[0]*e[3]
    background_c=sv[2]*e[2];background_o=sv[2]*e[3]
    g=_hill(c,ev[1],ev[2]);gate=(g*e[7]-(1-g)*e[8])/kv[1]
    binding=ev[4]*e[8]-ev[5]*e[9]
    loss_c=sv[1]*e[7];loss_o=iv[1]*e[8];d=ev[6]
    out[2]-=background_c;out[3]-=fragment+background_o
    out[5]+=0. # CAST-bound P is protected in this explicit assumption.
    out[6]+=background_c+background_o+loss_c+loss_o
    out[7]=-gate-loss_c-d*e[7]
    out[8]=gate+fragment-binding-loss_o-d*e[8]
    out[9]=binding-d*e[9]
    return out

def _from_stability_state(z):
    return np.r_[_prior._from_state(z),z[31]-z[32],z[32],z[33]]

def calpain_stability_equilibrium(c,ep,ip,kp,sp):
    ev=ep.vector();iv=ip.vector();kv=kp.vector();sv=sp.vector()
    if not np.isfinite(c) or c<0 or min(ev[6:])<=0:raise ValueError('Positive synthesis and turnover required')
    src=_stability_enzyme_rhs(np.zeros(10),c,ev,iv,kv,sv)
    mat=np.column_stack([_stability_enzyme_rhs(np.eye(10)[i],c,ev,iv,kv,sv)-src for i in range(10)])
    e=np.linalg.solve(mat,-src)
    if e.min()<-1e-12 or np.max(abs(_stability_enzyme_rhs(e,c,ev,iv,kv,sv)))>1e-10:raise ArithmeticError('Invalid equilibrium')
    return e

@njit
def _stability_rhs(y,j,p,kv,sv):
    out=np.zeros(37);out[:31]=_old_rhs(y[:31],j,p,kv)
    e=np.array([y[1]-y[28],y[28],y[2]-y[29],y[29],y[3],y[4],y[5],y[31]-y[32],y[32],y[33]])
    de=_stability_enzyme_rhs(e,p[0]+y[0],p[2:10],p[10:12],kv,sv)
    out[1]=de[0]+de[1];out[2]=de[2]+de[3];out[3]=de[4];out[4]=de[5];out[5]=de[6]
    out[28]=de[1];out[29]=de[3];out[31]=de[7]+de[8];out[32]=de[8];out[33]=de[9]
    out[34]=sv[0]*y[29];out[35]=sv[1]*e[7]+p[11]*e[8];out[36]=sv[2]*y[2]
    out[15]+=p[8]*(y[31]+y[33]);out[17]+=out[35]+out[36]
    # Assumption: associated X has P's activity and substrate specificity.
    act=y[32]*(1-p[30]);u=p[12]*act*y[6];v=p[12]*act*p[19]*y[7];n=p[13]*act*y[8]
    out[6]-=u;out[7]-=v;out[8]+=u+v-n;out[9]+=n
    out[20]+=u+v;out[21]+=n;out[25]+=act;out[26]+=u;out[27]+=v
    return out

def calcium_stability_chain(segments,*,basal_ca_uM,calcium_tau_s,enzyme_parameters,tdp_parameters,
        localization_parameters,stmn2_parameters,inactivation_parameters,phosphorylation_parameters,
        reactivation_parameters,stability_parameters,provenance,checkpoint=None,output_step_s=30.,rtol=1e-8,atol=1e-10,
        max_step_s=600.,calpain_block_fraction=0.):
    """Continuous Ca -> calpain -> TDP -> STMN2 with optional stability routes.

    Open processed P can undergo further cleavage to associated fragile X.
    X stays Ca-dependent and catalytically competent until closed-state loss.
    A separate optional stability loss affects free P at any calcium level.
    This is an uncertain topology comparison, not a human kinetic fit.
    Legacy calcium_reactivation_chain remains unchanged and separately callable.
    """
    sg=np.asarray(segments,float)
    if sg.ndim!=2 or sg.shape[1]!=3 or len(sg)==0 or not np.isfinite(sg).all() or np.any(sg[:,0]<0) or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[:,2]<0) or np.any(sg[1:,0]!=sg[:-1,1]):raise ValueError('Contiguous nonnegative flux segments required')
    scales=[basal_ca_uM,calcium_tau_s,output_step_s,rtol,atol,max_step_s]
    if not np.isfinite(scales).all() or min(scales)<=0 or not np.isfinite(calpain_block_fraction) or not 0<=calpain_block_fraction<=1:raise ValueError('Positive scales and fractional block required')
    if not isinstance(provenance,str) or not provenance.strip():raise ValueError('Input provenance required')
    ep=enzyme_parameters;tp=tdp_parameters;sp=stmn2_parameters;lp=localization_parameters;kp=reactivation_parameters
    ev=ep.vector();iv=inactivation_parameters.vector();tv=tp.vector();pv=phosphorylation_parameters.vector();kv=kp.vector();lp.validate()
    sv=stability_parameters.vector()
    e=calpain_stability_equilibrium(basal_ca_uM,ep,inactivation_parameters,kp,stability_parameters)
    # Initialization is the untreated equilibrium even in a catalytic-block arm.
    activity0=float(e[1]+e[3]+e[8]);pi=phospho_tdp_basal_state(activity0,tp,phosphorylation_parameters)
    ini=healthy_chain_state(basal_ca_uM=basal_ca_uM,enzyme_parameters=ep,tdp_parameters=tp,
        localization_parameters=lp,stmn2_parameters=sp,inactivation_parameters=inactivation_parameters)
    ybase=np.r_[0.,e[0]+e[1],e[2]+e[3],e[4:7],pi['pools'],ini['STMN2'],np.zeros(14),e[1],e[3],0.,e[7]+e[8],e[8],e[9],0.,0.,0.]
    config=dict(topology='stability_v1_37states',stability=asdict(stability_parameters),basal_ca_uM=basal_ca_uM,calcium_tau_s=calcium_tau_s,
        enzyme=asdict(ep),TDP=asdict(tp),STMN2=asdict(sp),localization=asdict(lp),
        inactivation=asdict(inactivation_parameters),phosphorylation=asdict(phosphorylation_parameters),
        reactivation=asdict(kp),calpain_block_fraction=calpain_block_fraction)
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();syn=pi['synthesis_per_s']
    if checkpoint is None:y=ybase.copy();origin=float(sg[0,0])
    else:
        if checkpoint['fingerprint']!=fingerprint or checkpoint['time_s']!=sg[0,0]:raise ValueError('Exact time/parameter continuation required')
        y=np.asarray(checkpoint['state'],float).copy();origin=checkpoint['origin_time_s']
        if y.shape!=(37,) or not np.isfinite(y).all() or y.min()<-1e-7 or _from_stability_state(y).min()<-1e-7:raise ValueError('Invalid checkpoint')
        if checkpoint['synthesis_per_s']!=syn or not np.array_equal(checkpoint['baseline_state'],ybase):raise ValueError('Changed baseline')
    st=np.array([sp.tdp_half_relative,sp.tdp_hill,sp.mrna_tau_s,sp.somatodendritic_protein_half_life_s,
        sp.axonal_availability_tau_s,sp.regrowth_response_tau_s,sp.axonal_gain,sp.regrowth_half_relative,sp.regrowth_hill])
    p=np.r_[basal_ca_uM,calcium_tau_s,ev,iv,tv[5:10],pv,syn,st,calpain_block_fraction]
    times=[np.array([sg[0,0]])];states=[y[None,:]];nfev=0
    for start,end,j in sg:
        near=start+np.geomspace(1e-4,min(end-start,3600.),100)
        grid=np.unique(np.r_[np.arange(start+output_step_s,end,output_step_s),near[near<end],end])
        sol=solve_ivp(lambda t,z:_stability_rhs(z,j,p,kv,sv),(start,end),y,t_eval=grid,method='LSODA',rtol=rtol,atol=atol,max_step=max_step_s)
        if not sol.success:raise RuntimeError(sol.message)
        y=sol.y[:,-1].copy();times.append(sol.t);states.append(sol.y.T);nfev+=sol.nfev
    t=np.concatenate(times);z=np.concatenate(states);c=basal_ca_uM+z[:,0]
    checks=dict(enzyme_balance=float(abs(z[:,1:6].sum(1)+z[:,31]+z[:,33]+z[:,15]-z[:,14]-ybase[1:6].sum()-ybase[31]-ybase[33]).max()),
        TDP_lineage_balance=float(abs(z[:,6:10].sum(1)+z[:,19]-z[:,18]-ybase[6:10].sum()).max()),
        calcium_balance_uM=float(abs(z[:,0]+z[:,23]-z[:,22]).max()),
        calcium_AUC_identity=float(abs(z[:,24]-calcium_tau_s*z[:,23]).max()),
        minimum_state=float(min(z.min(),(z[:,1]-z[:,28]).min(),(z[:,2]-z[:,29]).min(),(z[:,31]-z[:,32]).min())))
    if not np.isfinite(z).all() or checks['minimum_state']<-1e-7 or max(v for k,v in checks.items() if k!='minimum_state')>1e-5:raise ArithmeticError(str(checks))
    cp=dict(time_s=float(t[-1]),state=y.tolist(),baseline_state=ybase.tolist(),origin_time_s=origin,synthesis_per_s=syn,fingerprint=fingerprint)
    return dict(time_s=t,state=z,columns=STABILITY_NAMES,calcium_uM=c,activity=(z[:,28]+z[:,29]+z[:,32])*(1-calpain_block_fraction),
        baseline_state=ybase,baseline_activity=activity0,checkpoint=cp,config=config,checks=checks,nfev=nfev,segments=sg,
        intact_TDP=z[:,6]+z[:,7],nuclear_TDP_relative=z[:,6]+z[:,7],nuclear_TDP=lp.nuclear_fraction*(z[:,6]+z[:,7]),
        free_recoverable_pool=z[:,1]+z[:,2]+z[:,31],total_nondestroyed_pool=z[:,1:5].sum(1)+z[:,31]+z[:,33],provenance=provenance,
        human_kinetic_fit=False,electrical_feedback=False,
        limitations=['Fragile complex has processed-enzyme gating, CAST kinetics and activity by assumption; CAPN1/2 not distinguished',
            'Extra rates are stress-test assumptions, not measured neuronal or physiological salt effects',
            'No membrane/ATP/necrosis/NCX feedback model; DOrsi rapid excitotoxic collapse is outside scope',
            'Finite gating and closed-processed loss rates are not measured human rates',
            'Free and total nondestroyed enzyme are pool sizes, not measured saturating-Ca assay activity',
            'Maintained CAST reservoir, one effective calpain population, bulk/local Ca not resolved',
            'Conditional equal-partition TDP and STMN2 proxies; no NMJ or disease prediction',
            'Old acute/electrical API retained separately; this opt-in module receives hypothetical flux'])

def calpain_stability_evidence():
    return json.loads((_HERE/'evidence_codex.json').read_text(encoding='utf-8'))

def source_training_data():
    result=_prior.source_training_data()
    result['calpain_off_and_reactivation_context']=calpain_reactivation_evidence()
    result['calpain_stability_full_text_audit']=calpain_stability_evidence()
    return result

def calpain_reactivation_evidence():
    rows=_prior.calpain_reactivation_evidence()
    for row in rows:
        if row['source']=='Murphy2006':
            row.pop('activity_ceases_near_Ca_uM',None)
            row['no_significant_excess_force_decline_at_Ca_uM']=.05
            row['exact_zero_activity_established']=False
            row['depth']='Full local PDF and Fig9C visually checked 2026-10-23'
            row['measure']='Passive-force decline proxy above control, not direct enzyme concentration'
        if row['source'] in ['Dransfield1998','DOrsi2012']:
            row['depth']='Full local PDF, relevant figures and References checked 2026-10-23'
    return rows

def calpain_source_assay_reference(source):
    """Published assay observables, never passed to human ODE rates."""
    rows=calpain_stability_evidence()['sources']
    row=next((x for x in rows if x['id']==source),None)
    if row is None:raise KeyError(source)
    if source=='Dransfield1998':
        time=np.array([0.,2.,5.,10.,12.,15.,20.,30.])
        rates=np.array(row['published_exponential_rates_per_min'])
        return dict(preactivation_time_min=time,recoverable_activity_from_published_fits=np.exp(-rates[:,None]*time),
            raw_points=False,off_rate=False,human_fit=False,source=row)
    if source=='Li2004':
        v=row['Table1_rows']
        return dict(table=v,columns=row['Table1_columns'],
            post_dialysis_salt_over_matched_no_salt={k:vals[4]/vals[2] for k,vals in v.items()},
            raw_time_series=False,off_rate=False,human_fit=False,source=row)
    return dict(source=row,human_fit=False)
