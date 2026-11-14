"""[codex] 2026-10-23. Conditional aggregation coupled to continuous Ca/calpain/TDP.

Amounts are N-lineage equivalents relative to basal soluble intact TDP, NOT
aggregate counts, protein mass, insoluble band intensity or human concentrations.
Old APIs/electrical models remain available unchanged. No human kinetic fit.
"""
from pathlib import Path
from dataclasses import dataclass,asdict
import importlib.util,sys,json,hashlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from numba import njit
_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_before_aggregation20261023',_HERE.parent/'2026-10-23-calpain-seven-papers/model_api.py')
_prior=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_prior;_spec.loader.exec_module(_prior)
for _name in dir(_prior):
    if not _name.startswith('_') and _name not in globals():globals()[_name]=getattr(_prior,_name)
AGGREGATION_NAMES=STABILITY_NAMES+tuple('aggregate_fast_'+x for x in ('U','P','large_N','small_N'))+tuple('aggregate_slow_'+x for x in ('U','P','large_N','small_N'))+(
    'aggregation_cumulative','aggregate_clearance_cumulative','disaggregation_cumulative','aggregate_AUC','aggregate_proteolytic_release_cumulative')

@dataclass(frozen=True)
class TDPAggregationParameters:
    formation_per_s: tuple
    fast_clearance_per_s: tuple
    slow_clearance_per_s: tuple
    slow_formation_fraction: float
    disaggregation_per_s: float
    formation_order: float
    aggregate_calpain_accessibility: float
    provenance: str
    def vector(self):
        arrays=[np.asarray(x,float) for x in (self.formation_per_s,self.fast_clearance_per_s,self.slow_clearance_per_s)]
        if any(x.shape!=(4,) for x in arrays):raise ValueError('Four U/P/large-N/small-N rates required for each field')
        v=np.r_[*arrays,self.slow_formation_fraction,self.disaggregation_per_s,self.formation_order,self.aggregate_calpain_accessibility]
        if not np.isfinite(v).all() or min(v)<0 or min(v[4:12])<=0 or not 0<=v[12]<=1 or v[14] not in (1.,2.) or not 0<=v[15]<=1:
            raise ValueError('Nonnegative rates, positive clearance, fractions in [0,1] and order 1 or 2 required')
        if not isinstance(self.provenance,str) or not self.provenance.strip():raise ValueError('Explicit source/assumption required')
        return v

_old_rhs=_prior._stability_rhs

@njit
def _aggregate_fluxes(s,fast,slow,act,p,av):
    formation=av[:4]*np.maximum(s,0.)**av[14]
    back=av[13]*(fast+slow)
    df=(1-av[12])*formation-(av[4:8]+av[13])*fast
    ds=av[12]*formation-(av[8:12]+av[13])*slow
    soluble=-formation+back
    clear=np.sum(av[4:8]*fast)+np.sum(av[8:12]*slow)
    # Optional unmeasured accessibility. Cleavage of aggregate large N releases
    # soluble small N. Source Fig5 motivates sequential substrates, NOT this rate.
    qU=0.;qP=0.;qN=0.
    for j in range(2):
        a=fast if j==0 else slow
        da=df if j==0 else ds
        u=av[15]*p[12]*act*a[0];v=av[15]*p[12]*p[19]*act*a[1];n=av[15]*p[13]*act*a[2]
        da[0]-=u;da[1]-=v;da[2]+=u+v-n
        soluble[3]+=n;qU+=u;qP+=v;qN+=n
    return soluble,df,ds,formation.sum(),back.sum(),clear,qU,qP,qN

@njit
def _aggregation_rhs(y,j,p,kv,sv,av,block):
    # Instantaneous catalytic suppression, no pharmacokinetics/binding prediction.
    # Suppress substrate cleavage, autolysis and active self-damage consistently.
    pp=p.copy();ss=sv.copy();remaining=1-block
    pp[5]*=remaining;pp[10]*=remaining;pp[11]*=remaining;pp[30]=block;ss[0]*=remaining
    out=np.zeros(50);out[:37]=_old_rhs(y[:37],j,pp,kv,ss)
    act=(y[28]+y[29]+y[32])*remaining
    s,df,ds,formed,back,clear,u,v,n=_aggregate_fluxes(y[6:10],y[37:41],y[41:45],act,p,av)
    out[6:10]+=s;out[37:41]=df;out[41:45]=ds
    out[19]+=clear;out[20]+=u+v;out[21]+=n;out[26]+=u;out[27]+=v
    out[45]=formed;out[46]=clear;out[47]=back;out[48]=np.sum(y[37:45]);out[49]=n
    return out

def _baseline(kw,av):
    ep=kw['enzyme_parameters'];ip=kw['inactivation_parameters'];kp=kw['reactivation_parameters'];sp=kw['stability_parameters']
    e=_prior.calpain_stability_equilibrium(kw['basal_ca_uM'],ep,ip,kp,sp)
    ev=ep.vector();iv=ip.vector();tv=kw['tdp_parameters'].vector();pv=kw['phosphorylation_parameters'].vector();st=kw['stmn2_parameters'];lp=kw['localization_parameters'];lp.validate()
    activity=e[1]+e[3]+e[8]
    sv=sp.vector();kv=kp.vector()
    stv=np.array([st.tdp_half_relative,st.tdp_hill,st.mrna_tau_s,st.somatodendritic_protein_half_life_s,st.axonal_availability_tau_s,st.regrowth_response_tau_s,st.axonal_gain,st.regrowth_half_relative,st.regrowth_hill])
    p=np.r_[kw['basal_ca_uM'],kw['calcium_tau_s'],ev,iv,tv[5:10],pv,0.,stv,0.]
    # For a given soluble pool the aggregate equations are linear; solve their
    # stationary amounts, including optional aggregate cleavage, then solve the
    # soluble steady state with U+P=1 and fixed untreated synthesis.
    def agg(s):
        a0=np.zeros(4);f0=np.zeros(4)
        _,df,ds,*_= _aggregate_fluxes(s,a0,f0,activity,p,av)
        src=np.r_[df,ds]
        mat=np.empty((8,8))
        for k in range(8):
            a=np.eye(8)[k]
            _,af,ass,*_= _aggregate_fluxes(s,a[:4],a[4:],activity,p,av)
            mat[:,k]=np.r_[af,ass]-src
        return np.linalg.solve(mat,-src)
    # The stable triangular biochemical network permits scalar root solves.
    def flux(s):
        a=agg(s);d,*_=_aggregate_fluxes(s,a[:4],a[4:],activity,p,av);return d,a
    def p_res(P):
        U=1-P;d,_=flux(np.array([U,P,0.,0.]));return pv[0]*U-pv[1]*P-p[14]*P-p[12]*activity*p[19]*P+d[1]
    P=brentq(p_res,0.,1.,xtol=1e-14);U=1-P
    def n_res(N):
        s=np.array([U,P,N,0.]);d,_=flux(s)
        return p[12]*activity*(U+p[19]*P)-(p[13]*activity+p[15])*N+d[2]
    bound=1.
    while n_res(bound)>0:bound*=2
    N=0. if n_res(0.)==0 else brentq(n_res,0.,bound,xtol=1e-14)
    def s_res(S):
        s=np.array([U,P,N,S]);d,_=flux(s);return p[13]*activity*N-p[16]*S+d[3]
    bound=1.
    while s_res(bound)>0:bound*=2
    S=0. if s_res(0.)==0 else brentq(s_res,0.,bound,xtol=1e-14)
    soluble=np.array([U,P,N,S]);d,a=flux(soluble)
    syn=(p[14]+p[12]*activity)*U+pv[0]*U-pv[1]*P-d[0];p[20]=syn
    ini=healthy_chain_state(basal_ca_uM=kw['basal_ca_uM'],enzyme_parameters=ep,tdp_parameters=kw['tdp_parameters'],localization_parameters=lp,stmn2_parameters=st,inactivation_parameters=ip)
    base=np.r_[0.,e[0]+e[1],e[2]+e[3],e[4:7],soluble,ini['STMN2'],np.zeros(14),e[1],e[3],0.,e[7]+e[8],e[8],e[9],0.,0.,0.,a,np.zeros(5)]
    physical=list(range(14))+[28,29,31,32,33]+list(range(37,45))
    err=float(abs(_aggregation_rhs(base,0.,p,kv,sv,av,0.)[physical]).max())
    if syn<0 or base.min()<-1e-10 or err>1e-9:raise ArithmeticError(f'Invalid untreated equilibrium {err}')
    return base,p,kv,sv,activity

def calcium_aggregation_chain(segments,*,aggregation_parameters,provenance,checkpoint=None,
        output_step_s=1800.,rtol=1e-8,atol=1e-10,max_step_s=3600.,**kw):
    """Rows: start_s,end_s,Ca_flux_uM_per_s,calpain_block_fraction.

    All arms start from the SAME untreated equilibrium for a given parameter
    configuration. Flux and suppression may change only at explicit boundaries.
    Checkpoints preserve all states and untreated synthesis; changed inputs are
    allowed, changed kinetic parameters/baseline are rejected.
    """
    required={'basal_ca_uM','calcium_tau_s','enzyme_parameters','tdp_parameters','localization_parameters','stmn2_parameters','inactivation_parameters','phosphorylation_parameters','reactivation_parameters','stability_parameters'}
    if set(kw)!=required:raise ValueError('Explicit biochemical parameters required; unexpected/missing fields: '+str(set(kw)^required))
    sg=np.asarray(segments,float)
    if sg.ndim!=2 or sg.shape[1]!=4 or not len(sg) or not np.isfinite(sg).all() or np.any(sg[:,0]<0) or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[:,2:]<0) or np.any(sg[:,3]>1) or np.any(sg[1:,0]!=sg[:-1,1]):raise ValueError('Contiguous valid flux/suppression segments required')
    scales=[kw['basal_ca_uM'],kw['calcium_tau_s'],output_step_s,rtol,atol,max_step_s]
    if not np.isfinite(scales).all() or min(scales)<=0 or not isinstance(provenance,str) or not provenance.strip():raise ValueError('Positive scales and provenance required')
    av=aggregation_parameters.vector();base,p,kv,sv,a0=_baseline(kw,av)
    config={k:asdict(v) if hasattr(v,'__dataclass_fields__') else v for k,v in kw.items()}
    config.update(topology='aggregation_50states_v1',aggregation=asdict(aggregation_parameters),block_mode='instantaneous catalytic, including autolysis/self-processing; not a drug concentration')
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if checkpoint is None:y=base.copy();origin=sg[0,0]
    else:
        if checkpoint['fingerprint']!=fingerprint or checkpoint['time_s']!=sg[0,0] or checkpoint['synthesis_per_s']!=p[20] or not np.array_equal(checkpoint['baseline_state'],base):raise ValueError('Changed time, parameters or untreated baseline')
        y=np.asarray(checkpoint['state'],float).copy();origin=checkpoint['origin_time_s']
        if y.shape!=(50,) or not np.isfinite(y).all() or y.min()<-1e-7 or _prior._from_stability_state(y).min()<-1e-7:raise ValueError('Invalid checkpoint')
    times=[np.array([sg[0,0]])];states=[y[None,:]];nfev=0
    for start,end,j,block in sg:
        near=start+np.geomspace(1e-4,min(end-start,3600.),70)
        grid=np.unique(np.r_[np.arange(start+output_step_s,end,output_step_s),near[near<end],end])
        sol=solve_ivp(lambda t,z:_aggregation_rhs(z,j,p,kv,sv,av,block),(start,end),y,t_eval=grid,method='LSODA',rtol=rtol,atol=atol,max_step=max_step_s)
        if not sol.success:raise RuntimeError(sol.message)
        y=sol.y[:,-1].copy();times.append(sol.t);states.append(sol.y.T);nfev+=sol.nfev
    t=np.concatenate(times);z=np.concatenate(states);ag=z[:,37:45].sum(1)
    indexes=np.clip(np.searchsorted(sg[:,0],t,side='right')-1,0,len(sg)-1);b=sg[indexes,3]
    enzyme=z[:,1:6].sum(1)+z[:,31]+z[:,33];enzyme0=base[1:6].sum()+base[31]+base[33]
    td=z[:,6:10].sum(1)+ag;td0=base[6:10].sum()+base[37:45].sum()
    checks=dict(enzyme_balance=float(abs(enzyme+z[:,15]-z[:,14]-enzyme0).max()),
        TDP_lineage_balance=float(abs(td+z[:,19]-z[:,18]-td0).max()),
        aggregate_balance=float(abs(ag-base[37:45].sum()-z[:,45]+z[:,46]+z[:,47]+z[:,49]).max()),
        calcium_balance_uM=float(abs(z[:,0]+z[:,23]-z[:,22]).max()),calcium_AUC_identity=float(abs(z[:,24]-p[1]*z[:,23]).max()),
        minimum_state=float(min(z.min(),(z[:,1]-z[:,28]).min(),(z[:,2]-z[:,29]).min(),(z[:,31]-z[:,32]).min())))
    if not np.isfinite(z).all() or checks['minimum_state']<-1e-7 or max(v for k,v in checks.items() if k!='minimum_state')>1e-5:raise ArithmeticError(str(checks))
    cp=dict(time_s=float(t[-1]),state=y.tolist(),baseline_state=base.tolist(),origin_time_s=float(origin),synthesis_per_s=float(p[20]),fingerprint=fingerprint)
    return dict(time_s=t,state=z,columns=AGGREGATION_NAMES,calcium_uM=p[0]+z[:,0],activity=(z[:,28]+z[:,29]+z[:,32])*(1-b),
        calpain_block_fraction=b,aggregate_total=ag,aggregate_full_length=z[:,37]+z[:,38]+z[:,41]+z[:,42],
        aggregate_N_fragments=z[:,39]+z[:,40]+z[:,43]+z[:,44],soluble_intact_TDP=z[:,6]+z[:,7],
        total_intact_TDP=z[:,6]+z[:,7]+z[:,37]+z[:,38]+z[:,41]+z[:,42],nuclear_function_proxy=z[:,6]+z[:,7],
        baseline_state=base,baseline_activity=float(a0),checkpoint=cp,config=config,checks=checks,segments=sg,nfev=nfev,provenance=provenance,
        human_kinetic_fit=False,electrical_feedback=False,
        limitations=['Effective soluble-to-aggregate conversion; no explicit seeding, fibril growth, oligomer size or spatial compartment',
            'N-lineage accounting; calpain N-fragments are not caspase CTF or Cascella C-terminal inclusion bodies',
            'Fast/slow are operational pools, not identified UPS/ALP pathways',
            'Aggregate pools do not provide normal nuclear TDP function; soluble fraction uses inherited equal-partition assumption',
            'Baseline includes unknown steady aggregates; compare load with matching no-load arm',
            'Instantaneous catalytic suppression is not validated pharmacology; possible indirect/off-target effects absent',
            'No prediction of NMJ survival, neuron death or clinical benefit'])

def aggregation_evidence():
    return json.loads((_HERE/'evidence.json').read_text(encoding='utf-8'))

def source_training_data():
    out=_prior.source_training_data();out['connected_TDP_aggregation']=aggregation_evidence();return out
