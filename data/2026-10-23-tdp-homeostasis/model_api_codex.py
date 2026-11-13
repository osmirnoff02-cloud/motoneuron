"""[codex] 2026-10-23. TDP autoregulation extension; no human MN kinetic fit.

One basal soluble intact TDP unit is a normalization, not a concentration.
The new state is effective synthesis capacity, NOT measured total TARDBP RNA.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util, sys, json, hashlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq
from numba import njit
_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_aggregation_before_homeostasis',_HERE.parent/'2026-10-23-tdp-aggregation-chain/model_api_codex.py')
_agg=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_agg;_spec.loader.exec_module(_agg)
for _name in dir(_agg):
    if not _name.startswith('_') and _name not in globals():globals()[_name]=getattr(_agg,_name)
HOMEOSTASIS_NAMES=AGGREGATION_NAMES+('effective_TDP_synthesis_capacity',)

@dataclass(frozen=True)
class TDPAutoregulationParameters:
    enabled: bool
    minimum_capacity: float
    maximum_capacity: float
    hill: float
    response_tau_s: float
    provenance: str
    def vector(self):
        a=np.array([float(self.enabled),self.minimum_capacity,self.maximum_capacity,self.hill,self.response_tau_s])
        if not isinstance(self.enabled,bool) or not np.isfinite(a).all() or not 0<=a[1]<1<a[2] or min(a[3:])<=0 or not self.provenance.strip():
            raise ValueError('Explicit bounded feedback, positive response time/Hill exponent and provenance required')
        return a

@njit
def _feedback_target(n,hv):
    if hv[0]==0.:return 1.
    # Exactly 1 at basal functional proxy=1. Limits floor/ceiling are assumptions.
    return hv[1]+(hv[2]-hv[1])/(1.+(hv[2]-1.)/(1.-hv[1])*max(n,0.)**hv[3])

_old_rhs=_agg._aggregation_rhs
@njit
def _homeostasis_rhs(y,j,p,kv,sv,av,hv,block):
    out=np.zeros(51);out[:50]=_old_rhs(y[:50],j,p,kv,sv,av,block)
    change=p[20]*(y[50]-1.)
    out[6]+=change;out[18]+=change
    out[50]=(_feedback_target(y[6]+y[7],hv)-y[50])/hv[4]
    return out

def calcium_homeostasis_chain(segments,*,autoregulation_parameters,aggregation_parameters,
        provenance,checkpoint=None,output_step_s=1800.,rtol=1e-8,atol=1e-10,max_step_s=3600.,**kw):
    """Contiguous rows: start_s,end_s,Ca flux,calpain catalytic block [0,1].

    All arms use the same untreated equilibrium and basal synthesis. Inhibitor
    does not trigger reinitialization or a new synthesis calibration. Physical
    state and the feedback state persist across pauses/checkpoints.
    """
    required={'basal_ca_uM','calcium_tau_s','enzyme_parameters','tdp_parameters','localization_parameters','stmn2_parameters','inactivation_parameters','phosphorylation_parameters','reactivation_parameters','stability_parameters'}
    if set(kw)!=required:raise ValueError('Unexpected/missing fields: '+str(set(kw)^required))
    sg=np.asarray(segments,float)
    if sg.ndim!=2 or sg.shape[1]!=4 or not len(sg) or not np.isfinite(sg).all() or np.any(sg[:,0]<0) or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[:,2:]<0) or np.any(sg[:,3]>1) or np.any(sg[1:,0]!=sg[:-1,1]):raise ValueError('Contiguous valid flux/suppression segments required')
    scales=[kw['basal_ca_uM'],kw['calcium_tau_s'],output_step_s,rtol,atol,max_step_s]
    if not np.isfinite(scales).all() or min(scales)<=0 or not isinstance(provenance,str) or not provenance.strip():raise ValueError('Positive scales and provenance required')
    av=aggregation_parameters.vector();hv=autoregulation_parameters.vector()
    b,p,kv,sv,a0=_agg._baseline(kw,av);base=np.r_[b,1.]
    config={k:asdict(v) if hasattr(v,'__dataclass_fields__') else v for k,v in kw.items()}
    config.update(topology='TDP_homeostasis_51states_v1',aggregation=asdict(aggregation_parameters),autoregulation=asdict(autoregulation_parameters))
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if checkpoint is None:y=base.copy();origin=sg[0,0]
    else:
        if checkpoint['fingerprint']!=fingerprint or checkpoint['time_s']!=sg[0,0] or checkpoint['synthesis_per_s']!=p[20] or not np.array_equal(checkpoint['baseline_state'],base):raise ValueError('Changed time, parameters or untreated baseline')
        y=np.asarray(checkpoint['state'],float).copy();origin=checkpoint['origin_time_s']
        if y.shape!=(51,) or not np.isfinite(y).all() or y.min()<-1e-7 or y[50]<hv[1]-1e-7 or y[50]>hv[2]+1e-7:raise ValueError('Invalid checkpoint')
    times=[np.array([sg[0,0]])];states=[y[None,:]];nfev=0
    for start,end,j,block in sg:
        near=start+np.geomspace(1e-4,min(end-start,3600.),70)
        grid=np.unique(np.r_[np.arange(start+output_step_s,end,output_step_s),near[near<end],end])
        sol=solve_ivp(lambda t,z:_homeostasis_rhs(z,j,p,kv,sv,av,hv,block),(start,end),y,t_eval=grid,method='LSODA',rtol=rtol,atol=atol,max_step=max_step_s)
        if not sol.success:raise RuntimeError(sol.message)
        y=sol.y[:,-1].copy();times.append(sol.t);states.append(sol.y.T);nfev+=sol.nfev
    t=np.concatenate(times);z=np.concatenate(states);ag=z[:,37:45].sum(1)
    indexes=np.clip(np.searchsorted(sg[:,0],t,side='right')-1,0,len(sg)-1);blocks=sg[indexes,3]
    enzyme=z[:,1:6].sum(1)+z[:,31]+z[:,33];enzyme0=base[1:6].sum()+base[31]+base[33]
    td=z[:,6:10].sum(1)+ag;td0=base[6:10].sum()+base[37:45].sum()
    checks=dict(enzyme_balance=float(abs(enzyme+z[:,15]-z[:,14]-enzyme0).max()),TDP_lineage_balance=float(abs(td+z[:,19]-z[:,18]-td0).max()),
        aggregate_balance=float(abs(ag-base[37:45].sum()-z[:,45]+z[:,46]+z[:,47]+z[:,49]).max()),
        calcium_balance_uM=float(abs(z[:,0]+z[:,23]-z[:,22]).max()),calcium_AUC_identity=float(abs(z[:,24]-p[1]*z[:,23]).max()),
        minimum_state=float(min(z.min(),(z[:,1]-z[:,28]).min(),(z[:,2]-z[:,29]).min(),(z[:,31]-z[:,32]).min())))
    if not np.isfinite(z).all() or checks['minimum_state']<-1e-7 or max(v for k,v in checks.items() if k!='minimum_state')>1e-5 or z[:,50].min()<hv[1]-1e-7 or z[:,50].max()>hv[2]+1e-7:raise ArithmeticError(str(checks))
    cp=dict(time_s=float(t[-1]),state=y.tolist(),baseline_state=base.tolist(),origin_time_s=float(origin),synthesis_per_s=float(p[20]),fingerprint=fingerprint)
    return dict(time_s=t,state=z,columns=HOMEOSTASIS_NAMES,calcium_uM=p[0]+z[:,0],activity=(z[:,28]+z[:,29]+z[:,32])*(1-blocks),
        calpain_block_fraction=blocks,aggregate_total=ag,soluble_intact_TDP=z[:,6]+z[:,7],
        aggregate_full_length=z[:,37]+z[:,38]+z[:,41]+z[:,42],aggregate_N_fragments=z[:,39]+z[:,40]+z[:,43]+z[:,44],
        total_intact_TDP=z[:,6]+z[:,7]+z[:,37]+z[:,38]+z[:,41]+z[:,42],nuclear_function_proxy=z[:,6]+z[:,7],
        synthesis_capacity=z[:,50],synthesis_per_s=p[20]*z[:,50],basal_synthesis_per_s=float(p[20]),
        baseline_state=base,baseline_activity=float(a0),checkpoint=cp,config=config,checks=checks,segments=sg,nfev=nfev,provenance=provenance,
        human_kinetic_fit=False,electrical_feedback=False,absolute_TDP_per_cell=None,
        limitations=['Normalized soluble intact baseline=1; total/aggregate burden and molecules per human MN unknown',
            'Effective synthesis feedback, not explicit TARDBP isoforms, ribosomes or nuclear import/export',
            'Feedback depends on inherited soluble-function proxy; no calibrated nuclear concentration',
            'Feedback gain and response time are sensitivity assumptions, not fitted from 72h endpoints',
            'N-lineage units are not protein mass, full immunoblot signal or peptide-specific SILAC',
            'No prediction of human clinical benefit or NMJ survival'])

def baseline_turnover_observer(*,aggregation_parameters,**kw):
    """Pulse-chase of two explicitly stated proxies, not a fitted SILAC assay.

    First-order transfer network at fixed untreated Ca/calpain. Label cannot be
    newly synthesized. Loss of intact epitope differs from loss of N-lineage.
    """
    kw={k:v for k,v in kw.items() if k not in ('provenance','autoregulation_parameters')}
    av=aggregation_parameters.vector()
    if av[14]!=1:raise ValueError('Linear pulse-chase observer requires formation_order=1')
    base,p,kv,sv,_=_agg._baseline(kw,av);p=p.copy();p[20]=0.
    ix=np.r_[np.arange(6,10),np.arange(37,45)]
    fixed=base.copy();fixed[ix]=0.
    zero=_old_rhs(fixed,0.,p,kv,sv,av,0.)[ix]
    M=np.column_stack([_old_rhs(fixed+np.eye(50)[j],0.,p,kv,sv,av,0.)[ix]-zero for j in ix])
    if abs(zero).max()>1e-12:raise ArithmeticError('Unexpected label source')
    label=base[ix];times=np.array([0.,6.,24.,48.,96.,168.,336.,720.])*3600.
    states=np.array([expm(M*t)@label for t in times])
    weights={'intact_epitope':np.array([1.,1.,0.,0.,1.,1.,0.,0.,1.,1.,0.,0.]),'N_lineage':np.ones(12)}
    out={}
    for name,w in weights.items():
        initial=w@label;fn=lambda t:float(w@expm(M*t)@label/initial-.5)
        high=3600.
        while fn(high)>0 and high<100*365*86400.:high*=2
        half=brentq(fn,0.,high) if fn(high)<=0 else None
        out[name]=dict(half_life_h=half/3600 if half is not None else None,relative_signal=(states@w/initial).tolist())
    return dict(time_h=(times/3600).tolist(),signals=out,baseline_soluble_intact=1.,baseline_total_N_lineage=float(label.sum()),
        baseline_aggregate=float(base[37:45].sum()),baseline_N_fragments=float(base[8:10].sum()),
        basal_synthesis_per_h=float(_agg._baseline(kw,av)[1][20]*3600),absolute_amount_unknown=True,
        interpretation='Assumed epitope observers; NOT peptide-resolved SILAC or measured human MN turnover')

def homeostasis_evidence():
    return json.loads((_HERE/'evidence_codex.json').read_text(encoding='utf-8'))

def source_training_data():
    out=_agg.source_training_data();out['TDP_homeostasis']=homeostasis_evidence();return out
