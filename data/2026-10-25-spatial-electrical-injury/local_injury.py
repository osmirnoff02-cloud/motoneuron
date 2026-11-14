"""[codex] 2026-10-25. Local calpain and distinct process protein lineages.

One-way Ca -> local CAPN -> substrates; no invented conversion of axonal
fragments into nuclear TDP loss, STMN2 failure, NMJ failure or cell death.
"""
from pathlib import Path
from dataclasses import dataclass,asdict
import sys,importlib.util,json,hashlib
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
B=Path(__file__).resolve().parent
def _load(path,name):
    if name in sys.modules:return sys.modules[name]
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
capn=_load(B.parent/'2026-10-23-calpain-seven-papers/model_api.py','_local_injury_capn_source_20261025')
enzyme_rhs=capn._stability_enzyme_rhs

@dataclass(frozen=True)
class LocalSubstrateParameters:
    name: str
    initial_intact_amount: float
    cleavage_per_activity_s: float
    intact_half_life_s: float
    fragment_half_life_s: float
    provenance: str
    def vector(self):
        vals=[self.initial_intact_amount,self.cleavage_per_activity_s,self.intact_half_life_s,self.fragment_half_life_s]
        if not self.name or not self.provenance.strip() or not np.isfinite(vals).all() or min(vals)<0 or min(vals[0],vals[2],vals[3])<=0:raise ValueError('Explicit positive local pool/half lives, nonnegative cleavage and source required')
        return np.array([vals[1],np.log(2)/vals[2],np.log(2)/vals[3]])

LOCAL_INJURY_RANGES=dict(calpain_Ca_half_scale=[.3,1.,3.],calpain_pool_scale=[.1,1.,3.],gating_tau_s=[.1,10.,100.],
    cleavage_rate_scale=[.3,1.,3.],intact_half_life_s=[3600.,36000.,172800.],fragment_half_life_s=[600.,3600.,36000.],
    process_TDP_initial_amount_relative_to_soma=[.01,.05,.2],effective_substrate_block_fraction=[0.,.5,.9,1.],
    betaII_spectrin_cleavage_per_activity_s=[.0001,.001,.01])

@njit
def _rhs(z,c,block,ev,iv,kv,sv,pv,syn):
    out=np.zeros(len(z));out[:10]=enzyme_rhs(z[:10],c,ev,iv,kv,sv)
    act=(z[1]+z[3]+z[8])*(1-block)
    out[10]=ev[7];out[11]=ev[6]*np.sum(z[:10]);out[12]=act
    for j in range(len(pv)):
        b=13+5*j;rate,du,df=pv[j]
        cleave=rate*act*z[b];lu=du*z[b];lf=df*z[b+1]
        out[b]=syn[j]-lu-cleave;out[b+1]=cleave-lf
        out[b+2]=lu+lf;out[b+3]=syn[j];out[b+4]=cleave
    return out

def simulate_local_injury(time_s,calcium_uM,*,basal_ca_uM,enzyme_parameters,inactivation_parameters,
        reactivation_parameters,stability_parameters,substrates,provenance,compartment='process',
        enzyme_pool_scale=1.,block_schedule=None,checkpoint=None,max_step_s=2.,rtol=1e-8,atol=1e-10):
    """Calcium is prescribed, not recreated by a second Ca model.

    TDP amount is a NEW explicitly local pool relative to the somatic reference;
    betaII-spectrin has its OWN reference unit. No sums of unlike proteins.
    One retained fragment per precursor = lineage, not full molecular mass.
    Source and basal fragment equilibrium fixed once in untreated origin.
    block_schedule rows [start,end,fraction] affect substrate catalysis only;
    not drug concentration, autolysis blockade or pharmacokinetics.
    """
    t=np.asarray(time_s,float);ca=np.asarray(calcium_uM,float)
    if t.ndim!=1 or len(t)<2 or ca.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(ca).all() or np.any(np.diff(t)<=0) or t[0]<0 or ca.min()<0:raise ValueError('Finite increasing seconds and nonnegative Ca required')
    if not provenance.strip() or not compartment.strip() or not np.isfinite([basal_ca_uM,enzyme_pool_scale,max_step_s,rtol,atol]).all() or min(basal_ca_uM,enzyme_pool_scale,max_step_s,rtol,atol)<=0:raise ValueError('Explicit input origin and positive kinetic settings required')
    if not substrates or len({p.name for p in substrates})!=len(substrates):raise ValueError('Distinct explicit local substrates required')
    ev=enzyme_parameters.vector().copy();ev[7]*=enzyme_pool_scale
    iv=inactivation_parameters.vector();kv=reactivation_parameters.vector();sv=stability_parameters.vector()
    source=enzyme_rhs(np.zeros(10),basal_ca_uM,ev,iv,kv,sv)
    mat=np.column_stack([enzyme_rhs(np.eye(10)[i],basal_ca_uM,ev,iv,kv,sv)-source for i in range(10)])
    enz=np.linalg.solve(mat,-source);a0=enz[1]+enz[3]+enz[8]
    if enz.min()<-1e-12:raise ArithmeticError('Invalid enzyme equilibrium')
    pv=np.array([p.vector() for p in substrates]);syn=np.array([(p[1]+p[0]*a0)*s.initial_intact_amount for s,p in zip(substrates,pv)])
    base=np.r_[enz,np.zeros(3),np.concatenate([[s.initial_intact_amount,p[0]*a0*s.initial_intact_amount/p[2],0.,0.,0.] for s,p in zip(substrates,pv)])]
    bs=np.array([[t[0],t[-1],0.]]) if block_schedule is None else np.asarray(block_schedule,float)
    if bs.ndim!=2 or bs.shape[1]!=3 or not len(bs) or not np.isfinite(bs).all() or bs[0,0]!=t[0] or bs[-1,1]!=t[-1] or np.any(bs[:,1]<=bs[:,0]) or np.any(bs[1:,0]!=bs[:-1,1]) or np.any((bs[:,2]<0)|(bs[:,2]>1)):raise ValueError('Block schedule must cover entire absolute clock')
    config=dict(topology='local_CAPN10_substrate_lineages_v1',compartment=compartment,basal_ca_uM=basal_ca_uM,enzyme=asdict(enzyme_parameters),
        inactivation=asdict(inactivation_parameters),reactivation=asdict(reactivation_parameters),stability=asdict(stability_parameters),
        enzyme_pool_scale=enzyme_pool_scale,substrates=[asdict(s) for s in substrates],provenance=provenance)
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if checkpoint is None:y=base.copy();origin=t[0]
    else:
        if checkpoint['fingerprint']!=fingerprint or checkpoint['time_s']!=t[0] or not np.array_equal(base,checkpoint['baseline_state']) or not np.isclose(checkpoint['last_calcium_uM'],ca[0],atol=1e-10,rtol=0):raise ValueError('Changed local pool/parameters/time/calcium at checkpoint')
        y=np.asarray(checkpoint['state'],float);origin=checkpoint['origin_time_s']
        if y.shape!=base.shape or not np.isfinite(y).all() or y.min()<-1e-7:raise ValueError('Invalid local checkpoint')
    zs=[y[None,:]];ts=[t[:1]];nfev=0
    for start,end,block in bs:
        # Split at explicit treatment transitions. Max step limits missed Ca peaks.
        grid=np.unique(np.r_[t[(t>start)&(t<end)],end])
        sol=solve_ivp(lambda ti,z:_rhs(z,float(np.interp(ti,t,ca)),block,ev,iv,kv,sv,pv,syn),(start,end),y,
            t_eval=grid,method='LSODA',max_step=max_step_s,rtol=rtol,atol=atol)
        if not sol.success:raise RuntimeError(sol.message)
        y=sol.y[:,-1].copy();ts.append(sol.t);zs.append(sol.y.T);nfev+=sol.nfev
    tt=np.concatenate(ts);z=np.concatenate(zs);idx=np.clip(np.searchsorted(bs[:,0],tt,side='right')-1,0,len(bs)-1)
    act=(z[:,1]+z[:,3]+z[:,8])*(1-bs[idx,2]);err=lambda v:float(np.max(np.abs(v)))
    checks=dict(enzyme_lineage=err(z[:,:10].sum(1)+z[:,11]-z[:,10]-base[:10].sum()),minimum_state=float(z.min()))
    products={};columns=['native_closed','native_open','processed_closed','processed_open','native_CAST','processed_CAST','inactive','fragile_closed','fragile_open','fragile_CAST','enzyme_synthesized','enzyme_removed','activity_integral_s']
    for j,s in enumerate(substrates):
        b=13+5*j;checks[s.name+'_lineage']=err(z[:,b]+z[:,b+1]+z[:,b+2]-z[:,b+3]-base[b]-base[b+1])
        products[s.name]=dict(intact=z[:,b],fragment=z[:,b+1],removed=z[:,b+2],synthesized=z[:,b+3],cleavage_cumulative=z[:,b+4],
            intact_fraction_of_initial=z[:,b]/base[b],baseline_fragment=float(base[b+1]),initial_intact=float(base[b]))
        columns += [s.name+'_'+k for k in ('intact','fragment','removed','synthesized','cleavage_cumulative')]
    if not np.isfinite(z).all() or checks['minimum_state']<-1e-7 or max(v for k,v in checks.items() if k!='minimum_state')>1e-5:raise ArithmeticError(str(checks))
    cp=dict(time_s=float(tt[-1]),last_calcium_uM=float(ca[-1]),state=y.tolist(),baseline_state=base.tolist(),origin_time_s=float(origin),fingerprint=fingerprint)
    return dict(time_s=tt,state=z,columns=columns,calcium_uM=np.interp(tt,t,ca),activity=act,baseline_activity=float(a0),baseline_state=base,
        products=products,config=config,checkpoint=cp,checks=checks,nfev=nfev,block_schedule=bs,
        nuclear_TDP_effect=None,STMN2_effect=None,protein_synthesis_ATP_cost=None,human_kinetic_fit=False,feedback_to_Ca_voltage=False,
        limitations=['Inherited CAPN10 processing/CAST/gating, one effective isoform population; human local rates uncalibrated.',
        'Local TDP is cytoplasmic substrate; no automatic loss of nuclear TDP/STMN2 or neuronal function.',
        'New local pools do not duplicate the somatic pool; absolute amounts and soma-process protein transport are unknown.',
        'No substrate competition, secondary fragment cleavage, aggregation, phosphorylation or epitope assignment in these two local reporters.',
        'Protein synthesis/clearance energy is not yet returned to ATP; only pump energy is closed in the source model.',
        'Ca interpolation is a feed-forward coupling. No repetition of short traces into hours or inferred calpain from FRET directly.'])

def run_spatial_injury_network(segments,*,injury_parameters,checkpoint=None,**spatial_parameters):
    sys.path.insert(0,str(B.parent/'2026-10-25-pmca-spatial-model'))
    from spatial_pmca import run_spatial_pmca_network
    r=run_spatial_pmca_network(segments,checkpoint=None if checkpoint is None else checkpoint['spatial'],**spatial_parameters)
    local=simulate_local_injury(r['time_s'],r['process_calcium_uM'],checkpoint=None if checkpoint is None else checkpoint['process_injury'],**injury_parameters)
    return dict(spatial=r,process_injury=local,checkpoint=dict(spatial=r['checkpoint'],process_injury=local['checkpoint']),
        human_kinetic_fit=False,protein_damage_feedback=False)

def injury_from_electrical(recording,*,injury_parameters,compartment='process',sample_ms=.5,max_step_s=.002):
    """Feed actual native bulk Ca to CAPN; no tiling, extension or nuclear effect.

    Parameters must name the compartment and match the input baseline. A local
    enzyme/protein baseline is constructed explicitly at the start of this
    short experiment; this is not a bridge from a previous long checkpoint.
    """
    if compartment not in ('soma','process') or not np.isfinite(sample_ms) or sample_ms<=0:raise ValueError('Explicit soma/process and positive sample_ms required')
    t=np.asarray(recording['t']);ca=np.asarray(recording[compartment+'_cb'])
    if injury_parameters.get('compartment', 'process')!=compartment or not np.isclose(injury_parameters['basal_ca_uM'],ca[0],rtol=0,atol=1e-12):raise ValueError('Local parameters must match recorded compartment/baseline')
    indices=np.unique(np.r_[np.searchsorted(t,np.arange(t[0],t[-1],sample_ms)),len(t)-1])
    r=simulate_local_injury(t[indices]/1000,ca[indices],max_step_s=max_step_s,**injury_parameters)
    r['native_input_sampling_ms']=sample_ms
    return r

def run_spatial_electrical_injury(*,injury_parameters,compartment='process',sample_ms=.5,**electrical_parameters):
    if electrical_parameters.get('geometry') is None:raise ValueError('Explicit electrical two-section geometry required')
    sys.path.insert(0,str(B))
    from electrical_spatial_codex import run_spatial_electrical
    r=run_spatial_electrical(**electrical_parameters)
    injury=injury_from_electrical(r,injury_parameters=injury_parameters,compartment=compartment,sample_ms=sample_ms)
    return dict(electrical=r,local_injury=injury,protein_damage_feedback=False,human_kinetic_fit=False)
