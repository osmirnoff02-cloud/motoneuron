"""[codex] 2026-10-23. Ayala-informed RNA feedback, NOT a human MN fit.

Published protein turnover references are separate assay-specific benchmarks.
Prior APIs are re-exported unchanged; new chain has the same 51 physical states.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util, sys, json, hashlib
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_homeostasis_before_rna',_HERE.parent/'2026-10-23-tdp-homeostasis/model_api_codex.py')
_home=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_home;_spec.loader.exec_module(_home)
for _name in dir(_home):
    if not _name.startswith('_') and _name not in globals():globals()[_name]=getattr(_home,_name)
_agg=_home._agg
_feedback_target=_home._feedback_target
_old_home_rhs=_home._homeostasis_rhs

@dataclass(frozen=True)
class RNAFeedbackParameters:
    decay_fraction: float
    provenance: str
    def vector(self):
        if not np.isfinite(self.decay_fraction) or not 0<=self.decay_fraction<=1 or not self.provenance.strip():
            raise ValueError('RNA decay partition must be in [0,1] with explicit provenance')
        return np.array([self.decay_fraction])

@njit
def _rna_feedback_rhs(y,j,p,kv,sv,av,hv,block):
    out=_old_home_rhs(y,j,p,kv,sv,av,hv[:5],block)
    target=_feedback_target(y[6]+y[7],hv)
    # Equal steady state m=target; only kinetics/route of RNA regulation differ.
    # w=0 exactly recovers prior regulated-input/fixed-decay feedback.
    # w=1: constant productive input, TDP-dependent RNA degradation.
    out[50]=(target**(1.-hv[5])-y[50]/target**hv[5])/hv[4]
    return out

def calcium_rna_feedback_chain(segments,*,autoregulation_parameters,aggregation_parameters,rna_feedback_parameters,
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
    av=aggregation_parameters.vector();hv=np.r_[autoregulation_parameters.vector(),rna_feedback_parameters.vector()]
    if hv[5]>0 and hv[1]==0:raise ValueError("Decay regulation requires a strictly positive feedback floor")
    b,p,kv,sv,a0=_agg._baseline(kw,av);base=np.r_[b,1.]
    config={k:asdict(v) if hasattr(v,'__dataclass_fields__') else v for k,v in kw.items()}
    config.update(topology='TDP_RNA_feedback_51states_v2',rna_feedback=asdict(rna_feedback_parameters),aggregation=asdict(aggregation_parameters),autoregulation=asdict(autoregulation_parameters))
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
        sol=solve_ivp(lambda t,z:_rna_feedback_rhs(z,j,p,kv,sv,av,hv,block),(start,end),y,t_eval=grid,method='LSODA',rtol=rtol,atol=atol,max_step=max_step_s)
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
        effective_productive_RNA=z[:,50],rna_decay_fraction=float(hv[5]),synthesis_capacity=z[:,50],synthesis_per_s=p[20]*z[:,50],basal_synthesis_per_s=float(p[20]),
        baseline_state=base,baseline_activity=float(a0),checkpoint=cp,config=config,checks=checks,segments=sg,nfev=nfev,provenance=provenance,
        human_kinetic_fit=False,electrical_feedback=False,absolute_TDP_per_cell=None,
        limitations=['Normalized soluble intact baseline=1; total/aggregate burden and molecules per human MN unknown',
            'Effective productive-RNA/translation-capacity proxy, not measured total TARDBP RNA, explicit isoforms or ribosomes',
            'Partition between productive RNA input and RNA decay is a sensitivity assumption; equilibrium feedback is matched across mechanisms',
            'Feedback depends on inherited soluble-function proxy; no calibrated nuclear concentration',
            'Feedback gain and response time are sensitivity assumptions, not fitted from 72h endpoints',
            'N-lineage units are not protein mass, full immunoblot signal or peptide-specific SILAC',
            'No prediction of human clinical benefit or NMJ survival'])

def rna_feedback_step(time_s, *, functional_proxy, autoregulation_parameters,
        rna_feedback_parameters, initial_capacity=1., productive_input_on=True):
    """Clamped-driver analytic reference; ActD mode is NOT the full Ayala assay.

    In Ayala the inducing transgene also decays. A clamped driver cannot be
    fitted to that curve without its measured trajectory. Each chase is
    normalized to its own starting level; input-off can go below feedback floor.
    """
    t=np.asarray(time_s,float);hv=autoregulation_parameters.vector();w=rna_feedback_parameters.vector()[0]
    if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or np.any(t<0) or np.any(np.diff(t)<0) or not np.isfinite([functional_proxy,initial_capacity]).all() or min(functional_proxy,initial_capacity)<0 or not isinstance(productive_input_on,bool):
        raise ValueError('Finite nonnegative states and ordered times required')
    if w>0 and hv[1]==0:raise ValueError('Decay regulation requires a strictly positive feedback floor')
    target=float(_feedback_target(functional_proxy,hv))
    k=1./(hv[4]*target**w);production=target**(1.-w)/hv[4] if productive_input_on else 0.
    eq=production/k;m=eq+(initial_capacity-eq)*np.exp(-k*t)
    return dict(time_s=t,capacity=m,production_per_s=production,decay_per_s=k,
        fixed_driver_equilibrium=eq,response_tau_s=1./k,
        relative_to_start=m/initial_capacity if initial_capacity>0 else None,
        human_fit=False,source_assay_fit=False)

def protein_chase_reference(time_h,*,system,variant):
    """Published approximate single-exponential guides, NOT digitized raw data.

    HeLa tagged transgenes and heterozygous fibroblast mixtures remain separate.
    Values beyond the 12-hour HeLa observation window are extrapolations.
    No reference is assigned to intrinsic proteolytic loss in the MN model.
    """
    table={('HeLa_tagged','WT'):12.,('HeLa_tagged','Q331K'):24.,
        ('HeLa_tagged','G298S'):48.,('HeLa_tagged','M337V'):48.,
        ('primary_fibroblast','WT'):4.,('primary_fibroblast','heterozygous_G298S'):11.}
    if (system,variant) not in table:raise ValueError('Only the measured source systems/mixture labels are supported; no human-MN reference')
    t=np.asarray(time_h,float)
    if t.ndim!=1 or not np.isfinite(t).all() or np.any(t<0):raise ValueError('Finite nonnegative hours required')
    half=table[system,variant]
    return dict(time_h=t,relative_label=np.exp(-np.log(2)*t/half),half_life_h=half,
        doi='10.1073/pnas.1008227107',source='Fig2B/C; approximate published estimates, not raw-point fit',
        system=system,variant=variant,heterozygous_mixture=(variant=='heterozygous_G298S'),
        beyond_HeLa_observation_window=(t>12) if system=='HeLa_tagged' else None,human_MN_fit=False)

def source_evidence():
    return json.loads((_HERE/'evidence_codex.json').read_text(encoding='utf-8'))

def source_training_data():
    out=_home.source_training_data()
    reference=out['TDP_homeostasis']['baseline']['HeLa_reference']
    reference['read_at_previous_stage']=reference['read']
    reference['read']='Full user-provided Ling PDF now read; Fig1/2 and References checked; S1 remains unavailable'
    out['Ling_Ayala_RNA_feedback']=source_evidence();return out
