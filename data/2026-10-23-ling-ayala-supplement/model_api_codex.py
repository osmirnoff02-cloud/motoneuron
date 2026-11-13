"""[codex] 2026-10-23. Ling SI: finite pulse, extraction and band normalization.

The 51 biological/accounting variables are unchanged. Twelve passive label
variables neither change kinetics nor represent new organelles. No human fit.
"""
from pathlib import Path
from dataclasses import asdict
import importlib.util,sys,json,hashlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from numba import njit
_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_rna_before_LingSI',_HERE.parent/'2026-10-23-ling-ayala-synthesis/model_api_codex.py')
_prior=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_prior;_spec.loader.exec_module(_prior)
for _name in dir(_prior):
    if not _name.startswith('_') and _name not in globals():globals()[_name]=getattr(_prior,_name)
_agg=_prior._agg
_rna_rhs=_prior._rna_feedback_rhs
_agg_rhs=_agg._aggregation_rhs
LABEL_INDEX=np.array([6,7,8,9,37,38,39,40,41,42,43,44])
TRACER_NAMES=HOMEOSTASIS_NAMES+tuple('labelled_'+AGGREGATION_NAMES[i] for i in LABEL_INDEX)

@njit
def _label_rhs(label,total,p,kv,sv,av,block):
    y=total[:50].copy();pp=p.copy();pp[20]=0.
    for k in range(12):y[LABEL_INDEX[k]]=label[k]
    return _agg_rhs(y,0.,pp,kv,sv,av,block)[LABEL_INDEX]

@njit
def _pulse_chase_rhs(y,j,p,kv,sv,av,hv,block):
    out=np.zeros(63);out[:51]=_rna_rhs(y[:51],j,p,kv,sv,av,hv,block)
    out[51:]=_label_rhs(y[51:],y[:51],p,kv,sv,av,block)
    return out

def _equilibrium_pulse(total,p,kv,sv,av,duration,fraction):
    # Continuous translation during finite pulse; no pre-existing label.
    zero=_label_rhs(np.zeros(12),total,p,kv,sv,av,0.)
    if abs(zero).max()>1e-12:raise ArithmeticError('Unexpected labelled source')
    M=np.column_stack([_label_rhs(np.eye(12)[k],total,p,kv,sv,av,0.) for k in range(12)])
    aug=np.zeros((13,13));aug[:12,:12]=M;aug[0,12]=p[20]*fraction
    initial=np.zeros(13);initial[12]=1.
    label=(expm(aug*duration)@initial)[:12]
    if label.min()<-1e-10 or np.max(label-total[LABEL_INDEX])>1e-8:raise ArithmeticError('Invalid finite-pulse state')
    return label

def calcium_pulse_chase_chain(segments,*,autoregulation_parameters,aggregation_parameters,rna_feedback_parameters,
        provenance,pulse_duration_s=1800.,label_fraction=1.,checkpoint=None,output_step_s=1800.,rtol=1e-8,atol=1e-10,max_step_s=3600.,**kw):
    """Contiguous rows: start_s,end_s,Ca flux,calpain catalytic block [0,1].

    All arms use the same untreated equilibrium and basal synthesis. Inhibitor
    does not trigger reinitialization or a new synthesis calibration. Physical
    state and the feedback state persist across pauses/checkpoints.
    """
    required={'basal_ca_uM','calcium_tau_s','enzyme_parameters','tdp_parameters','localization_parameters','stmn2_parameters','inactivation_parameters','phosphorylation_parameters','reactivation_parameters','stability_parameters'}
    if set(kw)!=required:raise ValueError('Unexpected/missing fields: '+str(set(kw)^required))
    sg=np.asarray(segments,float)
    if sg.ndim!=2 or sg.shape[1]!=4 or not len(sg) or not np.isfinite(sg).all() or np.any(sg[:,0]<0) or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[:,2:]<0) or np.any(sg[:,3]>1) or np.any(sg[1:,0]!=sg[:-1,1]):raise ValueError('Contiguous valid flux/suppression segments required')
    scales=[kw['basal_ca_uM'],kw['calcium_tau_s'],output_step_s,rtol,atol,max_step_s,pulse_duration_s]
    if not np.isfinite(scales).all() or min(scales)<=0 or not isinstance(provenance,str) or not provenance.strip():raise ValueError('Positive scales and provenance required')
    av=aggregation_parameters.vector();hv=np.r_[autoregulation_parameters.vector(),rna_feedback_parameters.vector()]
    if hv[5]>0 and hv[1]==0:raise ValueError("Decay regulation requires a strictly positive feedback floor")
    b,p,kv,sv,a0=_agg._baseline(kw,av);base=np.r_[b,1.]
    if av[14]!=1:raise ValueError("Passive tracer currently requires first-order aggregation")
    if not np.isfinite(label_fraction) or not 0<label_fraction<=1:raise ValueError("Label fraction in (0,1] required")
    base=np.r_[base,_equilibrium_pulse(base,p,kv,sv,av,pulse_duration_s,label_fraction)]
    config={k:asdict(v) if hasattr(v,'__dataclass_fields__') else v for k,v in kw.items()}
    config.update(topology='RNA51_plus_passive_tracer12_v1',pulse_duration_s=pulse_duration_s,label_fraction=label_fraction,rna_feedback=asdict(rna_feedback_parameters),aggregation=asdict(aggregation_parameters),autoregulation=asdict(autoregulation_parameters))
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if checkpoint is None:y=base.copy();origin=sg[0,0]
    else:
        if checkpoint['fingerprint']!=fingerprint or checkpoint['time_s']!=sg[0,0] or checkpoint['synthesis_per_s']!=p[20] or not np.array_equal(checkpoint['baseline_state'],base):raise ValueError('Changed time, parameters or untreated baseline')
        y=np.asarray(checkpoint['state'],float).copy();origin=checkpoint['origin_time_s']
        if y.shape!=(63,) or not np.isfinite(y).all() or y.min()<-1e-7 or y[50]<hv[1]-1e-7 or y[50]>hv[2]+1e-7:raise ValueError('Invalid checkpoint')
    times=[np.array([sg[0,0]])];states=[y[None,:]];nfev=0
    for start,end,j,block in sg:
        near=start+np.geomspace(1e-4,min(end-start,3600.),70)
        grid=np.unique(np.r_[np.arange(start+output_step_s,end,output_step_s),near[near<end],end])
        sol=solve_ivp(lambda t,z:_pulse_chase_rhs(z,j,p,kv,sv,av,hv,block),(start,end),y,t_eval=grid,method='LSODA',rtol=rtol,atol=atol,max_step=max_step_s)
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
    if np.max(z[:,51:]-z[:,LABEL_INDEX])>1e-7:raise ArithmeticError("Label exceeds total molecular lineage")
    if not np.isfinite(z).all() or checks['minimum_state']<-1e-7 or max(v for k,v in checks.items() if k!='minimum_state')>1e-5 or z[:,50].min()<hv[1]-1e-7 or z[:,50].max()>hv[2]+1e-7:raise ArithmeticError(str(checks))
    cp=dict(time_s=float(t[-1]),state=y.tolist(),baseline_state=base.tolist(),origin_time_s=float(origin),synthesis_per_s=float(p[20]),fingerprint=fingerprint)
    return dict(time_s=t,state=z,columns=TRACER_NAMES,biochemical_state=z[:,:51],labelled_lineage_states=z[:,51:],calcium_uM=p[0]+z[:,0],activity=(z[:,28]+z[:,29]+z[:,32])*(1-blocks),
        calpain_block_fraction=blocks,aggregate_total=ag,soluble_intact_TDP=z[:,6]+z[:,7],
        aggregate_full_length=z[:,37]+z[:,38]+z[:,41]+z[:,42],aggregate_N_fragments=z[:,39]+z[:,40]+z[:,43]+z[:,44],
        total_intact_TDP=z[:,6]+z[:,7]+z[:,37]+z[:,38]+z[:,41]+z[:,42],nuclear_function_proxy=z[:,6]+z[:,7],
        effective_productive_RNA=z[:,50],rna_decay_fraction=float(hv[5]),synthesis_capacity=z[:,50],synthesis_per_s=p[20]*z[:,50],basal_synthesis_per_s=float(p[20]),
        baseline_state=base,baseline_activity=float(a0),checkpoint=cp,config=config,checks=checks,segments=sg,nfev=nfev,provenance=provenance,
        human_kinetic_fit=False,electrical_feedback=False,absolute_TDP_per_cell=None,
        pulse_protocol=dict(duration_s=pulse_duration_s,label_fraction=label_fraction,unlabelled_synthesis_continues=True,prehistory="Untreated stationary model before chase; not reconstructed 18h transgene induction"),
        limitations=['Normalized soluble intact baseline=1; total/aggregate burden and molecules per human MN unknown',
            'Effective productive-RNA/translation-capacity proxy, not measured total TARDBP RNA, explicit isoforms or ribosomes',
            'Partition between productive RNA input and RNA decay is a sensitivity assumption; equilibrium feedback is matched across mechanisms',
            'Feedback depends on inherited soluble-function proxy; no calibrated nuclear concentration',
            'Feedback gain and response time are sensitivity assumptions, not fitted from 72h endpoints',
            'N-lineage units are not protein mass, full immunoblot signal or peptide-specific SILAC',
            'No prediction of human clinical benefit or NMJ survival',
            'Tracer tracks labelled N-lineage molecules, NOT all sulfur atoms or C-terminal fragments',
            'Only full-length band autoradiography is supported; extraction recovery is separately assumed',
            '30min pulse is source-based; steady pre-pulse history and unit label fraction are model assumptions'])

def pulse_chase_readout(result,*,aggregate_recovery,provenance,band='full_length'):
    """L(t)/P(t), normalized to chase zero; L alone is also returned.

    Recovery fractions for fast/slow aggregates are assumptions. Both labelled
    and total full-length molecules share capture efficiency within each pool.
    A RIPA-supernatant sample is not assumed to recover all aggregates or only
    model-soluble TDP. N/C fragment bands require additional sequence/epitope
    calibration and C-terminal states absent from this model.
    """
    r=np.asarray(aggregate_recovery,float)
    if r.shape!=(2,) or not np.isfinite(r).all() or np.any(r<0) or np.any(r>1) or not isinstance(provenance,str) or not provenance.strip():raise ValueError('Two explicit recovery fractions in [0,1] and provenance required')
    if band!='full_length':raise ValueError('Only full-length band supported; fragment epitopes and sulfur content are not calibrated')
    z=np.asarray(result['state'],float)
    if z.ndim!=2 or z.shape[1]!=63 or not np.isfinite(z).all():raise ValueError('63-state pulse-chase result required')
    weights=np.array([1.,1.,0.,0.,r[0],r[0],0.,0.,r[1],r[1],0.,0.])
    total=z[:,LABEL_INDEX]@weights;label=z[:,51:]@weights
    if np.any(total<=0) or label[0]<=0:raise ValueError('Positive band denominator and initial label required')
    base=np.asarray(result['baseline_state'],float)
    initial_total=base[LABEL_INDEX]@weights;initial_label=base[51:]@weights
    if initial_total<=0 or initial_label<=0:raise ValueError('Positive original chase-zero reference required')
    signal=label/total;normalized=signal/(initial_label/initial_total)
    # No fitted or extrapolated half-life: first observed bracket only.
    t=np.asarray(result['time_s'],float);cross=np.where(normalized<=.5)[0]
    origin=result['checkpoint']['origin_time_s']
    before_window=bool(len(cross) and cross[0]==0)
    bracket=None if not len(cross) or before_window else [float(t[cross[0]-1]-origin),float(t[cross[0]]-origin)]
    return dict(time_s=t,full_length_label=label,full_length_total=total,label_relative=label/initial_label,total_relative=total/initial_total,
        label_per_protein_relative=normalized,half_signal_bracket_s=bracket,already_below_half_at_window_start=before_window,aggregate_recovery=r,
        band=band,provenance=provenance,source_protocol='Ling2010 Fig2 and SI p1; no matched transgene induction/capture calibration',
        human_fit=False,fragment_signal_supported=False)

def antibody_calibrator_reference(antibody):
    """Qualitative S1 binding of recombinant fragments; not epitope coordinates."""
    table={'FL4':(False,True),'FL9':(False,True),'ProteinTech_10782':(True,False)}
    if antibody not in table:raise ValueError('Only source-tested antibodies supported')
    nt,ct=table[antibody]
    return dict(antibody=antibody,full_1_414=True,NT_1_251=nt,CT_251_414=ct,reference='Ling2010 SI FigS1B/D',
        exact_epitope=None,calpain_fragment_visibility=None,band_affinity=None,
        note='Recombinant constructs localize recognition broadly; no numerical affinity or visibility of arbitrary proteolytic fragments assigned')

def source_evidence():return json.loads((_HERE/'evidence_codex.json').read_text(encoding='utf-8'))

def source_training_data():
    out=_prior.source_training_data()
    h=out['TDP_homeostasis']['baseline']['HeLa_reference'];h['read_before_supplement']=h['read'];h['read']='Ling main and SI now read; S1 antibody calibration verified'
    ling=out['Ling_Ayala_RNA_feedback']['Ling2010'];ling['read_before_supplement']=ling['read'];ling['read']='Full main and six-page SI read; S1/S2/S4/S5 visually inspected'
    out['Ling_SI_assay_observer']=source_evidence();return out
