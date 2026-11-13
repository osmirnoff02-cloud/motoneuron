"""[codex] 2026-10-23. Independent TDP turnover and assay-specific axon outcomes.

Ryan motivates separating production from loss, not a measured BORC->translation
law. Beccari supplies human MN endpoint data, not calcium/clearance time constants.
Defaults delegate unchanged to the Yamashita-enabled chain.
"""
from pathlib import Path
from dataclasses import dataclass, asdict, replace
import importlib.util, sys, json
import numpy as np

_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_before_library20261023',_HERE.parent/'2026-10-23-yamashita-phospho/model_api_codex.py')
_prior=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_prior;_spec.loader.exec_module(_prior)
for _name in dir(_prior):
    if not _name.startswith('_') and _name not in globals():globals()[_name]=getattr(_prior,_name)


def _source(value):
    if not isinstance(value,str) or not value.strip():raise ValueError('Explicit provenance required')


@dataclass(frozen=True)
class TDPHomeostasisIntervention:
    """Acute multipliers at t[0], with the SAME untreated initial molecular pools.

    These are independent effective controls, not measured BORC knockdown effects.
    Full-length loss affects both U/P equally, but not calpain cleavage. Fragment
    removal is separate. No automatic rebalancing of synthesis after intervention.
    """
    synthesis_multiplier: float
    intact_clearance_multiplier: float
    large_N_clearance_multiplier: float
    small_N_clearance_multiplier: float
    provenance: str

    def validate(self):
        _source(self.provenance)
        v=np.array([self.synthesis_multiplier,self.intact_clearance_multiplier,
                    self.large_N_clearance_multiplier,self.small_N_clearance_multiplier],float)
        if not np.isfinite(v).all() or v[0]<0 or np.any(v[1:]<=0):
            raise ValueError('Finite nonnegative synthesis and positive clearance multipliers required')


def chain_from_calcium(time_s,ca_uM,*,homeostasis_intervention=None,
        phosphorylation_parameters=None,stmn2_functional_fraction=1.,
        stmn2_rna_retention=1.,splice_rescue_fraction=0.,stmn2_drive_provenance=None,**kwargs):
    if homeostasis_intervention is not None:homeostasis_intervention.validate()
    out=_prior.chain_from_calcium(time_s,ca_uM,phosphorylation_parameters=phosphorylation_parameters,
        stmn2_functional_fraction=stmn2_functional_fraction,stmn2_rna_retention=stmn2_rna_retention,
        splice_rescue_fraction=splice_rescue_fraction,stmn2_drive_provenance=stmn2_drive_provenance,**kwargs)
    if homeostasis_intervention is None:return out
    h=homeostasis_intervention;ini=out['initial_state'];tp=kwargs['tdp_parameters'];lp=kwargs['localization_parameters']
    changed=replace(tp,intact_half_life_s=tp.intact_half_life_s/h.intact_clearance_multiplier,
        large_half_life_s=tp.large_half_life_s/h.large_N_clearance_multiplier,
        small_half_life_s=tp.small_half_life_s/h.small_N_clearance_multiplier)
    synthesis=ini['TDP_synthesis_per_s']*h.synthesis_multiplier
    active=out['enzyme']['effective_activity']*(1-out['interventions']['calpain_block_fraction'])
    args=dict(parameters=changed,synthesis_per_s=synthesis,activity_provenance=h.provenance)
    if phosphorylation_parameters is None:
        td=simulate_tdp_from_activity(time_s,active,initial_TDP=ini['TDP'],**args)
    else:
        td=simulate_phospho_tdp_from_activity(time_s,active,initial_TDP=ini['TDP_phospho'],
            phosphorylation_parameters=phosphorylation_parameters,**args)
    out['TDP']=td
    f=lp.nuclear_fraction
    out['localization']=partition_intact_tdp(time_s,td['intact_TDP'],import_per_s=lp.exchange_per_s*f,
        export_per_s=lp.exchange_per_s*(1-f),synthesis_per_s=synthesis,synthesis_nuclear_fraction=f,
        initial_nuclear_fraction=f,provenance=lp.provenance+'; equal per-molecule loss and unchanged partition assumed')
    st=drive_stmn2_from_chain(out,functional_fraction=stmn2_functional_fraction,
        splice_rescue_fraction=splice_rescue_fraction,stmn2_rna_retention=stmn2_rna_retention,
        parameters=kwargs['stmn2_parameters'],drive_provenance=stmn2_drive_provenance or h.provenance)
    out['STMN2']=st
    out['summary'].update(intact_TDP_end=float(td['intact_TDP'][-1]),
        primary_cleavage_cumulative=float(td['formed_large_cumulative'][-1]),
        nuclear_TDP_relative_end=float(out['localization']['nuclear_intact'][-1]/ini['nuclear_reference']))
    if phosphorylation_parameters is not None:out['summary']['phosphorylated_intact_end']=float(td['phosphorylated_intact'][-1])
    for key,channel in [('STMN2_mRNA_end','full_length_mrna'),('STMN2_mRNA_target_end','full_length_mrna_target'),
            ('axonal_STMN2_proxy_end','axonal_stmn2_proxy'),('regrowth_support_end','regrowth_support_index')]:
        out['summary'][key]=float(st[channel][-1])
    out['checks'].update(TDP_lineage_balance=td['lineage_balance_max_error'],
        nuclear_partition_balance=out['localization']['partition_balance_error'])
    out['interventions']['TDP_homeostasis']=asdict(h)
    out['provenance']['TDP_homeostasis']=dict(intervention=asdict(h),effective_TDP_parameters=asdict(changed),
        untreated_synthesis_per_s=ini['TDP_synthesis_per_s'],applied_synthesis_per_s=synthesis)
    out['limitations']+=['Homeostasis changes start at trace onset from unchanged untreated pools; no hidden preconditioning',
        'Ryan dSILAC half-lives assume steady state; SunTag did not detect reduced translation; mechanism remains unresolved',
        'Effective total removal is not a resolved lysosome/proteasome pathway; no BORC dose calibration']
    return out


def apparent_silac_half_life(heavy_light_ratio,*,label_duration_s,steady_state_assumed,provenance):
    """Ryan/Hasan single-timepoint observation formula; not a decay-curve fit."""
    _source(provenance)
    r=np.asarray(heavy_light_ratio,float)
    if steady_state_assumed is not True:raise ValueError('The single-timepoint formula requires explicit steady-state assumption')
    if not np.isfinite(r).all() or np.any(r<=0) or not np.isfinite(label_duration_s) or label_duration_s<=0:
        raise ValueError('Positive finite H/L ratio and labeling duration required')
    return dict(apparent_half_life_s=label_duration_s*np.log(2)/np.log1p(r),
        steady_state_assumed=True,provenance=provenance,human_motoneuron_fit=False)


def steady_state_synthesis_ratio(abundance_ratio,half_life_ratio,*,steady_state_assumed,provenance):
    """Conditional S'/S=(P'/P)/(HL'/HL), NOT evidence that translation changed."""
    _source(provenance)
    if steady_state_assumed is not True:raise ValueError('Explicit stationary first-order balance assumption required')
    if not np.isfinite([abundance_ratio,half_life_ratio]).all() or abundance_ratio<0 or half_life_ratio<=0:
        raise ValueError('Nonnegative abundance and positive half-life ratios required')
    return dict(conditional_synthesis_ratio=abundance_ratio/half_life_ratio,
        interpretation='Mass-balance hypothesis; translation not measured by this calculation',provenance=provenance)


def beccari_reference():
    return json.loads((_HERE/'beccari-endpoints_codex.json').read_text(encoding='utf-8'))


def beccari_regrowth_endpoint(stmn2_effective_fraction,*,shape_exponent,provenance,
        assay='Beccari_Fig7_day35_fluid_shear_axotomy'):
    """Source-specific endpoint envelope anchored to WT/KO, not a kinetic fit.

    The input is an explicitly ASSUMED effective STMN2 fraction maintained across
    the assay. Its mapping to a model axonal proxy is uncalibrated. Interpolation
    between complete knockout and WT is unknown (sensitivity exponent required).
    Lengths concern measured positive axons; do not multiply by the 0-24h outcome
    fraction to claim a population total. No extrapolation beyond 24/48h or WT.
    """
    _source(provenance)
    if assay!='Beccari_Fig7_day35_fluid_shear_axotomy':raise ValueError('Unsupported assay; Chan/Baughn endpoints differ')
    x=np.asarray(stmn2_effective_fraction,float)
    if not np.isfinite(x).all() or np.any((x<0)|(x>1)) or not np.isfinite(shape_exponent) or shape_exponent<=0:
        raise ValueError('Fraction in [0,1] and positive finite interpolation exponent required')
    a=x**shape_exponent;ref=beccari_reference();g=ref['groups']
    out={}
    for key in ('regenerated_0_24h_fraction','stalled_0_24h_fraction','fragmented_0_24h_fraction'):
        lo=np.mean(g['KO'][key]);hi=np.mean(g['WT'][key]);out[key]=lo+(hi-lo)*a
    for t in ('24','48'):
        key='length_'+t+'h_um';out['mean_measured_axon_'+key]=g['KO'][key]['mean']+(g['WT'][key]['mean']-g['KO'][key]['mean'])*a
    out.update(assay=assay,shape_exponent=shape_exponent,provenance=provenance,
        fitted_endpoints=('WT','KO'),held_out_rescue_groups=('WT_rescue','TBN_rescue'),
        independently_validated=False,human_calcium_fit=False,
        limitation='Two differentiations; lengths lack replicate labels. Partial STMN2 shape and history not measured.')
    return out


def source_training_data():
    result=_prior.source_training_data()
    result['Library20261023_source_scope']=json.loads((_HERE/'source-evidence_codex.json').read_text(encoding='utf-8'))
    result['Beccari2025_regrowth_endpoints']=beccari_reference()
    result['Hasan2023_cortical_turnover']=json.loads((_HERE/'hasan-turnover-reference_codex.json').read_text(encoding='utf-8'))
    return result


def hasan_cortical_label_chase(time_s,*,gene,cohort='WT_global'):
    """Reference old-label decay from published apparent protein half-lives.

    Separate assay component, NOT an override of intact TDP or axonal STMN2 rates.
    First-order label turnover/constant abundance is an explicit source assumption.
    The two WT cohorts are not pooled and their spread is not a confidence interval.
    """
    ref=json.loads((_HERE/'hasan-turnover-reference_codex.json').read_text(encoding='utf-8'))
    if cohort not in ref['cohorts'] or gene not in ref['cohorts'][cohort]:raise ValueError('Choose a recorded cohort and STMN2 or TARDBP')
    t=np.asarray(time_s,float)
    if t.ndim!=1 or len(t)<1 or not np.isfinite(t).all() or np.any(t<0) or np.any(np.diff(t)<0):raise ValueError('Nonnegative increasing time since label switch required')
    hl=ref['cohorts'][cohort][gene]['half_life_days']*86400
    return dict(time_s=t,old_label_fraction=np.exp(-np.log(2)*t/hl),apparent_half_life_s=hl,
        source_cell=ref['cohorts'][cohort][gene],cell_type=ref['cell_type'],
        interpretation='First-order source reference, not raw timepoints or fitted motor-neuron kinetics',
        source_assumptions=ref['limitations'],updates_neuron_parameters=False)


def chain_from_electrical_trace(trace,*,basal_ca_uM,enzyme_parameters,tdp_parameters,
        localization_parameters,stmn2_parameters,compartment,calcium_sample_ms=.1,slow_dt_s=.05,**interventions):
    if compartment not in ('local','bulk'):raise ValueError('Choose local shell or bulk')
    if not np.isfinite(calcium_sample_ms) or calcium_sample_ms<=0:raise ValueError('Positive sampling interval required')
    t=np.asarray(trace['t']);dt=float(np.median(np.diff(t)))
    stride=max(1,int(round(calcium_sample_ms/dt)));ix=np.unique(np.r_[np.arange(0,len(t),stride),len(t)-1]).astype(int)
    r=chain_from_calcium(t[ix]/1000,np.asarray(trace['cs' if compartment=='local' else 'cb'])[ix],
        basal_ca_uM=basal_ca_uM,enzyme_parameters=enzyme_parameters,tdp_parameters=tdp_parameters,
        localization_parameters=localization_parameters,stmn2_parameters=stmn2_parameters,compartment=compartment,
        ca_provenance='Native NEURON Ca/SK/Vm; explicitly sampled',slow_dt_s=slow_dt_s,**interventions)
    r['sampling']=dict(source_dt_ms=dt,retained_Ca_dt_ms=stride*dt,STMN2_grid_s=slow_dt_s)
    return r


def run_integrated_chain(profile,*,adar2_activity,editing_parameters,preconditioning_h,mapping_family,
        editing_provenance,coupling_parameters,enzyme_parameters,tdp_parameters,localization_parameters,
        stmn2_parameters,compartment,protocol,calcium_sample_ms=.1,slow_dt_s=.05,**interventions):
    receptor=prepare_receptor_state(adar2_activity=adar2_activity,editing_parameters=editing_parameters,
        preconditioning_h=preconditioning_h,mapping_family=mapping_family,editing_provenance=editing_provenance)
    duration=protocol.get('pre_ms',200.)+protocol.get('duration_ms',500. if profile=='Burley' else 1000.)+protocol.get('post_ms',1800.)
    if duration>.01*min(editing_parameters.tau_rna_h,editing_parameters.tau_surface_h)*3600000:raise ValueError('Frozen receptor approximation invalid')
    electrical=run_coupled(profile,surface_Q=receptor['surface_Q'],parameters=coupling_parameters,**protocol)
    out=chain_from_electrical_trace(electrical,basal_ca_uM=coupling_parameters.basal_ca_uM,
        enzyme_parameters=enzyme_parameters,tdp_parameters=tdp_parameters,localization_parameters=localization_parameters,
        stmn2_parameters=stmn2_parameters,compartment=compartment,calcium_sample_ms=calcium_sample_ms,
        slow_dt_s=slow_dt_s,**interventions)
    out.update(receptor=receptor,electrical=electrical)
    return out

