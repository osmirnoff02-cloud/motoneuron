"""[codex] 2026-10-21. Explicit ADAR2 -> GluA2 -> Ca/CAPN/TDP -> STMN2.

One-way biochemical coupling; inherited Ca/SK/Vm feedback is retained.
Unknown human links remain assumptions. No implicit extension of a Ca trace.
"""
from pathlib import Path
from dataclasses import dataclass,asdict,replace
import importlib.util,sys,json
import numpy as np

_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_before_integrated20261021',_HERE.parent/'2026-10-21-cascella-cast-source-update/model_api.py')
_prior=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_prior;_spec.loader.exec_module(_prior)
for _name in dir(_prior):
    if not _name.startswith('_') and _name not in globals(): globals()[_name]=getattr(_prior,_name)


def _source(value):
    if not isinstance(value,str) or not value.strip(): raise ValueError('Explicit provenance required')


@dataclass(frozen=True)
class LocalizationParameters:
    nuclear_fraction: float
    exchange_per_s: float
    provenance: str
    def validate(self):
        _source(self.provenance)
        if not np.isfinite([self.nuclear_fraction,self.exchange_per_s]).all() or not 0<self.nuclear_fraction<1 or self.exchange_per_s<0:
            raise ValueError('Nuclear fraction in (0,1), nonnegative transport rate required')


def healthy_chain_state(*,basal_ca_uM,enzyme_parameters,tdp_parameters,localization_parameters,stmn2_parameters):
    """A conditional basal equilibrium, not measured healthy concentrations.

    Intact TDP is normalized to one. Its synthesis balances basal degradation
    AND basal cleavage. Fragment pools are equilibrium lineage units. Balanced
    initialization avoids interpreting startup drift as an ADAR2 effect.
    """
    ep=enzyme_parameters;tp=tdp_parameters;lp=localization_parameters;sp=stmn2_parameters
    ep.vector();tp.vector();lp.validate()
    if not np.isfinite(basal_ca_uM) or basal_ca_uM<=0 or ep.turnover_per_s<=0 or ep.synthesis_per_s<=0:
        raise ValueError('Positive basal Ca, enzyme turnover and synthesis needed for unique healthy equilibrium')
    c=basal_ca_uM;n=ep.hill_n
    gu=1/(1+(ep.native_ca_half_uM/c)**n);ga=1/(1+(ep.processed_ca_half_uM/c)**n)
    d=ep.turnover_per_s;u=ep.cast_unbinding_per_s
    aut=ep.autolysis_per_s*gu;bu=ep.cast_binding_per_s*gu;ba=ep.cast_binding_per_s*ga
    M=np.array([[-aut-bu-d,0,u,0],[aut,-ba-d,0,u],[bu,0,-u-d,0],[0,ba,0,-u-d]])
    e=np.linalg.solve(M,-np.array([ep.synthesis_per_s,0,0,0]))
    if np.any(e<0): raise ArithmeticError('Invalid enzyme equilibrium')
    a=gu*e[0]+ga*e[1]
    dp,df,dr=np.log(2)/np.array([tp.intact_half_life_s,tp.large_half_life_s,tp.small_half_life_s])
    q1=tp.cleavage1_per_s*a;q2=tp.cleavage2_per_s*a
    frag=q1/(df+q2);small=q2*frag/dr
    syn=dp+q1
    growth=float(normalized_hill(sp.axonal_gain,sp.regrowth_half_relative,sp.regrowth_hill))
    return dict(enzyme=e,activity=float(a),TDP=np.array([1.,frag,small]),TDP_synthesis_per_s=float(syn),
                nuclear_reference=lp.nuclear_fraction,STMN2=np.array([1.,1.,sp.axonal_gain,growth]),
                interpretation='Assumed basal steady state, lineage-normalized; not observed human pools')


def chain_from_calcium(time_s,ca_uM,*,basal_ca_uM,enzyme_parameters,tdp_parameters,
        localization_parameters,stmn2_parameters,ca_provenance,compartment,
        slow_dt_s=.05,calpain_block_fraction=0.,splice_rescue_fraction=0.,
        freeze_tdp_input_to_stmn2=False):
    """Continuous acute cascade on the supplied time domain, no future hold.

    Proteolysis uses the SAME enzyme activity output (CAST is not applied twice).
    Calpain inhibition is an explicit idealized catalytic block, not drug dose.
    All intervention arms start from the same untreated equilibrium.
    Nuclear/cytoplasmic losses are assumed equal. Functional intact nuclear TDP
    is proportional to its amount, with Chan transfer families as uncertainty.
    STMN2 is evaluated on a declared coarser grid; the full upstream trace stays
    intact. A convergence check is required for a selected grid.
    """
    _source(ca_provenance);_source(compartment)
    t=np.asarray(time_s,float);c=np.asarray(ca_uM,float)
    if t.ndim!=1 or len(t)<2 or c.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(c).all() or np.any(np.diff(t)<=0) or np.any(c<0):
        raise ValueError('Matching finite Ca/uM and increasing time/seconds required')
    if not np.isfinite([slow_dt_s,calpain_block_fraction,splice_rescue_fraction]).all() or slow_dt_s<=0 or not 0<=calpain_block_fraction<=1 or not 0<=splice_rescue_fraction<=1:
        raise ValueError('Positive slow step and fractional interventions required')
    initial=healthy_chain_state(basal_ca_uM=basal_ca_uM,enzyme_parameters=enzyme_parameters,tdp_parameters=tdp_parameters,localization_parameters=localization_parameters,stmn2_parameters=stmn2_parameters)
    e=simulate_calpain_pools(t,c,parameters=enzyme_parameters,initial_pools=initial['enzyme'],ca_provenance=ca_provenance,compartment=compartment)
    active=e['effective_activity']*(1-calpain_block_fraction)
    td=simulate_tdp_from_activity(t,active,parameters=tdp_parameters,initial_TDP=initial['TDP'],synthesis_per_s=initial['TDP_synthesis_per_s'],activity_provenance='Same simulated free enzyme activity including CAST, times idealized catalytic availability')
    lp=localization_parameters;f=lp.nuclear_fraction
    loc=partition_intact_tdp(t,td['intact_TDP'],import_per_s=lp.exchange_per_s*f,export_per_s=lp.exchange_per_s*(1-f),synthesis_per_s=initial['TDP_synthesis_per_s'],synthesis_nuclear_fraction=f,initial_nuclear_fraction=f,provenance=lp.provenance)
    slow=np.unique(np.r_[np.arange(t[0],t[-1],slow_dt_s),t[-1]])
    # Explicit sampling/interpolation of intact amounts, never extend the domain.
    n=np.interp(slow,t,loc['nuclear_intact'])/initial['nuclear_reference']
    drive=np.ones_like(n) if freeze_tdp_input_to_stmn2 else n
    st=simulate_stmn2_regrowth(slow,drive,functional_fraction=1.,splice_rescue_fraction=splice_rescue_fraction,stmn2_rna_retention=1.,parameters=stmn2_parameters,initial_state=initial['STMN2'],drive_provenance='Same chain intact nuclear TDP relative to basal; proportional function assumed; no seconds-to-hours extrapolation')
    checks=dict(enzyme_balance=e['pool_balance_max_error'],TDP_lineage_balance=td['lineage_balance_max_error'],nuclear_partition_balance=loc['partition_balance_error'],time_domain_preserved=bool(slow[0]==t[0] and slow[-1]==t[-1]))
    summary=dict(duration_s=float(t[-1]-t[0]),peak_Ca_uM=float(c.max()),Ca_excess_auc_uM_s=float(np.trapezoid(c-basal_ca_uM,t)),activity_auc=float(np.trapezoid(active,t)),intact_TDP_end=float(td['intact_TDP'][-1]),primary_cleavage_cumulative=float(td['formed_large_cumulative'][-1]),nuclear_TDP_relative_end=float(n[-1]),STMN2_mRNA_end=float(st['full_length_mrna'][-1]),STMN2_mRNA_target_end=float(st['full_length_mrna_target'][-1]),axonal_STMN2_proxy_end=float(st['axonal_stmn2_proxy'][-1]),regrowth_support_end=float(st['regrowth_support_index'][-1]))
    return dict(enzyme=e,TDP=td,localization=loc,STMN2=st,initial_state=initial,summary=summary,checks=checks,
                provenance=dict(calcium=ca_provenance,compartment=compartment,enzyme=asdict(enzyme_parameters),TDP=asdict(tdp_parameters),localization=asdict(lp),STMN2=asdict(stmn2_parameters)),
                interventions=dict(calpain_block_fraction=calpain_block_fraction,splice_rescue_fraction=splice_rescue_fraction,freeze_tdp_input_to_stmn2=freeze_tdp_input_to_stmn2),
                limitations=['No TDP/STMN2 feedback to voltage or ADAR2','No aggregate, p35 or NMJ substitution','No clinical time-to-disease prediction','RNA expression is not ADAR2 activity','Ca compartment to whole-cell cleavage is an explicit simplification'])


def prepare_receptor_state(*,adar2_activity,editing_parameters,preconditioning_h,mapping_family,editing_provenance):
    """GluA2 RNA/surface turnover; preconditioning does not simulate injury history."""
    _source(editing_provenance)
    e=editing_trajectory(adar2_activity,editing_parameters,hours=preconditioning_h,family=mapping_family)
    return dict(**e,surface_Q=1-e['surface_R_at_stimulus'],adar2_activity=adar2_activity,mapping_family=mapping_family,preconditioning_h=preconditioning_h,provenance=editing_provenance,
                scope='Receptor preparation only; downstream basal initialization occurs at electrical challenge, not at ADAR2 perturbation')


def run_integrated_chain(profile,*,adar2_activity,editing_parameters,preconditioning_h,mapping_family,
        editing_provenance,coupling_parameters,enzyme_parameters,tdp_parameters,
        localization_parameters,stmn2_parameters,compartment,protocol,calcium_sample_ms=.1,slow_dt_s=.05):
    """Single public entry point. All unknown biochemical choices are explicit.

    Surface composition is frozen during the short electrical challenge. Reject
    durations above 1% of the fastest supplied receptor turnover time.
    """
    receptor=prepare_receptor_state(adar2_activity=adar2_activity,editing_parameters=editing_parameters,preconditioning_h=preconditioning_h,mapping_family=mapping_family,editing_provenance=editing_provenance)
    total_ms=protocol.get('pre_ms',200.)+protocol.get('duration_ms',500. if profile=='Burley' else 1000.)+protocol.get('post_ms',1800.)
    if total_ms>0.01*min(editing_parameters.tau_rna_h,editing_parameters.tau_surface_h)*3600000:
        raise ValueError('Frozen surface approximation invalid: explicitly simulate receptor turnover')
    electrical=run_coupled(profile,surface_Q=receptor['surface_Q'],parameters=coupling_parameters,**protocol)
    result=chain_from_electrical_trace(electrical,basal_ca_uM=coupling_parameters.basal_ca_uM,enzyme_parameters=enzyme_parameters,tdp_parameters=tdp_parameters,localization_parameters=localization_parameters,stmn2_parameters=stmn2_parameters,compartment=compartment,calcium_sample_ms=calcium_sample_ms,slow_dt_s=slow_dt_s)
    result.update(receptor=receptor,electrical=electrical)
    return result


def chain_from_electrical_trace(trace,*,basal_ca_uM,enzyme_parameters,tdp_parameters,
        localization_parameters,stmn2_parameters,compartment,calcium_sample_ms=.1,slow_dt_s=.05,**interventions):
    if compartment not in ('local','bulk'): raise ValueError('Choose local shell or bulk; no unnamed compartment')
    if not np.isfinite(calcium_sample_ms) or calcium_sample_ms<=0: raise ValueError('Positive calcium sampling interval required')
    t=np.asarray(trace['t']);dt=float(np.median(np.diff(t)))
    stride=max(1,int(round(calcium_sample_ms/dt)));ix=np.unique(np.r_[np.arange(0,len(t),stride),len(t)-1]).astype(int)
    r=chain_from_calcium(t[ix]/1000,np.asarray(trace['cs' if compartment=='local' else 'cb'])[ix],basal_ca_uM=basal_ca_uM,enzyme_parameters=enzyme_parameters,tdp_parameters=tdp_parameters,localization_parameters=localization_parameters,stmn2_parameters=stmn2_parameters,ca_provenance='Native NEURON Ca/SK/Vm trace; declared sampling interval '+str(calcium_sample_ms)+' ms',compartment=compartment,slow_dt_s=slow_dt_s,**interventions)
    r['sampling']=dict(source_dt_ms=dt,retained_Ca_dt_ms=stride*dt,STMN2_grid_s=slow_dt_s)
    return r
