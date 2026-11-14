"""[codex] 2026-10-26. Conditional adaptive-recycling 308-state model; prior sources retained.

Opt-in continuous two-compartment voltage and bulk-Ca model. Inherited
somatic biochemistry plus distinct local CAPN/substrate pools; no duplicate Ca.
All spatial/traffic parameters below are hypotheses, not a human fit.
"""
from pathlib import Path
from dataclasses import dataclass, asdict, replace as dc_replace
import sys, json, hashlib, importlib.util
import numpy as np
from scipy.integrate import solve_ivp
B=Path(__file__).resolve().parent.parent/'2026-10-26-nak-energy-neuron'
sys.path.insert(0,str(B.parent/'neuron-model'))
from model_contracts_codex import load_module, state_schema, check_checkpoint_schema, solver_atol
sys.path.insert(0,str(B))
sys.path.insert(0,str(B.parent/'2026-10-26-connected-spatial-neuron'))
from ion_parameters_codex import IonParameters, ION_NAMES, prepare_ions, ion_extension
from mito_parameters_codex import MitoParameters, MITO_NAMES, mito_extension
sys.path.insert(0,str(B.parent/'2026-10-26-nak-source-calibration'))
sys.path.insert(0,str(B.parent/'2026-10-26-proteasome-energy'))
sys.path.insert(0,str(B.parent/'2026-10-26-proteasome-ampa'))
sys.path.insert(0,str(B.parent/'2026-10-26-geometry-mito-nmj'))
sys.path.insert(0,str(B.parent/'2026-10-26-prakash-activity-atp'))
sys.path.insert(0,str(B.parent/'2026-10-26-takeuchi-recycling-atp'))
from adaptive_recycling_codex import AdaptiveRecyclingParameters,ADAPTIVE_NAMES,adaptive_extension,phluorin_signal,capacity
sys.path.insert(0,str(B.parent/'2026-10-28-parsons-grabe'))
from cholinergic_cargo_codex import CARGO_NAMES,cargo_extension,cargo_jump,release_observation
sys.path.insert(0,str(B.parent/'2026-10-28-vesicle-supplements'))
from uncoating_codex import UNCOATING_NAMES,uncoating_extension
sys.path.insert(0,str(B.parent/'2026-10-28-donor-energy-integration'))
from creatine_codex import CREATINE_NAMES,creatine_extension,equilibrium_PCr
sys.path.insert(0,str(B.parent/'2026-10-29-rna-calcium-modules'))
from rna_calcium_codex import RNA_CALCIUM_NAMES,ligand_schedule,gene_ion_extension,append_surface
sys.path.insert(0,str(B.parent/'2026-10-30-nucleus-ribosome'))
from nucleus_ribosome_codex import NUCLEUS_NAMES,initial_nucleus,nucleus_schedule,replace_biosynthesis,append_nucleus,nucleus_checks
sys.path.insert(0,str(B.parent/'2026-10-30-nuclear-targets-cycle'))
from nuclear_cycle_codex import EXTRA_NAMES as NUCLEAR_CYCLE_NAMES,initial_cycle,initial_kcnq,kcnq_vector,cycle_schedule,replace_cycle_targets,kcnq_current_extension,append_cycle,cycle_checks,kcnq_ratios
sys.path.insert(0,str(B.parent/'2026-10-30-ribosome-biogenesis'))
from biogenesis_codex import NAMES as BIOGENESIS_NAMES,prepare as bio_prepare,initial as bio_initial,controls as bio_schedule,replace_targets as bio_replace,append_biogenesis,checks as bio_checks
sys.path.insert(0,str(B.parent/'2026-10-30-shared-translation'))
from shared_translation_codex import NAMES as SHARED_NAMES,setup as shared_setup,controls as shared_schedule,replace_shared,append_shared,checks as shared_checks
sys.path.insert(0,str(B.parent/'2026-10-30-translation-substrates'))
from substrates_codex import NAMES as SUBSTRATE_NAMES,initial as substrate_initial,controls as substrate_schedule,gate as substrate_gate,append_substrates,checks as substrate_checks
sys.path.insert(0,str(B.parent/'2026-10-30-nuclear-recycling'))
from recycling_codex import NAMES as NUC_REC_NAMES,initial as nrec_initial,controls as nrec_schedule,modulate as nrec_modulate,append_recycling,checks as nrec_checks,removal_sources,intact_TDP_loss
sys.path.insert(0,str(B.parent/'2026-10-30-isr-selective'))
import selective_isr_codex as isr_module
sys.path.insert(0,str(B.parent/'2026-10-30-fusco-supplement'))
import maturation_codex as maturation_module
sys.path.insert(0,str(B.parent/"2026-10-31-local-ribosomes"))
import local_ribosomes_codex as local_ribo
sys.path.insert(0,str(B.parent/"2026-11-02-local-translation"))
import local_translation_codex as local_translation
sys.path.insert(0,str(B.parent/"2026-11-02-local-mrna"))
import local_mrna_codex as local_mrna
sys.path.insert(0,str(B.parent/"2026-11-02-gap43-regrowth"))
import gap43_regrowth_codex as gap43_regrowth
sys.path.insert(0,str(B.parent/"2026-11-02-local-rqc"))
import local_rqc_codex as local_rqc
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"2026-11-02-ribosome-maintenance"))
import maintenance_codex as maintenance
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"2026-11-02-ribosome-renewal"))
import renewal_codex as renewal
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"2026-11-02-ribosome-surplus"))
import surplus_codex as surplus
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"2026-11-03-precursor-quality"))
import precursor_quality_codex as precursor_quality
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"2026-11-03-rna-salvage"))
import rna_salvage_codex as rna_salvage
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'2026-11-03-translation-coupling'))
import coupled_translation_codex as coupled_translation
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'2026-11-03-ion-homeostasis'))
import ion_homeostasis_codex as ion_homeostasis
from recycling_energy_codex import RecyclingParameters,TerminalPathwayParameters,RECYCLING_NAMES,recycling_extension,recycling_jump
from activity_energy_codex import ActivityEnergyParameters,ACTIVITY_NAMES,activity_extension,activity_jump
from source_observers_codex import ProteasomeEntryParameters
from energy_terminal_codex import TerminalEnergyParameters,TERMINAL_NAMES,NMJParameters,nmj_vector,terminal_extension,terminal_jump
from proteasome_work_codex import ProteasomeWorkParameters,work_costs,work_budget,work_loss
from proteostasis_parameters_codex import ProteostasisParameters, PROTEOSTASIS_NAMES, proteostasis_fluxes, proteostasis_budget, split_costs
from extension_parameters_codex import RecoveryParameters, BudgetParameters, BUDGET_NAMES, recovery_extension, budget_extension, protein_costs
sys.path.insert(0, str(B.parent/'2026-10-25-pmca-geometry-model'))
from surface_pmca_codex import PMCASurfaceParameters, map_surface_pmca, IONS_PER_UM3_PER_UM
import long_pmca_codex as core
from pmca_memory_codex import PMCAMemoryParameters, memory_equilibrium, memory_rates, pmca2_flux
from energy_parameters_codex import EnergyParameters, energy_fluxes, validate_schedule
from connected_parameters_codex import ConnectedParameters, local, gates, derived, membrane_rhs
compiled_memory = core.rhs_memory  # Its 133-state biochemical RHS is already JIT compiled.

from connected_parameters_codex import spatial as spatial_source
EXTRA_NAMES=spatial_source.EXTRA_NAMES
passive_exchange_rate=spatial_source.passive_exchange_rate
traffic_rates=spatial_source.traffic_rates
sat_norm=spatial_source.sat_norm

from numba import njit
from compiled_kernels_codex import memory_jit,pmca_flux_jit,rates_jit,sat_jit,traffic_jit
from connected_parameters_codex import injury_rhs

@njit
def connected_rhs_kernel(z,row,target,drive,srow,injection,ctx):
    p,kv,sv,av,hv,nv,ag0,iv,bn,ep,rv,qv,base,ev,en,pv,unit_ratio,ppv,punit,delta_s,cost0s,delta_p,cost0p,pmp0,ncp0,tf,trev,vs,vp,sgamma,skappa,pgamma,pkappa,ca_diffusion,aden_diffusion,trvec,profile_p,profile_c,geomvec,couplevec,fbvec,c0,lev,liv,lkv,lsv,lpv,lsyn=ctx
    out=np.zeros(199);localpv=pv.copy();localpv[8]=unit_ratio*max(z[139],0)
    # Original biochemical AMPA flux is disabled: membrane current below owns it once.
    norow=row.copy();norow[2]=0.
    damage=min(1.,max(0.,1.-z[180]));support=1-fbvec[1]*damage
    out[:134]=memory_jit(z[:134],p,kv,sv,av,hv,nv,ag0,iv,bn,norow,ep,target,rv,qv,base[83],ev,z[185],en,drive,localpv)
    cs=max(c0+z[0],0);cp=max(c0+z[134],0)
    fs=sat_jit(z[120],en[4],en[0]) if en[12] else 1.
    fp=sat_jit(z[136],en[4],en[0]) if en[12] else 1.
    # Transport ATP sensitivity is independent of pump_feedback switch.
    rates,af,ar=traffic_jit(z[139:143],trvec,tf,trev,srow[3]*support,srow[4]*support,sat_jit(z[120],en[4],en[0]),sat_jit(z[136],en[4],en[0]))
    out[139:143]=rates;out[159:161]=[af,ar]
    costs=af*trvec[2]/(IONS_PER_UM3_PER_UM*vs)
    costp=ar*trvec[2]/(IONS_PER_UM3_PER_UM*vp)
    out[154:156]=[costs,costp]
    out[0]+=delta_s/(1+skappa);out[23]-=delta_s/(1+skappa);out[132]+=delta_s
    oxextra=(delta_s+cost0s)*(1-en[8])*drive[0]*sat_jit(z[121],en[6],en[1])
    glyextra=(delta_s+cost0s)*en[8]*drive[1]*sat_jit(z[121],en[7],en[1])
    out[120]+=oxextra+glyextra-costs;out[121]-=oxextra+glyextra-costs
    out[123]+=oxextra;out[124]+=glyextra
    pvec=ppv.copy();pvec[8]=punit*max(z[141],0)
    pm=pmca_flux_jit(cp,z[135],c0,pgamma,en[9],fp,pvec)
    nc=(1-en[9])*pgamma*cp*fp
    _,_,_,other,ox,gly,ak=energy_fluxes(z[136],z[137],z[138],cp,0.,pgamma,c0,0.,en,drive)
    ox+=(delta_p+cost0p)*(1-en[8])*drive[0]*sat_jit(z[137],en[6],en[1])
    gly+=(delta_p+cost0p)*en[8]*drive[1]*sat_jit(z[137],en[7],en[1])
    cost=other+pm+nc*en[10]+costp;prod=ox+gly
    out[136:139]=[prod-cost+ak,cost-prod-2*ak,ak]
    out[148:154]=[ox,gly,other,pm,nc*en[10],ak]
    on,off=rates_jit(cp,pvec);out[135]=on*(1-z[135])-off*z[135]
    out[144:147]=[pm,nc,pmp0+ncp0]
    cv=couplevec.copy();cv[6]=srow[2]
    dv,ca_current=membrane_rhs(z,profile_p,profile_c,geomvec,cv,fbvec,row[2],row[23],z[106],ep[3:8],ep,injection,c0)
    # SOCE was already included in the somatic Ca ledger by the ER source.
    # Return the SAME charge to voltage; do not add its Ca twice.
    dv[0]+=1000*out[119]*vs/5182.1348283/(geomvec[0]/100.)
    out[185:199]=dv
    soma_input=ca_current[0]*5182.1348283/vs
    proc_input=ca_current[1]*5182.1348283/vp
    out[0]+=soma_input/(1+skappa);out[22]+=soma_input/(1+skappa);out[143]=proc_input
    out[162:185]=injury_rhs(z[162:185],max(c0+z[134],0.),row[3],lev,liv,lkv,lsv,lpv,lsyn)
    f=ca_diffusion*(cs-cp);out[147]=f
    out[0]-=f/(vs*(1+skappa))
    out[134]=(proc_input+pmp0+ncp0-pm-nc+f/vp)/(1+pkappa)
    out[161]=z[134]
    fa=aden_diffusion*(z[120:123]-z[136:139])
    out[120:123]-=fa/vs;out[136:139]+=fa/vp;out[156:159]=fa
    return out


# [codex]2026-11-03. Optional per-gene RP inventory; shared resources stay upstream.
sys.path.insert(0,str(B.parent/'2026-11-03-rp-neuron-coupling'))
import rp_coupling_codex as individual_rp

# [codex]2026-11-03. Conservative optional nuclear transport.
sys.path.insert(0,str(B.parent/'2026-11-03-nuclear-transport'))
nct_module=load_module(B.parent/'2026-11-03-nuclear-transport/transport_codex.py','_detailed_nct48')

rp_production=load_module(B.parent/'2026-11-03-rp-production/production_codex.py','_stable_RP_production49')

# [codex] 2026-11-03. RNA/RP/pre-ribosome transport extends the shared Ran cycle.
nrt_module=load_module(B.parent/'2026-11-03-rna-ribosome-transport/rna_transport_codex.py','_rna_ribosome_transport50')

# [codex]2026-11-03. Gene-specific RP-mRNA source/decay in existing transport states.
rmk_module=load_module(B.parent/'2026-11-03-rp-mrna-kinetics/kinetics_codex.py','_RP_mRNA_kinetics51')

def run_adaptive_recycling_neuron(segments, *, adaptive_parameters=None, recycling_parameters=None,pathway_parameters=None,glycolysis_support=1.,oxidative_support=1., activity_parameters=None,entry_parameters=None, terminal_parameters=None,nmj_parameters=None,terminal_supply_support=1.,respiratory_reserve=2.,association_support=1.,initiation_support=1., work_parameters=None, processing_efficiency=1., ATPase_support=1., proteostasis_parameters=None, proteasome_support=1., folding_load=1., recovery_parameters=None, budget_parameters=None, alpha3_support=1., respiratory_support=1., mito_parameters=None, uptake_support=1., ion_parameters=None, pump_support=1., geometry, traffic_parameters, pump_kinetics,
        profile,profile_p,profile_c,coupling,injury_parameters,connected_parameters=None,electrical_schedule=None,window_s=None,
        spatial_schedule=None, memory_parameters, energy_parameters, checkpoint=None,
        differential_KCNQ_activity_multiplier=None,etc_backpressure_exponent=None,
        output_step_s=300., max_step_s=300., rtol=1e-8, atol=1e-10,
        cargo_parameters=None, filling_support=1., uncoating_parameters=None, uncoating_support=1.,
        creatine_parameters=None,creatine_support=1.,creatine_activity_multiplier=(1.,1.,1.), rna_calcium_parameters=None,rna_ligand_schedule=None,
        nucleus_parameters=None,nuclear_schedule=None, cycle_parameters=None,ribosome_schedule=None,kcnq_parameters=None,kcnq_coupling_parameters=None,biogenesis_parameters=None,biogenesis_schedule=None,shared_parameters=None,shared_translation_schedule=None,substrate_parameters=None,substrate_schedule_input=None,nuclear_recycling_parameters=None,nuclear_recycling_schedule=None,isr_parameters=None,isr_schedule=None,maturation_schedule=None,local_ribosome_parameters=None,local_ribosome_schedule=None,local_translation_parameters=None,local_translation_schedule=None,local_mrna_parameters=None,local_mrna_schedule=None,gap43_regrowth_parameters=None,gap43_regrowth_schedule=None,local_rqc_parameters=None,local_rqc_schedule=None,ribosome_maintenance_parameters=None,ribosome_maintenance_schedule=None,ribosome_renewal_parameters=None,ribosome_renewal_schedule=None,ribosome_surplus_parameters=None,ribosome_surplus_schedule=None,precursor_quality_parameters=None,precursor_quality_schedule=None,rna_salvage_parameters=None,rna_salvage_schedule=None, **kw):
    """Spatial schedule columns: start,end,AMPA fraction to process,forward,reverse.

    Continuous local voltage determines AMPA/VGCC Ca current; shared bulk Ca
    drives SK and local CAPN. Somatic ER SOCE returns its charge to voltage.
    pump_kinetics is PMCASurfaceParameters with density_per_um2=1 (unit marker);
    ACTUAL densities are calculated from traffic counts / membrane areas.
    Basal Ca influx and ADP-controlled supply are constructed ONCE at origin
    for matched rest, then held fixed through all stimulation/traffic changes.
    """
    recp=recycling_parameters or RecyclingParameters();recv=recp.vector()
    prior_controls=(checkpoint or {}).get('diagnostic_interventions',{})
    if differential_KCNQ_activity_multiplier is None:
        differential_KCNQ_activity_multiplier=prior_controls.get('differential_KCNQ_activity_multiplier',1.)
    if etc_backpressure_exponent is None:
        etc_backpressure_exponent=prior_controls.get('etc_backpressure_exponent',2.)
    if not np.isfinite([differential_KCNQ_activity_multiplier,etc_backpressure_exponent]).all() or not 0<=differential_KCNQ_activity_multiplier<=2 or not 1<=etc_backpressure_exponent<=8:
        raise ValueError('Diagnostic KCNQ multiplier [0,2] and ETC backpressure exponent [1,8] required')
    if cycle_parameters is None and differential_KCNQ_activity_multiplier!=1:
        raise ValueError('Differential KCNQ intervention requires the nuclear/cycle branch')
    diagnostic_controls=dict(differential_KCNQ_activity_multiplier=float(differential_KCNQ_activity_multiplier),
        etc_backpressure_exponent=float(etc_backpressure_exponent))
    adp=adaptive_parameters or AdaptiveRecyclingParameters();adv=adp.vector()
    if adv[0] and not recv[0]:raise ValueError('Staged recycling requires RecyclingParameters(enabled=True)')
    pathp=pathway_parameters or TerminalPathwayParameters();pathv=pathp.vector()
    if not np.isfinite([glycolysis_support,oxidative_support]).all() or min(glycolysis_support,oxidative_support)<0 or max(glycolysis_support,oxidative_support)>1:raise ValueError('Pathway support [0,1]')
    ap=activity_parameters or ActivityEnergyParameters();activityv=ap.vector()
    original_initiation_support=initiation_support
    if entry_parameters is not None:initiation_support*=entry_parameters.probability()
    tp=terminal_parameters or TerminalEnergyParameters();tv=tp.vector()
    np_=nmj_parameters or NMJParameters();npv=nmj_vector(np_)
    cv=cargo_parameters.vector() if cargo_parameters is not None else None
    uv=uncoating_parameters.vector() if uncoating_parameters is not None else None
    ckv=creatine_parameters.vector() if creatine_parameters is not None else None
    # External kinetic intervention, like pump_support: it may change at a
    # checkpoint without altering pools, equilibrium, baseline or ledgers.
    # Both forward and reverse rates scale together; this is not ATP supply.
    ck_activity=np.asarray(creatine_activity_multiplier,dtype=float)
    if ck_activity.shape!=(3,) or not np.isfinite(ck_activity).all() or np.any(ck_activity<0) or np.any(ck_activity>1000):
        raise ValueError('Three finite CK activity multipliers in [0,1000] required')
    if ckv is None and np.any(ck_activity!=1):raise ValueError('CK activity intervention requires explicit creatine pools')
    if ckv is not None and np.any(ck_activity!=1):
        ckv=ckv.copy();ckv[5:8]*=ck_activity
    rnav=rna_calcium_parameters.vector() if rna_calcium_parameters is not None else None
    nucv=nucleus_parameters.vector() if nucleus_parameters is not None else None
    cyc=cycle_parameters.vector() if cycle_parameters is not None else None
    biov=biogenesis_parameters.vector() if biogenesis_parameters is not None else None
    sub=substrate_parameters.vector() if substrate_parameters is not None else None
    nrv=nuclear_recycling_parameters.vector() if nuclear_recycling_parameters is not None else None
    isrv=isr_parameters.vector() if isr_parameters is not None else None
    lrv=local_ribosome_parameters.vector() if local_ribosome_parameters is not None else None
    ltv=local_translation_parameters.vector() if local_translation_parameters is not None else None
    lmrv=local_mrna_parameters.vector() if local_mrna_parameters is not None else None
    if lmrv is not None and ltv is None:raise ValueError("mRNA requires local translation branch")
    if lmrv is None and local_mrna_schedule is not None:raise ValueError("mRNA controls require parameters")
    grv=gap43_regrowth_parameters.vector() if gap43_regrowth_parameters is not None else None
    rqv=local_rqc_parameters.vector() if local_rqc_parameters is not None else None
    rsv=rna_salvage_parameters.vector() if rna_salvage_parameters is not None else None
    detailed_parameters=kw.pop('coupled_translation_parameters',None)
    basal_translation_capacity_scale=kw.pop('basal_translation_capacity_scale',1.)
    if not np.isfinite(basal_translation_capacity_scale) or not 0<basal_translation_capacity_scale<=2:
        raise ValueError('Conditional basal translation capacity scale must be in (0,2]')
    conditional_RNA_export_initialization=kw.pop('conditional_RNA_export_initialization',False)
    if not isinstance(conditional_RNA_export_initialization,bool):raise ValueError('RNA initialization switch must be boolean')
    gene_specific_RP_translation_reference=kw.pop('gene_specific_RP_translation_reference',False)
    if not isinstance(gene_specific_RP_translation_reference,bool):raise ValueError('Gene-specific RNA reference switch must be boolean')
    diagnostic_rates=kw.pop('diagnostic_rates',False)
    if not isinstance(diagnostic_rates,bool):raise ValueError('diagnostic_rates must be boolean')
    center_solver_coordinates=kw.pop('center_solver_coordinates',False)
    segment_output_steps=kw.pop('segment_output_steps_s',None)
    segment_max_steps=kw.pop('segment_max_steps_s',None)
    retain_dense_solver_output=kw.pop('retain_dense_solver_output',True)
    if not isinstance(retain_dense_solver_output,bool):raise ValueError('retain_dense_solver_output must be boolean')
    solver_observer=kw.pop('solver_observer',None)
    if solver_observer is not None and not callable(solver_observer):raise ValueError('solver_observer must be callable')
    solver_method=kw.pop('solver_method','LSODA')
    if solver_method not in ('LSODA','BDF'):raise ValueError('Supported solver_method: LSODA or BDF')
    solver_sparse_jacobian=kw.pop('solver_sparse_jacobian',False)
    if not isinstance(solver_sparse_jacobian,bool) or (solver_sparse_jacobian and solver_method!='BDF'):
        raise ValueError('solver_sparse_jacobian requires a boolean and BDF')
    sparse_stats=dict(calls=0,rhs_evaluations=0,last_nnz=0)
    if not isinstance(center_solver_coordinates,bool):raise ValueError('center_solver_coordinates must be boolean')
    ctv=detailed_parameters.vector() if detailed_parameters is not None else None
    individual_rp_parameters=kw.pop('individual_rp_parameters',None)
    irates=individual_rp_parameters.rates() if individual_rp_parameters is not None else None
    if irates is not None and ctv is None:raise ValueError('Individual RP requires coupled translation topology')
    rpp_parameters=kw.pop('rp_production_parameters',None)
    rpp_supplied=kw.pop('rp_production_schedule',None)
    rppv=rpp_parameters.vector() if rpp_parameters is not None else None
    if rppv is not None and irates is None:raise ValueError('RP production requires individual RP inventories')
    if rppv is None and rpp_supplied is not None:raise ValueError('RP schedule requires parameters')
    nct_parameters=kw.pop('nuclear_transport_parameters',None)
    nct_supplied=kw.pop('nuclear_transport_schedule',None)
    nctv=nct_parameters.vector() if nct_parameters is not None else None
    if nctv is not None and ctv is None:raise ValueError('Detailed NCT requires coupled translation')
    if nctv is None and nct_supplied is not None:raise ValueError('NCT schedule requires parameters')
    nrt_parameters=kw.pop('rna_export_parameters',None)
    nrt_supplied=kw.pop('rna_export_schedule',None)
    nrtv=nrt_parameters.vector() if nrt_parameters is not None else None
    if nrtv is not None and (nctv is None or precursor_quality_parameters is None):raise ValueError('RNA export requires detailed NCT and precursor quality')
    if nrtv is not None and irates is not None and rppv is None:raise ValueError('Individual RP nuclear transport requires individual production')
    if nrtv is None and nrt_supplied is not None:raise ValueError('RNA export schedule requires parameters')
    rmk_parameters=kw.pop('rp_mrna_kinetics_parameters',None)
    rmk_supplied=kw.pop('rp_mrna_kinetics_schedule',None)
    rmkv=rmk_parameters.vector() if rmk_parameters is not None else None
    if rmkv is not None and (nrtv is None or rppv is None):raise ValueError('RP-mRNA kinetics require RNA transport and individual RP production')
    if rmkv is None and rmk_supplied is not None:raise ValueError('RP-mRNA schedule requires kinetic parameters')
    background_parameters=kw.pop('ion_background_parameters',None)
    if background_parameters is not None:background_parameters.validate()
    pump_leak_fraction=kw.pop('pump_leak_closure_fraction',None)
    if pump_leak_fraction is not None and (background_parameters is None or background_parameters.passive_fraction!=1.):
        raise ValueError('Joint pump/leak closure requires fully passive Na/K background')
    pump_response_parameters=kw.pop('pump_response_parameters',None)
    pump_response_module=None;pump_response_vector=None
    if pump_response_parameters is not None:
        pump_response_parameters.validate()
        if not (ion_parameters or IonParameters()).enabled or alpha3_support!=1.:
            raise ValueError('Aggregate pump-response experiment requires active ions and alpha3_support=1')
        if rna_calcium_parameters is not None and rna_calcium_parameters.fxyd7_strength!=0:
            raise ValueError('Aggregate pump-response experiment cannot resolve an active FXYD7 intervention')
        pump_response_module=load_module(B.parent/'2026-11-13-pump-response-test/pump_response_codex.py','_stable_pump_response13')
    water_parameters=kw.pop('water_volume_parameters',None)
    absolute_parameters=kw.pop('absolute_kcnq_parameters',None)
    water_module=None;water_scale=None;absolute_module=None;absolute_delta0=None
    if water_parameters is not None:
        water_parameters.validate()
        water_module=load_module(B.parent/'2026-11-12-ion-water-kcnq/compartment_volumes_codex.py','_stable_mobile_water12')
    if absolute_parameters is not None:
        absolute_parameters.validate()
        if pump_leak_fraction is None or cyc is None:
            raise ValueError('Absolute KCNQ requires joint passive/pump closure and explicit KCNQ transport/gate')
        absolute_module=load_module(B.parent/'2026-11-12-ion-water-kcnq/absolute_current_coupling_codex.py','_stable_absolute_coupling12')
    chloride_parameters=kw.pop('passive_chloride_parameters',None)
    chloride_module=None
    if chloride_parameters is not None:
        chloride_parameters.validate()
        if pump_leak_fraction is None:raise ValueError('Passive Cl requires explicit joint pump/passive-leak closure')
        chloride_module=load_module(B.parent/'2026-11-11-kcnq-passive/passive_chloride_codex.py','_stable_passive_chloride11')
    # [claude] 2026-11-13: optional acid-extrusion closure for PMCA counter-protons (plan N05.1.4); default None leaves the model unchanged.
    acid_parameters=kw.pop('acid_extrusion_parameters',None)
    acid_module=None;acid_ref=None;acid_offset=None
    if acid_parameters is not None:
        acid_parameters.validate()
        # [codex] 2026-11-14: host PMCA has no voltage current; do not vary H/Ca independently.
        # Zero is reserved for the engineering zero-coupling control.
        if acid_parameters.H_per_Ca not in (0.,2.):
            raise ValueError('Host electroneutral PMCA requires H_per_Ca=2; 0 is only a zero-coupling control')
        if not (ion_parameters or IonParameters()).enabled:raise ValueError('Acid extrusion requires active Na/K ions')
        acid_module=load_module(B.parent/'2026-11-13-acid-extrusion/acid_extrusion_claude.py','_stable_acid_extrusion13_claude')
    if ctv is not None and rsv is None:raise ValueError('Coupled codon branch requires full RNA salvage topology')
    if rsv is not None and precursor_quality_parameters is None:raise ValueError("RNA salvage requires explicit precursor quality")
    if rsv is None and rna_salvage_schedule is not None:raise ValueError("RNA controls require parameters")
    pqv=precursor_quality_parameters.vector() if precursor_quality_parameters is not None else None
    if pqv is not None and ribosome_surplus_parameters is None:raise ValueError("Precursor quality requires surplus recipe/feedback")
    if pqv is None and precursor_quality_schedule is not None:raise ValueError("Precursor quality controls require parameters")
    spv=ribosome_surplus_parameters.vector() if ribosome_surplus_parameters is not None else None
    if spv is not None and ribosome_renewal_parameters is None:raise ValueError("Surplus requires continuous renewal")
    if spv is None and ribosome_surplus_schedule is not None:raise ValueError("Surplus controls require parameters")
    ribomass=biov.copy() if biov is not None else None
    if spv is not None:ribomass[13:15]=4*surplus.RECIPE_VECTOR[:,1]
    rnv=ribosome_renewal_parameters.vector() if ribosome_renewal_parameters is not None else None
    if rnv is not None and ribosome_maintenance_parameters is None:raise ValueError("Renewal requires maintenance")
    if rnv is None and ribosome_renewal_schedule is not None:raise ValueError("Renewal controls require parameters")
    rmv=ribosome_maintenance_parameters.vector() if ribosome_maintenance_parameters is not None else None
    if rmv is not None and rqv is None:raise ValueError("Maintenance requires local RQC branch")
    if rmv is None and ribosome_maintenance_schedule is not None:raise ValueError("Maintenance controls require parameters")
    if rqv is not None and grv is None:raise ValueError("Local RQC requires full GAP43 branch")
    if rqv is None and local_rqc_schedule is not None:raise ValueError("RQC controls require parameters")
    if grv is not None and lmrv is None:raise ValueError("GAP43 recovery requires mRNA branch")
    if grv is None and gap43_regrowth_schedule is not None:raise ValueError("Recovery controls require parameters")
    if ltv is not None and lrv is None:raise ValueError("Local translation requires local ribosome branch")
    if ltv is None and local_translation_schedule is not None:raise ValueError("Local translation controls require parameters")
    if lrv is not None and isrv is None:raise ValueError("Local ribosomes require the full selective-ISR branch")
    if lrv is None and local_ribosome_schedule is not None:raise ValueError("Local controls require local ribosome parameters")
    if isrv is not None and (nrv is None or nrv[16]==0):raise ValueError('Selective ISR requires active nutrient/recycling branch')
    if isrv is None and isr_schedule is not None:raise ValueError('ISR schedule requires parameters')
    if nrv is not None and sub is None:raise ValueError("Nuclear recycling requires finite substrates")
    if nrv is None and nuclear_recycling_schedule is not None:raise ValueError("Recycling schedule requires parameters")
    if sub is not None and shared_parameters is None:raise ValueError("Substrates require shared translation")
    if sub is None and substrate_schedule_input is not None:raise ValueError("Substrate schedule requires parameters")
    shv=shared_parameters.vector() if shared_parameters is not None else None
    if shv is not None and biov is None:raise ValueError('Shared translation requires explicit biogenesis')
    if shv is None and shared_translation_schedule is not None:raise ValueError('Shared schedule requires parameters')
    if biov is not None and cyc is None:raise ValueError('Biogenesis requires API0.26 cycle topology')
    if biov is None and biogenesis_schedule is not None:raise ValueError('Biogenesis controls require explicit parameters')
    if biov is None and maturation_schedule is not None:raise ValueError('Late maturation requires explicit biogenesis pools')
    if cyc is not None and (nucv is None or kcnq_parameters is None or kcnq_coupling_parameters is None):raise ValueError('Cycle branch requires nucleus and explicit KCNQ parameters')
    if cyc is None and any(x is not None for x in (ribosome_schedule,kcnq_parameters,kcnq_coupling_parameters)):raise ValueError('KCNQ/cycle controls require cycle parameters')
    if nucv is not None and rnav is None:raise ValueError('Nucleus extension requires API0.24 topology')
    if nucv is None and nuclear_schedule is not None:raise ValueError('Nuclear schedule requires explicit parameters')
    if nucv is not None and window_s is not None:raise ValueError('Use explicit segments for nuclear schedules; implicit window expansion is not supported')
    if rnav is not None and ckv is None:raise ValueError('RNA/calcium extension requires API0.23 topology')
    if rnav is None and rna_ligand_schedule is not None:raise ValueError('Ligands require RNA/calcium parameters')
    if not np.isfinite(creatine_support) or not 0<=creatine_support<=1:raise ValueError('Creatine support [0,1]')
    if ckv is not None and uv is None:raise ValueError('Creatine extension requires API0.22 topology')
    if ckv is None and creatine_support!=1.:raise ValueError('Creatine intervention requires explicit pools')
    if not np.isfinite(uncoating_support) or not 0<=uncoating_support<=1:raise ValueError('Uncoating support [0,1]')
    if uv is not None and cv is None:raise ValueError('Uncoating requires explicit ACh cargo')
    if uv is None and uncoating_support!=1.:raise ValueError('Uncoating intervention requires an explicit stage')
    if not np.isfinite(filling_support) or not 0<=filling_support<=1:raise ValueError('Filling support must be [0,1]')
    if cv is not None and (not adv[0] or not tv[0]):raise ValueError('ACh cargo requires active terminal and staged recycling')
    if cv is None and filling_support!=1.:raise ValueError('Filling intervention requires explicit cargo model')
    if not np.isfinite([terminal_supply_support,respiratory_reserve,association_support,initiation_support]).all() or not 0<=terminal_supply_support<=1 or not 0<=association_support<=1 or not 0<=initiation_support<=1 or respiratory_reserve<=1:raise ValueError('Invalid supply/reserve/recognition controls')
    energy_parameters=dc_replace(energy_parameters,ox_half_ADP_uM=(respiratory_reserve-1)*energy_parameters.ADP0_uM)
    wp=work_parameters or ProteasomeWorkParameters();wv=wp.vector()
    if not np.isfinite([processing_efficiency,ATPase_support]).all() or min(processing_efficiency,ATPase_support)<0 or max(processing_efficiency,ATPase_support)>1:raise ValueError('Efficiency and ATPase support must be in [0,1]')
    prot=proteostasis_parameters or ProteostasisParameters();prv=prot.vector()
    if not np.isfinite([proteasome_support,folding_load]).all() or not 0<=proteasome_support<=1 or folding_load<1:raise ValueError('Support [0,1], folding load >=1 required')
    rp=recovery_parameters or RecoveryParameters();rpv=rp.vector()
    bp=budget_parameters or BudgetParameters();bpv=bp.vector()
    # External response-shape intervention. Basal ETC and its limiting capacity
    # are unchanged; this is not a transfer of FCCP OCR to ATP capacity.
    if etc_backpressure_exponent!=2.:bpv=np.r_[bpv,etc_backpressure_exponent]
    if not np.isfinite([alpha3_support,respiratory_support]).all() or min(alpha3_support,respiratory_support)<0 or max(alpha3_support,respiratory_support)>1:raise ValueError('Support outside [0,1]')
    mp=mito_parameters or MitoParameters();mv=mp.vector()
    if not np.isfinite(uptake_support) or not 0<=uptake_support<=1:raise ValueError('Uptake support outside [0,1]')
    ip=ion_parameters or IonParameters();ip.validate()
    if rnav is not None and not ip.enabled:raise ValueError('RNA/calcium requires explicit Na/K current and energy ownership')
    if not np.isfinite(pump_support) or not 0<=pump_support<=1:raise ValueError('Pump support in [0,1] required')
    # [codex]2026-11-13: explicit capacity intervention, distinct from inhibition.
    # Do not re-close leak or re-normalize ATP supply after this perturbation.
    pump_capacity_multiplier=kw.pop('pump_capacity_multiplier',prior_controls.get('pump_capacity_multiplier',1.))
    if not np.isfinite(pump_capacity_multiplier) or pump_capacity_multiplier<0:
        raise ValueError('Pump capacity multiplier must be finite and nonnegative')
    pump_effective_support=pump_support*pump_capacity_multiplier
    diagnostic_controls['pump_capacity_multiplier']=float(pump_capacity_multiplier)
    geometry.validate();traffic_parameters.validate();pump_kinetics.validate()
    feedback_parameters=connected_parameters or ConnectedParameters();feedback_parameters.validate()
    if geometry.process_area_um2<=0 or geometry.axial_conductance_nS<=0:raise ValueError('Shared electrical geometry required')
    if not np.isclose(geometry.soma_area_um2+geometry.process_area_um2,profile_c[0]*100,rtol=1e-12):raise ValueError('Total membrane area must preserve frozen profile capacitance')
    if window_s is not None and (not np.isfinite(window_s) or window_s<=0):raise ValueError('Positive window_s required')
    if pump_kinetics.density_per_um2!=1:
        raise ValueError('Use density_per_um2=1 unit kinetics; actual counts come only from traffic')
    e=kw['editing_parameters'];c0=kw['basal_ca_uM'];g=geometry;tr=traffic_parameters
    if not np.isclose(e.volume_um3,g.soma_volume_um3,rtol=1e-12,atol=0):
        raise ValueError('Editing input volume must equal soma accessible volume')
    if not np.isfinite([output_step_s,max_step_s,rtol]).all() or min(output_step_s,max_step_s,rtol)<=0:
        raise ValueError('Positive solver settings required')
    sg=np.asarray(segments,float)
    ncts=nct_module.controls(sg,nct_supplied) if nctv is not None else None
    if sg.ndim!=2 or sg.shape[1]!=34 or not len(sg) or not np.isfinite(sg).all() or np.any(sg<0) or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[1:,0]!=sg[:-1,1]):
        raise ValueError('Contiguous finite 34-column schedule required')
    if np.any(sg[:,[2,3,4,5,6,8,9,10,11,12,22,24,25,26]]>1):raise ValueError('Driver fractions outside [0,1]')
    if window_s is not None and (segment_output_steps is not None or segment_max_steps is not None):
        raise ValueError('Explicit per-segment numerical settings require window_s=None')
    for label,setting in [('output',segment_output_steps),('max',segment_max_steps)]:
        if setting is not None:
            a=np.asarray(setting,float)
            if a.shape!=(len(sg),) or not np.isfinite(a).all() or np.any(a<=0):
                raise ValueError('Positive per-segment '+label+' steps with one value per input row required')
    ss=np.c_[sg[:,:2],np.full(len(sg),g.process_area_um2/(g.soma_area_um2+g.process_area_um2)),np.ones((len(sg),2))] if spatial_schedule is None else np.asarray(spatial_schedule,float)
    if ss.shape!=(len(sg),5) or not np.isfinite(ss).all() or not np.array_equal(ss[:,:2],sg[:,:2]) or np.any(ss[:,2:]<0) or np.any(ss[:,2:]>1):
        raise ValueError('Spatial rows must match source times; fraction/support in [0,1]')
    es=validate_schedule(kw.pop('energy_schedule',None),sg)
    electrical=np.c_[sg[:,:2],np.zeros((len(sg),2))] if electrical_schedule is None else np.asarray(electrical_schedule,float)
    if electrical.shape!=(len(sg),4) or not np.isfinite(electrical).all() or not np.array_equal(electrical[:,:2],sg[:,:2]):raise ValueError('Electrical schedule must match segment boundaries')
    if window_s is not None:
        expanded=[];ee=[];spat=[];energy=[]
        for row,erow,srow,energyrow in zip(sg,electrical,ss,es):
            edges=np.r_[np.arange(row[0],row[1],window_s),row[1]]
            for a,b in zip(edges[:-1],edges[1:]):
                for target,src in ((expanded,row),(ee,erow),(spat,srow),(energy,energyrow)):
                    v=src.copy();v[:2]=[a,b];target.append(v)
        sg,ss,es,electrical=map(np.asarray,(expanded,spat,energy,ee))
    # The original runner performs its full parameter validation and supplies
    # the ORIGINAL baseline, not the endpoint of its tiny validation interval.
    ghost=sg[:1].copy();ghost[0,:2]=[0,1e-4];ghost[0,2]=0
    old=core.run_pmca_network(ghost,pmca_parameters=memory_parameters,energy_parameters=energy_parameters,**kw)
    en=energy_parameters.vector();ev=kw['er_parameters'].vector();rv=kw['iron_release_parameters'].vector();qv=kw['ncoa4_parameters'].vector()
    base=np.r_[old['baseline_state'],np.zeros(28)];base[135]=base[133];base[136:139]=en[:3]
    base[139:143],tf,trev=tr.initial(g)
    vs,vp=g.soma_volume_um3,g.process_volume_um3
    ep=np.array([e.tau_rna_h*3600,e.tau_surface_h*3600,e.incorporation_R_over_Q,*e.weights,
        e.g_reference_nS,e.reversal_mV-e.voltage_mV,e.cp_conductance_ratio,e.cp_unblocked_fraction,e.pf_ci,e.pf_cp,e.volume_um3,e.kappa])
    iv=kw['iron_parameters'].vector();nv=kw['organelle_parameters'].vector();av=kw['aggregation_parameters'].vector()
    hv=np.r_[kw['autoregulation_parameters'].vector(),kw['rna_feedback_parameters'].vector()]
    required={'basal_ca_uM','calcium_tau_s','enzyme_parameters','tdp_parameters','localization_parameters','stmn2_parameters','inactivation_parameters','phosphorylation_parameters','reactivation_parameters','stability_parameters'}
    _,p,kv,sv,_=core._prior._prior._agg._baseline({k:kw[k] for k in required},av)
    ag0=base[37:45].sum();bn=base[6]+base[7]
    from dataclasses import replace
    unit=replace(pump_kinetics,density_per_um2=1/g.soma_area_um2)
    mapped,_=map_surface_pmca(unit,area_um2=g.soma_area_um2,accessible_volume_um3=vs,basal_ca_uM=c0,
        gamma_per_s=e.gamma_per_s,memory_parameters=memory_parameters,energy_parameters=energy_parameters)
    pv=mapped.vector();unit_ratio=pv[8];pv[8]*=base[139]
    pp=replace(pump_kinetics,density_per_um2=1/g.process_area_um2)
    mappedp,_=map_surface_pmca(pp,area_um2=g.process_area_um2,accessible_volume_um3=vp,basal_ca_uM=c0,
        gamma_per_s=g.process_gamma_per_s,memory_parameters=memory_parameters,energy_parameters=energy_parameters)
    ppv=mappedp.vector();punit=ppv[8];ppv[8]*=base[141]
    pm0=pmca2_flux(c0,base[133],c0,e.gamma_per_s,en[9],1.,pv)
    delta_s=pm0-en[9]*e.gamma_per_s*c0
    pmp0=pmca2_flux(c0,base[135],c0,g.process_gamma_per_s,en[9],1.,ppv)
    ncp0=(1-en[9])*g.process_gamma_per_s*c0
    _,af0,ar0=traffic_rates(base[139:143],tr,tf,trev,1.,1.)
    cost0s=af0*tr.ATP_molecules_per_delivery/(IONS_PER_UM3_PER_UM*vs)
    cost0p=ar0*tr.ATP_molecules_per_delivery/(IONS_PER_UM3_PER_UM*vp)
    delta_p=pmp0-en[9]*g.process_gamma_per_s*c0
    config=dict(old['config'],topology='spatial_PMCA_162_v1',geometry=asdict(g),traffic=asdict(tr),pump_kinetics=asdict(pump_kinetics),
        initial_pump_pools=base[139:143].tolist(),initial_PMCA_soma_uM_s=float(pm0),initial_PMCA_process_uM_s=float(pmp0),
        matched_rest='Basal Ca input and supply set once from initial pumps, never reset after perturbation',
        process_biochemistry='Ca, ATP/ADP/AMP and pumps only; inherited CAPN/TDP/STMN2/ER/iron network is somatic')
    local_options=dict(injury_parameters)
    if local_options['basal_ca_uM']!=c0:raise ValueError('Local baseline must equal common calcium baseline')
    if len(local_options['substrates'])!=2 or local_options['substrates'][1].name!='betaII_spectrin' or local_options['substrates'][1].initial_intact_amount!=1.:raise ValueError('Explicit two local lineages and unit spectrin reference required')
    li=local.simulate_local_injury([0.,1e-4],[c0,c0],**local_options)
    lev=local_options['enzyme_parameters'].vector().copy();lev[7]*=local_options.get('enzyme_pool_scale',1.)
    liv=local_options['inactivation_parameters'].vector();lkv=local_options['reactivation_parameters'].vector();lsv=local_options['stability_parameters'].vector()
    lpv=np.array([q.vector() for q in local_options['substrates']]);la=li['baseline_activity']
    lsyn=np.array([(v[1]+v[0]*la)*q.initial_intact_amount for v,q in zip(lpv,local_options['substrates'])])
    gl,el,hold=derived(profile_p,profile_c);init=profile_c[5] if profile=='Burley' else profile_c[2]
    if profile!='Burley':hold=0.
    vg0=1/(1+np.exp(-(init-coupling['vg_half_mV'])/coupling['vg_slope_mV']))
    sk0=c0**coupling['sk_hill']/(coupling['sk_half_uM']**coupling['sk_hill']+c0**coupling['sk_hill'])
    electrical_base=np.tile(np.r_[init,gates(init,profile_p),vg0,sk0],2)
    base=np.r_[base,li['baseline_state'],electrical_base]
    geomvec=np.array([g.soma_area_um2,g.process_area_um2,g.axial_conductance_nS,init,hold,gl,el])
    couplevec=np.array([coupling[k] for k in ('vg_half_mV','vg_slope_mV','vg_tau_ms','sk_half_uM','sk_hill','sk_tau_ms')]+[.9])
    fbvec=np.array([feedback_parameters.spectrin_to_Nav_loss_gain,feedback_parameters.spectrin_to_delivery_loss_gain,feedback_parameters.g_vgcc_nS,feedback_parameters.g_sk_nS])
    config.update(topology='connected_spatial_199_v1',profile=profile,HH_parameters=profile_p.tolist(),HH_context=profile_c.tolist(),
        bulk_coupling=coupling,feedback=asdict(feedback_parameters),local_injury=li['config'],
        calcium_owner='One bulk cytosolic Ca pool in each compartment; somatic ER. No independent shell pool in this reduced driver.',
        geometry_owner='Shared native cylinder geometry; lateral membrane area, actual accessible volumes; series half-length diffusion resistance',
        electrical_solver='Continuous HH/biochemistry LSODA, native NEURON reference tested separately; no repeated short traces',
        process_biochemistry='Local CAPN and TDP/spectrin lineages plus Ca/ATP and conserved PMCA traffic',
        protein_synthesis_ATP_cost=None)
    base,ionvec,ioncv,ionbase=prepare_ions(base,profile_p,profile_c,geomvec,couplevec,fbvec,ep,c0,np.array([vs,vp]),np.array([(1-en[9])*e.gamma_per_s*c0,ncp0]),ip,pump_leak_fraction)
    if pump_leak_fraction is not None:
        config['pump_leak_closure']=dict(fraction=float(pump_leak_fraction),capacity_multipliers=ionbase[:,9].tolist(),
            source='Krishnan2015 pump/leak equilibrium architecture; fraction and human capacity unmeasured',
            scope='Fixed pump capacities derived jointly with passive leaks at initialization. Same 3Na/2K/1ATP owner and basal supply ledger; no state clamp, reset or restorative ion source. Does not resolve Cl/water or prove healthy physiology.')
    if ip.enabled and min(pm0+ionbase[0,8],pmp0+ionbase[1,8]) < -1e-9:
        raise ValueError('Resting Ca balance would require unowned active Ca export; recalibrate PMCA/NCX before accepting this combination')
    config.update(topology='ion_energy_239_v1',ions=asdict(ip),ion_baseline=ionbase.tolist(),
      resting_ion_background='Fixed fluxes constructed once to preserve calibrated rest; not measured leaks. Explicit reservoir ledgers.',
      NCX='Reversible bounded mass-action flux with independent uncertain capacity; basal external Ca input matched once. No singular normalization at reversal.',
      ion_energy='NaK actual cycles consume ATP; prior indirect NCX ATP proxy removed; Na-entry/3 is a separate demand metric, never another ATP debit')
    background_ref=None
    if background_parameters is not None:
        background_ref=ion_homeostasis.prepare(base,profile_c,geomvec,ionvec,ionbase,np.array([vs,vp]),gl,background_parameters)
        config.update(ion_background=asdict(background_parameters),ion_background_reference=background_ref.tolist(),
            resting_ion_background='Passive Na/K partition of existing fitted leak: replace fixed material fluxes; voltage adds only reversal shifts, not another conductance. Residual leak/holding current is phenomenological; Cl/water unresolved.')
    base=np.r_[base,np.tile([mv[1]*(1+mv[2])*mv[3],0.,0.,0.],2)]
    config.update(topology='matrix_ion_energy_247_v1',mitochondria=asdict(mp),
      matrix_model='Finite effective buffered Ca store in each compartment; effective exit is not a resolved NCLX flux. No matrix Na, proton gradient, oxygen/substrate or MCU energy debit.',
      oxidative_modulation='Optional bounded matrix-Ca modifier of existing OXPHOS production, with matching ATP/ADP and production ledgers; not a new independent ATP source')
    base=np.r_[base,np.tile(np.r_[bpv[1:3],np.zeros(8)],2)]
    config.update(topology='source_recovery_budget_267_v1',recovery=asdict(rp),budget=asdict(bp),
        protein_synthesis_ATP_cost='Gross soma/local TDP tracer fluxes times uncertain absolute uM; 4 high-energy equivalents/aa. Basal allocation replaces part of existing other-demand, not an additional basal debit.',
        respiratory_model='Finite lumped proton-motive reserve and O2 balance; existing OXPHOS modulated once. Not resolved matrix Na, delta-psi, pH, respiratory complexes or substrate chemistry.')
    if prv[0] and (not bpv[0] or nv[25]==0):raise ValueError('Proteostasis coupling requires active energy budget and explicit inherited UPS/lysosome partition')
    base=np.r_[base,np.zeros(8)]
    config.update(topology='proteostasis_energy_275_v1',proteostasis=asdict(prot),proteasome_driver='External support/folding can change on restart; no new baseline or synthesis')
    if wv[0] and not prv[0]:raise ValueError('Pool ATPase requires explicit proteostasis branch')
    config.update(topology='proteasome_work_275_v1',proteasome_work=asdict(wp),pool_model='Lumped ATPase pool separate from productive TDP loss; no enzyme occupancy or fixed residence time. No finite capacity coupling.')
    base=np.r_[base,np.array([0.,1.,0.,npv[12],0.,0.,0.]),en[:3],np.zeros(9)]
    base=np.r_[base,np.zeros(2)]
    base=np.r_[base,np.zeros(7)]
    base=np.r_[base,np.zeros(5)]
    config.update(recycling=asdict(recp),terminal_pathways=asdict(pathp),adaptive_recycling=asdict(adp))
    config.update(topology='adaptive_recycling_terminal_308_v1',terminal=asdict(tp),nmj=asdict(np_),respiratory_reserve=respiratory_reserve,activity=asdict(ap))
    if cv is not None:
        base=np.r_[base,npv[5]*cv[1],0.,0.,0.]
        config.update(topology='cholinergic_cargo_terminal_312_v1',cargo=asdict(cargo_parameters),
            cargo_energy='Existing per-cycled-vesicle ATP budget retained; no extra VAChT ATP debit. Filling uses a lumped ATP/gradient support, no measured pH law or ChAT/choline balance.')
    if uv is not None:
        base=np.r_[base,np.zeros(3)]
        config.update(topology='uncoating_terminal_315_v1',uncoating=asdict(uncoating_parameters))
    if ckv is not None:
        pc0=equilibrium_PCr(base[[120,136,282]],base[[121,137,283]],ckv[:3],ckv[4])
        base=np.r_[base,pc0]
        config.update(topology='creatine_terminal_318_v1',creatine=asdict(creatine_parameters),
            creatine_initialization='CK equilibrium at original ATP/ADP, never reset on restart; finite pools, no creatine transport')
    # External pump_support may change on restart, like applied current; biological capacities may not.
    rnals=ligand_schedule(sg,rna_ligand_schedule) if rnav is not None else None
    if rnav is not None:
        base=np.r_[base,rnav[22],rnav[22]]
        config.update(topology='rna_calcium_surface_320_v1',rna_calcium=asdict(rna_calcium_parameters),
            rna_scope='Functional protein factors are explicit hypotheses, not RNA fold changes; CLU/PRNP retained as unresolved evidence. GABA Cl/HCO3 reversal clamped; no chloride homeostasis or extra trafficking energy debit.')
    nucls=nucleus_schedule(sg,nuclear_schedule) if nucv is not None else None
    f0=kw['localization_parameters'].nuclear_fraction
    if biov is not None:
        if not bpv[0]:raise ValueError('Nascent-chain accounting requires active protein energy budget')
        nucv,cyc=bio_prepare(nucv,cyc,p,bpv,biov)
    if nucv is not None:
        base=np.r_[base,initial_nucleus(p,hv,nucv,f0)]
        config.update(topology='nucleus_ribosome_330_v1',nucleus=asdict(nucleus_parameters),
            nuclear_scope='Dynamic partition of existing soluble TDP; equal losses across locations; no new TDP copy. Explicit TARDBP/STMN2 RNA processing/export and finite selected ribosome slots. Human rates unknown.')
    cycls=cycle_schedule(sg,ribosome_schedule) if cyc is not None else None
    if cyc is not None:
        cybase=initial_cycle(nucv,cyc);kp=kcnq_vector(kcnq_parameters,kcnq_coupling_parameters)
        kbase=initial_kcnq(kcnq_parameters,kcnq_coupling_parameters,base[[185,192]])
        if shv is not None:shbase,cybase,shref=shared_setup(cybase,nucv,cyc,biov,nv,kp,kbase,shv)
        base=np.r_[base,cybase,kbase]
        config.update(topology='nuclear_targets_ribosome_cycle_364_v1',ribosome_cycle=asdict(cycle_parameters),kcnq_transport=asdict(kcnq_parameters),kcnq_coupling=asdict(kcnq_coupling_parameters))
        current_kp=kp
        if differential_KCNQ_activity_multiplier!=1:
            current_kp=kp.copy();current_kp[15]*=differential_KCNQ_activity_multiplier
    if absolute_module is not None:
        ionbase,absolute_delta0,absolute_config=absolute_module.prepare(absolute_parameters,base,profile_p,profile_c,geomvec,ionvec,ionbase,background_ref,np.array([vs,vp]))
        background_ref=ion_homeostasis.prepare(base,profile_c,geomvec,ionvec,ionbase,np.array([vs,vp]),gl,background_parameters)
        config.update(absolute_kcnq=absolute_config,ion_baseline=ionbase.tolist(),ion_background_reference=background_ref.tolist())
        config['pump_leak_closure']['capacity_multipliers']=ionbase[:,9].tolist()
    bios=bio_schedule(sg,biogenesis_schedule) if biov is not None else None
    latecontrols=maturation_module.controls(sg,maturation_schedule) if maturation_schedule is not None else None
    shares=shared_schedule(sg,shared_translation_schedule) if shv is not None else None
    if biov is not None:
        biobase=bio_initial(nucv,cyc,biov);base=np.r_[base,biobase]
        config.update(topology='biogenesis_nascent_403_v1',biogenesis=asdict(biogenesis_parameters),effective_ribosome_slots=nucv[6:8].tolist(),effective_selected_concentration_uM=cyc[4],nascent_scope='Four elongation bins; concentration-derived slots, external RP kit supply, idle-subunit retirement; not whole-cell self-replicating ribosomes or full RQC.')
    if shv is not None:
        base=np.r_[base,shbase]
        config.update(topology='shared_translation_497_v1',shared_translation=asdict(shared_parameters),shared_reference=shref.tolist(),nascent_scope='Eight selected classes share40S/60S/ATP; RP kits are representative-chain equivalents, not all genes; 5S separate from47S support.')
    subs=substrate_schedule(sg,substrate_schedule_input) if sub is not None else None
    if sub is not None:
        subbase,subref=substrate_initial(sub,biov,nucv,shref,base[120],base[121]);base=np.r_[base,subbase]
        config.update(topology="translation_substrates_513_v1",translation_substrates=asdict(substrate_parameters),substrate_reference=subref.tolist())
    nrcontrols=nrec_schedule(sg,nuclear_recycling_schedule) if nrv is not None else None
    if nrv is not None:
        rrrow=sg[0].copy();rrrow[10]=1.;rrrow[12]=1.
        rrzero=np.zeros(513);rrzero[398:400]=np.log(2)/(biov[1]*86400)
        rrtdp=intact_TDP_loss(base,p,nv,prv,wv,en[0],bpv[9],rrrow,1.,1.,1.,1.)
        rrsource=removal_sources(base,rrzero,rrtdp,ribomass,nucv,nv,shv,kp)
        rrbase,nrref=nrec_initial(rrsource,nrv,subref);base=np.r_[base,rrbase]
        config.update(topology='nuclear_recycling_537_v1',nuclear_recycling=asdict(nuclear_recycling_parameters),recycling_reference=nrref.tolist(),recycling_source_baseline=rrsource.tolist())
    isrcontrols=isr_module.controls(sg,isr_schedule) if isrv is not None else None
    if isrv is not None:
        base,cybase,subref,nrref,isrref=isr_module.initialize(base,isrv,biov,nucv,cyc,cybase,sub,subref,nrv,nrref)
        config.update(topology='selective_isr_562_v1',selective_isr=asdict(isr_parameters),ISR_reference=isrref.tolist(),substrate_reference=subref.tolist(),recycling_reference=nrref.tolist())
    if latecontrols is not None:config.update(late_maturation_schema='independent_40S_60S_v1',late_maturation_scope='Selected soma pools only; no abundance-to-rate transfer or resolved late maturation energy cost')
    lrcontrols=local_ribo.controls(sg,local_ribosome_schedule) if lrv is not None else None
    if lrv is not None:
        base=np.r_[base,np.zeros(12)]
        config.update(topology="local_ribosomes_574_v1",local_ribosomes=asdict(local_ribosome_parameters),local_ribosomes_initialization="Empty local queues; startup scenario, not calibrated healthy equilibrium")
    ltcontrols=local_translation.controls(sg,local_translation_schedule) if ltv is not None else None
    if ltv is not None:
        base=local_translation.initialize(base,ltv)
        config.update(topology="local_translation_605_v1",local_translation=asdict(local_translation_parameters),local_translation_scope="One representative cargo, explicit local substrates and two-way exchange; unchanged pre-existing protein production; no gene-specific functional effect")
    lmrcontrols=local_mrna.controls(sg,local_mrna_schedule) if lmrv is not None else None
    if lmrv is not None:
        base,ltv,lmrref=local_mrna.initialize(base,ltv,lrv,biov,lmrv,vs/vp,nucv[8])
        config.update(topology="GAP43_mrna_617_v1",local_mrna=asdict(local_mrna_parameters),local_mrna_reference=lmrref.tolist(),local_translation_effective=ltv.tolist(),local_initialization="Stationary selected molecular pools; derived demand-matched local NDPK capacity; not equilibrium of all electrical/mitochondrial states")
    grcontrols=gap43_regrowth.controls(sg,gap43_regrowth_schedule) if grv is not None else None
    if grv is not None:
        base,grref=gap43_regrowth.initialize(base,grv)
        config.update(topology="GAP43_recovery_620_v1",gap43_regrowth=asdict(gap43_regrowth_parameters),gap43_regrowth_reference=grref.tolist(),gap43_regrowth_scope="Normalized virtual-lesion readout; no geometry/NMJ feedback or repair-mass/ATP budget")
    rqcontrols=local_rqc.controls(sg,local_rqc_schedule) if rqv is not None else None
    if rqv is not None:
        base=np.r_[base,np.zeros(20)]
        config.update(topology="local_RQC_640_v1",local_RQC=asdict(local_rqc_parameters),local_RQC_scope="Effective eligible stalls, separate40S/60S/chain release and delayed AA return;unmeasured human rates;no polysome collision detection")
    rmcontrols=maintenance.controls(sg,ribosome_maintenance_schedule) if rmv is not None else None
    if rmv is not None:
        base=maintenance.initialize(base,rmv)
        config.update(topology="ribosome_maintenance_675_v1",ribosome_maintenance=asdict(ribosome_maintenance_parameters),ribosome_maintenance_scope="RPS26 exchange hypothesis and selected local retirement;finite substrates;unmeasured human rates")
    rncontrols=renewal.controls(sg,ribosome_renewal_schedule) if rnv is not None else None
    if rnv is not None:
        base,rnref=renewal.initialize(base,rnv,rmv,ltv,lrv,biov,vs/vp,nucv[8])
        config.update(topology="ribosome_renewal_678_v1",ribosome_renewal=asdict(ribosome_renewal_parameters),renewal_reference=rnref.tolist())
    spcontrols=surplus.controls(sg,ribosome_surplus_schedule) if spv is not None else None
    if spv is not None:
        base=np.r_[base,np.zeros(14)]
        config.update(topology="ribosome_surplus_692_v1",ribosome_surplus=asdict(ribosome_surplus_parameters),RP_recipe=surplus.RECIPE)
    pqcontrols=precursor_quality.controls(sg,precursor_quality_schedule) if pqv is not None else None
    if pqv is not None:
        base=np.r_[base,np.zeros(34)]
        config.update(topology="precursor_quality_726_v1",precursor_quality=asdict(precursor_quality_parameters))
    rscontrols=rna_salvage.controls(sg,rna_salvage_schedule) if rsv is not None else None
    if rsv is not None:
        base=np.r_[base,rna_salvage.initial(rsv)]
        config.update(topology="rna_salvage_745_v1",rna_salvage=asdict(rna_salvage_parameters),rna_recipe=rna_salvage.RECIPE)
    if ctv is not None:
        base,handoff=coupled_translation.initialize(base,ctv,ltv,biov[0],vs/vp)
        config.update(topology='coupled_codon_1600_v1',coupled_translation=asdict(detailed_parameters),coupled_handoff=handoff)
        if not config['coupled_translation'].get('balanced_coarse_AA',False):
            config['coupled_translation'].pop('balanced_coarse_AA',None) # preserve legacy checkpoint identity
    if irates is not None:
        base=individual_rp.initialize(base,biov[0])
        config.update(topology='individual_RP_2160_v1',individual_RP=asdict(individual_rp_parameters),individual_RP_rates_per_s=irates.tolist())
    rppcontrols=rp_production.controls(sg,rpp_supplied) if rppv is not None else None
    if rppv is not None:
        base=rp_production.initialize(base,shref)
        config.update(topology='individual_RP_production_3360_v1',RP_production=asdict(rpp_parameters),RP_production_genes=rp_production.GENES)
    nct_offset=len(base)
    if nctv is not None:
        # The NCT initializer owns only its appended pools; keep all RP queues.
        nct_prefix=base[:2160] if rppv is not None else base
        nct_initial=nct_module.initialize(nct_prefix,bpv[9],nctv)
        base=np.r_[base,nct_initial[len(nct_prefix):]]
        config.update(topology=config['topology']+'_nct48',nuclear_transport=asdict(nct_parameters),nuclear_transport_offset=nct_offset)
    nrt_offset=len(base)
    nrtshape=nrt_module.layout(rppv is not None)
    nrtcontrols=nrt_module.controls(sg,nrt_supplied) if nrtv is not None else None
    if nrtv is not None:
        base=nrt_module.initialize(base,nrtv,nctv,biov[0],nrtshape,rppv if rppv is not None else np.ones((2,80)))
        config.update(topology=config['topology']+'_nrt50',RNA_export=asdict(nrt_parameters),RNA_export_offset=nrt_offset,RNA_export_layout=nrtshape.tolist())
    rmkcontrols=rmk_module.controls(sg,rmk_supplied) if rmkv is not None else None
    if rmkv is not None:
        base=rmk_module.initialize(base,rmkv,nrtshape,nrt_offset,rppv)
        if base[120]<=0:raise ValueError('Positive baseline ATP required by RP-mRNA kinetics')
        config.update(topology=config['topology']+'_rmk51',RP_mRNA_kinetics=asdict(rmk_parameters))
    if conditional_RNA_export_initialization:
        if rmkv is None or nrtv is None or nctv is None:raise ValueError('RNA initialization requires full RNA/NCT/RP kinetics')
        q3init=load_module(B.parent/'2026-11-05-Q3-nuclear-baseline/rna_initialization_codex.py','_Q3_rna_initialization05')
        base,q3report=q3init.prepare(base,v=nrtv,nv=nctv,shape=nrtshape,off=nrt_offset,noff=nct_offset,
          rmkv=rmkv,rppv=rppv,p=p,hv=hv,nucv=nucv,kp=kp,lmrref=lmrref,nrt=nrt_module)
        config['RNA_initialization']=q3report
    rp_RNA_reference=None
    if gene_specific_RP_translation_reference:
        if not conditional_RNA_export_initialization or rmkv is None:raise ValueError('Gene-specific translation reference requires conditional RNA initialization and RP kinetics')
        rp_RNA_reference=np.array([base[nrt_offset+8*(5+i)+4] for i in range(80)])
        if np.any(rp_RNA_reference<=0):raise ValueError('Positive per-gene basal RNA required')
        config['RP_translation_reference_cyt_uM']=rp_RNA_reference.tolist()
        config['RP_translation_reference_scope']='Conditional per-gene basal loading efficiency; each gene normalized once to its own initial cytoplasmic RNA. No refit during perturbation and no measured human translation efficiencies.'
    if basal_translation_capacity_scale!=1.:
        config['basal_translation_capacity_scale']=float(basal_translation_capacity_scale)
        config['basal_translation_capacity_scope']='Q3 conditional initiation capacity; finite ribosomes and all substrate/energy costs retained; not measured human capacity'
    chloride_offset=len(base);chloride_ref=None
    if chloride_module is not None:
        base,chloride_ref=chloride_module.prepare(base,ionvec,background_ref,chloride_parameters)
        config.update(passive_chloride=asdict(chloride_parameters),passive_chloride_offset=chloride_offset,
            passive_chloride_reference=chloride_ref.tolist(),topology=config['topology']+'_passiveCl12')
    # [claude] 2026-11-13: six acid states follow the optional Cl block; resting PMCA fluxes pm0/pmp0 are the reference.
    if acid_module is not None:
        acid_offset=len(base)
        base,acid_ref=acid_module.prepare(base,np.array([pm0,pmp0]),ionvec,acid_parameters)
        config.update(acid_extrusion=asdict(acid_parameters),acid_extrusion_offset=acid_offset,acid_extrusion_version=acid_module.VERSION,
            acid_extrusion_reference=acid_ref.tolist(),topology=config['topology']+'_acidNaH13')
    def checked_cycle(z,initial):
        # [codex] 2026-11-14: legacy molecular positivity views exclude signed optional acid states.
        z=z[:,:acid_offset];initial=initial[:acid_offset]
        ch=cycle_checks(z,initial)
        return nrt_module.revise_host_checks(ch,z,initial,nrtv,nrtshape,nrt_offset,lmrv) if nrtv is not None else ch
    def checked_shared(z,initial):
        actual_rpp=z;initial_rpp=initial
        if rppv is not None:z,initial=rp_production.particle_view(z,initial)
        if ctv is not None:z,initial=coupled_translation.particle_view(z,initial,biov[0],vs/vp)
        if pqv is not None:z,initial=precursor_quality.adjusted(z,initial)
        if spv is not None:z,initial=surplus.adjusted(z,initial)
        if rmv is not None:z,initial=maintenance.adjusted_particles(z,initial)
        if rqv is not None:z,initial=local_rqc.adjusted_particles(z,initial)
        if ltv is not None:z,initial=local_translation.adjusted_particles(z,initial)
        if lrv is not None:z,initial=local_ribo.adjusted(z,initial)
        zz,bb=isr_module.adjusted_ribosomes(z,initial) if isrv is not None else (z,initial)
        ch=shared_checks(zz,bb,nucv,biov,shv,shref)
        return rp_production.revise_shared(ch,actual_rpp,initial_rpp,biov[0],shref) if rppv is not None else ch
    def checked_recycling(z,initial):
        # [codex] 2026-11-14: legacy molecular positivity views exclude signed optional acid states.
        z=z[:,:acid_offset];initial=initial[:acid_offset]
        actual=z
        if nctv is not None:z,initial=nct_module.revise_somatic_checks(None,z,initial,nct_offset)
        ch=nrec_checks(z,initial,nrv,sub)
        ch=isr_module.revise_checks(ch,z,initial,isrv,biov) if isrv is not None else ch
        ch=local_translation.revise_somatic_checks(ch,z,initial,vs/vp) if ltv is not None else ch
        # Transport ledger views add cumulative exchanged guanylates to GTP
        # and subtract them from GDP. These accounting coordinates can become
        # negative even while every physical pool remains positive. Use them
        # for conservation only; positivity always refers to the actual state.
        ch['substrate_nonnegative']=float(max(0.,-actual[:,497:513].min()))
        return ch
    def checked_local(z,initial):
        # [codex] 2026-11-14: legacy molecular positivity views exclude signed optional acid states.
        z=z[:,:acid_offset];initial=initial[:acid_offset]
        actual=z;actual_initial=initial
        extra=local_translation.checks(z,initial,ltv,biov[0],vs/vp) if ltv is not None else {}
        if lmrv is not None:extra.update(local_mrna.checks(z,initial,lmrv,biov[0],vs/vp))
        if grv is not None:extra.update(gap43_regrowth.checks(z,initial))
        if rqv is not None:
            extra=local_rqc.revise_checks(extra,z,initial,rqv,ltv[0],biov[0],vs/vp)
            z,initial=local_rqc.adjusted_particles(z,initial)
        if rmv is not None:
            extra=maintenance.revise_checks(extra,actual,actual_initial,rmv,ltv,biov[0],vs/vp,ribomass[13]/4,ribomass[14]/4)
            z,initial=maintenance.adjusted_particles(z,initial)
        if ltv is not None:z,initial=local_translation.adjusted_particles(z,initial)
        if spv is not None:extra.update(surplus.checks(actual,actual_initial,spv,biov[0]))
        if pqv is not None:extra.update(precursor_quality.checks(actual,actual_initial,pqv,biov[0],surplus.RECIPE_VECTOR))
        if rsv is not None:extra.update(rna_salvage.checks(actual,actual_initial,rsv,biov))
        if ctv is not None:
            z,initial=coupled_translation.particle_view(z,initial,biov[0],vs/vp)
            extra=coupled_translation.revise_checks(extra,actual,actual_initial,ctv,ltv,biov[0],vs/vp,ribomass[13]/4,ribomass[14]/4)
        if irates is not None:extra.update(individual_rp.checks(actual,actual_initial,biov[0]))
        if rppv is not None:
            extra=rp_production.revise_inventory(extra,actual,actual_initial,biov[0])
            extra.update(rp_production.checks(actual,actual_initial,shref,biov[0]))
        if nctv is not None:
            nx,nb=nrt_module.nct_check_view(actual,actual_initial,nrtv,nrtshape,nrt_offset,nct_offset) if nrtv is not None else (actual,actual_initial)
            extra.update(nct_module.checks(nx,nb,bpv[9],nct_offset))
        if nrtv is not None:
            extra.update(nrt_module.checks(actual,actual_initial,nrtv,nrtshape,nrt_offset,nct_offset,biov[0],lmrv))
            extra=nrt_module.revise_host_checks(extra,actual,actual_initial,nrtv,nrtshape,nrt_offset,lmrv)
        return dict(local_ribo.checks(z,initial,lrv,biov[0],vs/vp),**extra)
    state_columns=old['columns']+EXTRA_NAMES+tuple('process_'+q for q in li['columns'])+tuple(k+'_'+q for k in ('soma','process') for q in ('voltage_mV','m','h','n','w','VG_gate','SK_gate'))+ION_NAMES+MITO_NAMES+BUDGET_NAMES+tuple(q.replace('TDP_UPS_ATP','UPS_pool_and_overhead_ATP') for q in PROTEOSTASIS_NAMES)+TERMINAL_NAMES+ACTIVITY_NAMES+tuple('terminal_internal_alkaline_quanta' if adv[0] and q=='terminal_maturation_queue_quanta' else q for q in RECYCLING_NAMES)+ADAPTIVE_NAMES+(CARGO_NAMES if cv is not None else ())+(UNCOATING_NAMES if uv is not None else ())+(CREATINE_NAMES if ckv is not None else ())+(RNA_CALCIUM_NAMES if rnav is not None else ())+(NUCLEUS_NAMES if nucv is not None else ())+(NUCLEAR_CYCLE_NAMES if cyc is not None else ())+(BIOGENESIS_NAMES if biov is not None else ())+(SHARED_NAMES if shv is not None else ())+(SUBSTRATE_NAMES if sub is not None else ())+(NUC_REC_NAMES if nrv is not None else ())+(isr_module.NAMES if isrv is not None else ())+(local_ribo.NAMES if lrv is not None else ())+(local_translation.NAMES if ltv is not None else ())+(local_mrna.NAMES if lmrv is not None else ())+(gap43_regrowth.NAMES if grv is not None else ())+(local_rqc.NAMES if rqv is not None else ())+(maintenance.NAMES if rmv is not None else ())+(renewal.NAMES if rnv is not None else ())+(surplus.NAMES if spv is not None else ())+(precursor_quality.NAMES if pqv is not None else ())+(rna_salvage.NAMES if rsv is not None else ())+(coupled_translation.NAMES if ctv is not None else ())+(individual_rp.NAMES if irates is not None else ())+(rp_production.NAMES if rppv is not None else ())+(nct_module.NAMES if nctv is not None else ())+(nrt_module.names(nrtshape) if nrtv is not None else ())
    if chloride_module is not None:state_columns+=chloride_module.NAMES
    if acid_module is not None:state_columns+=acid_module.NAMES  # [claude] 2026-11-13
    if len(state_columns)!=len(base):raise ValueError('State labels and vector lengths differ')
    if water_module is not None:
        water_scale,water_config=water_module.prepare(water_parameters,g,len(base),chloride_offset if chloride_module is not None else None)
        config['mobile_water_volumes']=water_config
        config['geometry_owner']='Fixed native membrane geometry; explicit mobile-water capacity separate from effective calcium/reference volume; see mobile_water_volumes passport'
    if pump_response_module is not None:
        pump_response_vector,pump_response_config=pump_response_module.prepare(pump_response_parameters,base,ionbase,g)
        config['pump_response']=pump_response_config
    # [codex] 2026-11-14: acid deviations/counters are signed; old nonnegative tail checks stop before this block.
    schema=state_schema(state_columns,config)
    checkpoint_schema_status=check_checkpoint_schema(checkpoint,schema)
    atol=solver_atol(atol,len(base))
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    signed=[0,22,23,24,118,129,131,134,143,145,147,153,156,157,158,161]
    def minimum(z):
        return float(min(z[:,np.setdiff1d(np.arange(162),signed)].min(),(z[:,0]+c0).min(),(z[:,134]+c0).min(),
            (z[:,1]-z[:,28]).min(),(z[:,2]-z[:,29]).min(),(z[:,31]-z[:,32]).min(),z[:,162:185].min()))
    if checkpoint is None:y=base.copy();origin=float(sg[0,0])
    else:
        if checkpoint['fingerprint']!=fingerprint or checkpoint['time_s']!=sg[0,0] or not np.array_equal(checkpoint['baseline_state'],base):
            raise ValueError('Checkpoint geometry/parameters/baseline/time changed')
        y=np.asarray(checkpoint['state'],float);origin=checkpoint['origin_time_s']
        if y.shape!=base.shape or not np.isfinite(y).all() or minimum(y[None,:]) < -1e-7 or np.any(y[[105,106,133,135]]>1+1e-7):raise ValueError('Invalid checkpoint state')
        if cv is not None and (min(y[308:acid_offset]) < -1e-7 or y[308]>npv[5]*y[276]+1e-7 or y[309]>y[303]+1e-7):raise ValueError('Invalid checkpoint cargo pools')
        if ckv is not None and (np.min(y[315:318]) < -1e-7 or np.max(y[315:318]-ckv[:3])>1e-7):raise ValueError('Invalid checkpoint PCr pools')
        if rnav is not None and (np.min(y[318:320]) < -1e-7 or np.max(y[318:320])>1+1e-7):raise ValueError('Invalid CaV surface checkpoint')
        if nucv is not None:
            chk=nucleus_checks(y[None,:])
            if biov is not None:chk.pop('ribosome_capacity_excess')
            if max(chk.values())>1e-7:raise ValueError('Invalid nuclear/ribosome checkpoint')
        if cyc is not None:
            chk=checked_cycle(y[None,:],base)
            if biov is not None:
                chk.pop('selected_40S_balance');chk.pop('selected_60S_balance');chk.update(checked_shared(y[None,:],base) if shv is not None else bio_checks(y[None,:],base,nucv,biov))
            if max(chk.values())>1e-7:raise ValueError('Invalid cycle/biogenesis checkpoint')

        if sub is not None and max((checked_recycling(y[None,:],base) if nrv is not None else substrate_checks(y[None,:],base,sub)).values())>1e-7:raise ValueError("Invalid substrate/recycling checkpoint")

    if lrv is not None and max(checked_local(y[None,:],base).values())>1e-7:raise ValueError("Invalid local-ribosome checkpoint")

    def rhs_reference(z,row,target,drive,srow,injection):
        out=np.zeros(199);localpv=pv.copy();localpv[8]=unit_ratio*max(z[139],0)
        # Original biochemical AMPA flux is disabled: membrane current below owns it once.
        norow=row.copy();norow[2]=0.
        damage=min(1.,max(0.,1.-z[180]));support=1-feedback_parameters.spectrin_to_delivery_loss_gain*damage
        out[:134]=compiled_memory(z[:134],p,kv,sv,av,hv,nv,ag0,iv,bn,norow,ep,target,rv,qv,base[83],ev,z[185],en,drive,localpv)
        cs=max(c0+z[0],0);cp=max(c0+z[134],0)
        fs=sat_norm(z[120],en[4],en[0]) if en[12] else 1.
        fp=sat_norm(z[136],en[4],en[0]) if en[12] else 1.
        # Transport ATP sensitivity is independent of pump_feedback switch.
        rates,af,ar=traffic_rates(z[139:143],tr,tf,trev,srow[3]*support,srow[4]*support,sat_norm(z[120],en[4],en[0]),sat_norm(z[136],en[4],en[0]))
        out[139:143]=rates;out[159:161]=[af,ar]
        costs=af*tr.ATP_molecules_per_delivery/(IONS_PER_UM3_PER_UM*vs)
        costp=ar*tr.ATP_molecules_per_delivery/(IONS_PER_UM3_PER_UM*vp)
        out[154:156]=[costs,costp]
        out[0]+=delta_s/(1+e.kappa);out[23]-=delta_s/(1+e.kappa);out[132]+=delta_s
        oxextra=(delta_s+cost0s)*(1-en[8])*drive[0]*sat_norm(z[121],en[6],en[1])
        glyextra=(delta_s+cost0s)*en[8]*drive[1]*sat_norm(z[121],en[7],en[1])
        out[120]+=oxextra+glyextra-costs;out[121]-=oxextra+glyextra-costs
        out[123]+=oxextra;out[124]+=glyextra
        pvec=ppv.copy();pvec[8]=punit*max(z[141],0)
        pm=pmca2_flux(cp,z[135],c0,g.process_gamma_per_s,en[9],fp,pvec)
        nc=(1-en[9])*g.process_gamma_per_s*cp*fp
        _,_,_,other,ox,gly,ak=energy_fluxes(z[136],z[137],z[138],cp,0.,g.process_gamma_per_s,c0,0.,en,drive)
        ox+=(delta_p+cost0p)*(1-en[8])*drive[0]*sat_norm(z[137],en[6],en[1])
        gly+=(delta_p+cost0p)*en[8]*drive[1]*sat_norm(z[137],en[7],en[1])
        cost=other+pm+nc*en[10]+costp;prod=ox+gly
        out[136:139]=[prod-cost+ak,cost-prod-2*ak,ak]
        out[148:154]=[ox,gly,other,pm,nc*en[10],ak]
        on,off=memory_rates(cp,pvec);out[135]=on*(1-z[135])-off*z[135]
        out[144:147]=[pm,nc,pmp0+ncp0]
        cv=couplevec.copy();cv[6]=srow[2]
        dv,ca_current=membrane_rhs(z,profile_p,profile_c,geomvec,cv,fbvec,row[2],row[23],z[106],ep[3:8],ep,injection,c0)
        # SOCE was already included in the somatic Ca ledger by the ER source.
        # Return the SAME charge to voltage; do not add its Ca twice.
        dv[0]+=1000*out[119]*vs/5182.1348283/(g.soma_area_um2/100.)
        out[185:199]=dv
        soma_input=ca_current[0]*5182.1348283/vs
        proc_input=ca_current[1]*5182.1348283/vp
        out[0]+=soma_input/(1+e.kappa);out[22]+=soma_input/(1+e.kappa);out[143]=proc_input
        out[162:185]=local._rhs(z[162:185],max(c0+z[134],0.),row[3],lev,liv,lkv,lsv,lpv,lsyn)
        f=passive_exchange_rate(cs,cp,g);out[147]=f
        out[0]-=f/(vs*(1+e.kappa))
        out[134]=(proc_input+pmp0+ncp0-pm-nc+f/vp)/(1+g.process_kappa)
        out[161]=z[134]
        fa=g.adenylate_diffusion_um2_s*g.connection_factor_um*(z[120:123]-z[136:139])
        out[120:123]-=fa/vs;out[136:139]+=fa/vp;out[156:159]=fa
        return out

    sgamma=e.gamma_per_s;skappa=e.kappa;pgamma=g.process_gamma_per_s;pkappa=g.process_kappa
    ca_diffusion=g.free_ca_diffusion_um2_s*g.connection_factor_um;aden_diffusion=g.adenylate_diffusion_um2_s*g.connection_factor_um
    trvec=np.array([tr.departure_soma_per_s,tr.departure_process_per_s,tr.ATP_molecules_per_delivery])
    ctx=(p,kv,sv,av,hv,nv,ag0,iv,bn,ep,rv,qv,base,ev,en,pv,unit_ratio,ppv,punit,delta_s,cost0s,delta_p,cost0p,pmp0,ncp0,tf,trev,vs,vp,sgamma,skappa,pgamma,pkappa,ca_diffusion,aden_diffusion,trvec,profile_p,profile_c,geomvec,couplevec,fbvec,c0,lev,liv,lkv,lsv,lpv,lsyn)
    neutral=sg[0].copy();neutral[2]=0.;neutral[3]=0.
    basefast=connected_rhs_kernel(base,neutral,core.editing_target(1.,e.normal_editing,kw['mapping_family']),np.ones(3),np.array([0.,1.,.9,1.,1.]),np.zeros(2),ctx)
    baseion=ion_extension(base,basefast,neutral,np.ones(3),np.array([0.,1.,.9,1.,1.]),np.zeros(2),ctx,ionvec,ionbase,ioncv,1.,g.connection_factor_um)
    basemito=mito_extension(base,baseion,ctx,mv,1.)
    basebio,baseups=proteostasis_fluxes(base,basemito,ctx,neutral,prv,1.,1.)
    budgetref=np.zeros((2,4))
    for j in range(2):
        ox=123 if j==0 else 148;other=125 if j==0 else 150;m=239+4*j
        pc=protein_costs(basebio,bpv,j)[0]+sum(work_costs(base,basebio,baseups,bpv,prv,wv,1.,j,en[0],1.,1.)) if prv[0] else sum(protein_costs(basemito,bpv,j))
        if bpv[0] and pc>basemito[other]+1e-10:raise ValueError('Absolute protein cost exceeds allocated basal other-demand; independent ATP production/budget calibration needed')
        h=bpv[6]*basemito[ox]+bpv[5]+bpv[8]*basemito[m+1]
        budgetref[j]=[h,bpv[2]+h/bpv[7]*bpv[4],pc,basemito[other]]
    if nucv is not None and budgetref[0,2]+np.log(2)/nucv[11]*nucv[9]*nucv[10]*4>budgetref[0,3]:raise ValueError('STMN2 allocation exceeds remaining basal ATP demand')
    if cyc is not None:
        cycle_cost=sum(nucv[5]/np.array([414.,nucv[10]])*nucv[6:8])*cyc[4]*cyc[5]
        if budgetref[0,2]+np.log(2)/nucv[11]*nucv[9]*nucv[10]*4+cycle_cost>budgetref[0,3]:raise ValueError('Recycling cost exceeds basal allocation')
    if biov is not None:
        biocost=biov[0]*np.log(2)/(biov[1]*86400)*(sum(biov[11:15])+2*biov[15])
        if budgetref[0,2]+np.log(2)/nucv[11]*nucv[9]*nucv[10]*4+cycle_cost+biocost>budgetref[0,3]:raise ValueError('Biogenesis exceeds basal energy allocation')
    if shv is not None and budgetref[0,2]+np.log(2)/nucv[11]*nucv[9]*nucv[10]*4+cycle_cost+biocost+shref[14]>budgetref[0,3]:raise ValueError('Shared translation exceeds basal energy allocation')
    if isrv is not None and budgetref[0,2]+np.log(2)/nucv[11]*nucv[9]*nucv[10]*4+cycle_cost+biocost+shref[14]+isrref[3]>budgetref[0,3]:raise ValueError('ISR production exceeds basal energy allocation')
    # One explicit equivalence check before integration protects scalar rewrites.
    checkrow=sg[0].copy();checkrow[2]=.003
    probe=base.copy();probe[0]+=.1;probe[134]+=.2;probe[180]*=.8;probe[185]+=5.;probe[192]-=3.
    target=core.editing_target(checkrow[22],e.normal_editing,kw['mapping_family'])
    reference=rhs_reference(probe,checkrow,target,es[0,2:],ss[0],electrical[0,2:])
    fast=connected_rhs_kernel(probe,checkrow,target,es[0,2:],ss[0],electrical[0,2:],ctx)
    kernel_error=float(np.max(abs(reference-fast)/(1+abs(reference))))
    if kernel_error>1e-11:raise ArithmeticError('Compiled/reference RHS mismatch: '+str(kernel_error))
    def rhs(z,row,target,drive,srow,injection,ligand,nuccontrol,cyclecontrol,biocontrol,shcontrol,subcontrol,nrcontrol,isrcontrol,latecontrol,lrcontrol,ltcontrol):
        nuccontrol=nuccontrol.copy()
        nuccontrol[1]*=basal_translation_capacity_scale
        raw_nuccontrol=nuccontrol.copy()
        mcontrol=rmcontrol.copy() if rmv is not None else np.ones(7)
        if rnv is not None:biocontrol,shcontrol,mcontrol=renewal.modulate(z,rnv,rnref,rncontrol,biocontrol,shcontrol,mcontrol)
        if spv is not None:biocontrol,shcontrol=surplus.modulate(z,base,spv,spcontrol,biocontrol,shcontrol)
        if nrv is not None:
            nutrient_z=rp_production.regulatory_view(z) if rppv is not None else z
            nuccontrol,cyclecontrol,biocontrol,shcontrol=nrec_modulate(nutrient_z,nrv,sub,cyc,nuccontrol,cyclecontrol,biocontrol,shcontrol)
        if isrv is not None:nuccontrol=isr_module.general_control(nuccontrol,z,nrv)
        if sub is not None:
            nuccontrol=nuccontrol.copy();nuccontrol[2]*=substrate_gate(z,sub)
        if rsv is not None:
            biocontrol=biocontrol.copy();shcontrol=shcontrol.copy()
            rgate=rna_salvage.gate(z,rsv);biocontrol[0]*=rgate;shcontrol[6]*=rgate
        old=connected_rhs_kernel(z,row,target,drive,srow,injection,ctx)
        ions=ion_extension(z,old,row,drive,srow,injection,ctx,ionvec,ionbase,ioncv,pump_effective_support,g.connection_factor_um)
        matrices=mito_extension(z,ions,ctx,mv,uptake_support)
        recovered=recovery_extension(z,matrices,ctx,ionvec,ionbase,rpv,alpha3_support)
        if rnav is not None:recovered=gene_ion_extension(z,recovered,ctx,ionvec,rpv,alpha3_support,rnav,ligand)
        if pump_response_module is not None:recovered=pump_response_module.apply(z,recovered,ctx,ionvec,pump_response_vector,pump_effective_support)
        if cyc is not None:recovered=kcnq_current_extension(z,recovered,ctx,ionvec,current_kp,kbase)
        if absolute_module is not None:
            recovered=absolute_module.apply(z,recovered,ctx,ionvec,current_kp,kbase,absolute_parameters.g_total_nS,absolute_parameters.replace_slow_fraction,absolute_delta0)
        underlying_other=recovered[125]
        assembly_input,cytoplasmic_z=nrt_module.host_views(z,nrtv,nrtshape,nrt_offset,biov[0]) if nrtv is not None else (z,z)
        assembly_z=individual_rp.assembly_view(assembly_input,biov[0]) if irates is not None else assembly_input
        if biov is not None:recovered,nucdx,cycdx,biodx,aacost=bio_replace(assembly_z,recovered,p,hv,nv,nucv,cyc,biov,f0,en[0],nuccontrol,cyclecontrol,biocontrol,cybase,biobase,row)
        elif cyc is not None:recovered,nucdx,cycdx=replace_cycle_targets(z,recovered,p,hv,nv,nucv,cyc,f0,en[0],nuccontrol,cyclecontrol,cybase,row)
        elif nucv is not None:recovered,nucdx=replace_biosynthesis(z,recovered,p,hv,nucv,f0,en[0],nuccontrol)
        if shv is not None:
            shared_z,shared_control=rp_production.empty_legacy(z,shcontrol) if rppv is not None else (z,shcontrol)
            recovered,cycdx,biodx,shdx,shcomplete,shcost=replace_shared(shared_z,recovered,nucdx,cycdx,biodx,p,nucv,cyc,biov,shv,shref,kp,nv,en[0],nuccontrol,cyclecontrol,biocontrol,shared_control,cybase)
            if rppv is not None:
                if rp_RNA_reference is not None:
                    rpp_effective,rpp_effective_control=nrt_module.referenced_message_controls(z,rppv,rppcontrol,nrtv,nrtshape,nrt_offset,rp_RNA_reference)
                else:
                    rpp_effective,rpp_effective_control=nrt_module.message_controls(z,rppv,rppcontrol,nrtv,nrtshape,nrt_offset) if nrtv is not None else (rppv,rppcontrol)
                rppdx,rppfree,rppcost=rp_production.fluxes(z,nucv,cyc,biov,shref,en[0],nuccontrol,cyclecontrol,biocontrol,shcontrol,cybase,rpp_effective,rpp_effective_control)
                cycdx,biodx,shdx,shcomplete,shcost=rp_production.add_shared(rppdx,rppfree,rppcost,cycdx,biodx,shdx,shcomplete,shcost,shref)
        if irates is not None or nrtv is not None:individual_assembly=biodx[26:28]+biodx[32:34]
        bio,ups=work_loss(z,recovered,ctx,row,prv,wv,proteasome_support,folding_load,processing_efficiency*association_support*initiation_support,ATPase_support)
        out=proteostasis_budget(z,bio,ups,ctx,bpv,budgetref,respiratory_support,drive[0],prv,folding_load)
        worked=work_budget(z,out,bio,ups,bpv,prv,wv,folding_load,en[0],proteasome_support,ATPase_support)
        result=adaptive_extension(z,recycling_extension(z,activity_extension(z,terminal_extension(z,worked,tv,npv,en,vp,bpv,respiratory_reserve,terminal_supply_support),tv,activityv),tv,npv,en,activityv,recv,pathv,terminal_supply_support,glycolysis_support,oxidative_support),tv,npv,en,recv,adv)
        result=cargo_extension(z,result,tv,npv,en,cv,filling_support) if cv is not None else result
        result=uncoating_extension(z,result,tv,npv,en,recv,uv,uncoating_support) if uv is not None else result
        result=creatine_extension(z,result,ckv,creatine_support) if ckv is not None else result
        result=append_surface(z,result,ctx,rnav) if rnav is not None else result
        result=append_nucleus(z,result,nucdx,budgetref[0,3],underlying_other) if nucv is not None else result
        result=append_cycle(z,result,cycdx,cyc,kp,kbase,f0,budgetref[0,3],underlying_other) if cyc is not None else result
        result=append_biogenesis(z,result,biodx,aacost,underlying_other,budgetref[0,3]) if biov is not None else result
        result=append_shared(z,result,shdx,shcomplete,shcost,shv,biov,kp,underlying_other,budgetref[0,3]) if shv is not None else result
        if isrv is not None:
            isrq,isrfree,isrph=isr_module.fluxes(z,isrv,isrref,biov,nucv,cyc,sub,nrv,raw_nuccontrol,cyclecontrol,isrcontrol,nrcontrol,substrate_gate(z,sub))
            result,isrcredit=isr_module.before_substrates(result,isrq,isrfree,isrref,underlying_other,budgetref[0,3])
        result=append_substrates(z,result,sub,subref,biov,nucv,subcontrol) if sub is not None else result
        if isrv is not None:result[493]-=isrq[22]
        if nrv is not None:
            rrtdp=intact_TDP_loss(z,p,nv,prv,wv,en[0],bpv[9],row,proteasome_support,folding_load,processing_efficiency*association_support*initiation_support,ATPase_support)
            rrsource=removal_sources(z,result,rrtdp,ribomass,nucv,nv,shv,kp)
            result=append_recycling(z,result,rrsource,nrv,nrref,sub,nucv,en[0],nrcontrol,subcontrol[0],z[72])
        if isrv is not None:result=isr_module.finish(z,result,isrq,isrph,isrcredit,isrv)
        if latecontrols is not None:result=maturation_module.apply_maturation(result,latecontrol)
        if lrv is not None:result=local_ribo.extension(cytoplasmic_z,result,lrv,lrcontrol,biov[0],vs/vp,base[120],base[136],nucv[8])
        if rnv is not None:result=renewal.delivery(cytoplasmic_z,result,rnv,rncontrol,lrv,lrcontrol,biov[0],vs/vp,base[120],nucv[8])
        if ctv is not None:
            # Keep existing exchange/NDPK/degradation/transport exactly once.
            # Position-resolved branch owns GAP43 load, elongation, charging, RQC.
            coarse=z.copy();coarse[578:583]=0.
            disabled=ltcontrol.copy();disabled[[0,1,4]]=0.
            result=local_mrna.extension(coarse,result,ltv,disabled,lmrv,lmrcontrol,lmrref,biov[0],vs/vp,nucv[8])
        elif lmrv is not None:result=local_mrna.extension(z,result,ltv,ltcontrol,lmrv,lmrcontrol,lmrref,biov[0],vs/vp,nucv[8])
        elif ltv is not None:result=local_translation.extension(z,result,ltv,ltcontrol,biov[0],vs/vp,nucv[8])
        if grv is not None:result=gap43_regrowth.extension(z,result,grv,grcontrol,grref)
        if ctv is not None:result=np.r_[result,np.zeros(20)]
        elif rqv is not None:result=local_rqc.extension(z,result,rqv,rqcontrol,ltv[0],biov[0],vs/vp,nucv[8],ltv[14])
        if rmv is not None:
            maintenance_z=coupled_translation.coarse_charged_view(z) if ctv is not None and len(ctv)>12 and ctv[12] else z
            result=maintenance.extension(maintenance_z,result,rmv,mcontrol,ltv,biov[0],vs/vp,nucv[8],ribomass[13]/4,ribomass[14]/4)
        if rnv is not None:result=renewal.finish(z,result,rnv,rnref)
        if spv is not None:
            cleanup=spcontrol.copy()
            if pqv is not None:cleanup[0]=0.
            result=surplus.finish(z,result,base,spv,cleanup,surplus.RECIPE_VECTOR,biov[0],nucv[8],proteasome_support)
        if pqv is not None:result=precursor_quality.finish(z,result,base,pqv,pqcontrol,spv,spcontrol,surplus.RECIPE_VECTOR,biov[0],nucv[8],proteasome_support)
        if rsv is not None:result=rna_salvage.finish(z,result,rsv,rscontrol,biov,nucv[8],base[120],z[72])
        if ctv is not None:result=coupled_translation.finish(z,result,ctv,ltv,ltcontrol,biov[0],vs/vp,nucv[8],rsv[6])
        if background_ref is not None:result=ion_homeostasis.finish(z,result,ionvec,background_ref,background_parameters.passive_fraction)
        if irates is not None:
            idamage,imark,idispose,iorphan,_=precursor_quality.fluxes(z,base,pqv,pqcontrol,spv,spcontrol,surplus.RECIPE_VECTOR,biov[0],nucv[8],proteasome_support)
            ie=max(z[120],0.)/(nucv[8]+max(z[120],0.))
            result=individual_rp.finish(z,result,biov[0],irates,individual_assembly,idamage,imark,idispose,iorphan,ie,proteasome_support,spv[3])
        if rppv is not None:result=rp_production.finish(z,result,rppdx,biov[0])
        nct_effective=nctcontrol.copy()
        if nrtv is not None:
            _,pore_ratio=nrt_module.common_pore(z[nrt_offset:],z[nct_offset:nct_offset+33],nrtv,nctv,nrtshape)
            nct_effective[1]*=pore_ratio
        if nctv is not None:result=nct_module.finish(z,result,bpv[9],nctv,nct_effective,nct_offset)
        if nrtv is not None:result=nrt_module.finish(z,result,nrtv,nctv,nrtcontrol,nctcontrol,nrtshape,nrt_offset,nct_offset,biov[0],nucv,kp,individual_assembly,rppv if rppv is not None else np.ones((2,80)),rppcontrol)
        if rmkv is not None:result=rmk_module.replace_rates(z,result,rmkv,rmkcontrol,nrtv,nrtcontrol,nrtshape,nrt_offset,rppv,rppcontrol,base[120])
        if chloride_module is not None:result=chloride_module.finish(z,result,ionvec,chloride_ref,chloride_offset)
        if acid_module is not None:result=acid_module.finish(z,result,acid_ref,acid_offset)  # [claude] 2026-11-13: after all PMCA terms, before water scaling
        if absolute_module is not None and absolute_parameters.gate_on_s is not None:
            if absolute_parameters.gate_scheme=='voltage_anchors_v2':
                result=absolute_module.finish_voltage_gates(z,result,current_kp,absolute_parameters.gate_on_s,absolute_parameters.gate_off_s,absolute_parameters.gate_low_voltage_mV,absolute_parameters.gate_high_voltage_mV)
            else:
                result=absolute_module.finish_gates(z,result,current_kp,absolute_parameters.gate_on_s,absolute_parameters.gate_off_s)
        if water_scale is not None:result=result*water_scale
        return result
    ie=ion_extension(probe,fast,checkrow,es[0,2:],ss[0],electrical[0,2:],ctx,ionvec,ionbase,ioncv,pump_effective_support,g.connection_factor_um)
    ir=ion_extension.py_func(probe,reference,checkrow,es[0,2:],ss[0],electrical[0,2:],ctx,ionvec,ionbase,ioncv,pump_effective_support,g.connection_factor_um)
    ion_kernel_error=float(np.max(abs(ie-ir)/(1+abs(ir))))
    if ion_kernel_error>1e-11:raise ArithmeticError('Ion compiled/reference mismatch')

    mi=mito_extension(probe,ie,ctx,mv,uptake_support)
    mr=mito_extension.py_func(probe,ir,ctx,mv,uptake_support)
    mito_error=float(np.max(abs(mi-mr)/(1+abs(mr))))
    if mito_error>1e-11:raise ArithmeticError('Mito compiled/reference mismatch')
    rpcheck=recovery_extension(probe,mi,ctx,ionvec,ionbase,rpv,alpha3_support)
    rpref=recovery_extension.py_func(probe,mr,ctx,ionvec,ionbase,rpv,alpha3_support)
    br=budget_extension(probe,rpcheck,ctx,bpv,budgetref,respiratory_support,1.)
    brr=budget_extension.py_func(probe,rpref,ctx,bpv,budgetref,respiratory_support,1.)
    budget_error=float(np.max(abs(br-brr)/(1+abs(brr))))
    if budget_error>1e-11:raise ArithmeticError('Budget compiled/reference mismatch')
    biou,up=proteostasis_fluxes(probe,rpcheck,ctx,checkrow,prv,proteasome_support,folding_load)
    bior,upr=proteostasis_fluxes.py_func(probe,rpref,ctx,checkrow,prv,proteasome_support,folding_load)
    pu=proteostasis_budget(probe,biou,up,ctx,bpv,budgetref,respiratory_support,1.,prv,folding_load)
    pur=proteostasis_budget.py_func(probe,bior,upr,ctx,bpv,budgetref,respiratory_support,1.,prv,folding_load)
    proto_error=float(np.max(abs(pu-pur)/(1+abs(pur))))
    if proto_error>1e-11:raise ArithmeticError('Proteostasis RHS mismatch')
    workfast=work_budget(probe,pu,biou,up,bpv,prv,wv,folding_load,en[0],proteasome_support,ATPase_support)
    workref=work_budget.py_func(probe,pur,bior,upr,bpv,prv,wv,folding_load,en[0],proteasome_support,ATPase_support)
    workerror=float(np.max(abs(workfast-workref)/(1+abs(workref))))
    if workerror>1e-11:raise ArithmeticError('ATPase pool compiled mismatch')
    ts=[sg[0,:1]];zs=[y[None,:]];nfev=0;events_s=[];events_p=[];release_events=[];rate_diagnostics=[]
    armed=True if checkpoint is None else bool(checkpoint['terminal_armed'])
    for row_index,(row,srow,erow,elecrow) in enumerate(zip(sg,ss,es,electrical)):
        nctcontrol=ncts[row_index,2:].copy() if ncts is not None else np.ones(7)
        nrtcontrol=nrtcontrols[row_index,2:] if nrtcontrols is not None else np.ones(9)
        rmkcontrol=rmkcontrols[row_index,2:] if rmkcontrols is not None else np.ones(160)
        ligand=rnals[row_index,2:] if rnals is not None else np.zeros(2)
        nuccontrol=nucls[row_index,2:] if nucls is not None else np.ones(4)
        if nctv is not None:nctcontrol[0]*=nuccontrol[0]
        cyclecontrol=cycls[row_index,2:] if cycls is not None else np.ones(4)
        rncontrol=rncontrols[row_index,2:] if rncontrols is not None else np.ones(3)
        spcontrol=spcontrols[row_index,2:] if spcontrols is not None else np.ones(2)
        biocontrol=bios[row_index,2:] if bios is not None else np.ones(7)
        shcontrol=shares[row_index,2:] if shares is not None else np.ones(7)
        rppcontrol=rppcontrols[row_index,2:] if rppv is not None else np.ones(160)
        subcontrol=subs[row_index,2:] if subs is not None else np.ones(3)
        nrcontrol=nrcontrols[row_index,2:] if nrcontrols is not None else np.ones(6)
        isrcontrol=isrcontrols[row_index,2:] if isrv is not None else np.array([0.,1.,1.])
        latecontrol=latecontrols[row_index,2:] if latecontrols is not None else np.ones(2)
        lrcontrol=lrcontrols[row_index,2:] if lrcontrols is not None else np.ones(3)
        ltcontrol=ltcontrols[row_index,2:] if ltcontrols is not None else np.ones(6)
        lmrcontrol=lmrcontrols[row_index,2:] if lmrcontrols is not None else np.ones(4)
        grcontrol=grcontrols[row_index,2:] if grcontrols is not None else np.array([0.,1.,1.])
        rqcontrol=rqcontrols[row_index,2:] if rqcontrols is not None else np.array([0.,1.,1.,1.])
        rscontrol=rscontrols[row_index,2:] if rscontrols is not None else np.ones(6)
        pqcontrol=pqcontrols[row_index,2:] if pqcontrols is not None else np.array([0.,0.,0.,1.,1.,1.,1.])
        rmcontrol=rmcontrols[row_index,2:] if rmcontrols is not None else np.array([0.,1.,0.,1.,1.,0.,0.])
        start,end=row[:2];target=core.editing_target(row[22],e.normal_editing,kw['mapping_family'])
        sample_step=output_step_s if segment_output_steps is None else segment_output_steps[row_index]
        solver_step=max_step_s if segment_max_steps is None else segment_max_steps[row_index]
        grid=np.unique(np.r_[np.arange(start+sample_step,end,sample_step),start+np.geomspace(1e-4,min(end-start,3600),70),end])
        cursor=start
        if diagnostic_rates:
            rate_diagnostics.append(dict(time_s=float(start),side='segment_start',segment=int(row_index),
                rate=rhs(y,row,target,erow[2:],srow,elecrow[2:],ligand,nuccontrol,cyclecontrol,biocontrol,shcontrol,subcontrol,nrcontrol,isrcontrol,latecontrol,lrcontrol,ltcontrol)))
        while cursor<end-1e-12:
            # Optional coordinate translation only: the RHS/events ALWAYS see
            # the full physical state. Large history values are not erased.
            # Daily solver intervals bound accumulation/cancellation error.
            anchor=np.where(np.abs(y)>=1e6,y,0.) if center_solver_coordinates else np.zeros_like(y)
            shifted=bool(np.any(anchor))
            def physical(z):return z+anchor if shifted else z
            def depletion(t,zz):
                z=physical(zz)
                return min(np.min(z[120:123]),np.min(z[136:139]),np.min(z[199:203]),np.min(z[219:223]),z[239],z[243],z[247],z[248],z[257],z[258],np.min(z[282:285]))+1e-7
            depletion.terminal=True
            threshold=0. if profile=='Burley' else -10.
            def spike_s(t,z):return physical(z)[185]-threshold
            def spike_p(t,z):return physical(z)[192]-threshold
            spike_s.direction=1;spike_p.direction=1 if armed else -1
            spike_p.terminal=True
            # [codex]2026-11-02: each autonomous, constant-control interval uses
            # a local solver clock. Large absolute times (e.g. after24h) can
            # otherwise collapse tiny LSODA steps in dense-output timestamps.
            # Physical states, absolute output/event times and history persist.
            offset=cursor;chunk_end=min(end,cursor+86400.) if center_solver_coordinates else end;duration=chunk_end-offset
            # Sparse sampling retains only requested points and event states;
            # dense=True retains every internal interpolant in a long stiff run.
            # No physical state, solver tolerance or event is changed.
            requested=grid[(grid>cursor+1e-12)&(grid<=chunk_end)]
            eval_t=None if retain_dense_solver_output else np.unique(np.r_[0.,np.clip(requested-offset,0.,duration),duration])
            fun=lambda t,z:rhs(physical(z),row,target,erow[2:],srow,elecrow[2:],ligand,nuccontrol,cyclecontrol,biocontrol,shcontrol,subcontrol,nrcontrol,isrcontrol,latecontrol,lrcontrol,ltcontrol)
            jac_options={}
            if solver_sparse_jacobian:
                sparse_module=load_module(Path(__file__).resolve().parent.parent/'2026-11-04-joint-baseline-calibration/sparse_solver_codex.py','_stable_sparse_full_jacobian04')
                jac_options['jac']=sparse_module.make_jacobian(fun,atol,rtol,sparse_stats)
            sol=solve_ivp(fun,(0.,duration),y-anchor,method=solver_method,rtol=rtol,atol=atol,max_step=solver_step,events=[depletion,spike_s,spike_p],dense_output=retain_dense_solver_output,t_eval=eval_t,**jac_options)
            nfev+=sol.nfev
            if len(sol.t_events[0]):raise ArithmeticError('Depletion; negative pool rejected')
            if not sol.success:raise RuntimeError(sol.message)
            if retain_dense_solver_output:
                stop=chunk_end if sol.t[-1]==duration else offset+float(sol.t[-1]);sample=grid[(grid>cursor+1e-12)&(grid<=stop)]
                if not len(sample) or sample[-1]!=stop:sample=np.r_[sample,stop]
                values=sol.sol(np.clip(sample-offset,0.,sol.t[-1])).T+anchor;y=sol.y[:,-1].copy()+anchor
            else:
                event_time=float(sol.t_events[2][-1]) if len(sol.t_events[2]) else duration
                stop=chunk_end if event_time==duration else offset+event_time
                keep=sol.t>1e-12;sample=offset+sol.t[keep];values=sol.y[:,keep].T+anchor
                y=(sol.y_events[2][-1] if len(sol.t_events[2]) else sol.y[:,-1]).copy()+anchor
                if not len(sample) or sample[-1]!=stop:sample=np.r_[sample,stop];values=np.vstack([values,y])
            ts.append(sample);zs.append(values)
            events_s.extend(offset+sol.t_events[1])
            if len(sol.t_events[2]):
                if armed:
                    before_release=y.copy()
                    events_p.append(stop);y,released=terminal_jump(y,tv,npv);y=activity_jump(y,tv,activityv);y=recycling_jump(y,released,tv,recv)
                    event=dict(time_s=stop,expected_quanta=released)
                    if cv is not None:
                        y,effective=cargo_jump(before_release,y,released,npv)
                        event.update(ACh_quantal_equivalents=effective,**release_observation(released,effective,cargo_parameters.interpretation))
                    release_events.append(event);ts.append(np.array([stop]));zs.append(y[None,:])
                armed=not armed
            if stop<=cursor:raise ArithmeticError('Hybrid event did not advance')
            cursor=stop
            if solver_observer is not None and stop==chunk_end:
                # Copies only: an external progress/archive callback cannot
                # mutate the solver state or reset its physical history.
                solver_observer(dict(time_s=float(stop),state=y.tolist(),baseline_state=base.tolist(),origin_time_s=origin,fingerprint=fingerprint,terminal_armed=armed,state_schema=schema,diagnostic_interventions=diagnostic_controls),int(nfev))
        if diagnostic_rates:
            rate_diagnostics.append(dict(time_s=float(end),side='segment_end',segment=int(row_index),
                rate=rhs(y,row,target,erow[2:],srow,elecrow[2:],ligand,nuccontrol,cyclecontrol,biocontrol,shcontrol,subcontrol,nrcontrol,isrcontrol,latecontrol,lrcontrol,ltcontrol)))
    t=np.concatenate(ts);z=np.concatenate(zs);err=lambda x:float(np.max(np.abs(x)))
    # Concentration deltas converted to amount/Vref only for mass/energy checks.
    # Kinetics, positivity, output and checkpoint always use physical z.
    dc=(z-base) if water_scale is None else water_module.reference_delta(z,base,water_scale)
    ag=z[:,37:45].sum(1);enzyme=z[:,1:6].sum(1)+z[:,31]+z[:,33];enzyme0=base[1:6].sum()+base[31]+base[33]
    ca_s=(1+e.kappa)*vs*(z[:,0]+z[:,23]-z[:,22])+vs*(z[:,115]-base[115])+z[:,147]+vs*(z[:,239]-base[239])
    ca_p=(1+g.process_kappa)*vp*(z[:,134]-base[134])-vp*(z[:,143]+z[:,146]-z[:,144]-z[:,145])-z[:,147]+vp*(z[:,243]-base[243])
    atps=dc[:,120]-z[:,123]-z[:,124]+z[:,125:129].sum(1)-z[:,129]+z[:,154]+z[:,156]/vs+z[:,203]
    atpp=z[:,285]/vp+dc[:,136]-z[:,148]-z[:,149]+z[:,150:153].sum(1)-z[:,153]+z[:,155]-z[:,156]/vp+z[:,223]
    if ckv is not None:
        # CK adds ATP exactly when PCr decreases; no production counter is credited.
        atps+=dc[:,315];atpp+=dc[:,316]
    adens=dc[:,120:123].sum(1)+z[:,156:159].sum(1)/vs
    adenp=z[:,285:288].sum(1)/vp+dc[:,136:139].sum(1)-z[:,156:159].sum(1)/vp
    checks=dict(soma_Ca_balance_uM=err(ca_s/vs),process_Ca_balance_uM=err(ca_p/vp),global_Ca_balance_uM=err((ca_s+ca_p)/(vs+vp)),
        soma_ATP_balance_uM=err(atps),process_ATP_balance_uM=err(atpp),global_ATP_balance_uM=err((vs*atps+vp*atpp)/(vs+vp)),
        soma_adenylate_balance_uM=err(adens),process_adenylate_balance_uM=err(adenp),pump_count_relative_error=err(z[:,139:143].sum(1)-tr.total_pumps)/max(1,tr.total_pumps),
        enzyme_balance=err(enzyme+z[:,15]-z[:,14]-enzyme0),TDP_lineage_balance=err(z[:,6:10].sum(1)+ag+z[:,19]-z[:,18]-base[6:10].sum()-ag0),
        aggregate_balance=err(ag-ag0-z[:,45]+z[:,46]+z[:,47]+z[:,49]),ER_balance_uM=err(z[:,115]-base[115]-z[:,116]+z[:,117]+z[:,118]),
        soma_PMCA_ATP=err(z[:,126]-z[:,130]),process_PMCA_ATP=err(z[:,151]-z[:,144]),SERCA_ATP=err(z[:,127]-z[:,116]/2),
        gross_soma_clearance=err((1+e.kappa)*z[:,23]-z[:,130]-z[:,131]+z[:,132]),
        iron_mass_balance=err(z[:,core.FE_INDEX].sum(1)+z[:,97]-z[:,96]-base[core.FE_INDEX].sum()),
        memory_editing_bounds=float(max(0,-z[:,[105,106,133,135]].min(),z[:,[105,106,133,135]].max()-1)),minimum_state=minimum(z))
    checks['compiled_rhs_scaled_error']=kernel_error
    checks['local_enzyme_lineage']=err(z[:,162:172].sum(1)+z[:,173]-z[:,172]-base[162:172].sum())
    for j,name in enumerate(('process_TDP','process_spectrin')):
        k=175+5*j;checks[name+'_lineage']=err(z[:,k]+z[:,k+1]+z[:,k+2]-z[:,k+3]-base[k]-base[k+1])
    gi=[186,187,188,189,190,191,193,194,195,196,197,198]
    checks['electrical_gate_bounds']=float(max(0.,-z[:,gi].min(),z[:,gi].max()-1.))
    if rnav is not None:checks['CaV_surface_bounds']=float(max(0.,-z[:,318:320].min(),z[:,318:320].max()-1.))
    if nucv is not None:checks.update(nucleus_checks(z))
    if cyc is not None:
        checks.update(checked_cycle(z,base))
        checks['PKN_full_AF3_balance']=err(z[:,63]+z[:,66]+z[:,76]-z[:,75]-base[63]-base[66])
        checks['PKN207_balance']=err(z[:,65]+z[:,78]-z[:,77]-base[65])
    if biov is not None:
        for key in ('ribosome_capacity_excess','selected_40S_balance','selected_60S_balance'):checks.pop(key,None)
        checks.update(checked_shared(z,base) if shv is not None else bio_checks(z,base,nucv,biov))
        checks['completed_TDP_from_ribosomes']=err((z[:,380]-base[380])*biov[0]/414-(z[:,18]-base[18])*bpv[9])
    if sub is not None:checks.update(checked_recycling(z,base) if nrv is not None else substrate_checks(z,base,sub))
    if lrv is not None:checks.update(checked_local(z,base))
    checks['ion_compiled_rhs_error']=ion_kernel_error
    checks['mito_compiled_rhs_error']=mito_error
    for j,side in enumerate(('soma','process')):
        q=239+4*j
        checks[side+'_matrix_Ca_balance_uM']=err(z[:,q]-base[q]-z[:,q+1]+z[:,q+2])
        if np.min(z[:,q]) < -1e-8 or np.max(z[:,q]) > mv[1]*(1+mv[2])*mv[4]+1e-7:raise ArithmeticError('Matrix Ca bounds')
    for j,label in enumerate(('soma','process')):
        b=199+20*j;vol=vs if j==0 else vp
        # [claude] 2026-11-13: Na entering by electroneutral Na/H exchange has its own ledger (acid block); the term is 0. when the block is off.
        acid_na=z[:,acid_offset+3*j+2] if acid_module is not None else 0.
        checks[label+'_Na_balance_uM']=err(dc[:,b]+z[:,b+5]+3*z[:,b+4]-z[:,b+7]-z[:,b+8]-z[:,b+12]/vol-acid_na)
        checks[label+'_K_balance_uM']=err(dc[:,b+1]+z[:,b+6]-2*z[:,b+4]-z[:,b+9]-z[:,b+13]/vol)
        checks[label+'_Na_cell_bath_uM']=err(dc[:,b]+ionvec[10]*(z[:,b+2]-base[b+2])-z[:,b+10]-z[:,b+12]/vol)
        checks[label+'_K_cell_bath_uM']=err(dc[:,b+1]+ionvec[10]*(z[:,b+3]-base[b+3])-z[:,b+11]-z[:,b+13]/vol)
        checks[label+'_NaK_charge_ATP_uM']=err(z[:,b+14]*10364.26965661773/vol-z[:,b+4])
        if np.min(z[:,b:b+4])<=0:raise ArithmeticError('Nonpositive ionic concentration')
    checks['budget_compiled_rhs_error']=budget_error
    if bpv[0]:
        for j,side in enumerate(('soma','process')):
            b=247+10*j;ox=123 if j==0 else 148
            checks[side+'_proton_budget_uM']=err(z[:,b]-base[b]-z[:,b+2]+bpv[6]*z[:,ox]+z[:,b+5]+z[:,b+6])
            checks[side+'_oxygen_budget_uM']=err(z[:,b+1]-base[b+1]+z[:,b+3]-z[:,b+4])
            if np.min(z[:,b:b+2])<-1e-7:raise ArithmeticError('Negative oxygen/proton reserve')
    checks['proteostasis_compiled_rhs_error']=proto_error
    checks['work_compiled_rhs_error']=workerror
    if prv[0]:
        for j,side in enumerate(('soma','process')):
            b=267+4*j;d=19 if j==0 else 177
            checks[side+'_TDP_clearance_split']=err(z[:,b]+z[:,b+1]-z[:,d])
            checks[side+'_TDP_clearance_energy_split']=err(z[:,b+2]+z[:,b+3]-z[:,255+10*j])
        if z[:,267:275].min()<-1e-7:raise ArithmeticError('Negative proteostasis ledger')
    va=tv[1]
    checks['terminal_ATP_balance_uM']=err(z[:,282]-base[282]-z[:,288]+z[:,289]+z[:,290]+z[:,281]/(602.214076*va)-z[:,291]-z[:,285]/va)
    if ckv is not None:
        checks['terminal_ATP_balance_uM']=err(z[:,282]-base[282]-z[:,288]+z[:,289]+z[:,290]+z[:,281]/(602.214076*va)-z[:,291]-z[:,285]/va+z[:,317]-base[317])
        checks['PCr_pool_bounds']=float(max(0.,-z[:,315:318].min(),(z[:,315:318]-ckv[:3]).max()))
    checks['terminal_adenylate_balance_uM']=err(z[:,282:285].sum(1)-base[282:285].sum()-z[:,285:288].sum(1)/va)
    checks['terminal_Ca_balance_uM']=err((1+tv[5])*z[:,275]-z[:,292]+z[:,293])
    checks['terminal_PMCA_ATP_Ca']=err(z[:,290]-z[:,293])
    checks['terminal_vesicle_balance']=err(npv[5]*z[:,276]+z[:,279]-z[:,280]-npv[5])
    coat=z[:,312] if uv is not None else 0.
    uncoat_cost=z[:,314] if uv is not None else 0.
    if recv[0]:
        checks['terminal_surface_queue']=err(z[:,296]-z[:,279]+z[:,298])
        checks['terminal_internal_queue']=err(z[:,297]+coat+z[:,303]-z[:,298]+z[:,280])
        checks['terminal_whole_vesicle_pool']=err(npv[5]*z[:,276]+z[:,296]+z[:,297]+coat+z[:,303]-npv[5])
        checks['terminal_recycling_cost']=err(z[:,281]-z[:,299]-z[:,300]-uncoat_cost)
        checks['terminal_endocytosis_cost']=err(z[:,299]-npv[15]*recv[4]*z[:,298]*(1.-uv[1]*uv[2] if uv is not None else 1.))
        if adv[0]:
            checks['terminal_internal_alkaline_balance']=err(z[:,297]+coat-z[:,298]+z[:,305])
            checks['terminal_acidified_unready_balance']=err(z[:,303]-z[:,305]+z[:,280])
            checks['terminal_acidification_cost']=err(z[:,306]-npv[15]*(1-recv[4])*adv[4]*z[:,305])
            checks['terminal_priming_cost']=err(z[:,307]-npv[15]*(1-recv[4])*(1-adv[4])*z[:,280])
            checks['terminal_maturation_cost']=err(z[:,300]-z[:,306]-z[:,307])
            checks['terminal_adaptive_nonnegative']=float(max(0.,-z[:,303:acid_offset].min()))
        else:checks['terminal_maturation_cost']=err(z[:,300]-npv[15]*(1-recv[4])*z[:,280])
        checks['terminal_queues_nonnegative']=float(max(0.,-z[:,296:301].min()))
    else:checks['terminal_recycling_cost']=err(z[:,281]-npv[15]*z[:,280])
    if pathv[0]:checks['terminal_pathway_sum']=err(z[:,301]+z[:,302]-z[:,288])
    checks['activity_nonnegative']=float(max(0.,-z[:,294:296].min()))
    checks['terminal_pool_bounds']=float(max(0.,-z[:,282:285].min(),-z[:,276].min(),z[:,276].max()-1.))
    checks['Na_axial_amount']=err(z[:,211]+z[:,231])
    checks['K_axial_amount']=err(z[:,212]+z[:,232])
    if cv is not None:
        checks['ACh_cargo_balance']=err(z[:,308]+z[:,309]-base[308]-z[:,310]+z[:,311])
        checks['ACh_cargo_bounds']=float(max(0.,-z[:,308:acid_offset].min(),(z[:,308]-npv[5]*z[:,276]).max(),(z[:,309]-z[:,303]).max()))
        checks['ACh_release_not_above_membrane']=float(max(0.,(z[:,311]-z[:,279]).max()))
    if uv is not None:
        checks['uncoating_membrane_balance']=err(z[:,312]-uv[1]*z[:,298]+z[:,313])
        checks['uncoating_ATP_cost']=err(z[:,314]-z[:,313]*npv[15]*recv[4]*uv[2])
    if chloride_module is not None:
        cz=z if water_scale is None else base+dc
        cl_checks=chloride_module.checks(cz,base,ionvec,chloride_offset)
        for j in range(2):cl_checks[f'chloride_{j}_negative_pool']=float(max(0.,-z[:,chloride_offset+6*j:chloride_offset+6*j+6].min()))
        checks.update(cl_checks)
    if acid_module is not None:checks.update(acid_module.checks(z,base,acid_ref,acid_offset,t-origin))  # [claude] 2026-11-13
    if not np.isfinite(z).all() or checks['minimum_state'] < -1e-7 or max(v for k,v in checks.items() if k!='minimum_state')>5e-5:raise ArithmeticError(str(checks))
    background_obs=ion_homeostasis.observe(z,ionvec,background_ref,background_parameters.passive_fraction) if background_ref is not None else None
    if water_module is not None:background_obs=water_module.background_observation(background_obs,config['mobile_water_volumes'])
    idx=np.clip(np.searchsorted(sg[:,0],t,side='right')-1,0,len(sg)-1)
    cpdict=dict(time_s=float(t[-1]),state=y.tolist(),baseline_state=base.tolist(),origin_time_s=origin,fingerprint=fingerprint,terminal_armed=armed,state_schema=schema,diagnostic_interventions=diagnostic_controls)
    return dict(time_s=t,state=z,columns=state_columns,
        isr_parameters=asdict(isr_parameters) if isrv is not None else None,isr_segments=isrcontrols,
        maturation_segments=latecontrols,
        local_ribosome_segments=lrcontrols,local_ribosome_observation=local_ribo.observe(z[:,:574],biov[0],vs/vp) if lrv is not None else None,
        rna_salvage_segments=rscontrols,rna_salvage_observation=rna_salvage.observe(z) if rsv is not None else None,
        coupled_translation_observation=coupled_translation.observe(z) if ctv is not None else None,
        coupled_translation_config=config.get('coupled_translation'),coupled_handoff=config.get('coupled_handoff'),
        nuclear_transport_observation=nct_module.observe(z,nctv,nct_offset,nct_parameters) if nctv is not None else None,
        nuclear_transport_segments=ncts,
        RP_mRNA_kinetics_segments=rmkcontrols,RP_mRNA_kinetics_observation=rmk_module.observe(z,rmkv,rmk_parameters,nrtshape,nrt_offset,nrtv,base[120]) if rmkv is not None else None,
        RNA_export_segments=nrtcontrols,RNA_export_observation=nrt_module.observe(z,nrtv,nrtshape,nrt_offset,nct_offset,nrt_parameters) if nrtv is not None else None,
        RP_production_observation=rp_production.observe(z,biov[0],rpp_parameters) if rppv is not None else None,
        RP_production_segments=rppcontrols,
        individual_RP_observation=individual_rp.observe(z,biov[0],individual_rp_parameters) if irates is not None else None,
        ion_background_observation=background_obs,
        passive_chloride_observation=chloride_module.observe(z,ionvec,chloride_ref,chloride_offset) if chloride_module is not None else None,
        acid_extrusion_observation=acid_module.observe(z,acid_ref,acid_offset,acid_parameters,config.get('mobile_water_volumes')) if acid_module is not None else None,  # [claude] 2026-11-13
        absolute_kcnq_observation=absolute_module.observe(z,config,base) if absolute_module is not None else None,
        mobile_water_volume_observation=config.get('mobile_water_volumes'),
        rate_diagnostics=rate_diagnostics,
        solver_coordinate_centering=dict(enabled=center_solver_coordinates,threshold=1e6,maximum_interval_s=86400.,physical_state_and_history_preserved=True),
        precursor_quality_segments=pqcontrols,precursor_quality_observation=precursor_quality.observe(z,base,spv) if pqv is not None else None,
        ribosome_surplus_segments=spcontrols,ribosome_surplus_observation=surplus.observe(z,base) if spv is not None else None,
        ribosome_renewal_segments=rncontrols,ribosome_renewal_observation=renewal.observe(z,base,rnref) if rnv is not None else None,
        ribosome_maintenance_segments=rmcontrols,ribosome_maintenance_observation=maintenance.observe(z,biov[0],vs/vp) if rmv is not None else None,
        local_rqc_segments=rqcontrols,local_rqc_observation=local_rqc.observe(z,biov[0],vs/vp) if rqv is not None else None,
        gap43_regrowth_segments=grcontrols,gap43_regrowth_observation=gap43_regrowth.observe(t,z,grcontrols,grv,grref) if grv is not None else None,
        local_mrna_segments=lmrcontrols,local_mrna_observation=local_mrna.observe(z,biov[0],vs/vp) if lmrv is not None else None,
        local_translation_segments=ltcontrols,local_translation_observation=local_translation.observe(z[:,:605],biov[0],vs/vp) if ltv is not None else None,
        nuclear_recycling_parameters=asdict(nuclear_recycling_parameters) if nrv is not None else None,nuclear_recycling_segments=nrcontrols,
        substrate_parameters=asdict(substrate_parameters) if sub is not None else None,substrate_segments=subs,
        substrate_reference=subref if sub is not None else None,
        shared_translation_segments=shares,shared_translation_parameters=asdict(shared_parameters) if shv is not None else None,shared_reference=shref if shv is not None else None,
        biogenesis_segments=bios,biogenesis_parameters=asdict(biogenesis_parameters) if biov is not None else None,effective_nucleus_vector=nucv,
        nucleus_parameters=asdict(nucleus_parameters) if nucv is not None else None,nuclear_segments=nucls,
        nuclear_TDP_fraction=z[:,320] if nucv is not None else None,
        free_ribosome_slots=z[:,330] if cyc is not None else (1-z[:,326]-z[:,327] if nucv is not None else None),
        free_60S_selected=z[:,331] if cyc is not None else None,
        cycle_segments=cycls,KCNQ_working_ratios=np.array([kcnq_ratios(q[342:362],kbase,kp) for q in z]) if cyc is not None else None,
        rna_calcium_parameters=asdict(rna_calcium_parameters) if rnav is not None else None,rna_ligand_segments=rnals,
        CaV12_surface_fraction=z[:,318:320] if rnav is not None else None,
        creatine_parameters=asdict(creatine_parameters) if ckv is not None else None,creatine_support=creatine_support,
        creatine_activity_multiplier=ck_activity.tolist(),
        diagnostic_interventions=dict(differential_KCNQ_activity_multiplier=float(differential_KCNQ_activity_multiplier),
            etc_backpressure_exponent=float(etc_backpressure_exponent),
            pump_capacity_multiplier=float(pump_capacity_multiplier),
            scope='Explicit sensitivity controls; inherited from new checkpoints unless overridden. KCNQ changes only differential current, with the same charge in K/voltage ledgers. ETC changes feedback shape, not basal or maximal capacity. Not human calibration.'),
        PCr_uM=z[:,315:318] if ckv is not None else None,creatine_uM=ckv[:3]-z[:,315:318] if ckv is not None else None,
        CK_net_ATP_uM=base[315:318]-z[:,315:318] if ckv is not None else None,
        uncoating_parameters=asdict(uncoating_parameters) if uv is not None else None,uncoating_support=uncoating_support,
        coated_quanta=z[:,312] if uv is not None else None,
        cargo_parameters=asdict(cargo_parameters) if cv is not None else None,filling_support=filling_support,
        ready_ACh_quantal_equivalents=z[:,308] if cv is not None else None,
        unready_ACh_quantal_equivalents=z[:,309] if cv is not None else None,
        released_ACh_quantal_equivalents=z[:,311] if cv is not None else None,
        adaptive_parameters=asdict(adp),acidified_unready_quanta=z[:,303],capacity_memory_uM=z[:,304],endocytic_capacity_quanta=np.array([capacity(q,recv,npv,adv) for q in z[:,304]]) if adv[0] else np.full(len(z),recv[2]),phluorin_equivalents=phluorin_signal(z[:,296],z[:,297]+coat) if adv[0] else None,acid_quench_equivalents=z[:,297]+coat if adv[0] else None,
        recycling_parameters=asdict(recp),pathway_parameters=asdict(pathp),surface_queue_quanta=z[:,296],maturation_queue_quanta=z[:,297],glycolytic_ATP_uM=z[:,301],oxidative_ATP_uM=z[:,302],glycolysis_support=glycolysis_support,oxidative_support=oxidative_support,
        filtered_activity_Hz=z[:,294],activity_produced_ATP_uM=z[:,295],activity_parameters=asdict(ap),entry_parameters=asdict(entry_parameters) if entry_parameters else None,original_initiation_support=original_initiation_support,
        terminal_ATP_uM=z[:,282],terminal_Ca_proxy_uM=z[:,275],endplate_voltage_mV=z[:,278],release_events=release_events,
        terminal_exchange_amount=z[:,285:288],terminal_supply_support=terminal_supply_support,respiratory_reserve=respiratory_reserve,association_support=association_support,initiation_support=initiation_support,
        TDP_UPS_cleared=z[:,[267,271]],TDP_nonUPS_cleared=z[:,[268,272]],
        UPS_pool_and_overhead_ATP_uM=z[:,[269,273]],TDP_nonUPS_ATP_uM=z[:,[270,274]],
        proteasome_support=proteasome_support,folding_load=folding_load,processing_efficiency=processing_efficiency,ATPase_support=ATPase_support,
        proton_reserve_relative=z[:,[247,257]]/bpv[1],oxygen_uM=z[:,[248,258]],
        protein_ATP_uM=z[:,[254,264]]+z[:,[255,265]],budget_reference=budgetref,
        matrix_calcium_uM=z[:,[239,243]]/(mv[1]*(1+mv[2])),
        sodium_in_uM=z[:,[199,219]],potassium_in_uM=z[:,[200,220]],sodium_out_uM=z[:,[201,221]],potassium_out_uM=z[:,[202,222]],NaK_ATP_uM=z[:,[203,223]],
        soma_voltage_mV=z[:,185],process_voltage_mV=z[:,192],spike_times_s=dict(soma=events_s,process=events_p) if profile=='Burley' else None,
        voltage_crossing_times_s=dict(soma=events_s,process=events_p),
        event_definition='Upward Vm crossing: Burley0mV, Selvaraj-10mV. Selvaraj spike_times_s=None because crossings are NOT its source peak/dV/dt detector; use the original source detector for frequency validation.',
        process_spectrin_intact_fraction=z[:,180],process_TDP_intact=z[:,175],process_TDP_fragment=z[:,176],
        process_Nav_remaining=1-feedback_parameters.spectrin_to_Nav_loss_gain*np.clip(1-z[:,180],0,1),
        delivery_remaining=1-feedback_parameters.spectrin_to_delivery_loss_gain*np.clip(1-z[:,180],0,1),
        electrical_segments=electrical,window_s=window_s,baseline_state=base,checkpoint=cpdict,config=config,checks=checks,nfev=nfev,state_schema=schema,checkpoint_schema_status=checkpoint_schema_status,
        solver_settings=dict(method=solver_method,rtol=rtol,atol=np.asarray(atol).tolist(),max_step_s=max_step_s,center_coordinates=center_solver_coordinates,sparse_jacobian=solver_sparse_jacobian,sparse_jacobian_stats=sparse_stats),
        segments=sg,spatial_segments=ss,energy_segments=es,calcium_uM=c0+z[:,0],process_calcium_uM=c0+z[:,134],ATP_uM=z[:,120],process_ATP_uM=z[:,136],
        activity=(z[:,28]+z[:,29]+z[:,32])*(1-sg[idx,3]),aggregate_total=ag,nuclear_function_proxy=(z[:,6]+z[:,7])*(z[:,320]/f0 if nucv is not None else 1.),
        pump_surface_soma=z[:,139],pump_surface_process=z[:,141],human_kinetic_fit=False,electrical_feedback=True,
        inherited_source_limitations=old['limitations'],limitations=[
        'Reduced bulk-Ca coupling is distinct from native shell-Ca mode; no claim of matching submembrane microdomains.',
        'Dynamic Nernst Na/K, finite perfused bath and NaK stoichiometry implemented. Resting background fluxes and pump kinetics are assumptions; residual leak/holding charge has no full Cl/osmotic model.',
        'Somatic ER, matrix Ca, lumped proton-motive reserve, O2 and a lumped Ca-cycle energy cost. No resolved matrix Na/NCLX, delta-psi/pH, substrates or individual respiratory complexes. Not a complete mitochondrion.',
        'Spectrin-to-Nav and delivery gains are bounded hypotheses, not observed human transfer functions.',
        'TDP local lineage does not automatically impair nuclear TDP or STMN2. Gross TDP turnover energy is included with uncertain absolute scale; other proteins not individually costed.',
        'Optional three-stage recycling separates alkaline, acidified unready and releasable pools. pHluorin observes surface+alkaline; Ca-capacity scale/memory and ATP allocation hypothetical. No calibrated vesicle pH or human Ca dependence.',
        'Optional finite recycling queues replace old refill and cost; terminal pathways split existing production. No terminal glucose/O2 balance or human kinetic fit.',
        'Optional filtered-AP ATP production is phenomenological; gain=0 preserves prior kinetics. Entry/release competing rates unmeasured in human TDP. No automatic rat-source parameter transfer.',
        'Continuous stiff solver, not an accelerated multiday simulation. No independent human validation.',
        'Source Ca/energy balances cover represented pumps/traffic, not a whole-cell energy budget. Terminal AP-triggered release/replenishment has its own ATP/ADP/AMP and bidirectional amount exchange with process; endplate is passive, not muscle AP/force. ATPase pool energy replaces prior per-cleared-TDP catalytic cost; pool size/activity/ATP sensitivity unknown. Inherited loss rates are not constrained by finite enzyme occupancy.',
        'Closed four-pool PMCA traffic. Exponential delivery, no synthesis/degradation. Actual geometry unmeasured.',
        ],source_spatial_limitations=['Two well-mixed volumes; connector is an effective resistance without its own stored mass.',
        'Free-Ca exchange assumes stationary buffers; mobile Ca-buffer transport and actual branched morphology absent.',
        'Closed pump pool, no new PMCA synthesis/degradation. Transit pumps cannot extrude Ca. Transit is exponential, not deterministic travel.',
        'Matched basal influx and ATP supply are constructed assumptions, not measured homeostasis or fitted rest.',
        'Only soma has the inherited ER/STMN2/iron network. Process has distinct local CAPN/TDP/spectrin states; they are not silently identified with the somatic nuclear pools.',
        'AMPA input is partitioned at fixed voltage. Separate native electrical controls are not continuous spatial spike coupling.',
        'PMCA delivery energy and ATP sensitivity are effective uncertain assumptions, not measured molecular motor stoichiometry.'])
