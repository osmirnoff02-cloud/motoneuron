"""[codex] Versioned neuron facade, through experimental API0.51, 2026-11-03.
Long-duration evidence is scenario-specific: conditional Q3v3 has a documented
72h/4413-state test; this does not validate the historical default or human rest.

Historical models are preserved in place. This facade pins the current backend
and adds qualified measurement outputs. It does not silently change kinetics.
"""
from pathlib import Path
import importlib.util
import sys

B=Path(__file__).resolve().parent
BACKEND=B.parent/'2026-10-24-kcnq2-transport/model_api_codex.py'

# Shared, path-based identity; legacy files elsewhere are still versioned.
sys.path.insert(0,str(B))
from model_contracts_codex import load_module as _module, retire_limitations, revise_limitations
RUNTIME_CONTRACT_REVISION='2026-11-03-review1'

# [codex]2026-11-03. Isolated source-assay probes; no connected ODE changes.
_ion_sources03 = _module(B.parent/'2026-11-03-ion-pump-papers/ion_references_codex.py',
                         '_stable_ion_source_probes03')
for _name in ('jurk2025_ion_reference', 'compare_jurk2025_ion_current',
              'probe_frozen_ion_clamp', 'leroy2015_recruitment_reference',
              'amini2004_pump_reference'):
    globals()[_name] = getattr(_ion_sources03, _name)

_extensions=_module(B/'evidence_extensions_codex.py','_stable_neuron_evidence_extensions')
for _name in ('watt_stoichiometry','calcium_energy_cost','trpm5_endpoint_response',
              'walton_sustained_firing','sxrf_total_element'):
    globals()[_name]=getattr(_extensions,_name)

def backend():
    return _module(BACKEND,'_stable_neuron_backend_20261024')

def __getattr__(name):
    if name.startswith('_'): raise AttributeError(name)
    if name in ('EnergyParameters','ENERGY_RANGES','run_energy_network'):
        return getattr(energy_backend(),name)
    return getattr(backend(),name)

def energy_backend():
    return _module(B.parent/'2026-10-24-atp-calcium-feedback/model_api_codex.py','_stable_energy_backend_20261024')

def run_neuron(*,mode,annotate=True,energy_parameters=None,energy_schedule=None,**kwargs):
    if energy_parameters is None:
        if energy_schedule is not None:raise ValueError('energy_schedule requires explicit EnergyParameters')
        result=backend().run_neuron(mode=mode,**kwargs)
    else:
        result=energy_backend().run_neuron(mode=mode,energy_parameters=energy_parameters,energy_schedule=energy_schedule,**kwargs)
    result['interface_version']='0.11'
    result['backend_path']=str(BACKEND if energy_parameters is None else B.parent/'2026-10-24-atp-calcium-feedback/model_api_codex.py')
    if annotate and mode=='fast':
        result['evidence_checks']=_extensions.annotate_fast_recording(result['electrical'])
    elif annotate:
        # No false PMCA split from long-mode effective clearance. Counter stays
        # on the original absolute clock and therefore survives checkpoints.
        result['evidence_checks']=dict(
            SERCA_ATP_hydrolysis_uM=result['core']['serca_ATP_equivalent_uM'].copy(),
            total_ATP_cost=None,ATP_level=None,upstream_feedback=False,
            limitation='Effective long-mode clearance is not a measured PMCA flux; absolute ATP supply remains unknown.')
    if annotate and energy_parameters is not None:
        r=result['electrical'] if mode=='fast' else result['core']
        result['evidence_checks']['explicit_energy']=dict(
            ATP_uM=r['atp' if mode=='fast' else 'ATP_uM'].copy(),
            ADP_uM=r['adp' if mode=='fast' else 'ADP_uM'].copy(),
            AMP_uM=r['amp' if mode=='fast' else 'AMP_uM'].copy(),
            pump_feedback=energy_parameters.pump_feedback,human_kinetic_fit=False,
            limitation='ATP budget includes explicit Ca pumps and aggregate other demand; no complete cellular energy inventory, free MgATP or thermodynamic pump reversal.')
        if mode=='long':
            result['evidence_checks'].update(ATP_level=r['ATP_uM'].copy(),upstream_feedback=energy_parameters.pump_feedback,
                limitation='Gross PMCA/SERCA and instantaneous NCX-gradient proxy; source pools and kinetic ranges remain hypotheses.')
    return result

# [codex] 2026-10-30. Explicit optional biosynthesis; prior APIs stay unchanged.
_nucleus30=_module(B.parent/'2026-10-30-nucleus-ribosome/nucleus_ribosome_codex.py','_stable_nucleus30')
NucleusRibosomeParameters=_nucleus30.NucleusRibosomeParameters
NUCLEUS_RANGES=_nucleus30.NUCLEUS_RANGES
hachoumi2022_reference=_nucleus30.hachoumi2022_reference
hachoumi_pulse_train=_nucleus30.hachoumi_pulse_train
nuclear_recovery_reference=_nucleus30.nuclear_recovery_reference
translation_source_reference=_nucleus30.translation_source_reference

def run_nucleus_ribosome_neuron(*args,nucleus_parameters=None,**kwargs):
    result=run_rna_calcium_neuron(*args,nucleus_parameters=nucleus_parameters or NucleusRibosomeParameters(),**kwargs)
    result['interface_version']='0.25'
    result['limitations'].append('Nucleus/ribosome: selected TARDBP/STMN2 RNA lifecycle and finite allocatable slots only. Equal compartment losses, effective ATP-supported Ran import, no DNA/chromatin/rRNA biogenesis/40S-60S assembly/collisions. Existing TDP synthesis/ATP is replaced, not duplicated. RNA synthesis/export costs remain within aggregate background; no full transcription energy budget. Somatic STMN2 translation is explicit; local axonal translation is still an inherited proxy. No human kinetic calibration.')
    return result

# [codex] 2026-10-30. Opt-in unified nuclear targets and recycled subunits.
_cycle30=_module(B.parent/'2026-10-30-nuclear-targets-cycle/nuclear_cycle_codex.py','_stable_cycle30')
RibosomeCycleParameters=_cycle30.RibosomeCycleParameters
KCNQCouplingParameters=_cycle30.KCNQCouplingParameters
NuclearKCNQTransportParameters=_cycle30.TransportParameters
CYCLE_RANGES=_cycle30.CYCLE_RANGES

def run_nuclear_cycle_neuron(*args,nucleus_parameters=None,cycle_parameters=None,kcnq_parameters=None,kcnq_coupling_parameters=None,**kwargs):
    r=run_rna_calcium_neuron(*args,nucleus_parameters=nucleus_parameters or NucleusRibosomeParameters(),
        cycle_parameters=cycle_parameters or RibosomeCycleParameters(),kcnq_parameters=kcnq_parameters or NuclearKCNQTransportParameters(),
        kcnq_coupling_parameters=kcnq_coupling_parameters or KCNQCouplingParameters(),**kwargs)
    r['interface_version']='0.26'
    r['limitations'].append('One nuclear TDP signal for TARDBP/STMN2/PKN/KCNQ2. Selected 40S/60S cycle only for TARDBP/STMN2. KCNQ/PKN turnover retains its separate phenomenological rates and aggregate energy allocation. Recycling and eligible-stall rescue are ATP-supported; no rRNA biogenesis, universal rescue, full RQC or nascent-chain energy/mass accounting. Translation energy remains completed-protein equivalent. Differential M-current references the implicit healthy potassium fit. Not a human kinetic calibration.')
    return r

# [codex] 2026-10-30. Opt-in physical selected-pool and nascent energy branch.
_bio30=_module(B.parent/'2026-10-30-ribosome-biogenesis/biogenesis_codex.py','_stable_biogenesis30')
BiogenesisParameters=_bio30.BiogenesisParameters
BIOGENESIS_RANGES=_bio30.BIOGENESIS_RANGES

def run_biogenesis_neuron(*args,biogenesis_parameters=None,**kwargs):
    r=run_nuclear_cycle_neuron(*args,biogenesis_parameters=biogenesis_parameters or BiogenesisParameters(),**kwargs)
    r['interface_version']='0.27'
    r['limitations']=retire_limitations(r,'legacy-retire-1','One nuclear TDP signal')
    r['limitations'].append('Selected somatic biogenesis, four-bin nascent chain accounting and delayed60S/RQC release. Canonical TDP/STMN lengths and one concentration scale derive occupancy from basal gross protein flux. Does not model every rRNA modification, individual RP, full RQC or local axonal biogenesis. RP kit supply is an external aggregate, not self-replication. GTP/ATP costs are high-energy equivalents; amino-acid and nucleotide substrates are not finite pools. KCNQ/PKN retain separate phenomenological synthesis rates and aggregate energy allocation; their nuclear TDP input remains shared. Differential M-current references the implicit healthy potassium fit. Human rates/capacities/costs unknown.')
    return r

# [codex] 2026-10-30. Opt-in selected common ribosomal resource.
_shared30=_module(B.parent/'2026-10-30-shared-translation/shared_translation_codex.py','_stable_shared30')
SharedTranslationParameters=_shared30.SharedTranslationParameters
SHARED_RANGES=_shared30.SHARED_RANGES

def run_shared_translation_neuron(*args,shared_parameters=None,**kwargs):
    r=run_biogenesis_neuron(*args,shared_parameters=shared_parameters or SharedTranslationParameters(),**kwargs)
    r['interface_version']='0.28'
    r['limitations']=retire_limitations(r,'legacy-retire-2','Selected somatic biogenesis')
    r['limitations'].append('Eight selected translation classes share40S/60S and ATP-equivalent budget. RP40/RP60 are representative-chain kit equivalents, not individual RPL/RPS genes. KCNQ/PKN use unknown absolute concentration scales; deltaKCNQ chain length is a cost proxy; PKN207207aa is source-verified. 5S support separate from47S, no complete ActD pharmacology. No finite amino-acid/GTP pools, precursor quality control or human kinetic fit. Differential M-current retains the prior healthy electrical reference. Other proteins remain aggregate demands.')
    return r

def source_survival_contrasts():
    """Safren source-specific Cox contrasts, never death rates per hour."""
    import json
    path=B.parent/'2026-10-24-plan-proteostasis-audit/safren_survival_references_codex.json'
    return json.loads(path.read_text(encoding='utf-8'))

def protein_inventory(gene=None,model_status=None):
    """Reference/implementation inventory, not an automatically active ODE pool.

    Unknown quantities remain null. A human reference entry does not establish
    its presence or abundance in a healthy iPSC motor neuron.
    """
    import json
    path=B.parent/'2026-10-24-neuron-protein-atlas/protein_inventory_codex.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    rows=[]
    for original in data['proteins']:
        r=dict(original)
        if r['gene'] in ('KCNQ2','ATP2B2'):
            r['prior_model_status']=r['model_status']
            r['prior_model_note']=r['note']
            r['model_status']='dynamic_forms'
            r['evidence_cards']=list(dict.fromkeys(r['evidence_cards']+['E081']))
            r['evidence_reports']=list(dict.fromkeys(r['evidence_reports']+['research/2026-10-25-rna-kcnq2-pmca2-model_codex.md']))
            r['note']='[codex] 2026-10-25: optional normalized RNA/protein/surface states and native fast functional probe; unknown kinetics remain sensitivity hypotheses. No measured absolute abundance. Historical implementation status/note are retained in prior_model_status/prior_model_note.'
        rows.append(r)
        if r['gene']=='ATP2B2':
            r['evidence_cards']=list(dict.fromkeys(r['evidence_cards']+['E083','E084']))
            r['evidence_reports']=list(dict.fromkeys(r['evidence_reports']+['research/2026-10-25-pmca-memory-model_codex.md']))
            r['note']+=' Optional PMCA2 memory/support with shared Ca/ATP ledgers added in API0.7; source anchors from heterologous systems, human kinetics remain unknown.'
    if gene is not None:rows=[r for r in rows if r['gene'].casefold()==gene.casefold()]
    if model_status is not None:rows=[r for r in rows if r['model_status']==model_status]
    return dict(metadata=data['metadata'],proteins=rows,changes_simulated_dynamics=False)


# [codex] 2026-10-25: source experiments; no implicit change to human kinetics.
_source25=_module(B.parent/'2026-10-25-watt-liu-kiehn/source_extensions_codex.py','_stable_source_experiments_20261025')
for _name in ('liu_trpm5_response','TRPM5HistoryParameters','TRPM5_SCENARIO_RANGES','run_trpm5_source_protocol','kiehn_firing_evidence','watt_complex_reference'):
    globals()[_name]=getattr(_source25,_name)

# Paired Gerencser observations: no implicit Ca/calpain/TDP parameter transfer.
_fret25=_module(B.parent/'2026-10-25-gerencser-fret/fret_bridge_codex.py','_stable_gerencser_fret_20261025')
for _name in ('gerencser_example','gerencser_spectral_correction','observe_gerencser_calpain','compare_gerencser_fret'):
    globals()[_name]=getattr(_fret25,_name)

# Whole-field optical reference with independent quality masks; legacy API intact.
_field25=_module(B.parent/'2026-10-25-gerencser-field-model/field_observer_codex.py','_stable_gerencser_field_20261025')
for _name in ('gerencser_field_reference','compare_gerencser_field'):
    globals()[_name]=getattr(_field25,_name)

# Explicit RNA/product scenarios; existing run_neuron defaults remain unchanged.
_rna25=_module(B.parent/'2026-10-25-rna-kcnq2-pmca2-model/rna_effectors_codex.py','_stable_rna_effectors_20261025')
for _name in ('RNAEffectorParameters','RNA_EFFECTOR_RANGES','rna_effector_reference',
              'simulate_rna_effectors','run_rna_effector_probe'):
    globals()[_name]=getattr(_rna25,_name)

# Conditional source-informed pump memory. Legacy run_neuron remains unchanged.
_pmca25=_module(B.parent/'2026-10-25-pmca-memory-model/pmca_memory_codex.py','_stable_pmca_memory_20261025')
for _name in ('PMCAMemoryParameters','PMCA_MEMORY_RANGES','pmca_source_reference','simulate_pmca_memory',
              'pmca_density_capacity','run_pmca_memory_probe','run_pmca_memory_network'):
    globals()[_name]=getattr(_pmca25,_name)

# Explicit physical PMCA2 capacity and source-specific measurement observers.
# Legacy run_neuron and all calibrated electrical parameters remain unchanged.
_surface25=_module(B.parent/'2026-10-25-pmca-geometry-model/surface_pmca_codex.py','_stable_pmca_surface_20261025')
for _name in ('PMCASurfaceParameters','PMCA_SURFACE_RANGES','surface_capacity','surface_pmca_flux',
              'map_surface_pmca','run_surface_pmca_probe','run_surface_pmca_network',
              'bautista_buffer_factor','bautista_reference_curve','paired_effective_volume'):
    globals()[_name]=getattr(_surface25,_name)
_bkprotocol25=_module(B.parent/'2026-10-25-pmca-geometry-model/bk_protocol_codex.py','_stable_bk_protocol_20261025')
bk_source_protocol=_bkprotocol25.bk_source_protocol

# Opt-in soma/process Ca/ATP and conserved surface/transit PMCA2 pools.
# This does not replace the fast electrical backend or imply long spatial Vm.
_spatial25=_module(B.parent/'2026-10-25-pmca-spatial-model/spatial_pmca_codex.py','_stable_pmca_spatial_20261025')
for _name in ('SpatialGeometry','PumpTrafficParameters','SPATIAL_RANGES','run_spatial_pmca_network'):
    globals()[_name]=getattr(_spatial25,_name)

# [codex] 2026-10-25: two-section electrical and local injury are opt-in.
# Native DLLs are loaded only when the new electrical function is called.
_electricalgeo25=_module(B.parent/'2026-10-25-spatial-electrical-injury/electrical_geometry_codex.py','_stable_electrical_geometry_20261025')
ElectricalGeometry=_electricalgeo25.ElectricalGeometry
ELECTRICAL_GEOMETRY_RANGES=_electricalgeo25.ELECTRICAL_GEOMETRY_RANGES

def run_spatial_electrical(*args,**kwargs):
    module=_module(B.parent/'2026-10-25-spatial-electrical-injury/electrical_spatial_codex.py','_stable_electrical_spatial_20261025')
    return module.run_spatial_electrical(*args,**kwargs)

_localinjury25=_module(B.parent/'2026-10-25-spatial-electrical-injury/local_injury_codex.py','_stable_local_injury_20261025')
for _name in ('LocalSubstrateParameters','LOCAL_INJURY_RANGES','simulate_local_injury','run_spatial_injury_network',
              'injury_from_electrical','run_spatial_electrical_injury'):
    globals()[_name]=getattr(_localinjury25,_name)

# [codex] 2026-10-26. Opt-in continuous shared-geometry driver.
# Legacy fast/long remain unchanged; bulk-Ca reduction is explicit in new output.
_connected26=_module(B.parent/'2026-10-26-connected-spatial-neuron/connected_parameters_codex.py','_stable_connected_parameters_20261026')
ConnectedParameters=_connected26.ConnectedParameters
CONNECTED_RANGES=_connected26.CONNECTED_RANGES
connected_neuron_inputs=_connected26.inputs
connected_neuron_schedule=_connected26.schedule

def run_connected_neuron(*args,**kwargs):
    import sys
    folder=B.parent/'2026-10-26-connected-spatial-neuron'
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))
    module=_module(folder/'connected_model_codex.py','_stable_connected_neuron_20261026')
    result=module.run_connected_neuron(*args,**kwargs)
    result['interface_version']='0.11'
    return result


# [codex] 2026-10-26. Experimental ion budgets and optional matrix-Ca stores.
# Old run_neuron / run_connected_neuron are unchanged. See the 6/9 new-mode
# frequency result before using these branches as an electrical calibration.
_ionfolder26=B.parent/'2026-10-26-nak-energy-neuron'
if str(_ionfolder26) not in sys.path:sys.path.insert(0,str(_ionfolder26))
_ions26=_module(_ionfolder26/'ion_parameters_codex.py','_stable_ion_parameters26')
_mito26=_module(_ionfolder26/'mito_parameters_codex.py','_stable_mito_parameters26')
IonParameters=_ions26.IonParameters
ION_RANGES=_ions26.ION_RANGES
MitoParameters=_mito26.MitoParameters
MITO_RANGES=_mito26.MITO_RANGES

def run_ion_neuron(*args,**kwargs):
    module=_module(_ionfolder26/'ion_model_codex.py','_stable_ion_neuron26')
    result=module.run_ion_neuron(*args,**kwargs)
    result['interface_version']='0.12'
    return result

def run_mito_ion_neuron(*args,**kwargs):
    module=_module(_ionfolder26/'mito_ion_model_codex.py','_stable_mito_ion_neuron26')
    result=module.run_mito_ion_neuron(*args,**kwargs)
    result['interface_version']='0.12'
    return result

# [codex] 2026-10-26. Source-aware recovery / respiratory budget and NMJ.
# These are optional prototypes, not replacements for frozen human profiles.
_budgetfolder26=B.parent/'2026-10-26-nak-source-calibration'
if str(_budgetfolder26) not in sys.path:sys.path.insert(0,str(_budgetfolder26))
_budgets26=_module(_budgetfolder26/'extension_parameters_codex.py','_stable_recovery_budget26')
_nmj26=_module(_budgetfolder26/'nmj_codex.py','_stable_nmj26')
RecoveryParameters=_budgets26.RecoveryParameters
BudgetParameters=_budgets26.BudgetParameters
NMJParameters=_nmj26.NMJParameters
RECOVERY_RANGES=_budgets26.RECOVERY_RANGES
BUDGET_RANGES=_budgets26.BUDGET_RANGES
NMJ_RANGES=_nmj26.NMJ_RANGES
simulate_nmj=_nmj26.simulate_nmj
_recoveryinputs26=_module(_budgetfolder26/'recovery_inputs_codex.py','_stable_recovery_inputs26')
recovery_neuron_inputs=_recoveryinputs26.recovery_neuron_inputs

def run_recovery_neuron(*args,**kwargs):
    module=_module(_budgetfolder26/'energetic_neuron_codex.py','_stable_recovery_neuron26')
    result=module.run_recovery_neuron(*args,**kwargs)
    result['interface_version']='0.13'
    return result

def sodium_recovery_source_data():
    """Descriptive endpoints with cell coordinates/species; no fitted kinetics."""
    import json
    return json.loads((_budgetfolder26/'workbook_summary_codex.json').read_text(encoding='utf8'))

# [codex] 2026-10-26. Peth/Zahler source assays and conditional ATP -> TDP loss.
_proteofolder26=B.parent/'2026-10-26-proteasome-energy'
if str(_proteofolder26) not in sys.path:sys.path.insert(0,str(_proteofolder26))
_proteop26=_module(_proteofolder26/'proteostasis_parameters_codex.py','_stable_proteostasis_parameters26')
_proteosource26=_module(_proteofolder26/'source_assays_codex.py','_stable_proteostasis_sources26')
ProteostasisParameters=_proteop26.ProteostasisParameters
PROTEOSTASIS_RANGES=_proteop26.PROTEOSTASIS_RANGES
peth_proteasome_reference=_proteop26.peth_reference
zahler_pump_reference=_proteosource26.zahler_assay

def run_proteostasis_neuron(*args,**kwargs):
    module=_module(_proteofolder26/'proteostasis_neuron_codex.py','_stable_proteostasis_neuron26')
    result=module.run_proteostasis_neuron(*args,**kwargs)
    result['interface_version']='0.14'
    return result

# [codex] 2026-10-26. Source-specific unfolding assays and optional ATPase pool.
_workfolder26=B.parent/'2026-10-26-proteasome-ampa'
if str(_workfolder26) not in sys.path:sys.path.insert(0,str(_workfolder26))
_workp26=_module(_workfolder26/'proteasome_work_codex.py','_stable_proteasome_work26')
ProteasomeWorkParameters=_workp26.ProteasomeWorkParameters
PROTEASOME_WORK_RANGES=_workp26.PROTEASOME_WORK_RANGES
henderson_proteasome_reference=_workp26.henderson_reference

def run_proteasome_work_neuron(*args,**kwargs):
    module=_module(_workfolder26/'proteasome_work_neuron_codex.py','_stable_proteasome_work_neuron26')
    result=module.run_proteasome_work_neuron(*args,**kwargs)
    result['interface_version']='0.15'
    return result

# [codex] 2026-10-26. Explicit geometry audit and terminal ATP feedback.
_terminalfolder26=B.parent/'2026-10-26-geometry-mito-nmj'
if str(_terminalfolder26) not in sys.path:sys.path.insert(0,str(_terminalfolder26))
_terminalp26=_module(_terminalfolder26/'energy_terminal_codex.py','_stable_terminal_energy26')
_terminalin26=_module(_terminalfolder26/'inputs_codex.py','_stable_terminal_inputs26')
_recognition26=_module(_terminalfolder26/'proteasome_sources_codex.py','_stable_recognition_sources26')
TerminalEnergyParameters=_terminalp26.TerminalEnergyParameters
TERMINAL_ENERGY_RANGES=_terminalp26.TERMINAL_ENERGY_RANGES
energy_terminal_inputs=_terminalin26.energy_terminal_inputs
hoffman_proteasome_reference=_recognition26.hoffman_reference
thrower_competition_reference=_recognition26.thrower_competition_reference

def run_energy_terminal_neuron(*args,**kwargs):
    module=_module(_terminalfolder26/'energy_terminal_neuron_codex.py','_stable_energy_terminal_neuron26')
    result=module.run_energy_terminal_neuron(*args,**kwargs)
    result['interface_version']='0.16'
    return result

# [codex] 2026-10-26. Opt-in activity-dependent ATP production and source observer.
_activityfolder26=B.parent/'2026-10-26-prakash-activity-atp'
if str(_activityfolder26) not in sys.path:sys.path.insert(0,str(_activityfolder26))
_activityp26=_module(_activityfolder26/'activity_energy_codex.py','_stable_activity_parameters26')
_activityobs26=_module(_activityfolder26/'source_observers_codex.py','_stable_activity_observers26')
ActivityEnergyParameters=_activityp26.ActivityEnergyParameters
ACTIVITY_ENERGY_RANGES=_activityp26.ACTIVITY_ENERGY_RANGES
ProteasomeEntryParameters=_activityobs26.ProteasomeEntryParameters
synatp_observe=_activityobs26.synatp_observe
synatp_invert=_activityobs26.synatp_invert

def run_activity_energy_neuron(*args,**kwargs):
    module=_module(_activityfolder26/'activity_energy_neuron_codex.py','_stable_activity_energy26')
    result=module.run_activity_energy_neuron(*args,**kwargs)
    result['interface_version']='0.17'
    return result

# [codex] 2026-10-26. Optional finite vesicle queues and terminal pathway allocation.
_recyclingfolder26=B.parent/'2026-10-26-takeuchi-recycling-atp'
if str(_recyclingfolder26) not in sys.path:sys.path.insert(0,str(_recyclingfolder26))
_recyclingp26=_module(_recyclingfolder26/'recycling_energy_codex.py','_stable_recycling_parameters26')
RecyclingParameters=_recyclingp26.RecyclingParameters
TerminalPathwayParameters=_recyclingp26.TerminalPathwayParameters
RECYCLING_RANGES=_recyclingp26.RECYCLING_RANGES
TERMINAL_PATHWAY_RANGES=_recyclingp26.PATHWAY_RANGES
takeuchi_reference_constraint=_recyclingp26.takeuchi_constraints

def run_recycling_energy_neuron(*args,**kwargs):
    module=_module(_recyclingfolder26/'recycling_energy_neuron_codex.py','_stable_recycling_energy26')
    result=module.run_recycling_energy_neuron(*args,**kwargs)
    result['interface_version']='0.18'
    return result

# [codex] 2026-10-28. Explicit alkaline/acidified stages and Ca-capacity scenarios.
_adaptivefolder28=B.parent/'2026-10-28-takeuchi-balaji'
if str(_adaptivefolder28) not in sys.path:sys.path.insert(0,str(_adaptivefolder28))
_adaptivep28=_module(_adaptivefolder28/'adaptive_recycling_codex.py','_stable_adaptive_parameters28')
AdaptiveRecyclingParameters=_adaptivep28.AdaptiveRecyclingParameters
ADAPTIVE_RECYCLING_RANGES=_adaptivep28.ADAPTIVE_RANGES
phluorin_recycling_signal=_adaptivep28.phluorin_signal
takeuchi_assay_constraint=_adaptivep28.takeuchi_assay_gate

def run_adaptive_recycling_neuron(*args,**kwargs):
    module=_module(_adaptivefolder28/'adaptive_recycling_neuron_codex.py','_stable_adaptive_recycling28')
    result=module.run_adaptive_recycling_neuron(*args,**kwargs)
    result['interface_version']='0.19'
    return result

# [codex] 2026-10-28. Protocol-aware source observations; same kinetic backend.
_synapticfolder28=B.parent/'2026-10-28-atluri-tabares'
_synapticobs28=_module(_synapticfolder28/'synaptic_observers_codex.py','_stable_synaptic_observers28')
ReleaseObservationParameters=_synapticobs28.ReleaseObservationParameters
SYNAPTIC_OBSERVATION_RANGES=_synapticobs28.OBSERVATION_RANGES
acid_quench_assay=_synapticobs28.acid_quench_assay
transmission_readouts=_synapticobs28.transmission_readouts
atluri_reference_parameters=_synapticobs28.atluri_reference_parameters

def run_synaptic_observation_neuron(*args,observation_parameters=None,**kwargs):
    """API0.20: separate reporter/trafficking/effective-release observations.

    New observation parameters have no biochemical or electrical feedback.
    The 308-state mechanism and its previous limitations are preserved.
    """
    if kwargs.get('cargo_parameters') is not None:
        raise ValueError('Dynamic ACh cargo requires run_cholinergic_neuron; API0.20 assumes the original release observer')
    result=run_adaptive_recycling_neuron(*args,**kwargs)
    if result['adaptive_parameters']['enabled']:
        result['synaptic_observations']=_synapticobs28.annotate_synaptic_result(result,observation_parameters)
    elif observation_parameters is not None:
        raise ValueError('Observation parameters require explicit adaptive recycling')
    else:result['synaptic_observations']=None
    result['interface_version']='0.20'
    return result

# [codex] 2026-10-28. Optional ACh cargo, distinct from membrane turnover.
_cargofolder28=B.parent/'2026-10-28-parsons-grabe'
_cargop28=_module(_cargofolder28/'cholinergic_cargo_codex.py','_stable_cargo28')
_protonp28=_module(_cargofolder28/'organelle_acidity_codex.py','_stable_proton28')
CholinergicCargoParameters=_cargop28.CholinergicCargoParameters
CHOLINERGIC_CARGO_RANGES=_cargop28.CARGO_RANGES
ProtonBalanceParameters=_protonp28.ProtonBalanceParameters
PROTON_SCENARIO_RANGES=_protonp28.PROTON_SCENARIO_RANGES
simulate_proton_balance=_protonp28.simulate_proton_balance
synaptic_source_constraints=_cargop28.source_constraints

def run_cholinergic_neuron(*args,cargo_parameters=None,**kwargs):
    """API0.21: 312 states, ACh filling changes endplate conductance.

    Requires explicitly enabled staged recycling and terminal. The isolated
    Grabe proton assay is separate; it does not silently calibrate live pH.
    """
    p=cargo_parameters or CholinergicCargoParameters()
    result=run_adaptive_recycling_neuron(*args,cargo_parameters=p,**kwargs)
    result['interface_version']='0.21'
    result['limitations'].append('ACh cargo has unknown filling/initial occupancy; no ChAT, choline supply, VAChT stoichiometry or muscle action potentials. Binary/graded cargo distributions are observational alternatives.')
    return result

# [codex] 2026-10-28. Coat removal before acidification; source-aware readouts.
_uncoatingfolder28=B.parent/'2026-10-28-vesicle-supplements'
_uncoatingp28=_module(_uncoatingfolder28/'uncoating_codex.py','_stable_uncoating28')
_quantalp28=_module(_uncoatingfolder28/'quantal_observer_codex.py','_stable_quantal28')
_surfacep28=_module(_uncoatingfolder28/'proton_surface_codex.py','_stable_surface28')
UncoatingParameters=_uncoatingp28.UncoatingParameters
UNCOATING_RANGES=_uncoatingp28.UNCOATING_RANGES
QUANTAL_OBSERVATION_RANGES=_quantalp28.QUANTAL_OBSERVATION_RANGES
observe_quantal_mixture=_quantalp28.observe_quantal_mixture
searl_reference=_quantalp28.searl_reference
PROTON_SURFACE_RANGES=_surfacep28.SURFACE_RANGES
surface_proton_flux=_surfacep28.surface_proton_flux
buffer_identifiability_transform=_surfacep28.buffer_identifiability_transform

def run_uncoating_neuron(*args,uncoating_parameters=None,**kwargs):
    """API0.22: 315 states, membrane -> uncoating -> acidification -> cargo.

    Gate is mechanistic; human rates/route/cost allocation remain scenarios.
    Bulk organellar pH and transmitter distributions are separate assays.
    """
    p=uncoating_parameters or UncoatingParameters()
    result=run_cholinergic_neuron(*args,uncoating_parameters=p,**kwargs)
    result['interface_version']='0.22'
    result['limitations'].append('Coat removal is an ATP-supported queue with unknown human rates; not all recycling routes are clathrin-dependent. Existing retrieval ATP cost is partitioned, not independently calibrated. No full neutral-vesicle reuse or dynamic organellar pH in this driver.')
    return result

# [codex] 2026-10-28. Explicit finite CK pools; all prior entry points preserved.
_donorfolder28=B.parent/'2026-10-28-donor-energy-integration'
_ck28=_module(_donorfolder28/'creatine_codex.py','_stable_creatine28')
_debt28=_module(_donorfolder28/'energy_observer_codex.py','_stable_energy_debt28')
CreatineParameters=_ck28.CreatineParameters
CREATINE_RANGES=_ck28.CREATINE_RANGES
creatine_source_parameters=_ck28.creatine_source_parameters
energy_recovery_observer=_debt28.energy_recovery_observer
voltage_recovery_observer=_debt28.voltage_recovery_observer

def run_creatine_neuron(*args,creatine_parameters=None,**kwargs):
    """API0.23: finite PCr in soma/process/terminal; explicit ATP/ADP ownership.

    Source-model CK parameters are scenarios, not a human kinetic fit. The
    ion-debt observer measures represented Na restoration without ATP debit.
    """
    result=run_uncoating_neuron(*args,creatine_parameters=creatine_parameters or CreatineParameters(),**kwargs)
    result['interface_version']='0.23'
    result['energy_recovery']=energy_recovery_observer(result)
    result['limitations'].append('Finite reversible CK/PCr uses source-model scenarios and matched initial equilibrium; no creatine transport, pH/Mg dependence, mitochondrial CK shuttle or human calibration. No additional ATP production or pump debit; ion debt is diagnostic only.')
    return result

# [codex] 2026-10-29. Conditional RNA-candidate protein mechanisms.
_rnafolder29=B.parent/'2026-10-29-rna-calcium-modules'
_rna29=_module(_rnafolder29/'rna_calcium_codex.py','_stable_rna_calcium29')
RNACalciumParameters=_rna29.RNACalciumParameters
RNA_CALCIUM_RANGES=_rna29.RNA_CALCIUM_RANGES
rna_to_protein_scenario=_rna29.protein_scenario
endocytic_support_scenario=_rna29.endocytic_support_scenario
FXYD7_PRIMARY_RANGES=_rna29.FXYD7_PRIMARY_RANGES

def run_rna_calcium_neuron(*args,rna_calcium_parameters=None,**kwargs):
    """API0.24: 320 states, optional GABA/NMDA/FXYD7 and CaV trafficking.

    Protein factors and transfers remain scenarios; this is not a fit of RNA
    to conductance. Default effects are off; old entry points retain topology.
    """
    result=run_creatine_neuron(*args,rna_calcium_parameters=rna_calcium_parameters or RNACalciumParameters(),**kwargs)
    result['interface_version']='0.24.1'
    result['limitations'].append('GABA reversal is clamped: no Cl/HCO3 concentrations, KCC2/NKCC1 energy or developmental calibration. CaV traffic has finite surface/internal pools but no synthesis or separately resolved traffic ATP. FXYD7 source alternatives are oocyte assays, not human MN; CLU/PRNP have no assigned kinetic law.')
    result['limitations'].append('FXYD7 sources are separate: Meyer2020 human alpha1/beta1/FXYD7 versus Beguin2002/Crambert2004 rat alpha1/beta1 + mouse FXYD7 in Xenopus. The latter K-Hill profiles use -50mV table means flat across voltages, without an intracellular-Na shift. Surface delivery, association and per-complex K effect are separate scenario inputs, not fitted ER/Golgi dynamics. No alpha3 or beta2 effect is inferred; Imax is not copied from whole-oocyte current.')
    return result


# [codex] 2026-10-30. Finite substrate pools.
_substrates30=_module(B.parent/'2026-10-30-translation-substrates/substrates_codex.py','_stable_substrates30')
TranslationSubstrateParameters=_substrates30.TranslationSubstrateParameters
SUBSTRATE_RANGES=_substrates30.SUBSTRATE_RANGES

def run_translation_substrates_neuron(*args,substrate_parameters=None,**kwargs):
    r=run_shared_translation_neuron(*args,substrate_parameters=substrate_parameters or TranslationSubstrateParameters(),**kwargs)
    r['interface_version']='0.29'
    r['limitations']=retire_limitations(r,'legacy-retire-3','Eight selected translation classes')
    r['limitations'].append('Finite pooled AA/charged-tRNA and GTP/GDP feed eight selected somatic translation classes. Charging uses fast AMP-recycling2ATP->2ADP approximation, not explicit PPi/AMP kinetics; NDPK reversible ATP+GDP<->ADP+GTP. Old254/328/493 counters are4-equivalent reference costs, removed and replaced in the actual ATP ledger. No20AA/codon specificity, initiation/termination GTP, Ran/other GTP users, whole-cell guanylate synthesis or full nutrient metabolism. Only orphan amino-acid recovery explicit. Concentrations/transport/enzyme kinetics unknown; basal capacities derived, not measured. RP kit/deltaKCNQ and differential M-current limitations retained.')
    return r


# [codex] 2026-10-30. Selected mature-AA return and nutrient response.
_nrec30=_module(B.parent/'2026-10-30-nuclear-recycling/recycling_codex.py','_stable_nrec30')
NuclearRecyclingParameters=_nrec30.NuclearRecyclingParameters
NUCLEAR_RECYCLING_RANGES=_nrec30.RECYCLING_RANGES

def run_nuclear_recycling_neuron(*args,nuclear_recycling_parameters=None,**kwargs):
    r=run_translation_substrates_neuron(*args,nuclear_recycling_parameters=nuclear_recycling_parameters or NuclearRecyclingParameters(),**kwargs)
    r['interface_version']='0.30'
    r['limitations']=revise_limitations(r,'legacy-revise-4',[('Only orphan amino-acid recovery explicit.', 'Selected intact mature protein and orphan amino-acid recovery explicit; fragments and generic cargo excluded.')])
    r['limitations'].append('537 states. Existing mature loss feeds separate cytosolic peptide, lysosomal peptide and lysosomal AA queues. Pools are soma-volume residue equivalents, not measured lysosomal molarity. Basal recycling replaces external AA supply once. GCN2/eIF2/ATF4/GADD34/mTOR are effective activity proxies, not newly quantified protein pools; individual AA/codon specificity, Ran GTP, all nuclear targets, RNA nucleotide salvage and whole proteome remain unresolved. Existing proteolysis/background energy retained, hydrolysis returns no ATP; separate H-coupled transporter energetics not calibrated.')
    return r


# [codex]2026-10-30. PERK vs GCN2, resource-owned ATF4/GADD34.
_isr30=_module(B.parent/'2026-10-30-isr-selective/selective_isr_codex.py','_stable_isr30')
SelectiveISRParameters=_isr30.SelectiveISRParameters
SELECTIVE_ISR_RANGES=_isr30.ISR_RANGES

def run_selective_isr_neuron(*args,isr_parameters=None,**kwargs):
    r=run_nuclear_recycling_neuron(*args,isr_parameters=isr_parameters or SelectiveISRParameters(),**kwargs)
    r['interface_version']='0.31'
    r['limitations']=retire_limitations(r,'legacy-retire-5','537 states.')
    r['limitations'].append('562 states. PERK is driven by an explicit ER-stress scenario; GCN2 by charged-tRNA deficiency. Two extra selected ATF4/GADD34 translation queues share 40S/60S/AA/GTP/ATP. Old ATF4/GADD34 proxies are observers of these proteins, not duplicate mass. GADD34 functional loss is distinct from loss of production. uORF competition is an effective selection law; uORF peptide/scanning costs, ISR-specific RQC and absolute GADD34 transcription/nucleotide costs unresolved. Relative message drive and all human-MN kinetics remain hypotheses. No direct conversion of ER calcium or unfolded proteins to stress drive; IRE1/ATF6/HRI/PKR absent. Existing mature-AA queues reclaim removed ISR proteins without creating ATP.')
    return r

# Source-aware observation interfaces, no automatic cross-system rate transfer.
_isrobs30=_module(B.parent/'2026-10-30-isr-selective/isr_observers_codex.py','_stable_isrobs30')
observe_ribosome_maturity=_isrobs30.ribosome_maturity_observer
observe_nuclear_concentration_ratio=_isrobs30.nuclear_concentration_ratio
nuclear_fraction_from_concentration_ratio=_isrobs30.fraction_from_concentration_ratio
validate_Ziff_fraction_input=_isrobs30.validate_fraction_calibration_input

# [codex]2026-10-30. Source audit + independent late maturation controls.
_mat30=_module(B.parent/'2026-10-30-fusco-supplement/maturation_codex.py','_stable_mat30')
MATURATION_RANGES=_mat30.MATURATION_RANGES
ribosome_factor_evidence=_mat30.factor_evidence

def run_ribosome_maturation_neuron(segments,*,maturation_schedule=None,**kwargs):
    ctl=_mat30.controls(segments,maturation_schedule)
    r=run_selective_isr_neuron(segments,maturation_schedule=ctl,**kwargs)
    r['interface_version']='0.32'
    r['limitations'].append('Independent late40S and late60S maturation supports multiply existing fluxes. Precursors and mature pools conserve their lineages, previous energy accounting retained. Supports are interventions, not calibrated protein activities. Fusco rat-neurite proteomics are separate observation evidence; selected dynamic ribosomes remain somatic, local transport/maturation and late-maturation ATP/GTP costs unmeasured. No inference of zero activity from nondetection or maturation times from centrifugation durations.')
    return r


# [codex]2026-10-31. Finite local precursor/mature pools, separate from RP exchange.
_lr31=_module(B.parent/'2026-10-31-local-ribosomes/local_ribosomes_codex.py','_stable_lr31')
LocalRibosomeParameters=_lr31.LocalRibosomeParameters
LOCAL_RIBOSOME_RANGES=_lr31.LOCAL_RIBOSOME_RANGES

def run_local_ribosome_neuron(segments,*,local_ribosome_parameters=None,local_ribosome_schedule=None,**kwargs):
    r=run_ribosome_maturation_neuron(segments,local_ribosome_parameters=local_ribosome_parameters or LocalRibosomeParameters(),local_ribosome_schedule=local_ribosome_schedule,**kwargs)
    r['interface_version']='0.33'
    r['limitations'].append('574 states. Precursor transport, local late40/60 maturation and mature return conserve global rRNA/RP lineages. Departure/return ATP charged at source once, with volume conversion. Empty local initial pools are an explicit startup condition; not a measured healthy distribution. Local mature pools are idle capacity observers, not an implemented local translation cycle. No direct local retirement, RP exchange or resolved late maturation GTP cost. Human transport/rate/cost parameters remain scenario ranges; Fusco enrichment is never converted to a kinetic constant.')
    return r

# Source-system observations; never silently calibrate human MN rates.
_org31=_module(B.parent/"2026-10-31-local-ribosomes/organelle_observers_codex.py","_stable_org31")
bmp_assay_timecourse=_org31.bmp_assay_timecourse
bmp_assay_pH=_org31.bmp_assay_pH
organelle_signal=_org31.organelle_signal
organelle_evidence=_org31.evidence


# [codex]2026-11-02. Resource-limited local protein production, one representative cargo.
_lt34=_module(B.parent/'2026-11-02-local-translation/local_translation_codex.py','_stable_local_translation34')
LocalTranslationParameters=_lt34.LocalTranslationParameters
LOCAL_TRANSLATION_RANGES=_lt34.LOCAL_TRANSLATION_RANGES

def run_local_translation_neuron(segments,*,local_translation_parameters=None,local_translation_schedule=None,**kwargs):
    r=run_local_ribosome_neuron(segments,local_translation_parameters=local_translation_parameters or LocalTranslationParameters(),local_translation_schedule=local_translation_schedule,**kwargs)
    r['interface_version']='0.34'
    r['limitations']=retire_limitations(r,'legacy-retire-6','574 states.')
    r['limitations'].append('605 states. One representative local cargo uses the existing local40S/60S pool, finite local AA/charged-tRNA/GTP, explicit NDPK and process ATP. Two-way AA/GTP/GDP exchange conserves soma+process amounts with volume conversion. Four-bin elongation; fast AMP recycling remains lumped, no20-aa specificity. Clamped message-slot availability is not an RNA-transport model. Existing local TDP/STMN2/other sources are unchanged; this additional cargo has no assigned gene or direct disease phenotype. Human rates, pools, costs and initial distribution are scenarios. Local ribosome retirement/RP exchange and complete RQC remain open.')
    r['local_ribosome_observation']['scope']='Free local subunits; bound and posttermination particles counted separately by local_translation_observation.'
    return r


# [codex]2026-11-02. Named GAP43 cargo with source/transit/local mRNA.
_lm35=_module(B.parent/'2026-11-02-local-mrna/local_mrna_codex.py','_stable_local_mrna35')
LocalMRNAParameters=_lm35.LocalMRNAParameters
MRNA_RANGES=_lm35.MRNA_RANGES

def run_local_mrna_neuron(segments,*,local_mrna_parameters=None,local_mrna_schedule=None,local_translation_parameters=None,**kwargs):
    r=run_local_translation_neuron(segments,local_mrna_parameters=local_mrna_parameters or LocalMRNAParameters(),local_mrna_schedule=local_mrna_schedule,local_translation_parameters=local_translation_parameters or _lm35.cargo_parameters(),**kwargs)
    r['interface_version']='0.35'
    r['limitations']=retire_limitations(r,'legacy-retire-7','605 states.')
    r['limitations'].append('617 states. Human GAP43 sequence238aa; mechanism informed by mouse motor-neuron localization experiments, human kinetic rates unknown. Explicit soma/transit/free-local mRNA, message binding/release, free-message decay and ATP transport costs. One ribosome/message, no polysomes or bound-message decay. Stationary selected molecular pools and derived local NDPK capacity; whole-neuron electrical/metabolic equilibrium is not claimed. No GAP43-to-regrowth/NMJ efficacy function. Costs/turnover/localization remain scenarios; sequence length is not a fitted rate.')
    r['local_translation_observation']['scope']='GAP43 selected local production; no calibrated functional effect'
    cols=list(r['columns']);cols[583]='local_GAP43_protein_uM';r['columns']=tuple(cols)
    return r


# [codex]2026-11-02. Conditional GAP43-dependent virtual-lesion assay.
_gr36=_module(B.parent/'2026-11-02-gap43-regrowth/gap43_regrowth_codex.py','_stable_gap43_regrowth36')
GAP43RegrowthParameters=_gr36.GAP43RegrowthParameters
GAP43_REGROWTH_RANGES=_gr36.REGROWTH_RANGES

def gap43_source_evidence():
    """Read source observations with cell/species/assay labels, never human fits."""
    import json
    return json.loads((B.parent/'2026-11-02-gap43-source-audit/observations_codex.json').read_text(encoding='utf8'))

def run_gap43_regrowth_neuron(segments,*,gap43_regrowth_parameters=None,gap43_regrowth_schedule=None,**kwargs):
    r=run_local_mrna_neuron(segments,gap43_regrowth_parameters=gap43_regrowth_parameters or GAP43RegrowthParameters(),gap43_regrowth_schedule=gap43_regrowth_schedule,**kwargs)
    r['interface_version']='0.36'
    r['limitations']=revise_limitations(r,'legacy-revise-8',[('617 states.', 'RNA transport subsystem.'), ('Stationary selected molecular pools and derived local NDPK capacity; whole-neuron electrical/metabolic equilibrium is not claimed.', 'Selected molecular baseline and local NDPK capacity derived by balance; whole-neuron24h resting controls tested for documented default profiles only, not arbitrary parameters or stimulation.'), ('No GAP43-to-regrowth/NMJ efficacy function.', 'Conditional GAP43 virtual-lesion readout follows; no calibrated NMJ efficacy or anatomical regrowth.')])
    r['limitations'].append('620 states. GAP43 amount and process ATP drive a bounded normalized recovery readout, with explicit injury and permissive support. All response parameters are assumptions including zero GAP43 dependence; animal localization/rescue data establish no human dose-response. No physical axon length, new tissue mass, repair ATP cost or feedback to geometry/conduction/NMJ is inferred. This separates a functional hypothesis from mechanistic synthesis/energy accounting.')
    r['local_translation_observation']['scope']='GAP43 local synthesis with conditional recovery readout; no measured human functional calibration'
    return r

# [codex]2026-11-02. Alternative two-stage downstream growth assay;620D core unchanged.
_growth_stages=_module(B.parent/'2026-11-02-axon-growth-supplements/growth_stages_codex.py','_stable_growth_stages_20261102')
GrowthStageParameters=_growth_stages.GrowthStageParameters
GROWTH_STAGE_RANGES=_growth_stages.GROWTH_STAGE_RANGES
axon_growth_source_evidence=_growth_stages.evidence
run_axon_growth_stages=_growth_stages.run_growth_stages


# [codex]2026-11-02. Local effective rescue, split subunits and delayed AA recovery.
_rqc37=_module(B.parent/'2026-11-02-local-rqc/local_rqc_codex.py','_stable_local_rqc37')
LocalRQCParameters=_rqc37.LocalRQCParameters
LOCAL_RQC_RANGES=_rqc37.LOCAL_RQC_RANGES
local_rqc_growth_source_evidence=_rqc37.growth_source_evidence

def run_local_rqc_neuron(segments,*,local_rqc_parameters=None,local_rqc_schedule=None,**kwargs):
    r=run_gap43_regrowth_neuron(segments,local_rqc_parameters=local_rqc_parameters or LocalRQCParameters(),local_rqc_schedule=local_rqc_schedule,**kwargs)
    r['interface_version']='0.37'
    r['limitations'].append('640 states including20 local RQC pools/ledgers. Only eligible stalled GAP43 chains at quarter-length boundaries;default extra stall support0 preserves prior baseline.40S and60S release separately. RNA fate is a scenario;unfinished chains enter proteolysis and existing peptide hydrolysis, never mature protein or calpain TDP fragments. ATP and optional GTP debit only once;no ATP refund. No explicit collisions,ubiquitin,CAT tails,tRNA damage/repair,all-cargo RQC or human rate calibration.')
    r['local_mrna_observation']['bound_local_mRNA_uM']=r['local_rqc_observation']['bound_message_uM']
    r['local_translation_observation']['scope']='GAP43 productive translation;stalled and60S-bound queues and extra costs are in local_rqc_observation'
    return r


# [codex]2026-11-02. Selected RP exchange, finite synthesis and local retirement.
_rm38=_module(B.parent/'2026-11-02-ribosome-maintenance/maintenance_codex.py','_stable_maintenance38')
RibosomeMaintenanceParameters=_rm38.RibosomeMaintenanceParameters
RIBOSOME_MAINTENANCE_RANGES=_rm38.MAINTENANCE_RANGES

def ribosome_maintenance_source_evidence():
    """Source-conditioned observations; no automatic human rate substitution."""
    import json
    return json.loads((B.parent/'2026-11-02-ribosome-maintenance/source_constraints_codex.json').read_text(encoding='utf8'))

def run_ribosome_maintenance_neuron(segments,*,ribosome_maintenance_parameters=None,ribosome_maintenance_schedule=None,**kwargs):
    r=run_local_rqc_neuron(segments,ribosome_maintenance_parameters=ribosome_maintenance_parameters or RibosomeMaintenanceParameters(),ribosome_maintenance_schedule=ribosome_maintenance_schedule,**kwargs)
    r['interface_version']='0.38'
    r['limitations'].append('675 states. Selected RPS26 exchange with finite spare protein and inducible four-bin synthesis sharing existing free40S/60S,charged tRNA,GTP,ATP. Free-subunit impairment is a hypothesis, not measured oxidative chemistry or mRNA-selective fidelity. Additional local retirement competes with repair; whole-RP residues pass lysosomal hydrolysis/export, rRNA disposal remains separate. Old somatic retirement remains its own flux. No human rate calibration, all-RP turnover, NUFIP1 law or nucleotide salvage. Default extra damage,retirement,RP expression and orphan-clearance supports0 preserve the prior resting control; challenged states are not claimed physiological steady state.')
    return r

# [codex]2026-11-02. Continuous maintenance coupled to existing production/delivery.
_rn39=_module(B.parent/'2026-11-02-ribosome-renewal/renewal_codex.py','_stable_renewal39')
RibosomeRenewalParameters=_rn39.RibosomeRenewalParameters
RIBOSOME_RENEWAL_RANGES=_rn39.RENEWAL_RANGES

def ribosome_renewal_source_evidence():
    import json
    return json.loads((B.parent/'2026-11-02-ribosome-renewal/source_constraints_codex.json').read_text(encoding='utf8'))

def run_ribosome_renewal_neuron(segments,*,ribosome_renewal_parameters=None,ribosome_renewal_schedule=None,ribosome_maintenance_schedule=None,**kwargs):
    import numpy as np
    sg=np.asarray(segments,float)
    if ribosome_maintenance_schedule is None:ribosome_maintenance_schedule=np.c_[sg[:,:2],np.ones((len(sg),7))]
    r=run_ribosome_maintenance_neuron(sg,ribosome_renewal_parameters=ribosome_renewal_parameters or RibosomeRenewalParameters(),ribosome_renewal_schedule=ribosome_renewal_schedule,ribosome_maintenance_schedule=ribosome_maintenance_schedule,**kwargs)
    r['interface_version']='0.39'
    r['limitations']=retire_limitations(r,'legacy-retire-9','675 states.')
    r['limitations'].append('678 states/counters including3 bounded phenomenological capacity regulators. Continuous RPS26 maintenance, whole-subunit retirement, existing resource-limited biogenesis and delivery operate fromt=0. No resetting or hidden source of particles/AA/ATP. Control times/gains are unknown human scenarios. Original675D inventories begin a transient; no exact whole-neuron physiological equilibrium or all-RP repair is asserted.47S makes both subunits, so unequal loss may accumulate precursor surplus; source assays remain separately identified.')
    return r


# [codex]2026-11-02. Source-resolved RP cost and surplus precursor surveillance.
_sp40=_module(B.parent/'2026-11-02-ribosome-surplus/surplus_codex.py','_stable_surplus40')
RibosomeSurplusParameters=_sp40.RibosomeSurplusParameters
RIBOSOME_SURPLUS_RANGES=_sp40.SURPLUS_RANGES
observe_RP_heavy_fraction=_sp40.heavy_fraction
ribosomal_pulse_chase_observer=_sp40.pulse_chase_observer

def ribosome_surplus_source_evidence():
    import json
    p=B.parent/'2026-11-02-ribosome-surplus'
    return dict(recipe=_sp40.RECIPE,source_data=json.loads((p/'RPS26_observations_codex.json').read_text()))

def run_ribosome_surplus_neuron(segments,*,ribosome_surplus_parameters=None,ribosome_surplus_schedule=None,biogenesis_parameters=None,shared_parameters=None,**kwargs):
    from dataclasses import replace
    recipe=_sp40.RECIPE_VECTOR
    bio=biogenesis_parameters or BiogenesisParameters()
    shared=shared_parameters or SharedTranslationParameters()
    bio=replace(bio,RP40_kit_ATP_equivalents=4*recipe[0,0],RP60_kit_ATP_equivalents=4*recipe[1,0],
      provenance=bio.provenance+' API0.40 RP kit costs now use selected canonical human sequence recipe, including fusion coproducts; other effective costs unchanged.')
    shared=replace(shared,RP40_representative_length_aa=recipe[0,0]/recipe[0,3],RP60_representative_length_aa=recipe[1,0]/recipe[1,3],
      provenance=shared.provenance+' API0.40 two aggregate RP classes use per-chain averages from human sequence/copy inventory; not80 independent translation equations.')
    r=run_ribosome_renewal_neuron(segments,ribosome_surplus_parameters=ribosome_surplus_parameters or RibosomeSurplusParameters(),ribosome_surplus_schedule=ribosome_surplus_schedule,biogenesis_parameters=bio,shared_parameters=shared,**kwargs)
    r['interface_version']='0.40'
    r['limitations']=retire_limitations(r,'legacy-retire-10','678 states/')
    r['limitations'].append('692 states/counters. Selected80-gene human RP recipe (82 chains incl.P1/P2 duplicates), two aggregate translation classes; full encoded costs and bound masses separated by Ub/FUBI coproduct pool. Surplus clearance and feedback are hypotheses, human rates unknown. Removed proteins enter existing peptide/AA queues, RNA disposal has no nucleotide salvage. Common47S production stays coupled; elevated throughput buffers are allowed. Pulse/chase and stress observations from rat neurons remain separate; no human rate substitution, no full proteome or exact physiological steady state.')
    return r


# [codex] 2026-11-03. Normal waiting, surplus protein, and defective precursors.
_pq41=_module(B.parent/'2026-11-03-precursor-quality/precursor_quality_codex.py','_stable_precursor_quality41')
PrecursorQualityParameters=_pq41.PrecursorQualityParameters
PRECURSOR_QUALITY_RANGES=_pq41.QUALITY_RANGES

def run_precursor_quality_neuron(segments,*,precursor_quality_parameters=None,precursor_quality_schedule=None,**kwargs):
    r=run_ribosome_surplus_neuron(segments,precursor_quality_parameters=precursor_quality_parameters or PrecursorQualityParameters(),precursor_quality_schedule=precursor_quality_schedule,**kwargs)
    r['interface_version']='0.41'
    r['ribosome_surplus_observation']['scope']='API0.41: legacy disposal counters now combine defective disposal and healthy orphan RP clearance; see precursor_quality_observation for separation. Healthy RNA/RNP is not cleared merely for abundance.'
    r['limitations']=retire_limitations(r,'legacy-retire-11','692 states/')
    r['limitations'].append('726 states/counters. Healthy waiting and excess are nonduplicating views of productive pools. Healthy RNA/RNP is not destroyed merely for delay/abundance; only excess free RP has separate orphan clearance. Independently imposed irreversible damage is quarantined from assembly, recognized, then disposed with mass/energy ownership. Baseline has no imposed damage, not a measured zero defect rate. Kinetics and classifier are hypotheses, not human calibration or explicit HUWE1/exosome. No repair/nucleotide salvage. Existing long-time ionic drift remains unresolved.')
    return r


# [codex]2026-11-03. Selected rRNA nucleotide reuse and activation energy.
_rs42=_module(B.parent/'2026-11-03-rna-salvage/rna_salvage_codex.py','_stable_rna_salvage42')
RNASalvageParameters=_rs42.RNASalvageParameters
RNA_SALVAGE_RANGES=_rs42.SALVAGE_RANGES

def run_rna_salvage_neuron(segments,*,rna_salvage_parameters=None,rna_salvage_schedule=None,**kwargs):
    r=run_precursor_quality_neuron(segments,rna_salvage_parameters=rna_salvage_parameters or RNASalvageParameters(),rna_salvage_schedule=rna_salvage_schedule,**kwargs)
    r['interface_version']='0.42'
    r['precursor_quality_observation']['scope']='Healthy/defective precursor handling from API0.41; actual disposed RNA now feeds rna_salvage_observation. No structural classifier or repair kinetics calibrated.'
    r['ribosome_surplus_observation']['scope']='Disposal counters combine defective disposal and healthy orphan RP clearance; RNA fate in rna_salvage_observation, protein fate in existing peptide/AA pools. Historical per-RNA synthesis energy is offset once by legacy_RNA_ATP_replaced counter.'
    r['limitations']=retire_limitations(r,'legacy-retire-12','726 states/')
    r['limitations'].append('745 states/counters. Selected rRNA debris -> finite nucleoside/NMP -> RNA-reserved activated precursor -> actual new rRNA. Four bases pooled; nuclear and lysosomal routes effective; all human rates/yields/pool concentrations assumptions. One ATP ledger replaces prior RNA debit; no free ATP/GTP or AA from RNA. External NMP is an open material boundary, not de-novo synthesis. No whole-cell 72h ionic stability or benefit of faster clearance established.')
    return r


def rna_salvage_source_evidence():
    """Sequence inventory and descriptive human RNA detection, not kinetic fits."""
    import json
    p=B.parent/'2026-11-03-rna-salvage'
    return dict(recipe=_rs42.RECIPE,human_RNA=json.loads((p/'human_salvage_RNA_codex.json').read_text(encoding='utf8')),
                report='research/2026-11-03-rna-salvage_codex.md',human_kinetics_calibrated=False)


# [codex] 2026-11-03. API0.43 selected-cargo branch, not a second global debit.
_ct43=_module(B.parent/'2026-11-03-codon-translation/codon_translation_codex.py','_stable_codon_translation43')
CodonTranslationParameters=_ct43.CodonTranslationParameters
CODON_TRANSLATION_RANGES=_ct43.RANGES
run_detailed_gap43_translation=_ct43.run
translation_resources_from_neuron=_ct43.resources_from_neuron
fibroblast_adenosine_deamination=_ct43.fibroblast_adenosine_deamination

def detailed_translation_source_evidence():
    import json
    p=B.parent/'2026-11-03-codon-translation'
    return dict(recipe=_ct43.RECIPE,sources=json.loads((p/'source_constraints_codex.json').read_text(encoding='utf8')),
                human_kinetics_calibrated=False,whole_neuron_bidirectional_coupling=False)


# [codex]2026-11-03. Alternative mean-field codon translation inside full RHS.
_ct44=_module(B.parent/'2026-11-03-translation-coupling/coupled_translation_codex.py','_stable_coupled_translation44')
CoupledTranslationParameters=_ct44.CoupledTranslationParameters
COUPLED_TRANSLATION_RANGES=_ct44.RANGES

def run_coupled_translation_neuron(segments,*,coupled_translation_parameters=None,**kwargs):
    if any(kwargs.get(k) is not None for k in ('local_rqc_parameters','local_rqc_schedule')):
        raise ValueError('Coupled translation replaces the legacy local RQC branch; use CoupledTranslationParameters.')
    inherited=kwargs.get('local_translation_parameters')
    if inherited is not None and inherited.initiation_GTP_per_chain!=2:
        raise ValueError('Coupled translation currently uses 2 GTP per initiation; the legacy initiation_GTP_per_chain override is unsupported.')
    r=run_rna_salvage_neuron(segments,coupled_translation_parameters=coupled_translation_parameters or CoupledTranslationParameters(),**kwargs)
    r['interface_version']='0.44-coupled-mean-field'
    o=r['coupled_translation_observation'];z=r['state']
    r['local_translation_observation'].update(busy_ribosomes_uM=z[:,_ct44.B].sum(1),
        total_occupied_ribosomes_uM=z[:,_ct44.B].sum(1)+r['local_translation_observation']['posttermination_ribosomes_uM'],
        RQC_bound60S_uM=z[:,_ct44.R].sum(1),scope=o['scope'])
    r['local_mrna_observation'].pop('free_local_mRNA_uM',None)
    r['local_mrna_observation'].pop('bound_local_mRNA_uM',None)
    r['local_mrna_observation'].update(total_local_mRNA_uM=z[:,607],scope='Total GAP43 messages, with polysome mean-field occupancy; decay independent of occupancy is an effective assumption.')
    r['local_rqc_observation']=dict(bound60S_uM=z[:,_ct44.R].sum(1),scope=o['scope'])
    cols=list(r['columns']);cols[607]='GAP43_total_local_mRNA_uM';r['columns']=tuple(cols)
    r['limitations']=retire_limitations(r,'legacy-retire-13',('RNA transport subsystem.','640 states','745 states'))
    r['limitations'].append(o['scope'])
    return r


# [codex]2026-11-03. Optional passive partition of existing calibrated leak.
_ih45=_module(B.parent/'2026-11-03-ion-homeostasis/ion_homeostasis_codex.py','_stable_ion_background45')
IonBackgroundParameters=_ih45.IonBackgroundParameters
ION_BACKGROUND_RANGES=_ih45.RANGES
_ks45=_module(B.parent/'2026-11-03-ion-homeostasis/potassium_sources_codex.py','_potassium_sources45')
perinatal_K_reference=_ks45.perinatal_K_reference
luthi_slow_K_reference=_ks45.luthi_slow_K_reference
luthi_slow_K_protocol=_ks45.luthi_slow_K_protocol
compare_perinatal_K_response=_ks45.compare_perinatal_K_response

def run_ion_homeostasis_neuron(segments,*,ion_background_parameters=None,**kwargs):
    r=run_coupled_translation_neuron(segments,ion_background_parameters=ion_background_parameters or IonBackgroundParameters(),**kwargs)
    r['interface_version']='0.45-passive-ion-background'
    r['limitations'].append(r['ion_background_observation']['scope'])
    return r


# [codex]2026-11-03. API0.46, existing shared translation -> 80 RP inventories.
_irp46=_module(B.parent/'2026-11-03-rp-neuron-coupling/rp_coupling_codex.py','_stable_individual_RP46')
IndividualRPParameters=_irp46.IndividualRPParameters
INDIVIDUAL_RP_RANGES=_irp46.RANGES

def run_individual_ribosomal_neuron(segments,*,individual_rp_parameters=None,**kwargs):
    r=run_coupled_translation_neuron(segments,individual_rp_parameters=individual_rp_parameters or IndividualRPParameters(),**kwargs)
    r['interface_version']='0.46-individual-RP-inventory'
    r['coupled_translation_observation']['counters']=r['state'][:,1589:1600]
    r['limitations'].append(r['individual_RP_observation']['scope'])
    return r


# [codex]2026-11-03. Consistent coarse AA recipe; exact GAP43 unchanged.
def run_balanced_amino_acid_neuron(segments,*,coupled_translation_parameters=None,**kwargs):
    from dataclasses import replace
    p=replace(coupled_translation_parameters or CoupledTranslationParameters(),balanced_coarse_AA=True)
    kwargs.setdefault('retain_dense_solver_output',False)
    if kwargs.get('individual_rp_parameters') is not None:
        kwargs.setdefault('ion_background_parameters',IonBackgroundParameters())
    runner=run_individual_ribosomal_neuron if kwargs.get('individual_rp_parameters') is not None else run_ion_homeostasis_neuron
    r=runner(segments,coupled_translation_parameters=p,**kwargs)
    r['interface_version']='0.47-consistent-coarse-AA'
    r['limitations'].append('Coarse local RP uses the SAME uniform20-AA recipe for consumption and return; limiting charged elongator gates elongation/completion and elongation costs. Legacy loading can still spend initiationGTP and wait. Exact GAP43 codons are unchanged. Uniform coarse composition is an approximation, not measured RPS26 sequence or human AA flux. No inferred human AHP fit.')
    return r


# [codex]2026-11-03. Detailed, finite nuclear transport, compatible with RP inventory.
_nct48=_module(B.parent/'2026-11-03-nuclear-transport/transport_codex.py','_stable_nuclear_transport48')
NuclearTransportParameters=_nct48.NuclearTransportParameters
NUCLEAR_TRANSPORT_RANGES=_nct48.RANGES

def run_detailed_nuclear_transport_neuron(segments,*,nuclear_transport_parameters=None,nuclear_transport_schedule=None,**kwargs):
    r=run_balanced_amino_acid_neuron(segments,nuclear_transport_parameters=nuclear_transport_parameters or NuclearTransportParameters(),nuclear_transport_schedule=nuclear_transport_schedule,**kwargs)
    r['interface_version']='0.48-detailed-nuclear-transport'
    r['coupled_translation_observation']['counters']=r['state'][:,1589:1600]
    r['limitations'].append(r['nuclear_transport_observation']['scope'])
    return r


# [codex]2026-11-03. API0.49: 80 independent somatic RP translation queues.
_rpp49=_module(B.parent/'2026-11-03-rp-production/production_codex.py','_stable_RP_production49')
RPProductionParameters=_rpp49.RPProductionParameters
RP_PRODUCTION_GENES=_rpp49.GENES
RP_PRODUCTION_RANGES=_rpp49.RANGES

def run_ribosomal_production_neuron(segments,*,rp_production_parameters=None,rp_production_schedule=None,individual_rp_parameters=None,**kwargs):
    kwargs.setdefault('retain_dense_solver_output',False)
    kwargs.setdefault('coupled_translation_parameters',CoupledTranslationParameters(balanced_coarse_AA=True))
    # Match API0.47/0.48. Explicit None replays the earlier constant-flux branch.
    kwargs.setdefault('ion_background_parameters',IonBackgroundParameters())
    r=run_individual_ribosomal_neuron(segments,
        rp_production_parameters=rp_production_parameters or RPProductionParameters(),
        rp_production_schedule=rp_production_schedule,
        individual_rp_parameters=individual_rp_parameters or IndividualRPParameters(),**kwargs)
    r['interface_version']='0.49-individual-RP-production'
    scope=r['RP_production_observation']['scope']
    r['limitations']=[x for x in r['limitations'] if x!=r['individual_RP_observation']['scope']]
    r['individual_RP_observation']['scope']=scope
    r['limitations'].append(scope)
    return r


# [codex] 2026-11-03. RNA and pre-ribosome nuclear transport, shared Ran/ATP.
_nrt50=_module(B.parent/'2026-11-03-rna-ribosome-transport/rna_transport_codex.py','_stable_RNA_export50')
RNAExportParameters=_nrt50.RNAExportParameters
RNA_EXPORT_RANGES=_nrt50.RANGES

def run_rna_ribosome_transport_neuron(segments,*,rna_export_parameters=None,rna_export_schedule=None,include_individual_rp=True,**kwargs):
    kwargs.setdefault('nuclear_transport_parameters',NuclearTransportParameters())
    runner=run_ribosomal_production_neuron if include_individual_rp else run_detailed_nuclear_transport_neuron
    r=runner(segments,rna_export_parameters=rna_export_parameters or RNAExportParameters(),rna_export_schedule=rna_export_schedule,**kwargs)
    r['interface_version']='0.50-RNA-ribosome-nuclear-transport'
    # Supersede transport limitations from ancestor APIs without erasing provenance.
    inherited={}
    for key in ('nuclear_transport_observation','RP_production_observation'):
        obs=r.get(key)
        if obs is None:continue
        previous=obs['scope'];inherited[key]=previous
        updated={
            'nuclear_transport_observation': 'Finite KPNB1/KPNA/NTF2/CAS/Ran cycle with shared RNA/RP/pre-ribosome pore competition; passive TDP shuttling and selected import. No XPO1-TDP export, detailed RNA retention, individual NUP turnover, RCC1 sequestration, full transcriptome or human MN calibration.',
            'RP_production_observation': '80 separate somatic RP translation queues use common ribosomes/AA/GTP/ATP. RP message availability is the cytoplasmic RNA pool from API0.50; external message controls regulate its transcription. No codon decoding, amino-acid-specific soma pools or independent bound-RP exchange.',
        }[key]
        r.setdefault('limitation_records',{})[key]={'id':key,'supersedes_api':'0.48/0.49','scope':updated,'inherited_scope':previous}
        obs['scope']=updated
        r['limitations']=[updated if x==previous else x for x in r['limitations']]
    r['RNA_export_observation']['inherited_scopes_before_transport_extension']=inherited
    r['limitations'].append(r['RNA_export_observation']['scope'])
    columns=list(r['columns']);columns[608]='GAP43_mRNA_exported_to_soma_uM';r['columns']=tuple(columns)
    return r


# [codex]2026-11-03. API0.51: evidence-tagged RP-mRNA synthesis and decay.
_rmk51=_module(B.parent/'2026-11-03-rp-mrna-kinetics/kinetics_codex.py','_stable_RP_mRNA_kinetics51')
RPmRNAKineticsParameters=_rmk51.RPmRNAKineticsParameters
RP_MRNA_GENES=_rmk51.GENES
rp_mrna_evidence_parameters=_rmk51.evidence_parameters

def run_rp_mrna_kinetics_neuron(segments,*,rp_mrna_kinetics_parameters=None,rp_mrna_kinetics_schedule=None,**kwargs):
    if kwargs.pop('include_individual_rp',True) is not True:
        raise ValueError('RP-mRNA kinetics require all80 individual RP genes')
    p=rp_mrna_kinetics_parameters or rp_mrna_evidence_parameters()
    r=run_rna_ribosome_transport_neuron(segments,rp_mrna_kinetics_parameters=p,
        rp_mrna_kinetics_schedule=rp_mrna_kinetics_schedule,include_individual_rp=True,**kwargs)
    r['interface_version']='0.51-individual-RP-mRNA-kinetics'
    r['limitations'].append(r['RP_mRNA_kinetics_observation']['scope'])
    r['RNA_export_observation']['RP_mRNA_source_and_decay_overridden_by']=p.evidence_profile
    return r


# [codex] 2026-11-04. U41 source assays and read-only compartment observations.
# These functions do not replace the transport or energy equations in API0.50/0.51.
_u41cal04=_module(B.parent/'2026-11-04-U41-compartment-data/compartment_calibration_codex.py',
                  '_stable_U41_compartment_calibration04')
for _name in ('RNACompartmentRates','u41_reference_panel','u41_reference_rates',
              'predict_fractionated_rna_labeling','u41_labeling_observations','observe_u41_transport',
              'u41_ribosome_reference'):
    globals()[_name]=getattr(_u41cal04,_name)


# [codex] 2026-11-04. Explicit complete composition of the existing API0.51.
_integrated04=_module(B/'integrated_neuron_codex.py','_stable_integrated_neuron04')
integrated_model_components=_integrated04.components

def run_integrated_neuron(segments,*,include_u41_readout=True,include_human_mn_rna_evidence=True,require_validated_baseline=False,control_context=None,**kwargs):
    """Run individual RP + RNA + nuclear transport with existing common resources.

    Preserves API0.51 equations/parameters/checkpoints. Not a human-calibrated
    whole cell; U41 is an optional read-only observation, never an extra ODE.
    require_validated_baseline=True rejects this uncalibrated model before
    calculation; see integrated_baseline_requirements() for unmet conditions.
    """
    if not isinstance(require_validated_baseline,bool):
        raise TypeError('require_validated_baseline must be a boolean')
    if require_validated_baseline:
        raise ValueError('Validated healthy human-MN baseline is not available; inspect integrated_baseline_requirements() before calibration')
    _q1passports04.validate_context(control_context,kwargs)
    _integrated04.validate_options(kwargs)
    result=run_rp_mrna_kinetics_neuron(segments,**kwargs)
    readout=observe_u41_transport(result) if include_u41_readout else None
    result=_integrated04.attach(result,rp_genes=RP_PRODUCTION_GENES,readout=readout)
    if include_human_mn_rna_evidence:
        result['human_MN_RP_mRNA_evidence']=_human_mn04.compare_parameters(
            result['RP_mRNA_kinetics_observation']['parameters'])
    result['energy_baseline_audit']=energy_baseline_audit(result)
    result['control_passport']=_q1passports04.run_passport(result,control_context)
    result['ion_energy_audit']=ion_energy_audit(result)
    result['mn_compartment_energy_observation']=observe_mn_compartment_energy(result)
    return result


# [codex] 2026-11-04. Supplied RNA papers: explicit mouse scenarios and assay readout.
_supplied_rna04=_module(B.parent/'2026-11-04-rna-supplied-papers/supplied_rna_evidence_codex.py',
                        '_stable_supplied_rna_evidence04')
rp_mrna_loedige_panel=_supplied_rna04.reference_panel
rna_supplied_paper_evidence=_supplied_rna04.supplied_paper_evidence
rna_labeling_measurement=_supplied_rna04.labeling_measurement
rna_labeling_mixture_pmf=_supplied_rna04.labeling_mixture_pmf

def rp_mrna_loedige_parameters(profile='PCN_GFP_strict', *, base_parameters=None):
    """Explicit mouse-source transfer scenario; neuron defaults are unchanged."""
    return _supplied_rna04.loedige_parameters(
        rp_mrna_evidence_parameters() if base_parameters is None else base_parameters,profile)

def compare_rp_mrna_loedige_compartments(parameters=None):
    return _supplied_rna04.compare_compartments(
        rp_mrna_evidence_parameters() if parameters is None else parameters)


# [codex] 2026-11-04. Assay-aware human protein turnover, never a silent default fit.
_pt04=_module(B.parent/'2026-11-04-protein-turnover-review/turnover_evidence_codex.py','_stable_human_turnover04')
protein_turnover_evidence=_pt04.evidence
protein_turnover_reference_curve=_pt04.reference_curve

def run_protein_turnover_scenario(segments,*,genes,allow_effective_pool_approximation=False,**kwargs):
    """Explicit dSILAC-to-effective-pool sensitivity scenario on the full neuron.

    Source observations are not assigned to enzyme activity, transport time or
    fragment clearance. Defaults and the established electrical profiles remain.
    """
    options,record=_pt04.transfer_parameters(sys.modules[__name__],kwargs,genes=genes,
        allow_effective_pool_approximation=allow_effective_pool_approximation)
    result=run_integrated_neuron(segments,**options)
    result['protein_turnover_scenario']=record
    result['limitations'].append('Protein turnover source transfer is an explicit effective-pool approximation, not a fitted whole-protein half-life or a shared physiological calibration. CHX and optical-label references remain separate.')
    return result


# [codex] 2026-11-04. Human-MN U41 localization evidence with explicit source QC.
_u41mn04=_module(B.parent/'2026-11-04-U41-Helton-Markmiller/human_mn_reference_codex.py',
                 '_stable_U41_human_mn_reference04')
for _name in ('u41_human_mn_reference','u41_human_mn_target',
              'compare_u41_human_mn_localization','u41_markmiller_model_labels'):
    globals()[_name]=getattr(_u41mn04,_name)


# [codex] 2026-11-04. Human-MN RNA evidence gate; no inferred physical overrides.
_human_mn04=_module(B.parent/'2026-11-04-human-MN-rp-mrna/human_mn_evidence_codex.py',
                   '_stable_human_MN_RP_mRNA_evidence04')
rp_mrna_human_mn_evidence=_human_mn04.evidence

def compare_rp_mrna_human_mn(parameters=None):
    return _human_mn04.compare_parameters(
        rp_mrna_evidence_parameters() if parameters is None else parameters)


# human-pet-axon-20261104_codex
# Assay-specific reference equations; default neuron RHS and energy owners unchanged.
_petaxon04=_module(B.parent/'2026-11-04-human-pet-axon/human_reference_codex.py',
                   '_stable_human_pet_axon_reference04')
for _name in ('human_pet_axon_sources','bishu_leucine_reference','bishu_leucine_flux',
              'bishu_label_observer','howells_hcn_kinetics','howells_hcn_voltage_steps',
              'howells_axon_fit','compare_howells_axon_targets'):
    globals()[_name]=getattr(_petaxon04,_name)


# calpain-baseline-evidence-20261104_codex
# Observation/source constraints only: no physiological default/RHS override.
_calpainbase04=_module(B.parent/'2026-11-04-calpain-baseline-evidence/baseline_observers_codex.py',
                       '_stable_calpain_baseline_observers04')
calpain_baseline_sources=_calpainbase04.calpain_baseline_sources
calpain_form_observer=_calpainbase04.calpain_form_observer
tdp_prm_observer=_calpainbase04.tdp_prm_observer


# U41-supplied-export-20261104_codex
# Source observations only; no RNA/ribosome kinetics or shared energy changes.
_u41sup04=_module(B.parent/'2026-11-04-U41-supplied-export/supplied_export_reference_codex.py',
                  '_stable_U41_supplied_export04')
for _name in ('u41_supplied_export_reference','u41_pir_observations','compare_u41_pir'):
    globals()[_name]=getattr(_u41sup04,_name)


# calpain-fulltext-integration-20261104_codex
# Sequence-specific assay projection only; physical ODE and rates are unchanged.
_calpainfull04=_module(B.parent/'2026-11-04-calpain-fulltext-integration/fragment_observers_codex.py',
                       '_stable_calpain_fulltext_observers04')
calpain_fulltext_evidence=_calpainfull04.calpain_fulltext_evidence
tdp_fragment_assay=_calpainfull04.tdp_fragment_assay


# U41-development-QC-20261104_codex
# Supplied sample evidence only; no kinetic or energy override.
u41_development_sample_qc=_u41sup04.u41_development_sample_qc


# calpain-turnover-review-20261104_codex
# Independent source estimates and conditional loss accounting; no rate override.
_calpainturn04=_module(B.parent/'2026-11-04-calpain-turnover-review/turnover_constraints_codex.py',
                       '_stable_calpain_turnover_review04')
hasan_cortical_turnover=_calpainturn04.hasan_cortical_turnover
tdp_first_order_loss_budget=_calpainturn04.tdp_first_order_loss_budget

# energy-baseline-audit-20261104_codex
# Source/condition checks; never silently replace CK/ADP or temperature.
_energybase04=_module(B.parent/'2026-11-04-resting-state-review/energy_baseline_codex.py',
                     '_stable_energy_baseline_audit04')
for _name in ('ck_equilibrium_reference','conditional_ck_adp','energy_baseline_audit',
              'integrated_baseline_requirements'):
    globals()[_name]=getattr(_energybase04,_name)

# joint-baseline-candidate-20261104_codex
# One explicit factory for the existing integrated entry point. Historical
# defaults are preserved; the candidate is NOT a validated healthy baseline.
_jointbase04=_module(B.parent/'2026-11-04-joint-baseline-calibration/profile_codex.py',
                     '_stable_joint_baseline_candidate04')
ck_conditioned_equilibrium=_jointbase04.ck_equilibrium

def joint_baseline_inputs(profile='Burley',**conditions):
    """Return (integrated kwargs, provenance) for a conditional joint baseline.

    Pass kwargs to run_integrated_neuron and retain the provenance alongside
    its output. No change to the default model or claim of human calibration.
    """
    options,provenance=_jointbase04.joint_inputs(sys.modules[__name__],profile,**conditions)
    options['control_context']=provenance
    return options,provenance

# Q1-control-passports-20261104_codex
# Metadata only: same solver, parameters, state vector and checkpoint ownership.
_q1passports04=_module(B.parent/'2026-11-04-Q1-control-passports/control_passports_codex.py',
                      '_stable_Q1_control_passports04')
neuron_control_passport=_q1passports04.cell_passport
integrated_run_passport=_q1passports04.run_passport

# Q2-ion-energy-20261104_codex: explicit bath scenarios and read-only balances.
_q2ions04=_module(B.parent/'2026-11-04-Q2-ion-energy/ion_energy_codex.py',
                 '_stable_Q2_ion_energy04')
ion_energy_audit=_q2ions04.audit

def ion_energy_inputs(profile='Burley',**conditions):
    """Joint candidate with a named ionic scenario; no new run entry point."""
    return _q2ions04.inputs(sys.modules[__name__],profile,**conditions)

# Q3-conditional-baseline-20261105_codex. Explicit candidates only.
_q3thermo05=_module(B.parent/'2026-11-05-Q3-nuclear-baseline/thermodynamics_codex.py',
                   '_stable_Q3_thermodynamics05')
teague_conditioned_equilibrium=_q3thermo05.equilibrium
_golding05=_module(B.parent/'2026-11-05-Golding-Q3-recovery/source_chemistry_codex.py',
                  '_stable_Golding_source_chemistry05')
golding_conditioned_equilibrium=_golding05.golding_equilibrium
creatine_exchange_observer=_golding05.ck_exchange
_prepared05=_module(B.parent/'2026-11-05-Golding-Q3-recovery/prepared_state_codex.py',
                    '_stable_Q3_prepared_state05')
q3_prepared_checkpoint=_prepared05.prepared_checkpoint
_q3observers05=_module(B.parent/'2026-11-05-Q3-nuclear-baseline/source_observers_codex.py',
                       '_stable_Q3_source_observers05')
straub_ateam_observer=_q3observers05.straub_ateam
_q3base05=_module(B.parent/'2026-11-05-Q3-nuclear-baseline/profile_codex.py',
                 '_stable_Q3_candidate05')
def q3_baseline_inputs(segments,profile='Burley',**conditions):
    """Conditional candidate; retain its provenance and check rest before use."""
    return _q3base05.inputs(sys.modules[__name__],segments,profile,**conditions)

# [codex]2026-11-11. Explicit chemistry/assay benchmarks; no default-rate replacement.
_energy_sources11=_module(B.parent/'2026-11-11-Con2-neurite-energy/source_constraints_codex.py',
                         '_stable_energy_sources11')
energy_source_constraints=_energy_sources11.source_constraints
teague1992_temperature_equilibrium=_energy_sources11.teague1992_temperature
vinnakota_binding_species=_energy_sources11.vinnakota_binding_species
lei_forward_CK_exchange=_energy_sources11.lei_forward_exchange
seahorse_ATP_assay=_energy_sources11.seahorse_ATP_assay

# [codex]2026-11-11. Species-separated pump and compartmental-energy assays.
_ion_sources11=_module(B.parent/'2026-11-11-ion-balance/source_constraints_codex.py','_stable_ion_sources11')
ion_pump_source_constraints=_ion_sources11.source_constraints
hamada_rat_DRG_pump_assay=_ion_sources11.hamada_pump_density
krishnan_pump_activation=_ion_sources11.krishnan_pump_activation
wei_compartment_inhibitor_index=_ion_sources11.wei_inhibitor_index
wei_mouse_reference_indices=_ion_sources11.wei_mouse_reference_indices

# [codex]2026-11-11. Optional passive Cl with finite pools and one leak owner.
_passivecl11=_module(B.parent/'2026-11-11-kcnq-passive/passive_chloride_codex.py','_stable_passive_chloride11')
PassiveChlorideParameters=_passivecl11.PassiveChlorideParameters
# [claude] 2026-11-13. Optional acid-extrusion closure for PMCA counter-protons (plan N05.1.4); scenario, not human calibration.
_acid13_claude=_module(B.parent/'2026-11-13-acid-extrusion/acid_extrusion_claude.py','_stable_acid_extrusion13_claude')
AcidExtrusionParameters=_acid13_claude.AcidExtrusionParameters
_kcnqsources11=_module(B.parent/'2026-11-11-kcnq-passive/source_constraints_codex.py','_stable_kcnq_sources11')
kcnq_passive_source_constraints=_kcnqsources11.source_constraints
gerkau_recovery_comparison=_kcnqsources11.gerkau_recovery_comparison

_current_partition11=_module(B.parent/'2026-11-11-kcnq-passive/current_partition_codex.py','_stable_current_partition11')
neuron_current_partition=_current_partition11.observe

# [codex]2026-11-11. Source-specific acquisition and patch assays; no ODE changes.
_assays_pdf11=_module(B.parent/'2026-11-11-user-pdf-batch/source_assays_codex.py','_stable_user_pdf_assays11')
user_pdf_source_constraints=_assays_pdf11.source_constraints
patch_chloride_context=_assays_pdf11.patch_chloride_context
eberhardt_nociceptor_gating=_assays_pdf11.eberhardt_nociceptor_gating
sample_gameiro_fluorescence=_assays_pdf11.sample_gameiro_fluorescence
gameiro_event_acceptance=_assays_pdf11.gameiro_event_acceptance
normalized_patch_current=_assays_pdf11.normalized_patch_current

# [codex]2026-11-11. Maturity/MEA assays and one-way KNa candidate, not a default current.
_maturity11=_module(B.parent/'2026-11-11-neuronal-maturity-assays/source_assays_codex.py','_stable_neuronal_maturity11')
neuronal_maturity_source_constraints=_maturity11.source_constraints
bardy_sweep_classification=_maturity11.bardy_sweep_classification
jurk2025_maturity_reference=_maturity11.jurk2025_maturity_reference
compare_jurk2025_properties=_maturity11.compare_jurk2025_properties
fofie_retigabine_reference=_maturity11.fofie_retigabine_reference
mea_well_observer=_maturity11.mea_well_observer
KNaCandidateParameters=_maturity11.KNaCandidateParameters
replay_kna_candidate=_maturity11.replay_kna_candidate

# human-electrical-numbers-20261111_codex
# Published assay constraints only; no replacement of Burley/Selvaraj or neuron RHS.
_humanelec11=_module(B.parent/'2026-11-11-human-electrical-numbers/electrical_references_codex.py',
                    '_stable_human_electrical_references11')
for _name in ('human_electrical_source_constraints','measured_medium_reference',
              'medium_nernst_reference','halliwell_ligand_reference',
              'halliwell_normalized_response','lyckenvik_network_reference',
              'kleckner_pulse_reference'):
    globals()[_name]=getattr(_humanelec11,_name)

# mn-pump-source-assays-20261111_codex
_mnpump11=_module(B.parent/'2026-11-11-mn-pump-loads/source_assays_codex.py',
                 '_stable_mn_pump_source_assays11')
for _name in ('mn_pump_source_constraints','safronov1996_open_probability',
              'safronov1996_candidate','ck_inferred_free_nucleotides',
              'simkin2021_postburst_observer','torregrosa2025_energy_reference',
              'compare_torregrosa_pathway_fraction'):
    globals()[_name]=getattr(_mnpump11,_name)

# potassium-flux-audit-20261112_codex
# Verified read-only observers: positive K flux enters cytosol; not intervention models.
_kledger12=_module(B.parent/'2026-11-12-potassium-flux-audit/ledger_audit_codex.py',
                   '_stable_potassium_ledger12')
potassium_flux_ledger=_kledger12.observe
_kinstant12=_module(B.parent/'2026-11-12-potassium-flux-audit/instantaneous_codex.py',
                    '_stable_potassium_instantaneous12')
potassium_flux_partition=_kinstant12.observe
_kassays12=_module(B.parent/'2026-11-12-potassium-flux-audit/source_assays_codex.py',
                  '_stable_potassium_assays12')
potassium_recovery_source_constraints=_kassays12.source_constraints
paired_voltage_resistance_recovery=_kassays12.recovery_assay

# absolute-kcnq-assays-20261112_codex
# Independent clamp/rest constraints. Not a calibrated human-MN replacement.
_absK12=_module(B.parent/'2026-11-12-absolute-kcnq/absolute_kcnq_codex.py',
                '_stable_absolute_kcnq12')
absolute_kcnq_source_constraints=_absK12.source_constraints
absolute_kcnq_tail_current=_absK12.tail_current
absolute_kcnq_replacement_current=_absK12.replacement_current
passive_NaK_rest_closure=_absK12.passive_pump_closure
process_volume_conversion=_absK12.geometry_conversion

# mobile-water-and-absolute-kcnq-20261112_codex
_water12=_module(B.parent/'2026-11-12-ion-water-kcnq/compartment_volumes_codex.py','_stable_mobile_water12')
_absCoupled12=_module(B.parent/'2026-11-12-ion-water-kcnq/absolute_current_coupling_codex.py','_stable_absolute_coupling12')
WaterVolumeParameters=_water12.WaterVolumeParameters
_pump13=_module(B.parent/'2026-11-13-pump-response-test/pump_response_codex.py','_stable_pump_response13')
PumpResponseParameters=_pump13.PumpResponseParameters
observe_pump_response=_pump13.observe
AbsoluteKCNQParameters=_absCoupled12.AbsoluteKCNQParameters
ion_water_source_constraints=_water12.source_constraints

# joint-kcnq-source-constraints-20261112_codex
# Source-specific measurements/conditions, not imported human-MN kinetic rates.
_jointK12=_module(B.parent/'2026-11-12-kcnq-joint-calibration/source_constraints_codex.py',
                  '_stable_joint_kcnq_constraints12')
joint_kcnq_source_constraints=_jointK12.source_constraints
yang_named_salt_nernst_reference=_jointK12.yang_named_salt_nernst_reference
telezhkin_reference_comparison=_jointK12.telezhkin_reference_comparison


# [codex]2026-11-12. Primary KCNQ/pump assays; no mutation of ODE defaults.
_primaryKpump12 = _module(B.parent/'2026-11-12-kcnq-pump-primary/assay_constraints_codex.py',
                         '_stable_primary_kcnq_pump12')
primary_pump_kcnq_sources = _primaryKpump12.primary_pump_kcnq_sources
smith_kcnq_activation = _primaryKpump12.smith_kcnq_activation
pump_current_to_fluxes = _primaryKpump12.pump_current_to_fluxes
typical_biophysics_table_audit = _primaryKpump12.typical_table_audit
ma2023_gating_reference = _primaryKpump12.ma2023_gating_reference

# [codex] 2026-11-12. Explicit clamp kinetics and mouse assay references.
# These exports do not replace integrated-neuron gating or human calibration.
_kkinetics12 = _module(B.parent/'2026-11-12-kcnq-kinetics-search/kinetic_assays_codex.py',
                       '_stable_kcnq_kinetics12')
for _name in ('kcnq_kinetic_sources','cyl3059_deactivation','kcnq_clamp_protocol',
              'fit_kcnq_relaxation','sharples2023_reference'):
    globals()[_name]=getattr(_kkinetics12,_name)

# [codex] 2026-11-13. Species-specific reference assays and read-only ATP output.
_mn_modulation13 = _module(B.parent/'2026-11-13-mn-modulation-atp/source_assays_codex.py',
                           '_stable_mn_modulation13')
for _name in ('mn_modulation_energy_sources','sharples2021_hcn_reference',
              'hcn_ionic_current','sharples2022_pump_reference',
              'adenylate_extract_observer','ateam_flim_observer',
              'compare_cart_response','observe_mn_compartment_energy'):
    globals()[_name]=getattr(_mn_modulation13,_name)
