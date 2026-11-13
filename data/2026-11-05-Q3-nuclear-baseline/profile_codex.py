"""[codex] Q3 explicit candidate derived from the prior neutral Q2 run.

These are conditional normalization constraints, NOT human measurements.
The donor Q2 run had no differential KCNQ current; this candidate restores it.
No state is clamped or reset. Every additional completed protein pays its cost.
"""
from pathlib import Path
from dataclasses import replace
import json,numpy as np

B=Path(__file__).resolve().parent
RANGES={'nuclear_reference_fraction':[.6,.6733782240461704,.8320592217706559,.9],
        'basal_initiation_gain':[1.,1.3345234618112376,1.5],
        'CK_source_model':['Wiseman2023','Teague1996'],
        'pH':[6.8,7.,7.2],'free_Mg_mM':[.5,1.,2.],'ionic_strength_M':[.15,.25]}

def inputs(api,segments,profile='Burley',*,reference_fraction=None,initiation_gain=None,CK_source='Wiseman2023',initialize_RNA=True,rp_translation_reference=True,ionic_scenario='legacy_reference'):
    thermo=api._module(B/'thermodynamics_codex.py','_Q3_thermodynamics05')
    donor=json.loads((B/'Q2_donor_constraints_codex.json').read_text(encoding='utf8'))
    old=donor['source']
    # Explicit calibration target: keep the inherited basal completed-protein
    # flux while using the now-expanded resource network. It remains unmeasured.
    f=donor['nuclear_fraction'] if reference_fraction is None else float(reference_fraction)
    gain=(1. if rp_translation_reference else donor['derived_initiation_gain']) if initiation_gain is None else float(initiation_gain)
    if not .1<f<.99 or not 0<gain<=2:raise ValueError('Conditional fraction .1..99 and positive initiation support<=2')
    eq=thermo.equilibrium()
    o,record=api.joint_baseline_inputs(profile,AK_Kobs=eq['AK_Kobs'])
    if ionic_scenario!='legacy_reference':
        o,record=api.ion_energy_inputs(profile,AK_Kobs=eq['AK_Kobs'],ionic_scenario=ionic_scenario)
    if CK_source=='Teague1996':
        joint=api._module(B.parent/'2026-11-04-joint-baseline-calibration/profile_codex.py','_Q3_joint05')
        en=o['energy_parameters'];ck=o['creatine_parameters']
        pools=joint.energy_pools(adenylate_total_uM=en.ATP0_uM+en.ADP0_uM+en.AMP0_uM,
          creatine_total_uM=ck.total_creatine_uM[0],PCr_uM=record['energy']['PCr0_uM'],KCK=eq['CK_Kobs'],KAK=eq['AK_Kobs'])
        o['energy_parameters']=replace(en,**pools,gly_half_ADP_uM=pools['ADP0_uM'])
        o['creatine_parameters']=replace(ck,reverse_equilibrium_factor=1/eq['CK_Kobs'],provenance='Conditional Teague1996 Table1 source-model alternative; not human MN calibration')
        record['energy'].update(pools)
    elif CK_source!='Wiseman2023':raise ValueError(CK_source)
    o['localization_parameters']=replace(o['localization_parameters'],nuclear_fraction=f,
      provenance='Q3 conditional basal nuclear fraction from Q2 neutral 72h transport; not a measured healthy-human fraction')
    o['basal_translation_capacity_scale']=gain
    o['conditional_RNA_export_initialization']=initialize_RNA
    o['gene_specific_RP_translation_reference']=rp_translation_reference
    record['Q3']=dict(healthy_human_calibrated=False,nuclear_reference_fraction=f,basal_initiation_gain=gain,
      calibration_donor=str(old),CK_source=CK_source,Teague1996=eq,
      assumptions='Fixed conditional basal reference and constant initiation capacity; derived before challenge, not adjusted during a trajectory. Shared ribosomes, AA, GTP and ATP remain finite.',
      differential_KCNQ_conductance_nS=.5,state_clamps=False,state_resets=False,conditional_RNA_export_initialization=initialize_RNA,
      whole_network_stationarity_validated=False,gene_specific_RP_translation_reference=rp_translation_reference)
    record['energy']['AK_Kobs']=eq['AK_Kobs'];record['energy']['AK_source']=eq['source']
    o['control_context']=record
    return o,record
