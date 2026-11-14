"""[codex] 2026-11-04. Auditable composition of the existing API0.51 network.

No additional ODE, energy debit, biological rate or source-cell calibration.
The common driver remains the sole state/checkpoint/ATP owner.
"""
from copy import deepcopy
import numpy as np

CONTRACT = 'integrated-neuron-2026-11-04-v1'
COMPONENTS = (
    ('individual_RP_observation', '80 individual RP inventories and turnover',
     'data/2026-11-03-rp-neuron-coupling/rp_coupling.py'),
    ('RP_production_observation', '80 translation queues using common resources',
     'data/2026-11-03-rp-production/production.py'),
    ('RP_mRNA_kinetics_observation', '80 RP transcript production and loss rates',
     'data/2026-11-03-rp-mrna-kinetics/kinetics.py'),
    ('nuclear_transport_observation', 'Finite karyopherins, pore passage and Ran cycle',
     'data/2026-11-03-nuclear-transport/transport.py'),
    ('RNA_export_observation', 'RNA export, RP import and pre40S/pre60S export',
     'data/2026-11-03-rna-ribosome-transport/rna_transport.py'),
)


def components():
    """Return a portable inventory; it is not a calibration certificate."""
    return [dict(observation=key,role=role,source=source) for key,role,source in COMPONENTS]


def validate_options(kwargs):
    if kwargs.get('include_individual_rp', True) is not True:
        raise ValueError('Integrated neuron requires all 80 individual RP genes')
    if 'nuclear_transport_parameters' in kwargs and kwargs['nuclear_transport_parameters'] is None:
        raise ValueError('Integrated neuron requires nuclear transport; use the explicit legacy API for disabled branches')


def attach(result, *, rp_genes, readout=None):
    """Validate composition and attach descriptive metadata without changing arrays."""
    absent=[key for key,_,_ in COMPONENTS if result.get(key) is None]
    if absent:
        raise ValueError('Missing integrated components: '+', '.join(absent))
    genes=tuple(rp_genes)
    if len(genes)!=80 or len(set(genes))!=80:
        raise ValueError('Expected the canonical 80 distinct RP genes')
    for key in ('individual_RP_observation','RP_production_observation','RP_mRNA_kinetics_observation'):
        if tuple(result[key]['genes'])!=genes:
            raise ValueError('RP gene order mismatch: '+key)
    export=result['RNA_export_observation']
    if tuple(export['RP_genes'])!=genes or tuple(export['genes'][5:])!=genes:
        raise ValueError('RNA export and RP production use different gene orders')
    n=len(result['time_s'])
    chloride_enabled='passive_chloride' in result['config']
    # [claude] 2026-11-13: optional six-state acid-extrusion block (plan N05.1.4) follows passive Cl.
    acid_enabled='acid_extrusion' in result['config']
    expected_count=(4425 if chloride_enabled else 4413)+(6 if acid_enabled else 0)
    if result['state'].shape!=(n,expected_count) or result['state_schema']['count']!=expected_count:
        raise ValueError('Integrated-v1 requires the reviewed 4413 states plus the optional 12-state passive Cl block')
    if chloride_enabled and (result['config']['passive_chloride_offset']!=4413 or result.get('passive_chloride_observation') is None):
        raise ValueError('Incomplete passive chloride topology')
    if acid_enabled and (result['config']['acid_extrusion_offset']!=expected_count-6 or result.get('acid_extrusion_observation') is None):  # [claude] 2026-11-13
        raise ValueError('Incomplete acid extrusion topology')
    rna=np.asarray(export['RNA_pools_counters'])
    rmk=result['RP_mRNA_kinetics_observation']
    if not np.array_equal(rna[:,5:,:],rmk['RNA_pools_counters']):
        raise ValueError('RP kinetics must observe the same RNA pools as transport')
    if result['RP_production_observation']['queues_Ceq'].shape!=(n,80,15):
        raise ValueError('Incomplete RP translation queues')
    # Legacy export names used the same prefix for mRNA and protein. Preserve
    # backend labels/checkpoint hashes, but provide an unambiguous analysis view.
    columns=list(result['columns'])
    block_size=(rna.shape[1]*rna.shape[2]+np.asarray(export['RP_pools']).shape[1]*4+
                np.asarray(export['pre_ribosome_pools']).shape[1]*4+
                len(export['carrier_names'])+len(export['counter_names']))
    offset=len(columns)-block_size-(12 if chloride_enabled else 0)-(6 if acid_enabled else 0)  # [claude] 2026-11-13: acid block
    rna_names=['nrt_'+g+'_'+f+'_uM' for g in export['genes'] for f in export['RNA_fields']]
    protein_fields=('cytoplasmic_free','KPNB1_cytoplasmic','KPNB1_nuclear','nuclear_free')
    protein_names=['nrt_'+g+'_'+f+'_uM' for g in genes for f in protein_fields]
    old=rna_names+protein_names
    if columns[offset:offset+len(old)]!=old:
        raise ValueError('RNA/protein export label block differs from the reviewed schema')
    columns[offset:offset+len(old)]=([x.replace('nrt_','nrt_RNA_',1) for x in rna_names]+
                                   [x.replace('nrt_','nrt_protein_',1) for x in protein_names])
    if len(set(columns))!=len(columns):
        raise ValueError('Unresolved ambiguous labels in integrated analysis columns')
    result['integrated_columns']=tuple(columns)
    if readout is not None:
        if tuple(readout['genes'])!=tuple(export['genes']):
            raise ValueError('U41 readout uses different RNA genes')
        result['integrated_transport_readout']=readout
    result['integrated_model']={
        'contract':CONTRACT,'entry_point':'run_integrated_neuron',
        'numerical_backend':result['interface_version'],
        'state_count':expected_count,'components':components(),
        'one_state_vector':True,'one_checkpoint':True,
        'new_physical_states':(4 if chloride_enabled else 0)+(2 if acid_enabled else 0),'new_energy_debits':0,  # [claude] 2026-11-13: +2 acid pools
        'new_bookkeeping_states':(8 if chloride_enabled else 0)+(4 if acid_enabled else 0),  # [claude] 2026-11-13: +4 acid ledgers
        'optional_passive_chloride':chloride_enabled,
        'optional_acid_extrusion':acid_enabled,  # [claude] 2026-11-13
        'analysis_columns':'integrated_columns separates RNA from protein labels; columns/state_schema preserve the backend checkpoint contract',
        'resource_ownership':{
            'ATP_ADP_AMP':'Existing soma/process/terminal pools; no second RP or nuclear ATP pool',
            'GTP_GDP':'Existing guanylate pools; NDPK is the recharge ATP owner',
            'RP_mRNA':'Existing 85-class RNA-export state; kinetics replaces source/loss only for 80 RP messages',
            'RP_protein':'Existing individual inventories receive the actual production queues; coarse RP source is replaced',
            'Ran':'Shared finite transport cycle used by cargo and pre-ribosome export',
            'U41':'Read-only source assays and observations; no automatic source-cell parameter override',
        },
        'RP_mRNA_external_transfer_validation':deepcopy(rmk['external_transfer_validation']),
        'human_motor_neuron_calibrated':False,
        'limitations':[
            'Coupling is implemented; a common long-term steady state is not established',
            'RP-mRNA source-cell rate transfer to human motor neurons did not pass the recorded conditional screen',
            'Absolute RNA/protein pools and many transport rates remain assumptions/ranges',
            'Source fluorescence/NTR/FRAP and pore dwell observations are not interchangeable with model concentrations or rates',
        ],
    }
    return result
