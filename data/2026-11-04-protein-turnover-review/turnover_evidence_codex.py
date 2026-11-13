"""[codex] Assay-aware protein turnover references and explicit transfer scenarios.

Whole-protein dSILAC is not intrinsic cleavage, trafficking, enzyme deactivation
or fragment clearance. No default neuron parameter is replaced by this module.
"""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np

B=Path(__file__).resolve().parent
CONTEXT='Frankenfield2026_MN_dSILAC'
MAPPINGS={
    'TARDBP':('tdp_parameters','intact_half_life_s',3600.,
        'Total peptide turnover approximates intrinsic intact-TDP loss only in this scenario. Cleavage, phosphorylation, aggregation and fragments remain separate; not a fitted apparent TDP half-life.'),
    'STMN2':('nucleus_parameters','STMN2_protein_half_life_s',3600.,
        'Whole-culture turnover transferred to the soma. The independent Ellis human-MN CHX reference is retained; this does not calibrate axonal delivery or regrowth.'),
    'GAP43':('local_translation_parameters','protein_half_h',1.,
        'Whole-culture turnover transferred to a local pool. Existing ATP-dependent removal gate is retained, so observed local half-life need not equal the input parameter.'),
    'KCNQ2':('kcnq_parameters','channel_half_h',1.,
        'Total KCNQ2 turnover assigned to the existing equal-loss WT/delta trafficking pools. Mutant-isoform loss, surface residence and channel conductance were not measured by dSILAC.'),
}


def evidence(gene=None):
    data=json.loads((B/'protein_turnover_panel_codex.json').read_text(encoding='utf8'))
    if gene is None:return data
    hit=[r for r in data['records'] if r['gene']==gene]
    if not hit:raise ValueError('No reviewed source record for '+str(gene))
    return hit[0]


def reference_curve(gene,time_hours,*,assay=CONTEXT):
    """First-order reference signal, not raw observations or a confidence band."""
    t=np.asarray(time_hours,dtype=float)
    if not np.isfinite(t).all() or np.any(t<0):raise ValueError('Finite nonnegative hours required')
    if assay==CONTEXT:
        r=evidence(gene)
        if not r['eligible_source_reference']:raise ValueError('Unqualified source fit: '+str(r['exclusion_reason']))
        h=r['MN_half_hours']
    else:
        matches=[r for r in evidence()['alternative_assays'] if r['context']==assay and r['gene']==gene]
        if not matches:raise ValueError('No matched gene and assay')
        h=matches[0]['half_hours_approx']
    return dict(gene=gene,assay=assay,half_hours=h,time_hours=t,
        old_label_fraction=np.exp(-np.log(2)*t/h),raw_measurements=False,
        scope='Assay-specific first-order reference; no prediction of intact/fragment mixture or human whole-neuron function')


def transfer_parameters(api,options,*,genes,allow_effective_pool_approximation=False):
    """Copy selected parameters; explicit opt-in prevents assay/pool confusion.

    Reinitialization follows the existing model. This is a separate baseline
    scenario, not an acute intervention using the same biochemical initial state.
    """
    if allow_effective_pool_approximation is not True:
        raise ValueError('Set allow_effective_pool_approximation=True only for the documented source-to-pool sensitivity scenario')
    genes=tuple(genes)
    if not genes or len(set(genes))!=len(genes) or any(g not in MAPPINGS for g in genes):
        raise ValueError('Choose distinct reviewed mappings from TARDBP, STMN2, GAP43, KCNQ2')
    out=dict(options);changes=[]
    for g in genes:
        source=evidence(g)
        assert source['eligible_source_reference']
        key,field,scale,scope=MAPPINGS[g]
        base=out.get(key)
        if base is None:
            if g=='TARDBP':raise ValueError('Provide existing tdp_parameters explicitly; other TDP rates are not invented')
            if g=='STMN2':base=api.NucleusRibosomeParameters()
            elif g=='GAP43':base=api._lm35.cargo_parameters()
            elif g=='KCNQ2':base=api.NuclearKCNQTransportParameters()
        value=source['MN_half_hours']*scale
        fields={field:value}
        if hasattr(base,'provenance'):
            fields['provenance']=base.provenance+'; Conditional '+CONTEXT+' '+g+': '+scope
        out[key]=replace(base,**fields)
        changes.append(dict(gene=g,parameter=key+'.'+field,old_value=getattr(base,field),new_value=value,
            units='s' if scale==3600. else 'h',source_total_half_hours=source['MN_half_hours'],
            source_good_peptides=source['good_peptides'],assumption=scope))
    return out,dict(context=CONTEXT,changes=changes,
        scenario_type='assay-to-effective-pool transfer; not fitted human parameters',
        baseline='Existing baseline constructors rerun; not an acute equal-initial-state intervention',
        new_ODE_states=0,new_energy_debits=0,human_motor_neuron_calibrated=False,
        untouched='Calpain activation/deactivation, TDP cleavage and fragment/aggregate loss, RNA half-lives, transport residence times, electrical conductances',
        alternative_assays=evidence()['alternative_assays'])
