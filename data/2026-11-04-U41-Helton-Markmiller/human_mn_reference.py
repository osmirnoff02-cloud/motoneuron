"""Human-MN U41 observation targets. Never alter physical rates or energy ledgers.

HSPA1A/B are not dedicated states in the current full ODE. Predictions must
therefore be supplied explicitly for these genes; generic RNA cannot stand in.
"""
from pathlib import Path
import json
import math
from numbers import Real

B=Path(__file__).resolve().parent
HELTON_CONTEXT='human_iPSC_spinal_MN_BrainXell'
MARKMILLER_CONTEXT='human_PSC_derived_motor_neurons'


def u41_markmiller_model_labels(*, cell_context):
    """Map categorical source evidence to model RNA names; no rates or fractions."""
    if cell_context!=MARKMILLER_CONTEXT:
        raise ValueError('Markmiller evidence requires human_PSC_derived_motor_neurons context')
    return json.loads((B/'markmiller_model_mapping.json').read_text(encoding='utf8'))


def u41_human_mn_reference(source, *, cell_context):
    """Read source-qualified observations; each call returns an independent copy."""
    sources={'Helton2026':('helton_humanMN_observations.json',HELTON_CONTEXT),
             'Markmiller2021':('markmiller_localization.json',MARKMILLER_CONTEXT)}
    if source not in sources:
        raise ValueError('Choose Helton2026 or Markmiller2021')
    filename,context=sources[source]
    if cell_context!=context:
        raise ValueError(f'{source} requires cell_context={context!r}; cell types are not interchangeable')
    return json.loads((B/filename).read_text(encoding='utf8'))


def u41_human_mn_target(gene, condition, *, qc_policy, cell_context):
    """Fraction target, not a kinetic rate. Policy is mandatory due to source errors.

    reported_all reproduces Nuc/Total for all source rows. consistent_only drops
    rows whose Nuc+Cyto differs from Total; neither policy edits original counts.
    Equal weight is given to independent replicate means, not individual cells.
    """
    panel=u41_human_mn_reference('Helton2026',cell_context=cell_context)
    if qc_policy not in panel['summaries']:
        raise ValueError('qc_policy must be reported_all or consistent_only')
    target=next((r for r in panel['summaries'][qc_policy]
                 if r['gene']==gene and r['condition']==condition),None)
    if target is None:
        raise ValueError('Only HSPA1A/HSPA1B and Unstressed/Stressed are measured in this human-MN panel')
    target.update(source='Helton2026',source_sheet=panel['source_sheet'],cell_context=cell_context,
        qc_policy=qc_policy,unit='fraction of detected RNA spots in nucleus',
        condition_protocol=panel['protocol']['Stressed'] if condition=='Stressed' else panel['protocol']['control'],
        caution='Two-condition localization target; export, synthesis and decay are not separately identified. No automatic full-ODE gene mapping.',
        source_inconsistent_rows=[r['excel_row'] for r in panel['qc']['inconsistent_rows']])
    return target


def compare_u41_human_mn_localization(predictions, *, qc_policy, cell_context):
    """Compare explicitly gene-matched nuclear fractions with Helton endpoints.

    predictions: sequence of dictionaries with gene, condition, nuclear_fraction.
    A fraction is N/(N+C) of compatible amounts, never a ratio of concentrations
    in unequal compartment volumes. This is a diagnostic, not a fit or validation
    of human-MN export. Partial predictions are reported as partial coverage.
    """
    predictions=list(predictions)
    if not predictions:
        raise ValueError('At least one explicit gene/condition prediction is required')
    seen=set();out=[]
    for pred in predictions:
        key=(pred['gene'],pred['condition'])
        if key in seen:
            raise ValueError('Duplicate gene/condition predictions would double-count evidence')
        seen.add(key)
        value=pred['nuclear_fraction']
        if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('nuclear_fraction must be a finite real fraction in [0,1]')
        target=u41_human_mn_target(*key,qc_policy=qc_policy,cell_context=cell_context)
        out.append(dict(gene=key[0],condition=key[1],prediction=float(value),
            observed=target['mean_nuclear_fraction'],residual=float(value)-target['mean_nuclear_fraction'],
            sem_between_replicates=target['sem_between_replicates'],n_replicates=target['n_replicates'],n_cells=target['n_cells']))
    return dict(source='Helton2026',cell_context=cell_context,qc_policy=qc_policy,
        comparisons=out,rmse=math.sqrt(sum(r['residual']**2 for r in out)/len(out)),
        covered_targets=len(out),available_targets=4,
        status='Observation comparison only; no export-rate calibration or treatment efficacy inference',
        energy_charged=False,physical_model_modified=False)
