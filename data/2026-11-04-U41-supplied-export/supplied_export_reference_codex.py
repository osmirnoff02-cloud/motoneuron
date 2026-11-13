"""Codex 2026-11-04: assay-specific U41 observations; no physical-rate override.

PIR is an intron/event-specific percentage within a fraction, not N/(N+C).
Differentiation days are not residence times. Mouse ALIBi and human neurogenesis
remain different source contexts. Source sample conflicts require acknowledgement.
"""
from pathlib import Path
import json
import math

B = Path(__file__).resolve().parent
SOURCES = {
    'PetricHowe2022': ('petric_pir_panel_codex.json', 'human_iPSC_motor_neurogenesis'),
    'Zhang2025': ('zhang_alibi_panel_codex.json', 'mouse_ESC_ALIBi'),
    'Wang2021': ('wang_probe_panel_codex.json', 'probe_sequences_only'),
    'Tyzack2021': ('../2026-11-04-U41-development-QC/tyzack_aberrant_ir_panel_codex.json', 'human_iPSC_motor_neurogenesis'),
}

DEVELOPMENT = B.parent / '2026-11-04-U41-development-QC'


def u41_development_sample_qc(sample_id=None, *, cell_context):
    """Read original QC provenance; no pass/fail inference or genotype relabeling."""
    if cell_context != 'human_iPSC_motor_neurogenesis':
        raise ValueError('Development QC applies to human_iPSC_motor_neurogenesis only')
    panel = json.loads((DEVELOPMENT/'development_qc_panel_codex.json').read_text(encoding='utf-8'))
    if sample_id is None:
        return panel
    if sample_id not in panel['samples']:
        raise ValueError('Unknown source sample ID')
    return {'sample': panel['samples'][sample_id], 'limitations': panel['limitations'],
            'aggregate_group_calibration_allowed': False, 'export_rate_identified': False}


def u41_supplied_export_reference(source, *, cell_context):
    """Return independent source data, with context guard and unmodified source labels."""
    if source not in SOURCES:
        raise ValueError(f'Unknown source; choose from {tuple(SOURCES)}')
    file, expected = SOURCES[source]
    if cell_context != expected:
        raise ValueError(f'{source} requires cell_context={expected!r}; no automatic transfer')
    result = json.loads((B / file).read_text(encoding='utf-8'))
    if source in ('PetricHowe2022', 'Tyzack2021'):
        assay = json.loads((DEVELOPMENT/'assay_constraints_codex.json').read_text(encoding='utf-8'))
        result['supplement_assays'] = [a for a in assay['assays'] if a['source'] == source]
    if source == 'PetricHowe2022':
        qc = u41_development_sample_qc(cell_context=cell_context)
        result['qc']['supplement_followup'] = {k:v for k,v in qc.items()
                                               if k not in ('samples', 'source_lines')}
        fields = ('source_line', 'S1_mutation', 'S2_genotypes', 'mutation_conflict',
                  'RIN_values', 'comments', 'any_metric_outlier',
                  'author_final_exclusion_decision_available')
        result['sample_qc'] = {sid:{**{k:s[k] for k in fields},
                                   'QC_source_excel_rows':[r['excel_row'] for r in s['file_records']]}
                               for sid,s in qc['samples'].items() if s['in_Petric_PIR_columns']}
        result['event_annotations'] = {r['event']:{'class_interpretation':qc['class_interpretation'] if r['exact_class_match'] else 'No matched Tyzack S6 row; source class remains unresolved.',
               'Tyzack_S6_match':r, 'SVD_groups':qc['SVD_event_groups'].get(r['event'], [])}
               for r in qc['class_matches']}
    return result


def _pir_panel(cell_context, qc_policy):
    if qc_policy != 'source_labels_only':
        raise ValueError('Unresolved sample/legend conflicts require qc_policy="source_labels_only"')
    return u41_supplied_export_reference('PetricHowe2022', cell_context=cell_context)


def _rows(panel):
    rows = []
    for event in panel['events']:
        for col, value in zip(panel['sample_columns'][event['source_sheet']], event['source_values']):
            observed = isinstance(value, (float, int)) and not isinstance(value, bool)
            rows.append({
                'gene': event['gene'], 'event': event['event'],
                'source_sheet': event['source_sheet'], 'source_file': panel['source_file'],
                'source_cell': f"{col['excel_column']}{event['excel_row']}",
                **{k: col[k] for k in ('sample_id', 'day', 'line', 'compartment', 'group')},
                'source_class': event['source_class'], 'source_value': value,
                'sample_qc': panel['sample_qc'][col['sample_id']],
                'event_annotation': panel['event_annotations'][event['event']],
                'pir_percent': value if observed else None,
                'unit': 'PIR_percent', 'is_observed': observed,
            })
    return rows


def u41_pir_observations(gene, *, cell_context, qc_policy, event=None, day=None, compartment=None):
    """Return event-matched observations without group pooling or missing-value imputation."""
    panel = _pir_panel(cell_context, qc_policy)
    if gene not in {e['gene'] for e in panel['events']}:
        raise ValueError('Gene absent from the selected source panel; absence is not zero')
    if day is not None and (type(day) is not int or day not in (0, 3, 7, 14, 22, 35)):
        raise ValueError('day must be a source differentiation day: 0, 3, 7, 14, 22, 35')
    if compartment is not None and compartment not in ('nuc', 'cyto'):
        raise ValueError('compartment must be nuc or cyto')
    rows = [r for r in _rows(panel) if r['gene'] == gene
            and (event is None or r['event'] == event)
            and (day is None or r['day'] == day)
            and (compartment is None or r['compartment'] == compartment)]
    if not rows:
        raise ValueError('No matching source observations')
    return {'source': panel['source'], 'cell_context': cell_context, 'qc': panel['qc'],
            'rows': rows, 'export_rate_identified': False,
            'limitation': 'PIR within each fraction; not RNA abundance or nuclear fraction. '
                          'ODE has no event-specific intron states; caller must supply an explicit compatible prediction.'}


def compare_u41_pir(predictions, *, cell_context, qc_policy):
    """Compare explicitly supplied PIR predictions with source cells; no fit or ODE mutation.

    Each prediction requires source_sheet, event, sample_id, unit='PIR_percent',
    pir_percent. Sheet is mandatory because some source events occur in both S2/S3.
    """
    panel = _pir_panel(cell_context, qc_policy)
    keys = ('source_sheet', 'event', 'sample_id')
    targets = {tuple(r[k] for k in keys): r for r in _rows(panel)}
    results = []; seen = set()
    for prediction in predictions:
        if not isinstance(prediction, dict) or not all(k in prediction for k in (*keys, 'unit', 'pir_percent')):
            raise ValueError('Prediction requires source_sheet/event/sample_id/unit/pir_percent')
        if prediction['unit'] != 'PIR_percent':
            raise ValueError('Only within-fraction PIR_percent is supported; N/C abundance is not PIR')
        key = tuple(prediction[k] for k in keys)
        if key in seen or key not in targets:
            raise ValueError('Duplicate or unknown source observation')
        seen.add(key)
        value = prediction['pir_percent']; target = targets[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('PIR prediction must be finite and between 0 and 100 percent')
        if not target['is_observed']:
            raise ValueError('Source observation is missing; missing values cannot be fitted')
        results.append({**target, 'predicted_pir_percent': value,
                        'residual_percentage_points': value - target['pir_percent']})
    if not results:
        raise ValueError('At least one explicit prediction is required')
    return {'source': panel['source'], 'cell_context': cell_context, 'qc': panel['qc'],
            'comparisons': results, 'compared_observations': len(results),
            'available_numeric_observations': sum(r['is_observed'] for r in targets.values()),
            'export_rate_fitted': False, 'physical_parameters_changed': False}
