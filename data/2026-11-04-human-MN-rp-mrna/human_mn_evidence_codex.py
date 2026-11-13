"""[codex] 2026-11-04. Human-MN evidence gate; no physical rate overrides.

ActD estimates are mixed spinal-neuron source measurements under explicit
normalization assumptions. NTR is not a decay constant. This module preserves
both datasets without promoting either to measured basal pure-MN kinetics.
"""
from pathlib import Path
import json
import math

B = Path(__file__).resolve().parent
SCOPE = ('Human motor-neuron RP-mRNA evidence audit: 80 genes observed, '
         '0 newly identifiable basal human-MN decay rates and 0 absolute synthesis rates. '
         'GSE203579 is mixed spinal neuronal culture; its ActD fits are conditional. '
         'Supplied Li Tables contain sample metadata, not rates; C9-4/C9-5 identity conflicts between GEO and Table S7c. '
         'Publisher Source Data contain qPCR timecourses for three non-RP genes, not Figure3a gene-wise half-lives; BioSample/SRA do not resolve donor identity. '
         'E-MTAB-14537 NTR/counts lack resolved labeling and absolute RNA scale. '
         'Ye/DCPS adds 80 conditional ActD endpoint estimates and all-point nonlinear fits with replicate uncertainty; not basal MN calibration. '
         'Chua DataS1 contains 40 autophagy genes with relative RNA indices and zero canonical RP overlap. '
         'Physical priors are retained; this readout is not calibration.')


def _read(name):
    return json.loads((B / name).read_text(encoding='utf8'))


def evidence(gene=None, *, include_source_fits=False):
    """Return fresh evidence records; unknown rates remain null, never zero."""
    actd = _read('actd_rp80_fits_codex.json')
    vargas = _read('vargas_identifiability_codex.json')
    metadata = _read('li_supplement_metadata_codex.json')
    author_audit = _read('li_author_audit_codex.json')
    ye = _read('ye_chua_RP80_fits_codex.json')
    chua = _read('ye_chua_Chua_DataS1_codex.json')
    recovery = _read('ercc_bru_recovery_codex.json')
    genes = actd['genes']
    vr = {r['gene']: r for r in vargas['genes']}
    ar = {(r['group'], r['gene']): r for r in actd['rows']}
    yr = {r['gene']: r for r in ye['genes']}
    if len(genes) != 80 or set(vr) != set(genes) or set(yr) != set(genes):
        raise ValueError('Human-MN evidence must match all80 canonical genes')
    if gene is not None and gene not in genes:
        raise KeyError(gene)
    rows = []
    for g in genes if gene is None else (gene,):
        row = dict(gene=g, human_MN_decay_s_inv=None,
                   human_MN_absolute_production_uM_s=None,
                   physical_parameter_update_allowed=False,
                   status='unresolved_human_MN_kinetics',
                   Li_author_gene_wise_half_life_available=g in author_audit['canonical80_overlap'],
                   Vargas_observed_samples=vr[g]['samples_present'],
                   Vargas_author_width_QC=vr[g]['all_time_condition_groups_pass_author_2of3_width_qc'],
                   ActD_cell_type='human mixed spinal neuronal culture; MN enrichment not quantified per sample',
                   ActD_control_technical_eligible=ar['Ctrl', g]['source_profile_eligible'],
                   ActD_C9_technical_eligible=ar['C9', g]['source_profile_eligible'],
                   ActD_C9_source_identity_resolved=not bool(metadata['conflict_accessions']),
                   ActD_C9_calibration_eligible=metadata['calibration_eligible_by_group']['C9'],
                   ActD_control_calibration_eligible=metadata['calibration_eligible_by_group']['Ctrl'])
        row['Ye_DCPS_assay_estimates'] = {key: yr[g][key] for key in (
            'source_cell_type', 'source_condition', 'endpoint_arithmetic', 'log_linear',
            'nonlinear_all9', 'bootstrap_endpoint', 'zero_observations', 'means_monotone', 'flags')}
        row['Ye_DCPS_calibration_eligible'] = False
        if include_source_fits:
            row['Vargas_observations'] = vr[g]
            row['ActD_source_fits'] = {group: ar[group, g] for group in ('Ctrl', 'C9')}
            row['Ye_DCPS_observations'] = {key: yr[g][key] for key in (
                'source_columns', 'time_h', 'normalized_observations', 'timepoint_means', 'timepoint_CV')}
        rows.append(row)
    return dict(author='codex', date='2026-11-04', scope=SCOPE, rows=rows,
                new_identified_human_MN_decay=0, new_absolute_human_MN_synthesis=0,
                source_technical_screen=actd['summary'],
                Li_supplement_sample_metadata=metadata,
                Li_author_source_audit=author_audit,
                ERCC_BruChase_source_recovery=recovery,
                Ye_DCPS_source_audit={key: ye[key] for key in (
                    'source', 'doi', 'source_sha256', 'summary', 'policy', 'fixed77_panel_bootstrap',
                    'single_sample_pair_sensitivity', 'normalization_inventory', 'other_AI_reproduction')},
                Chua_DataS1_evidence={key: chua[key] for key in (
                    'source', 'doi', 'status', 'inventory', 'genes', 'unique_genes',
                    'canonical_RP_overlap', 'absent_from_Ln_sheets', 'normalization_check',
                    'cell_type', 'limitations', 'physical_parameter_update_allowed')},
                sources=['https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE203579',
                         'https://doi.org/10.1126/scisignal.adx6752',
                         'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE298758',
                         'https://doi.org/10.1016/j.cub.2022.04.053'],
                missing=['Ye decay-sample ERCC counts, spike amounts and calibration coefficients',
                         'Chua transcriptome-wide BruChase data beyond the supplied40 genes',
                         'author gene-wise Figure3a half-life table and fit code', 'resolution of C9-4/C9-5 sample identity', 'sample-resolved motor-neuron purity', 'labeling kinetics/effective time',
                         'absolute RNA per cell and volume', 'matched total/new/old RNA timecourses',
                         'mature RNA and compartment mapping'])


def compare_parameters(parameters):
    """Attach declared model values with their unresolved human-MN evidence status."""
    p = parameters if isinstance(parameters, dict) else vars(parameters)
    panel = evidence()
    for key in ('decay_s_inv', 'production_uM_s', 'initial_total_uM'):
        values = p[key]
        if len(values) != 80 or any(not math.isfinite(float(x)) or x < 0 for x in values):
            raise ValueError('Expected finite nonnegative canonical80 model parameters')
    for i, row in enumerate(panel['rows']):
        row['model_decay_s_inv'] = float(p['decay_s_inv'][i])
        row['model_production_uM_s'] = float(p['production_uM_s'][i])
        row['model_initial_total_uM'] = float(p['initial_total_uM'][i])
        model_k = row['model_decay_s_inv']
        row['Ye_DCPS_half_life_over_current_model'] = {
            estimator: (fit['half_life_h'] * model_k * 3600 / math.log(2)
                        if fit is not None and fit['half_life_h'] is not None and model_k > 0 else None)
            for estimator, fit in ((key, row['Ye_DCPS_assay_estimates'][key])
                                   for key in ('endpoint_arithmetic', 'log_linear', 'nonlinear_all9'))}
    panel['model_profile'] = p['evidence_profile']
    panel['model_provenance'] = p['provenance']
    return panel
