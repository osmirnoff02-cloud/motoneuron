"""Codex: evidence-aware calibration input; inherited physiology unchanged.

Onodera normalized RNA and Hirunagi well-level fluorescence are separate
measurements. Neither is automatically a conductance, calcium pump or ADAR2 rate.
"""
from pathlib import Path
from itertools import combinations
import importlib.util,sys,json,csv
import numpy as np
B=Path(__file__).resolve().parent

def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

_prior=_load('_sources_before_onodera_hirunagi',B.parent/'2026-10-19-source-model-update/model_api.py')
for _name in (*_prior._INHERITED,'source_training_data','apply_boivin_drift','simulate_substrate','simulate_fluorescent_intact','robinson_fret','epitope_linear_mixture','gerencser_corrected_fret','observe_biochemistry_substrate'):
    globals()[_name]=getattr(_prior,_name)
_obs=_load('_source_observers_before_onodera_hirunagi',B.parent/'2026-10-19-source-model-update/source_observers.py')
for _name in ('unmix_fura_fret','renewing_reporter','sample_photometry','competing_tdp_chase','observe_fragment_band'):globals()[_name]=getattr(_obs,_name)
_qc=_load('_optical_qc_before_onodera_hirunagi',B.parent/'2026-10-19-source-model-update/optical_qc.py')
optical_event_features=_qc.optical_event_features
calibrate_negative_control=_qc.calibrate_negative_control

def calibration_evidence():
    """Return measured references and explicit unresolved donor/assay metadata."""
    return {name:json.loads((B/name).read_text(encoding='utf-8')) for name in ('hirunagi_reference.json','rna_intake_and_qc.json')}

def source_training_data():
    """Keep legacy assay sources and append the corrected donor/RNA evidence."""
    sources=_prior.source_training_data()
    sources['onodera_hirunagi_evidence']=calibration_evidence()
    return sources

def healthy_gene_reference(gene,*,fraction='P4'):
    """Descriptive normalized expression only, with shared-donor provenance."""
    if fraction not in ('P4','NT'):raise ValueError('P4/NT fractions must remain separate')
    with (B/'model_gene_evidence_codex.tsv').open(encoding='utf-8',newline='') as f:
        rows=[r for r in csv.DictReader(f,delimiter='\t') if r['gene']==gene]
    if len(rows)!=1 or rows[0]['usable']!='True':raise ValueError('No unambiguous source row for this gene')
    clones=['EKA','TIGE9','YFE16','YFE19']
    return dict(gene=gene,fraction=fraction,values={c:float(rows[0][c+'_'+fraction]) for c in clones},
        units='deposited_DESeq2_normalized_expression',source='GSE142612',healthy_donors=3,
        shared_donor_clones=['YFE16','YFE19'],EKA_EKN3_alias_confirmed=False,
        conductance=None,pump_rate=None,QR_editing=None,protein_activity=None)

def compare_hirunagi_optical(predicted_by_line,*,context):
    """Compare a matching well-level optical model; reject single-cell AP/Ca.

    Donor identities of HC1-4 are not published. Return all six possible
    donor-weighted RMSEs instead of inventing their mapping or confidence interval.
    Current isolated-neuron NEURON output is not eligible for this comparison.
    """
    ref=calibration_evidence()['hirunagi_reference.json'];expected=ref['context']
    for key in ['species','cell_type','group','indicator','instrument','metric','unit','aggregation','recording_s','recording_DIV','cells_plated_per_well','DHT_nM','DHT_days']:
        if context.get(key)!=expected[key]:raise ValueError('Incompatible assay context: '+key)
    labels=list(ref['healthy_line_statistics'])
    if set(predicted_by_line)!=set(labels):raise ValueError('One explicit prediction per source HC label required')
    predictions=np.array([predicted_by_line[k] for k in labels],dtype=float)
    if predictions.shape!=(4,) or not np.isfinite(predictions).all() or np.any(predictions<0):raise ValueError('Finite nonnegative optical rates required')
    target=np.array([ref['healthy_line_statistics'][k]['optical_bursts_per_min']['mean'] for k in labels])
    error=predictions-target;results=[]
    for possibility in ref['possible_donor_mappings']:
        w=np.array([possibility['line_weights'][k] for k in labels])
        results.append(dict(hypothetical_same_donor_pair=possibility['hypothetical_same_donor_pair'],RMSE=float(np.sqrt(np.sum(w*error*error)))))
    return dict(source='Hirunagi2026 Fig5g',unit='events/min',residuals=dict(zip(labels,error.tolist())),mapping_sensitivity=results,
        RMSE_min=min(r['RMSE'] for r in results),RMSE_max=max(r['RMSE'] for r in results),
        mapping_uncertainty_is_not_confidence_interval=True,independent_human_validation=False)
