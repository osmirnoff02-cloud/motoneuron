"""[codex] 2026-10-25. Qualified whole-field optical reference for the neuron API.

Rat cortical example; one field, no measured human Ca/CAPN kinetic constants.
All 48 candidate traces remain accessible. Invalid measurements are never zero.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

B = Path(__file__).resolve().parent
ROOT = B.parents[1]
SOURCE = ROOT / 'data/2026-10-25-gerencser-fret'


def _json(name):
    return json.loads((SOURCE / name).read_text(encoding='utf-8'))


def _rows(name):
    with (SOURCE / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def gerencser_field_reference(roi=None):
    """Return raw optical ratios and independent Fura/FRET/joint quality masks.

    roi accepts candidate C01..C48 or historical R1/R2/R3. Array shape is
    (regions, 56), even for one ROI. Baseline refers to the first 3 acquisitions,
    not known pre-stimulus data. Every load verifies pinned source bytes.
    """
    manifest = json.loads((B / 'field_sources_codex.json').read_text(encoding='utf-8'))
    for path, expected in manifest['sources'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source changed; review and version the reference: {path}')
    audit = _json('field_analysis_audit_codex.json')
    details = audit['candidate_details']
    aliases = {d['original_roi']: d['id'] for d in details if d['original_roi']}
    canonical = aliases.get(roi, roi)
    if canonical is not None:
        details = [d for d in details if d['id'] == canonical]
        if not details:
            raise ValueError('Unknown candidate ROI')
    ids = [d['id'] for d in details]
    values = _rows('field_all_candidate_traces_codex.csv')
    flags = _rows('field_timepoint_qc_codex.csv')
    value_keys = {(v['id'], v['part'], v['local_frame']) for v in values}
    flag_keys = {(q['id'], q['global_frame']) for q in flags}
    if (len(values) != 48 * 56 or len(flags) != 48 * 56 or
            len(value_keys) != len(values) or len(flag_keys) != len(flags)):
        raise ValueError('Duplicate or missing source records')
    ratios = {s: [] for s in ('Fura', 'FRET')}
    masks = {s: [] for s in ratios}
    quality_flags = []
    clock = None
    for d in details:
        v = [r for r in values if r['id'] == d['id']]
        q = [r for r in flags if r['id'] == d['id']]
        t = np.array([float(r['time_s']) for r in v])
        tq = np.array([float(r['time_s']) for r in q])
        if len(v) != 56 or len(q) != 56 or not np.array_equal(t, tq) or np.any(np.diff(t) <= 0):
            raise ValueError('Invalid acquisition alignment')
        if [int(r['global_frame']) for r in q] != list(range(1, 57)):
            raise ValueError('Invalid QC frame order')
        if clock is not None and not np.array_equal(clock, t):
            raise ValueError('Different clocks within sample field')
        clock = t
        for signal in ratios:
            a = np.array([float(r[signal + '_ratio']) if r[signal + '_ratio'] else np.nan for r in v])
            ratios[signal].append(a)
            if signal == 'Fura':
                ok = np.array([r['Fura_quality_flag'] == 'pass' for r in q])
                ok &= d['fura_baseline_qualified']
            else:
                ok = np.array([float(r['FRET_valid_pixel_fraction']) >= .95 for r in v])
                ok &= d['fret_baseline_qualified']
            masks[signal].append(ok & np.isfinite(a))
        quality_flags.append([r['Fura_quality_flag'] for r in q])
    ratios = {s: np.array(v) for s, v in ratios.items()}
    masks = {s: np.array(v, dtype=bool) for s, v in masks.items()}
    return dict(
        roi_ids=ids, aliases=aliases, time_s=clock, candidate_details=details,
        ratios=ratios, valid=masks, valid_pair=masks['Fura'] & masks['FRET'],
        qualified_ratios={s: np.where(masks[s], v, np.nan) for s, v in ratios.items()},
        Fura_quality_flags=quality_flags, source_sha256=manifest['sources'],
        native_R1_comparison=_json('native_pair_validation_codex.json'),
        units='dimensionless_optical_ratio', source_field_count=1,
        first_three_frames_confirmed_prestimulus=False,
        stimulus_onset_s=None, treatment_group=None, calcium_uM=None,
        calpain_activity=None, TDP43_cleavage=None,
        human_kinetic_fit_allowed=False, calpain_washout_fit_allowed=False,
        qc_thresholds_validated=False,
        limitation='Fixed candidate boxes, no validated tracking. Fura censoring can depend on response. '
                   'No whole-field native validation, independent culture count, or absolute Ca calibration.')


def compare_gerencser_field(time_s, predicted_optical_ratio, *, roi, signal,
                            recording_offset_s, provenance, paired_only=False):
    """Sample a predicted optical observable on the recorded clock, using QC.

    Model time = recording time + recording_offset_s; no extrapolation.
    Fura expects a separately calibrated optical prediction, never Ca in uM.
    No optimizer, human-rate update, or biological goodness-of-fit is implied.
    First-3 normalization is refused if any required baseline point is flagged.
    """
    if signal not in ('Fura', 'FRET'):
        raise ValueError('signal must name an optical ratio: Fura or FRET, not calcium/activity')
    t = np.asarray(time_s, dtype=float)
    p = np.asarray(predicted_optical_ratio, dtype=float)
    if (t.ndim != 1 or p.shape != t.shape or len(t) < 2 or
            not np.isfinite(t).all() or not np.isfinite(p).all() or
            np.any(np.diff(t) <= 0) or not np.isfinite(recording_offset_s) or
            not isinstance(provenance, str) or not provenance.strip()):
        raise ValueError('Finite ordered optical prediction, explicit alignment and provenance required')
    d = gerencser_field_reference(roi)
    clock = d['time_s']
    tt = clock + recording_offset_s
    if tt[0] < t[0] or tt[-1] > t[-1]:
        raise ValueError('Prediction must cover the full recording window; extrapolation refused')
    eligible = d['valid_pair'][0] if paired_only else d['valid'][signal][0]
    if not eligible[:3].all():
        raise ValueError('ROI lacks three qualified baseline points for this comparison')
    observed = d['ratios'][signal][0]
    pred = np.interp(tt, t, p)
    bases = (observed[:3].mean(), pred[:3].mean())
    if not np.isfinite(bases).all() or min(bases) <= 0:
        raise ValueError('Positive finite observed and predicted baselines required')
    obs_norm = observed / bases[0]
    pred_norm = pred / bases[1]
    residual = np.where(eligible, pred_norm - obs_norm, np.nan)
    n = int(eligible.sum())
    return dict(
        roi=d['roi_ids'][0], signal=signal, time_s=clock.copy(),
        observed_ratio=observed.copy(), predicted_ratio=pred,
        observed_normalized=obs_norm, predicted_normalized=pred_norm,
        valid=eligible.copy(), residual=residual, used_frames=n, excluded_frames=56-n,
        RMSE=float(np.sqrt(np.mean(residual[eligible] ** 2))),
        paired_only=paired_only, provenance=provenance,
        temporal_warning=d['candidate_details'][0]['temporal_warning'],
        inferential_p_value=None, independent_donors=None,
        human_calibration=False, calpain_off_rate_identified=False,
        source_field_count=1, source_sha256=d['source_sha256'],
        meaning='Descriptive discrepancy on eligible optical samples. Unknown sample protocol, '
                'conditional censoring and preliminary masks preclude biological parameter fitting.')
