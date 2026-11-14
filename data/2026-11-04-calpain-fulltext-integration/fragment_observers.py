"""[codex] 2026-11-04. Sequence-aware assay projections, not new neuron rates.

Fragments are explicit canonical TARDBP Q13148 1-based inclusive segments.
Existing ODE lineage-equivalent pools cannot be passed as molecular fragments.
"""
from pathlib import Path
import json
import numpy as np

B = Path(__file__).resolve().parent
PANELS = {
    'feneberg_prm': (
        ('N5_27', 5, 27, False),
        ('C375_385', 375, 385, False),
        ('neoN176_188', 176, 188, True),
        ('neoN266_276', 266, 276, True),
    ),
    'igaz_epitope': (
        ('N6_24', 6, 24, False),
        ('C394_414', 394, 414, False),
    ),
}


def calpain_fulltext_evidence(source_id=None):
    data = json.loads((B / 'fulltext_constraints.json').read_text(encoding='utf8'))
    if source_id is None:
        return data
    hits = [row for row in data['sources'] if row['id'] == source_id]
    if len(hits) != 1:
        raise ValueError('Unknown source_id')
    return hits[0]


def tdp_fragment_assay(fragments, *, assay, compartment, fraction, amount_basis,
                       response_yields=None):
    """Return sequence coverage, and signals only with explicit assay responses.

Each fragment: {name, start, end, amount}. Amounts share a molar basis.
response_yields is channels x fragments, including extraction, digestion,
missed cleavage and detection. No response or efficiency is fitted here.
For Igaz this is a latent epitope proxy, NOT the ordinal IHC score 0..4.
Full immunogen coverage is conservative; partial epitope binding is unknown.
"""
    if assay not in PANELS:
        raise ValueError('Unknown assay')
    if amount_basis != 'molar':
        raise ValueError('Molecular molar amounts required, not ODE lineage equivalents')
    if compartment not in ('cortex', 'spinal_cord', 'iPSC_MN'):
        raise ValueError('Specify source compartment')
    if fraction not in ('urea_insoluble', 'soluble', 'whole_cell', 'histological_section'):
        raise ValueError('Specify source fraction')
    if not isinstance(fragments, (list, tuple)) or not fragments:
        raise ValueError('At least one explicit fragment required')
    names, spans, amounts = [], [], []
    for f in fragments:
        name, start, end, amount = f['name'], f['start'], f['end'], f['amount']
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Unique nonempty fragment names required')
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, int) or not isinstance(end, int)
                or not 1 <= start <= end <= 414):
            raise ValueError('Canonical 1-based inclusive integer segments required')
        if isinstance(amount, bool) or not np.isfinite(amount) or amount < 0:
            raise ValueError('Nonnegative finite molar amounts required')
        names.append(name); spans.append((start, end)); amounts.append(float(amount))
    panel = PANELS[assay]
    coverage = np.array([[s <= lo and e >= hi and (not neo or s == lo)
                          for s, e in spans] for _, lo, hi, neo in panel], dtype=bool)
    out = dict(assay=assay, channels=[p[0] for p in panel], fragment_names=names,
        structural_coverage=coverage.tolist(), compartment=compartment, fraction=fraction,
        signals=None, C_over_N=None, response_status='unspecified',
        calibration_status='not_calibrated', calpain_specific=False,
        scope='Explicit observation only; no ODE states, cleavage rates, threshold or energy debit.',
        limitations=[
            'No conversion from existing N-lineage ODE states without an explicit molecular map.',
            'C/N cannot distinguish intact TDP from complementary N/C fragments.',
            'Peptide or epitope signal is not enzyme activity or a universal healthy bound.',
        ])
    if assay == 'feneberg_prm':
        out['limitations'] += [
            'neoN266 requires an internal chymotryptic missed cleavage; its yield is assay dependent.',
            'Canonical neoN176 differs from published start175; main N sequence omits one D.',
            'Supplement isotope label R versus K/+8 is inconsistent; no absolute mass calibration.',
        ]
    else:
        out['limitations'].append('Full immunogen coverage is only a conservative epitope proxy, not an ordinal IHC score or a validated antibody binding model.')
    if response_yields is not None:
        y = np.asarray(response_yields, dtype=float)
        if y.shape != coverage.shape or not np.isfinite(y).all() or np.any(y < 0):
            raise ValueError('Finite nonnegative channels x fragments response matrix required')
        signal = (coverage * y) @ np.asarray(amounts)
        out.update(signals=dict(zip(out['channels'], signal.tolist())),
                   C_over_N=float(signal[1] / signal[0]) if signal[0] > 0 else None,
                   response_status='explicit_caller_supplied_not_validated')
    return out
