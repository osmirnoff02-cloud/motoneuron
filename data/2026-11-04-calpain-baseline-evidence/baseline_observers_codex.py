"""[codex] Separate biochemical states from experimental observation.

No neuron RHS, default parameter, initial pool or energy debit is changed.
"""
from pathlib import Path
import json
import numpy as np
B=Path(__file__).resolve().parent

def calpain_baseline_sources(source_id=None):
    data=json.loads((B/'source_constraints_codex.json').read_text(encoding='utf8'))
    if source_id is None:return data
    hits=[r for r in data['records'] if r['id']==source_id]
    if len(hits)!=1:raise ValueError('Unknown reviewed source')
    return hits[0]

def calpain_form_observer(*,native,processed,fragile,inactive,active_open):
    """Amounts are disjoint forms; open activity is a SUBSET, never an extra pool."""
    a=np.array([native,processed,fragile,inactive,active_open],float)
    if not np.isfinite(a).all() or np.any(a<0):raise ValueError('Nonnegative finite amounts required')
    live=native+processed+fragile;total=live+inactive
    if total<=0 or live<=0:raise ValueError('Nonzero total and noninactive amount required')
    if active_open>live+1e-12:raise ValueError('Open subset exceeds noninactive amount')
    return dict(total=total,noninactive=live,processed_fraction_of_total=processed/total,
        processed_fraction_of_noninactive=processed/live,
        processed_with_fragile_fraction_of_noninactive=(processed+fragile)/live,
        active_open_fraction_of_total=active_open/total,
        processed_native_proxy=processed/native if native>0 else None,
        tissue_band_ratio_comparable=False,
        reason='Model processed/fragile forms have no validated76/78/80kDa antibody observation map; Saito tissue ratios are not automatic pass/fail thresholds.')

def tdp_prm_observer(amounts,peptide_yields,*,compartment,fraction,amount_basis):
    """Explicit molar pools x calibrated peptide response; never infer missing CTFs.

    Rows of peptide_yields are N,C; columns match supplied amounts. User must
    supply both the molecular composition and extraction/response mapping.
    """
    if amount_basis!='molar':raise ValueError('Explicit molar pools required; N-lineage equivalents cannot be mapped automatically')
    if compartment not in ('cortex','spinal_cord','iPSC_MN'):raise ValueError('Explicit supported compartment required')
    if fraction not in ('urea_insoluble','soluble','whole_cell'):raise ValueError('Explicit extraction fraction required')
    x=np.asarray(amounts,float);m=np.asarray(peptide_yields,float)
    if x.ndim!=1 or m.shape!=(2,x.size) or not np.isfinite(x).all() or not np.isfinite(m).all() or np.any(x<0) or np.any(m<0):raise ValueError('Finite nonnegative pools and2xN yield matrix required')
    n,c=m@x
    if n<=0:raise ValueError('N-terminal signal must be positive; ratio not identifiable')
    return dict(N_signal=float(n),C_signal=float(c),C_over_N=float(c/n),compartment=compartment,fraction=fraction,
        diagnosis=None,healthy_bound=None,
        scope='Explicit observation projection only. The cortex ROC cutoff1.5 is not a universal healthy bound or a calpain-specific signal.')
