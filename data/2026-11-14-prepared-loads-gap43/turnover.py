"""Codex 2026-11-14. Explicit local GAP43 turnover intervention, not a fit.

The old degradation flux already exists. Add only (multiplier - 1) of that
flux, including its peptide, removal and ATP ledgers. Default 1 is a no-op.
No initial pool, reference concentration, fingerprint or clock is reset.
"""
import numpy as np
from numba import njit

def resolve(value, checkpoint=None, has_GAP43=True):
    prior=(checkpoint or {}).get('diagnostic_interventions',{})
    m=float(prior.get('local_GAP43_turnover_multiplier',1.) if value is None else value)
    if not np.isfinite(m) or not 0<=m<=4:
        raise ValueError('Local GAP43 turnover multiplier must be finite in [0,4]')
    if not has_GAP43 and m!=1:
        raise ValueError('Local GAP43 turnover requires the local-mRNA cargo branch')
    return m

@njit
def apply(z, out, local_parameters, ATP_half_uM, multiplier):
    if multiplier==1.:return out
    ATP=max(z[136],0.)
    delta=(multiplier-1.)*np.log(2)/(local_parameters[21]*3600)*max(z[583],0.)*ATP/(ATP_half_uM+ATP)
    residues=local_parameters[0]*delta
    cost=local_parameters[23]*residues
    out[583]-=delta
    out[584]+=residues
    out[596]+=delta
    out[598]+=cost
    out[136]-=cost
    out[137]+=cost
    out[150]+=cost
    return out
