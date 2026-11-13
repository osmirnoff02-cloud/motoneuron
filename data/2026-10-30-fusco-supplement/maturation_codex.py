"""[codex]2026-10-30. Independent late40S/60S interventions, not abundance fits.
Fusco2025 supports stage distinctions; human rates retain prior uncertainty.
No local axon compartment added. Source enrichments never become rate constants.
"""
from pathlib import Path
import json,numpy as np
from numba import njit
B=Path(__file__).resolve().parent
MATURATION_RANGES=dict(late40_support=[0.,.25,1.],late60_support=[0.,.25,1.])

def controls(segments,supplied=None):
    sg=np.asarray(segments,float)
    a=np.c_[sg[:,:2],np.ones((len(sg),2))] if supplied is None else np.asarray(supplied,float)
    if a.shape!=(len(sg),4) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>1):
        raise ValueError('Late maturation schedule: start/end, late40/late60 supports in [0,1], same boundaries as neuron segments')
    return a.copy()

@njit
def apply_maturation(out,support):
    # Production counters contain the existing ATP- and stress-gated fluxes.
    # Move identical amount from mature production back into its precursor.
    # Existing rRNA/RP/assembly costs stay charged exactly once. The current
    # model has no resolved late-maturation GTP/ATP cost; do not invent one.
    for j in range(2):
        old=out[396+j];new=old*support[j];delta=new-old
        out[330+j]+=delta
        out[390+j]-=delta
        out[396+j]=new
    return out

def factor_evidence(gene=None):
    d=json.loads((B/'ribosome_factor_evidence_codex.json').read_text(encoding='utf8'))
    if gene is not None:d['groups']=[x for x in d['groups'] if x['gene'].casefold()==str(gene).casefold()]
    return d
