"""Codex 2026-10-20. Assay observation equations, not new human calcium constants."""
from pathlib import Path
import json
import math

B=Path(__file__).resolve().parent

def ampa_assay_reference():
    return json.loads((B/"ampa_assay_reference_codex.json").read_text(encoding="utf-8"))

def editing_evidence():
    return json.loads((B/"editing_evidence_codex.json").read_text(encoding="utf-8"))

def _probability(x, label):
    if not math.isfinite(x) or not 0<=x<=1:
        raise ValueError(label+" must be a finite fraction in [0,1]")

def naspm_current_fraction(surface_cp_fraction, *, effective_cp_ci_ratio, cp_block_efficiency):
    """CI block=0 assumed. r includes per-channel conductance, gating and basal block.

    B=q*r*p/(1-p+r*p). p counts receptors; B measures current. Pf is absent.
    """
    _probability(surface_cp_fraction,"surface_cp_fraction")
    _probability(cp_block_efficiency,"cp_block_efficiency")
    if not math.isfinite(effective_cp_ci_ratio) or effective_cp_ci_ratio<=0:
        raise ValueError("Positive effective current ratio required")
    p,r,q=surface_cp_fraction,effective_cp_ci_ratio,cp_block_efficiency
    # Roundoff at p=1 must not manufacture blockade greater than q.
    return min(q, q*r*p/(1-p+r*p))

def surface_cp_from_naspm(block_fraction, *, effective_cp_ci_ratio, cp_block_efficiency):
    _probability(block_fraction,"block_fraction")
    _probability(cp_block_efficiency,"cp_block_efficiency")
    r,q=effective_cp_ci_ratio,cp_block_efficiency
    if not math.isfinite(r) or r<=0 or q<=0 or block_fraction>q:
        raise ValueError("Observation incompatible with assumed ratio/block; q=0 cannot identify p")
    return block_fraction/(q*r-block_fraction*(r-1))

def _check_context(context):
    expected=ampa_assay_reference()["context"]
    for key,value in expected.items():
        if context.get(key)!=value:
            raise ValueError("Incompatible Selvaraj assay context: "+key)

def compare_selvaraj_naspm(surface_cp_fraction, *, effective_cp_ci_ratio,
                           cp_block_efficiency, context):
    """Published means and reading bounds; no claim of population acceptance."""
    _check_context(context)
    ref=ampa_assay_reference()
    predicted=naspm_current_fraction(surface_cp_fraction,
        effective_cp_ci_ratio=effective_cp_ci_ratio,cp_block_efficiency=cp_block_efficiency)
    low,high=ref["mean_reading_envelope_fraction"]
    return dict(predicted_block_fraction=predicted,
        mean_reading_envelope=[low,high],
        within_mean_reading_envelope=low<=predicted<=high,
        residuals_percentage_points={r["line"]:100*predicted-r["value"] for r in ref["measured_lines"]},
        independent_validation=False,
        interpretation="Assay-specific consistency with two graph-derived means, not a biological confidence interval.")

def conditional_surface_cp_interval(*, effective_cp_ci_ratio, cp_block_efficiency, context):
    """Conditions, including q and r, must be explicit. No default narrowing."""
    _check_context(context)
    bounds=ampa_assay_reference()["mean_reading_envelope_fraction"]
    return [surface_cp_from_naspm(b,effective_cp_ci_ratio=effective_cp_ci_ratio,
             cp_block_efficiency=cp_block_efficiency) for b in bounds]

def conditional_glua2_free_weights(block_fraction, *, surface_R, glua2_copies,
                                   effective_cp_ci_ratio, cp_block_efficiency, context):
    """Conditional assembly family: w0 and one n>0 class; never uniquely inferred."""
    _check_context(context)
    _probability(surface_R,"surface_R")
    if glua2_copies not in (1,2,3,4) or surface_R==0:
        raise ValueError("Specify n=1..4 and nonzero surface R")
    p=surface_cp_from_naspm(block_fraction,effective_cp_ci_ratio=effective_cp_ci_ratio,
                           cp_block_efficiency=cp_block_efficiency)
    cp_if_glua2=(1-surface_R)**glua2_copies
    w0=(p-cp_if_glua2)/(1-cp_if_glua2)
    if not 0<=w0<=1:
        raise ValueError("Editing/assembly assumption incompatible with observed current")
    weights=[0.]*5
    weights[0]=w0; weights[glua2_copies]=1-w0
    return tuple(weights)
