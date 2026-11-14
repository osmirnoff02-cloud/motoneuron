"""[codex] 2026-10-20. Independent AMPAR composition/capacity and rat assay models.

Legacy human electrical profiles and physiological defaults are unchanged.
RNA abundance is NOT a surface protein fraction; P_Ca/P_Cs is NOT calcium charge Pf.
"""
from pathlib import Path
import importlib.util
import sys
import json
import math
from dataclasses import replace
import numpy as np

B = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_qr_before_ampa20261020", B.parent / "2026-10-19-adar2-qr-calibration/model_api.py")
_prior = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _prior
_spec.loader.exec_module(_prior)
for _name in (*_prior._names, "calibration_evidence", "healthy_gene_reference",
              "compare_hirunagi_optical", "ampa_assay_reference", "editing_evidence",
              "naspm_current_fraction", "surface_cp_from_naspm", "compare_selvaraj_naspm",
              "conditional_surface_cp_interval", "conditional_glua2_free_weights"):
    globals()[_name] = getattr(_prior, _name)


def ampa_new_evidence():
    return json.loads((B / "evidence.json").read_text(encoding="utf-8"))


def source_training_data():
    sources = _prior.source_training_data()
    sources["AMPAR_density_permeability_and_SOD1_editing"] = ampa_new_evidence()
    return sources


def _fraction(value, name):
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + " must be a finite fraction")
    return float(value)


def random_tetramer_weights(surface_glua2_fraction, *, abundance_kind):
    """Optional random assembly, NOT a fit of receptor stoichiometry.

    f is the GluA2 fraction among all incorporated surface AMPAR subunits.
    Weights specify n=0..4 GluA2 copies, regardless of their Q/R editing.
    Independent assembly + any R dominates implies CP = (1-f*surface_R)**4.
    """
    if abundance_kind != "surface_subunit_fraction":
        raise ValueError("Requires surface protein subunit fraction; RNA fold is not this quantity")
    f = _fraction(surface_glua2_fraction, "Surface GluA2")
    return tuple(math.comb(4, n) * f**n * (1-f)**(4-n) for n in range(5))


def composition_parameters(p, *, receptor_capacity_scale=1., all_ampa_block=0.,
                           surface_glua2_fraction=None, abundance_kind=None):
    """Change functional receptor capacity at FIXED geometry; optionally composition.

    Capacity scales both CP and CI populations. It can represent current-density
    scaling only when area, voltage, single-channel conductance and gating are held
    fixed. It is not a measured count of receptors. cp_block remains separate.
    """
    if not np.isfinite(receptor_capacity_scale) or receptor_capacity_scale < 0:
        raise ValueError("Nonnegative finite receptor capacity required")
    block = _fraction(all_ampa_block, "All-AMPAR blockade")
    weights = p.weights
    if surface_glua2_fraction is not None:
        weights = random_tetramer_weights(surface_glua2_fraction, abundance_kind=abundance_kind)
    elif abundance_kind is not None:
        raise ValueError("Abundance kind supplied without a surface abundance")
    return replace(p, weights=weights,
                   g_reference_nS=p.g_reference_nS * receptor_capacity_scale * (1-block))


def ampa_composition_calcium(surface_R, p=None, *, receptor_capacity_scale=1.,
                             all_ampa_block=0., surface_glua2_fraction=None,
                             abundance_kind=None, **protocol):
    p = AMPAParameters() if p is None else p
    modified = composition_parameters(p, receptor_capacity_scale=receptor_capacity_scale,
        all_ampa_block=all_ampa_block, surface_glua2_fraction=surface_glua2_fraction,
        abundance_kind=abundance_kind)
    result = ampa_calcium(surface_R, modified, **protocol)
    result["composition_assumptions"] = dict(
        receptor_capacity_scale=receptor_capacity_scale, all_ampa_block=all_ampa_block,
        surface_glua2_fraction=surface_glua2_fraction,
        assembly="random tetramer" if surface_glua2_fraction is not None else "legacy fixed weights",
        measured_human_fit=False)
    return result


def paired_composition_chain(api, p, biop, *, receptor_capacity_scale=1.,
                              all_ampa_block=0., surface_glua2_fraction=None,
                              abundance_kind=None, **protocol):
    """New composition/capacity branch feeds the existing Ca->calpain->TDP chain."""
    modified = composition_parameters(p, receptor_capacity_scale=receptor_capacity_scale,
        all_ampa_block=all_ampa_block, surface_glua2_fraction=surface_glua2_fraction,
        abundance_kind=abundance_kind)
    result = paired_chain(api, modified, biop, **protocol)
    result["composition_assumptions"] = dict(
        receptor_capacity_scale=receptor_capacity_scale, all_ampa_block=all_ampa_block,
        surface_glua2_fraction=surface_glua2_fraction, measured_human_fit=False,
        unchanged_geometry=True, endogenous_ADAR2_activity_not_inferred_from_RNA=True)
    return result


def _biionic_constants(ca_external_mM, temperature_C):
    if ca_external_mM not in (15., 50.):
        raise ValueError("Only the two Na-free Vandenberghe assay baths are specified")
    if not np.isfinite(temperature_C) or not 15 <= temperature_C <= 30:
        raise ValueError("Supply an explicit room-temperature assumption in 15..30 C")
    cs_activity = 132. * .718
    mg_activity = 3. * .416
    ca_activity = ca_external_mM * (.496 if ca_external_mM == 15. else .357)
    a = cs_activity + 4 * .8 * mg_activity
    rt_over_f_mV = 1000 * 8.314462618 * (temperature_C + 273.15) / 96485.33212
    return a, cs_activity, ca_activity, rt_over_f_mV


def rat_biionic_reversal(p_ca_over_cs, *, ca_external_mM, temperature_C):
    """Extended constant-field equation; activities and P_Mg/P_Cs=.8 as published."""
    if not np.isfinite(p_ca_over_cs) or p_ca_over_cs <= 0:
        raise ValueError("Positive finite relative permeability required, NOT a Pf fraction")
    a, b, ca, thermal = _biionic_constants(ca_external_mM, temperature_C)
    c = -4 * p_ca_over_cs * ca
    # Stable positive root of a*x*x + b*x + c = 0.
    x = -2*c / (b + np.sqrt(b*b - 4*a*c))
    return float(thermal * np.log(x))


def rat_biionic_permeability(reversal_mV, *, ca_external_mM, temperature_C):
    if not np.isfinite(reversal_mV) or abs(reversal_mV) > 200:
        raise ValueError("Reversal outside the supported finite assay domain")
    a, b, ca, thermal = _biionic_constants(ca_external_mM, temperature_C)
    x = np.exp(reversal_mV / thermal)
    return float((a*x*x + b*x) / (4*ca))


def rat_isolated_glutamate_response(time_ms, *, pulse_ms, context):
    """Empirical peak-normalized response; rat 3 mM glutamate, 1 or 100 ms.

    Replays reported means, not independent biological validation. Instantaneous
    rise is an approximation. No recovery-from-desensitization is inferred, hence
    this is deliberately a SINGLE pulse API, not a repeated-train human channel.
    """
    reference = ampa_new_evidence()["rat_kinetics"]
    if context != reference["context"]:
        raise ValueError("Species, agonist, bath/voltage and recording context must match")
    if pulse_ms not in (1., 100.):
        raise ValueError("Only published 1-ms and 100-ms isolated pulse protocols")
    t = np.asarray(time_ms, float)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Finite strictly increasing one-dimensional time axis required")
    fast, slow = reference["tau_fast_ms"]["mean"], reference["tau_slow_ms"]["mean"]
    w, residual = reference["fast_weight"]["mean"], reference["residual_fraction"]["mean"]
    def during(s):
        return residual + (1-residual)*(w*np.exp(-s/fast)+(1-w)*np.exp(-s/slow))
    current = np.zeros_like(t)
    active = (t >= 0) & (t <= pulse_ms)
    current[active] = during(t[active])
    after = t > pulse_ms
    current[after] = during(pulse_ms)*np.exp(
        -(t[after]-pulse_ms)/reference["deactivation_ms"]["mean"])
    return dict(time_ms=t, normalized_inward_current=current, pulse_ms=pulse_ms,
                context=context, calcium_charge_fraction=None,
                recovery_from_desensitization_ms=None,
                interpretation="Rat source replay; not human calibration or absolute Ca influx")
