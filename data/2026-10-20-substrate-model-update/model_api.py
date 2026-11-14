"""[codex] 2026-10-20. Substrate-specific extension of the coupled neuron.

The same simulated calpain activity drives independent, unsaturated substrates.
No conversion of SNTF/p25 into TDP cleavage, neuronal death, or electrical rescue.
MPS redistribution is an explicit static observation hypothesis, not an axon mesh.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util
import sys
import json
import numpy as np
from numba import njit

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_coupled_before_substrates20261020",
    _HERE.parent / "2026-10-20-coupled-calpain-tdp/model_api.py")
_previous = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _previous
_spec.loader.exec_module(_previous)
for _name in dir(_previous):
    if not _name.startswith("_") and _name not in globals():
        globals()[_name] = getattr(_previous, _name)

SUBSTRATES = {
    "alphaII_spectrin": "SNTF-compatible tracked cleavage product",
    "betaII_spectrin": "tracked calpain product; not alphaII SNTF",
    "p35": "p25; not a measurement of Cdk5 kinase activity",
}


@dataclass(frozen=True)
class SubstrateParameters:
    cleavage_per_activity_s: float
    intact_half_life_s: float
    fragment_half_life_s: float
    provenance: str

    def vector(self):
        p = [self.cleavage_per_activity_s, self.intact_half_life_s,
             self.fragment_half_life_s]
        if (not np.isfinite(p).all() or p[0] < 0 or min(p[1:]) <= 0
                or not isinstance(self.provenance, str) or not self.provenance.strip()):
            raise ValueError("Explicit finite rates/half-lives and provenance required")
        return np.array([p[0], np.log(2)/p[1], np.log(2)/p[2]])


@njit
def _substrate_rhs(y, activity, p, synthesis):
    cleavage = p[0]*activity*y[0]
    du, df = p[1]*y[0], p[2]*y[1]
    return np.array([synthesis-du-cleavage, cleavage-df,
                     du+df, synthesis, cleavage])


@njit
def _substrate_integrate(t, a, multiplier, p, y0, synthesis, max_step):
    y = np.empty((len(t), 5))
    y[0] = y0
    safe = min(max_step, .1/max(p[0]+p[1], p[2]))
    for i in range(len(t)-1):
        steps = max(1, int(np.ceil((t[i+1]-t[i])/safe)))
        h = (t[i+1]-t[i])/steps
        v = y[i].copy()
        for j in range(steps):
            # Activation is continuous; inhibitor commands are left-constant.
            a0 = (a[i]+(a[i+1]-a[i])*j/steps)*multiplier[i]
            am = (a[i]+(a[i+1]-a[i])*(j+.5)/steps)*multiplier[i]
            a1 = (a[i]+(a[i+1]-a[i])*(j+1)/steps)*multiplier[i]
            k1 = _substrate_rhs(v, a0, p, synthesis)
            k2 = _substrate_rhs(v+h*k1/2, am, p, synthesis)
            k3 = _substrate_rhs(v+h*k2/2, am, p, synthesis)
            k4 = _substrate_rhs(v+h*k3, a1, p, synthesis)
            v += h*(k1+2*k2+2*k3+k4)/6
        y[i+1] = v
    return y


def substrate_equilibrium(basal_effective_activity, parameters):
    """Shared unblocked baseline, precursor=1; not a healthy experimental fit."""
    p = parameters.vector()
    a = float(basal_effective_activity)
    if not np.isfinite(a) or not 0 <= a <= 1:
        raise ValueError("Effective activity must use the coupled model's [0,1] scale")
    return dict(initial_state=np.array([1., p[0]*a/p[2], 0., 0., 0.]),
                synthesis_per_s=float(p[1]+p[0]*a))


def simulate_substrate_reporter(time_s, activity, *, substrate, parameters,
                               baseline, activity_multiplier=1., max_step_s=.01):
    """One retained product per precursor cleavage; lineage accounting, not mass.

    Does not model secondary spectrin/p25 cleavage or substrate competition.
    These rates have no fitted human values. Baseline must be shared by arms.
    activity is interpolated linearly; activity_multiplier is a left-constant
    fractional command. Put all abrupt inhibitor switches in the time vector.
    """
    if substrate not in SUBSTRATES or not isinstance(parameters, SubstrateParameters):
        raise ValueError("Explicit supported substrate and SubstrateParameters required")
    t, a = np.asarray(time_s, float), np.asarray(activity, float)
    if (t.ndim != 1 or len(t) < 2 or a.shape != t.shape or
            not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or
            not np.isfinite(a).all() or np.any((a < 0) | (a > 1))):
        raise ValueError("Increasing seconds and matching [0,1] calpain activity required")
    mult = (np.full(t.shape, float(activity_multiplier)) if np.ndim(activity_multiplier)==0
            else np.asarray(activity_multiplier, float))
    if (mult.shape != t.shape or not np.isfinite(mult).all()
            or np.any((mult < 0) | (mult > 1))):
        raise ValueError("Multiplier must be finite [0,1], scalar or matching time")
    y0 = np.asarray(baseline["initial_state"], float)
    synthesis = float(baseline["synthesis_per_s"])
    if (y0.shape != (5,) or not np.isfinite(y0).all() or np.any(y0 < 0)
            or not np.isfinite(synthesis) or synthesis < 0
            or not np.isfinite(max_step_s) or max_step_s <= 0):
        raise ValueError("Invalid baseline or integration step")
    y = _substrate_integrate(t, a, mult, parameters.vector(), y0, synthesis, max_step_s)
    balance = y[:, :3].sum(axis=1)-y[:, 3]
    error = float(np.max(abs(balance-balance[0])))
    if (not np.isfinite(y).all() or y.min() < -1e-10 or
            error > 1e-8*max(1., float(np.max(y[:, :3].sum(axis=1))))):
        raise ArithmeticError("Invalid substrate trajectory or lineage imbalance")
    return dict(time_s=t, intact=y[:, 0], product=y[:, 1], removed=y[:, 2],
                synthesized=y[:, 3], cleavage_events=y[:, 4],
                substrate=substrate, product_definition=SUBSTRATES[substrate],
                parameters=asdict(parameters), lineage_balance_error=error,
                status="conditional independent substrate; no human kinetic fit",
                predicts_viability=False, predicts_Cdk5_activity=False)


def coupled_trace_to_substrates(trace, *, parameters, compartment, basal_ca_uM,
                               substrate_parameters, inhibition=0., max_step_s=.01):
    """Keep original Ca/TDP dynamics; add separately parameterized substrates."""
    if not isinstance(substrate_parameters, dict) or not substrate_parameters:
        raise ValueError("An explicit nonempty substrate parameter mapping is required")
    if any(k not in SUBSTRATES for k in substrate_parameters):
        raise ValueError("Unknown substrate; no implicit assignment by band mass")
    tdp = _previous.coupled_trace_to_tdp(
        trace, parameters=parameters, compartment=compartment,
        basal_ca_uM=basal_ca_uM, inhibition=inhibition, max_step_s=max_step_s)
    # Baseline is from unblocked equilibrium even for the treated arm.
    base = _previous.baseline_equilibrium(basal_ca_uM, parameters)
    basal_activity = base["initial_state"][0]/(1+parameters.cast_over_ki)
    outputs = {}
    for name, p in substrate_parameters.items():
        outputs[name] = simulate_substrate_reporter(
            tdp["time_s"], tdp["active_fraction"]/(1+parameters.cast_over_ki),
            activity_multiplier=1-tdp["inhibition"], substrate=name, parameters=p,
            baseline=substrate_equilibrium(basal_activity, p), max_step_s=max_step_s)
    return dict(TDP=tdp, substrates=outputs,
                nuclear_TDP_function=None, neuronal_viability=None, NMJ_function=None,
                status="common calpain input; independent substrate kinetics",
                biochemical_feedback_to_voltage=False,
                substrate_competition_modeled=False)


def mps_distribution(intact_betaII, *, gap_fraction, gap_to_patch_ratio, provenance):
    """Static two-region density observation; no protein destruction is needed.

    betaII mean = gap_fraction*gap_density + (1-gap_fraction)*patch_density.
    The ratio may be constrained by Gazal's reported ~0.2-0.3, but gap_fraction
    and assembly dynamics are not measured by that ratio. This function neither
    predicts gaps from calpain nor treats the pattern as neuronal damage.
    """
    mean = np.asarray(intact_betaII, float)
    f, r = float(gap_fraction), float(gap_to_patch_ratio)
    if (not np.isfinite(mean).all() or np.any(mean < 0) or
            not np.isfinite([f, r]).all() or not 0 < f < 1 or not 0 <= r <= 1 or
            not isinstance(provenance, str) or not provenance.strip()):
        raise ValueError("Nonnegative amount, explicit region fraction/ratio/provenance required")
    patch = mean/(1-f+f*r)
    gap = r*patch
    return dict(mean_betaII=mean, gap_density=gap, patch_density=patch,
                gap_fraction=f, gap_to_patch_ratio=r, provenance=provenance,
                proteolysis_inferred=False, neuronal_damage=None,
                status="static redistribution scenario, not a fitted axonal morphology")


def substrate_evidence():
    return json.loads((_HERE/"evidence.json").read_text(encoding="utf-8"))


def source_training_data():
    result = _previous.source_training_data()
    result["independent_substrate_and_function_observations"] = substrate_evidence()
    return result


def source_readout_reference(source_id):
    """Return actual assay values without inventing a Ca -> readout mapping."""
    return substrate_evidence()["source_readouts"][source_id]
