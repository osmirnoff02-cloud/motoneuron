"""[codex] 2026-10-20. Explicit replay of coupled free Ca into calpain/TDP.

Preserves the prior electrical API. Biochemistry does not feed back to electrical
parameters. Local/bulk are alternative site assumptions, not additive pools.
No extrapolation, time dilation, injury threshold or human biochemical fit.
"""
from pathlib import Path
import importlib.util
import sys
import numpy as np

B = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "_coupled_before_tdp", B.parent/"2026-10-20-coupled-ampa-calcium-spikes/model_api_codex.py")
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
for _name in dir(previous):
    if not _name.startswith("_") and _name not in globals():
        globals()[_name] = getattr(previous, _name)


def coupled_trace_to_tdp(trace, *, parameters, compartment, basal_ca_uM,
                         inhibition=0., max_step_s=.01):
    """Convert t in ms to seconds exactly once; cs/cb are absolute free uM.

    trace is the result of run_coupled/run_compatible or a stored equivalent.
    The unblocked equilibrium and synthesis are shared by intervention arms.
    Inhibition is fractional catalytic block, NOT a drug dose.
    """
    if compartment not in ("bulk", "local"):
        raise ValueError("Choose explicit bulk or local compartment")
    t = np.asarray(trace["t"], dtype=float)
    ca = np.asarray(trace["cb" if compartment == "bulk" else "cs"], dtype=float)
    if (not np.isfinite(basal_ca_uM) or basal_ca_uM < 0 or
            abs(ca[0]-basal_ca_uM) > 1e-7):
        raise ValueError("The shared baseline must match the initial free calcium")
    out = simulate_calpain_tdp(
        t/1000., ca, parameters=parameters,
        ca_provenance="NEURON AMPA/VGCC/Ca/SK trajectory; unchanged absolute time and Ca",
        compartment=compartment+" scenario; calpain location not measured in human iPSC-MN",
        baseline=baseline_equilibrium(basal_ca_uM, parameters),
        inhibition=inhibition, max_step_s=max_step_s)
    out.update(input_duration_s=float((t[-1]-t[0])/1000.),
               time_extrapolated=False, calpain_to_voltage_feedback=False,
               damage_definition="intact TDP loss and cleavage; not aggregation, death or NMJ")
    return out


def coupled_calpain_scenario(profile, scenario_id, surface_Q, *, biochemistry,
                             compartment, inhibition=0., **protocol):
    """Run the accepted electrical model, then replay its calcium downstream."""
    electric = run_compatible(profile, scenario_id, surface_Q, **protocol)
    chemistry = coupled_trace_to_tdp(
        electric, parameters=biochemistry, compartment=compartment,
        basal_ca_uM=electric["parameters"]["basal_ca_uM"], inhibition=inhibition)
    return dict(electrical=electric, biochemistry=chemistry,
                status="conditional connected scenario; no human biochemical calibration")
