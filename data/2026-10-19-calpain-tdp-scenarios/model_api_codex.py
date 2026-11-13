"""Codex 2026-10-19: additive biochemical API with unchanged electrical exports."""
from pathlib import Path
import importlib.util
import sys

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
from calpain_tdp_codex import CalpainTDPParameters, baseline_equilibrium, simulate_calpain_tdp

prior = B.parent/'2026-10-19-electrical-recovery-refinement'
sys.path.insert(0, str(prior))
spec = importlib.util.spec_from_file_location('_electrical_api_before_calpain_20261019', prior/'model_api_codex.py')
_old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _old
spec.loader.exec_module(_old)
for name in ('refined_profile', 'refined_electrical', 'run_recovery', 'preserved_electrical',
             'bursch_healthy_profile', 'bursch_calcium_scenario', 'compare_bursch_ampa_amplitude',
             'observe_calibrated_fura2', 'observe_spikes', 'nmda_fura2_response',
             'hess_dye_calibration', 'ERScenario', 'er_exchange'):
    globals()[name] = getattr(_old, name)


def bursch_calpain_scenario(time_s, influx_uM_per_s, *, parameters,
                            external_mg_mM, external_ca_mM, age_days,
                            endogenous_kappa, gamma_per_s, kinetic_provenance,
                            compartment, inhibition=0., max_step_s=2.):
    """Existing source-gated Ca model -> new conditional biochemical states.

    Uses Bursch baseline only, not a fitted Bursch calpain or cleavage rate.
    No assumption that an electrical profile has Bursch's imaging conditions.
    """
    ca = bursch_calcium_scenario(time_s, influx_uM_per_s,
        external_mg_mM=external_mg_mM, external_ca_mM=external_ca_mM, age_days=age_days,
        endogenous_kappa=endogenous_kappa, gamma_per_s=gamma_per_s,
        kinetic_provenance=kinetic_provenance)
    baseline = baseline_equilibrium(float(ca['ca_uM'][0]), parameters)
    bio = simulate_calpain_tdp(time_s, ca['ca_uM'], parameters=parameters,
        ca_provenance=ca['status']+'; '+kinetic_provenance, compartment=compartment,
        baseline=baseline, inhibition=inhibition, max_step_s=max_step_s)
    return dict(calcium=ca, biochemistry=bio,
                status='Bursch baseline only; conditional kinetics; no membrane feedback')
