"""Codex source-observer extension; explicit entry avoids concurrent API edits.

Both previous physiology and local observation helpers are imported by path.
The separate model_api.py in this directory is preserved as found.
"""
from pathlib import Path
import importlib.util,sys
B=Path(__file__).resolve().parent

def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

_old=_load('_equal_api_for_observers_20261019',B.parent/'2026-10-19-equal-calcium-load/model_api.py')
for _name in ('equal_entry_protocol','calcium_equal_entry','inward_current_to_ca_flux','observe_cleavage_reporter',
    'observe_pulse_chase','enzyme_driven_chase','simulate_equal_load','CalpainTDPParameters','baseline_equilibrium',
    'simulate_calpain_tdp','bursch_calpain_scenario','refined_profile','refined_electrical','run_recovery',
    'preserved_electrical','bursch_healthy_profile','bursch_calcium_scenario','compare_bursch_ampa_amplitude',
    'observe_calibrated_fura2','observe_spikes','nmda_fura2_response','hess_dye_calibration','ERScenario','er_exchange'):
    globals()[_name]=getattr(_old,_name)
_observations=_load('_explicit_source_observations_20261019',B/'source_observers.py')
for _name in ('unmix_fura_fret','renewing_reporter','sample_photometry','competing_tdp_chase','observe_fragment_band'):
    globals()[_name]=getattr(_observations,_name)
_optical=_load('_explicit_optical_qc_20261019',B/'optical_qc.py')
optical_event_features=_optical.optical_event_features
calibrate_negative_control=_optical.calibrate_negative_control
