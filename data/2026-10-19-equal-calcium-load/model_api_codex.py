"""[codex] 2026-10-19: additive assay-aware/equal-dose API; electrical exports preserved."""
from pathlib import Path
import importlib.util,sys
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
from equal_load_codex import (equal_entry_protocol,calcium_equal_entry,inward_current_to_ca_flux,
    observe_cleavage_reporter,observe_pulse_chase,enzyme_driven_chase,simulate_equal_load)
prior=B.parent/'2026-10-19-calpain-tdp-scenarios'
spec=importlib.util.spec_from_file_location('_calpain_api_before_equal_load_20261019',prior/'model_api_codex.py')
_old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=_old;spec.loader.exec_module(_old)
for name in ('CalpainTDPParameters','baseline_equilibrium','simulate_calpain_tdp','bursch_calpain_scenario',
    'refined_profile','refined_electrical','run_recovery','preserved_electrical','bursch_healthy_profile',
    'bursch_calcium_scenario','compare_bursch_ampa_amplitude','observe_calibrated_fura2','observe_spikes',
    'nmda_fura2_response','hess_dye_calibration','ERScenario','er_exchange'):
    globals()[name]=getattr(_old,name)
