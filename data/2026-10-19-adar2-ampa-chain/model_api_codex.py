"""[codex] 2026-10-19. Additive ADAR2 scenarios; both observation APIs retained."""
from pathlib import Path
import importlib.util
import sys
B = Path(__file__).resolve().parent
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
source_api = _load('_source_api_for_adar2', B.parent/'2026-10-19-source-model-update/model_api_codex.py')
observer_api = _load('_observer_api_for_adar2', B.parent/'2026-10-19-source-model-update/model_api_observers_codex.py')
for _name in source_api._INHERITED + ('source_training_data','apply_boivin_drift','simulate_substrate',
        'simulate_fluorescent_intact','robinson_fret','epitope_linear_mixture','gerencser_corrected_fret',
        'observe_biochemistry_substrate'):
    globals()[_name] = getattr(source_api, _name)
for _name in ('unmix_fura_fret','renewing_reporter','sample_photometry','competing_tdp_chase',
              'observe_fragment_band','optical_event_features','calibrate_negative_control'):
    globals()[_name] = getattr(observer_api, _name)
_chain = _load('_adar2_ampa_chain_20261019', B/'adar2_ampa_codex.py')
for _name in ('AMPAParameters','editing_target','editing_trajectory','cp_fraction','ampa_calcium',
              'paired_chain','chain_summary'):
    globals()[_name] = getattr(_chain, _name)
