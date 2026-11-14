"""Additive entry point: frozen electrical/Ca observers plus recovery and ER scenarios."""
from pathlib import Path
import importlib.util,sys
from recovery_model import run_recovery,recovery_features
from er_exchange import ERScenario,er_exchange
B=Path(__file__).resolve().parent
prior=B.parent/'2026-10-19-calcium-buffer-calibration'
sys.path.insert(0,str(prior))
spec=importlib.util.spec_from_file_location('_preserved_calcium_api_recovery',prior/'model_api.py')
_old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=_old;spec.loader.exec_module(_old)
for _name in ['preserved_electrical','bursch_healthy_profile','bursch_calcium_scenario','compare_bursch_ampa_amplitude','observe_calibrated_fura2','observe_spikes','nmda_fura2_response','hess_dye_calibration']:
    globals()[_name]=getattr(_old,_name)

def bursch_healthy_profile(*,external_mg_mM,external_ca_mM,age_days):
    """Preserve numerical anchor and condition checks; refresh source metadata."""
    result=dict(_old.bursch_healthy_profile(external_mg_mM=external_mg_mM,external_ca_mM=external_ca_mM,age_days=age_days))
    result.update(source_scope='Full article ddz107.pdf and supplements obtained 2026-10-19; prior source-scope text was historical',
                  local_pdf=str(B.parent.parent/'papers/ddz107.pdf'),
                  group='pooled healthy controls, 7 lines from 4 donors; not 7 independent donors',
                  donor_mapping_source='Naujock 2016, 10.1002/stem.2354, not Naujock 2014',
                  provenance_update='2026-10-19; baseline, uncertainty and all numerical parameters unchanged')
    return result
