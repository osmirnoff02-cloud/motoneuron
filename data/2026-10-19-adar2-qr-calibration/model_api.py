"""Codex 2026-10-20. Unified additive API: legacy physiology + assay constraints.

No default electrical, ADAR2, receptor, calcium or proteolysis parameter changed.
"""
from pathlib import Path
import importlib.util
import sys

B=Path(__file__).resolve().parent

def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

legacy=_load("_legacy_chain_for_qr20261020",B.parent/"2026-10-19-adar2-ampa-chain/model_api.py")
evidence_api=_load("_onodera_for_qr20261020",B.parent/"2026-10-19-onodera-hirunagi/model_api.py")
_names=(*legacy.source_api._INHERITED,
    "apply_boivin_drift","simulate_substrate","simulate_fluorescent_intact",
    "robinson_fret","epitope_linear_mixture","gerencser_corrected_fret","observe_biochemistry_substrate",
    "unmix_fura_fret","renewing_reporter","sample_photometry","competing_tdp_chase",
    "observe_fragment_band","optical_event_features","calibrate_negative_control",
    "AMPAParameters","editing_target","editing_trajectory","cp_fraction","ampa_calcium",
    "paired_chain","chain_summary")
for _name in _names:
    globals()[_name]=getattr(legacy,_name)
for _name in ("calibration_evidence","healthy_gene_reference","compare_hirunagi_optical"):
    globals()[_name]=getattr(evidence_api,_name)
_constraints=_load("_ampa_observer_for_qr20261020",B/"ampa_constraints.py")
for _name in ("ampa_assay_reference","editing_evidence","naspm_current_fraction",
              "surface_cp_from_naspm","compare_selvaraj_naspm",
              "conditional_surface_cp_interval","conditional_glua2_free_weights"):
    globals()[_name]=getattr(_constraints,_name)

def source_training_data():
    sources=evidence_api.source_training_data()
    sources["ADAR2_QR_evidence"]=editing_evidence()
    sources["Selvaraj_AMPAR_assay"]=ampa_assay_reference()
    return sources
