"""[codex] Separate assay constraints, never automatic human parameters."""
from pathlib import Path
import json,numpy as np
B=Path(__file__).resolve().parent
def source_constraints():
 return json.loads((B/'source_constraints_codex.json').read_text(encoding='utf8'))
def recovery_assay(voltage_mV,rin_MOhm,*,holding_mV,baseline_rin_MOhm):
 """Distinguish hyperpolarization from shunting; mechanism still needs blockers."""
 v=np.asarray(voltage_mV,dtype=float);r=np.asarray(rin_MOhm,dtype=float)
 if v.shape!=r.shape or not np.isfinite(v).all() or not np.isfinite(r).all() or np.any(r<=0) or not np.isfinite([holding_mV,baseline_rin_MOhm]).all() or baseline_rin_MOhm<=0:raise ValueError('Paired finite voltage/positive Rin required')
 return dict(voltage_shift_mV=v-holding_mV,Rin_fraction=r/baseline_rin_MOhm,
  scope='AHP alone does not separate pump and Kv7. Use paired Rin and pharmacological difference currents; no molecular classification from these two numbers alone.')
