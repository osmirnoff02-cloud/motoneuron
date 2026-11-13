"""[codex] Explicit prepared candidate, preserving the entire physical history."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent

def prepared_checkpoint(profile='Burley'):
    """Return all4413 states AND original baseline; never promote to healthy norm.

    Caller creates options with q3_baseline_inputs(..., same profile). This
    checkpoint uses the legacy145mM extracellular Na scenario. It is not a
    dialysis/patch-condition conversion. The integrated runner verifies its
    configuration fingerprint and state schema before continuation.
    """
    manifest=json.loads((B/'prepared_states_manifest_codex.json').read_text(encoding='utf8'))
    if profile not in manifest['profiles']:raise ValueError('Prepared candidate supports only listed Burley/Selvaraj profiles')
    entry=manifest['profiles'][profile];p=B/entry['checkpoint'];raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Prepared checkpoint changed; revalidate its provenance before use')
    cp=json.loads(raw)
    if len(cp['state'])!=4413 or cp['time_s']!=312*3600:raise ValueError('Unexpected prepared checkpoint topology/time')
    return cp,dict(profile=profile,source=entry,healthy_human_calibrated=False,
      full_network_stationarity_validated=False,preparation_is_not_culture_age=True,
      extracellular_Na_mM=145.,temperature_C=23.,reset=False)
