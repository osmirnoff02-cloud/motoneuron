"""[codex] Explicit exchanging geometry, not a healthy-human calibration."""
from dataclasses import replace
from recovery_inputs import recovery_neuron_inputs
def energy_terminal_inputs(profile='Burley',*,adenylate_diffusion_um2_s=100.,**kwargs):
    result=recovery_neuron_inputs(profile,**kwargs)
    result['geometry']=replace(result['geometry'],adenylate_diffusion_um2_s=adenylate_diffusion_um2_s)
    result['geometry'].validate()
    return result
