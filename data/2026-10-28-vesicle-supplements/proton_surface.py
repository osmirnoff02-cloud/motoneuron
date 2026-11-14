"""[codex] Grabe2001 SI p7: surface potentials change inferred permeability.

Isolated fixed-bulk-voltage assay. Bulk voltage is NOT the intramembrane
voltage. This does not replace the neuronal Na/K pump or install original
MATLAB/Madonna programs. The historical six-site rotor is not a human law.
"""
import numpy as np

SURFACE_RANGES=dict(cytoplasmic_surface_mV=[-50.,-25.,0.],luminal_surface_mV=[0.],bulk_voltage_mV=[-30.,0.,30.],buffer_multiplier=[.5,1.,2.])

def surface_proton_flux(pH_lumen,*,pH_cytoplasm,bulk_voltage_mV,cytoplasmic_surface_mV,luminal_surface_mV,permeability_cm_s,area_um2,temperature_K=298.15):
    vals=np.array([pH_lumen,pH_cytoplasm,bulk_voltage_mV,cytoplasmic_surface_mV,luminal_surface_mV,permeability_cm_s,area_um2,temperature_K])
    if not np.isfinite(vals).all() or min(permeability_cm_s,area_um2)<0 or temperature_K<=0:raise ValueError('Invalid surface assay')
    rtf=8.314462618*temperature_K/96485.33212*1000
    hi=10.**(-pH_lumen)*np.exp(-luminal_surface_mV/rtf)
    ho=10.**(-pH_cytoplasm)*np.exp(-cytoplasmic_surface_mV/rtf)
    u=(bulk_voltage_mV+luminal_surface_mV-cytoplasmic_surface_mV)/rtf
    a=1. if abs(u)<1e-8 else u/(-np.expm1(-u))
    b=1. if abs(u)<1e-8 else u/np.expm1(u)
    return permeability_cm_s*area_um2*1e-8*1e-3*6.02214076e23*(a*hi-b*ho)

def buffer_identifiability_transform(beta,volume,pump,leak,scale):
    """Exact degeneracy for constant volume/buffer and scaled source/fluxes.
    This produces an equally shaped pH trace; it is not parameter fitting.
    """
    if not np.isfinite([beta,volume,pump,leak,scale]).all() or min(beta,volume,scale)<=0 or min(pump,leak)<0:raise ValueError('Invalid balance')
    return dict(beta=beta*scale,volume=volume,pump=pump*scale,leak=leak*scale)
