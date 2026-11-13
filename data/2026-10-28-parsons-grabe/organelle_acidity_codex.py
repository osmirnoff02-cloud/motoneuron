"""[codex] Grabe--Oster2001: reduced, voltage-clamped proton-balance assay.

This is NOT a replication of their full V-ATPase/NaK/counterion model and
NOT a second pH state attached to the live vesicle queues. Eq7f establishes
the buffer/volume conversion. Eq2 motivates passive proton leakage.
No human cell values are calibrated. Pump flux is externally prescribed.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

NA=6.02214076e23

@dataclass(frozen=True)
class ProtonBalanceParameters:
    volume_um3: float = 6.
    area_um2: float = 800.
    buffer_mM_per_pH: float = 10.
    permeability_cm_s: float = 8.1e-6
    cytoplasmic_pH: float = 7.4
    voltage_mV: float = 0.
    temperature_K: float = 298.15
    provenance: str = 'Grabe2001 TableIV Golgi scale; geometry rat kidney, leak inferred using HeLa pH. SI clarification: published leak includes surface-charge assumptions; this legacy zero-surface assay is not a reproduction of that fit. Voltage0/temperature25C are scenario assumptions. Not human motor-neuron vesicles.'

    def validate(self):
        vals=[self.volume_um3,self.area_um2,self.buffer_mM_per_pH,self.permeability_cm_s,self.cytoplasmic_pH,self.voltage_mV,self.temperature_K]
        if not np.isfinite(vals).all() or min(vals[0:3]+[vals[6]])<=0 or self.permeability_cm_s<0 or not self.provenance:raise ValueError('Invalid proton balance parameters')


# Scenario multipliers / clamps, NOT measured confidence limits.
PROTON_SCENARIO_RANGES=dict(buffer_multiplier=[.5,1.,2.],volume_multiplier=[.5,1.,2.],area_multiplier=[.5,1.,2.],
                          pump_support=[0.,.5,1.],leak_multiplier=[.5,1.,2.],voltage_mV=[-30.,0.,30.],
                          temperature_K=[294.15,298.15,310.15])

def proton_capacity(p):
    p.validate()
    return p.buffer_mM_per_pH*1e-3*p.volume_um3*1e-15*NA


def outward_proton_flux(pH,p):
    """GHK outward H+/s; positive lumen voltage drives cations outward.

    Uses bulk concentrations, no membrane surface-charge corrections.
    M converted to mol/cm3; um2 to cm2. Valid under a fixed-voltage clamp.
    """
    hi=10.**(-pH);ho=10.**(-p.cytoplasmic_pH)
    u=p.voltage_mV*1e-3*96485.33212/(8.314462618*p.temperature_K)
    a=1. if abs(u)<1e-8 else u/(-np.expm1(-u))
    b=1. if abs(u)<1e-8 else u/np.expm1(u)
    return p.permeability_cm_s*p.area_um2*1e-8*1e-3*NA*(a*hi-b*ho)


def simulate_proton_balance(times,*,initial_pH,pump_H_per_s,parameters=None):
    p=parameters or ProtonBalanceParameters();p.validate()
    t=np.asarray(times,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0) or not np.isfinite([initial_pH,pump_H_per_s]).all() or pump_H_per_s<0:raise ValueError('Invalid protocol')
    cap=proton_capacity(p)
    def rhs(_,z):
        leak=outward_proton_flux(z[0],p)
        return [-(pump_H_per_s-leak)/cap,pump_H_per_s,leak]
    sol=solve_ivp(rhs,(t[0],t[-1]),[initial_pH,0.,0.],t_eval=t,rtol=1e-10,atol=1e-11,method='LSODA')
    if not sol.success:raise RuntimeError(sol.message)
    h,pumped,leaked=sol.y
    balance=cap*(initial_pH-h)-pumped+leaked
    return dict(time_s=t,pH=h,pumped_H=pumped,leaked_H=leaked,proton_balance_max_H=float(np.max(abs(balance))),
                limitations='Voltage-clamped isolated assay; prescribed pump flux; no fitted V-ATPase surface, counterion/osmotic dynamics or ATP feedback.')
