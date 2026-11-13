"""[codex] Absolute-current assays and physical rest-closure constraints.

This explicit submodel does NOT replace the differential KCNQ in the full
neuron. A fitted cortical tail amplitude is not a human-MN conductance.
Currents are outward positive, pA; voltages mV; conductances nS.
"""
from pathlib import Path
import json
import numpy as np
FCONV=1e9/96485.33212  # pA / um3 -> uM/s

def source_constraints():
    return json.loads((Path(__file__).parent/'absolute_source_constraints_codex.json').read_text(encoding='utf8'))

def gate(voltage_mV, vhalf_mV=-27.3966, slope_mV=11.7788):
    if not np.isfinite([voltage_mV,vhalf_mV,slope_mV]).all() or slope_mV<=0:
        raise ValueError('Finite voltage and positive slope required')
    return float(1/(1+np.exp(np.clip((vhalf_mV-voltage_mV)/slope_mV,-700,700))))

def tail_current(g_nS, EK_mV, *, tau_s, vhalf_mV=-27.3966, slope_mV=11.7788,
                 holding_mV=-70., prepulse_mV=20., tail_mV=-50., prepulse_s=1., tail_s=.5):
    """End-minus-start tail amplitude; analytic first-order gate, no capacitive transient.

    Constant tau and steady holding are explicit approximations. No automatic
    temperature transfer, leak subtraction or ion-composition inference.
    """
    vals=[g_nS,EK_mV,tau_s,holding_mV,prepulse_mV,tail_mV,prepulse_s,tail_s]
    if not np.isfinite(vals).all() or g_nS<0 or min(tau_s,prepulse_s,tail_s)<=0:
        raise ValueError('Nonnegative conductance and positive finite timing required')
    h=gate(holding_mV,vhalf_mV,slope_mV);p=gate(prepulse_mV,vhalf_mV,slope_mV)
    t=gate(tail_mV,vhalf_mV,slope_mV)
    start=p+(h-p)*np.exp(-prepulse_s/tau_s)
    end=t+(start-t)*np.exp(-tail_s/tau_s)
    first=g_nS*start*(tail_mV-EK_mV);last=g_nS*end*(tail_mV-EK_mV)
    return dict(start_pA=first,end_pA=last,end_minus_start_pA=last-first,
                start_gate=start,end_gate=end)

def passive_pump_closure(*, voltage_mV, ENa_mV, EK_mV, sodium_channel_pA,
                         potassium_channel_pA, ncx_cycles_charge_pA, passive_NaK_nS):
    """Solve Na and K mass balance at ONE reference point, not ion restoration.

    J denotes pump cycle charge: 3 Na out, 2 K in, 1 ATP. X denotes forward
    NCX cycle charge: 3 Na in, 1 Ca out. Given gNa+gK, solve J and the two
    passive conductances. Negative values are returned as INFEASIBLE, never
    clipped. The voltage balance additionally requires inward Ca current -2X
    and all other currents; it is not automatically asserted here.
    """
    vals=[voltage_mV,ENa_mV,EK_mV,sodium_channel_pA,potassium_channel_pA,
          ncx_cycles_charge_pA,passive_NaK_nS]
    if not np.isfinite(vals).all() or not EK_mV<voltage_mV<ENa_mV or passive_NaK_nS<0:
        raise ValueError('Require EK < V < ENa and nonnegative Na/K passive budget')
    dn=voltage_mV-ENa_mV;dk=voltage_mV-EK_mV
    # Na: I_Na + gNa*(V-ENa) + 3J - 3X = 0.
    # K : I_K  + gK *(V-EK)  - 2J      = 0.
    slope=-3/dn+2/dk
    J=(passive_NaK_nS+(sodium_channel_pA-3*ncx_cycles_charge_pA)/dn+
       potassium_channel_pA/dk)/slope
    gn=(-sodium_channel_pA-3*J+3*ncx_cycles_charge_pA)/dn
    gk=(-potassium_channel_pA+2*J)/dk
    na=sodium_channel_pA+gn*dn+3*J-3*ncx_cycles_charge_pA
    k=potassium_channel_pA+gk*dk-2*J
    return dict(pump_cycle_pA=J,gNa_nS=gn,gK_nS=gk,
                feasible=bool(min(J,gn,gk)>=0),Na_residual_pA=na,K_residual_pA=k,
                passive_budget_residual_nS=gn+gk-passive_NaK_nS,
                net_NaK_NCX_charge_pA=sodium_channel_pA+gn*dn+potassium_channel_pA+gk*dk+J-ncx_cycles_charge_pA,
                required_Ca_inward_pA=-2*ncx_cycles_charge_pA)

def replacement_current(*, old_slow_pA, old_differential_pA, absolute_g_nS,
                        surface_ratio, gate_open, voltage_mV, EK_mV, replace_fraction):
    """Current bookkeeping only: remove identified slow share AND differential owner.

    replace_fraction is an uncertain molecular attribution, not a block fraction.
    Full-neuron activation requires a new rest closure and electrical validation.
    """
    vals=[old_slow_pA,old_differential_pA,absolute_g_nS,surface_ratio,gate_open,
          voltage_mV,EK_mV,replace_fraction]
    if not np.isfinite(vals).all() or min(absolute_g_nS,surface_ratio)<0 or not 0<=gate_open<=1 or not 0<=replace_fraction<=1:
        raise ValueError('Finite inputs, nonnegative g/ratio and bounded fractions required')
    absolute=absolute_g_nS*surface_ratio*gate_open*(voltage_mV-EK_mV)
    retained=(1-replace_fraction)*old_slow_pA
    new=retained+absolute
    return dict(absolute_KCNQ_pA=absolute,retained_unidentified_slow_pA=retained,
                new_total_pA=new,delta_pA=new-old_slow_pA-old_differential_pA)

def geometry_conversion(geometry, fractions):
    """Volume-only conversion: not a simulated dynamic rescue or swelling model."""
    fs=np.asarray(fractions,float)
    if fs.ndim!=1 or not np.isfinite(fs).all() or np.any(fs<=0) or np.any(fs>1):
        raise ValueError('Accessible fractions in (0,1] required')
    r=geometry['process_radius_um'];L=geometry['process_length_um']
    if not np.isfinite([r,L]).all() or min(r,L)<=0:raise ValueError('Positive cylinder size required')
    total=np.pi*r*r*L;area=2*np.pi*r*L;vol=total*fs
    return dict(whole_process_volume_um3=total,membrane_area_um2=area,
                accessible_fraction=fs,accessible_volume_um3=vol,
                K_rate_mM_s_per_pA=-FCONV/vol/1000,
                scope='Same fixed charge amount only; all fluxes/axial exchange/Ca/energy would need recomputation for a new trajectory')
