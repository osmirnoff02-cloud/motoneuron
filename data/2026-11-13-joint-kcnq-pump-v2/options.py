"""[codex] Explicit joint candidates; no human confidence bounds."""
from dataclasses import replace
from capture import old

BASE=dict(g=.15,on=.08,off=.08,vhalf=-27.3966,replacement=.025,leak=.5,alpha3=.5,khalf=1500.)
RANGES=dict(g_total_nS=[.05,.1,.25,.5,1.,2.],gate_on_s=[.02,.04,.08,.15],gate_off_s=[.02,.08,.15,.3],
    replace_slow_fraction=[0.,.025,.1,.25],leak_NaK_fraction=[.25,.5,.75],alpha3_max_fraction=[0.,.5,.9],pump_K_half_uM=[500.,1500.,3000.])
def options(sg,profile,p):
    o,rec=old.legacy.options(sg,profile,p['leak']);api=old.r.api
    o['water_volume_parameters']=api.WaterVolumeParameters(process_water_fraction=.5)
    if not p.get('legacy',False):
        o['absolute_kcnq_parameters']=api.AbsoluteKCNQParameters(g_total_nS=p['g'],replace_slow_fraction=p['replacement'],gate_on_s=p['on'],gate_off_s=p['off'],
            gate_scheme='voltage_anchors_v2',gate_low_voltage_mV=-60.,gate_high_voltage_mV=-20.,
            provenance='Joint conditional fit13-v2: voltage-dependent tau at scenario anchors -60/-20mV; inferred rest pump/leaks. Not human MN calibration.')
        o['kcnq_coupling_parameters']=replace(o.get('kcnq_coupling_parameters',api.KCNQCouplingParameters()),vhalf_mV=p['vhalf'])
    o['recovery_parameters']=replace(o.get('recovery_parameters',api.RecoveryParameters()),alpha3_max_fraction=p['alpha3'])
    o['ion_parameters']=replace(o.get('ion_parameters',api.IonParameters()),pump_K_half_uM=p['khalf'])
    o['pump_capacity_multiplier']=1.
    return o,rec
