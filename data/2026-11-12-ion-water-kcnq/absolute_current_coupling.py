"""[codex] Opt-in absolute KCNQ with one-time joint pump/passive rest closure.

One K-current owner: retain (1-f) of unidentified slow HH K, remove the old
differential KCNQ, add absolute surface-dependent KCNQ. No dynamic clamp or
forced ion return. Existing gate and surface transport retain ownership.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
from numba import njit
from model_contracts import load_module
assay=load_module(Path(__file__).resolve().parent.parent/'2026-11-12-absolute-kcnq/absolute_kcnq.py','_absolute_assays12')
FCONV=10364.26965661773
RANGES={'g_total_nS':[0.,.5,2.,8.],'replace_slow_fraction':[0.,.5,1.]}

@dataclass(frozen=True)
class AbsoluteKCNQParameters:
    g_total_nS:float=.5
    replace_slow_fraction:float=.5
    provenance:str='Scenario, not measured human MN conductance or molecular slow-current attribution. Reuses explicit CHO-referenced gate with uncertain tau.'
    gate_on_s:float|None=None
    gate_off_s:float|None=None
    gate_scheme:str='directional_v1'
    gate_low_voltage_mV:float=-60.
    gate_high_voltage_mV:float=-20.
    def validate(self):
        if not np.isfinite([self.g_total_nS,self.replace_slow_fraction]).all() or self.g_total_nS<0 or not 0<=self.replace_slow_fraction<=1 or not self.provenance.strip():
            raise ValueError('Nonnegative finite absolute g, fraction [0,1], provenance required')
        if (self.gate_on_s is None)!=(self.gate_off_s is None):
            raise ValueError('Specify both activation and deactivation times, or neither')
        if self.gate_on_s is not None and (not np.isfinite([self.gate_on_s,self.gate_off_s]).all() or min(self.gate_on_s,self.gate_off_s)<=0):
            raise ValueError('Positive finite KCNQ activation/deactivation times required')
        if self.gate_scheme not in ('directional_v1','voltage_anchors_v2'):
            raise ValueError('Unknown KCNQ gate scheme')
        if not np.isfinite([self.gate_low_voltage_mV,self.gate_high_voltage_mV]).all() or self.gate_low_voltage_mV>=self.gate_high_voltage_mV:
            raise ValueError('Ordered finite voltage anchors required')

def prepare(p,base,hh,c,gv,iv,ib,background_ref,volumes):
    p.validate();ib=ib.copy();delta0=[];rows=[]
    for j in range(2):
        vi=185+7*j;b=199+20*j;vm=base[vi];share=gv[j]/sum(gv[:2]);cap=c[0]*share
        ena=iv[1]*np.log(base[b+2]/base[b]);ek=iv[1]*np.log(base[b+3]/base[b+1])
        old_slow=cap*hh[12]*base[vi+4]*(vm-ek)
        absolute=p.g_total_nS*share*base[362+j]*(vm-ek)
        delta=absolute-p.replace_slow_fraction*old_slow
        fac=FCONV/volumes[j];oldJ=ib[j,0]
        closure=assay.passive_pump_closure(voltage_mV=vm,ENa_mV=ena,EK_mV=ek,
            sodium_channel_pA=ib[j,1],potassium_channel_pA=ib[j,2]+delta,
            ncx_cycles_charge_pA=ib[j,8]/fac,passive_NaK_nS=sum(background_ref[j,3:5]))
        if not closure['feasible'] or oldJ<=0:raise ValueError('Absolute KCNQ has no nonnegative fixed rest closure: '+str(closure))
        J=closure['pump_cycle_pA']*fac
        ib[j,0]=J;ib[j,2]+=delta
        ib[j,3]=ib[j,1]*fac+3*J-3*ib[j,8]
        ib[j,4]=ib[j,2]*fac-2*J
        ib[j,9]*=J/oldJ
        delta0.append(delta)
        rows.append(dict(**closure,old_slow_K_pA=float(old_slow),absolute_KCNQ_pA=float(absolute),
            current_delta_at_origin_pA=float(delta),capacity_multiplier=float(ib[j,9])))
    parameters=asdict(p)
    # Preserve fingerprints of prior configurations when the new opt-in is off.
    if p.gate_on_s is None:
        parameters.pop('gate_on_s');parameters.pop('gate_off_s')
    if p.gate_on_s is None or p.gate_scheme=='directional_v1':
        for key in ('gate_scheme','gate_low_voltage_mV','gate_high_voltage_mV'):parameters.pop(key)
    return ib,np.array(delta0),dict(parameters=parameters,rest_closure=rows,
        scope='Absolute current replaces differential KCNQ and selected slow HH share. Fixed passive conductance sum; pump capacity derived once. The fixed voltage offset is the changed passive/pump origin, not a holding/ion-restoration controller.')

@njit
def apply(z,old,ctx,iv,kp,kbase,g,f,delta0):
    out=old.copy();gv=ctx[38]
    for j in range(2):
        vi=185+7*j;b=199+20*j;share=gv[j]/(gv[0]+gv[1]);cap=ctx[37][0]*share
        ek=iv[1]*np.log(max(z[b+3],1e-12)/max(z[b+1],1e-12));vm=z[vi]
        si=4+2*j
        ratio=(max(z[342+si],0.)+kp[13]*max(z[342+si+6],0.))/(kbase[si]+kp[13]*kbase[si+6])
        ratio=max(0.,1+kp[14]*(ratio-1))
        slow=cap*ctx[36][12]*z[vi+4]*(vm-ek)
        differential=(ratio-1)*kp[15]*share*z[362+j]*(vm-ek)
        absolute=g*share*ratio*z[362+j]*(vm-ek)
        delta=absolute-f*slow-differential
        fac=FCONV/(ctx[27] if j==0 else ctx[28])
        out[b+1]-=delta*fac;out[b+3]+=delta*fac/iv[10];out[b+6]+=delta*fac
        out[vi]-=1000*(delta-delta0[j])/cap
    return out


@njit
def finish_gates(z,out,kp,on_s,off_s):
    """Separate relaxation times; same steady G-V and single current owner.

    Continuous derivative at gate==target. Times are scenarios, not human fits.
    Called after append_cycle, which owns the legacy constant-tau gate.
    """
    for j in range(2):
        target=1/(1+np.exp(-max(-700.,min(700.,(z[185+7*j]-kp[16])/kp[17]))))
        delta=target-z[362+j]
        out[362+j]=delta/(on_s if delta>=0 else off_s)
    return out

@njit
def finish_voltage_gates(z,out,kp,on_s,off_s,low_mV,high_mV):
    """Positive voltage-dependent relaxation, smooth at both scenario anchors.

    tau(low)=off, tau(high)=on. Cubic smoothstep interpolates in between;
    values stay constant outside. Steady G-V and current ownership unchanged.
    Equivalent positive two-state rates alpha=m_inf/tau, beta=(1-m_inf)/tau.
    The anchors and interpolation are modelling assumptions, not human data.
    """
    for j in range(2):
        vm=z[185+7*j]
        target=1/(1+np.exp(-max(-700.,min(700.,(vm-kp[16])/kp[17]))))
        x=max(0.,min(1.,(vm-low_mV)/(high_mV-low_mV)));x=x*x*(3.-2.*x)
        tau=off_s+(on_s-off_s)*x
        out[362+j]=(target-z[362+j])/tau
    return out

def observe(z,config,baseline):
    z=np.atleast_2d(z);p=config['absolute_kcnq']['parameters'];rows=[]
    hh=np.asarray(config['HH_parameters']);c=config['HH_context'];geom=config['geometry']
    # Relative area is also the capacitance allocation used by the driver.
    areas=[geom['soma_area_um2'],2*np.pi*geom['process_radius_um']*geom['process_length_um']]
    kp=config['kcnq_transport'];theta=.08617333262145*(273.15+config['ions']['temperature_C'])
    for j in range(2):
        vi=185+7*j;b=199+20*j;share=areas[j]/sum(areas);si=346+2*j
        ratio=(np.maximum(z[:,si],0)+kp['mutant_residual_function']*np.maximum(z[:,si+6],0))/(baseline[si]+kp['mutant_residual_function']*baseline[si+6])
        ratio=np.maximum(0,1+kp['electrical_transfer_gain']*(ratio-1))
        ek=theta*np.log(z[:,b+3]/z[:,b+1]);drive=z[:,vi]-ek
        rows.append(dict(absolute_KCNQ_pA=p['g_total_nS']*share*ratio*z[:,362+j]*drive,
            retained_slow_K_pA=(1-p['replace_slow_fraction'])*c[0]*share*hh[12]*z[:,vi+4]*drive,
            surface_ratio=ratio))
    return rows
