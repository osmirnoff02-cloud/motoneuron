"""[codex] 2026-10-26. Conditional mitochondrial reserve and terminal energy.

An effective terminal, not a reconstructed NMJ: Ca increment/recycling costs
are hypotheses. ATP exchange is an amount flux, debited from the process.
No claim that NCLX is exclusively Na/Ca; molecular mechanism remains open.
"""
from dataclasses import dataclass,asdict
import numpy as np
from numba import njit
from nmj_codex import NMJParameters

@dataclass(frozen=True)
class TerminalEnergyParameters:
    enabled:bool=True
    volume_um3:float=1.
    exchange_um3_s:float=.3
    other_ATP_uM_s:float=6.
    production_half_ADP_uM:float=200.
    ca_buffer_kappa:float=50.
    provenance:str='Scenario effective terminal volume/exchange/ATP supply and bulk Ca; no human NMJ calibration. Explicit molecules, no muscle AP/force or complete terminal metabolism.'
    def vector(self):
        a=np.array([float(self.enabled),self.volume_um3,self.exchange_um3_s,self.other_ATP_uM_s,self.production_half_ADP_uM,self.ca_buffer_kappa])
        if not isinstance(self.enabled,bool) or not np.isfinite(a).all() or np.any(a[1:]<0) or min(a[1],a[4])<=0 or not self.provenance:raise ValueError('Invalid terminal energy parameters')
        return a

TERMINAL_ENERGY_RANGES=dict(volume_um3=[.1,1.,10.],exchange_um3_s=[0.,.03,.3,3.],other_ATP_uM_s=[1.,6.,30.],production_half_ADP_uM=[50.,200.,1000.],ca_buffer_kappa=[10.,50.,200.],respiratory_reserve=[2.,4.,8.],adenylate_diffusion_um2_s=[0.,10.,100.,300.],process_length_um=[30.,60.,100.],process_AMPA_fraction=[.1,.9],association_support=[0.,.5,1.],initiation_support=[0.,.5,1.])
TERMINAL_NAMES=('terminal_Ca_excess_proxy_uM','terminal_vesicle_available_fraction','terminal_ACh_conductance_nS','endplate_mV','terminal_released_quanta','terminal_replenished_quanta','terminal_recycling_ATP_molecules','terminal_ATP_uM','terminal_ADP_uM','terminal_AMP_uM','terminal_ATP_exchange_amount_uM_um3','terminal_ADP_exchange_amount_uM_um3','terminal_AMP_exchange_amount_uM_um3','terminal_produced_ATP_uM','terminal_other_ATP_uM','terminal_PMCA_ATP_uM','terminal_AK_net_ATP_uM','terminal_external_Ca_uM','terminal_extruded_Ca_uM')

# Input interventions, not measured confidence intervals.
TERMINAL_SUPPLY_RANGES=dict(terminal_supply_support=[0.,.5,1.])

def nmj_vector(p):
    a=np.array([p.terminal_Ca_increment_uM,p.terminal_Ca_decay_s,p.release_half_uM,p.release_hill,p.max_release_probability,p.releasable_vesicles,p.replenishment_s,p.ATP_half_uM,p.ACh_conductance_decay_s,p.conductance_per_vesicle_nS,p.endplate_capacitance_pF,p.endplate_leak_nS,p.resting_mV,p.ACh_reversal_mV,p.threshold_mV,p.ATP_per_recycled_vesicle])
    if not np.isfinite(a).all() or min(a[:12])<=0 or not 0<a[4]<=1 or a[15]<0 or not a[12]<a[14]<a[13]:raise ValueError('Invalid NMJ parameters')
    return a

@njit
def terminal_extension(z,old,tp,npv,en,process_volume,bpv,reserve,local_support):
    out=np.zeros(294);out[:275]=old
    # Same basal ETC; enlarge only the explicitly unknown response reserve.
    if bpv[0] and reserve!=2.:
        for j in range(2):
            b=247+10*j;q=max(z[b],0.)/bpv[1]
            ratio=reserve*(1+q*q)/(2*(1+(reserve-1)*q*q))
            delta=old[b+2]*(ratio-1)
            out[b]+=delta;out[b+2]+=delta;out[b+1]-=delta/bpv[7];out[b+3]+=delta/bpv[7]
    if not tp[0]:return out
    c,a,g,v=z[275:279];A,D,M=z[282:285];vol=tp[1]
    f=max(A,0.)/(npv[7]+max(A,0.))
    f0=en[0]/(npv[7]+en[0]);fn=f/f0
    repl=max(0.,1-a)/npv[6]*f*npv[5]
    caout=max(c,0.)/npv[1]*(1+tp[5])*fn
    other=tp[3]*fn
    prod=tp[3]*(max(D,0.)/(tp[4]+max(D,0.)))/(en[1]/(tp[4]+en[1]))*local_support
    recycle=repl*npv[15]/(602.214076*vol)
    ak=en[11]*(D*D-A*M/(en[0]*en[2]/en[1]**2))
    flux=tp[2]*(z[136:139]-z[282:285])
    out[136:139]-=flux/process_volume
    out[275:282]=np.array([-caout/(1+tp[5]),repl/npv[5],-g/npv[8],1000*(npv[11]*(npv[12]-v)+g*(npv[13]-v))/npv[10],0.,repl,repl*npv[15]])
    cost=other+recycle+caout
    out[282:285]=np.array([prod-cost+ak,cost-prod-2*ak,ak])+flux/vol
    out[285:288]=flux
    out[288:294]=np.array([prod,other,caout,ak,0.,caout])
    return out

def terminal_jump(y,tp,npv):
    z=y.copy()
    if not tp[0]:return z,0.
    z[275]+=npv[0];z[292]+=(1+tp[5])*npv[0]
    prob=npv[4]*z[275]**npv[3]/(npv[2]**npv[3]+z[275]**npv[3])
    released=npv[5]*z[276]*prob
    z[276]-=released/npv[5];z[277]+=released*npv[9];z[279]+=released
    return z,float(released)
