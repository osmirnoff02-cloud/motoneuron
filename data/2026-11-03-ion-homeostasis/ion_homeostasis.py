"""[codex]2026-11-03. Partition the existing leak into passive Na/K components.

The original voltage leak already contains a conductance. Do not add its
Na/K part again. Replace fixed material backgrounds by that part's current,
and change voltage only by the shift of the two Nernst potentials.
This does NOT resolve chloride, water, all background transport or glia.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit

CONVERSION=1e9/96485.33212
RANGES={'passive_fraction':[0.,.5,1.]}

@dataclass(frozen=True)
class IonBackgroundParameters:
    passive_fraction:float=1.
    provenance:str='Derived from original resting Na/K fluxes and fitted leak; passive interpretation is a hypothesis, not measured human conductance.'
    def validate(self):
        if not np.isfinite(self.passive_fraction) or not 0<=self.passive_fraction<=1 or not self.provenance.strip():
            raise ValueError('Passive fraction in [0,1] and explicit provenance required')

def prepare(base,c,gv,iv,ib,volumes,gl,parameters):
    parameters.validate()
    if not iv[0]:raise ValueError('Passive ion background requires enabled ion dynamics')
    rows=[]
    for j in range(2):
        k=199+20*j;vm=float(base[185+7*j]);cap=c[0]*gv[j]/(gv[0]+gv[1]);fac=CONVERSION/volumes[j]
        ena=iv[1]*np.log(base[k+2]/base[k]);ek=iv[1]*np.log(base[k+3]/base[k+1])
        if abs(vm-ena)<1e-6 or abs(vm-ek)<1e-6:raise ValueError('Background decomposition undefined near reversal')
        gn=-ib[j,3]/fac/(vm-ena);gk=-ib[j,4]/fac/(vm-ek)
        total=cap*gl;residual=total-parameters.passive_fraction*(gn+gk)
        if min(gn,gk)<-1e-12 or residual < -1e-10:
            raise ValueError(f'No nonnegative passive leak partition: compartment{j}, gNa={gn},gK={gk},total={total},residual={residual} nS')
        rows.append([vm,ena,ek,gn,gk,cap,fac,ib[j,3],ib[j,4],total,residual])
    return np.array(rows)

@njit
def finish(z,old,iv,ref,fraction):
    out=old.copy()
    for j in range(2):
        k=199+20*j;vi=185+7*j
        vm=z[vi];ena=iv[1]*np.log(max(z[k+2],1e-12)/max(z[k],1e-12));ek=iv[1]*np.log(max(z[k+3],1e-12)/max(z[k+1],1e-12))
        gn,gk,cap,fac=ref[j,3:7]
        newNa=-gn*(vm-ena)*fac;newK=-gk*(vm-ek)*fac
        dn=fraction*(newNa-ref[j,7]);dk=fraction*(newK-ref[j,8])
        out[k]+=dn;out[k+1]+=dk;out[k+2]-=dn/iv[10];out[k+3]-=dk/iv[10]
        out[k+8]+=dn;out[k+9]+=dk
        # Vm-dependent slope was ALREADY in the calibrated voltage leak.
        # Only changes of reversal potentials are new electrical currents.
        dI=fraction*(gn*(ref[j,1]-ena)+gk*(ref[j,2]-ek))
        out[vi]-=1000*dI/cap
    return out

def observe(z,iv,ref,fraction):
    z=np.atleast_2d(z);rows=[]
    for j in range(2):
        k=199+20*j;vm=z[:,185+7*j]
        ena=iv[1]*np.log(z[:,k+2]/z[:,k]);ek=iv[1]*np.log(z[:,k+3]/z[:,k+1])
        gn,gk,cap,fac=ref[j,3:7]
        na=(1-fraction)*ref[j,7]-fraction*gn*(vm-ena)*fac
        kk=(1-fraction)*ref[j,8]-fraction*gk*(vm-ek)*fac
        rows.append(dict(Na_background_in_uM_s=na,K_background_in_uM_s=kk,
                         reversal_voltage_correction_pA=fraction*(gn*(ref[j,1]-ena)+gk*(ref[j,2]-ek))))
    return dict(compartments=rows,passive_fraction=fraction,
                derived_columns=['V0_mV','ENa0_mV','EK0_mV','gNa_nS','gK_nS','cap_pF','uM_s_per_pA','Na0_background_in_uM_s','K0_background_in_uM_s','total_existing_leak_nS','remaining_leak_nS'],
                derived=ref,scope='Existing leak partition only. No ion clamping, no new ATP debit. Residual leak/holding current and Cl/osmotic balance unresolved; not a human fit.')
