"""[codex] 2026-11-11. Optional finite passive chloride, not a human Cl fit.

The existing voltage leak already owns the slope. Its residual Na/K-unassigned
conductance is assigned to Cl. Only the changing ECl correction is added to Vm.
Initial ECl=V0 is a declared passive-equilibrium assumption. No KCC/NKCC,
chloride axial transport, cell swelling, or restorative intracellular source.
The well-stirred external reservoir has explicit material ledgers.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit

CONVERSION=1e9/96485.33212
NAMES=tuple(c+'_'+s for c in ('soma','process') for s in
    ('Cl_i_uM','Cl_o_uM','Cl_membrane_in_uM','Cl_membrane_out_uM',
     'Cl_bath_added_uM','Cl_bath_removed_uM'))
# Scenario values, NOT a measured human confidence interval.
RANGES={'Cl_out0_uM':[110000.,130000.,150000.],
        'NaK_leak_share':[.25,.5,.75,1.]}

@dataclass(frozen=True)
class PassiveChlorideParameters:
    Cl_out0_uM:float=130000.
    provenance:str='Scenario: external chloride; initial passive ECl=V0; no human intracellular chloride calibration.'
    def validate(self):
        if not np.isfinite(self.Cl_out0_uM) or self.Cl_out0_uM<=0 or not self.provenance.strip():
            raise ValueError('Positive finite external Cl and provenance required')

def prepare(base,iv,ref,p):
    p.validate()
    rows=[];tail=[]
    for j in range(2):
        v0,cap,fac,gcl=ref[j,0],ref[j,5],ref[j,6],max(0.,ref[j,10])
        if ref[j,10]<-1e-10:raise ValueError('Negative residual chloride conductance')
        ci=p.Cl_out0_uM*np.exp(v0/iv[1])
        tail.extend([ci,p.Cl_out0_uM,0.,0.,0.,0.])
        rows.append([v0,gcl,cap,fac,p.Cl_out0_uM])
    return np.r_[base,tail],np.asarray(rows)

@njit
def finish(z,old,iv,ref,offset):
    out=np.zeros(len(z));out[:len(old)]=old
    for j in range(2):
        k=offset+6*j;vi=185+7*j
        ecl=iv[1]*np.log(max(z[k],1e-12)/max(z[k+1],1e-12))
        v0,g,cap,fac,co=ref[j]
        # Outward positive charge equals inward flux of the negative Cl ion.
        flux=g*(z[vi]-ecl)*fac
        bath=(co-z[k+1])/iv[11]
        out[k]=flux;out[k+1]=-flux/iv[10]+bath
        out[k+2]=max(flux,0.);out[k+3]=max(-flux,0.)
        out[k+4]=max(bath,0.);out[k+5]=max(-bath,0.)
        # Baseline g*(V-V0) is already part of the fitted membrane leak.
        out[vi]-=1000*g*(v0-ecl)/cap
    return out

def observe(z,iv,ref,offset):
    z=np.atleast_2d(z);rows=[]
    for j in range(2):
        k=offset+6*j;v0,g,cap,fac,co=ref[j]
        ecl=iv[1]*np.log(z[:,k]/z[:,k+1])
        rows.append(dict(ECl_mV=ecl,gCl_nS=g,Cl_current_outward_pA=g*(z[:,185+7*j]-ecl),
            additional_voltage_current_pA=g*(v0-ecl)))
    return dict(compartments=rows,offset=offset,
        scope='Passive-only Cl; implicit fitted baseline leak slope retained once. No KCC/NKCC/Cl axial transport or human fit.')

def checks(z,base,iv,offset):
    result={}
    for j in range(2):
        k=offset+6*j;d=z[:,k:k+6]-base[k:k+6]
        scale=max(1.,base[k]+iv[10]*base[k+1])
        result[f'chloride_{j}_material_balance']=float(np.max(abs(d[:,0]+iv[10]*d[:,1]-iv[10]*(d[:,4]-d[:,5])))/scale)
        result[f'chloride_{j}_membrane_ledger']=float(np.max(abs(d[:,0]-d[:,2]+d[:,3]))/scale)
        result[f'chloride_{j}_negative_pool']=float(max(0.,-z[:,k:k+6].min()))
    return result
