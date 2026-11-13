"""[codex] Scalar JIT equivalents of inherited PMCA equations; no new kinetics."""
import numpy as np
from numba import njit
from connected_parameters_codex import spatial
energy_rhs=spatial.core._rhs_energy
@njit
def sat_jit(x,k,x0):
    x=max(x,0.);return x/(k+x)*(k+x0)/x0
@njit
def rates_jit(c,v):
    c=max(c,0.)
    return v[2]*v[3]*1e-6*c**v[5]/(c**v[5]+v[4]**v[5]),v[7]+(v[1]-v[7])*(1+(.05/v[6])**4)/(1+(c/v[6])**4)
@njit
def pmca_flux_jit(c,activation,c0,gamma,pmca_fraction,atp_factor,v):
    c=max(c,0.);a,b=rates_jit(c0,v);base=a/(a+b)
    memory=(1+v[0]*activation)/(1+v[0]*base)
    hill=c**v[13]/(v[12]**v[13]+c**v[13]);hill0=c0**v[13]/(v[12]**v[13]+c0**v[13])
    shaped=(1-v[11])*c+v[11]*c0*hill/hill0
    return pmca_fraction*gamma*atp_factor*((1-v[9])*c+v[9]*v[8]*v[10]*shaped*memory)
@njit
def memory_jit(z,p,kv,sv,av,hv,nv,ag0,iv,base_n,row,ep,target,rv,qv,base_iron,ev,voltage,en,drive,pv):
    out=np.zeros(134)
    out[:133]=energy_rhs(z[:133],p,kv,sv,av,hv,nv,ag0,iv,base_n,row,ep,target,rv,qv,base_iron,ev,voltage,en,drive)
    c=max(p[0]+z[0],0.);gamma=(1+ep[15])/p[1]
    pf=z[120]/(en[4]+z[120])*(en[4]+en[0])/en[0] if en[12]>0 else 1.
    delta=pmca_flux_jit(c,z[133],p[0],gamma,en[9],pf,pv)-out[130]
    out[0]-=delta/(1+ep[15]);out[23]+=delta/(1+ep[15])
    out[120]-=delta;out[121]+=delta;out[126]+=delta;out[130]+=delta
    on,off=rates_jit(c,pv);out[133]=on*(1-z[133])-off*z[133]
    return out
@njit
def traffic_jit(pools,tr,tf,trev,forward,reverse,atps,atpp):
    ns,nf,np_,nr=np.maximum(pools,0)
    ds=tr[0]*ns;dp=tr[1]*np_;af=nf/tf*forward*atps;ar=nr/trev*reverse*atpp
    return np.array([-ds+ar,ds-af,af-dp,dp-ar]),af,ar
