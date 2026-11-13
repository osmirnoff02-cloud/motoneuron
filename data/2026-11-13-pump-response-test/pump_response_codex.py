"""[codex] Matched-rest pump shape experiment, not a human calibration.

Sharples2022/Cressman sigmoid is adapted to zero flux at zero substrate.
The published source-only assay remains unchanged. ATP dependence is inherited
from the host model; all capacities are fixed once, not rebalanced in time.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit

FCONV=1e9/96485.33212
RANGES={'capacity_scale':[.5,1.,2.], 'ATP_half_uM':[50.,300.,1000.]}

@dataclass(frozen=True)
class PumpResponseParameters:
    Na_half_mM:float=25.
    Na_slope_mM:float=3.
    K_half_mM:float=6.
    K_slope_mM:float=1.
    capacity_scale:float=1.
    ATP_half_uM:float=300.
    provenance:str='Sharples2022 p16 model shape; zero-substrate adaptation and inherited ATP saturation; matched host resting current, not measured human MN capacity.'
    def validate(self):
        values=[self.Na_half_mM,self.Na_slope_mM,self.K_half_mM,self.K_slope_mM,self.capacity_scale,self.ATP_half_uM]
        if not np.isfinite(values).all() or min(values)<=0 or not self.provenance.strip():
            raise ValueError('Positive finite pump shape/capacity/ATP parameters and provenance required')

@njit
def substrate(x,half,slope):
    x=max(x,0.)
    floor=1/(1+np.exp(min(700.,half/slope)))
    value=1/(1+np.exp(min(700.,max(-700.,(half-x)/slope))))
    return max(0.,(value-floor)/(1-floor))

@njit
def cycles(na_uM,k_uM,atp_uM,v,j,support=1.):
    atp=max(atp_uM,0.)
    return v[j,6]*v[j,4]*support*substrate(na_uM/1000.,v[j,0],v[j,1])*substrate(k_uM/1000.,v[j,2],v[j,3])*atp/(v[j,5]+atp)

def prepare(p,base,ionbase,geometry):
    p.validate();v=np.zeros((2,7));rows=[]
    for j in range(2):
        b=199+20*j;at=120 if j==0 else 136
        shape=substrate(base[b]/1000.,p.Na_half_mM,p.Na_slope_mM)*substrate(base[b+3]/1000.,p.K_half_mM,p.K_slope_mM)*base[at]/(p.ATP_half_uM+base[at])
        if shape<=0 or ionbase[j,0]<=0:raise ValueError('Positive resting substrates and pump flux required')
        vmax=ionbase[j,0]/shape
        v[j]=[p.Na_half_mM,p.Na_slope_mM,p.K_half_mM,p.K_slope_mM,p.capacity_scale,p.ATP_half_uM,vmax]
        vol=geometry.soma_volume_um3 if j==0 else geometry.process_volume_um3
        rows.append(dict(Na0_uM=float(base[b]),Kout0_uM=float(base[b+3]),ATP0_uM=float(base[at]),
            reference_volume_um3=vol,rest_cycles_uM_s=float(ionbase[j,0]),
            rest_current_pA=float(ionbase[j,0]*vol/FCONV),asymptotic_capacity_pA=float(vmax*vol/FCONV),
            actual_rest_current_pA=float(cycles(base[b],base[b+3],base[at],v,j)*vol/FCONV)))
    return v,dict(parameters=asdict(p),vector=v.tolist(),compartments=rows,
        source_DOI='10.3389/fncel.2021.715427',family='zero-substrate-adapted sigmoid, matched initial current',
        limitations=['Source is a mouse spinal CPG model, not measured human MN current.',
            'No alpha1/alpha3 or FXYD7 resolution; source 40.26pA is not copied.',
            'Source experiments 27C; host temperature unchanged in the matched comparison, no measured Q10.',
            'ATP saturation and zero-substrate subtraction are explicit host adaptations.',
            'Capacity scale and ATP half ranges are scenarios, not confidence intervals.',
            'Rest match applies at capacity_scale=1 and pump_support=1; no dynamic ion or energy restoration.'])

@njit
def apply(z,old,ctx,iv,v,support):
    out=old.copy()
    for j in range(2):
        b=199+20*j;at=120 if j==0 else 136;vi=185+7*j
        vol=ctx[27] if j==0 else ctx[28];fac=FCONV/vol
        cap=ctx[37][0]*ctx[38][j]/(ctx[38][0]+ctx[38][1])
        delta=cycles(z[b],z[b+3],z[at],v,j,support)-old[b+4]
        out[b]-=3*delta;out[b+1]+=2*delta
        out[b+2]+=3*delta/iv[10];out[b+3]-=2*delta/iv[10]
        out[at]-=delta;out[at+1]+=delta;out[vi]-=1000*delta/fac/cap
        out[b+4]+=delta;out[b+14]+=delta/fac;out[b+15]+=delta
    return out

def observe(state,passport,support=1.):
    z=np.atleast_2d(np.asarray(state,float));v=np.asarray(passport['vector'],float)
    if not np.isfinite(z).all() or not np.isfinite(support) or support<0:raise ValueError('Finite states and nonnegative support required')
    return np.array([[cycles(row[199+20*j],row[202+20*j],row[120 if j==0 else 136],v,j,support)*passport['compartments'][j]['reference_volume_um3']/FCONV for j in range(2)] for row in z])
