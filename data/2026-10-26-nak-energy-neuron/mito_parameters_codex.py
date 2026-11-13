"""[codex] 2026-10-26. Optional effective matrix-Ca store, not a full mitochondrion.

MCU-like entry and effective exit conserve Ca amount. Exit is NOT named NCLX:
matrix Na and membrane potential are not yet represented. Ca-to-OXPHOS gain
is an independently switchable hypothesis, not fitted to Dafinca fluorescence.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit

@dataclass(frozen=True)
class MitoParameters:
    enabled: bool = True
    matrix_volume_per_cytosol: float = .05
    matrix_kappa: float = 100.
    resting_matrix_Ca_uM: float = .1
    maximum_matrix_Ca_uM: float = 10.
    uptake_capacity_uM_s: float = 1.
    uptake_half_uM: float = .5
    uptake_hill: float = 2.
    oxidative_gain: float = 0.
    oxidative_half_uM: float = .5
    provenance: str = 'Effective mechanism supported by human iPSC-MN Dafinca2020; all numeric kinetics/volumes are exploratory assumptions; no absolute-Ca human fit'
    def vector(self):
        a=np.array([float(self.enabled),self.matrix_volume_per_cytosol,self.matrix_kappa,self.resting_matrix_Ca_uM,
          self.maximum_matrix_Ca_uM,self.uptake_capacity_uM_s,self.uptake_half_uM,self.uptake_hill,
          self.oxidative_gain,self.oxidative_half_uM])
        if not np.isfinite(a).all() or np.any(a[1:8]<=0) or a[4]<=a[3] or not 0<=a[8]<=1 or a[9]<=0 or not self.provenance.strip():
            raise ValueError('Invalid matrix Ca parameters')
        return a

MITO_RANGES=dict(matrix_volume_per_cytosol=[.01,.05,.1],matrix_kappa=[20.,100.,500.],
 resting_matrix_Ca_uM=[.05,.1,.2],maximum_matrix_Ca_uM=[2.,10.,50.],uptake_capacity_uM_s=[.1,1.,10.],
 uptake_half_uM=[.2,.5,2.],uptake_hill=[1.,2.],oxidative_gain=[0.,.5,1.],oxidative_half_uM=[.2,.5,2.],
 uptake_support=[0.,.5,1.])
MITO_NAMES=tuple(side+'_'+n for side in ('soma','process') for n in ('matrix_Ca_total_cyt_equiv_uM','matrix_Ca_in_uM','matrix_Ca_out_uM','matrix_extra_OXPHOS_ATP_uM'))

@njit
def mito_rates(ca,store,c0,v,support):
    scale=v[1]*(1+v[2]);cm=max(store,0.)/scale
    h0=c0**v[7]/(v[6]**v[7]+c0**v[7])
    j0=v[5]*h0*(1-v[3]/v[4])
    h=max(ca,0.)**v[7]/(v[6]**v[7]+max(ca,0.)**v[7])
    entry=v[5]*h*max(0.,1-cm/v[4])*support
    exit=j0*cm/v[3]
    gain=v[8]*(cm/(v[9]+cm)-v[3]/(v[9]+v[3]))
    return entry,exit,gain

@njit
def mito_extension(z,old,ctx,v,support):
    out=np.zeros(247);out[:239]=old
    if v[0]==0:return out
    c0=ctx[41]
    for j in range(2):
        q=239+4*j;ci=0 if j==0 else 134;at=120 if j==0 else 136;ox=123 if j==0 else 148
        kappa=ctx[30] if j==0 else ctx[32]
        entry,exit,gain=mito_rates(c0+z[ci],z[q],c0,v,support)
        net=entry-exit;out[q:q+3]=[net,entry,exit]
        out[ci]-=net/(1+kappa)
        extra=old[ox]*gain
        out[at]+=extra;out[at+1]-=extra;out[ox]+=extra;out[q+3]=extra
    return out
