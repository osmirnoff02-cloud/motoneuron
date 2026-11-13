"""[codex] 2026-10-28. Finite reversible CK, adapted from Blue Brain.

Equation CKc_n: BlueBrain/metabolism-in-aging, commit1204a605a8d2397f2b78952f74d163df217ed2f9,
PolinaShichkova_NGVmodel_Julia166.ipynb. Copyright2023/2024 Blue Brain Project/EPFL,
Apache-2.0; LICENSE_BlueBrain.txt and NOTICE_codex.txt accompany this adaptation.
Changes: mM -> uM, explicit ATP/ADP ODEs, three separate finite pools.
No donor algebraic ADP/AK denominator, pump debit, or ATP production is copied.
These parameters are donor-model scenarios, NOT measured human MN constants.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit

@dataclass(frozen=True)
class CreatineParameters:
    total_creatine_uM:tuple=(10000.,10000.,10000.)
    forward_per_uM_s:float=0.0000214016075483191
    reverse_equilibrium_factor:float=0.04265840286184623
    rate_scale:tuple=(1.,1.,1.)
    provenance:str='Blue Brain young CKc_n scenario; same hypothetical pool in soma/process/terminal, no human MN CK calibration.'
    def vector(self):
        v=np.array([*self.total_creatine_uM,self.forward_per_uM_s,self.reverse_equilibrium_factor,*self.rate_scale],float)
        if len(self.total_creatine_uM)!=3 or len(self.rate_scale)!=3 or v.shape!=(8,) or not np.isfinite(v).all() or min(v)<0 or v[4]<=0 or not self.provenance:
            raise ValueError('Three nonnegative creatine pools/scales and positive equilibrium factor required')
        return v

CREATINE_RANGES=dict(total_creatine_uM=[0.,1000.,10000.],forward_per_uM_s=[0.0000214016075483191,0.0000433],
    reverse_equilibrium_factor=[0.00028/0.0433,.02,.04265840286184623,.1],rate_scale=[0.,.1,1.,10.],creatine_support=[0.,.5,1.])
CREATINE_NAMES=tuple(q+'_PCr_uM' for q in ('soma','process','terminal'))

def creatine_source_parameters(source='shichkova2025'):
    """Paired donor parameterizations, never a measured human interval.

    Jolivet2015 Table3: forward .0433 (optimized), reverse .00028 (constrained)
    mM^-1 s^-1; total10mM fixed. Preserve pairing instead of mixing models.
    """
    if source=='shichkova2025':return CreatineParameters()
    if source=='jolivet2015':
        return CreatineParameters(forward_per_uM_s=.0433/1000,reverse_equilibrium_factor=.00028/.0433,
            provenance='Jolivet2015 Table3/EqA38 donor model: optimized forward and constrained reverse rates, fixed10mM pool. Not measured human MN.')
    raise ValueError('Unknown source parameterization')

def equilibrium_PCr(atp,adp,total,keq):
    a=np.asarray(atp,float);d=np.asarray(adp,float)
    if np.any(a<=0) or np.any(d<=0):raise ValueError('Positive ATP/ADP required at baseline')
    return np.asarray(total)*keq*a/(d+keq*a)

@njit
def ck_flux(pcr,atp,adp,total,k,keq,scale=1.):
    # Boundary clipping only handles floating-point undershoots; the solver
    # separately rejects out-of-pool states. No ATP or PCr creation at zero.
    pc=min(max(pcr,0.),total)
    return scale*k*(pc*max(adp,0.)-keq*(total-pc)*max(atp,0.))

@njit
def creatine_extension(z,old,v,support):
    out=np.zeros(318);out[:315]=old
    for j,a in enumerate((120,136,282)):
        f=ck_flux(z[315+j],z[a],z[a+1],v[j],v[3],v[4],v[5+j]*support)
        out[a]+=f;out[a+1]-=f;out[315+j]=-f
    return out
