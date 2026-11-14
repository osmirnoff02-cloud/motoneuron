"""[codex] Optional coat-removal stage, supported by Farsi2018 mouse vesicles.

Human times/fractions/cost partition are hypotheses. This is a queue gate,
not a physical pH pump, clathrin molecule model or a fit to isolated vesicles.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit

@dataclass(frozen=True)
class UncoatingParameters:
    tau_s: float = 1.
    coated_fraction: float = 1.
    retrieval_cost_fraction: float = .25
    provenance: str = 'Farsi2018 mouse brain CCVs: coat inhibits V-ATPase; removal restores acidification. Human time, pathway fraction and allocation of existing ATP budget are unknown.'

    def vector(self):
        a=np.array([self.tau_s,self.coated_fraction,self.retrieval_cost_fraction])
        if not np.isfinite(a).all() or a[0]<=0 or np.any(a[1:]<0) or np.any(a[1:]>1) or not self.provenance:
            raise ValueError('Invalid uncoating parameters')
        return a

UNCOATING_RANGES=dict(tau_s=[.1,1.,10.],coated_fraction=[0.,.5,1.],retrieval_cost_fraction=[0.,.25,.5],uncoating_support=[0.,.1,1.])
UNCOATING_NAMES=('terminal_coated_quanta','terminal_uncoated_quanta','terminal_uncoating_ATP_molecules')

@njit
def uncoating_rates(coated,endo,ATP_factor,cycle_cost,retrieval_fraction,uv,support):
    uncoat=max(coated,0.)/uv[0]*ATP_factor*support
    moved=uv[1]*endo*cycle_cost*retrieval_fraction*uv[2]
    cost=uncoat*cycle_cost*retrieval_fraction*uv[2]
    return uncoat,moved,cost

@njit
def uncoating_extension(z,old,tv,nv,en,rc,uv,support):
    out=np.zeros(315);out[:312]=old
    a=max(z[282],0.)
    af=(a/(nv[7]+a))/(en[0]/(nv[7]+en[0]))
    endo=old[298]
    uncoat,moved,cost=uncoating_rates(z[312],endo,af,nv[15],rc[4],uv,support)
    # Redirect a fraction of the same endocytosed membrane; never duplicate it.
    out[297]+=uncoat-uv[1]*endo
    out[312]=uv[1]*endo-uncoat;out[313]=uncoat
    # Allocate part of the existing retrieval budget to later uncoating.
    # Do not add an unrelated second per-vesicle ATP cost.
    delta=cost-moved
    out[299]-=moved;out[281]+=delta;out[314]=cost
    out[282]-=delta/(602.214076*tv[1]);out[283]+=delta/(602.214076*tv[1])
    return out
