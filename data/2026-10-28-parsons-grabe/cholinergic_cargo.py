"""[codex] 2026-10-28. ACh cargo is distinct from cycling membrane.

Parsons1999 snake NMJ provides the qualitative blocked-filling constraint.
Human rates, initial occupancy and the distribution of cargo are unknown.
Cargo units are full-quantal equivalents, NOT molecules or vesicle counts.
Existing cycling ATP cost is retained; VAChT is not an extra ATPase.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit


@dataclass(frozen=True)
class CholinergicCargoParameters:
    filling_tau_s: float = 10.
    initial_ready_content_fraction: float = 1.
    interpretation: str = 'binary_mixture'
    provenance: str = 'Parsons1999 snake: empty vesicles cycle; human filling time/occupancy unknown. Binary mixture is source-specific; graded content is an alternative observation model.'

    def vector(self):
        v=np.array([self.filling_tau_s,self.initial_ready_content_fraction])
        if not np.isfinite(v).all() or v[0]<=0 or not 0<=v[1]<=1 or self.interpretation not in ('binary_mixture','graded_content') or not self.provenance:
            raise ValueError('Invalid ACh cargo parameters')
        return v


CARGO_RANGES=dict(filling_tau_s=[1.,10.,60.],initial_ready_content_fraction=[0.,.5,1.],filling_support=[0.,.1,.5,1.])
CARGO_NAMES=('terminal_ready_ACh_quantal_equivalents','terminal_unready_ACh_quantal_equivalents',
             'terminal_loaded_ACh_quantal_equivalents','terminal_released_ACh_quantal_equivalents')


@njit
def cargo_rates(ready,unready,ready_content,unready_content,priming_rate,ATP_factor,cv,support):
    """Random priming of a mixed pool; filling in acidified/ready pools only.

    ATP_factor is a lumped available-gradient proxy, not measured VAChT ATP
    affinity. No ACh synthesis/ChAT/choline pool, leak or partial fusion yet.
    """
    transfer=priming_rate*min(1.,max(0.,unready_content/max(unready,1e-20)))
    k=support*ATP_factor/cv[0]
    lr=k*max(0.,ready-ready_content);lu=k*max(0.,unready-unready_content)
    return np.array([transfer+lr,lu-transfer,lr+lu,0.])


@njit
def cargo_extension(z,old,tv,nv,en,cv,support):
    out=np.zeros(312);out[:308]=old
    if not tv[0]:return out
    A=max(z[282],0.)
    af=(A/(nv[7]+A))/(en[0]/(nv[7]+en[0]))
    out[308:]=cargo_rates(nv[5]*z[276],z[303],z[308],z[309],old[280],af,cv,support)
    return out


def cargo_jump(before,after,released,nv):
    """Replace the previous all-vesicles-full conductance increment once."""
    out=after.copy();ready=nv[5]*before[276]
    fraction=float(np.clip(before[308]/ready,0,1)) if ready>0 else 0.
    effective=released*fraction
    out[308]-=effective;out[311]+=effective
    out[277]=before[277]+effective*nv[9]
    return out,effective


def release_observation(membrane_quanta,cargo_quanta,interpretation):
    """Two unresolved distributions can produce the SAME mean ACh output.

    Binary: fewer full-size events. Graded: same count, smaller amplitude.
    These are ideal expectations, with no detection threshold or noise.
    """
    m,c=np.broadcast_arrays(np.asarray(membrane_quanta,float),np.asarray(cargo_quanta,float))
    if not np.isfinite(m+c).all() or np.any(m<0) or np.any(c<0) or np.any(c>m+1e-8):raise ValueError('Invalid release')
    frac=np.divide(c,m,out=np.zeros_like(c),where=m>0)
    if interpretation=='binary_mixture':return dict(expected_detectable_events=c,relative_event_amplitude=np.where(c>0,1.,np.nan))
    if interpretation=='graded_content':return dict(expected_detectable_events=np.where(c>0,m,0.),relative_event_amplitude=frac)
    raise ValueError('Unknown cargo interpretation')


def source_constraints():
    return dict(
        parsons=dict(species='Thamnophis snake',temperature_C=[21.,23.],vesamicol_uM=5.,K_propionate_mM=35.,
                     pretreatment_min=20.,load_FM1_43_min=[114.,120.],MEPC_frequency_at90min_per_s_upper=1.,
                     integrated_preformed_quanta_estimate=490000,FM_destaining_quantified=False,
                     amplitude='No decrease among detected events; not a universal partially-filled-vesicle law'),
        tabares=dict(species='adult mouse NMJ',S1_n=42,S1_half_time_intercept_s=5.9,S1_half_time_slope_s_per_shock=.02,
                     S1_half_time_r_squared=.42,S1_amplitude_r_squared=.20,S2_surface_reporter_fraction_estimate=.20,
                     S2_acid_bath_pH=5.6,S2_response_relative=.13,S2_folimycin_nM=67.,
                     caveat='S1 is half-time, not tau; S2 acid also inhibits release. Folimycin can erase optical increments without erasing cycling.'))
