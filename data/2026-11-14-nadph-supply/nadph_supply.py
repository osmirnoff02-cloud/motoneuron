"""[claude] 2026-11-15. Standalone NADPH supply block (pentose phosphate pathway).

Author: Claude (Fable 5.2). Plan row N11.2.3 (redox half). Version 1 is NOT
connected to the host neuron. It replaces the "solved NADPH supply capacity"
of the lipid block by an enzyme-like law whose reserve is tied to a measured
ratio, and it makes the resting NADPH/NADP+ ratio an output.

Pools, uM of cytosol, seconds
  G6P   glucose-6-phosphate          P5   pentose phosphates (lumped)

Reactions
  f0 hexokinase supply      J_glc * glucose_supply_rel            (input flux)
  f1 glycolysis             k_gly * G6P
  f2 oxidative branch       G6P + 2 NADP+ -> P5 + CO2 + 2 NADPH
                            Vmax*G6P/(Kg+G6P)*NADP/(Kn*(1+NADPH/Ki)+NADP)
                            (G6PD law; 6-phosphogluconate step lumped in)
  f3 pentose return         P5 -> 5/6 hexose; a share goes back to G6P,
                            the rest continues into glycolysis
  f4 other NADPH sources    Vo*NADP/(Ko+NADP)   (cytosolic IDH, malic enzyme)

Closure. Vmax is not declared: it follows from the declared reserve, i.e. the
ratio of the maximal to the resting oxidative flux, which is anchored to the
H2O2-stimulated rise of pentose-pathway flux in primary cerebrocortical
cultures (Ben-Yoseph 1994, 1996). The resting NADP+ then follows from the
G6PD law, so the resting NADPH fraction is a result, not an input.

Carbon ledger in C1 units: 6*G6P + 5*P5 + 6*glycolysis_out + CO2 = 6*glucose in.
The ATP that the lost carbon could have produced is reported as an
opportunity cost; it is NOT debited from any ATP pool.

Declared limits: one compartment; no glucose transport or hexokinase
kinetics (supply is an input); no feed-forward activation of glucose uptake
by peroxide (reported by Aburto 2025); no NADPH oxidase; no mitochondrial
NADPH. Kinetic constants are for purified recombinant human G6PD and the
NADPH inhibition constant for the mouse enzyme. Not a human motor-neuron
calibration.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit

VERSION = 'nadph-supply-claude-v1-2026-11-15'
POOLS = ('G6P', 'P5')
LEDGERS = ('glucose_phosphorylated', 'glycolysis_out_C6', 'CO2', 'PPP_oxidative', 'NADPH_from_PPP',
           'NADPH_from_other', 'pentose_returned')
NAMES = POOLS + LEDGERS
NS = len(NAMES)
INPUTS = ('NADPH_uM', 'glucose_supply_rel', 'm_G6PD', 'm_other_NADPH_sources')
NU = len(INPUTS)

# (low, default, high). Scenario grids unless a source is named in the provenance.
RANGES = {
    'PPP_reserve_fold': [11., 30., 89.],
    'G6PD_Km_NADP_uM': [6.1, 7.8, 7.8],
    'G6PD_Km_G6P_uM': [38.4, 42.8, 42.8],
    'G6PD_Ki_NADPH_uM': [9., 18., 36.],
    'other_source_share': [0., .3, .6],
    'other_source_Km_NADP_uM': [3., 10., 30.],
    'G6P_uM': [30., 100., 300.],
    'glucose_flux_uM_s': [3., 8., 15.],
    'pentose_return_tau_s': [10., 60., 600.],
    'recycle_fraction': [0., .5, 1.],
    'ATP_per_glucose': [30., 32., 36.],
}


@dataclass(frozen=True)
class NADPHSupplyParameters:
    PPP_reserve_fold: float = 30.
    G6PD_Km_NADP_uM: float = 7.8
    G6PD_Km_G6P_uM: float = 42.8
    G6PD_Ki_NADPH_uM: float = 18.
    other_source_share: float = .3
    other_source_Km_NADP_uM: float = 10.
    G6P_uM: float = 100.
    glucose_flux_uM_s: float = 8.
    pentose_return_tau_s: float = 60.
    recycle_fraction: float = .5
    ATP_per_glucose: float = 32.
    provenance: str = ('Scenario [claude] 2026-11-15. Reserve: H2O2 raises pentose-pathway flux in primary mixed '
                       'cerebrocortical cultures from 2.0 to 22.5 % and from 0.25 to 22.2 % of glucose (Ben-Yoseph 1994, '
                       '1996; rat, abstracts). Km for NADP+ and G6P: recombinant human G6PD (Hernandez-Ochoa 2025, Table 1). '
                       'Ki for NADPH: mouse enzyme, 18 uM (Lee 1979, abstract). Everything else is a declared scenario. '
                       'No human motor-neuron calibration.')

    def validate(self):
        d = asdict(self)
        values = [v for k, v in d.items() if k != 'provenance']
        if not np.isfinite(values).all() or min(values) < 0 or not self.provenance.strip():
            raise ValueError('Finite non-negative parameters and provenance required')
        if self.PPP_reserve_fold <= 1.:
            raise ValueError('Reserve must exceed one')
        if not 0. <= self.other_source_share < 1. or not 0. <= self.recycle_fraction <= 1.:
            raise ValueError('Shares must lie inside [0,1]')
        for k in ('G6PD_Km_NADP_uM', 'G6PD_Km_G6P_uM', 'G6PD_Ki_NADPH_uM', 'other_source_Km_NADP_uM', 'G6P_uM',
                  'glucose_flux_uM_s', 'pentose_return_tau_s', 'ATP_per_glucose'):
            if d[k] <= 0:
                raise ValueError(k + ' must be positive')


(S_T, S_VMAX, S_KG, S_KN, S_KI, S_VO, S_KO, S_JGLC, S_KGLY, S_KRET, S_RHO) = range(11)
NQ = 11


@njit
def fluxes(y, u, q):
    g6p, p5 = max(y[0], 0.), max(y[1], 0.)
    nadph = min(max(u[0], 0.), q[S_T])
    nadp = q[S_T] - nadph
    f = np.zeros(5)
    f[0] = q[S_JGLC] * u[1]
    f[1] = q[S_KGLY] * g6p
    f[2] = u[2] * q[S_VMAX] * g6p / (q[S_KG] + g6p) * nadp / (q[S_KN] * (1. + nadph / q[S_KI]) + nadp)
    f[3] = q[S_KRET] * p5
    f[4] = u[3] * q[S_VO] * nadp / (q[S_KO] + nadp)
    return f


@njit
def rhs(y, u, q):
    f = fluxes(y, u, q)
    d = np.zeros(NS)
    back = 5. / 6. * f[3]
    d[0] = f[0] - f[1] - f[2] + q[S_RHO] * back
    d[1] = f[2] - f[3]
    d[2] = f[0]
    d[3] = f[1] + (1. - q[S_RHO]) * back
    d[4] = f[2]; d[5] = f[2]; d[6] = 2. * f[2]; d[7] = f[4]; d[8] = f[3]
    return d


@njit
def supply(y, u, q):
    """NADPH produced per second by both routes."""
    f = fluxes(y, u, q)
    return 2. * f[2] + f[4]


def prepare(p, resting_demand_uM_s, NADP_total_uM):
    """Return (state, reference, report) stationary for the given resting NADPH demand."""
    p.validate()
    d0, total = float(resting_demand_uM_s), float(NADP_total_uM)
    if d0 <= 0 or total <= 0:
        raise ValueError('Positive resting demand and NADP total required')
    v_ox = (1. - p.other_source_share) * d0 / 2.
    v_other = p.other_source_share * d0
    sg = p.G6P_uM / (p.G6PD_Km_G6P_uM + p.G6P_uM)
    kn, ki = p.G6PD_Km_NADP_uM, p.G6PD_Ki_NADPH_uM
    top = total / (kn + total)                      # NADP saturation when all NADPH is oxidized
    vmax = p.PPP_reserve_fold * v_ox / (sg * top)
    a = top / p.PPP_reserve_fold                    # required resting saturation N/(kn*(1+(T-N)/ki)+N)
    nadp = a * kn * (1. + total / ki) / (1. - a + a * kn / ki)
    if not 0. < nadp < total:
        raise ArithmeticError('No resting NADP+ inside (0, total) for this reserve')
    vo = v_other * (p.other_source_Km_NADP_uM + nadp) / nadp if v_other > 0 else 0.
    kret = 1. / p.pentose_return_tau_s
    kgly = (p.glucose_flux_uM_s - v_ox + p.recycle_fraction * 5. / 6. * v_ox) / p.G6P_uM
    if kgly <= 0:
        raise ArithmeticError('Resting pentose-pathway flux exceeds the declared glucose flux')
    q = np.array([total, vmax, p.G6PD_Km_G6P_uM, kn, ki, vo, p.other_source_Km_NADP_uM, p.glucose_flux_uM_s,
                  kgly, kret, p.recycle_fraction])
    y = np.zeros(NS)
    y[0], y[1] = p.G6P_uM, v_ox / kret
    u = np.array([total - nadp, 1., 1., 1.])
    rate = rhs(y, u, q)
    residual = float(max(abs(rate[0]) / y[0], abs(rate[1]) / max(y[1], 1e-12), abs(supply(y, u, q) / d0 - 1.)))
    if residual > 1e-10:
        raise ArithmeticError('NADPH supply is not stationary at rest: ' + str(residual))
    report = dict(version=VERSION, resting_NADP_uM=float(nadp), resting_NADPH_uM=float(total - nadp),
                  resting_NADPH_fraction=float(1. - nadp / total), resting_NADPH_over_NADP=float((total - nadp) / nadp),
                  G6PD_Vmax_uM_s=float(vmax), other_source_Vmax_uM_s=float(vo),
                  resting_oxidative_flux_uM_s=float(v_ox), resting_G6PD_saturation=float(v_ox / vmax),
                  resting_PPP_share_of_glucose=float(v_ox / p.glucose_flux_uM_s),
                  maximal_PPP_share_of_glucose_at_resting_G6P=float(p.PPP_reserve_fold * v_ox / p.glucose_flux_uM_s),
                  maximal_NADPH_supply_uM_s=float(2. * vmax * sg * top + vo * total / (p.other_source_Km_NADP_uM + total)),
                  NADPH_supply_reserve_fold=float((2. * vmax * sg * top + vo * total / (p.other_source_Km_NADP_uM + total)) / d0),
                  ATP_equivalent_per_NADPH=float(p.ATP_per_glucose / 12.),
                  resting_ATP_equivalent_forgone_uM_s=float(v_ox * p.ATP_per_glucose / 6.),
                  resting_relative_rate=residual)
    return y, q, report


def checks(z, base):
    z = np.atleast_2d(z)
    d = z - base
    big = lambda i: max(1e-6, float(np.max(np.abs(z[:, i]))))
    return dict(
        carbon=float(np.max(np.abs(6. * d[:, 0] + 5. * d[:, 1] - 6. * d[:, 2] + 6. * d[:, 3] + d[:, 4])) / (6. * big(2))),
        pentose=float(np.max(np.abs(d[:, 1] - d[:, 5] + d[:, 8])) / big(5)),
        NADPH_ledger=float(np.max(np.abs(d[:, 6] - 2. * d[:, 5])) / big(6)),
        negative_pool=float(max(0., -z[:, :2].min())))
