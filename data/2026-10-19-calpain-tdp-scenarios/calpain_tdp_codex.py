"""Codex, 2026-10-19: conditional local Ca -> calpain -> TDP processing.

No human biochemical fit, CAPN isoform assignment, aggregation, nuclear function,
viability, or NMJ output. P -> F -> R follows one N-terminal lineage, not protein
mass: F pools large N-retaining products (1-286/1-324), R smaller products.
Omitted complementary C-terminal products must not be counted as missing mass.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from numba import njit

STATE_NAMES = ('active_fraction', 'intact_TDP', 'large_N_fragment',
               'small_N_product', 'removed_lineages', 'formed_large_cumulative',
               'secondary_cleavage_cumulative', 'large_removed_cumulative',
               'synthesized_cumulative')


@dataclass(frozen=True)
class CalpainTDPParameters:
    ca_half_uM: float
    hill_n: float
    activation_tau_s: float
    deactivation_tau_s: float
    cast_over_ki: float
    cleavage1_per_s: float
    cleavage2_per_s: float
    intact_half_life_s: float
    large_half_life_s: float
    small_half_life_s: float
    provenance: str

    def vector(self):
        positive = [self.ca_half_uM, self.hill_n, self.activation_tau_s,
                    self.deactivation_tau_s, self.intact_half_life_s,
                    self.large_half_life_s, self.small_half_life_s]
        nonnegative = [self.cast_over_ki, self.cleavage1_per_s, self.cleavage2_per_s]
        if (not np.isfinite(positive + nonnegative).all() or min(positive) <= 0
                or min(nonnegative) < 0 or not self.provenance.strip()):
            raise ValueError('Finite positive times/Ca/Hill, nonnegative rates/CAST and provenance required')
        return np.array([self.ca_half_uM, self.hill_n, self.activation_tau_s,
                         self.deactivation_tau_s, self.cast_over_ki,
                         self.cleavage1_per_s, self.cleavage2_per_s,
                         math.log(2)/self.intact_half_life_s,
                         math.log(2)/self.large_half_life_s,
                         math.log(2)/self.small_half_life_s], dtype=float)


@njit
def _hill(c, p):
    if c <= 0:
        return 0.
    z = p[1]*(math.log(c)-math.log(p[0]))
    if z >= 0:
        return 1./(1.+math.exp(-z))
    e = math.exp(z)
    return e/(1.+e)


@njit
def _rhs(y, c, block, synthesis, p):
    target = _hill(c, p)
    tau = p[2] if target > y[0] else p[3]
    effective = y[0]*(1.-block)/(1.+p[4])
    q1 = p[5]*effective*y[1]
    q2 = p[6]*effective*y[2]
    dP, dF, dR = p[7]*y[1], p[8]*y[2], p[9]*y[3]
    return np.array([(target-y[0])/tau, synthesis-dP-q1,
                     q1-q2-dF, q2-dR, dP+dF+dR, q1, q2, dF, synthesis])


@njit
def _integrate(t, c, blocks, y0, synthesis, p, max_step):
    out = np.empty((len(t), len(y0)))
    out[0] = y0
    # Nonnegative compartment system; keep h*largest rate <= 0.15.
    rate = max(1./min(p[2], p[3]), p[5]+p[7], p[6]+p[8], p[9])
    safe_step = min(max_step, .15/rate)
    for i in range(len(t)-1):
        dt = t[i+1]-t[i]
        # Resolve the on/off relaxation switch during changing Ca; constant
        # recovery tails can retain the larger, rate-limited stable step.
        local_step = min(safe_step, .1) if abs(c[i+1]-c[i]) > 1e-12 else safe_step
        steps = max(1, int(math.ceil(dt/local_step)))
        h = dt/steps
        y = out[i].copy()
        for j in range(steps):
            c1 = c[i]+(c[i+1]-c[i])*j/steps
            cm = c[i]+(c[i+1]-c[i])*(j+.5)/steps
            c2 = c[i]+(c[i+1]-c[i])*(j+1.)/steps
            # Inhibition is a left-constant command; switching times are input knots.
            k1 = _rhs(y, c1, blocks[i], synthesis, p)
            k2 = _rhs(y+h*k1/2, cm, blocks[i], synthesis, p)
            k3 = _rhs(y+h*k2/2, cm, blocks[i], synthesis, p)
            k4 = _rhs(y+h*k3, c2, blocks[i], synthesis, p)
            y = y+h*(k1+2*k2+2*k3+k4)/6
        out[i+1] = y
    return out


def baseline_equilibrium(basal_ca_uM, parameters):
    """Normalize accessible intact P=1; balance source at uninhibited baseline.

    F and R can be large for some assumptions. This equilibrium is a mathematical
    initialization, not evidence of a healthy state. Keep synthesis fixed across
    matched intervention arms; never rebalance separately under inhibition.
    """
    p = parameters.vector()
    if not np.isfinite(basal_ca_uM) or basal_ca_uM < 0:
        raise ValueError('Finite nonnegative baseline required')
    a = float(_hill(basal_ca_uM, p))
    eff = a/(1+p[4])
    f = p[5]*eff/(p[8]+p[6]*eff)
    r = p[6]*eff*f/p[9]
    return dict(initial_state=np.array([a, 1., f, r, 0., 0., 0., 0., 0.]),
                synthesis_per_s=float(p[7]+p[5]*eff), basal_ca_uM=float(basal_ca_uM),
                status='mathematical baseline, P=1; not human healthy calibration')


def simulate_calpain_tdp(time_s, ca_uM, *, parameters, ca_provenance, compartment,
                         baseline, inhibition=0., max_step_s=2.):
    """Piecewise-linear absolute FREE Ca in uM, seconds; no Vm/AP conversion.

    baseline must be created before intervention, or explicitly supplied as a
    pulse-chase initial condition. Inhibition is fractional activity block [0,1],
    not drug concentration or a calibrated dose. CAST/Ki is a dimensionless
    scenario ratio, not RNA expression. Amounts are P-baseline N-lineage units.
    """
    if not isinstance(parameters, CalpainTDPParameters):
        raise ValueError('Explicit CalpainTDPParameters required')
    p = parameters.vector()
    t, c = np.asarray(time_s, float), np.asarray(ca_uM, float)
    if (t.ndim != 1 or len(t) < 2 or c.shape != t.shape or
            not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or
            not np.isfinite(c).all() or np.any(c < 0)):
        raise ValueError('Ordered finite seconds and matching nonnegative free Ca in uM required')
    if not ca_provenance.strip() or not compartment.strip():
        raise ValueError('Explicit Ca provenance and compartment required')
    b = np.full(t.shape, float(inhibition)) if np.ndim(inhibition) == 0 else np.asarray(inhibition, float)
    if b.shape != t.shape or not np.isfinite(b).all() or np.any((b < 0) | (b > 1)):
        raise ValueError('Activity inhibition must be in [0,1] and match time')
    y0 = np.asarray(baseline['initial_state'], float)
    synthesis = float(baseline['synthesis_per_s'])
    if (y0.shape != (9,) or not np.isfinite(y0).all() or np.any(y0 < 0) or y0[0] > 1
            or not np.isfinite(synthesis) or synthesis < 0 or
            not np.isfinite(max_step_s) or max_step_s <= 0):
        raise ValueError('Invalid state/source/integration step')
    y = _integrate(t, c, b, y0, synthesis, p, max_step_s)
    mass = y[:, 1:5].sum(axis=1)-y[:, 8]
    residual = float(np.max(abs(mass-mass[0])))
    if not np.isfinite(y).all() or y.min() < -1e-9 or y[:, 0].max() > 1+1e-9:
        raise ArithmeticError('Nonphysical state; no clipping applied')
    if residual > 1e-8*max(1., float(y[:, 1:5].sum(axis=1).max())):
        raise ArithmeticError('N-terminal lineage accounting failed')
    effective = y[:, 0]*(1-b)/(1+p[4])
    result = {name: y[:, j] for j, name in enumerate(STATE_NAMES)}
    result.update(time_s=t, ca_uM=c, effective_activity=effective,
                  formation_rate_per_s=p[5]*effective*y[:, 1],
                  secondary_rate_per_s=p[6]*effective*y[:, 2],
                  large_removal_rate_per_s=p[8]*y[:, 2], inhibition=b,
                  lineage_balance_max_error=residual, parameters=asdict(parameters),
                  ca_provenance=ca_provenance, compartment=compartment,
                  nuclear_TDP_function=None, aggregation=None, NMJ_function=None,
                  status='conditional biochemical scenario; not calibrated human physiology',
                  electrical_feedback=False)
    return result
