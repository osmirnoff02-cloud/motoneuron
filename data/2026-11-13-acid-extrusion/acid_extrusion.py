"""[claude] 2026-11-13. Optional acid-extrusion closure for PMCA counter-protons.

Author: Claude (Fable 5.2). Plan row N05.1.4, assumption A08. The hooks that
connect this module to Codex files are marked "[claude] 2026-11-13" there.

Problem. PMCA exports Ca2+ electroneutrally against protons (2 H+ per Ca2+),
but the host model had no proton pool. Ca2+ entered as current, K+ left to
balance the charge, and the cell kept a permanent K+ deficit equal to twice
the excess PMCA export (Codex charge diagnostic of 2026-11-13).

Closure, written for deviations from the fitted rest only:

    dH/dt    = H_per_Ca*(J_PMCA - J_PMCA_rest) - J_ex
    J_ex     = H/recovery_tau_s * Na_o/Na_o_rest
    dNa_i/dt += J_ex            dNa_o/dt -= J_ex/outside_volume_ratio

H is the proton-equivalent acid load (free plus buffered) as amount per
reference volume, like the host ledgers. J_ex is an electroneutral Na+/H+
exchange: no voltage term and no direct ATP term. The existing Na/K pump then
removes the Na+ load, pays the ATP and brings K+ back.

Declared limits of version 1 (scenario, NOT a human motor-neuron fit):
  * the basal PMCA proton load and the basal exchanger flux are assumed to be
    inside the already fitted resting Na+ background; only deviations move;
  * one linear law on both sides of rest (alkaline side = reduced basal flux);
  * no bicarbonate/Cl- routes, no proton diffusion between compartments,
    no pH feedback on channels, pumps or enzymes;
  * the terminal PMCA has no ion block in the host and is not covered;
  * pH is an observer only: delta_pH = -H_mobile/buffer.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit

VERSION = 'acid-extrusion-claude-v1-2026-11-13'
STATES_PER_COMPARTMENT = 3
# Host cumulative PMCA Ca2+ export ledgers (uM per reference volume).
PMCA_INDEX = (130, 144)
NAMES = tuple(c + '_' + s for c in ('soma', 'process') for s in
              ('acid_excess_uM', 'acid_PMCA_excess_in_uM', 'acid_NaH_exchange_out_uM'))
# Scenario grids, NOT measured human confidence intervals.
RANGES = {'H_per_Ca': [2.],
          'recovery_tau_s': [20., 120., 600.],
          'buffer_mM_per_pH': [10., 20., 46.],
          'resting_pH': [7.02, 7.18, 7.24]}


@dataclass(frozen=True)
class AcidExtrusionParameters:
    H_per_Ca: float = 2.
    recovery_tau_s: float = 120.
    buffer_mM_per_pH: float = 20.
    resting_pH: float = 7.18
    provenance: str = ('Scenario [claude] 2026-11-13: 2H/Ca PMCA (Thomas 2009, 2011; snail neurons); '
                       'recovery time and buffering from rat neurons and motor terminals, abstracts only; '
                       'no human motor-neuron calibration.')

    def validate(self):
        values = [self.H_per_Ca, self.recovery_tau_s, self.buffer_mM_per_pH, self.resting_pH]
        if (not np.isfinite(values).all() or self.H_per_Ca < 0 or self.recovery_tau_s <= 0
                or self.buffer_mM_per_pH <= 0 or not 6. < self.resting_pH < 8.
                or not self.provenance.strip()):
            raise ValueError('Acid extrusion needs finite H_per_Ca>=0, recovery_tau_s>0, '
                             'buffer_mM_per_pH>0, resting_pH in (6,8) and provenance')


def prepare(base, pmca_rest_uM_s, iv, p):
    """Append six zero states; return (state, reference rows).

    Reference row per compartment: H_per_Ca, resting PMCA flux (uM/s per
    reference volume), exchange rate constant (1/s), resting external Na+
    (uM), outside/inside volume ratio.
    """
    p.validate()
    rest = np.asarray(pmca_rest_uM_s, float)
    if rest.shape != (2,) or not np.isfinite(rest).all() or np.any(rest < 0):
        raise ValueError('Two finite non-negative resting PMCA fluxes required')
    rows = []
    for j in range(2):
        b = 199 + 20 * j
        if base[b + 2] <= 0 or iv[10] <= 0:
            raise ValueError('Positive external Na and outside volume ratio required')
        rows.append([p.H_per_Ca, rest[j], 1. / p.recovery_tau_s, base[b + 2], iv[10]])
    return np.r_[base, np.zeros(2 * STATES_PER_COMPARTMENT)], np.asarray(rows)


@njit
def finish(z, old, ref, offset):
    """Add the acid block to an already complete host derivative.

    Called after every host contribution to the PMCA ledgers and before the
    mobile-water scaling, so all terms are amounts per reference volume.
    """
    out = np.zeros(len(z))
    out[:len(old)] = old
    for j in range(2):
        k = offset + 3 * j
        b = 199 + 20 * j
        hca, rest, rate, nao0, outside = ref[j]
        load = hca * (old[PMCA_INDEX[j]] - rest)
        exchange = rate * z[k] * z[b + 2] / nao0
        out[k] = load - exchange
        out[k + 1] = load
        out[k + 2] = exchange
        # Electroneutral Na+/H+ exchange: Na+ in, proton equivalent out.
        out[b] += exchange
        out[b + 2] -= exchange / outside
    return out


def checks(z, base, ref, offset, elapsed_s):
    """Balances of the block. elapsed_s is time since the model origin."""
    z = np.atleast_2d(z)
    elapsed = np.asarray(elapsed_s, float)
    result = {}
    for j in range(2):
        k = offset + 3 * j
        hca, rest = ref[j, 0], ref[j, 1]
        d = z[:, k:k + 3] - base[k:k + 3]
        result[f'acid_{j}_proton_balance_uM'] = float(np.max(np.abs(d[:, 0] - d[:, 1] + d[:, 2])))
        pm = z[:, PMCA_INDEX[j]] - base[PMCA_INDEX[j]]
        scale = max(1., hca * float(np.max(np.abs(pm))))
        result[f'acid_{j}_PMCA_coupling_relative'] = float(
            np.max(np.abs(d[:, 1] - hca * (pm - rest * elapsed))) / scale)
    return result


def observe(z, ref, offset, p, water=None):
    """Read-only view. pH uses the declared buffer; it does not act on the model."""
    z = np.atleast_2d(z)
    ratios = [1., 1.] if water is None else list(water['water_to_reference_ratio'])
    rows = []
    for j, name in enumerate(('soma', 'process')):
        k = offset + 3 * j
        b = 199 + 20 * j
        hca, rest, rate, nao0, outside = ref[j]
        mobile = z[:, k] / ratios[j]
        exchange = rate * z[:, k] * z[:, b + 2] / nao0
        delta_ph = -mobile / (1000. * p.buffer_mM_per_pH)
        rows.append(dict(
            compartment=name,
            acid_excess_reference_uM=z[:, k],
            acid_excess_mobile_water_uM=mobile,
            delta_pH=delta_ph,
            pH=p.resting_pH + delta_ph,
            exchange_Na_in_uM_per_s_reference=exchange,
            implied_total_exchange_uM_per_s_reference=hca * rest + exchange,
            cumulative_PMCA_proton_excess_uM=z[:, k + 1],
            cumulative_exchange_uM=z[:, k + 2],
            resting_PMCA_uM_per_s_reference=float(rest),
            water_to_reference_ratio=float(ratios[j])))
    return dict(compartments=rows, offset=offset, version=VERSION,
                scope=('Deviation-from-rest proton-equivalent pool with linear electroneutral Na/H exchange. '
                       'pH is an observer with a declared buffer. No bicarbonate/Cl routes, proton diffusion, '
                       'pH feedback or terminal coverage. Scenario, not a human motor-neuron calibration.'))
