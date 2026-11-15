"""Codex 2026-11-15: read-only GSH turnover/assay diagnostics.

No ODE parameters, initial state or fluxes are changed. A first-order loss
time constant is not a half-life, and a BSO or GEE assay is not an isotope
pulse-chase of a healthy human motor neuron. Concentrations and fluxes must
refer to the same physical water volume.
"""
import math
import numpy as np


def _finite_nonnegative(name, value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(name + ' must be finite and non-negative')
    return value


def gsh_turnover_budget(*, gsh_uM, loss_tau_s, intracellular_recovery_fraction,
                        other_use_fraction, additional_synthesis_uM_s=0.):
    """Stationary cysteine budget, with explicit assumptions, not a refit.

gross GSH synthesis = first-order loss + additional replacement demand;
Cys uptake = (gross synthesis - returned Cys)/(1-other_use_fraction).
This matches the algebra in Claude's independent supply.prepare. It does
not assume the integrated neuron has this equilibrium or this recycling.
The ATP value covers GSH synthesis only (2 ATP per molecule), excluding
transport, other protein synthesis, and any recycling pathway cost.
"""
    gsh = _finite_nonnegative('gsh_uM', gsh_uM)
    tau = _finite_nonnegative('loss_tau_s', loss_tau_s)
    recovery = _finite_nonnegative('intracellular_recovery_fraction', intracellular_recovery_fraction)
    other = _finite_nonnegative('other_use_fraction', other_use_fraction)
    extra = _finite_nonnegative('additional_synthesis_uM_s', additional_synthesis_uM_s)
    if tau == 0 or recovery > 1 or other >= 1:
        raise ValueError('tau>0, recovery<=1 and other_use_fraction<1 required')
    loss = gsh / tau
    synthesis = loss + extra
    returned = recovery * loss
    net_gsh = synthesis - returned
    uptake = net_gsh / (1-other)
    return dict(loss_tau_s=tau, loss_only_half_life_s=math.log(2)*tau,
                gross_GSH_synthesis_uM_s=synthesis,
                recovered_Cys_uM_s=returned,
                net_Cys_for_GSH_uM_s=net_gsh,
                other_Cys_use_uM_s=other*uptake,
                total_Cys_uptake_uM_s=uptake,
                total_Cys_uptake_mM_per_day=uptake*86.4,
                GSH_synthesis_ATP_uM_s=2*synthesis,
                scope='Conditional stationary budget, no ODE refit; all fluxes per same water volume')


def gsh_assay_total(gsh_uM, gssg_uM):
    """GSx = GSH + 2 GSSG in GSH equivalents (Dringen 1999 assay).

GSSG must be supplied in molecules, not already in GSH equivalents.
Protein mixed disulfides, conjugates and extracellular GSH are excluded.
"""
    gsh, gssg = np.asarray(gsh_uM, dtype=float), np.asarray(gssg_uM, dtype=float)
    if gsh.shape != gssg.shape:
        raise ValueError('GSH and GSSG require the same shape and compartment')
    if not np.isfinite(gsh).all() or not np.isfinite(gssg).all() or np.any(gsh<0) or np.any(gssg<0):
        raise ValueError('Physical concentrations must be finite and non-negative')
    return gsh + 2*gssg


def gsh_turnover_sources():
    """Source-specific assay anchors; deliberately no pooled human range."""
    return [
        dict(source='Dringen et al. 1999', doi='10.1523/JNEUROSCI.19-02-00562.1999',
             system='E16 rat brain neuron-rich cultures, DIV5-7',
             assay='GSx after 0.5mM BSO for 4h, Fig4',
             loss_fraction_lower_bound=.30, time_h=4.,
             status='Single endpoint under synthesis inhibitor; not a fitted basal half-life'),
        dict(source='Sedlak et al. 2019', doi='10.1073/pnas.1817885116',
             system='Rat cortical neurons, Neurobasal/B27',
             assay='BSO200uM,24h; SI FigS3B single endpoint, not a time series',
             reported_half_life_h=[1.,4.],
             endpoint_fraction_approx=.17, endpoint_time_h=24.,
             status='SI supplied and fully read 2026-11-15: half-life1-4h is stated in main text but cannot be fitted from S3B; no human-MN calibration'),
        dict(source='Zeevalk et al. 2007', doi='10.1016/j.expneurol.2006.09.004',
             system='Rat ventral mesencephalic culture, E15, GEE added DIV8; 37C',
             assay='5mM GEE 24h preload then clearance, Fig3; 0/8/16/24/48h',
             reported_half_life_h=[10.14,13.86],
             status='Control and BSO-predepleted groups, not bounds or basal turnover'),
        dict(source='Gutbier et al. 2018', doi='10.1038/s41418-018-0229-x',
             system='Human LUHMES neurons; murine/human astrocyte-conditioned medium',
             assay='GSH, thiol supplementation and proteasome-inhibition responses, Fig4',
             status='Human mechanistic benchmark; not healthy iPSC-MN turnover kinetics'),
    ]
