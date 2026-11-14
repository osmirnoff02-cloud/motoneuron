"""Explicit assay models, Codex 2026-10-19. No human-fitted/default rates.

Amounts follow one specified N-terminal molecular lineage, not protein mass.
Calpain/caspase products are distinct latent classes, not assignments from kDa.
These linear, unsaturated assay models are not a physiological neuron model.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def _provenance(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Explicit nonempty provenance is required")


def _nonnegative(value, name):
    x = np.asarray(value, dtype=float)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return x


def _times(time_s):
    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be a finite strictly increasing vector")
    return t


def _interval(value, n, name):
    x = _nonnegative(value, name)
    if x.ndim == 0:
        return np.full(n, float(x))
    if x.shape != (n,):
        raise ValueError(f"{name} must be scalar or have one value per interval ({n})")
    return x


def simulate_substrate(time_s, *, substrate_name, initial_amounts,
                       synthesis_per_s, calpain_activity, caspase_activity,
                       calpain_cleavage_per_activity_s, caspase_cleavage_per_activity_s,
                       removal_per_s, provenance):
    """Exact piecewise-constant U -> Fcalpain / Fcaspase with synthesis/removal.

    initial_amounts: [U, Fcalpain, Fcaspase]; removal_per_s: three rates.
    All interval quantities apply on [time_s[i], time_s[i+1]). Activities use
    the explicit caller's scale; cleavage constants must match that scale.
    No conversion of spectrin reporter kinetics into TDP kinetics is implied.
    """
    _provenance(provenance)
    if not isinstance(substrate_name, str) or not substrate_name.strip():
        raise ValueError("An explicit substrate_name is required")
    t = _times(time_s)
    n = len(t) - 1
    initial = _nonnegative(initial_amounts, "initial_amounts")
    removal = _nonnegative(removal_per_s, "removal_per_s")
    if initial.shape != (3,) or removal.shape != (3,):
        raise ValueError("initial_amounts and removal_per_s must have three entries")
    synthesis = _interval(synthesis_per_s, n, "synthesis_per_s")
    ca = _interval(calpain_activity, n, "calpain_activity")
    cs = _interval(caspase_activity, n, "caspase_activity")
    kc = _interval(calpain_cleavage_per_activity_s, n, "calpain_cleavage_per_activity_s")
    ks = _interval(caspase_cleavage_per_activity_s, n, "caspase_cleavage_per_activity_s")
    # U, Fcalpain, Fcaspase, removed, synthesized, calpain events, caspase events, 1
    y = np.zeros((len(t), 8))
    y[0, :3] = initial
    y[0, 7] = 1
    for i, dt in enumerate(np.diff(t)):
        hc, hs = kc[i] * ca[i], ks[i] * cs[i]
        matrix = np.zeros((8, 8))
        matrix[0, 0] = -(hc + hs + removal[0])
        matrix[0, 7] = synthesis[i]
        matrix[1, 0], matrix[1, 1] = hc, -removal[1]
        matrix[2, 0], matrix[2, 2] = hs, -removal[2]
        matrix[3, :3] = removal
        matrix[4, 7] = synthesis[i]
        matrix[5, 0], matrix[6, 0] = hc, hs
        y[i + 1] = expm(matrix * dt) @ y[i]
    return dict(time_s=t, substrate_name=substrate_name,
                species_names=["intact", "calpain_product", "caspase_product"],
                amounts=y[:, :3], removed=y[:, 3], synthesized=y[:, 4],
                calpain_cleavage_events=y[:, 5], caspase_cleavage_events=y[:, 6],
                intact_loss_per_s=kc * ca + ks * cs + removal[0],
                synthesis_per_s=synthesis,
                balance_residual=y[:, :3].sum(axis=1) + y[:, 3] - y[:, 4] - initial.sum(),
                provenance=provenance,
                status="Conditional assay; one N-lineage; no human calibration")


def simulate_fluorescent_intact(time_s, *, initial_fluorescent_amount,
                               fluorescent_synthesis_per_s, biochemical_loss_per_s,
                               illumination_fraction, bleach_per_illuminated_s,
                               provenance):
    """Track intact fluorescent sensor, including fresh unbleached synthesis.

    Exact intervals: dV/dt=s-(biochemical_loss+b*illumination_fraction)*V.
    Light is zero between acquisition windows. No maturation delay, dark
    recovery or donor/acceptor-specific bleaching is inferred. A fractional
    duty cycle averages fast alternating illumination inside each interval.
    """
    _provenance(provenance)
    t = _times(time_s)
    n = len(t) - 1
    initial = _nonnegative(initial_fluorescent_amount, "initial_fluorescent_amount")
    if initial.ndim != 0:
        raise ValueError("initial_fluorescent_amount must be scalar")
    synthesis = _interval(fluorescent_synthesis_per_s, n, "fluorescent_synthesis_per_s")
    loss = _interval(biochemical_loss_per_s, n, "biochemical_loss_per_s")
    light = _interval(illumination_fraction, n, "illumination_fraction")
    bleach = _interval(bleach_per_illuminated_s, n, "bleach_per_illuminated_s")
    if np.any(light > 1):
        raise ValueError("illumination_fraction must be between zero and one")
    v = np.empty(len(t))
    v[0] = float(initial)
    integral = np.zeros(n)
    for i, dt in enumerate(np.diff(t)):
        rate = loss[i] + bleach[i] * light[i]
        if rate == 0:
            v[i + 1] = v[i] + synthesis[i] * dt
            integral[i] = v[i] * dt + 0.5 * synthesis[i] * dt ** 2
        else:
            factor = -np.expm1(-rate * dt) / rate
            v[i + 1] = v[i] * np.exp(-rate * dt) + synthesis[i] * factor
            # Matrix exponential avoids cancellation for small rate*dt.
            m = np.array([[-rate, 0, synthesis[i]], [1, 0, 0], [0, 0, 0]])
            integral[i] = (expm(m * dt) @ np.array([v[i], 0, 1]))[1]
    return dict(time_s=t, fluorescent_intact=v,
                interval_mean_fluorescent_intact=integral / np.diff(t),
                cumulative_illumination_s=np.r_[0, np.cumsum(light * np.diff(t))],
                provenance=provenance,
                status="Conditional fluorescent population; new synthesis is unbleached")


def robinson_fret(fluorescent_intact, *, gain, led_off,
                  excitation_background, baseline_index, provenance):
    """Robinson EYFP emission, LED-off subtraction, then percent delta F/Fpre.

    Input may be acquisition-window means from simulate_fluorescent_intact.
    led_off is excitation-independent background; excitation_background is
    residual excited background (not removed by LED-off). No CFP/YFP ratio.
    """
    _provenance(provenance)
    v = _nonnegative(fluorescent_intact, "fluorescent_intact")
    if v.ndim != 1 or len(v) == 0:
        raise ValueError("fluorescent_intact must be a nonempty vector")
    g = _nonnegative(gain, "gain")
    if g.ndim != 0 or g <= 0:
        raise ValueError("gain must be a positive scalar")
    off = _interval(led_off, len(v), "led_off")
    background = _interval(excitation_background, len(v), "excitation_background")
    if isinstance(baseline_index, bool) or not isinstance(baseline_index, (int, np.integer)) or not 0 <= baseline_index < len(v):
        raise ValueError("baseline_index must identify an observation")
    on = float(g) * v + background + off
    corrected = on - off
    if corrected[baseline_index] <= 0:
        raise ValueError("Positive baseline signal required for normalization")
    return dict(excitation_on=on, led_off=off, corrected_eyfp=corrected,
                delta_f_over_f=corrected / corrected[baseline_index] - 1,
                delta_fret_percent=100 * (corrected / corrected[baseline_index] - 1),
                provenance=provenance, definition="EYFP emission; not donor/acceptor ratio")


def epitope_linear_mixture(amounts, *, species_names, band_names,
                           epitope_gain_matrix, background, provenance):
    """Explicit bands = amounts @ gain.T + background. No kDa inference.

    Matrix columns identify specified species, rows observed bands/epitopes.
    Rank deficiency records inability to infer origins from those bands alone.
    """
    _provenance(provenance)
    a = _nonnegative(amounts, "amounts")
    gains = _nonnegative(epitope_gain_matrix, "epitope_gain_matrix")
    if a.ndim != 2 or gains.ndim != 2 or gains.shape[1] != a.shape[1]:
        raise ValueError("amounts must be time x species, gains bands x species")
    if len(species_names) != a.shape[1] or len(band_names) != gains.shape[0]:
        raise ValueError("Species and band labels must match matrix dimensions")
    if len(set(species_names)) != len(species_names) or len(set(band_names)) != len(band_names):
        raise ValueError("Labels must be unique")
    if any(not isinstance(x, str) or not x.strip() for x in [*species_names, *band_names]):
        raise ValueError("Explicit nonempty species and band names required")
    bg = _interval(background, gains.shape[0], "background")
    rank = int(np.linalg.matrix_rank(gains))
    return dict(signal=a @ gains.T + bg, species_names=list(species_names),
                band_names=list(band_names), gain_rank=rank,
                unidentifiable_dimensions=a.shape[1] - rank, provenance=provenance)


def gerencser_corrected_fret(unmixed_fret, unmixed_cfp, unmixed_yfp, *, provenance):
    """Gerencser: F'_FRET / sqrt(F'_CFP * F'_YFP), AFTER external unmixing.

    This function performs no spectral correction. Negative finite FRET after
    unmixing is retained; nonpositive denominators become NaN with valid=False.
    This normalization is not the Robinson photometry measurement.
    """
    _provenance(provenance)
    f, c, y = (np.asarray(x, dtype=float) for x in (unmixed_fret, unmixed_cfp, unmixed_yfp))
    if f.shape != c.shape or f.shape != y.shape or f.ndim != 1:
        raise ValueError("Unmixed channels must be equal-length vectors")
    if not all(np.isfinite(x).all() for x in (f, c, y)):
        raise ValueError("Unmixed channels must be finite")
    valid = (c > 0) & (y > 0)
    corrected = np.full(f.shape, np.nan)
    corrected[valid] = f[valid] / (np.sqrt(c[valid]) * np.sqrt(y[valid]))
    return dict(corrected_fret=corrected, valid=valid, provenance=provenance,
                definition="F'_FRET / sqrt(F'_CFP * F'_YFP); input already unmixed")
