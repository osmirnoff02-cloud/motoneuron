"""[claude] 2026-11-14. Standalone cytosolic hydrogen-peroxide block.

Author: Claude (Fable 5.2). Plan row N11.2.3 (redox half). Version 1 is NOT
connected to the host neuron. It can run alone (NADPH, glutathione and ATP as
inputs) or together with the lipid block through coupled_redox.py.

Structure follows "Model 1" of Selvaggio, Coelho & Salvador 2018
(doi 10.1016/j.redox.2017.12.008): typical 2-Cys peroxiredoxins, thioredoxin,
thioredoxin reductase, sulfiredoxin and lumped alternative sinks.

Pools, uM of cytosol, time in seconds
  H        hydrogen peroxide
  PS       peroxiredoxin thiolate          PSOH  sulfenate
  PSS      disulfide                       PSO2  hyperoxidized (sulfinate)
  TSH      reduced thioredoxin             (oxidized = total - TSH)

Reactions
  supply  v_sup = v0*((1-m) + m*respiration_rel)*supply_multiplier
  v1 PS + H -> PSOH                kOx
  v2 PSOH + H -> PSO2              kSulf
  v3 PSOH -> PSS                   kCond
  v4 PSS + TSH -> PS + TSS         kRed
  v5 PSO2 -> PSOH                  kSrx, one ATP, vanishes without ATP
  v6 TSS -> TSH                    thioredoxin reductase, one NADPH
  v7 H -> water                    alternative sinks kAlt; a declared share
                                   uses glutathione (2 GSH -> GSSG)

Outputs for the lipid block: peroxide relative to rest (drive of radical
initiation), glutathione oxidation by the glutathione-dependent sinks and
NADPH use by thioredoxin reductase.

Declared limits: one well-mixed cytosol (the source paper notes real
gradients around supply sites), no mitochondrial matrix pool, no NADPH
oxidases, no peroxynitrite, peroxiredoxins 1 and 2 lumped. Enzyme amounts and
rate constants are estimates for human cell lines, not for neurons; the
supply rate is a scenario. Not a human motor-neuron calibration.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

VERSION = 'peroxide-claude-v1-2026-11-14'
POOLS = ('H2O2', 'Prx_S', 'Prx_SOH', 'Prx_SS', 'Prx_SO2', 'Trx_SH')
LEDGERS = ('H2O2_supplied', 'H2O2_by_Prx', 'H2O2_by_hyperoxidation', 'H2O2_by_GSH_sinks', 'H2O2_by_other_sinks',
           'Prx_condensed', 'Prx_reduced', 'Prx_repaired', 'Trx_reduced', 'NADPH_consumed', 'GSH_oxidized', 'ATP_consumed')
NAMES = POOLS + LEDGERS
NS = len(NAMES)
INPUTS = ('respiration_rel', 'supply_multiplier', 'ATP_uM', 'NADPH_uM', 'GSH_uM',
          'm_TrxR', 'm_Prx_oxidation', 'm_sulfiredoxin', 'm_alternative_sinks')
NU = len(INPUTS)

# Thirteen nucleated human cell types, Selvaggio 2018 Table 2 (cell lines and hepatocytes; erythrocytes excluded).
CELL_TABLE = {
    'Prx_total_uM': [49., 76., 140., 65., 93., 170., 100., 86., 92., 87., 84., 110., 86.],
    'Trx_total_uM': [29., 44., 46., 24., 27., 36., 28., 17., 23., 69., 18., 24., 63.],
    'kAlt_per_s': [44., 65., 160., 180., 130., 85., 180., 130., 79., 140., 97., 74., 210.],
    'kCond_per_s': [8.7, 8.5, 7.3, 7.3, 7.1, 7.0, 6.9, 5.9, 6.4, 6.6, 8.0, 7.4, 7.4],
    'kSulf_per_uM_s': [1.7e-3, 2.0e-3, 3.7e-3, 3.8e-3, 4.1e-3, 4.3e-3, 4.4e-3, 5.8e-3, 5.1e-3, 4.7e-3, 2.7e-3, 3.6e-3, 3.7e-3],
    'kSrx_per_s': [4.5e-3, 2.6e-3, .41e-3, 1.1e-3, 2.5e-3, .52e-3, .34e-3, .93e-3, 3.3e-3, 6.6e-3, 2.4e-3, 1.5e-3, .29e-3],
    'TrxR_Vmax_uM_s': [590., 580., 190., 230., 120., 180., 130., 310., 230., 270., 340., 47., 50.],
}
# (low, default, high). Cell-line spans for the seven table parameters; scenarios for the rest.
RANGES = {k: [min(v), float(np.median(v)), max(v)] for k, v in CELL_TABLE.items()}
RANGES.update({
    'kOx_per_uM_s': [40., 40., 40.],
    'kRed_per_uM_s': [.21, .21, .21],
    'TrxR_Km_uM': [1.8, 1.8, 1.8],
    'supply_uM_s': [.03, .3, 3.],
    'mitochondrial_share': [.2, .5, .8],
    'GSH_dependent_share_of_alternative_sinks': [.1, .5, .9],
    'GSH_half_uM': [10., 50., 200.],
    'NADPH_half_uM': [1., 5., 20.],
    'ATP_half_uM': [100., 500., 1500.],
    'repair_ATP': [1., 1., 1.],
})


@dataclass(frozen=True)
class PeroxideParameters:
    Prx_total_uM: float = RANGES['Prx_total_uM'][1]
    Trx_total_uM: float = RANGES['Trx_total_uM'][1]
    kAlt_per_s: float = RANGES['kAlt_per_s'][1]
    kCond_per_s: float = RANGES['kCond_per_s'][1]
    kSulf_per_uM_s: float = RANGES['kSulf_per_uM_s'][1]
    kSrx_per_s: float = RANGES['kSrx_per_s'][1]
    TrxR_Vmax_uM_s: float = RANGES['TrxR_Vmax_uM_s'][1]
    kOx_per_uM_s: float = 40.
    kRed_per_uM_s: float = .21
    TrxR_Km_uM: float = 1.8
    supply_uM_s: float = .3
    mitochondrial_share: float = .5
    GSH_dependent_share_of_alternative_sinks: float = .5
    GSH_half_uM: float = 50.
    NADPH_half_uM: float = 5.
    ATP_half_uM: float = 500.
    repair_ATP: float = 1.
    resting_ATP_uM: float = 2900.
    resting_NADPH_uM: float = 49.
    resting_GSH_uM: float = 2000.
    provenance: str = ('Scenario [claude] 2026-11-14. Structure and seven enzyme parameters: Selvaggio, Coelho & Salvador '
                       '2018, Table 2, medians of thirteen nucleated human cell types (estimates for cell lines and '
                       'hepatocytes, not neurons). Supply rate, its mitochondrial share, the glutathione-dependent share '
                       'of alternative sinks and all half-saturation constants are declared scenarios. '
                       'No human motor-neuron calibration.')

    def validate(self):
        d = asdict(self)
        values = [v for k, v in d.items() if k != 'provenance']
        if not np.isfinite(values).all() or min(values) < 0 or not self.provenance.strip():
            raise ValueError('Finite non-negative parameters and provenance required')
        for k in ('mitochondrial_share', 'GSH_dependent_share_of_alternative_sinks'):
            if not 0. <= d[k] <= 1.:
                raise ValueError(k + ' must lie inside [0,1]')
        for k in ('Prx_total_uM', 'Trx_total_uM', 'kCond_per_s', 'TrxR_Vmax_uM_s', 'kOx_per_uM_s', 'kRed_per_uM_s',
                  'TrxR_Km_uM', 'supply_uM_s', 'resting_ATP_uM', 'resting_NADPH_uM', 'resting_GSH_uM',
                  'GSH_half_uM', 'NADPH_half_uM', 'ATP_half_uM'):
            if d[k] <= 0:
                raise ValueError(k + ' must be positive')


(Q_PT, Q_TT, Q_KALT, Q_KCOND, Q_KSULF, Q_KSRX, Q_VMAX, Q_KOX, Q_KRED, Q_KM, Q_V0, Q_M, Q_G, Q_KG, Q_KN, Q_KA,
 Q_CATP, Q_SG0, Q_SN0, Q_SA0) = range(20)
NQ = 20
NF = 9


@njit
def fluxes(y, u, q):
    """Reaction rates (uM/s). f[7] is the glutathione-dependent part of the alternative sinks, f[8] the rest."""
    h = max(y[0], 0.)
    ps, psoh, pss, pso2 = max(y[1], 0.), max(y[2], 0.), max(y[3], 0.), max(y[4], 0.)
    tsh = min(max(y[5], 0.), q[Q_TT])
    tss = q[Q_TT] - tsh
    atp, nadph, gsh = max(u[2], 0.), max(u[3], 0.), max(u[4], 0.)
    f = np.zeros(NF)
    f[0] = q[Q_V0] * ((1. - q[Q_M]) + q[Q_M] * u[0]) * u[1]                    # supply
    f[1] = u[6] * q[Q_KOX] * ps * h                                              # Prx oxidation
    f[2] = q[Q_KSULF] * psoh * h                                                 # hyperoxidation
    f[3] = q[Q_KCOND] * psoh                                                     # condensation to disulfide
    f[4] = q[Q_KRED] * pss * tsh                                                 # reduction by thioredoxin
    f[5] = u[7] * q[Q_KSRX] * pso2 * (atp / (q[Q_KA] + atp)) / q[Q_SA0]          # sulfiredoxin repair
    f[6] = u[5] * q[Q_VMAX] * tss / (q[Q_KM] + tss) * (nadph / (q[Q_KN] + nadph)) / q[Q_SN0]   # thioredoxin reductase
    alt = u[8] * q[Q_KALT] * h
    f[7] = alt * q[Q_G] * (gsh / (q[Q_KG] + gsh)) / q[Q_SG0]
    f[8] = alt * (1. - q[Q_G])
    return f


@njit
def rhs(y, u, q):
    f = fluxes(y, u, q)
    d = np.zeros(NS)
    d[0] = f[0] - f[1] - f[2] - f[7] - f[8]
    d[1] = -f[1] + f[4]
    d[2] = f[1] - f[2] - f[3] + f[5]
    d[3] = f[3] - f[4]
    d[4] = f[2] - f[5]
    d[5] = -f[4] + f[6]
    d[6] = f[0]; d[7] = f[1]; d[8] = f[2]; d[9] = f[7]; d[10] = f[8]
    d[11] = f[3]; d[12] = f[4]; d[13] = f[5]; d[14] = f[6]
    d[15] = f[6]; d[16] = 2. * f[7]; d[17] = q[Q_CATP] * f[5]
    return d


def resting_inputs(p):
    u = np.ones(NU)
    u[2], u[3], u[4] = p.resting_ATP_uM, p.resting_NADPH_uM, p.resting_GSH_uM
    return u


def reference(p):
    p.validate()
    q = np.zeros(NQ)
    q[Q_PT], q[Q_TT], q[Q_KALT], q[Q_KCOND] = p.Prx_total_uM, p.Trx_total_uM, p.kAlt_per_s, p.kCond_per_s
    q[Q_KSULF], q[Q_KSRX], q[Q_VMAX], q[Q_KOX] = p.kSulf_per_uM_s, p.kSrx_per_s, p.TrxR_Vmax_uM_s, p.kOx_per_uM_s
    q[Q_KRED], q[Q_KM], q[Q_V0], q[Q_M] = p.kRed_per_uM_s, p.TrxR_Km_uM, p.supply_uM_s, p.mitochondrial_share
    q[Q_G], q[Q_KG], q[Q_KN], q[Q_KA] = p.GSH_dependent_share_of_alternative_sinks, p.GSH_half_uM, p.NADPH_half_uM, p.ATP_half_uM
    q[Q_CATP] = p.repair_ATP
    q[Q_SG0] = p.resting_GSH_uM / (p.GSH_half_uM + p.resting_GSH_uM)
    q[Q_SN0] = p.resting_NADPH_uM / (p.NADPH_half_uM + p.resting_NADPH_uM)
    q[Q_SA0] = p.resting_ATP_uM / (p.ATP_half_uM + p.resting_ATP_uM)
    return q


def steady_state(q, u, guess=None):
    """Stationary pools for constant inputs; conservation of Prx and Trx is imposed exactly."""
    def unpack(x):
        e = np.exp(x)
        h, psoh, pss, pso2, tss = e
        ps = q[Q_PT] - psoh - pss - pso2
        return np.array([h, ps, psoh, pss, pso2, q[Q_TT] - tss])

    def residual(x):
        y = unpack(x)
        d = rhs(np.r_[y, np.zeros(NS - 6)], u, q)
        scale = np.array([max(d[6], 1e-12), q[Q_KCOND] * y[2] + 1e-12, q[Q_KRED] * y[3] * max(y[5], 1e-9) + 1e-12,
                          max(q[Q_KSRX] * y[4], 1e-15), max(d[14], 1e-12)])
        return np.array([d[0], d[2], d[3], d[4], d[5]]) / scale
    if guess is None:
        sup = fluxes(np.r_[0., q[Q_PT], 0., 0., 0., q[Q_TT], np.zeros(NS - 6)], u, q)[0]
        h = sup / (q[Q_KOX] * q[Q_PT] + q[Q_KALT])
        psoh = sup / q[Q_KCOND]
        pss = sup / (q[Q_KRED] * q[Q_TT])
        pso2 = max(q[Q_KSULF] * psoh * h / max(q[Q_KSRX], 1e-12), 1e-12)
        tss = q[Q_KM] * sup / max(q[Q_VMAX] - sup, 1e-9)
        guess = np.log(np.maximum([h, psoh, pss, pso2, tss], 1e-14))
    def solved(x, flag):
        return flag == 1 and not np.any(unpack(x) < 0) and np.max(np.abs(residual(x))) <= 1e-8
    x, info, flag, message = fsolve(residual, guess, full_output=True, xtol=1e-13, maxfev=4000)
    if solved(x, flag):
        return unpack(x)
    # Fallback: relax the kinetics in time from the fully reduced state, then polish the root.
    start = np.r_[0., q[Q_PT], 0., 0., 0., q[Q_TT], np.zeros(NS - 6)]
    sol = solve_ivp(lambda t, s: rhs(s, u, q), (0., 1e6), start, method='Radau', rtol=1e-10, atol=1e-16)
    if not sol.success:
        raise ArithmeticError('No stationary peroxide state: ' + str(sol.message))
    y = sol.y[:6, -1]
    x, info, flag, message = fsolve(residual, np.log(np.maximum([y[0], y[2], y[3], y[4], q[Q_TT] - y[5]], 1e-300)),
                                    full_output=True, xtol=1e-13, maxfev=4000)
    if solved(x, flag):
        return unpack(x)
    rate = rhs(np.r_[y, np.zeros(NS - 6)], u, q)[:6]
    if np.any(y < 0) or np.max(np.abs(rate) / np.maximum(np.abs(y), 1e-12)) > 1e-9:
        raise ArithmeticError('No stationary peroxide state: ' + str(message))
    return y


def prepare(p):
    q = reference(p)
    u = resting_inputs(p)
    y = np.r_[steady_state(q, u), np.zeros(NS - 6)]
    f = fluxes(y, u, q)
    rate = rhs(y, u, q)
    residual = float(np.max(np.abs(rate[:6]) / np.maximum(np.abs(y[:6]), 1e-12)))
    report = dict(version=VERSION, resting_state={k: float(v) for k, v in zip(POOLS, y[:6])},
                  resting_relative_rate_per_s=residual,
                  H2O2_nM=float(1000. * y[0]),
                  supply_uM_s=float(f[0]),
                  share_by_Prx=float(f[1] / f[0]), share_by_GSH_sinks=float(f[7] / f[0]),
                  share_by_other_sinks=float(f[8] / f[0]), share_by_hyperoxidation=float(f[2] / f[0]),
                  Prx_reduced_fraction=float(y[1] / p.Prx_total_uM),
                  Prx_hyperoxidized_fraction=float(y[4] / p.Prx_total_uM),
                  Trx_reduced_fraction=float(y[5] / p.Trx_total_uM),
                  NADPH_use_uM_s=float(f[6]), GSSG_formation_uM_s=float(f[7]), ATP_use_uM_s=float(rate[17]),
                  TrxR_reserve_fold=float(p.TrxR_Vmax_uM_s / f[6]),
                  disulfide_reduction_capacity_uM_s=float(min(p.TrxR_Vmax_uM_s, p.kRed_per_uM_s * p.Prx_total_uM * p.Trx_total_uM)))
    return y, q, report


def checks(z, base, q):
    z = np.atleast_2d(z)
    d = z - base
    big = lambda i: max(1e-6, float(np.max(np.abs(z[:, i]))))
    err = lambda a, s: float(np.max(np.abs(a)) / s)
    return dict(
        peroxide=err(d[:, 0] - d[:, 6] + d[:, 7] + d[:, 8] + d[:, 9] + d[:, 10], big(6)),
        peroxiredoxin_total=err(d[:, 1:5].sum(1), q[Q_PT]),
        thiolate=err(d[:, 1] + d[:, 7] - d[:, 12], big(7)),
        sulfenate=err(d[:, 2] - d[:, 7] + d[:, 8] + d[:, 11] - d[:, 13], big(7)),
        disulfide=err(d[:, 3] - d[:, 11] + d[:, 12], big(11)),
        sulfinate=err(d[:, 4] - d[:, 8] + d[:, 13], max(1e-9, float(np.max(np.abs(z[:, 8]))))),
        thioredoxin=err(d[:, 5] + d[:, 12] - d[:, 14], big(12)),
        NADPH_ledger=err(d[:, 15] - d[:, 14], big(14)),
        GSH_ledger=err(d[:, 16] - 2. * d[:, 9], big(9)),
        negative_pool=float(max(0., -z[:, :6].min())),
        thioredoxin_bound=float(max(0., (z[:, 5] - q[Q_TT]).max())))


def observe(z, q):
    z = np.atleast_2d(z)
    return dict(H2O2_nM=1000. * z[:, 0], Prx_reduced_fraction=z[:, 1] / q[Q_PT],
                Prx_disulfide_fraction=z[:, 3] / q[Q_PT], Prx_hyperoxidized_fraction=z[:, 4] / q[Q_PT],
                Trx_reduced_fraction=z[:, 5] / q[Q_TT],
                scope='Well-mixed cytosol; model quantities, not calibrated neuronal measurements.')
