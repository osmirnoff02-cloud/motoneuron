"""[claude] 2026-11-14. Peroxide block and lipid block run together (plan N11.2.3).

Still standalone: no host neuron is loaded or changed. The lipid module
(data/2026-11-14-lipid-redox/lipid_redox.py, version 1) is used as it
is; nothing in it is edited. The two placeholders of the lipid block are
replaced by the peroxide block:

  * its "oxidant drive" becomes the peroxide level relative to rest;
  * its constant background of glutathione oxidation and of NADPH use becomes
    the actual flux of the glutathione-dependent peroxide sinks and of
    thioredoxin reductase.

Glutathione, GSSG and NADPH are shared pools (they live in the lipid state).
Glutathione reductase capacity is declared here instead of being solved from a
small resting flux: the resting GSSG level follows from it.

State = 27 lipid states followed by 18 peroxide states.

Run from the project root:
  python -I data/2026-11-14-peroxide/coupled_redox.py
"""
from pathlib import Path
from dataclasses import dataclass, replace
import json
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp

sys.stdout.reconfigure(encoding='utf-8')
B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
sys.path.insert(0, str(B.parent / '2026-11-14-lipid-redox'))
import lipid_redox as L  # noqa: E402
import peroxide as P  # noqa: E402

VERSION = 'coupled-redox-claude-v1-2026-11-14'
HOUR = 3600.
FAIL = .05  # same declared failure event as in the lipid block: damaged share of membrane lipid
NL, NP = L.NS, P.NS
INPUTS = ('Fe_rel', 'respiration_rel', 'supply_multiplier', 'ATP_uM', 'cysteine_supply_rel',
          'm_GPX4', 'm_GSH_synthesis', 'm_RTA_regeneration', 'm_repair', 'm_NADPH_supply', 'm_aldehyde_removal',
          'm_TrxR', 'm_Prx_oxidation', 'm_sulfiredoxin', 'm_alternative_sinks')
I = {k: i for i, k in enumerate(INPUTS)}
# Scenario grids (low, default, high), not measured intervals.
RANGES = {'glutathione_reductase_Vmax_uM_s': [3., 30., 100.],
          'peroxide_dependent_initiation_fraction': [0., 1., 1.],
          'other_NADPH_use_uM_s': [.005, .05, .5]}


@dataclass(frozen=True)
class CouplingParameters:
    glutathione_reductase_Vmax_uM_s: float = 30.
    peroxide_dependent_initiation_fraction: float = 1.
    other_NADPH_use_uM_s: float = .05
    provenance: str = ('Scenario [claude] 2026-11-14: reductase capacity, the share of radical initiation that follows '
                       'peroxide and the remaining NADPH use are declared assumptions without neuronal measurements.')

    def validate(self):
        if (self.glutathione_reductase_Vmax_uM_s <= 0 or not 0. <= self.peroxide_dependent_initiation_fraction <= 1.
                or self.other_NADPH_use_uM_s < 0 or not self.provenance.strip()):
            raise ValueError('Positive reductase capacity, initiation share in [0,1], non-negative NADPH use and provenance required')


def prepare(lp=None, pp=None, cp=None):
    lp = lp or L.LipidRedoxParameters()
    cp = cp or CouplingParameters()
    cp.validate()
    nadph0 = lp.NADPH_fraction * lp.NADP_total_uM
    pp = replace(pp or P.PeroxideParameters(), resting_NADPH_uM=nadph0, resting_GSH_uM=lp.GSH_uM, resting_ATP_uM=lp.resting_ATP_uM)
    yp, q, rep_p = P.prepare(pp)
    j_gsh, j_trx = rep_p['GSSG_formation_uM_s'], rep_p['NADPH_use_uM_s']
    first = replace(lp, background_peroxide_uM_s=j_gsh, background_NADPH_uM_s=cp.other_NADPH_use_uM_s + j_trx)
    _, _, rep0 = L.prepare(first)
    v = rep0['GPX4_turnover_per_s'] * lp.GPX4_uM
    sat_n = nadph0 / (lp.GR_Km_NADPH_uM + nadph0)
    a = (v + j_gsh) / (cp.glutathione_reductase_Vmax_uM_s * sat_n)
    if not 0. < a < 1.:
        raise ArithmeticError('Declared reductase capacity cannot carry the resting flux')
    gssg = lp.GR_Km_GSSG_uM * a / (1. - a)
    second = replace(first, GSSG_fraction=gssg / lp.GSH_uM)
    yl, ref, rep_l = L.prepare(second)
    run_ref = ref.copy()
    run_ref[L.R_JPER] = 0.
    run_ref[L.R_JBGN] = cp.other_NADPH_use_uM_s
    y = np.r_[yl, yp]
    u = resting_inputs(lp)
    rate = rhs(y, u, run_ref, q, yp[0], cp.peroxide_dependent_initiation_fraction)
    residual = float(np.max(np.abs(np.r_[rate[:11], rate[NL:NL + 6]]) / np.maximum(np.abs(np.r_[y[:11], y[NL:NL + 6]]), 1e-12)))
    if residual > 1e-8:
        raise ArithmeticError('Coupled resting state is not stationary: ' + str(residual))
    nadph_total = float(rate[23] + rate[NL + 15] * 0.)  # lipid ledger already contains the reductase part added in rhs
    report = dict(version=VERSION, lipid=rep_l, peroxide=rep_p, resting_relative_rate_per_s=residual,
                  resting_GSSG_uM=float(gssg), resting_GSH_over_GSSG=float(lp.GSH_uM / gssg),
                  glutathione_reductase_Vmax_uM_s=float(ref[L.R_VGR]),
                  resting_NADPH_use_uM_s=nadph_total,
                  NADPH_share_thioredoxin_reductase=float(j_trx / nadph_total),
                  NADPH_supply_Vmax_uM_s=float(ref[L.R_VPPP]),
                  NADPH_supply_reserve_fold=float(ref[L.R_VPPP] / nadph_total),
                  resting_ATP_use_uM_s=float(rate[26] + rate[NL + 17]))
    return y, (run_ref, q, float(yp[0]), cp.peroxide_dependent_initiation_fraction), report, (second, pp, cp)


def resting_inputs(lp):
    u = np.ones(len(INPUTS))
    u[I['ATP_uM']] = lp.resting_ATP_uM
    return u


def rhs(y, u, ref, q, h0, share):
    yl, yp = y[:NL], y[NL:]
    up = np.array([u[1], u[2], u[3], min(max(yl[9], 0.), ref[L.R_NTOT]), max(yl[7], 0.), u[11], u[12], u[13], u[14]])
    fp = P.fluxes(yp, up, q)
    dp = P.rhs(yp, up, q)
    drive = 1. + share * (max(yp[0], 0.) / h0 - 1.)
    ul = np.array([u[0], drive, u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10]])
    dl = L.rhs(yl, ul, ref)
    dl[7] -= 2. * fp[7]      # glutathione-dependent peroxide sinks
    dl[8] += fp[7]
    dl[9] -= fp[6]           # thioredoxin reductase
    dl[23] += fp[6]
    return np.r_[dl, dp]


def checks(z, base, ref, q):
    a = L.checks(z[:, :NL], base[:NL], ref)
    b = P.checks(z[:, NL:], base[NL:], q)
    out = {'lipid_' + k: v for k, v in a.items()}
    out.update({'peroxide_' + k: v for k, v in b.items()})
    return out


def integrate(y0, model, segments, points=240):
    ref, q, h0, share = model
    ts, zs, y = [np.array([segments[0][0]])], [y0[None, :]], y0.copy()
    for a, b, u in segments:
        grid = np.linspace(a, b, points + 1)[1:]
        fun = lambda t, s: rhs(s, u, ref, q, h0, share)
        sol = solve_ivp(fun, (a, b), y, method='LSODA', rtol=1e-9, atol=1e-13, t_eval=grid)
        if not sol.success:
            sol = solve_ivp(fun, (a, b), y, method='Radau', rtol=1e-8, atol=1e-12, t_eval=grid)
        if not sol.success:
            raise RuntimeError(sol.message)
        ts.append(sol.t); zs.append(sol.y.T); y = sol.y[:, -1].copy()
    return np.concatenate(ts), np.vstack(zs)


def scenario(changes, lp=None, pp=None, cp=None, hours=48.):
    y0, model, report, (lp2, pp2, cp2) = prepare(lp, pp, cp)
    u0 = resting_inputs(lp2)
    u1 = u0.copy()
    for k, v in changes.items():
        u1[I[k]] = v if k == 'ATP_uM' else u0[I[k]] * v
    t, z = integrate(y0, model, [(0., HOUR, u0), (HOUR, HOUR + hours * HOUR, u1)])
    ref, q, h0, share = model
    lo, po = L.observe(z[:, :NL], ref), P.observe(z[:, NL:], q)
    ch = checks(z, y0, ref, q)
    damaged = lo['damaged_lipid_fraction']
    k = int(np.argmax(damaged >= FAIL)) if np.any(damaged >= FAIL) else -1
    rate = rhs(z[-1], u1, ref, q, h0, share)
    return dict(
        H2O2_fold_final=float(z[-1, NL] / y0[NL]), H2O2_nM_final=float(po['H2O2_nM'][-1]), H2O2_nM_peak=float(po['H2O2_nM'].max()),
        LOOH_fold_final=float(z[-1, 2] / y0[2]), damaged_peak=float(damaged.max()), damaged_final=float(damaged[-1]),
        failure=bool(k >= 0), hours_to_failure=float((t[k] - HOUR) / HOUR) if k >= 0 else None,
        GSH_final_uM=float(z[-1, 7]), GSH_over_GSSG_final=float(lo['GSH_over_GSSG'][-1]),
        NADPH_fraction_min=float(lo['NADPH_fraction'].min()), RTA_reduced_min=float(lo['RTA_reduced_fraction'].min()),
        Prx_reduced_min=float(po['Prx_reduced_fraction'].min()), Prx_hyperoxidized_max=float(po['Prx_hyperoxidized_fraction'].max()),
        Trx_reduced_min=float(po['Trx_reduced_fraction'].min()),
        NADPH_use_final_uM_s=float(rate[23]), ATP_use_final_uM_s=float(rate[26] + rate[NL + 17]),
        worst_check=float(max(ch.values())), worst_check_name=max(ch, key=ch.get), checks=ch), report


def threshold(key, lo, hi, fixed=None, increasing=True, **kw):
    def bad(x):
        ch = dict(fixed or {}); ch[key] = x
        return scenario(ch, **kw)[0]['failure']
    if increasing:
        if not bad(hi):
            return None
        if bad(lo):
            return lo
    else:
        if not bad(lo):
            return None
        if bad(hi):
            return hi
    for _ in range(22):
        mid = np.sqrt(lo * hi)
        if bad(mid) == increasing:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


def main():
    started = time.monotonic()
    y0, model, report, (lp, pp, cp) = prepare()
    ref, q, h0, share = model
    out = dict(author='Claude (Fable 5.2)', date='2026-11-14', version=VERSION,
               scope='Two standalone blocks coupled through glutathione and NADPH; scenario parameters; not a motor-neuron prediction',
               coupling_parameters=cp.__dict__, resting_report={k: v for k, v in report.items() if k not in ('lipid', 'peroxide')},
               resting_lipid=report['lipid'], resting_peroxide=report['peroxide'])
    print('resting:', {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out['resting_report'].items()})
    u0 = resting_inputs(lp)
    t, z = integrate(y0, model, [(0., 72 * HOUR, u0)])
    pools = np.r_[np.arange(11), NL + np.arange(6)]
    drift = float(np.max(np.abs(z[:, pools] - y0[pools]) / np.maximum(np.abs(y0[pools]), 1e-15)))
    ch = checks(z, y0, ref, q)
    out['rest_72h'] = dict(max_relative_pool_drift=drift, worst_check=float(max(ch.values())), checks=ch)
    print(f'rest 72 h: drift {drift:.2e}, worst check {max(ch.values()):.2e}')
    scenarios = {
        'supply_x3': {'supply_multiplier': 3.}, 'supply_x10': {'supply_multiplier': 10.},
        'supply_x30': {'supply_multiplier': 30.}, 'supply_x100': {'supply_multiplier': 100.},
        'respiration_x2': {'respiration_rel': 2.},
        'Fe_x5': {'Fe_rel': 5.}, 'Fe_x10': {'Fe_rel': 10.}, 'Fe_x5_supply_x3': {'Fe_rel': 5., 'supply_multiplier': 3.},
        'GPX4_x0.1': {'m_GPX4': .1}, 'GPX4_x0.1_supply_x10': {'m_GPX4': .1, 'supply_multiplier': 10.},
        'GPX4_off': {'m_GPX4': 0.},
        'TrxR_x0.01': {'m_TrxR': .01}, 'TrxR_x0.01_supply_x10': {'m_TrxR': .01, 'supply_multiplier': 10.},
        'NADPH_supply_x0.3': {'m_NADPH_supply': .3}, 'NADPH_supply_x0.1': {'m_NADPH_supply': .1},
        'NADPH_supply_x0.3_supply_x3': {'m_NADPH_supply': .3, 'supply_multiplier': 3.},
        'RTA_regeneration_off_supply_x3': {'m_RTA_regeneration': 0., 'supply_multiplier': 3.},
        'Prx_activity_x0.1': {'m_Prx_oxidation': .1},
        'GSH_synthesis_off': {'m_GSH_synthesis': 0.},
    }
    out['scenarios'] = {}
    print(f"{'scenario':<30} H2O2x    LOOHx    dmg_pk  fail_h GSH   GSH/GSSG NADPH  RTA   PrxS  PrxSO2 NADPH/s  check")
    for name, changes in scenarios.items():
        s, _ = scenario(changes)
        out['scenarios'][name] = dict(changes=changes, **s)
        fh = '-' if s['hours_to_failure'] is None else f"{s['hours_to_failure']:.1f}"
        print(f"{name:<30} {s['H2O2_fold_final']:8.2f} {s['LOOH_fold_final']:8.2f} {s['damaged_peak']:7.1e} {fh:>6} {s['GSH_final_uM']:5.0f} "
              f"{s['GSH_over_GSSG_final']:8.0f} {s['NADPH_fraction_min']:5.2f} {s['RTA_reduced_min']:5.2f} {s['Prx_reduced_min']:5.2f} "
              f"{s['Prx_hyperoxidized_max']:6.3f} {s['NADPH_use_final_uM_s']:7.3f}  {s['worst_check']:.1e}"
              + (' ' + s['worst_check_name'] if s['worst_check'] > 1e-7 else ''))
    th = dict(
        supply_multiple_for_failure=threshold('supply_multiplier', 1., 3000.),
        Fe_multiple_for_failure=threshold('Fe_rel', 1., 300.),
        Fe_multiple_for_failure_at_supply_x3=threshold('Fe_rel', 1., 300., fixed={'supply_multiplier': 3.}),
        supply_multiple_for_failure_at_tenth_GPX4=threshold('supply_multiplier', 1., 3000., fixed={'m_GPX4': .1}),
        supply_multiple_for_failure_at_third_NADPH_supply=threshold('supply_multiplier', 1., 3000., fixed={'m_NADPH_supply': .3}),
        supply_multiple_for_failure_without_peroxide_driven_initiation=threshold(
            'supply_multiplier', 1., 3000., cp=replace(cp, peroxide_dependent_initiation_fraction=0.)))
    out['thresholds'] = th
    print('thresholds', th)
    sc = out['scenarios']
    worst = max(max(v['worst_check'] for v in sc.values()), out['rest_72h']['worst_check'])
    out['acceptance'] = dict(
        resting_state_stationary=bool(drift < 1e-7),
        ledgers_close=bool(worst < 1e-7),
        peroxide_raises_hydroperoxide=bool(sc['supply_x10']['LOOH_fold_final'] > sc['supply_x3']['LOOH_fold_final'] > 1.),
        reductase_capacity_as_declared=bool(abs(report['glutathione_reductase_Vmax_uM_s'] / cp.glutathione_reductase_Vmax_uM_s - 1.) < 1e-6),
        no_negative_pool=bool(all(v['checks']['lipid_negative_pool'] < 1e-9 and v['checks']['peroxide_negative_pool'] < 1e-9 for v in sc.values())))
    out['acceptance']['all_pass'] = bool(all(out['acceptance'].values()))
    print('acceptance', out['acceptance'])
    out['elapsed_s'] = time.monotonic() - started
    (B / 'coupled_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + '\n', encoding='utf8')
    print('elapsed', round(out['elapsed_s'], 1))


if __name__ == '__main__':
    main()
