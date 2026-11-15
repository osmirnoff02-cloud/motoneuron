"""[claude] 2026-11-14. Standalone lipid-peroxidation / glutathione / aldehyde block.

Author: Claude (Fable 5.2). Plan row N11.2.3. Version 1 is NOT connected to
the host neuron: it is a self-contained kinetic block with closed material
ledgers, meant to be tested alone and connected later as an opt-in module.
The existing effective block N11.2.2 (relative "lipid oxidation") is untouched.

One compartment, concentrations in uM of a cytosolic reference volume, time in
seconds. Membrane species are expressed as cytosol-equivalent uM; radical
chemistry uses mole fractions of the membrane lipid.

Pools
  LH   oxidizable (PUFA) phospholipid        LS   non-oxidizable phospholipid
  LOOH phospholipid hydroperoxide            LOH  reduced hydroxy-lipid
  LT   truncated lipid after fragmentation   HNE  free reactive aldehyde
  PA   protein-aldehyde adducts              GSH, GSSG, NADPH
  RTA  reduced radical-trapping antioxidant (ubiquinol / tocopherol equivalent)

Reactions
  R1 radicals:   r_rad  = k_init*ox*((1-a)+a*Fe_rel) + phi*r_dec, both scaled by LH availability
  R2 chain:      peroxyl radicals at quasi-steady state against trapping by RTA
                 (n radicals per RTA) and bimolecular termination;
                 r_form = r_rad + Ltot*kp*X_LH*x           (LH -> LOOH)
  R3 GPX4:       LOOH + 2 GSH -> LOH + GSSG, ping-pong
                 v = E/(1/(k1*LOOH) + 1/(k2*GSH))
  R4 iron:       r_dec = k_Fe*Fe_rel*LOOH -> LT + y*HNE + phi radicals
  R5 repair:     LOH, LT -> LH (fraction f_PUFA) or LS; 2 ATP-equivalents each
  R6 aldehyde:   HNE + GSH -> exported conjugate; enzymatic removal; adducts
  R7 adducts:    PA cleared first-order; ATP per cleared adducted protein
  R8 glutathione: GSSG + NADPH -> 2 GSH; synthesis 2 ATP per GSH with product
                 inhibition; first-order loss
  R9 NADPH:      regeneration by NADP+-dependent supply; consumed by R8, RTA
                 regeneration (one NADPH per RTA), part of R6 and a background
  R10 background: peroxide reduction 2 GSH -> GSSG scaling with the oxidant
                 drive, and other NADPH use; they set the reserve of R8/R9

The resting state is closed by construction: four supply capacities
(glutathione reductase, glutathione synthesis, RTA regeneration, NADPH
supply) are solved so that the declared resting pools are stationary.

Declared limits: no lysosomal or mitochondrial sub-pools, no diffusion
between compartments, no lipoxygenase route, no selenium/GPX4 turnover, no
cystine transport kinetics, no cell death. All defaults are scenario values;
only kp, 2kt, k1, k2 and the glutathione range come from cited measurements in
other systems. Not a human motor-neuron calibration.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit
from scipy.optimize import brentq

VERSION = 'lipid-redox-claude-v1-2026-11-14'
POOLS = ('LH', 'LS', 'LOOH', 'LOH', 'LT', 'HNE', 'PA', 'GSH', 'GSSG', 'NADPH', 'RTA')
LEDGERS = ('LOOH_formed', 'LOOH_reduced', 'LOOH_decomposed', 'lipid_repaired', 'HNE_formed',
           'HNE_conjugated', 'HNE_enzymatic', 'adducts_formed', 'adducts_cleared',
           'GSH_synthesized', 'GSH_lost', 'NADPH_regenerated', 'NADPH_consumed',
           'RTA_consumed', 'RTA_regenerated', 'ATP_consumed')
NAMES = POOLS + LEDGERS
NS = len(NAMES)
# Run-time inputs: host variables first, then intervention multipliers.
INPUTS = ('Fe_rel', 'oxidant_rel', 'ATP_uM', 'cysteine_supply_rel',
          'm_GPX4', 'm_GSH_synthesis', 'm_RTA_regeneration', 'm_repair', 'm_NADPH_supply', 'm_aldehyde_removal')
NU = len(INPUTS)

# Scenario grids (low, default, high), NOT measured human confidence intervals.
RANGES = {
    'lipid_total_uM': [15000., 30000., 60000.],
    'oxidizable_fraction': [.15, .3, .45],
    'kp_per_mole_fraction_s': [35., 115., 172.],
    'termination_2kt_M_s': [1.02e5, 1.27e5, 1.27e5],
    'membrane_lipid_M': [1.3, 2.1, 2.1],
    'kinh_over_kp': [100., 300., 3000.],
    'RTA_per_lipid': [2e-4, 5e-4, 2e-3],
    'RTA_reduced_fraction': [.5, .9, .98],
    'radicals_per_RTA': [2., 2., 2.],
    'initiation_uM_s': [1e-4, 1e-3, 1e-2],
    'iron_dependent_initiation_fraction': [0., .5, 1.],
    'iron_decomposition_per_s': [1e-4, 1e-3, 1e-2],
    'background_peroxide_uM_s': [.03, .3, 3.],
    'background_peroxide_GSH_half_uM': [10., 50., 200.],
    'background_NADPH_uM_s': [.005, .05, .5],
    'radicals_per_decomposition': [.5, 1., 1.],
    'HNE_per_decomposition': [.02, .1, .3],
    'GPX4_uM': [.05, .2, 1.],
    'GPX4_k1_per_uM_s': [5.28, 5.28, 5.28],
    'GPX4_k2_per_uM_s': [2.86e-3, 2.86e-3, 2.86e-3],
    'GSH_uM': [200., 2000., 5000.],
    'GSSG_fraction': [2e-4, 2e-3, 1e-2],
    'GSH_loss_tau_s': [3600., 21600., 172800.],
    'GSH_inhibition_uM': [1000., 2300., 5000.],
    'GR_Km_GSSG_uM': [20., 60., 100.],
    'GR_Km_NADPH_uM': [3., 9., 20.],
    'NADP_total_uM': [20., 50., 100.],
    'NADPH_fraction': [.9, .98, .995],
    'NADP_supply_Km_uM': [3., 10., 30.],
    'RTA_regeneration_Km_NADPH_uM': [3., 10., 30.],
    'repair_tau_s': [600., 3600., 21600.],
    'acyl_turnover_tau_s': [43200., 172800., 604800.],
    'repair_ATP': [2., 2., 2.],
    'ATP_half_uM': [100., 500., 1500.],
    'HNE_GSH_per_uM_s': [1.2e-6, 1e-5, 1e-4],
    'HNE_enzymatic_Vmax_uM_s': [.005, .05, .5],
    'HNE_enzymatic_Km_uM': [5., 20., 50.],
    'HNE_NADPH_fraction': [0., .3, .6],
    'HNE_adduction_per_s': [1e-4, 5e-4, 2e-3],
    'adduct_clearance_tau_s': [21600., 86400., 259200.],
    'adduct_clearance_ATP': [0., 300., 1000.],
    'conjugate_export_ATP': [0., 2., 2.],
}


@dataclass(frozen=True)
class LipidRedoxParameters:
    lipid_total_uM: float = 30000.
    oxidizable_fraction: float = .3
    kp_per_mole_fraction_s: float = 115.
    termination_2kt_M_s: float = 1.27e5
    membrane_lipid_M: float = 2.1
    kinh_over_kp: float = 300.
    RTA_per_lipid: float = 5e-4
    RTA_reduced_fraction: float = .9
    radicals_per_RTA: float = 2.
    initiation_uM_s: float = 1e-3
    iron_dependent_initiation_fraction: float = .5
    iron_decomposition_per_s: float = 1e-3
    background_peroxide_uM_s: float = .3
    background_peroxide_GSH_half_uM: float = 50.
    background_NADPH_uM_s: float = .05
    radicals_per_decomposition: float = 1.
    HNE_per_decomposition: float = .1
    GPX4_uM: float = .2
    GPX4_k1_per_uM_s: float = 5.28
    GPX4_k2_per_uM_s: float = 2.86e-3
    GSH_uM: float = 2000.
    GSSG_fraction: float = .002
    GSH_loss_tau_s: float = 21600.
    GSH_inhibition_uM: float = 2300.
    GR_Km_GSSG_uM: float = 60.
    GR_Km_NADPH_uM: float = 9.
    NADP_total_uM: float = 50.
    NADPH_fraction: float = .98
    NADP_supply_Km_uM: float = 10.
    RTA_regeneration_Km_NADPH_uM: float = 10.
    repair_tau_s: float = 3600.
    acyl_turnover_tau_s: float = 172800.
    repair_ATP: float = 2.
    ATP_half_uM: float = 500.
    HNE_GSH_per_uM_s: float = 1e-5
    HNE_enzymatic_Vmax_uM_s: float = .05
    HNE_enzymatic_Km_uM: float = 20.
    HNE_NADPH_fraction: float = .3
    HNE_adduction_per_s: float = 5e-4
    adduct_clearance_tau_s: float = 86400.
    adduct_clearance_ATP: float = 300.
    conjugate_export_ATP: float = 2.
    resting_ATP_uM: float = 2900.
    provenance: str = ('Scenario [claude] 2026-11-14. Measured in other systems: kp and 2kt in phosphatidylcholine '
                       'liposomes at 37 C (Xu 2009; Valgimigli 2023), GPX4 k1/k2 for recombinant human enzyme with '
                       'PCOOH (Cheng 2021), glutathione range of cultured rodent neurons (Raps 1989; Pileblad 1991). '
                       'All other numbers are declared scenarios. No human motor-neuron calibration.')

    def validate(self):
        d = asdict(self)
        values = [v for k, v in d.items() if k != 'provenance']
        if not np.isfinite(values).all() or min(values) < 0 or not self.provenance.strip():
            raise ValueError('Finite non-negative parameters and provenance required')
        for k in ('oxidizable_fraction', 'RTA_reduced_fraction', 'NADPH_fraction', 'GSSG_fraction'):
            if not 0. < d[k] < 1.:
                raise ValueError(k + ' must lie inside (0,1)')
        for k in ('HNE_NADPH_fraction', 'iron_dependent_initiation_fraction'):
            if not 0. <= d[k] <= 1.:
                raise ValueError(k + ' must lie inside [0,1]')
        for k in ('lipid_total_uM', 'kp_per_mole_fraction_s', 'termination_2kt_M_s', 'membrane_lipid_M', 'kinh_over_kp',
                  'RTA_per_lipid', 'radicals_per_RTA', 'GPX4_uM', 'GPX4_k1_per_uM_s', 'GPX4_k2_per_uM_s', 'GSH_uM',
                  'GSH_loss_tau_s', 'NADP_total_uM', 'repair_tau_s', 'acyl_turnover_tau_s', 'adduct_clearance_tau_s',
                  'resting_ATP_uM', 'initiation_uM_s'):
            if d[k] <= 0:
                raise ValueError(k + ' must be positive')


# Reference vector layout used by the compiled kernel.
(R_LTOT, R_F, R_KP, R_2KT, R_KINH, R_N, R_KINIT, R_KFE, R_PHI, R_YHNE, R_EG, R_K1, R_K2, R_KLOSS, R_KI,
 R_KMG, R_KMN, R_NTOT, R_KMP, R_KMR, R_KREM, R_KTURN, R_CREM, R_KATP, R_KGST, R_VENZ, R_KMENZ, R_FAKR, R_KADD,
 R_KCLR, R_CCLR, R_CEXP, R_RTATOT, R_VGR, R_VSYN, R_KREG, R_VPPP,
 R_AFE, R_JPER, R_KGB, R_JBGN, R_SN0, R_SG0, R_KLH, R_SL0) = range(45)
NR = 45
NF = 18


@njit
def radical_fraction(r_rad_uM_s, rta_uM, ref):
    """Quasi-steady peroxyl radical mole fraction; stable for vanishing termination terms."""
    rx = max(r_rad_uM_s, 0.) / ref[R_LTOT]
    b = ref[R_N] * ref[R_KINH] * max(rta_uM, 0.) / ref[R_LTOT]
    a = ref[R_2KT]
    root = np.sqrt(b * b + 4. * a * rx)
    return 2. * rx / (b + root) if b + root > 0 else 0.


@njit
def fluxes(y, u, ref):
    """All reaction rates (uM/s) of one compartment."""
    lh, loo, loh, lt = max(y[0], 0.), max(y[2], 0.), max(y[3], 0.), max(y[4], 0.)
    hne, pa, gsh, gssg = max(y[5], 0.), max(y[6], 0.), max(y[7], 0.), max(y[8], 0.)
    nadph = min(max(y[9], 0.), ref[R_NTOT])
    rta = min(max(y[10], 0.), ref[R_RTATOT])
    fe, ox, atp, cys = u[0], u[1], max(u[2], 0.), u[3]
    f = np.zeros(NF)
    f[0] = ref[R_KFE] * fe * loo                                   # iron-dependent LOOH decomposition
    # Radicals need an oxidizable lipid to start a chain; normalized to one at rest.
    avail = lh / (lh + ref[R_KLH]) / ref[R_SL0]
    r_rad = (ref[R_KINIT] * ox * ((1. - ref[R_AFE]) + ref[R_AFE] * fe) + ref[R_PHI] * f[0]) * avail
    x = radical_fraction(r_rad, rta, ref)
    f[1] = r_rad + ref[R_KP] * lh * x                              # LOOH formation (LH consumed)
    f[2] = ref[R_KINH] * rta * x                                   # RTA consumed
    k1l = ref[R_K1] * loo
    k2g = ref[R_K2] * gsh
    f[3] = u[4] * ref[R_EG] * k1l * k2g / (k1l + k2g) if k1l + k2g > 0 else 0.   # GPX4
    energy = atp / (ref[R_KATP] + atp)
    krem = u[7] * ref[R_KREM] * energy
    f[4] = krem * loh                                              # repair of LOH
    f[5] = krem * lt                                               # repair of LT
    sn = nadph / (ref[R_KMR] + nadph) / ref[R_SN0]                 # NADPH availability, one at rest
    f[6] = u[9] * ref[R_KGST] * hne * gsh                          # HNE-glutathione conjugation
    enz = u[9] * ref[R_VENZ] * hne / (ref[R_KMENZ] + hne)
    f[16] = enz * ref[R_FAKR] * sn                                 # NADPH-dependent part of enzymatic HNE removal
    f[7] = enz * (1. - ref[R_FAKR]) + f[16]                        # enzymatic HNE removal, total
    f[8] = ref[R_KADD] * hne                                       # protein adduction
    f[9] = ref[R_KCLR] * pa                                        # adduct clearance
    f[10] = ref[R_VGR] * gssg / (ref[R_KMG] + gssg) * nadph / (ref[R_KMN] + nadph)   # glutathione reductase
    f[11] = u[5] * ref[R_VSYN] * cys * energy / (1. + gsh / ref[R_KI])              # GSH synthesis
    f[12] = ref[R_KLOSS] * gsh                                     # GSH loss
    f[13] = u[6] * ref[R_KREG] * (ref[R_RTATOT] - rta) * nadph / (ref[R_KMR] + nadph)   # RTA regeneration
    nadp = ref[R_NTOT] - nadph
    f[14] = u[8] * ref[R_VPPP] * nadp / (ref[R_KMP] + nadp)        # NADPH supply
    f[15] = ref[R_KTURN] * (ref[R_F] * max(y[1], 0.) - (1. - ref[R_F]) * lh)       # acyl turnover LS -> LH
    # Background: other peroxide reduction by glutathione, and other NADPH use.
    f[17] = ref[R_JPER] * ox * gsh / (ref[R_KGB] + gsh) / ref[R_SG0]
    return f


@njit
def rhs(y, u, ref):
    f = fluxes(y, u, ref)
    d = np.zeros(NS)
    rep = f[4] + f[5]
    d[0] = -f[1] + ref[R_F] * rep + f[15]
    d[1] = (1. - ref[R_F]) * rep - f[15]
    d[2] = f[1] - f[3] - f[0]
    d[3] = f[3] - f[4]
    d[4] = f[0] - f[5]
    hne_formed = ref[R_YHNE] * f[0]
    d[5] = hne_formed - f[6] - f[7] - f[8]
    d[6] = f[8] - f[9]
    background_nadph = ref[R_JBGN] * (min(max(y[9], 0.), ref[R_NTOT]) / (ref[R_KMR] + min(max(y[9], 0.), ref[R_NTOT]))) / ref[R_SN0]
    d[7] = -2. * f[3] - 2. * f[17] + 2. * f[10] - f[6] + f[11] - f[12]
    d[8] = f[3] + f[17] - f[10]
    nadph_used = f[10] + f[13] + f[16] + background_nadph
    d[9] = f[14] - nadph_used
    d[10] = -f[2] + f[13]
    d[11] = f[1]; d[12] = f[3]; d[13] = f[0]; d[14] = rep; d[15] = hne_formed
    d[16] = f[6]; d[17] = f[7]; d[18] = f[8]; d[19] = f[9]; d[20] = f[11]; d[21] = f[12] + f[6]
    d[22] = f[14]; d[23] = nadph_used; d[24] = f[2]; d[25] = f[13]
    d[26] = 2. * f[11] + ref[R_CREM] * rep + ref[R_CEXP] * f[6] + ref[R_CCLR] * f[9]
    return d


def resting_inputs(p):
    u = np.ones(NU)
    u[2] = p.resting_ATP_uM
    return u


def prepare(p):
    """Return (initial state, reference vector, report) with a stationary resting state."""
    p.validate()
    ref = np.zeros(NR)
    ref[R_LTOT] = p.lipid_total_uM
    ref[R_F] = p.oxidizable_fraction
    # Mole-fraction units: Ltot*kp*X_LH*x = kp*LH_uM*x and Ltot*kinh*X_RTA*x = kinh*RTA_uM*x,
    # where x is the peroxyl-radical mole fraction returned by radical_fraction().
    ref[R_KP] = p.kp_per_mole_fraction_s
    ref[R_2KT] = p.termination_2kt_M_s * p.membrane_lipid_M
    ref[R_KINH] = p.kinh_over_kp * p.kp_per_mole_fraction_s
    ref[R_N] = p.radicals_per_RTA
    ref[R_KINIT] = p.initiation_uM_s
    ref[R_KFE] = p.iron_decomposition_per_s
    ref[R_PHI] = p.radicals_per_decomposition
    ref[R_YHNE] = p.HNE_per_decomposition
    ref[R_EG], ref[R_K1], ref[R_K2] = p.GPX4_uM, p.GPX4_k1_per_uM_s, p.GPX4_k2_per_uM_s
    ref[R_KLOSS] = 1. / p.GSH_loss_tau_s
    ref[R_KI] = p.GSH_inhibition_uM
    ref[R_KMG], ref[R_KMN] = p.GR_Km_GSSG_uM, p.GR_Km_NADPH_uM
    ref[R_NTOT] = p.NADP_total_uM
    ref[R_KMP], ref[R_KMR] = p.NADP_supply_Km_uM, p.RTA_regeneration_Km_NADPH_uM
    ref[R_KREM] = 1. / p.repair_tau_s
    ref[R_KTURN] = 1. / p.acyl_turnover_tau_s
    ref[R_CREM], ref[R_KATP] = p.repair_ATP, p.ATP_half_uM
    ref[R_KGST] = p.HNE_GSH_per_uM_s
    ref[R_VENZ], ref[R_KMENZ], ref[R_FAKR] = p.HNE_enzymatic_Vmax_uM_s, p.HNE_enzymatic_Km_uM, p.HNE_NADPH_fraction
    ref[R_KADD] = p.HNE_adduction_per_s
    ref[R_KCLR] = 1. / p.adduct_clearance_tau_s
    ref[R_CCLR], ref[R_CEXP] = p.adduct_clearance_ATP, p.conjugate_export_ATP
    ref[R_RTATOT] = p.RTA_per_lipid * p.lipid_total_uM
    ref[R_AFE] = p.iron_dependent_initiation_fraction
    ref[R_JPER], ref[R_KGB] = p.background_peroxide_uM_s, p.background_peroxide_GSH_half_uM
    ref[R_JBGN] = p.background_NADPH_uM_s
    ref[R_KLH] = .01 * p.lipid_total_uM
    u = resting_inputs(p)
    gsh = p.GSH_uM
    gssg = p.GSSG_fraction * gsh
    nadph = p.NADPH_fraction * p.NADP_total_uM
    ref[R_SN0] = nadph / (ref[R_KMR] + nadph)
    ref[R_SG0] = gsh / (ref[R_KGB] + gsh)
    ref[R_SL0] = 1.  # replaced below once the resting LH is known
    rta = p.RTA_reduced_fraction * ref[R_RTATOT]
    energy = u[2] / (ref[R_KATP] + u[2])
    krem = ref[R_KREM] * energy

    def lipid_state(loo):
        """Stationary lipid pools for a given LOOH; LH solved from conservation."""
        dec = ref[R_KFE] * loo
        k1l, k2g = ref[R_K1] * loo, ref[R_K2] * gsh
        v = ref[R_EG] * k1l * k2g / (k1l + k2g) if loo > 0 else 0.
        loh, lt = v / krem, dec / krem
        rest = p.lipid_total_uM - loo - loh - lt
        # Stationary LH: repair returns the fraction f as LH, acyl turnover supplies the remainder.
        lh = p.oxidizable_fraction * rest - (1. - p.oxidizable_fraction) * (v + dec) / ref[R_KTURN]
        return dec, v, loh, lt, lh

    def net(loo):
        dec, v, loh, lt, lh = lipid_state(loo)
        r_rad = ref[R_KINIT] + ref[R_PHI] * dec
        x = radical_fraction(r_rad, rta, ref)
        return r_rad + ref[R_KP] * max(lh, 0.) * x - v - dec

    grid = np.logspace(-9, np.log10(.2 * p.lipid_total_uM), 600)
    values = np.array([net(g) for g in grid])
    cross = np.where((values[:-1] > 0) & (values[1:] <= 0))[0]
    if not len(cross):
        raise ArithmeticError('No stationary hydroperoxide level: resting defence cannot balance formation')
    loo = brentq(net, grid[cross[0]], grid[cross[0] + 1], xtol=1e-16, rtol=1e-14)
    dec, v, loh, lt, lh = lipid_state(loo)
    if lh <= 0:
        raise ArithmeticError('Resting oxidizable lipid would be non-positive')
    ref[R_SL0] = lh / (lh + ref[R_KLH])
    ls = p.lipid_total_uM - loo - loh - lt - lh
    formed_hne = ref[R_YHNE] * dec

    def hne_net(h):
        return formed_hne - ref[R_KGST] * h * gsh - ref[R_VENZ] * h / (ref[R_KMENZ] + h) - ref[R_KADD] * h
    hne = brentq(hne_net, 0., max(1., formed_hne / max(ref[R_KADD], 1e-30)) * 10. + 1., xtol=1e-18, rtol=1e-14) if formed_hne > 0 else 0.
    gst, enz, add = ref[R_KGST] * hne * gsh, ref[R_VENZ] * hne / (ref[R_KMENZ] + hne), ref[R_KADD] * hne
    pa = add / ref[R_KCLR]
    r_rad = ref[R_KINIT] + ref[R_PHI] * dec
    x = radical_fraction(r_rad, rta, ref)
    trap = ref[R_KINH] * rta * x
    reductase = v + ref[R_JPER]                                # stationary GSSG: reductase flux = all GSSG formed
    ref[R_VGR] = reductase / (gssg / (ref[R_KMG] + gssg) * nadph / (ref[R_KMN] + nadph))
    ref[R_VSYN] = (ref[R_KLOSS] * gsh + gst) * (1. + gsh / ref[R_KI]) / energy
    ref[R_KREG] = trap / ((ref[R_RTATOT] - rta) * nadph / (ref[R_KMR] + nadph))
    nadp = p.NADP_total_uM - nadph
    ref[R_VPPP] = (reductase + trap + ref[R_FAKR] * enz + ref[R_JBGN]) / (nadp / (ref[R_KMP] + nadp))
    y = np.zeros(NS)
    y[:11] = [lh, ls, loo, loh, lt, hne, pa, gsh, gssg, nadph, rta]
    rate = rhs(y, u, ref)
    scale = np.maximum(np.abs(y[:11]), 1e-9)
    residual = float(np.max(np.abs(rate[:11]) / scale))
    f = fluxes(y, u, ref)
    report = dict(version=VERSION, resting_state={k: float(a) for k, a in zip(POOLS, y[:11])},
                  resting_relative_rate_per_s=residual,
                  chain_length=float(ref[R_KP] * lh * x / r_rad),
                  radical_mole_fraction=float(x),
                  LOOH_per_lipid=float(loo / p.lipid_total_uM),
                  LOOH_formation_uM_per_day=float(f[1] * 86400.),
                  GPX4_turnover_per_s=float(v / p.GPX4_uM),
                  GPX4_capacity_uM_s=float(p.GPX4_uM * ref[R_K2] * gsh),
                  GPX4_share_of_reductase_flux=float(v / reductase),
                  reductase_reserve_fold=float(ref[R_VGR] / reductase),
                  NADPH_supply_reserve_fold=float(ref[R_VPPP] / rate[23]) if rate[23] > 0 else None,
                  resting_ATP_cost_uM_s=float(rate[26]),
                  resting_NADPH_cost_uM_s=float(rate[23]),
                  solved_capacities=dict(glutathione_reductase_Vmax_uM_s=float(ref[R_VGR]),
                                         GSH_synthesis_Vmax_uM_s=float(ref[R_VSYN]),
                                         RTA_regeneration_per_s=float(ref[R_KREG]),
                                         NADPH_supply_Vmax_uM_s=float(ref[R_VPPP])))
    if residual > 1e-9:
        raise ArithmeticError('Resting state is not stationary: ' + str(residual))
    return y, ref, report


def checks(z, base, ref):
    """Material ledgers; every value should be at solver tolerance."""
    z = np.atleast_2d(z)
    d = z - base
    err = lambda a, s: float(np.max(np.abs(a)) / s)
    return dict(
        lipid_total=err(d[:, 0:5].sum(1), ref[R_LTOT]),
        hydroperoxide=err(d[:, 2] - d[:, 11] + d[:, 12] + d[:, 13], max(1., np.max(np.abs(z[:, 11])))),
        repair=err(d[:, 3] + d[:, 4] - d[:, 12] - d[:, 13] + d[:, 14], max(1., np.max(np.abs(z[:, 14])))),
        aldehyde=err(d[:, 5] - d[:, 15] + d[:, 16] + d[:, 17] + d[:, 18], max(1e-3, np.max(np.abs(z[:, 15])))),
        aldehyde_yield=err(d[:, 15] - ref[R_YHNE] * d[:, 13], max(1e-3, np.max(np.abs(z[:, 15])))),
        adducts=err(d[:, 6] - d[:, 18] + d[:, 19], max(1e-3, np.max(np.abs(z[:, 18])))),
        glutathione=err(d[:, 7] + 2. * d[:, 8] - d[:, 20] + d[:, 21], max(1., np.max(np.abs(z[:, 20])))),
        NADPH=err(d[:, 9] - d[:, 22] + d[:, 23], max(1., np.max(np.abs(z[:, 22])))),
        RTA=err(d[:, 10] + d[:, 24] - d[:, 25], max(1e-3, np.max(np.abs(z[:, 24])))),
        negative_pool=float(max(0., -z[:, :11].min())),
        NADPH_bound=float(max(0., (z[:, 9] - ref[R_NTOT]).max())),
        RTA_bound=float(max(0., (z[:, 10] - ref[R_RTATOT]).max())))


def observe(z, ref):
    z = np.atleast_2d(z)
    gsh, gssg = np.maximum(z[:, 7], 1e-12), np.maximum(z[:, 8], 1e-12)
    return dict(LOOH_per_lipid=z[:, 2] / ref[R_LTOT],
                oxidizable_fraction=z[:, 0] / ref[R_LTOT],
                damaged_lipid_fraction=(z[:, 2] + z[:, 3] + z[:, 4]) / ref[R_LTOT],
                GSH_over_GSSG=gsh / gssg,
                # Nernst potential of the couple at 37 C, E0' = -240 mV at pH 7.0; concentrations in M.
                glutathione_potential_mV=-240. - 30.75 * np.log10((gsh * 1e-6) ** 2 / (gssg * 1e-6)),
                NADPH_fraction=z[:, 9] / ref[R_NTOT],
                RTA_reduced_fraction=z[:, 10] / ref[R_RTATOT],
                HNE_uM=z[:, 5], adducts_uM=z[:, 6], ATP_consumed_uM=z[:, 26],
                scope='Standalone block; observers are model quantities, not calibrated injury or death probabilities.')
