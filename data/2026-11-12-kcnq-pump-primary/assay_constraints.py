"""[codex] 2026-11-12. Source-specific assays, NOT a human-MN calibration.

No import-time modification of neuronal parameters. Currents measured in HEK,
CHO, cardiac or renal cells must not become motor-neuron defaults.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FARADAY_C_PER_MOL = 96485.33212
AVOGADRO_PER_MOL = 6.02214076e23

# Smith 2001 Table 1, PDF p5, HEK293, 20-22 C, 1 s step -70 -> +40 mV.
# Mean/SEM from TABLE; Q2L prose instead says SEM=14, retained in evidence file.
# Vhalf/slope are phenomenological conductance fits, not single-channel Po.
SMITH = {
    'Q2L': (-25.5, 5., 12., 1., 50., 9., 9),
    'Q2L_Q2S_1_1': (-28., 4., 12., 1., 37., 4., 6),
    'Q2L_Q2S_1_5': (-18., 3., 13., 1., 15., 3., 6),
    'Q3': (-40., 2., 11., 2., 36., 4., 7),
    'Q3_Q2S_1_1': (-19., 2., 15., 1., 13., 2., 10),
    'Q3_Q2S_1_5': (-6., 5., 10., 2., 14., 1., 10),
    'Q2L_Q3_1_1': (-35., 3., 14., .7, 144., 27., 5),
    'Q2L_Q3_Q2S_1_1_1': (-38., 3., 12., 2., 137., 28., 6),
    'Q2L_Q3_Q2S_1_1_5': (-14., 3., 12., 1., 202., 20., 5),
}


def primary_pump_kcnq_sources():
    """Measured endpoints, provenance and explicit transfer limitations."""
    return json.loads((ROOT / 'evidence.json').read_text(encoding='utf-8'))


def ma2023_gating_reference():
    """Original normalized G-V observations and separately labelled local fits."""
    return json.loads((ROOT / 'ma_gating.json').read_text(encoding='utf-8'))


def smith_kcnq_activation(voltage_mV, construct, *, temperature_C=21.):
    """Descriptive reconstruction of published HEK G-V fits, not MN kinetics.

    Temperature is a protocol guard, not a Q10 correction. No tau is inferred.
    Current density at +40 is provided as a separate measured endpoint and is
    not treated as maximal conductance or a motor-neuron current density.
    """
    if construct not in SMITH:
        raise ValueError('Select an explicit Smith Table 1 construct')
    if not math.isfinite(temperature_C) or not 20. <= temperature_C <= 22.:
        raise ValueError('Smith recordings: 20-22 C; no measured Q10 supplied')
    if not math.isfinite(voltage_mV):
        raise ValueError('Finite voltage required')
    vh, vhsem, slope, ssem, dens, dsem, n = SMITH[construct]
    x = (voltage_mV - vh) / slope
    conductance_fraction = 1. / (1. + math.exp(-x)) if x >= 0 else math.exp(x) / (1. + math.exp(x))
    return dict(normalized_conductance=conductance_fraction,
                Vhalf_mV=vh, Vhalf_SEM_mV=vhsem, slope_mV=slope, slope_SEM_mV=ssem,
                step40_density_pA_per_pF=dens, step40_density_SEM=dsem, n=n,
                tau_s=None, system='HEK293; human KCNQ constructs',
                human_MN_calibrated=False, doi='10.1523/JNEUROSCI.21-04-01096.2001')


def pump_current_to_fluxes(net_outward_pA, *, accessible_water_um3):
    """Stoichiometric observer of a supplied forward NKA current.

    3 Na out : 2 K in : 1 ATP : 1 net elementary charge. Volume must be the
    accessible Na/K/ATP aqueous volume, NOT the buffered effective Ca volume.
    This reports a budget; it does NOT add another ATP charge to the RHS.
    """
    if not math.isfinite(net_outward_pA) or net_outward_pA < 0:
        raise ValueError('Nonnegative forward NKA current required')
    if not math.isfinite(accessible_water_um3) or accessible_water_um3 <= 0:
        raise ValueError('Positive finite accessible water volume required')
    rate = net_outward_pA * 1e-12 / FARADAY_C_PER_MOL
    rate_uM_s = rate / (accessible_water_um3 * 1e-15) * 1e6
    return dict(ATP_mol_per_s=rate, ATP_molecules_per_s=rate * AVOGADRO_PER_MOL,
                ATP_cost_uM_per_s=rate_uM_s, Na_in_rate_uM_per_s=-3 * rate_uM_s,
                K_in_rate_uM_per_s=2 * rate_uM_s, net_outward_pA=net_outward_pA,
                observation_only=True, adds_RHS_cost=False)


def typical_table_audit():
    """User-supplied, unsourced table: independent-endpoint arithmetic only."""
    return deepcopy(dict(
        source_status='user supplied; original source and matched conditions missing',
        human_MN_calibrated=False, used_as_default=False,
        conductance_from_Rin_nS=[1000 / 1500, 1000 / 500],
        density_times_capacitance_pA=[1 * 20, 4 * 45],
        frozen_conductance_pump_voltage_magnitude_mV=[5 * 500 / 1000, 25 * 1500 / 1000],
        interpretation=[
            '1/Rin is total local slope conductance, not isolated K2P/leak.',
            'Independent extrema are not a measured interval in matched cells.',
            'I*R assumes frozen conductances; it is not the equilibrated pump contribution.',
            'Activation/deactivation tau require voltage, temperature, subunits and protocol.',
            'Pump-cycle time, Na recovery time and protein trafficking time are distinct.',
            'The table supplies no matched AMPA current, neurite water volume or ATP/ADP/PCr.'
        ]))
