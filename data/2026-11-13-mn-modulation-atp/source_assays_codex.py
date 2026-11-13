"""[codex] 2026-11-13. Species-separated MN modulation and energy observations.

These are source assays and read-only outputs, not fitted human-MN currents.
The HCN/pump reference is a network-model component, never a new current owner.
"""
from pathlib import Path
import json
import numpy as np

B = Path(__file__).resolve().parent
F = 96485.33212


def mn_modulation_energy_sources():
    return json.loads((B / 'source_constraints_codex.json').read_text(encoding='utf-8'))


def sharples2021_hcn_reference():
    return json.loads((B / 'hcn_source_data_codex.json').read_text(encoding='utf-8'))


def hcn_ionic_current(*, voltage_mV, ENa_mV, EK_mV, open_probability,
                      g_total_nS, sodium_conductance_fraction, provenance):
    """Partition current by conductance and driving force, NOT by net-current fraction.

    Supply an open probability (already exponentiated if the chosen model uses m^2).
    At the mixed reversal voltage, net current can be zero while Na/K still flow.
    No ATP is directly debited: subsequent pump work is a separate owner.
    """
    v, ena, ek, q, g, f = np.broadcast_arrays(*[np.asarray(x, float) for x in
        (voltage_mV, ENa_mV, EK_mV, open_probability, g_total_nS, sodium_conductance_fraction)])
    if not all(np.isfinite(x).all() for x in (v, ena, ek, q, g, f)):
        raise ValueError('Finite inputs required')
    if np.any((q < 0) | (q > 1) | (f < 0) | (f > 1) | (g < 0)) or not str(provenance).strip():
        raise ValueError('Probability/fraction in [0,1], nonnegative g and explicit provenance required')
    ina = g*q*f*(v-ena)
    ik = g*q*(1-f)*(v-ek)
    return dict(Na_out_pA=ina, K_out_pA=ik, total_out_pA=ina+ik,
                Na_in_mol_s=-ina*1e-12/F, K_in_mol_s=-ik*1e-12/F,
                mixed_reversal_mV=f*ena+(1-f)*ek, ATP_debit_mol_s=0.,
                provenance=provenance, human_MN_calibrated=False)


def sharples2022_pump_reference(Nai_mM, Ko_mM):
    """Published HCO-model current, not measured pump capacity of a human neuron.

    Source: fncel.2021.715427 p16, Imax40.26pA, Na half25mM/slope3mM,
    K half6mM/slope1mM. Source model fixes K and has no finite ATP pool.
    Output ATP demand is a stoichiometric observation only, not an ATP debit.
    """
    na, k = np.broadcast_arrays(np.asarray(Nai_mM, float), np.asarray(Ko_mM, float))
    if not np.isfinite(na).all() or not np.isfinite(k).all() or np.any(na < 0) or np.any(k < 0):
        raise ValueError('Nonnegative finite concentrations in mM required')
    current = 40.26/(1+np.exp(np.clip((25-na)/3, -700, 700)))/(1+np.exp(np.clip(6-k, -700, 700)))
    cycles = current*1e-12/F
    return dict(pump_out_pA=current, Na_out_mol_s=3*cycles, K_in_mol_s=2*cycles,
                hypothetical_ATP_demand_mol_s=cycles,
                scope='Published CPG model component; no finite ATP dependence or potassium dynamics',
                human_MN_calibrated=False)


def adenylate_extract_observer(ATP_uM, ADP_uM, AMP_uM, *, volumes_um3,
                               pool_semantics):
    """Amount-weighted model extract. Last array axis is compartment, not time.

    A concentration ratio is never averaged across compartments. Unknown binding,
    recovery and unrepresented cell types/organelles prevent a calibrated HPLC fit.
    Even total-extractable input describes only the supplied compartments.
    """
    if pool_semantics not in ('model_pool', 'free', 'total_extractable'):
        raise ValueError('Explicit pool_semantics required')
    atp, adp, amp = np.broadcast_arrays(*[np.asarray(x, float) for x in (ATP_uM, ADP_uM, AMP_uM)])
    vol = np.asarray(volumes_um3, float)
    if atp.ndim < 1 or vol.ndim != 1 or atp.shape[-1] != len(vol):
        raise ValueError('Last concentration axis must match explicit compartment volumes')
    if not all(np.isfinite(x).all() for x in (atp, adp, amp, vol)) or np.any(vol <= 0) or any(np.any(x < 0) for x in (atp, adp, amp)):
        raise ValueError('Nonnegative finite pools and positive volumes required')
    amounts = np.stack([np.sum(x*vol, axis=-1)*1e-21 for x in (atp, adp, amp)], axis=-1)
    if np.any(amounts[..., 1] <= 0) or np.any(amounts.sum(axis=-1) <= 0):
        raise ValueError('Positive ADP and total adenylates required for ratios')
    return dict(ATP_ADP_ratio=amounts[..., 0]/amounts[..., 1],
                adenylate_energy_charge=(amounts[..., 0]+.5*amounts[..., 1])/amounts.sum(axis=-1),
                amounts_mol=amounts, pool_semantics=pool_semantics,
                direct_HPLC_fit_allowed=False,
                limitation='Conditional represented-compartment extract; binding, recovery and culture normalization unmeasured')


def ateam_flim_observer(lifetime_ps, *, donor_lifetime_ps):
    """Apparent FRET efficiency from mean lifetimes, never an ATP concentration.

    Values outside [0,1] remain visible as QC failures; no silent clipping.
    Biexponential mixture and sensor-specific ATP calibration remain separate.
    """
    t, td = np.broadcast_arrays(np.asarray(lifetime_ps, float), np.asarray(donor_lifetime_ps, float))
    if not np.isfinite(t).all() or not np.isfinite(td).all() or np.any(t <= 0) or np.any(td <= 0):
        raise ValueError('Positive finite lifetimes required')
    efficiency = 1-t/td
    return dict(apparent_FRET_efficiency=efficiency,
                physically_in_range=(efficiency >= 0) & (efficiency <= 1),
                ATP_uM=None, ATP_calibration_available=False)


def compare_cart_response(baseline, drug, *, treatment, subtype):
    """Compare observable deltas to mouse CART/muscarine means, not channel gains.

    Only explicitly supplied observables are evaluated. Nonsignificant responses
    are not forced to zero. Paired SD/CI cannot be recovered from marginal SDs.
    """
    if treatment not in ('CART', 'muscarine') or subtype not in ('fast', 'slow'):
        raise ValueError('Named source treatment/subtype required')
    source = mn_modulation_energy_sources()['Eleftheriadis2023'][subtype][treatment]
    if set(baseline) != set(drug) or not baseline or not set(baseline) <= set(source['observables']):
        raise ValueError('Matching nonempty supported observable sets required')
    rows = {}
    for key, value in baseline.items():
        if not np.isfinite([value, drug[key]]).all():
            raise ValueError('Finite observations required')
        ref = source['observables'][key]
        rows[key] = dict(model_delta=drug[key]-value,
                         source_delta=ref['drug_mean']-ref['baseline_mean'],
                         units=ref['units'], paired_SD=None,
                         significant_in_source=ref['significant'])
    return dict(rows=rows, source='Mouse P7-12, C-bouton modulation',
                inferred_channel_multiplier=None, human_MN_calibrated=False)


def observe_mn_compartment_energy(result):
    """Read current integrated trajectory without mutating state/config/checkpoint.

    State indices are checked against the shared positional-v1 schema. Model ATP
    pools do not resolve free/bound/Mg species. Report direction only against
    Zimyanin2023; neither lifetime nor absolute healthy ATP is inferred.
    """
    z = np.asarray(result['state'], float)
    if z.ndim != 2 or z.shape[1] < 139:
        raise ValueError('Integrated state trajectory required')
    schema = json.loads((B.parent/'neuron-model/state_layout_v1_codex.json').read_text(encoding='utf-8'))['base_1600']
    expected = {120:'ATP_uM',121:'ADP_uM',122:'AMP_uM',136:'process_ATP_uM',137:'process_ADP_uM',138:'process_AMP_uM'}
    if any(schema[i] != n for i, n in expected.items()):
        raise ValueError('State layout mismatch')
    a, d, m = z[:, [120,136]], z[:, [121,137]], z[:, [122,138]]
    cfg = result['config']
    if 'mobile_water_volumes' in cfg:
        volume = cfg['mobile_water_volumes']['mobile_water_volume_um3']
    else:
        volume = [(1e9/F)/cfg['ion_background_reference'][j][6] for j in range(2)]
    valid = np.isfinite(np.c_[a,d,m]).all(axis=1) & (np.c_[a,d,m] >= 0).all(axis=1) & (d.sum(axis=1) > 0)
    ratio = np.full(len(z), np.nan); aec = np.full(len(z), np.nan)
    if np.any(valid):
        extract = adenylate_extract_observer(a[valid],d[valid],m[valid],volumes_um3=volume,pool_semantics='model_pool')
        ratio[valid] = extract['ATP_ADP_ratio']; aec[valid] = extract['adenylate_energy_charge']
    return dict(compartments=['soma','process'],volumes_um3=list(volume),
                ATP_uM=a.copy(),model_extract_ATP_ADP=ratio,model_extract_AEC=aec,
                valid_adenylate_observation=valid,
                process_ATP_lower_than_soma=np.where(valid,a[:,1] < a[:,0],False),
                ATP_sensor_lifetime_ps=None,quantitative_human_calibration=False,
                scope='Read-only represented pools. Published axonal sensor ordering is a qualitative comparator, not PCr or absolute ATP calibration.')
