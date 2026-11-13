"""[codex] 2026-10-24. Source-scoped measurements, not new human kinetic fits.

Watt: 10.1073/pnas.1011099107; Zhang: 10.1523/JNEUROSCI.4973-06.2007;
Walton: 10.1113/jphysiol.2002.025064. See the stage1 report for scope.
"""
import numpy as np


def _nonnegative(x, name):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError(name + ': finite nonnegative values required')
    return x


def _cumulative(x, name):
    x = _nonnegative(x, name)
    if x.ndim != 1 or len(x) < 2 or np.any(np.diff(x) < -1e-9):
        raise ValueError(name + ': cumulative transported amounts required')
    return x


def watt_stoichiometry():
    """Ideal structural accounting; P/O is per oxygen ATOM (half O2).

    Bovine heart c8 ring; 3 ATP/turn. Cytoplasmic export costs one additional
    proton-equivalent. These are NOT respiration rates or a human MN P/O fit.
    """
    return dict(synthase_H_per_ATP=8/3, cytoplasmic_H_equivalent_per_ATP=11/3,
                ideal_ATP_per_O_atom_NADH=30/11,
                ideal_ATP_per_O_atom_succinate=18/11,
                ideal_ATP_per_O2_NADH=60/11,
                ideal_ATP_per_O2_succinate=36/11,
                mito_export_ATP_equivalent_per_Ca=9/11,
                mitochondrial_export_basis='Conditional 3 Na/Ca and 1 H/Na from Connolly; not direct NCLX ATP hydrolysis',
                human_MN_calibrated=False)


def calcium_energy_cost(q_pmca_uM, q_serca_uM, q_mito_export_uM, *,
                        accessible_volume_um3=None, provenance):
    """Same-volume TOTAL transported Ca, never delta free Ca times buffer again.

    SERCA and mitochondrial cycling can occur before PMCA extrusion: distinct
    sequential costs, not mutually exclusive fractions. Unknown route amounts
    must be supplied explicitly; this function will not turn missing into zero.
    """
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('Explicit route provenance required')
    p, s, m = [_cumulative(x, n) for x, n in
               [(q_pmca_uM, 'PMCA'), (q_serca_uM, 'SERCA'), (q_mito_export_uM, 'mitochondrial export')]]
    if not (p.shape == s.shape == m.shape):
        raise ValueError('Route arrays must share the time grid and volume')
    direct = p + s/2
    indirect = m*9/11
    out = dict(direct_ATP_hydrolysis_uM=direct,
               mitochondrial_gradient_ATP_equivalent_uM=indirect,
               counted_ATP_equivalent_uM=direct+indirect,
               legacy_Connolly_mito_coefficient=0.8,
               precision_difference_uM=m*(9/11-.8),
               predicts_ATP_level=False, full_cell_energy_budget=False,
               provenance=provenance, stoichiometry=watt_stoichiometry())
    if accessible_volume_um3 is not None:
        if not np.isfinite(accessible_volume_um3) or accessible_volume_um3 <= 0:
            raise ValueError('Positive accessible volume required')
        out['counted_ATP_equivalent_molecules'] = (direct+indirect)*accessible_volume_um3*602.214076
    return out


TRPM5_ENDPOINTS = {
    'WT_fresh': (8., 3.2, 1.), 'WT_desensitized': (48., 1.4, .11),
    'heterozygous_fresh': (14., 2.5, 1.), 'heterozygous_desensitized': (133., 1.2, .10)
}


def trpm5_endpoint_response(calcium_uM, *, endpoint):
    """Zhang Fig8 excised mouse taste-cell patches; Vmax relative to fresh max.

    Endpoints are not a kinetic trajectory. No transition/recovery rate or
    human-MN conductance is inferred. TRPM5 conducts monovalent cations.
    """
    ca = _nonnegative(calcium_uM, 'free Ca')
    if endpoint not in TRPM5_ENDPOINTS:
        raise ValueError('Select a source-scoped TRPM5 endpoint')
    half, hill, vmax = TRPM5_ENDPOINTS[endpoint]
    # Stable at zero and for large inputs, avoiding ca**hill overflow.
    x = np.zeros_like(ca)
    positive = ca > 0
    x[positive] = vmax/(1+np.exp(np.clip(hill*(np.log(half)-np.log(ca[positive])), -700, 700)))
    return dict(relative_current=x, half_uM=half, hill=hill,
                maximum_relative_to_fresh=vmax, kinetics_identified=False,
                human_MN_calibrated=False, calcium_current=False)


def walton_sustained_firing(spike_times_s, *, stimulus_end_s, recording_end_s,
                            control_rate_before_Hz=None, control_rate_after_Hz=None,
                            control_ramp_verified=False, matched_human_protocol=False):
    """Conservative paired-unit validation gate, NOT a criterion for all neurons.

    Walton: >1s persistence, >=5 spikes for frequency estimates, ISI CV<20%,
    no increase of control-unit frequency. Even +0.05Hz was excluded. Native
    current steps lack paired motor units/background drive and remain unmatched.
    """
    t = _nonnegative(spike_times_s, 'spike times')
    if t.ndim != 1 or np.any(np.diff(t) <= 0) or not np.isfinite([stimulus_end_s, recording_end_s]).all() or stimulus_end_s < 0 or recording_end_s <= stimulus_end_s or np.any(t > recording_end_s):
        raise ValueError('Ordered spike times inside the recording required')
    post = t[t > stimulus_end_s]
    span = float(post[-1]-post[0]) if len(post)>1 else 0.
    cv = float(np.std(np.diff(post), ddof=0)/np.mean(np.diff(post))) if len(post)>2 else None
    rates_known = control_rate_before_Hz is not None and control_rate_after_Hz is not None
    control_ok = None
    if rates_known:
        _nonnegative([control_rate_before_Hz, control_rate_after_Hz], 'control rates')
        control_ok = bool(control_rate_after_Hz <= control_rate_before_Hz)
    temporal = bool(len(post)>=5 and span>1 and cv is not None and cv<.2)
    complete = bool(matched_human_protocol and rates_known and control_ramp_verified)
    return dict(post_spike_count=int(len(post)), post_train_span_s=span, ISI_CV=cv,
                temporal_gate=temporal, control_no_increase=control_ok,
                matched_protocol_complete=complete,
                source_criterion_pass=bool(temporal and control_ok) if complete else None,
                status='evaluated_source_criterion' if complete else 'insufficient_or_unmatched_protocol',
                inferred_TRPM5_or_ER_mechanism=False)


def sxrf_total_element(values_ng_mm2, *, element, numeric_calibration,
                       sample_preparation, thickness_um=None,
                       thickness_preparation=None):
    """Quantified areal density -> atoms/area; optional total molarity only.

    Cryo-hydrated and dried geometry cannot be interchanged. RGB-only maps and
    min/max colour bars do not supply numeric calibration. No oxidation-state,
    labile-ion, organelle or live geometry inference from total elemental maps.
    """
    weights={'P':30.973761998,'S':32.06,'Zn':65.38,'K':39.0983,'Fe':55.845,'Cu':63.546,'Ca':40.078}
    if element not in weights or numeric_calibration is not True:
        raise ValueError('Known element and actual numerical calibration required')
    if sample_preparation not in ('live','frozen_hydrated','freeze_dried','chemically_fixed'):
        raise ValueError('Declare specimen preparation')
    v=_nonnegative(values_ng_mm2, 'areal density')
    out=dict(atoms_nm2=v*602.214076/weights[element],
             sample_preparation=sample_preparation, element=element,
             free_ion_concentration_inferred=False, human_MN_calibrated=False)
    if thickness_um is not None:
        h=np.asarray(thickness_um,float)
        if thickness_preparation != sample_preparation or not np.isfinite(h).all() or np.any(h<=0) or h.shape not in ((),v.shape):
            raise ValueError('Measured thickness for the SAME preparation and matching ROI required')
        out['total_element_uM_in_prepared_specimen']=v/h*1e6/weights[element]
        out['live_concentration_inferred']=sample_preparation=='live'
    return out


def annotate_fast_recording(recording):
    """Read existing trajectories without changing any state or electrical fit."""
    r=recording; q=np.asarray(r['qout'],float);s=_cumulative(r['qserca'], 'SERCA')
    if q.shape!=s.shape or not np.isfinite(q).all():
        raise ValueError('Finite matched flux counters required')
    er_cost=s/2
    return dict(
        energy=dict(SERCA_ATP_hydrolysis_uM=er_cost,
                    effective_NET_extrusion_uM=q.copy(),
                    PMCA_ATP_hydrolysis_uM=None,
                    mitochondrial_gradient_cost=None,
                    total_cost=None, ATP_level=None,
                    missing=['Gross PMCA flux: qout is signed effective excess clearance, not basal-inclusive pump flux','mitochondrial Ca cycling','ATP supply and other demands'],
                    stoichiometry=watt_stoichiometry()),
        TRPM5_source_endpoints={key:trpm5_endpoint_response(r['cs'],endpoint=key)
                              for key in ('WT_fresh','WT_desensitized')},
        sustained_firing=walton_sustained_firing(
            np.asarray(r['summary']['spike_times_ms'])/1000,
            stimulus_end_s=(r['protocol']['pre_ms']+r['protocol']['duration_ms'])/1000,
            recording_end_s=float(r['t'][-1])/1000),
        upstream_feedback=False)
