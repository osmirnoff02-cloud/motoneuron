"""[codex] 2026-11-03. Source-specific ion-current probes, not a new human fit.

Jurk 2025 Fig6; Leroy 2015 Figs2/3/8; Amini 2004 TablesVI/VIII.
No mutation of accepted neuronal parameters or intracellular budgets.
"""
from pathlib import Path
import importlib.util
import sys
import json
import numpy as np

B = Path(__file__).resolve().parent


def jurk2025_ion_reference(channel):
    """Digitized mean I/V, not raw cells. Currents pA/pF, outward positive."""
    if channel not in ('Na_control', 'Na_TTX', 'K'):
        raise ValueError('Select Na_control, Na_TTX or K')
    raw = json.loads((B/'source_constraints.json').read_text(encoding='utf8'))
    out = raw['jurk2025'][channel]
    return dict(**out, source=raw['jurk2025']['source'],
                conditions=raw['jurk2025']['conditions'],
                raw_timecourses_available=False, human_joint_fit=False)


def _vector(x, name):
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError(name+': finite nonempty 1D array required')
    return a


def compare_jurk2025_ion_current(channel, voltage_mV, current_pA_per_pF, *,
                               holding_mV, pulse_ms, temperature_C,
                               readout, provenance):
    """Protocol-qualified comparison at published steps, without fitting.

RMSE is descriptive, NOT a probability, significance test or calibration pass.
Both experimental bath and model reversal-potential assumptions must be
described in provenance. No automatic conversion of density into total current.
"""
    r = jurk2025_ion_reference(channel)
    v = _vector(voltage_mV, 'voltage'); y = _vector(current_pA_per_pF, 'current')
    if v.shape != y.shape or np.any(np.diff(v) <= 0):
        raise ValueError('Matching arrays with increasing command voltages required')
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('Explicit source/solution assumptions required')
    if (holding_mV != r['holding_mV'] or pulse_ms != r['pulse_ms'] or
            not np.isfinite(temperature_C) or not 22 <= temperature_C <= 25 or
            readout != 'peak'):
        raise ValueError('Source requires its holding voltage, pulse, 22–25 C and peak readout')
    vr = np.array(r['voltage_mV']); yr = np.array(r['mean_pA_per_pF'])
    idx = []
    for x in v:
        hit = np.flatnonzero(np.isclose(vr, x, atol=1e-9, rtol=0))
        if len(hit) != 1:
            raise ValueError('Only digitized command voltages supported; no extrapolation')
        idx.append(int(hit[0]))
    error = y-yr[idx]
    return dict(residual_pA_per_pF=error, RMSE_pA_per_pF=float(np.sqrt(np.mean(error**2))),
                mean_absolute_error_pA_per_pF=float(np.mean(abs(error))),
                biological_goodness_of_fit_pass=None,
                digitization_error_is_not_SEM=True, human_joint_fit=False,
                provenance=provenance, reference=r)


def _hh():
    name = '_ion_probe_hh20261103'
    if name not in sys.modules:
        path = B.parent/'2026-10-14-ipsc-electrical-multisource/electrical_model.py'
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def probe_frozen_ion_clamp(profile, channel, *, reversal_mV, provenance, samples=4001):
    """Existing HH density and gates under a different, explicit ideal clamp.

Source baths block other ions; this probe isolates only effective Na or K.
No fitted TTX-resistant branch: Na_TTX is deliberately unsupported.
Reversal potential is mandatory, because salt labels do not determine exact
ionic activities. Existing gating kinetics are retained at nominal 23 C.
Gates start at equilibrium at the holding potential; finite preconditioning
is not reconstructed. Later coupled calcium/KCNQ feedback is not simulated.
This is a transfer diagnostic of the frozen electrical core across
preparations, not a same-cell fit or total current of the coupled neuron.
"""
    if channel not in ('Na_control', 'K'):
        raise ValueError('No identified TTX-resistant conductance in frozen HH profiles')
    if not np.isfinite(reversal_mV) or not isinstance(provenance,str) or not provenance.strip():
        raise ValueError('Explicit finite reversal and provenance required')
    if isinstance(samples, bool) or int(samples) != samples or samples < 101:
        raise ValueError('At least 101 time samples required')
    profiles = json.loads((B.parent/'2026-10-19-electrical-recovery-refinement/profiles-accepted.json').read_text())['profiles']
    if profile not in profiles: raise ValueError('Unknown accepted profile')
    p = np.array(profiles[profile]['parameters']); ref = jurk2025_ion_reference(channel)
    hh = _hh(); g0 = np.array(hh.gates(ref['holding_mV'], p))
    t = np.linspace(0., ref['pulse_ms'], int(samples))
    peak=[]; end=[]
    for v in ref['voltage_mV']:
        ginf = np.array(hh.gates(v,p)); tau = np.array(hh.taus(v,p))
        g = ginf[:,None] + (g0-ginf)[:,None]*np.exp(-t[None,:]/tau[:,None])
        # g in mS/cm2 divided by assumed 1 uF/cm2 gives nS/pF numerically.
        conductance = p[0]*g[0]**3*g[1] if channel == 'Na_control' else p[3]*g[2]**4+p[12]*g[3]
        current = conductance*(v-reversal_mV)
        # At a reversal above the test range Na is inward; if a user supplies
        # another reversal, choose the extreme in the driving-force direction.
        peak.append(float(np.min(current) if v < reversal_mV else np.max(current)))
        end.append(float(current[-1]))
    comparison = compare_jurk2025_ion_current(channel, ref['voltage_mV'], peak,
        holding_mV=ref['holding_mV'], pulse_ms=ref['pulse_ms'], temperature_C=23.,
        readout='peak', provenance=provenance)
    return dict(profile=profile, channel=channel, voltage_mV=ref['voltage_mV'],
                peak_pA_per_pF=np.array(peak), end_pA_per_pF=np.array(end),
                reversal_mV=float(reversal_mV), samples=int(samples), comparison=comparison,
                fitted_parameters=False, transferred_between_preparations=True,
                limitation='Frozen electrical core only; ideal isolated-channel clamp with equilibrated holding gates. No coupled calcium/KCNQ feedback, finite preconditioning, series resistance, junction correction, space-clamp error or preparation-specific refit. Effective K is not resolved into IA/Kslow/KCNQ/BK/SK; it is not the summed K current of the 1600-state neuron.')


def leroy2015_recruitment_reference():
    return json.loads((B/'source_constraints.json').read_text(encoding='utf8'))['leroy2015']


def amini2004_pump_reference(Na_i_mM, K_o_mM, voltage_mV, *, provenance):
    """Published donor equation ONLY; fixed-ion rat model, not human pump fit.

TableVI I_NaK = .6*(Na/(Na+5.46))^3*(K/(K+.621))^2*(V+150)/(V+200) nA.
Reject voltages outside a conservative numerical operating window instead of
passing through its unphysical singularity at -200 mV. That guard is not a
measured voltage-validity interval. ATP dependence was not parameterized.
"""
    na,k,v = np.broadcast_arrays(np.asarray(Na_i_mM,float), np.asarray(K_o_mM,float), np.asarray(voltage_mV,float))
    if not all(np.isfinite(x).all() for x in (na,k,v)) or np.any(na<0) or np.any(k<0) or np.any((v < -100)|(v > 50)):
        raise ValueError('Nonnegative finite ions and conservative voltage domain [-100,50] mV required')
    if not isinstance(provenance,str) or not provenance.strip():raise ValueError('Explicit use provenance required')
    current=600*(na/(na+5.46))**3*(k/(k+.621))**2*(v+150)/(v+200)
    return dict(net_outward_current_pA=current, source='Amini2004 TablesVI/VIII; DOI10.1109/TBME.2004.827949',
                fixed_ions_in_original=True, original_Na_i_mM=8.71, original_K_o_mM=5.9,
                ATP_dependence_identified=False, human_MN_calibrated=False,
                long_term_ion_stability_established=False, provenance=provenance)
