"""[codex] 2026-11-11. Protocol-specific observations and an isolated KNa candidate.

No replacement of human-MN conductances or native K/Na/ATP states. The KNa replay
is deliberately one-way: local Na, channel density and overlap with effective
adaptation are not identified. Its current must not be added to the fitted
membrane before choosing an owner and rechecking material/charge balance.
"""
from pathlib import Path
from dataclasses import dataclass,asdict
import json
import numpy as np
B=Path(__file__).resolve().parent
FARADAY=96485.33212
# Scenario grid, not measured bounds or a human confidence interval.
KNA_SCENARIO_RANGES={
 'Na_half_mM':[20.,39.9,80.], 'hill':[1.,2.,4.],
 'tau_s':[.005,.05,.5], 'gmax_nS':[0.,.1,1.]}

def source_constraints():
    return json.loads((B/'source_constraints.json').read_text(encoding='utf8'))

def bardy_sweep_classification(*,nav_present,ap_peaks_mV,firing_rate_hz,
                               plateau_after_first,aborted_spikes,
                               holding_mV,temperature_C,pulse_ms,ljp_corrected_mV):
    """Compatible AP class for a selected sweep, not full cell/PatchSeq typing.

    Frequency is supplied explicitly because source frequency estimator is not
    fully specified. No silent substitution of count/window for interspike rate.
    """
    values=[firing_rate_hz,holding_mV,temperature_C,pulse_ms,ljp_corrected_mV]
    if not np.all(np.isfinite(values)):raise ValueError('Finite protocol values required')
    if not (np.isclose(holding_mV,-70) and np.isclose(temperature_C,25)
            and np.isclose(pulse_ms,500) and np.isclose(ljp_corrected_mV,10)):
        raise ValueError('Bardy-specific -70mV,25C,500ms,10mV LJP protocol required')
    peaks=np.asarray(ap_peaks_mV,dtype=float)
    if peaks.ndim!=1 or not np.all(np.isfinite(peaks)):raise ValueError('Finite peak vector required')
    if firing_rate_hz<0 or aborted_spikes<0 or int(aborted_spikes)!=aborted_spikes:
        raise ValueError('Nonnegative rate and integer aborted count required')
    n=int(np.sum(peaks>-10))
    if not nav_present and n:raise ValueError('APs with absent Nav conflict with source classification')
    if not nav_present:kind=0
    elif n==0:kind=1
    elif n>1:kind=5 if firing_rate_hz>=10 else 4
    elif plateau_after_first and not aborted_spikes:kind=2
    elif aborted_spikes and not plateau_after_first:kind=3
    else:kind=None
    return dict(compatible_sweep_type=kind,full_AP_count=n,cell_maturity_confirmed=False,
        reason='One sweep only; full protocol requires current-step series and Nav assay. Single-spike ambiguity retained.')

def jurk2025_maturity_reference(stage):
    s=source_constraints()['Jurk2025']
    if stage not in s['stages']:raise ValueError('Use DIV5_8 or DIV9_15 (post-thaw)')
    return dict(stage=stage,reference=s['stages'][stage],protocol=s['protocol'],
                source=s['doi'],scope='Same paper as previous ion references; not independent replication')

def compare_jurk2025_properties(values,*,stage,temperature_C,holding_mV,single_pulse_pA,single_pulse_ms):
    """Residuals in physical units/SEM, not a fit, confidence interval or pass test."""
    ref=jurk2025_maturity_reference(stage)
    if not (22<=temperature_C<=25 and np.isclose(holding_mV,-70)
            and np.isclose(single_pulse_pA,280) and np.isclose(single_pulse_ms,5)):
        raise ValueError('Jurk AP protocol: 22-25C, hold -70mV,280pA/5ms')
    out={}
    for name,v in values.items():
        if name not in ('threshold_mV','rise_ms','halfwidth_ms'):
            raise ValueError('Only explicitly defined source metrics; peak amplitude definition unresolved')
        if not np.isfinite(v):raise ValueError('Finite metric required')
        r=ref['reference'][name]
        out[name]=dict(value=float(v),source_mean=r['mean'],residual=float(v-r['mean']),
                       residual_in_SEM=float((v-r['mean'])/r['sem']))
    return dict(metrics=out,accepted_as_calibrated=False,
        limitation='Source SEM is uncertainty of a group mean, not single-cell tolerance; threshold detection algorithm still needs matching')

def fofie_retigabine_reference(dose_uM,condition='37C'):
    d=json.loads((B/'retigabine_source.json').read_text(encoding='utf8'))
    rows=[r for r in d['rows'] if r['treatment']=='retigabine' and r['dose_uM']==dose_uM and r['condition']==condition]
    if len(rows)!=1:raise ValueError('Measured doses 2.5,5,10uM; conditions 37C or 37_to_42C_ramp')
    return dict(measurements=rows[0],system=d['system'],acquisition=d['acquisition'],
        source_sheet=d['sheet'],dataset_doi=d['dataset_doi'],
        conductance_multiplier=None,human_MN_calibration=False,
        limitation='MEA count change is not KCNQ2 expression, channel current, calcium load or a cell survival endpoint')

def mea_well_observer(counts,*,recording_s,baseline_counts=None):
    """Rows=wells, columns=electrodes. Keep all electrodes, including silent ones."""
    a=np.asarray(counts,dtype=float)
    if a.ndim!=2 or min(a.shape)<1 or not np.all(np.isfinite(a)) or np.any(a<0) or not np.isfinite(recording_s) or recording_s<=0:
        raise ValueError('Nonnegative finite well x electrode counts and positive recording time required')
    total=a.sum(axis=1)
    out=dict(total_counts=total,detected_events_per_well_s=total/recording_s,
             active_electrode_fraction=(a/(recording_s/60)>=1).mean(axis=1),
             n_wells=a.shape[0],n_electrodes_per_well=a.shape[1],single_neuron_rate_hz=None)
    if baseline_counts is not None:
        b=np.asarray(baseline_counts,dtype=float)
        if b.shape!=a.shape or not np.all(np.isfinite(b)) or np.any(b<0) or np.any(b.sum(axis=1)<=0):
            raise ValueError('Matched positive well baselines required; zero baseline cannot be normalized')
        out['change_from_same_well_baseline_pct']=(total/b.sum(axis=1)-1)*100
    return out

@dataclass(frozen=True)
class KNaCandidateParameters:
    Na_half_mM:float
    hill:float
    tau_s:float
    gmax_nS:float
    provenance:str

    def validate(self):
        q=np.array([self.Na_half_mM,self.hill,self.tau_s,self.gmax_nS])
        if not np.all(np.isfinite(q)) or np.any(q[:3]<=0) or q[3]<0 or not self.provenance.strip():
            raise ValueError('Positive half/Hill/tau, nonnegative conductance and provenance required')

def replay_kna_candidate(time_s,Na_local_mM,Vm_mV,EK_mV,*,parameters,initial_gate,
                         Na_interpretation):
    """Integrate gate under a supplied local-Na scenario; return isolated current.

    Exact gate update for piecewise-constant Na on [t[i],t[i+1]). The current and
    signed K transfer are outputs only. No feedback or restoration of ions, no
    direct ATP charge. Local Na cannot be inferred from bulk Na in these papers.
    """
    parameters.validate();p=parameters
    t=np.asarray(time_s,dtype=float);na=np.asarray(Na_local_mM,dtype=float)
    if t.ndim!=1 or len(t)<2 or na.shape!=t.shape or not np.all(np.isfinite(t)) or np.any(np.diff(t)<=0):
        raise ValueError('Increasing time and matching sodium vector required')
    if not np.all(np.isfinite(na)) or np.any(na<0):raise ValueError('Nonnegative local sodium required')
    if Na_interpretation not in ('measured_local','explicit_local_scenario'):
        raise ValueError('Bulk Na is not automatically the AMPA/Slack microdomain')
    if not np.isfinite(initial_gate) or not 0<=initial_gate<=1:raise ValueError('Gate must be in [0,1]')
    v=np.broadcast_to(np.asarray(Vm_mV,dtype=float),t.shape)
    ek=np.broadcast_to(np.asarray(EK_mV,dtype=float),t.shape)
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(ek)):raise ValueError('Finite voltages required')
    with np.errstate(divide='ignore'):
        logodds=p.hill*(np.log(p.Na_half_mM)-np.log(na))
    target=np.exp(-np.logaddexp(0,logodds))
    gate=np.empty(len(t));gate[0]=initial_gate
    for i,dt in enumerate(np.diff(t)):
        gate[i+1]=target[i]+(gate[i]-target[i])*np.exp(-dt/p.tau_s)
    current=p.gmax_nS*gate*(v-ek)  # nS*mV=pA, outward positive
    # Integrate current exactly per interval with piecewise-constant Vm and EK.
    dt=np.diff(t);area_gate=target[:-1]*dt+(gate[:-1]-target[:-1])*p.tau_s*(-np.expm1(-dt/p.tau_s))
    charge=p.gmax_nS*(v[:-1]-ek[:-1])*area_gate  # pC
    transfer=np.r_[0,np.cumsum(charge/FARADAY)]  # pmol of monovalent K
    return dict(time_s=t,gate=gate,steady_gate=target,current_outward_pA=current,
        cumulative_K_out_pmol=transfer,cumulative_K_inside_delta_pmol=-transfer,
        cumulative_K_outside_delta_pmol=transfer,ATP_direct_pmol=0.,parameters=asdict(p),
        coupled_to_neuron_RHS=False,human_MN_calibrated=False,
        scope='One-way isolated KNa candidate. No Vm/Ca/TDP prediction and no finite-pool depletion assumed. Pump restoration cost is owned by the pump, not charged here.')
