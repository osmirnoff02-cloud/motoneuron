"""[codex] 2026-11-11. Protocol-specific observations, not MN parameter fits.

Sources: Gameiro-Ros 2025 DOI10.1101/2025.04.07.646449 and archived
analysis code DOI10.5281/zenodo.14841953 (SlesingerLab, CC-BY-4.0);
Eberhardt 2015 DOI10.1016/j.stemcr.2015.07.010 Table S1;
Haythornthwaite 2012 DOI10.1177/1087057112457821 Methods/Table1.
No function changes a model state or owns an ionic/ATP flux.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit

B=Path(__file__).resolve().parent

def source_constraints():
    return json.loads((B/'source_constraints_codex.json').read_text(encoding='utf8'))

def patch_chloride_context(*, configuration, temperature_C, voltage_mV=-70.):
    """Ideal Cl-only reversal from nominal solutions, NOT native E_GABA.

    The 2012 article does not specify recording temperature. Caller must choose
    it explicitly. No bicarbonate permeability/activity or junction correction.
    Added chloride salts counted stoichiometrically; no KOH/NaOH guessed.
    """
    recipes={'Haythornthwaite2012_manual':(137.,158.4),
             'Haythornthwaite2012_automated':(60.,150.)}
    if configuration not in recipes: raise ValueError('Unknown source configuration')
    if not np.isfinite([temperature_C,voltage_mV]).all() or temperature_C<=-273.15:
        raise ValueError('Finite voltage and temperature above absolute zero required')
    ci,co=recipes[configuration]
    e=1000*8.31446261815324*(temperature_C+273.15)/96485.33212*np.log(ci/co)
    return dict(configuration=configuration,Cl_in_mM=ci,Cl_out_mM=co,
                temperature_C=float(temperature_C),E_Cl_mV=float(e),
                driving_force_mV=float(voltage_mV-e),native_neuron_calibration=False,
                interpretation='For V<ECl, negative conventional current and chloride efflux; for V>ECl, positive current and influx. This clamp recipe does not establish native GABA action.',
                limitations=['Ideal concentrations, Cl only, no activity/junction correction',
                             'Whole-cell pipette imposes intracellular composition',
                             'Recording temperature is a caller assumption, not reported here'])

def eberhardt_nociceptor_gating(voltage_mV,*,population,gate):
    """Normalized steady-state source curves; no density/time constant/MN transfer.

    Positive slope in S1 is a magnitude: activation rises, availability falls.
    Does not implement an AP model or multiply activation by an invented power.
    """
    table=source_constraints()['Eberhardt2015']['table_S1'][population]
    if gate not in ('activation','inactivation'): raise ValueError('Unknown gate')
    pars=table.get(gate)
    if pars is None: raise ValueError('No measured source curve for this gate/population')
    v=np.asarray(voltage_mV,float)
    if not np.isfinite(v).all(): raise ValueError('Finite voltages required')
    x=(v-pars['Vhalf_mV'])/pars['slope_mV']
    return dict(voltage_mV=v.copy(),fraction=expit(x if gate=='activation' else -x),
                population=population,gate=gate,source_fit=pars,
                human_motor_neuron_fit=False)

def sample_gameiro_fluorescence(time_s,fluorescence,*,frame_start_s=None):
    """150ms exposure means, 6.14Hz; input must already be fluorescence.

    Exact integral of the piecewise-linear input. No GCaMP8f kinetics are
    inferred from this paper. Never pass Ca concentration as fluorescence.
    Frame timestamps returned both at start and midpoint; no extrapolation.
    """
    t=np.asarray(time_s,float);f=np.asarray(fluorescence,float)
    if t.ndim!=1 or len(t)<2 or f.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(f).all() or np.any(np.diff(t)<=0):
        raise ValueError('Ordered finite time and same-length fluorescence required')
    exposure=.150;rate=6.14
    starts=(t[0]+np.arange(max(0,int(np.floor((t[-1]-t[0]-exposure)*rate))+1))/rate
            if frame_start_s is None else np.asarray(frame_start_s,float))
    if starts.ndim!=1 or not np.isfinite(starts).all() or np.any(np.diff(starts)<=0) or np.any(starts<t[0]) or np.any(starts+exposure>t[-1]+1e-12):
        raise ValueError('Complete ordered exposures must lie within the supplied trace')
    slopes=np.diff(f)/np.diff(t)
    cumulative=np.r_[0.,np.cumsum(.5*(f[:-1]+f[1:])*np.diff(t))]
    def integral(q):
        q=np.clip(q,t[0],t[-1]);idx=np.clip(np.searchsorted(t,q,side='right')-1,0,len(t)-2)
        dt=q-t[idx]
        return cumulative[idx]+f[idx]*dt+.5*slopes[idx]*dt*dt
    means=(integral(starts+exposure)-integral(starts))/exposure
    return dict(frame_start_s=starts.copy(),frame_midpoint_s=starts+exposure/2,
                fluorescence=means,exposure_s=exposure,frame_rate_Hz=rate,
                input_units='fluorescence, arbitrary units',absolute_Ca_uM=None,
                AP_count=None,source='Gameiro-Ros2025 Methods p14; acquisition only')

def gameiro_event_acceptance(*,rise_s,fall_s,height,amplitude_from_start,
                            trace_sd,background_max,policy):
    """Transparent feature gate, NOT a reproduction of the full R detector.

    Caller supplies features after its documented preprocessing. The archived
    R code differs from manuscript thresholds; policy is mandatory. Width
    here is rise+fall (full monotonic event, NOT FWHM). Paper boundary uses
    the printed strict inequalities; code uses frame counts 2/4 and <=40.
    """
    a=np.asarray([rise_s,fall_s,height,amplitude_from_start,trace_sd,background_max],float)
    if not np.isfinite(a).all() or np.any(a<0) or trace_sd<=0:
        raise ValueError('Finite nonnegative features and positive trace SD required')
    if policy=='manuscript_v1':
        checks=dict(rise=rise_s>.325,fall=fall_s>.650,width=rise_s+fall_s<6.4,
                    asymmetry=rise_s<fall_s,sd=amplitude_from_start>5*trace_sd,
                    background=height>5*background_max)
    elif policy=='archive_v1_defaults':
        checks=dict(rise=rise_s>=2/6.14,fall=fall_s>=4/6.14,width=rise_s+fall_s<=40/6.14,
                    asymmetry=rise_s<fall_s,
                    sd=(amplitude_from_start>3*trace_sd if trace_sd<20 else True),
                    background=height>3*background_max)
    else: raise ValueError('Explicit manuscript_v1 or archive_v1_defaults policy required')
    return dict(accepted=bool(all(checks.values())),checks=checks,policy=policy,
                status='feature sensitivity check only; raw human validation unavailable',
                limitation='Paper does not fully specify SD/height conventions; adjusted height is our explicit interpretation. R preprocessing/trim and findpeaks are not reproduced.')

def normalized_patch_current(current_pA,capacitance_pF):
    """pA/pF does not remove selection, dissociation or solution differences."""
    i,c=np.broadcast_arrays(np.asarray(current_pA,float),np.asarray(capacitance_pF,float))
    if not np.isfinite(i).all() or not np.isfinite(c).all() or np.any(c<=0):
        raise ValueError('Finite current and positive measured capacitance required')
    return i/c
