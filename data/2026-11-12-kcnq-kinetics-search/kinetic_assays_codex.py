"""[codex] 2026-11-12. Explicit KCNQ clamp kinetics, separate from neuron defaults.

Piecewise first-order relaxation is an assay approximation. Activation delays,
sigmoidicity, rundown, Rs/Cm transients and temperature transfer are NOT fitted.
The caller owns g, EK, q-infinity, activation tau and initial state. No new
conductance, pump, ATP cost or artificial ion restoration is added to the RHS.
"""
from pathlib import Path
import json
import numpy as np

B=Path(__file__).resolve().parent

def kcnq_kinetic_sources():
    return json.loads((B/'kinetic_sources_codex.json').read_text(encoding='utf8'))

def cyl3059_deactivation(voltage_mV, *, interpolate=False):
    """CHO-K1 human Kv7.2/7.3; measured points or explicit linear interpolation.

    This is NOT a human-motoneuron parameter provider. No extrapolation or Q10.
    The source does not define the type of its reported +/- error for these n=3.
    """
    if not np.isfinite(voltage_mV):raise ValueError('Finite voltage required')
    d=kcnq_kinetic_sources()['CYL3059']
    vs=np.asarray(d['voltage_mV']); tau=np.asarray(d['tau_deactivation_ms'])
    err=np.asarray(d['reported_error_ms']); idx=np.where(vs==voltage_mV)[0]
    if len(idx):
        t=tau[idx[0]]; e=float(err[idx[0]]); status='reported mean'
    elif interpolate and vs[0]<voltage_mV<vs[-1]:
        t=np.interp(voltage_mV,vs,tau); e=None;status='linear interpolation, not measurement'
    else:raise ValueError('Only measured -70/-60/-50/-40 mV; interpolation must be explicit and within support')
    return dict(voltage_mV=float(voltage_mV),tau_s=float(t/1000),
                reported_error_ms=e,error_type='unspecified',n=3,status=status,
                temperature_C=None,human_motor_neuron_calibration=False,
                source='CYL3059 p5; human proteins in CHO-K1')

def kcnq_clamp_protocol(steps, *, initial_gate, g_nS, EK_mV):
    """Analytic gate state/current/charge over piecewise constant voltage steps.

    Each step explicitly supplies voltage_mV, duration_s, q_inf, tau_s,
    tau_provenance. All units are explicit: nS*mV=pA, pA*s=pC. Initial gate is
    never reset between steps. outward positive K charge/F gives K amount;
    concentration requires a separately measured accessible water volume.
    """
    if not np.isfinite([initial_gate,g_nS,EK_mV]).all() or not 0<=initial_gate<=1 or g_nS<0:
        raise ValueError('Finite initial gate [0,1], nonnegative g and finite EK required')
    if not steps:raise ValueError('At least one explicit clamp step required')
    q=float(initial_gate);time=0.;out=[]
    for s in steps:
        v,d,qi,tau=[float(s[x]) for x in ('voltage_mV','duration_s','q_inf','tau_s')]
        if not np.isfinite([v,d,qi,tau]).all() or d<=0 or tau<=0 or not 0<=qi<=1:
            raise ValueError('Finite positive duration/tau and q_inf in [0,1] required')
        if not str(s.get('tau_provenance','')).strip():raise ValueError('Explicit tau provenance required')
        relax=-np.expm1(-d/tau)
        qend=q+(qi-q)*relax
        int_q=qi*d+(q-qi)*tau*relax
        drive=g_nS*(v-EK_mV)
        charge=drive*int_q
        out.append(dict(t_start_s=time,t_end_s=time+d,voltage_mV=v,
            gate_start=q,gate_end=float(qend),start_pA=drive*q,end_pA=drive*qend,
            end_minus_start_pA=drive*(qend-q),K_out_charge_pC=charge,
            K_out_mol=charge*1e-12/96485.33212,tau_s=tau,
            tau_provenance=s['tau_provenance']))
        q=float(qend);time+=d
    return dict(steps=out,final_gate=q,total_K_out_charge_pC=sum(x['K_out_charge_pC'] for x in out),
                scope='Isolated first-order voltage-clamp approximation; not the integrated human neuron')

def fit_kcnq_relaxation(time_s,current_pA):
    """Fit offset+A*exp(-(t-t0)/tau), with free sign of A and positive tau.

    Caller must exclude capacitive artifacts, subtract appropriate drug/leak
    controls and provide ONE constant-voltage segment. A fitted curve alone
    cannot establish KCNQ specificity. No uncertainty interval is inferred.
    """
    from scipy.optimize import least_squares
    t=np.asarray(time_s,dtype=float);y=np.asarray(current_pA,dtype=float)
    if t.ndim!=1 or y.shape!=t.shape or len(t)<5 or not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(np.diff(t)<=0):
        raise ValueError('At least five finite ordered paired time/current values required')
    if np.ptp(y)<1e-10:raise ValueError('Constant current does not identify tau')
    t=t-t[0];span=t[-1]
    tau_lo=max(np.min(np.diff(t))*1e-3,1e-12);tau_hi=span*1e3
    def residual(p):return p[0]+p[1]*np.exp(-t/np.exp(p[2]))-y
    fit=least_squares(residual,[y[-1],y[0]-y[-1],np.log(span/4)],
        bounds=([-np.inf,-np.inf,np.log(tau_lo)],[np.inf,np.inf,np.log(tau_hi)]),
        xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=2000)
    return dict(tau_s=float(np.exp(fit.x[2])),offset_pA=float(fit.x[0]),amplitude_pA=float(fit.x[1]),
        rmse_pA=float(np.sqrt(np.mean(fit.fun**2))),success=bool(fit.success),
        numerical_tau_bounds_s=[tau_lo,tau_hi],
        bound_limited=bool(np.any(fit.active_mask)),human_motor_neuron_calibration=False,
        limitation='Numerical fit bounds are not physiological ranges; short/noisy traces may not identify tau')

def sharples2023_reference():
    """Measured mouse per-cell amplitudes; no raw I(t) or native human-MN tau."""
    return json.loads((B/'sharples_reference_codex.json').read_text(encoding='utf8'))
