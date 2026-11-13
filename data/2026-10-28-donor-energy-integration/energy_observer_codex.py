"""[codex] Le Masson-inspired ion-restoration observer; never an ATP debit.

Uses all represented net Na entry: channels (includes AMPA), NCX, background,
and axial amount transfer. Net entry/3 - actual NaK cycles = delta intracellular
Na/3. K restoration is shown independently; Na and K costs must NOT be summed.
Terminal NaK is not explicit in this model, so no invented terminal metric.
"""
import numpy as np

def energy_recovery_observer(result):
    z=np.asarray(result['state']);base=np.asarray(result['baseline_state'])
    g=result['config']['geometry']
    vp=np.pi*g['process_radius_um']**2*g['process_length_um']*g['process_accessible_fraction']
    volumes=np.array([g['soma_volume_um3'],vp])
    needed=[];actual=[];debt=[];kdebt=[];residual=[]
    for j,b in enumerate((199,219)):
        req=(-z[:,b+5]+z[:,b+7]+z[:,b+8]+z[:,b+12]/volumes[j])/3.
        used=z[:,b+4];d=req-used
        needed.append(req);actual.append(used);debt.append(d)
        kdebt.append((base[b+1]-z[:,b+1])/2.)
        residual.append(np.max(abs(d-(z[:,b]-base[b])/3.)))
    debt=np.array(debt).T
    return dict(time_s=result['time_s'],Na_restoration_needed_net_uM=np.array(needed).T,
        NaK_actual_ATP_uM=np.array(actual).T,Na_restoration_debt_signed_uM=debt,
        K_restoration_debt_signed_uM=np.array(kdebt).T,
        represented_Na_debt_amount_uM_um3=debt@volumes,
        represented_Na_debt_volume_mean_uM=debt@volumes/volumes.sum(),
        balance_error_uM=float(max(residual)),changes_dynamics=False,
        terminal_Na_debt=None,
        limitation='Represented ions only. Signed net demand, not gross Nav cost or a cell death threshold. K debt overlaps Na pump work; do not add them. Terminal Na not explicitly modeled. Not a Le Masson morphology replication.')

def voltage_recovery_observer(result,baseline_window,recovery_window):
    """Allen/IPFX-inspired baseline stability readout, not a source detector.

    Explicit time windows, time-weighted statistics for irregular solver output.
    No learned cutoff or automatic pass; spontaneous spikes affect the result.
    """
    t=np.asarray(result['time_s']);v=np.asarray(result['soma_voltage_mV'])
    t,ix=np.unique(t,return_index=True);v=v[ix]
    def stats(window):
        a,b=window
        if not t[0]<=a<b<=t[-1]:raise ValueError('Windows must be inside recording')
        q=np.r_[a,t[(t>a)&(t<b)],b];w=np.interp(q,t,v)
        mean=np.trapezoid(w,q)/(b-a);rms=np.sqrt(np.trapezoid((w-mean)**2,q)/(b-a))
        return dict(mean_mV=float(mean),rms_mV=float(rms))
    base=stats(baseline_window);tail=stats(recovery_window)
    return dict(baseline=base,recovery=tail,delta_mV=tail['mean_mV']-base['mean_mV'],
        baseline_window_s=baseline_window,recovery_window_s=recovery_window,
        automatic_pass=None,scope='Model waveform, not experimental recording noise or an independent validation target.')
