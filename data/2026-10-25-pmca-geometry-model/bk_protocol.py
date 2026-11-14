"""[codex] Source-specific BK protocol observer, not a human membrane channel.
Berkefeld2013: Ca-step at constant voltage, sub-ms exchange, 2 SD noise onset.
No recorded noise or original digital trace is available: thresholds are inputs.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'2026-10-25-pmca-memory-model'))
from bk_observer import bk_generator,stationary

def bk_source_protocol(ca_uM,*,voltage_mV=50.,temperature_C=23.,exchange_tau_s=.0003,
                       noise_sd_fraction=.005,dt_s=.00005):
    if not np.isfinite([ca_uM,voltage_mV,temperature_C,exchange_tau_s,noise_sd_fraction,dt_s]).all() or ca_uM<=0 or exchange_tau_s<0 or dt_s<=0 or not 0<noise_sd_fraction<.1:
        raise ValueError('Invalid source protocol or assumed noise')
    z=stationary(bk_generator(0.,voltage_mV,temperature_C));base=z[5:].sum()
    plateau=stationary(bk_generator(ca_uM,voltage_mV,temperature_C))[5:].sum()
    def ca(t):return ca_uM if exchange_tau_s==0 else ca_uM*(-np.expm1(-t/exchange_tau_s))
    t=np.linspace(0,.1,int(round(.1/dt_s))+1)
    sol=solve_ivp(lambda t,z:bk_generator(ca(t),voltage_mV,temperature_C)@z,(0,.1),z,
                  t_eval=t,method='BDF',rtol=1e-10,atol=1e-12,max_step=.0005)
    if not sol.success:raise ArithmeticError(sol.message)
    prob=sol.y[5:].sum(0)
    # The experiment used measured baseline SD. Here SD is an explicit sensitivity assumption.
    threshold=base+2*noise_sd_fraction*(plateau-base)
    cross=np.flatnonzero(prob>=threshold)
    if not len(cross):raise ArithmeticError('No threshold crossing')
    index=int(cross[0]);delay=float(t[index]);ix=t>=delay
    def rise(t,a,tau,offset):return offset+a*(1-np.exp(-t/tau))
    fit,_=curve_fit(rise,t[ix]-delay,prob[ix],p0=[plateau-base,.01,base],
        bounds=([-np.inf,1e-6,-np.inf],[np.inf,1.,np.inf]),maxfev=20000)
    error=float(max(abs(sol.y.sum(0)-1)))
    if error>1e-8 or sol.y.min()<-1e-9:raise ArithmeticError('BK probability balance failed')
    return dict(time_s=t,open_probability=prob,calcium_uM=np.array([ca(x) for x in t]),
        delay_ms=delay*1000,rise_tau_ms=float(fit[1]*1000),noise_sd_fraction=noise_sd_fraction,
        exchange_tau_s=exchange_tau_s,probability_balance=error,voltage_mV=voltage_mV,
        limitation='Threshold/noise and fit window declared; no original trace, noise, channel composition or human MN calibration.')
