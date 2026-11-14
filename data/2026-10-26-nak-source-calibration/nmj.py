"""[codex] Downstream NMJ prototype. Input is propagated AP times, not soma counts.

Expected vesicle release, residual terminal-Ca proxy, finite releasable pool,
ATP-dependent replenishment, ACh conductance and a passive muscle endplate.
Threshold crossings are a readout, not a muscle action-potential/force model.
No NMJ cost is silently debited from the neuronal ATP pool (terminal volume
and transport ownership are not yet known); its demand is reported separately.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.integrate import solve_ivp

@dataclass(frozen=True)
class NMJParameters:
    terminal_Ca_increment_uM:float=.5
    terminal_Ca_decay_s:float=.05
    release_half_uM:float=.5
    release_hill:float=3.
    max_release_probability:float=.5
    releasable_vesicles:float=100.
    replenishment_s:float=1.
    ATP_half_uM:float=300.
    ACh_conductance_decay_s:float=.002
    conductance_per_vesicle_nS:float=2.
    endplate_capacitance_pF:float=100.
    endplate_leak_nS:float=5.
    resting_mV:float=-80.
    ACh_reversal_mV:float=0.
    threshold_mV:float=-55.
    ATP_per_recycled_vesicle:float=100.
    provenance:str='NMJ mechanism/context Tu2024 and Altman2021; all numeric terminal/release/endplate/ATP quantities are uncalibrated scenarios; no muscle force or full muscle AP'

NMJ_RANGES=dict(terminal_Ca_increment_uM=[.1,.5,2.],terminal_Ca_decay_s=[.02,.05,.2],release_half_uM=[.2,.5,2.],release_hill=[2.,3.,4.],max_release_probability=[.1,.5,.9],releasable_vesicles=[20.,100.,300.],replenishment_s=[.1,1.,10.],ATP_half_uM=[50.,300.,1000.],ACh_conductance_decay_s=[.001,.002,.01],conductance_per_vesicle_nS=[.2,2.,10.],endplate_capacitance_pF=[50.,100.,500.],endplate_leak_nS=[1.,5.,20.],resting_mV=[-90.,-80.,-70.],threshold_mV=[-60.,-55.,-40.],ATP_per_recycled_vesicle=[0.,100.,1000.])

def simulate_nmj(spikes_s, time_s, ATP_uM, *, parameters=None,checkpoint=None,step_s=.0002):
    p=parameters or NMJParameters();d=asdict(p)
    for k,v in d.items():
        if isinstance(v,float) and not np.isfinite(v):raise ValueError(k)
    if min(v for k,v in d.items() if isinstance(v,float) and k not in ('resting_mV','ACh_reversal_mV','threshold_mV','ATP_per_recycled_vesicle'))<=0 or p.max_release_probability>1 or p.ATP_per_recycled_vesicle<0:raise ValueError('Positive NMJ kinetic parameters required')
    if not p.resting_mV<p.threshold_mV<p.ACh_reversal_mV:raise ValueError('Endplate voltage order')
    t=np.asarray(time_s,float);atp=np.asarray(ATP_uM,float);sp=np.asarray(spikes_s,float)
    if t.ndim!=1 or atp.shape!=t.shape or len(t)<2 or not np.isfinite(t).all() or not np.isfinite(atp).all() or np.min(atp)<0 or np.any(np.diff(t)<=0):raise ValueError('Finite ordered ATP trace required')
    if not np.isfinite(sp).all() or np.any(np.diff(sp)<=0) or np.any(sp<t[0]) or np.any(sp>=t[-1]):raise ValueError('Unique spikes in half-open trace interval required')
    if step_s<=0:raise ValueError('Positive step')
    # c, available fraction, gACh, endplateVm, cumreleased, cumreplenished, cumATP
    y=np.array([0.,1.,0.,p.resting_mV,0.,0.,0.])
    if checkpoint:
        if checkpoint['time_s']!=t[0] or checkpoint['parameters']!=d:raise ValueError('NMJ checkpoint differs')
        y=np.asarray(checkpoint['state'],float)
        if y.shape!=(7,) or not np.isfinite(y).all() or np.min(y[[0,1,2,4,5,6]])<0 or y[1]>1:raise ValueError('Invalid NMJ checkpoint')
    def rhs(x,z):
        a=np.interp(x,t,atp);repl=(1-z[1])/p.replenishment_s*a/(p.ATP_half_uM+a)
        return [-z[0]/p.terminal_Ca_decay_s,repl,-z[2]/p.ACh_conductance_decay_s,
          1000*(p.endplate_leak_nS*(p.resting_mV-z[3])+z[2]*(p.ACh_reversal_mV-z[3]))/p.endplate_capacitance_pF,
          0.,repl*p.releasable_vesicles,repl*p.releasable_vesicles*p.ATP_per_recycled_vesicle]
    times=[t[0]];states=[y.copy()];events=[];last=t[0]
    for end in np.r_[sp,t[-1]]:
        if end>last:
            grid=np.arange(last+step_s,end,step_s);grid=grid[(grid>last)&(grid<end)]
            sol=solve_ivp(rhs,(last,end),y,rtol=1e-8,atol=1e-10,max_step=step_s,t_eval=np.r_[grid,end])
            if not sol.success:raise RuntimeError(sol.message)
            times.extend(sol.t);states.extend(sol.y.T);y=sol.y[:,-1]
        if end<t[-1]:
            y=y.copy();y[0]+=p.terminal_Ca_increment_uM
            pr=p.max_release_probability*y[0]**p.release_hill/(p.release_half_uM**p.release_hill+y[0]**p.release_hill)
            released=p.releasable_vesicles*y[1]*pr;y[1]-=released/p.releasable_vesicles;y[2]+=released*p.conductance_per_vesicle_nS;y[4]+=released
            times.append(end);states.append(y.copy());events.append(dict(time_s=float(end),expected_quanta=float(released)))
        last=end
    z=np.array(states);tt=np.array(times)
    balance=float(np.max(abs(p.releasable_vesicles*z[:,1]+z[:,4]-z[:,5]-p.releasable_vesicles)))
    if balance>1e-6 or np.min(z[:,1])<0 or np.max(z[:,1])>1+1e-9:raise ArithmeticError('Vesicle balance')
    return dict(time_s=tt,state=z,events=events,columns=['terminal_Ca_proxy_uM','vesicle_available_fraction','ACh_conductance_nS','endplate_mV','released_quanta','replenished_quanta','recycling_ATP_molecules'],
       endplate_threshold_crossings_s=tt[1:][(z[:-1,3]<p.threshold_mV)&(z[1:,3]>=p.threshold_mV)],vesicle_balance=balance,
       checkpoint=dict(time_s=float(t[-1]),state=y.tolist(),parameters=d),human_kinetic_fit=False,
       limitations='Downstream expected-release module; no NMJ-to-neuron ATP debit, muscle AP/force, network or calibrated terminal Ca. Counts of endplate crossings are NOT validated muscle successes.')
