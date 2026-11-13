"""Conditional ADAR2/AMPAR model. No human fit; fixed voltage, fixed receptor count.

RNA editing affects NEW protein; surface state changes only through turnover.
Pf is fractional calcium charge, never P_Ca/P_Na. See evidence/report for limits.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from numba import njit

@dataclass(frozen=True)
class AMPAParameters:
    normal_editing: float = .999
    tau_rna_h: float = 6.
    tau_surface_h: float = 12.
    incorporation_R_over_Q: float = 1.
    weights: tuple = (.02, 0., .98, 0., 0.)
    g_reference_nS: float = 10.
    cp_conductance_ratio: float = 3.
    cp_unblocked_fraction: float = 1.
    pf_cp: float = .035
    pf_ci: float = .001
    voltage_mV: float = -60.
    reversal_mV: float = 0.
    volume_um3: float = 1000.
    kappa: float = 50.
    gamma_per_s: float = 5.1
    basal_ca_uM: float = .08
    opening_per_s: float = 1000.
    closing_per_s: float = 200.
    desensitization_per_s: float = 100.
    recovery_per_s: float = 10.

def _validate(p):
    vals = [v for k,v in asdict(p).items() if k != 'weights']
    if not np.isfinite(vals).all(): raise ValueError('Finite parameters required')
    if not 0 < p.normal_editing < 1: raise ValueError('normal_editing strictly between 0 and 1')
    for k in ('tau_rna_h','tau_surface_h','incorporation_R_over_Q','volume_um3','gamma_per_s',
              'opening_per_s','closing_per_s','desensitization_per_s','recovery_per_s'):
        if getattr(p,k) <= 0: raise ValueError(k+' must be positive')
    for k in ('g_reference_nS','cp_conductance_ratio','kappa','basal_ca_uM'):
        if getattr(p,k) < 0: raise ValueError(k+' must be nonnegative')
    if not p.voltage_mV < p.reversal_mV: raise ValueError('Inward clamp only')
    for k in ('pf_cp','pf_ci','cp_unblocked_fraction'):
        if not 0 <= getattr(p,k) <= 1: raise ValueError(k+' outside [0,1]')
    cp_fraction(.5,p.weights)

def editing_target(activity, normal_editing=.999, family='exposure', hill=1.):
    """Two uncalibrated monotone maps; activity is functional, not ADAR2 RNA."""
    if not (np.isfinite([activity,normal_editing,hill]).all() and 0 <= activity <= 1
            and 0 < normal_editing < 1 and hill > 0): raise ValueError('Invalid editing map')
    if family == 'exposure': return float(-np.expm1(activity**hill*np.log1p(-normal_editing)))
    if family == 'hill': return float(activity**hill/(activity**hill+(1-normal_editing)/normal_editing))
    raise ValueError('Unknown mapping family')

def editing_trajectory(activity, p=AMPAParameters(), *, hours=48., family='exposure', hill=1.,
                       forced_target=None, initial_rna=None, initial_surface=None):
    """Two turnover pools. Surface r is edited fraction AMONG incorporated GluA2.

    Initial steady normal state unless explicit. No receptor is chemically re-edited.
    Forced target sets NEW RNA/protein editing (genetic-R bypass abstraction).
    """
    _validate(p)
    if not np.isfinite(hours) or hours < 0: raise ValueError('Nonnegative finite hours')
    target = editing_target(activity,p.normal_editing,family,hill) if forced_target is None else float(forced_target)
    m0 = p.normal_editing if initial_rna is None else initial_rna
    def incorporation(m):
        return p.incorporation_R_over_Q*m/(p.incorporation_R_over_Q*m+1-m)
    r0 = incorporation(m0) if initial_surface is None else initial_surface
    if not np.isfinite([target,m0,r0]).all() or not all(0<=x<=1 for x in (target,m0,r0)):
        raise ValueError('Editing states outside [0,1]')
    t = np.linspace(0,hours,241) if hours else np.array([0.])
    if hours:
        z = solve_ivp(lambda t,y: [(target-y[0])/p.tau_rna_h,
                     (incorporation(y[0])-y[1])/p.tau_surface_h],(0,hours),[m0,r0],
                     t_eval=t,rtol=1e-9,atol=1e-11).y
    else: z = np.array([[m0],[r0]])
    return dict(time_h=t,rna_editing=z[0],surface_R=z[1],target_editing=target,
                steady_surface_R=incorporation(target),surface_R_at_stimulus=float(z[1,-1]))

def cp_fraction(surface_R, weights=(.02,0,.98,0,0)):
    """Conditional independent sampling, any R dominates; n=0..4 GluA2 subunits."""
    w=np.asarray(weights,float)
    if w.shape != (5,) or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1.,atol=1e-12,rtol=0):
        raise ValueError('Five nonnegative weights must sum to one')
    if not np.isfinite(surface_R) or not 0<=surface_R<=1: raise ValueError('Invalid surface editing')
    return float(np.dot(w,(1-surface_R)**np.arange(5)))

@njit(cache=False)
def _propagate(matrices, index):
    z=np.zeros((len(index)+1,6));z[0,0]=1.
    for i in range(len(index)):
        z[i+1]=matrices[index[i]] @ z[i]
    return z

def ampa_calcium(surface_R, p=AMPAParameters(), *, frequency_Hz=10., pulse_s=.005,
                 duration_s=10., drive=1., dt_s=.001, end_s=21600., cp_block=0.):
    """Identical square glutamate drive; C/O/D gating; exact linear segment flow.

    Buffer/removal: (1+kappa) dx/dt = J_Ca - gamma*x. Start at rest, 1s onset.
    Integral J and integral excess Ca are propagated as states, not trapezoids.
    Population is frozen during this short stimulus after receptor preconditioning.
    """
    _validate(p)
    if not np.isfinite([frequency_Hz,pulse_s,duration_s,drive,dt_s,end_s,cp_block]).all(): raise ValueError('Finite protocol')
    if not (frequency_Hz>0 and 0<pulse_s<1/frequency_Hz and duration_s>0 and drive>=0
            and 0<dt_s<=.01 and end_s>duration_s+2 and 0<=cp_block<=1): raise ValueError('Invalid protocol')
    starts=1.+np.arange(int(np.floor(duration_s*frequency_Hz+1e-9)))/frequency_Hz
    ends=starts+pulse_s
    fine_end=duration_s+2.
    grids=[np.arange(0,fine_end,dt_s),starts,ends,[fine_end,end_s]]
    if end_s>fine_end: grids.append(np.arange(fine_end,min(60.,end_s),.05))
    if end_s>60: grids.append(np.arange(60.,min(900.,end_s),.5))
    if end_s>900: grids.append(np.arange(900.,end_s,10.))
    t=np.unique(np.round(np.concatenate(grids),10));t=t[(t>=0)&(t<=end_s)]
    mid=(t[:-1]+t[1:])/2
    index_pulse=np.searchsorted(starts,mid,side='right')-1
    u=np.zeros(len(mid));ok=index_pulse>=0
    u[ok]=drive*(mid[ok]<ends[index_pulse[ok]])
    pc=cp_fraction(surface_R,p.weights)
    inward_ci=p.g_reference_nS*(p.reversal_mV-p.voltage_mV)*(1-pc)
    inward_cp=p.g_reference_nS*(p.reversal_mV-p.voltage_mV)*pc*p.cp_conductance_ratio*p.cp_unblocked_fraction*(1-cp_block)
    q=(inward_ci*p.pf_ci+inward_cp*p.pf_cp)*1e9/(2*96485.33212*p.volume_um3)
    # Use unit q; linear calcium response scales exactly for each receptor mixture.
    pairs,idx=np.unique(np.column_stack([u,np.round(np.diff(t),10)]),axis=0,return_inverse=True)
    matrices=[]
    for level,step in pairs:
        a=p.opening_per_s*level;b=p.closing_per_s;d=p.desensitization_per_s;r=p.recovery_per_s
        M=np.zeros((6,6))
        M[:3,:3]=[[-a,b,r],[a,-b-d,0],[0,d,-r]]
        M[3,1]=1/(1+p.kappa);M[3,3]=-p.gamma_per_s/(1+p.kappa)
        M[4,1]=1.;M[5,3]=1.
        matrices.append(expm(M*step))
    z=_propagate(np.array(matrices),idx)
    x=z[:,3]*q;cum=z[:,4]*q;auc=z[:,5]*q
    balance=cum-p.gamma_per_s*auc-(1+p.kappa)*x
    return dict(time_s=t,ca_uM=p.basal_ca_uM+x,excess_ca_uM=x,open_fraction=z[:,1],
        inward_ci_pA=z[:,1]*inward_ci,inward_cp_pA=z[:,1]*inward_cp,
        total_inward_pA=z[:,1]*(inward_ci+inward_cp),ca_flux_uM_per_s=q*z[:,1],
        ca_entry_cumulative_uM=cum,ca_auc_uM_s=auc,cp_fraction=pc,
        calcium_balance_max_error=float(np.max(abs(balance))),
        gating_sum_error=float(np.max(abs(z[:,:3].sum(axis=1)-1))),
        gating_min=float(z[:,:3].min()),interval_drive=u,
        protocol=dict(frequency_Hz=frequency_Hz,pulse_s=pulse_s,duration_s=duration_s,drive=drive,
                      voltage_mV=p.voltage_mV,pulse_count=len(starts)),
        interpretation='Fixed-voltage conditional scenario; no electrical or NMJ feedback')

def paired_chain(api, p, biop, *, activity=.1, family='exposure', hill=1., hours=48.,
                 forced_target=None, cp_block=0., inhibition=0., dt_s=.001, max_step_s=2., drive=1.):
    edit=editing_trajectory(activity,p,hours=hours,family=family,hill=hill,forced_target=forced_target)
    ca=ampa_calcium(edit['surface_R_at_stimulus'],p,cp_block=cp_block,dt_s=dt_s,drive=drive)
    baseline=api.baseline_equilibrium(p.basal_ca_uM,biop)
    bio=api.simulate_calpain_tdp(ca['time_s'],ca['ca_uM'],parameters=biop,
        ca_provenance='ADAR2 scenario, assumed receptor/calcium parameters; not a human fit',
        compartment='well-mixed accessible cytosolic scenario',baseline=baseline,
        inhibition=inhibition,max_step_s=max_step_s)
    return dict(editing=edit,calcium=ca,biochemistry=bio)

def chain_summary(result):
    e=result['editing'];c=result['calcium'];b=result['biochemistry']
    return dict(target_editing=e['target_editing'],surface_R=e['surface_R_at_stimulus'],
        cp_fraction=c['cp_fraction'],peak_Ca_uM=float(c['ca_uM'].max()),
        Ca_entry_uM=float(c['ca_entry_cumulative_uM'][-1]),Ca_auc_uM_s=float(c['ca_auc_uM_s'][-1]),
        peak_inward_pA=float(c['total_inward_pA'].max()),
        peak_effective_calpain=float(b['effective_activity'].max()),
        peak_large_TDP=float(b['large_N_fragment'].max()),
        large_TDP_peak_increment=float(b['large_N_fragment'].max()-b['large_N_fragment'][0]),
        large_TDP_end_change=float(b['large_N_fragment'][-1]-b['large_N_fragment'][0]),
        large_TDP_end=float(b['large_N_fragment'][-1]),
        intact_TDP_min=float(b['intact_TDP'].min()),
        formed_large_total=float(b['formed_large_cumulative'][-1]),
        secondary_cleavage_total=float(b['secondary_cleavage_cumulative'][-1]),
        removed_large_total=float(b['large_removed_cumulative'][-1]),
        calcium_balance_error=c['calcium_balance_max_error'],lineage_balance_error=b['lineage_balance_max_error'])
