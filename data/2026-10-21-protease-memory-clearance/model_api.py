"""[codex] 2026-10-21. Optional calpain processing, aggregate and assay modules.

Inherits Chan/STMN2 and all electrical APIs. No human parameter was refitted.
Autolysis is distinct from reversible calcium-dependent catalytic competence.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util, sys, json, math
import numpy as np
from numba import njit

_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_before_protease_memory20261021',_HERE.parent/'2026-10-21-chan-stmn2-neurite/model_api.py')
_previous=importlib.util.module_from_spec(_spec); sys.modules[_spec.name]=_previous
_spec.loader.exec_module(_previous)
for _name in dir(_previous):
    if not _name.startswith('_') and _name not in globals(): globals()[_name]=getattr(_previous,_name)


def _source(s):
    if not isinstance(s,str) or not s.strip(): raise ValueError('Explicit provenance/assumption required')


def _series(time_s,values,label):
    t=np.asarray(time_s,float); x=np.asarray(values,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Strictly ordered finite times in seconds required')
    if x.shape!=t.shape or not np.isfinite(x).all() or np.any(x<0):
        raise ValueError(label+' must be finite, nonnegative and match time')
    return t,x


@njit
def _gate(c,k,n):
    if c<=0: return 0.
    z=n*(math.log(c)-math.log(k))
    if z>=0: return 1/(1+math.exp(-z))
    e=math.exp(z); return e/(1+e)


@dataclass(frozen=True)
class CalpainPoolParameters:
    native_ca_half_uM: float
    processed_ca_half_uM: float
    hill_n: float
    autolysis_per_s: float
    cast_binding_per_s: float
    cast_unbinding_per_s: float
    turnover_per_s: float
    synthesis_per_s: float
    provenance: str

    def vector(self):
        _source(self.provenance)
        v=np.array([v for k,v in asdict(self).items() if k!='provenance'],float)
        if not np.isfinite(v).all() or np.any(v[:3]<=0) or np.any(v[3:]<0):
            raise ValueError('Positive Ca/Hill and nonnegative finite rates required')
        return v


@njit
def _pool_rhs(y,c,p):
    # U,A: free native/processed enzyme; BU,BA: CAST-bound enzyme.
    # CAST is a maintained reservoir; binding is pseudo-first-order, not Ki.
    gu,ga=_gate(c,p[0],p[2]),_gate(c,p[1],p[2])
    auto=p[3]*gu*y[0]
    bu=p[4]*gu*y[0]-p[5]*y[2]
    ba=p[4]*ga*y[1]-p[5]*y[3]
    d=p[6]
    return np.array([p[7]-auto-bu-d*y[0], auto-ba-d*y[1],
                     bu-d*y[2],ba-d*y[3],auto,d*np.sum(y[:4]),p[7]])


@njit
def _pool_integrate(t,c,y0,p,max_step):
    out=np.empty((len(t),7)); out[0]=y0
    step=min(max_step,.1/max(1e-12,p[3]+p[4]+p[6],p[5]+p[6]))
    for i in range(len(t)-1):
        n=max(1,int(math.ceil((t[i+1]-t[i])/step)))
        dt=(t[i+1]-t[i])/n; y=out[i].copy()
        for j in range(n):
            c0=c[i]+(c[i+1]-c[i])*j/n
            cm=c[i]+(c[i+1]-c[i])*(j+.5)/n
            c1=c[i]+(c[i+1]-c[i])*(j+1)/n
            a=_pool_rhs(y,c0,p); b=_pool_rhs(y+dt*a/2,cm,p)
            d=_pool_rhs(y+dt*b/2,cm,p); e=_pool_rhs(y+dt*d,c1,p)
            y=y+dt*(a+2*b+2*d+e)/6
        out[i+1]=y
    return out


def simulate_calpain_pools(time_s,ca_uM,*,parameters,initial_pools,
                          ca_provenance,compartment,max_step_s=.02):
    """Conditional enzyme pool history, not an isoform-specific human Ca fit.

    Free Ca is linearly interpolated. Catalytic gating is assumed instantaneous;
    processed enzyme still needs Ca. No reverse autolysis. CAST-bound enzyme is
    inactive and not autolysed in this approximation. Four initial pools required;
    their sum is in arbitrary common enzyme units, not a fraction by necessity.
    """
    if not isinstance(parameters,CalpainPoolParameters): raise ValueError('Explicit parameters required')
    p=parameters.vector(); t,c=_series(time_s,ca_uM,'Free Ca (uM)')
    _source(ca_provenance); _source(compartment)
    y0=np.asarray(initial_pools,float)
    if y0.shape!=(4,) or not np.isfinite(y0).all() or np.any(y0<0): raise ValueError('Four nonnegative pools required')
    if not np.isfinite(max_step_s) or max_step_s<=0: raise ValueError('Positive integration step required')
    y=_pool_integrate(t,c,np.r_[y0,0.,0.,0.],p,max_step_s)
    balance=y[:,:4].sum(axis=1)+y[:,5]-y[:,6]
    error=float(np.max(abs(balance-balance[0])))
    if not np.isfinite(y).all() or np.min(y)<-1e-9 or error>1e-8*max(1.,float(balance[0])):
        raise ArithmeticError('Enzyme pool positivity/accounting failed')
    gu=np.array([_gate(v,p[0],p[2]) for v in c]); ga=np.array([_gate(v,p[1],p[2]) for v in c])
    r={k:y[:,i] for i,k in enumerate(('native_free','processed_free','native_CAST_bound',
        'processed_CAST_bound','autolysed_cumulative','enzyme_removed','enzyme_synthesized'))}
    r.update(time_s=t,ca_uM=c,effective_activity=gu*y[:,0]+ga*y[:,1],
        pool_balance_max_error=error,parameters=asdict(parameters),ca_provenance=ca_provenance,
        compartment=compartment,human_kinetic_fit=False,electrical_feedback=False,
        status='Conditional pool model; rapid Ca gating; CAST reservoir assumption')
    return r


@njit
def _tdp_rhs(y,a,s,p):
    q1=p[5]*a*y[0]; q2=p[6]*a*y[1]
    dp=p[7]*y[0]; df=p[8]*y[1]; dr=p[9]*y[2]
    return np.array([s-dp-q1,q1-q2-df,q2-dr,dp+df+dr,q1,q2,df,s])


@njit
def _tdp_integrate(t,a,y0,s,p,step):
    out=np.empty((len(t),8)); out[0]=y0
    safe=min(step,.1/max(1e-12,p[5]*np.max(a)+p[7],p[6]*np.max(a)+p[8],p[9]))
    for i in range(len(t)-1):
        n=max(1,int(math.ceil((t[i+1]-t[i])/safe))); h=(t[i+1]-t[i])/n; y=out[i].copy()
        for j in range(n):
            a0=a[i]+(a[i+1]-a[i])*j/n; am=a[i]+(a[i+1]-a[i])*(j+.5)/n; a1=a[i]+(a[i+1]-a[i])*(j+1)/n
            k1=_tdp_rhs(y,a0,s,p); k2=_tdp_rhs(y+h*k1/2,am,s,p)
            k3=_tdp_rhs(y+h*k2/2,am,s,p); k4=_tdp_rhs(y+h*k3,a1,s,p)
            y=y+h*(k1+2*k2+2*k3+k4)/6
        out[i+1]=y
    return out


def simulate_tdp_from_activity(time_s,effective_activity,*,parameters,initial_TDP,
                              synthesis_per_s,activity_provenance,max_step_s=.02):
    """Same N-lineage topology as old model, driven by explicit activity.

    initial_TDP=[intact,large N-product,small N-product]. Old parameters' Ca,
    relaxation and CAST entries are not used here: activity already includes CAST.
    No C-terminal band, phospho epitope, RNA function or viability is inferred.
    """
    _source(activity_provenance); p=parameters.vector()
    t,a=_series(time_s,effective_activity,'Effective enzyme activity')
    y0=np.asarray(initial_TDP,float); s=float(synthesis_per_s)
    if y0.shape!=(3,) or not np.isfinite(y0).all() or np.any(y0<0) or not np.isfinite(s) or s<0:
        raise ValueError('Three nonnegative initial pools and finite nonnegative synthesis required')
    if not np.isfinite(max_step_s) or max_step_s<=0: raise ValueError('Positive step required')
    y=_tdp_integrate(t,a,np.r_[y0,np.zeros(5)],s,p,max_step_s)
    bal=y[:,:4].sum(axis=1)-y[:,7]; err=float(np.max(abs(bal-bal[0])))
    if not np.isfinite(y).all() or y.min()<-1e-9 or err>1e-8*max(1.,float(bal[0])):
        raise ArithmeticError('N-lineage accounting failed')
    r={k:y[:,i] for i,k in enumerate(('intact_TDP','large_N_fragment','small_N_product',
        'removed_lineages','formed_large_cumulative','secondary_cleavage_cumulative',
        'large_removed_cumulative','synthesized_cumulative'))}
    r.update(time_s=t,effective_activity=a,formation_rate_per_s=p[5]*a*y[:,0],
        lineage_balance_max_error=err,activity_provenance=activity_provenance,
        parameters=asdict(parameters),human_kinetic_fit=False,electrical_feedback=False)
    return r


def aggregate_pulse_chase(time_s,*,fast_rate_per_s,slow_fraction,slow_rate_per_s,
                         provenance,assay):
    """Two non-interconverting aggregate pools, no production after pulse.

    A zero slow rate is a finite-window plateau approximation, not proof of an
    immortal biological pool. The fast component half-time differs from total
    half-clearance. This observer is not used to clear endogenous intact TDP.
    """
    _source(provenance); _source(assay)
    t=np.asarray(time_s,float)
    v=np.array([fast_rate_per_s,slow_fraction,slow_rate_per_s],float)
    if t.ndim!=1 or len(t)==0 or not np.isfinite(t).all() or np.any(t<0) or np.any(np.diff(t)<=0):
        raise ValueError('Ordered nonnegative pulse-chase times required')
    if not np.isfinite(v).all() or fast_rate_per_s<=0 or slow_rate_per_s<0 or not 0<=slow_fraction<=1:
        raise ValueError('Positive fast rate, nonnegative slow rate, fraction [0,1] required')
    fast=(1-slow_fraction)*np.exp(-fast_rate_per_s*t); slow=slow_fraction*np.exp(-slow_rate_per_s*t)
    return dict(time_s=t,fast=fast,slow=slow,total=fast+slow,removed=1-fast-slow,
        fast_component_half_time_s=math.log(2)/fast_rate_per_s,assay=assay,provenance=provenance,
        endogenous_human_clearance_calibrated=False)


def observe_fractionated_tdp(pools,*,response_matrix,channel_labels,provenance):
    """Linear recovery/epitope observation model for separately defined pools.

    Explicit matrix is required: equal band intensity does not imply equal mass.
    Columns can distinguish N/C epitopes, phospho state and solubility. This
    readout never converts loss of a soluble band into a degradation rate.
    """
    _source(provenance); x=np.asarray(pools,float); m=np.asarray(response_matrix,float)
    if x.ndim!=2 or m.ndim!=2 or x.shape[1]!=m.shape[1] or len(channel_labels)!=m.shape[0]:
        raise ValueError('Pool time matrix and compatible explicit observation matrix required')
    if not np.isfinite(x).all() or not np.isfinite(m).all() or np.any(x<0) or np.any(m<0):
        raise ValueError('Nonnegative finite pools and response coefficients required')
    if len(set(channel_labels))!=len(channel_labels) or any(not isinstance(s,str) or not s.strip() for s in channel_labels):
        raise ValueError('Unique channel labels required')
    return dict(signals=x@m.T,channel_labels=list(channel_labels),provenance=provenance,
                inferred_cleavage_rate=None,inferred_degradation_rate=None)


def protease_memory_evidence():
    return json.loads((_HERE/'evidence.json').read_text(encoding='utf-8'))


def scotter_construct_chase_reference(time_s,*,cell_line,construct):
    """Normalized decaying component from a published apparent half-time.

    Time zero is the beginning of the analysed interval: 24 h after DOX washout
    for SH-SY5Y, washout for HEK293. Restrict to the source observation window.
    Tagged expressed constructs and whole-culture readout are not endogenous
    motor-neuron turnover. CTF is residues 181-414, not the core N-lineage product.
    Absolute amplitude/plateau were not extracted; a unit amplitude and zero
    plateau describe the normalized decaying component, not a full band fit.
    """
    e=protease_memory_evidence()['Scotter2014']
    if cell_line not in e['half_life_h'] or construct not in e['half_life_h'][cell_line]:
        raise ValueError('Choose HEK293 or SH-SY5Y and WT, deltaNLS or CTF181_414')
    t=np.asarray(time_s,float); window=e['analysis_window_h'][cell_line]
    if np.any(t>window*3600): raise ValueError('Outside source analysis window; no unlabelled extrapolation')
    hl=e['half_life_h'][cell_line][construct]
    r=aggregate_pulse_chase(t,fast_rate_per_s=math.log(2)/(hl*3600),slow_fraction=0,
        slow_rate_per_s=0,provenance='Scotter2014 Fig2 published apparent half-life; full XML verified',
        assay=f'Scotter2014 DOX HA-tagged {construct} {cell_line}')
    r.update(time_from_DOX_washout_s=t+e['analysis_start_after_washout_h'][cell_line]*3600,
             published_apparent_half_life_h=hl,raw_data_fit_performed=False,
             observable='normalized decaying component; full signal amplitude/plateau unknown')
    return r


def source_training_data():
    r=_previous.source_training_data()
    r['protease_processing_and_assay_domains']=protease_memory_evidence()
    return r
