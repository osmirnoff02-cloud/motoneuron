"""[codex] 2026-10-28. Protocol-aware observations, not a human kinetic fit.

Atluri2006: acid perfusion quenches the surface AND new endocytic influx.
Tabares2007: optical membrane turnover does not identify functional quanta.
The isolated assay has prescribed exocytosis, not a recycling pool or ATP model.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

@dataclass(frozen=True)
class ReleaseObservationParameters:
    electrically_effective_fraction: float = 1.
    reporter_brightness: float = 1.
    provenance: str = 'Unidentified scenario; no human ACh loading measurement.'

    def validate(self):
        if not np.isfinite([self.electrically_effective_fraction,self.reporter_brightness]).all() or not 0 <= self.electrically_effective_fraction <= 1 or self.reporter_brightness <= 0 or not self.provenance:
            raise ValueError('Invalid release observation parameters')

OBSERVATION_RANGES = dict(electrically_effective_fraction=[0.,.5,1.],reporter_brightness=[.5,1.,2.])

def transmission_readouts(membrane_exocytosis, parameters=None):
    """Separate membrane traffic, electrically effective quanta and reporter.

    Effective fraction is phenomenological, NOT an identified empty-vesicle
    fraction or receptor defect. No postsynaptic feedback or muscle AP added.
    """
    p=parameters or ReleaseObservationParameters();p.validate()
    x=np.asarray(membrane_exocytosis,float)
    if not np.isfinite(x).all() or np.min(x)<-1e-6:raise ValueError('Invalid exocytosis')
    return dict(membrane_exocytosis=x.copy(),effective_quanta=p.electrically_effective_fraction*x,
                reporter_exocytosis=p.reporter_brightness*x,
                effective_to_reporter_ratio=p.electrically_effective_fraction/p.reporter_brightness,
                calibrated_human_release=False)

def acid_quench_assay(time_s, *, exocytosis_intervals, quench_intervals=(),
                      tau_end_s, tau_acid_s, acidification_support=1.,
                      initial_surface=0., initial_alkaline=0.):
    """S -> alkaline -> quenched reporter; explicit bath-acid intervention.

    During acid perfusion incoming reporter is already quenched. The physical
    trafficking flux continues. 'quenched' is an optical class, not readiness.
    Exocytosis intervals are (start,end,reporter-equivalents/s). Units arbitrary.
    Perfect surface quenching and constant first-order rates are assumptions.
    Not a whole-neuron acid-bath prediction, pH simulation, or ATP-cost fit.
    """
    t=np.asarray(time_s,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Strictly increasing finite time required')
    if not np.isfinite([tau_end_s,tau_acid_s,acidification_support,initial_surface,initial_alkaline]).all() or min(tau_end_s,tau_acid_s)<=0 or min(initial_surface,initial_alkaline)<0 or not 0<=acidification_support<=1:raise ValueError('Invalid assay rates/state')
    ex=[tuple(map(float,x)) for x in exocytosis_intervals];qq=[tuple(map(float,x)) for x in quench_intervals]
    for a,b,j in ex:
        if not np.isfinite([a,b,j]).all() or b<=a or j<0:raise ValueError('Invalid exocytosis interval')
    for a,b in qq:
        if not np.isfinite([a,b]).all() or b<=a:raise ValueError('Invalid quench interval')
    edges=sorted(set([float(t[0]),float(t[-1])]+[a for x in ex+qq for a in x[:2] if t[0]<a<t[-1]]))
    z=np.zeros((len(t),4));y=np.array([initial_surface,initial_alkaline,0.,0.]);z[0]=y
    for left,right in zip(edges[:-1],edges[1:]):
        mid=(left+right)/2;acid=any(a<=mid<b for a,b in qq);exo=sum(j for a,b,j in ex if a<=mid<b)
        def rhs(_,v):
            je=v[0]/tau_end_s;ja=acidification_support*v[1]/tau_acid_s
            return [exo-je,(0. if acid else je)-ja,ja+(je if acid else 0.),exo]
        sol=solve_ivp(rhs,(left,right),y,rtol=1e-10,atol=1e-12,dense_output=True)
        if not sol.success:raise ArithmeticError(sol.message)
        mask=(t>left)&(t<=right);z[mask]=sol.sol(t[mask]).T;y=sol.y[:,-1]
    acid=np.array([any(a<=u<b for a,b in qq) for u in t]);surface=z[:,0];alkaline=z[:,1]
    balance=np.max(abs(z[:,:3].sum(1)-z[:,3]-initial_surface-initial_alkaline))
    if balance>1e-7 or z.min() < -1e-8:raise ArithmeticError('Reporter balance failed')
    return dict(time_s=t,surface=surface,alkaline=alkaline,quenched=z[:,2],cumulative_exocytosis=z[:,3],
                fluorescence=np.where(acid,0.,surface)+alkaline,acid_bath=acid,
                reporter_balance=float(balance),full_neuron_feedback=False)

def annotate_synaptic_result(result, parameters=None):
    """Attach observables to API0.19 states without changing their dynamics."""
    if not result['adaptive_parameters']['enabled']:
        raise ValueError('Separated acidification/priming must be explicitly enabled')
    z=result['state'];out=transmission_readouts(z[:,279],parameters)
    p=parameters or ReleaseObservationParameters()
    out.update(phluorin=p.reporter_brightness*(z[:,296]+z[:,297]),
               instantaneous_surface_quenched=p.reporter_brightness*z[:,297],
               membrane_ready_fraction=z[:,276].copy(),
               warning='Instantaneous quenched projection is NOT a finite acid-perfusion trace. Effective quanta are a scenario, not fitted ACh output; inherited EPP assumes all released model quanta are functional.',
               finite_quench_protocol='Use acid_quench_assay; no acid-bath perturbation of the whole neuron is implemented.')
    return out

def atluri_reference_parameters():
    """Explicit source reference; never sets a human default or temperature law."""
    return dict(species='rat',cell='hippocampal culture',temperature_C=25.,
                stimulus_AP=300,stimulus_Hz=10.,tau_acid_s=4.4,SEM_s=.4,n=12,
                source='10.1523/JNEUROSCI.4425-05.2006 Fig2/Methods',human_calibration=False)
