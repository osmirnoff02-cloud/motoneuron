"""Codex 2026-10-16. Additive observation/NMDA branch around preserved electrical models.

No automatic animal->human kinetic transfer, no new Ca->SK conductance.
Time units for new APIs are seconds; old electrical functions retain milliseconds.
"""
from pathlib import Path
from dataclasses import dataclass
import json,sys,importlib.util
import numpy as np
from scipy.integrate import cumulative_trapezoid
from cascade_observer import kernel_predict
from dandrea_fit import pulse_response
B=Path(__file__).resolve().parent

def _load(name,path):
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def _times(t):
    t=np.asarray(t,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Finite increasing 1D times required')
    return t

def observe_spikes(spikes_s,bin_edges_s,*,dataset):
    """Source-specific arbitrary-unit fluorescence from AP times, NOT micromolar Ca.

    Uses the same bin convention as held-out validation. Spikes must all be inside
    the requested window; otherwise clipping would silently lose contributions.
    """
    edges=_times(bin_edges_s);dt=np.diff(edges)
    if not np.allclose(dt,dt[0],rtol=1e-6,atol=1e-9):raise ValueError('Uniform bins required')
    sp=np.asarray(spikes_s,float)
    if sp.ndim!=1 or not np.isfinite(sp).all() or np.any(np.diff(sp)<0) or np.any(sp<edges[0]) or np.any(sp>=edges[-1]):raise ValueError('Ordered in-window spikes required')
    doc=json.loads((B/'cascade-observer-profile.json').read_text());p=doc['profiles'][dataset]
    counts,_=np.histogram(sp,edges)
    signal=kernel_predict(counts,float(dt[0]),p['rise_s'],p['decay_s'])
    return dict(t_s=(edges[:-1]+edges[1:])/2,signal_au=signal,bin_counts=counts,profile=p,dataset=dataset,
                status='mouse technical observer applied as an explicit scenario; not calibrated human calcium')

def nmda_fura2_response(t_s,*,external_mg_mM,roi=5,treatment='control',onset_s=None):
    """Replays the observed 3 s NMDA protocol in 0 Mg for the reporter-culture profile.

    Cell-specific first-control amplitudes/kinetics. Drug is a fitted amplitude
    scenario from the median of five same-field ROI ratios, not a predicted drug
    mechanism. Other concentrations/durations/species are unsupported.
    """
    t=_times(t_s)
    if external_mg_mM!=0:raise ValueError('This 0-Mg response cannot be transferred to 2mM Mg; paired drug/wash check does not reproduce strong suppression there')
    if treatment not in ('control','EU1794-4'):raise ValueError('Unknown treatment')
    doc=json.loads((B/'dandrea-response-profile.json').read_text())
    p=next((r for r in doc['profiles'] if r['field']=='2380205D' and r['roi']==roi),None)
    if p is None:raise ValueError('Available source ROI IDs: 1..5 in field 2380205D')
    onset=p['inferred_onset_s'] if onset_s is None else float(onset_s)
    if not np.isfinite(onset):raise ValueError('Finite onset required')
    factor=1. if treatment=='control' else next(r['drug_ratio_median'] for r in doc['field_summary'] if r['field']=='2380205D')
    dr=factor*p['scale_deltaR']*pulse_response(t,onset,3.,p['effective_slow_s'],p['effective_fast_s'])
    return dict(t_s=t,delta_ratio=dr,profile=p,treatment=treatment,amplitude_factor=factor,
                status='empirical NMDA->Fura2 response; no free Ca concentration, no SK feedback')

@dataclass(frozen=True)
class Fura2Calibration:
    kd_uM:float
    rmin:float
    rmax:float
    beta:float
    provenance:str
    def __post_init__(self):
        if not np.isfinite([self.kd_uM,self.rmin,self.rmax,self.beta]).all() or self.kd_uM<=0 or self.beta<=0 or self.rmin<0 or self.rmax<=self.rmin or not self.provenance:
            raise ValueError('Explicit valid dye calibration and provenance required')

def observe_fura2(t_s,ca_uM,cycle_start_s,*,calibration,exposure_s=.1,channel_gap_s=.1):
    """Equilibrium dye scenario, separate sequential exposures then ratio of intensities.

    No in-cell Rmin/Rmax/beta/Kd is available for D'Andrea. This function requires
    them from the caller, does not silently convert observed ratios to calcium.
    Indicator buffering and finite binding kinetics remain external to this readout.
    """
    t=_times(t_s);starts=_times(cycle_start_s);ca=np.asarray(ca_uM,float);p=calibration
    if not isinstance(p,Fura2Calibration):raise ValueError('Fura2Calibration required')
    if ca.shape!=t.shape or not np.isfinite(ca).all() or np.any(ca<0):raise ValueError('Finite nonnegative paired Ca required')
    if exposure_s<=0 or channel_gap_s<0 or not np.isfinite([exposure_s,channel_gap_s]).all():raise ValueError('Invalid exposure')
    start380=starts+exposure_s+channel_gap_s
    if starts[0]<t[0] or start380[-1]+exposure_s>t[-1]+1e-10:raise ValueError('Complete exposures must be inside simulated trace')
    occupancy=ca/(p.kd_uM+ca)
    f380=p.beta*(1-occupancy)+occupancy
    f340=p.rmin*p.beta*(1-occupancy)+p.rmax*occupancy
    def avg(f,start):
        integ=cumulative_trapezoid(f,t,initial=0)
        return (np.interp(start+exposure_s,t,integ)-np.interp(start,t,integ))/exposure_s
    a=avg(f340,starts);b=avg(f380,start380)
    return dict(cycle_start_s=starts,F340_au=a,F380_au=b,ratio=a/b,
                status='explicit equilibrium Fura2 scenario with sequential exposure averaging',calibration_provenance=p.provenance)

def preserved_electrical(profile='Kiskinis',current_pA=25.):
    """Execute the unchanged NEURON profiles. Return source-specific AP count."""
    if profile=='Kiskinis':
        m=_load('preserved_kiskinis',B.parent/'2026-10-16-kiskinis-electrical/kiskinis_model.py')
        out=m.run(current_pA=current_pA);out['count']=int(out['kiskinis_features'][0]);out['spikes_s']=out['t'][m.peaks(out['v'],out['used_dt_ms'],200.,500.)]/1000
    elif profile in ('Selvaraj-Con1','Selvaraj-Con2'):
        m=_load('preserved_selvaraj_api',B.parent/'2026-10-16-selvaraj-waveform/model_api.py')
        out=m.run(line=profile.split('-')[1],current_pA=current_pA);out['count']=int(out['features']['count'])
    elif profile=='Burley':
        sys.path.insert(0,str(B.parent/'2026-10-14-ipsc-electrical-multisource'))
        import neuron_replay as nr
        from neuron import h
        h.celsius=23
        doc=json.loads((B.parent/'2026-10-14-ipsc-electrical-multisource/burley-local.json').read_text())
        out=nr.run(np.array(doc['parameters']),np.array(doc['context']),current=current_pA,dt=.005)
        out['count']=int(out['features']['zero_crossings'])
    else:raise ValueError('Unknown electrical source profile')
    return out
