"""Codex 2026-10-19. Recovery protocols around frozen human electrical profiles.

This extends observable behaviour, not human calcium calibration. Units ms/mV/pA.
No original parameter, membrane mechanism, or study-specific FI protocol is edited.
"""
from pathlib import Path
import sys,json,importlib.util
import numpy as np
from scipy.signal import find_peaks
B=Path(__file__).resolve().parent
ROOT=B.parent.parent
sys.path.insert(0,str(B.parent/'2026-10-14-ipsc-electrical-multisource'))
from publication_optional_neuron import nr, h

def frozen_profile(name):
    if name=='Burley':
        p=json.loads((B.parent/'2026-10-14-ipsc-electrical-multisource/burley-local.json').read_text())
        return np.array(p['parameters']),np.array(p['context']),False
    if name in ('Selvaraj-Con1','Selvaraj-Con2'):
        doc=json.loads((B.parent/'2026-10-16-selvaraj-waveform/profiles-frozen.json').read_text())
        p=next(x for x in doc['profiles'] if x['line']==name.split('-')[1])
        return np.array(p['parameters']),np.array(p['context']),True
    raise ValueError('Recovery protocols currently support Burley and both Selvaraj controls')

def spike_indices(t,v,baseline,onset_ms):
    """Relative detector, checked separately against preserved study detectors.

    Whole trace is searched, including delayed and post-stimulus events. A slow
    shoulder cannot borrow the rising edge of the preceding action potential.
    """
    dt=float(t[1]-t[0])
    candidates=find_peaks(v,height=baseline+15,prominence=10,distance=max(1,round(3/dt)))[0]
    keep=[]
    for k in candidates:
        if t[k]<onset_ms:continue
        left=k-1
        while left>0 and v[left]>v[left-1] and t[k]-t[left]<30:left-=1
        if k>left and np.max(np.diff(v[left:k+1])/dt)>=5:keep.append(k)
    return np.array(keep,dtype=int)

def recovery_features(t,v,*,onset_ms,offset_ms,tolerance_mV=1.,minimum_tail_ms=100.):
    t=np.asarray(t,float);v=np.asarray(v,float)
    if t.ndim!=1 or v.shape!=t.shape or len(t)<3 or not np.isfinite(t).all() or not np.isfinite(v).all() or np.any(np.diff(t)<=0):raise ValueError('Finite ordered voltage trace required')
    if not t[0]+100<=onset_ms<offset_ms<t[-1] or tolerance_mV<=0 or minimum_tail_ms<=0:raise ValueError('Invalid protocol or recovery definition')
    base=v[(t>=onset_ms-100)&(t<onset_ms)]
    baseline=float(np.median(base));pk=spike_indices(t,v,baseline,onset_ms)
    if not len(pk):return dict(ap_count=0,baseline_mV=baseline,status='no AP; recovery not applicable')
    last=float(t[pk[-1]]);start=max(offset_ms,last)
    mask=t>=start;tx=t[mask];delta=v[mask]-baseline
    # Return means stays within band until the end, with >=100 ms observed tail.
    stable_to_end=np.logical_and.accumulate((np.abs(delta)<=tolerance_mV)[::-1])[::-1]
    idx=np.where(stable_to_end & (tx<=t[-1]-minimum_tail_ms))[0]
    ret=float(tx[idx[0]]) if len(idx) else None
    minimum=int(np.argmin(delta));negarea=float(np.trapezoid(np.maximum(-delta,0),tx))
    return dict(ap_count=int(len(pk)),spike_times_ms=t[pk].tolist(),baseline_mV=baseline,
        prestimulus_range_mV=float(np.ptp(base)),last_peak_ms=last,
        measurement_start_ms=start,post_min_delta_mV=float(delta[minimum]),
        post_min_time_after_last_peak_ms=float(tx[minimum]-last),
        ahp_depth_mV=max(0.,float(-delta[minimum])),negative_area_mV_ms=negarea,
        return_after_last_peak_ms=None if ret is None else ret-last,
        return_after_offset_ms=None if ret is None else ret-offset_ms,
        return_censored=ret is None,observed_after_last_peak_ms=float(t[-1]-last),
        tolerance_mV=tolerance_mV,minimum_stable_tail_ms=minimum_tail_ms,
        delta_after_offset_mV={str(k):float(np.interp(offset_ms+k,t,v)-baseline) for k in (50,200,500,1000) if offset_ms+k<=t[-1]},
        late_AP_after_offset_plus20ms=int(np.count_nonzero(t[pk]>offset_ms+20)),
        final_delta_mV=float(v[-1]-baseline),
        status='model diagnostic; not an experimentally calibrated AHP duration')

def run_recovery(profile,*,current_pA,duration_ms,dt_ms=.005,pre_ms=1000.,post_ms=3000.):
    if not np.isfinite([current_pA,duration_ms,dt_ms,pre_ms,post_ms]).all() or current_pA<=0 or duration_ms<=0 or dt_ms<=0 or pre_ms<100 or post_ms<100:raise ValueError('Invalid protocol')
    p,c,free=frozen_profile(profile);h.celsius=23
    out=nr.run(p,c,current_pA,duration=duration_ms,dt=dt_ms,pre=pre_ms,post=post_ms,rest=free,init=c[2] if free else c[5])
    out['recovery']=recovery_features(out['t'],out['v'],onset_ms=pre_ms,offset_ms=pre_ms+duration_ms)
    out['protocol']=dict(profile=profile,current_pA=current_pA,duration_ms=duration_ms,
        pre_ms=pre_ms,post_ms=post_ms,dt_ms=dt_ms,temperature_C=23,
        holding='free_rest' if free else 'preserved_-70mV_bias',
        applied_bias_pA=0. if free else out['holding_pA'],
        source='new diagnostic command; published source parameters preserved')
    return out

def select_single_ap(profile):
    """Bounded engineering search, never a fit to a paper or a human rheobase."""
    tried=[]
    for duration in (2.,5.,10.):
        for current in (25.,50.,100.,200.,400.):
            out=run_recovery(profile,current_pA=current,duration_ms=duration,dt_ms=.01,post_ms=1200.)
            tried.append(dict(current_pA=current,duration_ms=duration,ap_count=out['recovery']['ap_count']))
            if out['recovery']['ap_count']==1:return current,duration,tried
    raise RuntimeError('No isolated single AP in bounded search: '+str(tried))
