"""[codex] Searl1991: amplitude distributions and observation censoring.

Mixture parameters describe emitted events, not anatomical vesicle fractions.
No dose-response law or spontaneous/evoked equivalence is assumed.
"""
import numpy as np
from scipy.special import ndtr

QUANTAL_OBSERVATION_RANGES=dict(detection_threshold_relative=[0.,.1,.3],component_cv=[.1,.2,.3],small_event_fraction=[0.,.5,1.])

def observe_quantal_mixture(weights,means,standard_deviations,*,threshold=0.):
    """Gaussian amplitude mixture, left truncation below a positive threshold.

Gaussian tails approximate measured electrical amplitudes, not negative ACh.
Underlying true_mean and detected_mean are deliberately separate.
"""
    w,m,s=map(lambda x:np.asarray(x,float),(weights,means,standard_deviations))
    if w.ndim!=1 or w.shape!=m.shape or w.shape!=s.shape or not len(w) or not np.isfinite(np.r_[w,m,s,threshold]).all() or np.any(w<0) or np.any(m<0) or np.any(s<0) or threshold<0 or not np.isclose(w.sum(),1.):
        raise ValueError('Invalid quantal mixture')
    tail=np.zeros_like(m);first=np.zeros_like(m);second=np.zeros_like(m)
    for i in range(len(w)):
        if m[i]==0 and s[i]==0:continue # Silent vesicles are not recorded events.
        if s[i]==0:
            if m[i]>threshold:tail[i]=1.;first[i]=m[i];second[i]=m[i]**2
        else:
            a=(threshold-m[i])/s[i];q=ndtr(-a);d=np.exp(-a*a/2)/np.sqrt(2*np.pi)
            tail[i]=q;first[i]=m[i]*q+s[i]*d
            second[i]=(m[i]**2+s[i]**2)*q+s[i]*(m[i]+threshold)*d
    detected=float(w@tail);mean=float(w@first/detected) if detected else None
    variance=max(0.,float(w@second/detected)-mean**2) if detected else None
    return dict(true_mean=float(w@m),detected_fraction=detected,detected_mean=mean,
                detected_cv=np.sqrt(variance)/mean if detected and mean else None,
                detected_component_weights=(w*tail/detected).tolist() if detected else [0.]*len(w))

def searl_reference():
    return dict(species='Sprague-Dawley rat hemidiaphragm',temperature_C=[18.,22.],holding_mV=-55.,frequency_Hz=10.,duration_s=300.,
        vesamicol_nM=[50.,100.],n=[5,5],small_event_fraction=[.536,.756],small_event_fraction_SEM=[.041,.058],
        normal_mean_nA=[2.3,2.7],small_mean_nA=[1.2,1.2],normal_CV=[.247,.194],small_CV=[.252,.218],
        paired_small_to_normal_ratio=[.543,.438],paired_ratio_SEM=[.036,.040],
        limitation='Table3 group means and paired mean ratios are different statistics; these are post-stimulation miniature currents, not raw events, live human pool counts or a drug dosing model.')
