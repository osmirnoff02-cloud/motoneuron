"""Codex: control-calibrated optical-event QC, not AP counts or calcium pumps."""
import numpy as np
from scipy.signal import find_peaks


def optical_event_features(fluorescence,time_s,*,prominence_z,minimum_separation_s=2.):
    f=np.asarray(fluorescence,float);t=np.asarray(time_s,float)
    if f.ndim!=2 or f.shape[1]!=len(t) or len(t)<5 or not np.isfinite(f).all() or not np.isfinite(t).all():raise ValueError('Finite ROI x time matrix required')
    dt=np.diff(t)
    if np.any(dt<=0) or not np.allclose(dt,dt[0]) or not np.isfinite(prominence_z) or prominence_z<=0 or minimum_separation_s<=0:raise ValueError('Regular time grid and positive threshold/separation required')
    f0=np.percentile(f,20,axis=1);valid=f0>0
    dff=np.full_like(f,np.nan);dff[valid]=(f[valid]-f0[valid,None])/f0[valid,None]
    dif=np.diff(dff,axis=1);noise=1.4826*np.nanmedian(abs(dif-np.nanmedian(dif,axis=1)[:,None]),axis=1)/np.sqrt(2)
    valid&=np.isfinite(noise)&(noise>0)
    z=np.full_like(f,np.nan);z[valid]=dff[valid]/noise[valid,None]
    peaks=[];prominences=[]
    for i in range(len(f)):
        if not valid[i]:peaks.append([]);prominences.append([]);continue
        pp,pr=find_peaks(z[i],prominence=prominence_z,distance=max(1,int(np.ceil(minimum_separation_s/dt[0]))))
        peaks.append(pp.tolist());prominences.append(pr['prominences'].tolist())
    return dict(dff=dff,z=z,noise_dff=noise,valid=valid,peak_indices=peaks,prominences=prominences,
                count=np.array([len(p) for p in peaks]),units='optical events per observed window, NOT action potentials',
                calcium_uM=None,calcium_removal_rate=None)


def calibrate_negative_control(fluorescence,time_s,*,quantile=.99,provenance):
    if not 0<quantile<1 or not provenance.strip():raise ValueError('Control provenance and quantile required')
    result=optical_event_features(fluorescence,time_s,prominence_z=1e-12)
    maxima=np.array([max(v,default=0.) for v,ok in zip(result['prominences'],result['valid']) if ok])
    if len(maxima)<5:raise ValueError('Too few valid negative-control ROIs')
    threshold=max(3.,float(np.quantile(maxima,quantile)))
    return dict(prominence_z=threshold,quantile=quantile,training_ROIs=len(maxima),provenance=provenance,
                calibration_scope='Threshold for this acquisition only; not independent biological validation')
