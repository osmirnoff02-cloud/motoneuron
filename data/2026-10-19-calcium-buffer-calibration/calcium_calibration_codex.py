"""Codex 2026-10-19. Calibrated Fura-2 readout and added-buffer inference.

Independent implementation of standard ratiometry/weighted least squares.
Hess reference: doi:10.1016/j.dib.2021.107494 and author analysis on GitLab.
All concentrations uM, all times seconds. gamma is effective rate per volume
(1/s), not the activity of an individually identified pump.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2


@dataclass(frozen=True)
class DyeCalibration:
    rmin: float
    rmax: float
    keff_uM: float
    kd_uM: float
    provenance: str
    rmin_se: float = 0.
    rmax_se: float = 0.
    keff_se_uM: float = 0.
    kd_se_uM: float = 0.

    def __post_init__(self):
        vals=[self.rmin,self.rmax,self.keff_uM,self.kd_uM,
              self.rmin_se,self.rmax_se,self.keff_se_uM,self.kd_se_uM]
        if not np.isfinite(vals).all() or self.rmin<0 or self.rmax<=self.rmin or min(self.keff_uM,self.kd_uM)<=0 or min(vals[4:])<0 or not self.provenance:
            raise ValueError('Finite physically valid calibration and source required')

    @property
    def beta(self):
        return self.keff_uM/self.kd_uM


def ratio_to_ca(ratio, calibration):
    """No extrapolation beyond the dye range; return NaN and a validity mask."""
    r=np.asarray(ratio,float);p=calibration
    valid=np.isfinite(r)&(r>=p.rmin)&(r<p.rmax)
    with np.errstate(divide='ignore',invalid='ignore'):
        ca=p.keff_uM*(r-p.rmin)/(p.rmax-r)
    return np.where(valid,ca,np.nan),valid


def ca_to_ratio(ca_uM, calibration):
    c=np.asarray(ca_uM,float);p=calibration
    if not np.isfinite(c).all() or np.any(c<0):raise ValueError('Nonnegative finite Ca required')
    return (p.keff_uM*p.rmin+c*p.rmax)/(p.keff_uM+c)


def decode_adu(adu, *, roi_pixels, background_pixels, exposure340_s,
               exposure380_s, gain, read_noise_e, calibration):
    """Decode summed ROI/background counts, correcting area and unequal exposures.

    Measurement SE uses the delta method with shot/read noise. Calibration SE is
    kept separate: calibration errors are shared across time, not independent
    noise for each frame. Its estimate assumes zero covariance (not provided).
    Acquisition timestamps cannot resolve within-cycle channel offsets here.
    """
    a=np.asarray(adu,float);p=calibration
    if a.ndim!=2 or a.shape[1]!=7 or not np.isfinite(a).all():raise ValueError('Seven finite ADU columns required')
    if min(roi_pixels,background_pixels,exposure340_s,exposure380_s,gain)<=0 or read_noise_e<0:raise ValueError('Invalid acquisition metadata')
    if not np.isfinite([roi_pixels,background_pixels,exposure340_s,exposure380_s,gain,read_noise_e]).all():raise ValueError('Nonfinite metadata')
    f340=(a[:,1]/roi_pixels-a[:,2]/background_pixels)/exposure340_s
    f380=(a[:,5]/roi_pixels-a[:,6]/background_pixels)/exposure380_s
    f360=a[:,3]/roi_pixels-a[:,4]/background_pixels
    def var(roi,bg,T):
        return ((gain*np.maximum(roi,0)+gain**2*roi_pixels*read_noise_e**2)/roi_pixels**2+
                (gain*np.maximum(bg,0)+gain**2*background_pixels*read_noise_e**2)/background_pixels**2)/T**2
    with np.errstate(divide='ignore',invalid='ignore'):
        r=f340/f380
        vr=var(a[:,1],a[:,2],exposure340_s)/f380**2+f340**2*var(a[:,5],a[:,6],exposure380_s)/f380**4
        deriv=p.keff_uM*(p.rmax-p.rmin)/(p.rmax-r)**2
        measure_se=np.abs(deriv)*np.sqrt(vr)
        cal_se=np.sqrt(((r-p.rmin)/(p.rmax-r)*p.keff_se_uM)**2+
                       (p.keff_uM/(p.rmax-r)*p.rmin_se)**2+
                       (p.keff_uM*(r-p.rmin)/(p.rmax-r)**2*p.rmax_se)**2)
    ca,physical=ratio_to_ca(r,p)
    valid=(f340>0)&(f380>0)&(r<p.rmax)&np.isfinite(measure_se)
    # Signed noisy estimates must survive in the likelihood. Censoring r<Rmin
    # would select positive noise and bias the low-amplitude tail fit.
    with np.errstate(divide='ignore',invalid='ignore'):
        estimate=p.keff_uM*(r-p.rmin)/(p.rmax-r)
    return dict(ratio=r,ca_uM=np.where(valid,ca,np.nan),ca_estimate_uM=np.where(valid,estimate,np.nan),
                valid=valid,physical_valid=physical&valid,
                measurement_se_uM=np.where(valid,measure_se,np.nan),
                shared_calibration_se_uM=np.where(valid,cal_se,np.nan),f360_net=f360)


def indicator_capacity(dye_uM, kd_uM, ca_uM):
    """Differential fast-equilibrium buffer capacity; dimensionless."""
    d=np.asarray(dye_uM,float);c=np.asarray(ca_uM,float)
    if not np.isfinite(d).all() or not np.isfinite(c).all() or not np.isfinite(kd_uM) or np.any(d<0) or np.any(c<0) or kd_uM<=0:raise ValueError('Invalid dye/Ca')
    return d*kd_uM/(kd_uM+c)**2


def fit_decay(time_s, ca_uM, se_uM, *, baseline_points, start_index=None, remaining_fraction=.5):
    """Fit baseline plus monoexponential tail, omitting rise/peak.

    The baseline window is protocol metadata, never inferred from the fit result.
    SE is measurement noise only. Return lack-of-fit and lag-1 diagnostics.
    """
    t=np.asarray(time_s,float);c=np.asarray(ca_uM,float);se=np.asarray(se_uM,float)
    if t.ndim!=1 or c.shape!=t.shape or se.shape!=t.shape or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Increasing paired trace required')
    if not 2<=baseline_points<len(t)-6:raise ValueError('Invalid baseline window')
    if not (np.isfinite(c[:baseline_points]).all() and np.isfinite(se[:baseline_points]).all()):raise ValueError('Invalid baseline measurements')
    base=float(np.mean(c[:baseline_points]));peak=int(np.nanargmax(c))
    if peak<baseline_points or c[peak]<=base:raise ValueError('No post-baseline positive transient')
    if start_index is None:
        if not 0<remaining_fraction<1:raise ValueError('Invalid tail fraction')
        ix=np.flatnonzero(c[peak+1:]<=base+remaining_fraction*(c[peak]-base))
        if not len(ix):raise ValueError('Tail never reaches fitting fraction')
        start_index=int(peak+1+ix[0])
    if not baseline_points<=start_index<len(t)-5:raise ValueError('Invalid tail start')
    ids=np.r_[np.arange(baseline_points),np.arange(start_index,len(t))]
    good=np.isfinite(c[ids])&np.isfinite(se[ids])&(se[ids]>0)
    ids=ids[good]
    if len(ids)<7:raise ValueError('Too few valid samples')
    tt=t[ids]-t[start_index];tail=ids>=start_index
    def predict(p):return p[0]+np.where(tail,p[1]*np.exp(-np.maximum(tt,0)/p[2]),0)
    def resid(p):return (predict(p)-c[ids])/se[ids]
    x0=[max(base,1e-6),max(c[start_index]-base,1e-5),max((t[-1]-t[start_index])/5,.1)]
    fit=least_squares(resid,x0,bounds=([0,0,1e-5],[np.inf,np.inf,1e5]),x_scale='jac',max_nfev=2000,ftol=1e-11,gtol=1e-11,xtol=1e-11)
    if not fit.success:raise ValueError('Decay optimization did not converge')
    rss=float(fit.fun@fit.fun);dof=len(ids)-3
    cov=np.linalg.pinv(fit.jac.T@fit.jac) # absolute known noise; not scaled by residual variance
    errs=np.sqrt(np.maximum(0,np.diag(cov)))
    r=-fit.fun;lag=float(np.dot(r[:-1],r[1:])/(len(r)-1))
    return dict(baseline_uM=float(fit.x[0]),delta_uM=float(fit.x[1]),tau_s=float(fit.x[2]),
                tau_se_s=float(errs[2]),baseline_se_uM=float(errs[0]),start_index=int(start_index),
                baseline_points=int(baseline_points),rss=rss,dof=dof,p_lack_of_fit=float(chi2.sf(rss,dof)),
                lag1_residual_product=lag,relative_tau_se=float(errs[2]/fit.x[2]),
                prediction=predict(fit.x),indices=ids,residuals=r,covariance=cov)


def fit_added_buffer(kappa_added, tau_s, tau_se_s, *, configuration, provenance):
    """tau=a+b*kappa; gamma=1/b; kappa_endogenous=a/b-1.

    Fits within ONE cell, across indicator loading. Stable endogenous buffers,
    influx and extrusion, fast binding and a well-mixed compartment are assumed.
    Never identify endogenous buffer from a single dye capacity or from pulse
    duration alone. Whole-cell does not guarantee preserved mobile buffers.
    """
    x=np.asarray(kappa_added,float);y=np.asarray(tau_s,float);s=np.asarray(tau_se_s,float)
    if x.ndim!=1 or x.shape!=y.shape or x.shape!=s.shape or len(x)<3 or not np.isfinite([x,y,s]).all() or np.any(x<0) or np.any(y<=0) or np.any(s<=0):raise ValueError('At least three valid matched loading points required')
    if configuration not in ('whole_cell','beta_escin_perforated','other_perforated','unknown') or not provenance:raise ValueError('Explicit configuration and source required')
    if np.ptp(x)<=1e-6*max(1,float(np.max(x))):raise ValueError('Buffer and removal not identifiable at one dye capacity')
    A=np.column_stack([np.ones(len(x)),x]);W=A/s[:,None]
    cov=np.linalg.inv(W.T@W);coef=cov@(W.T@(y/s));a,b=map(float,coef)
    rss=float(np.sum(((y-A@coef)/s)**2));dof=len(x)-2
    slope_se=float(np.sqrt(cov[1,1]));identified=bool(b>0 and b-1.96*slope_se>0 and a/b-1>=0)
    warnings=[]
    if configuration=='whole_cell':warnings.append('Mobile-buffer washout possible; estimate refers to recorded state, not intact-cell total buffer')
    if not identified:warnings.append('Nonphysical or weakly identified slope/intercept; do not use as calibrated physiology')
    model_p=float(chi2.sf(rss,dof))
    if model_p<.01:warnings.append('Linear added-buffer relation fails lack-of-fit check')
    gamma_se=np.sqrt(cov[1,1])/b**2 if b>0 else None
    grad=np.array([1/b,-a/b**2]) if b>0 else None
    kappa_se=float(np.sqrt(max(0,grad@cov@grad))) if b>0 else None
    return dict(intercept_s=a,slope_s=b,covariance=cov.tolist(),gamma_per_s=1/b if b>0 else None,
                endogenous_kappa=a/b-1 if b>0 else None,slope_se_s=slope_se,
                gamma_se_per_s=float(gamma_se) if gamma_se is not None else None,endogenous_kappa_se=kappa_se,
                intercept_se_s=float(np.sqrt(cov[0,0])),rss=rss,dof=dof,
                p_lack_of_fit=model_p,identified=identified,usable_under_linear_model=identified and model_p>=.01,
                configuration=configuration,provenance=provenance,warnings=warnings,
                status='conditional within-cell estimate; not human MN calibration')


def buffered_response(time_s, influx_uM_per_s, *, basal_ca_uM, endogenous_kappa,
                      gamma_per_s, dye_uM=0., dye_kd_uM=None, provenance):
    """Linearized fast-buffer balance, exact propagation of piecewise constant input.

    d(C-C0)/dt=(J-gamma*(C-C0))/(1+kappa_endo+kappa_dye(C0)).
    Input is total concentration flux before buffering. No implied AP->Ca
    conversion or feedback to voltage; caller must supply missing kinetics.
    Nonnegative input and nonnegative capacity guarantee Ca>=C0.
    """
    t=np.asarray(time_s,float);j=np.asarray(influx_uM_per_s,float)
    if t.ndim!=1 or len(t)<2 or j.shape!=t.shape or not np.isfinite(t).all() or np.any(np.diff(t)<=0) or not np.isfinite(j).all() or np.any(j<0):raise ValueError('Finite increasing time and nonnegative matching influx required')
    if not np.isfinite([basal_ca_uM,endogenous_kappa,gamma_per_s,dye_uM]).all() or min(basal_ca_uM,endogenous_kappa,dye_uM)<0 or gamma_per_s<=0 or not provenance:raise ValueError('Explicit valid physiology/scenario and provenance required')
    if dye_uM>0 and dye_kd_uM is None:raise ValueError('Dye binding constant required')
    kdye=float(indicator_capacity(dye_uM,dye_kd_uM,basal_ca_uM)) if dye_uM>0 else 0.
    capacity=1+endogenous_kappa+kdye;tau=capacity/gamma_per_s
    excess=np.zeros_like(t)
    for i,dt in enumerate(np.diff(t)):
        decay=np.exp(-dt/tau);excess[i+1]=excess[i]*decay+j[i]/gamma_per_s*(-np.expm1(-dt/tau))
    return dict(t_s=t,ca_uM=basal_ca_uM+excess,effective_tau_s=tau,kappa_dye=kdye,
                status='linearized fast-buffer scenario; kinetics supplied by caller',provenance=provenance)
