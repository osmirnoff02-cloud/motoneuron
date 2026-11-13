"""[codex] 2026-10-25. Data-aware five-channel/FRET observation bridge.

Optical observation only: this dataset does not fit human Ca, CAPN or TDP rates.
"""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
B=Path(__file__).resolve().parent

def _observer():
    name='_gerencser_renewing_observer_20261019'
    if name not in sys.modules:
        p=B.parent/'2026-10-19-source-model-update/source_observers_codex.py'
        s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
    return sys.modules[name]

def gerencser_spectral_correction(observed,mixing_matrix,*,background,provenance,calibration_confirmed=False):
    """Samples x [YFP,CFP,FRET,340,380]; each ratio has its own valid mask.

    Signed FRET residuals are kept. A negative FRET residual does not invalidate
    otherwise measurable Fura. No silent zero clipping or absolute Ca mapping.
    """
    a=np.asarray(observed,float);m=np.asarray(mixing_matrix,float)
    if a.ndim!=2 or a.shape[1]!=5 or m.shape!=(5,5) or not np.isfinite(a).all() or not np.isfinite(m).all() or not provenance.strip():raise ValueError('Finite five-channel samples/matrix and provenance required')
    cond=np.linalg.cond(m)
    if not np.isfinite(cond) or cond>1e6:raise ValueError('Spectral matrix is singular or ill-conditioned')
    bg=np.broadcast_to(np.asarray(background,float),a.shape)
    if not np.isfinite(bg).all():raise ValueError('Finite background required')
    z=np.linalg.solve(m,(a-bg).T).T
    vf=(z[:,0]>0)&(z[:,1]>0);vc=z[:,4]>0
    fret=np.full(len(a),np.nan);fura=np.full(len(a),np.nan)
    fret[vf]=z[vf,2]/np.sqrt(z[vf,0]*z[vf,1]);fura[vc]=z[vc,3]/z[vc,4]
    return dict(unmixed=z,FRET_ratio=fret,Fura_ratio=fura,valid_FRET=vf,valid_Fura=vc,
        negative_residual_mask=z<0,condition_number=float(cond),reconstruction_error=float(np.max(abs(z@m.T+bg-a))),
        calibration_confirmed=bool(calibration_confirmed),provenance=provenance,
        calcium_uM=None,calpain_activity=None,TDP43_cleavage=None)

def gerencser_example():
    """Read-only optical benchmark, not concentration or enzyme activation."""
    with np.load(B/'paired_traces_and_sensitivity_codex.npz',allow_pickle=False) as a:
        data={k:a[k].copy() for k in a.files}
    settings=json.loads((B/'analysis_settings_codex.json').read_text(encoding='utf-8'))
    return dict(**data,rois=['R1','R2','R3'],settings=settings,
        stimulus_onset_s=None,treatment_group=None,calpain_washout_fit_allowed=False,
        human_kinetic_fit_allowed=False,assay_calibration_complete=False,
        primary_path='papers/Examples-FRET',biological_context='Associated rat cortical excitotoxicity study; exact example treatment pending author confirmation')

def observe_gerencser_calpain(time_s,activity,*,cleavage_per_s,synthesis_per_s,
        intact_turnover_per_s,product_turnover_per_s,donor_dequench_ratio,
        initial=(1.,0.,1.,0.),optical_multiplier=1.,illumination_fraction=0.,
        bleach_per_illuminated_s=0.,provenance):
    """CAPN activity -> reporter pools -> FRET; never FRET -> CAPN by identity.

    Donor intensity can increase after cleavage. Acceptor/product yields and
    turnover are simplified assumptions, not fitted constants of the example.
    Reversible optical factor is independent of cleavage, not sensor repair.
    """
    if not np.isfinite(donor_dequench_ratio) or donor_dequench_ratio<1:raise ValueError('Explicit donor dequenching ratio >=1 required')
    r=_observer().renewing_reporter(time_s,activity,cleavage_per_s=cleavage_per_s,synthesis_per_s=synthesis_per_s,
        intact_turnover_per_s=intact_turnover_per_s,product_turnover_per_s=product_turnover_per_s,initial=initial,
        illumination_fraction=illumination_fraction,bleach_per_illuminated_s=bleach_per_illuminated_s,
        provenance=provenance)
    u=r['bright_intact'];c=r['bright_cleaved'];yfp=u+c;cfp=u+donor_dequench_ratio*c
    optical=np.broadcast_to(np.asarray(optical_multiplier,float),u.shape)
    if not np.isfinite(optical).all() or np.any(optical<0):raise ValueError('Nonnegative independent optical multiplier required')
    valid=(yfp>0)&(cfp>0);fret=np.full(u.shape,np.nan);fret[valid]=optical[valid]*u[valid]/np.sqrt(yfp[valid]*cfp[valid])
    return dict(time_s=r['time_s'],FRET_ratio=fret,reporter=r,YFP=yfp,CFP=cfp,valid=valid,
        calpain_activity=np.broadcast_to(np.asarray(activity,float),u.shape).copy(),parameters=dict(cleavage_per_s=cleavage_per_s,donor_dequench_ratio=donor_dequench_ratio),
        optical_multiplier=optical,provenance=provenance,human_fit=False,reporter_repair=False,
        assumptions='Equal acceptor yields and common bleaching; donor dequenching and reporter rates unspecified by the sample. No TDP rate inferred.')

def compare_gerencser_fret(time_s,predicted_FRET_ratio,*,roi,recording_offset_s,provenance):
    """Compare observation outputs on actual acquisition times; descriptive only.

    recording_offset_s explicitly maps the model origin to the recording origin.
    Fura ratios cannot be substituted as Ca/uM by this API. No likelihood or
    fitted human rates follow from descriptive residuals of one sample field.
    """
    t=np.asarray(time_s,float);f=np.asarray(predicted_FRET_ratio,float)
    if t.ndim!=1 or f.shape!=t.shape or len(t)<2 or not np.isfinite(t).all() or not np.isfinite(f).all() or np.any(np.diff(t)<=0) or not np.isfinite(recording_offset_s) or not provenance.strip():raise ValueError('Finite ordered predicted optical trace and explicit alignment required')
    d=gerencser_example()
    if roi not in d['rois']:raise ValueError('Unknown ROI')
    r=d['rois'].index(roi);tt=d['time_s']+recording_offset_s
    if tt[0]<t[0] or tt[-1]>t[-1]:raise ValueError('Prediction does not cover complete observed time window; extrapolation refused')
    pred=np.interp(tt,t,f);obs=d['values'][0,r];base=pred[:3].mean()
    if base<=0:raise ValueError('Positive baseline required')
    pred=pred/base;obs=obs/obs[:3].mean();residual=pred-obs
    return dict(roi=roi,time_s=d['time_s'],observed_normalized=obs,predicted_normalized=pred,
        residual=residual,RMSE=float(np.sqrt(np.mean(residual**2))),provenance=provenance,
        inferential_p_value=None,independent_donors=None,human_calibration=False,
        calpain_off_rate_identified=False,meaning='Descriptive optical discrepancy; protocol/treatment/calibration not matched, therefore not a biological goodness-of-fit test.')
