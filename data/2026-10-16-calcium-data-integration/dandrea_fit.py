"""Source-specific phenomenological NMDA->Fura-2 profile, not free Ca or pump calibration."""
from pathlib import Path
import json,csv
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent

def pulse_response(t,onset,duration,decay,rise):
    """Unit DC-gain cascade of two first-order processes, analytically integrated square pulse.
    Swapping rise and decay leaves the signal unchanged (identifiability example).
    """
    if min(decay,rise,duration)<=0 or abs(decay-rise)<1e-7:raise ValueError('Positive distinct time constants required')
    def step(x):
        x=np.maximum(x,0)
        return 1-(decay*np.exp(-x/decay)-rise*np.exp(-x/rise))/(decay-rise)
    return step(np.asarray(t)-onset)-step(np.asarray(t)-onset-duration)

def main():
    z=np.load(B/'dandrea-traces.npz');qc=json.loads((B/'dandrea-registration.json').read_text())
    invalid={r['field']+'_'+r['condition']:[i for i,s in enumerate(r['shifts_yx_corr']) if s[2]<.8] for r in qc}
    records=list(csv.DictReader((B/'dandrea-metrics.csv').open(encoding='utf-8')))
    # Derived mask is applied to source values; raw data/previous raw extraction are preserved.
    curves=[];rows=[];profiles=[]
    for key in z.files:
        t,ratio,dr=z[key];field,condition,roi=key.split('_');valid=np.ones(len(t),bool);valid[invalid[field+'_'+condition]]=False
        win=valid&(t>=8)&(t<=24);peak=int(np.flatnonzero(win)[np.argmax(dr[win])]);noise=float(np.std(dr[t<6],ddof=1))
        rows.append(dict(field=field,condition=condition,roi=int(roi[3:]),deltaR=float(dr[peak]),peak_s=float(t[peak]),
                         baseline_sd=noise,invalid_frames=int((~valid).sum()),control_positive_SNR3=bool(dr[peak]>3*noise)))
        curves.append((key,t,dr,valid))
    effects=[]
    for field in ['2380202D','2380205D']:
        control=[r for r in rows if r['field']==field and r['condition']=='first']
        for c in control:
            d=next(r for r in rows if r['field']==field and r['condition']=='drug' and r['roi']==c['roi'])
            w=next(r for r in rows if r['field']==field and r['condition']=='wash' and r['roi']==c['roi'])
            effects.append(dict(field=field,roi=c['roi'],control_snr_ok=c['control_positive_SNR3'],
                                drug_over_control=d['deltaR']/c['deltaR'] if c['control_positive_SNR3'] else None,
                                wash_over_control=w['deltaR']/c['deltaR'] if c['control_positive_SNR3'] else None))
    # Fit every control ROI with positive SNR, so an incompatible field is visible, not silently removed.
    for key,t,dr,valid in curves:
        field,cond,roi=key.split('_');c=next(r for r in rows if r['field']==field and r['condition']==cond and r['roi']==int(roi[3:]))
        if cond!='first' or not c['control_positive_SNR3']:continue
        def residual(p):return (p[0]*pulse_response(t, p[1],3.,p[2],p[3])-dr)[valid]
        opt=least_squares(residual,[max(dr.max()*5,.01),14.,10.,.5],bounds=([0.,10.,1.01,.05],[20.,17.,120.,1.]))
        amp,onset,decay,rise=opt.x;pred=amp*pulse_response(t,onset,3.,decay,rise)
        err=float(np.sqrt(np.mean((pred[valid]-dr[valid])**2)));nrmse=err/max(c['deltaR'],1e-12)
        profiles.append(dict(field=field,roi=c['roi'],scale_deltaR=float(amp),inferred_onset_s=float(onset),
                             effective_slow_s=float(decay),effective_fast_s=float(rise),rmse_deltaR=err,nrmse_peak=nrmse,
                             near_bound=bool(decay>119 or rise>.999 or rise<.051),status='empirical optical, not calcium kinetics'))
    # Strict same-field held-out washout: all amplitudes/times/kinetics frozen from first control.
    validation=[];fig,ax=plt.subplots(2,3,figsize=(14,8))
    for i,p in enumerate([p for p in profiles if p['field']=='2380205D']):
        t,_,dr=z[f"2380205D_wash_roi{p['roi']}"]
        predictions={shift:p['scale_deltaR']*pulse_response(t,p['inferred_onset_s']+shift,3.,p['effective_slow_s'],p['effective_fast_s']) for shift in [-1.,0.,1.]}
        rms={str(shift):float(np.sqrt(np.mean((pred-dr)**2))) for shift,pred in predictions.items()}
        validation.append(dict(field=p['field'],roi=p['roi'],rmse_deltaR_frozen=rms['0.0'],timing_sensitivity_rmse=rms,
                               rmse_zero_response=float(np.sqrt(np.mean(dr**2))),
                               Pearson_r=float(np.corrcoef(predictions[0.],dr)[0,1])))
        a=ax.flat[i];a.plot(t,dr,'o-',label='Washout measured',ms=3)
        a.plot(t,predictions[0.],label='Predicted from first NMDA')
        a.fill_between(t,np.min(list(predictions.values()),axis=0),np.max(list(predictions.values()),axis=0),alpha=.15,label='Onset +/-1 s')
        a.set(title='2380205D ROI '+str(p['roi']),xlabel='Time (s)',ylabel='Delta F340/F380');a.legend(fontsize=7)
    ax.flat[-1].axis('off');ax.flat[-1].text(.02,.9,'Same cells, later application\nNot independent donors\n3 s NMDA; no fitted washout parameters\nFura-2 ratio, not micromolar Ca',va='top')
    fig.tight_layout();fig.savefig(B/'dandrea-washout-check.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(14,8))
    for key,t,dr,valid in curves:
        field,cond,roi=key.split('_');a=ax[['2380202D','2380205D'].index(field),['first','drug','wash'].index(cond)]
        a.plot(t,np.where(valid,dr,np.nan),label=roi);a.set(title=field+' '+cond,xlabel='Time (s)',ylabel='Delta F340/F380');a.legend(fontsize=6,ncol=4)
    fig.tight_layout();fig.savefig(B/'dandrea-qc-traces.png',dpi=150);plt.close(fig)
    summary=[]
    for field in ['2380202D','2380205D']:
        values=[r for r in effects if r['field']==field and r['control_snr_ok']]
        summary.append(dict(field=field,positive_control_rois=len(values),drug_ratio_median=float(np.median([r['drug_over_control'] for r in values])),
                            wash_ratio_median=float(np.median([r['wash_over_control'] for r in values]))))
    output=dict(author='codex',date='2026-10-16',source='DAndrea2024 Fig6, 0Mg, SL FUS-eGFP reporter culture',
                 scope='Soma-enriched cores, not certified individually segmented motor neurons; exploratory ROI/field summaries, no population inference',
                 excluded_frames_registration_correlation_lt_08=invalid,profiles=profiles,paired_effects=effects,field_summary=summary,washout_checks=validation,
                 timing='Three-second pulse from Methods; onset estimated from control fluorescence, not known stimulus marker; +/-1s sensitivity only, not a statistical CI',
                 nonidentifiability='Fast and slow filters commute. Neither tau can be uniquely assigned to indicator, buffering, pump, receptor or organelles.')
    (B/'dandrea-response-profile.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    for name,rr in [('dandrea-qc-metrics',rows),('dandrea-paired-effects',effects)]:
        with (B/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=rr[0].keys());w.writeheader();w.writerows(rr)
    print(json.dumps(dict(field_summary=summary,profiles=profiles,validation=validation),indent=2))
if __name__=='__main__':main()
