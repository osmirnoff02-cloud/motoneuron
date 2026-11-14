"""Codex 2026-10-19. Source-informed assay models; NOT fitted human kinetics.

Gerencser: five-channel unmixing and corrected FRET, reversible optical effects.
Robinson: replacement of cleaved sensor by newly synthesized intact molecules.
Yang: competing protease origins; molecular mass alone is not enzyme identity.
Old electrical and Ca/calpain/TDP models remain available without modification.
"""
import numpy as np
from scipy.linalg import expm


def _time(time_s):
    t=np.asarray(time_s,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Strictly increasing finite seconds required')
    return t


def _series(value,t,name,upper=None):
    x=np.broadcast_to(np.asarray(value,float),t.shape)
    if not np.isfinite(x).all() or np.any(x<0) or (upper is not None and np.any(x>upper)):
        raise ValueError(name+' outside its nonnegative finite range')
    return x


def unmix_fura_fret(observed,mixing_matrix,*,background,provenance):
    """Rows are samples; channel order YFP,CFP,FRET,340,380.

    observed = corrected @ mixing_matrix.T + background.
    Matrix coefficients MUST come from single-fluorophore calibration; none here
    are taken from processed RGB videos. Fura6F ratio is not absolute calcium.
    Corrected FRET = FRET/sqrt(CFP*YFP), NOT FRET/(CFP*YFP).
    """
    a=np.asarray(observed,float);m=np.asarray(mixing_matrix,float)
    if a.ndim!=2 or a.shape[1]!=5 or m.shape!=(5,5) or not np.isfinite(a).all() or not np.isfinite(m).all():
        raise ValueError('Finite N x 5 intensities and 5 x 5 mixing matrix required')
    if not provenance.strip():raise ValueError('Calibration provenance required')
    condition=float(np.linalg.cond(m))
    if not np.isfinite(condition) or condition>1e6:raise ValueError('Ill-conditioned spectral calibration')
    bg=np.broadcast_to(np.asarray(background,float),a.shape)
    if not np.isfinite(bg).all():raise ValueError('Finite background required')
    y=np.linalg.solve(m,(a-bg).T).T
    valid=(y[:,0]>0)&(y[:,1]>0)&(y[:,2]>=0)&(y[:,3]>=0)&(y[:,4]>0)
    fret=np.full(len(a),np.nan);fura=np.full(len(a),np.nan)
    fret[valid]=y[valid,2]/np.sqrt(y[valid,0]*y[valid,1]);fura[valid]=y[valid,3]/y[valid,4]
    return dict(unmixed=y,fret_corrected=fret,fura_ratio=fura,valid=valid,
        mixing_condition_number=condition,reconstruction_max_error=float(np.max(abs(y@m.T+bg-a))),
        calcium_uM=None,enzyme_activity=None,provenance=provenance,
        limitation='Optical ratios only; reversible pH/Cl/conformation effects may remain')


def renewing_reporter(time_s,activity,*,cleavage_per_s,synthesis_per_s,
                     intact_turnover_per_s,product_turnover_per_s,
                     initial=(1.,0.,1.,0.),other_cleavage_per_s=0.,
                     illumination_fraction=0.,bleach_per_illuminated_s=0.,
                     gain=1.,intact_yield=1.,product_yield=0.,background=0.,
                     optical_multiplier=1.,provenance):
    """Exact propagation of a replenishable reporter with bright/dark fractions.

    States: total intact U, total cleaved C, bright intact BU, bright cleaved BC.
    Newly synthesized reporter is bright: bleaching is NOT applied retroactively
    to molecules made later. Inputs are left-constant between time knots.
    Reversible optical multiplier is an independently supplied nuisance input,
    not proteolysis or a fitted pH mechanism. Rates are assay-specific assumptions
    until fitted to raw sensor controls. Does not reverse cleavage of any molecule.
    """
    t=_time(time_s);e=_series(activity,t,'activity');s=_series(synthesis_per_s,t,'synthesis')
    other=_series(other_cleavage_per_s,t,'other cleavage');light=_series(illumination_fraction,t,'illumination',1.)
    optical=_series(optical_multiplier,t,'optical multiplier')
    pars=np.array([cleavage_per_s,intact_turnover_per_s,product_turnover_per_s,
                   bleach_per_illuminated_s,gain,intact_yield,product_yield,background],float)
    y=np.asarray(initial,float)
    if not np.isfinite(pars).all() or np.any(pars<0) or not provenance.strip():raise ValueError('Nonnegative finite rates/optics and provenance required')
    if y.shape!=(4,) or not np.isfinite(y).all() or np.any(y<0) or y[2]>y[0] or y[3]>y[1]:raise ValueError('Invalid total/bright reporter state')
    states=[y.copy()]
    for i,dt in enumerate(np.diff(t)):
        q=cleavage_per_s*e[i]+other[i];d=intact_turnover_per_s;dc=product_turnover_per_s
        bl=bleach_per_illuminated_s*light[i]
        m=np.array([[-d-q,0,0,0,s[i]],[q,-dc,0,0,0],
                    [0,0,-d-q-bl,0,s[i]],[0,0,q,-dc-bl,0],[0,0,0,0,0]],float)
        y=(expm(m*dt)@np.r_[y,1.])[:4];states.append(y.copy())
    states=np.asarray(states)
    signal=gain*optical*(intact_yield*states[:,2]+product_yield*states[:,3])+background
    if np.min(states)<-1e-10:raise ArithmeticError('Negative reporter amount')
    return dict(time_s=t,total_intact=states[:,0],total_cleaved=states[:,1],
        bright_intact=states[:,2],bright_cleaved=states[:,3],signal=signal,
        background_subtracted_signal=signal-background,provenance=provenance,
        calibration='No human synthesis/turnover/optical constants inferred',
        reporter_repair=False,TDP_repair=None,enzyme_activity_inferred=None)


def sample_photometry(time_s,signal,windows_s,*,background,provenance):
    """Window-average an interpolated signal, then baseline-relative percent.

    Mirrors a measurement transform, not a fit: time points are not replicates.
    background is matched LED-off/background at each acquisition window.
    Illumination history must already be included in the reporter dynamics.
    """
    t=_time(time_s);f=np.asarray(signal,float);w=np.asarray(windows_s,float)
    if f.shape!=t.shape or not np.isfinite(f).all() or w.ndim!=2 or w.shape[1]!=2 or not np.isfinite(w).all():raise ValueError('Finite signal and Nx2 windows required')
    if np.any(w[:,0]<t[0]) or np.any(w[:,1]>t[-1]) or np.any(w[:,1]<=w[:,0]) or not provenance.strip():raise ValueError('Windows outside signal or invalid provenance')
    values=[]
    for lo,hi in w:
        tt=np.r_[lo,t[(t>lo)&(t<hi)],hi]
        values.append(np.trapezoid(np.interp(tt,t,f),tt)/(hi-lo))
    values=np.asarray(values)-np.broadcast_to(np.asarray(background,float),(len(w),))
    if not np.isfinite(values).all() or values[0]<=0:raise ValueError('Positive baseline after matched background subtraction required')
    return dict(window_mean=values,delta_percent=100*(values-values[0])/values[0],windows_s=w,
                provenance=provenance,independent_replicates=None)


def competing_tdp_chase(time_s,calpain_activity,caspase_activity,*,calpain_rate_per_s,
                        caspase_rate_per_s,calpain_product_clearance_per_s=0.,
                        caspase_product_clearance_per_s=0.,initial=(1.,0.,0.,0.),provenance):
    """Labelled TDP cohort with two competing origins; no new synthesis.

    Each cleavage transfers ONE tracked lineage into an origin-specific pool;
    complementary fragments are not double-counted as protein mass. Pools do not
    imply a particular kDa band or N/C epitope. This is a separate short-assay
    module; it does not replace the old P->F->R cellular cascade. Caspase input
    is independent and is never inferred automatically from calcium.
    """
    t=_time(time_s);cp=_series(calpain_activity,t,'calpain');cs=_series(caspase_activity,t,'caspase')
    rates=np.asarray([calpain_rate_per_s,caspase_rate_per_s,calpain_product_clearance_per_s,caspase_product_clearance_per_s],float)
    y=np.asarray(initial,float)
    if not np.isfinite(rates).all() or np.any(rates<0) or y.shape!=(4,) or not np.isfinite(y).all() or np.any(y<0) or not provenance.strip():raise ValueError('Finite nonnegative kinetics/state and provenance required')
    out=[y.copy()]
    for i,dt in enumerate(np.diff(t)):
        a,b=rates[0]*cp[i],rates[1]*cs[i];dc,ds=rates[2:]
        m=np.array([[-a-b,0,0,0],[a,-dc,0,0],[b,0,-ds,0],[0,dc,ds,0]])
        y=expm(m*dt)@y;out.append(y.copy())
    out=np.asarray(out);balance=float(np.max(abs(out.sum(axis=1)-out[0].sum())))
    return dict(time_s=t,intact_label=out[:,0],calpain_origin=out[:,1],caspase_origin=out[:,2],
        removed_lineages=out[:,3],lineage_balance_error=balance,provenance=provenance,
        band_mass_kDa=None,epitope=None,nuclear_function=None,survival=None,
        status='Assay-specific conditional competition, not human kinetic calibration')


def observe_fragment_band(species_amounts,*,recognition_weights,species_metadata,substrate,epitope,provenance):
    """Explicit observation matrix: no inference of origin from a 25/35kDa label.

    Caller must supply assay/antibody-specific weights for each molecular species.
    TDP N-lineage cannot silently serve as spectrin SNTF or C-terminal TDP band.
    """
    if substrate not in ('TDP43','spectrin') or epitope not in ('N','C','internal','neoepitope') or not provenance.strip():raise ValueError('Explicit substrate, epitope and provenance required')
    a=np.asarray(species_amounts,float);w=np.asarray(recognition_weights,float)
    if a.ndim!=2 or w.shape!=(a.shape[1],) or not np.isfinite(a).all() or not np.isfinite(w).all() or np.any(a<0) or np.any(w<0):raise ValueError('Nonnegative sample x species amounts and recognition weights required')
    if len(species_metadata)!=a.shape[1]:raise ValueError('One identity record per molecular species required')
    for weight,identity in zip(w,species_metadata):
        if weight>0 and (identity.get('substrate')!=substrate or epitope not in identity.get('epitopes',[])):
            raise ValueError('Positive antibody weight assigned to wrong substrate/epitope')
    return dict(signal=a@w,substrate=substrate,epitope=epitope,provenance=provenance,
                enzyme_origin_identifiable_from_band_alone=False)
