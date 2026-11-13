"""[codex] Source-specific optical observer; never changes neuronal ATP."""
import numpy as np
def straub_ateam(atp_mM,*,source_conditions_confirmed=False,calibration_variant='Methods2.16'):
    if not source_conditions_confirmed:
        raise ValueError('Confirm Straub2025 ATeam1.03YEMK calibration/source optical settings first; not a universal FRET conversion')
    a=np.asarray(atp_mM,float)
    if not np.isfinite(a).all() or np.any(a<0):raise ValueError('Finite nonnegative ATP in mM')
    rmins={'Methods2.16':.27,'Fig2C_fit':.25}
    if calibration_variant not in rmins:raise ValueError('Select Methods2.16 or Fig2C_fit; source lists different Rmin values')
    kd=1.71;n=2.68;rmin=rmins[calibration_variant];rmax=.66
    q=(a/kd)**n;ratio=rmin+(rmax-rmin)*q/(1+q)
    return dict(ratio=ratio,within_reported_dynamic_window=(a>=.2*kd)&(a<=2*kd),
      below_paper_baseline_ratio_cutoff=ratio<.62,
      source='Straub2025 doi:10.1111/jnc.70212 Methods2.11/2.16 and Fig2; mouse cerebellar boutons, near-physiological temperature; optical calibration, not MN ATP setpoint',
      calibration=dict(Kd_mM=kd,Hill=n,Rmin=rmin,Rmax=rmax,variant=calibration_variant),
      Rmin_note='Fig2C legend fit states0.25; Methods2.16 calculation states0.27 or individual values. Do not conflate optical calibration variants.')

def inverse_straub_ateam(ratio,calibration_variant='Methods2.16'):
    r=np.asarray(ratio,float)
    rmin={'Methods2.16':.27,'Fig2C_fit':.25}[calibration_variant]
    if not np.isfinite(r).all() or np.any(r<rmin) or np.any(r>=.66):raise ValueError('Ratio must be >=Rmin and <Rmax; saturated readings cannot be inverted')
    return 1.71*((r-rmin)/(.66-r))**(1/2.68)
