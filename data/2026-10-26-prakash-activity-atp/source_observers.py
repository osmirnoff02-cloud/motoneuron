"""[codex] Source observations, not interchangeable human kinetic parameters."""
from dataclasses import dataclass
import numpy as np

RANGARAJU2014=dict(DOI='10.1016/j.cell.2013.12.042',system='Rat CA1-CA3 primary hippocampal culture, 13-25 DIV, 37 C; not human motor neuron/NMJ',ATP_rest_uM=1400.,frequency_Hz=10.,stimulus_s=60.,post_average_s=360.,luminescence_exposure_s=60.,control_stim_mean_drop_uM=12.,control_stim_SEM_uM=70.,control_post_mean_drop_uM=150.,control_post_SEM_uM=20.,control_n=30,oligo_stim_mean_drop_uM=350.,oligo_stim_SEM_uM=20.,oligo_post_mean_drop_uM=280.,oligo_post_SEM_uM=10.,oligo_n=9,dGlu_stim_mean_drop_uM=440.,dGlu_stim_SEM_uM=100.,dGlu_post_mean_drop_uM=270.,dGlu_post_SEM_uM=40.,dGlu_n=21,dGlu_unstimulated_tau_s=37.*60,dGlu_tau_SEM_s=5.*60,source='Fig3B, Methods Baseline Measurements/pH Correction, FigS3E. Positive drop=baseline-ATP; SEM is not a physiological range. dGlu uses exponentially extrapolated baseline; control/oligo use prestimulus mean. No raw paired measurements supplied.')

def synatp_observe(ATP_uM,pH,*,Km_pH7_uM=2300.,luciferase_pKa=7.03,LFmax=1.):
    a,ph=np.broadcast_arrays(np.asarray(ATP_uM,float),np.asarray(pH,float))
    if not np.isfinite(a).all() or not np.isfinite(ph).all() or np.min(a)<0 or min(Km_pH7_uM,LFmax)<=0:raise ValueError('Invalid ATP/pH/calibration')
    k=Km_pH7_uM*(1+10**(luciferase_pKa-ph))/(1+10**(luciferase_pKa-7.))
    return LFmax*a/(k+a)

def synatp_invert(LF,pH,*,Km_pH7_uM=2300.,luciferase_pKa=7.03,LFmax=1.):
    x,ph=np.broadcast_arrays(np.asarray(LF,float),np.asarray(pH,float))
    if not np.isfinite(x).all() or not np.isfinite(ph).all() or np.min(x)<0 or np.max(x)>=LFmax or min(Km_pH7_uM,LFmax)<=0:raise ValueError('Signal must be below calibrated LFmax')
    k=Km_pH7_uM*(1+10**(luciferase_pKa-ph))/(1+10**(luciferase_pKa-7.))
    return k*x/(LFmax-x)

def exposure_means(t,y,edges):
    """Time-integrated exposures; do not mistake nonlinear means for instant ATP."""
    t=np.asarray(t,float);y=np.asarray(y,float);edges=np.asarray(edges,float)
    if len(t)!=len(y) or np.any(np.diff(t)<=0) or np.any(np.diff(edges)<=0) or edges[0]<t[0] or edges[-1]>t[-1]:raise ValueError('Ordered covered sample/exposure times required')
    return np.array([np.trapezoid(np.interp(np.r_[a,t[(t>a)&(t<b)],b],t,y),np.r_[a,t[(t>a)&(t<b)],b])/(b-a) for a,b in zip(edges[:-1],edges[1:])])

@dataclass(frozen=True)
class ProteasomeEntryParameters:
    engagement_per_s:float=.1
    release_per_s:float=.1
    accessibility:float=1.
    provenance:str='Hypothetical competition after binding, motivated by Prakash2004 discussion/Fig6; no human TDP rates measured.'
    def probability(self):
        v=np.array([self.engagement_per_s,self.release_per_s,self.accessibility])
        if not np.isfinite(v).all() or np.min(v)<0 or self.accessibility>1 or not self.provenance:raise ValueError('Invalid competing entry rates')
        e=self.engagement_per_s*self.accessibility
        return float(e/(e+self.release_per_s)) if e+self.release_per_s>0 else 0.

ENTRY_RANGES=dict(engagement_per_s=[.01,.1,1.],release_per_s=[.01,.1,1.],accessibility=[0.,.5,1.])
PRAKASH2004=dict(DOI='10.1038/nsmb814',system='Rabbit reticulocyte lysate with engineered bacterial DHFR/barnase; not neurons/TDP',ATP_uM=500.,creatine_phosphate_mM=10.,ubiquitin_uM=25.,temperature_C=25.,MTX_uM=320.,notes='Fig1 at30C; other degradation assays25C. Binding and initiation are separable; internal and terminal initiation sites; release competes with engagement; partial processing can leave stable fragments. 20aa statement is data-not-shown and not a universal threshold. A new TDP fragment production route is NOT inferred from this assay.')
