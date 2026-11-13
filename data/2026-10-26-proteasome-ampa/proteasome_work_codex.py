"""[codex] 2026-10-26. Optional ATPase-pool bookkeeping, not human calibration.

Henderson2011: yeast, Rpn10 tethered non-TDP substrates. ATPase continues
without successful degradation. Pool size and transfer to human MN are unknown.
The inherited TDP loss laws are retained; this is NOT finite enzyme occupancy.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from proteostasis_parameters_codex import proteasome_factor
from proteostasis_parameters_codex import proteostasis_fluxes

@dataclass(frozen=True)
class ProteasomeWorkParameters:
    enabled: bool=True
    soma_pool_nM: float=100.
    process_pool_nM: float=100.
    ATPase_per_min_per_26S: float=110.
    ATPase_half_uM: float=300.
    ATPase_hill: float=1.
    ATP_dependent_TDP_loss: bool=True
    provenance: str='Henderson2011 Table2: 110 ATP/min/doubly capped yeast26S without substrate. Pool amounts and human transfer unknown; separate ATPase and productive loss, not a fitted full proteasome.'
    def vector(self):
        a=np.array([float(self.enabled),self.soma_pool_nM,self.process_pool_nM,self.ATPase_per_min_per_26S,self.ATPase_half_uM,self.ATPase_hill,float(self.ATP_dependent_TDP_loss)])
        if not isinstance(self.enabled,bool) or not isinstance(self.ATP_dependent_TDP_loss,bool) or not np.isfinite(a).all() or np.any(a[1:6]<=0) or not self.provenance.strip():raise ValueError('Positive finite proteasome pool/ATPase parameters required')
        return a

PROTEASOME_WORK_RANGES=dict(soma_pool_nM=[10.,100.,500.],process_pool_nM=[10.,100.,500.],ATPase_per_min_per_26S=[30.,110.,140.],ATPase_half_uM=[15.,300.,1000.],ATPase_hill=[1.,2.],processing_efficiency=[0.,.5,1.],ATPase_support=[0.,.5,1.])

@njit
def work_loss(z,old,ctx,row,prv,wv,support,folding,efficiency,ATPase_support):
    proxy=z
    if wv[0] and not wv[6]:
        proxy=z.copy();proxy[120]=ctx[14][0];proxy[136]=ctx[14][0]
    return proteostasis_fluxes(proxy,old,ctx,row,prv,support*(efficiency*ATPase_support if wv[0] else 1.),folding)

@njit
def pool_ATP_flux(atp,atp0,wv,j,support,ATPase_support):
    a=max(atp,0.)**wv[5];a0=atp0**wv[5];k=wv[4]**wv[5]
    return wv[1+j]*.001*wv[3]/60.*a/(k+a)*(k+a0)/a0*support*ATPase_support

@njit
def work_costs(z,out,ups,bpv,prv,wv,folding,j,atp0,support,ATPase_support):
    total=max(out[19 if j==0 else 177],0.)
    if not wv[0]:
        return ups[j]*bpv[9]*(prv[4]*folding+prv[5]),max(total-ups[j],0.)*bpv[9]*prv[6]
    # Replace the per-cleared-substrate catalytic cost, do not add it twice.
    pool=pool_ATP_flux(z[120 if j==0 else 136],atp0,wv,j,support,ATPase_support)
    return pool+ups[j]*bpv[9]*prv[5],max(total-ups[j],0.)*bpv[9]*prv[6]

@njit
def work_budget(z,out,bio,ups,bpv,prv,wv,folding,atp0,support,ATPase_support):
    if not wv[0]:return out
    result=out.copy()
    for j in range(2):
        at=120 if j==0 else 136;other=125 if j==0 else 150;c=267+4*j
        old=out[c+2]+out[c+3]
        u,n=work_costs(z,bio,ups,bpv,prv,wv,folding,j,atp0,support,ATPase_support)
        delta=u+n-old
        result[at]-=delta;result[at+1]+=delta;result[other]+=delta
        result[255+10*j]=u+n;result[c+2]=u;result[c+3]=n
    return result

# Source fits: no automatic transfer to the cell. ± columns preserved verbatim;
# article does not identify them as confidence intervals, so neither do we.
HENDERSON_KINETICS={
 'R-I27-ext':(145.,17.,.0248,.0009,40.3,.97),
 'R-I27V13P-ext':(247.,33.,.118,.007,8.5,.96),
 'R-DHFR-ext':(325.,30.,.197,.008,5.1,.98),
 'R-DHFR-ext+MTX':(271.,51.,.073,.006,13.7,.92),
 'R-DHFR':(176.,24.,.022,.001,45.5,.95)}
HENDERSON_ATPASE={'None':(110.,12.,4),'R-I27':(123.,8.2,4),'R-I27V13P':(136.,23.,4),'R-I27-ext':(138.,.27,2),'R-I27V13P-ext':(140.,3.2,4),'R-DHFR':(134.,13.,4),'R-DHFR+MTX':(123.,5.9,2),'R-DHFR-ext':(116.,21.,2),'R-DHFR-ext+MTX':(118.,1.,2)}

def henderson_reference(substrate,substrate_nM=1500.,proteasome_nM=50.):
    if substrate not in HENDERSON_KINETICS or not np.isfinite([substrate_nM,proteasome_nM]).all() or substrate_nM<0 or proteasome_nM<=0:raise ValueError('Known source substrate and nonnegative concentration required')
    km,ke,kcat,ce,t,r2=HENDERSON_KINETICS[substrate];rate,spread,n=HENDERSON_ATPASE[substrate]
    throughput=proteasome_nM*kcat*substrate_nM/(km+substrate_nM)
    return dict(substrate=substrate,Km_nM=km,Km_reported_pm=ke,kcat_per_min=kcat,kcat_reported_pm=ce,published_turnover_min=t,R_squared=r2,throughput_nM_min=throughput,ATPase_per_min_per_26S=rate,ATPase_reported_pm=spread,ATPase_n=n,ATPase_baseline_per_min_per_26S=110.,ATP_flux_scaled_to_requested_proteasome_nM_min=proteasome_nM*rate,ATPase_assay_proteasome_nM=30.,ATPase_assay_substrate_nM=1500.,apparent_saturating_total_ATP_per_substrate=rate/kcat,apparent_baseline_subtracted_ATP_per_substrate=(rate-110.)/kcat,ATPase_concentration_response_measured=False,ATP_assay_mM=5.,degradation_temperature_C=30.,source='Henderson2011 Tables1/2; purified doubly-capped yeast26S; no TDP/human fit. ATPase measured separately at 1.5uM, cost ratios are our calculation, not matched single-molecule costs; baseline subtraction uncertain.')
