"""[codex] 2026-10-23. Substrate-specific phosphorylation in the integrated chain.

Yamashita 2016 supports altered full-length susceptibility, not human rates,
not universal protection of N fragments, and not restored RNA function.
All prior APIs remain available. The new branch is explicitly opt-in.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util, sys, json, math
import numpy as np
from numba import njit

_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_before_yamashita20261023',_HERE.parent/'2026-10-23-six-papers-calpain-stmn2/model_api.py')
_prior=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_prior;_spec.loader.exec_module(_prior)
for _name in dir(_prior):
    if not _name.startswith('_') and _name not in globals():globals()[_name]=getattr(_prior,_name)


@dataclass(frozen=True)
class TDPPhosphorylationParameters:
    """Effective U <-> P full-length states, NOT site-resolved occupancy.

    Native non-phosphorylated full-length cleavage = k1*A*U;
    phosphorylated full-length cleavage = r*k1*A*P, 0 <= r <= 1.
    Large N products do not inherit this multiplier. Their independent k2 is
    retained. Equal basal clearance of U/P and equal RNA function per intact
    molecule are provisional assumptions, not results of Yamashita.
    """
    phosphorylation_per_s: float
    dephosphorylation_per_s: float
    phosphorylated_cleavage_multiplier: float
    provenance: str

    def vector(self):
        if not isinstance(self.provenance,str) or not self.provenance.strip():raise ValueError('Explicit provenance required')
        p=np.array([self.phosphorylation_per_s,self.dephosphorylation_per_s,self.phosphorylated_cleavage_multiplier],float)
        if not np.isfinite(p).all() or np.any(p<0) or p[2]>1:raise ValueError('Nonnegative finite rates and multiplier in [0,1] required')
        return p


def phospho_tdp_basal_state(activity,tdp_parameters,phosphorylation_parameters):
    """Normalize U+P=1 in a conditional equilibrium; synthesis enters U only.

    Each sensitivity arm is separately baseline matched. This is NOT an acute
    kinase treatment of an already existing cell, nor measured human fractions.
    """
    p=tdp_parameters.vector();kp,kd,r=phosphorylation_parameters.vector()
    if not np.isfinite(activity) or activity<0:raise ValueError('Nonnegative basal activity required')
    q=p[5]*activity;z=kp/(kp+kd+p[7]+r*q)
    first=q*((1-z)+r*z);large=first/(p[8]+p[6]*activity)
    return dict(pools=np.array([1-z,z,large,p[6]*activity*large/p[9]]),
                synthesis_per_s=float(p[7]+first),phosphorylated_fraction=float(z))


@njit
def _rhs(y,a,s,p,pp):
    kp,kd,r=pp
    ph=kp*y[0]-kd*y[1]
    qU=p[5]*a*y[0];qP=p[5]*a*r*y[1];qN=p[6]*a*y[2]
    du=p[7]*y[0];dp=p[7]*y[1];dn=p[8]*y[2];ds=p[9]*y[3]
    return np.array([s-du-qU-ph,ph-dp-qP,qU+qP-qN-dn,qN-ds,
                     du+dp+dn+ds,qU+qP,qN,dn,s,qU,qP])


@njit
def _integrate(t,a,y0,s,p,pp,max_step):
    out=np.empty((len(t),11));out[0]=y0
    safe=min(max_step,.1/max(1e-12,p[5]*np.max(a)+p[7]+pp[0],
             p[5]*np.max(a)*pp[2]+p[7]+pp[1],p[6]*np.max(a)+p[8],p[9]))
    for i in range(len(t)-1):
        n=max(1,int(math.ceil((t[i+1]-t[i])/safe)));h=(t[i+1]-t[i])/n;y=out[i].copy()
        for j in range(n):
            a0=a[i]+(a[i+1]-a[i])*j/n;am=a[i]+(a[i+1]-a[i])*(j+.5)/n;a1=a[i]+(a[i+1]-a[i])*(j+1)/n
            k1=_rhs(y,a0,s,p,pp);k2=_rhs(y+h*k1/2,am,s,p,pp)
            k3=_rhs(y+h*k2/2,am,s,p,pp);k4=_rhs(y+h*k3,a1,s,p,pp)
            y=y+h*(k1+2*k2+2*k3+k4)/6
        out[i+1]=y
    return out


def simulate_phospho_tdp_from_activity(time_s,effective_activity,*,parameters,
        phosphorylation_parameters,initial_TDP,synthesis_per_s,activity_provenance,max_step_s=.02):
    """Four N-lineage pools [U,P,large N,small N] and cumulative accounting.

    Activity is linearly interpolated between supplied times; no hidden enzyme
    bolus or extrapolation. Cleavage removes C-terminal sites from this N lineage.
    Complementary C products/aggregation/kinase-induced localization are not
    simulated. Existing C-fragment chase and antibody observers stay separate.
    """
    p=parameters.vector();pp=phosphorylation_parameters.vector()
    t=np.asarray(time_s,float);a=np.asarray(effective_activity,float);y0=np.asarray(initial_TDP,float)
    if t.ndim!=1 or len(t)<2 or a.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(a).all() or np.any(np.diff(t)<=0) or np.any(a<0):raise ValueError('Finite increasing time and nonnegative matching activity required')
    if y0.shape!=(4,) or not np.isfinite(y0).all() or np.any(y0<0):raise ValueError('Four nonnegative pools required')
    if not np.isfinite([synthesis_per_s,max_step_s]).all() or synthesis_per_s<0 or max_step_s<=0:raise ValueError('Nonnegative synthesis and positive integration step required')
    if not isinstance(activity_provenance,str) or not activity_provenance.strip():raise ValueError('Activity provenance required')
    z=_integrate(t,a,np.r_[y0,np.zeros(7)],synthesis_per_s,p,pp,max_step_s)
    balance=z[:,:5].sum(axis=1)-z[:,8];err=float(np.max(abs(balance-balance[0])))
    if not np.isfinite(z).all() or z.min()<-1e-9 or err>1e-8*max(1.,float(balance[0])):raise ArithmeticError('N-lineage positivity or accounting failed')
    names=('unphosphorylated_intact','phosphorylated_intact','large_N_fragment','small_N_product',
           'removed_lineages','formed_large_cumulative','secondary_cleavage_cumulative',
           'large_removed_cumulative','synthesized_cumulative','cleavage_from_U','cleavage_from_P')
    out={name:z[:,i] for i,name in enumerate(names)}
    out.update(time_s=t,effective_activity=a,intact_TDP=z[:,0]+z[:,1],
       formation_rate_per_s=p[5]*a*(z[:,0]+pp[2]*z[:,1]),lineage_balance_max_error=err,
       parameters=asdict(parameters),phosphorylation_parameters=asdict(phosphorylation_parameters),
       activity_provenance=activity_provenance,human_kinetic_fit=False,electrical_feedback=False)
    return out


def chain_from_calcium(time_s,ca_uM,*,phosphorylation_parameters=None,
        stmn2_functional_fraction=1.,stmn2_rna_retention=1.,splice_rescue_fraction=0.,
        stmn2_drive_provenance=None,**kwargs):
    """Optional phospho branch; absent option preserves the old chain exactly.

    Enzyme and calcium dynamics are unchanged. Recompute TDP and its downstream
    localization/STMN2, respecting the original recorded time domain. Equal
    functional activity per U/P molecule is explicit; the existing independent
    functional input remains available and must not be inferred from a pTDP band.
    """
    if phosphorylation_parameters is None:
        return _prior.chain_from_calcium(time_s,ca_uM,stmn2_functional_fraction=stmn2_functional_fraction,
          stmn2_rna_retention=stmn2_rna_retention,splice_rescue_fraction=splice_rescue_fraction,
          stmn2_drive_provenance=stmn2_drive_provenance,**kwargs)
    phosphorylation_parameters.vector()
    modified=(isinstance(stmn2_functional_fraction,FractionSchedule) or isinstance(stmn2_rna_retention,FractionSchedule) or
        isinstance(splice_rescue_fraction,FractionSchedule) or np.ndim(stmn2_functional_fraction)!=0 or
        np.ndim(stmn2_rna_retention)!=0 or np.ndim(splice_rescue_fraction)!=0 or
        stmn2_functional_fraction!=1 or stmn2_rna_retention!=1 or splice_rescue_fraction!=0)
    if modified and (not isinstance(stmn2_drive_provenance,str) or not stmn2_drive_provenance.strip()):raise ValueError('Independent STMN2 intervention provenance required')
    # Validate schedules before the upstream simulation.
    for val in (stmn2_functional_fraction,stmn2_rna_retention,splice_rescue_fraction):_prior._fraction(val,np.asarray(time_s,float))
    out=_prior.chain_from_calcium(time_s,ca_uM,**kwargs)
    old_init=out['initial_state'];tp=kwargs['tdp_parameters'];lp=kwargs['localization_parameters']
    ini=phospho_tdp_basal_state(old_init['activity'],tp,phosphorylation_parameters)
    active=out['enzyme']['effective_activity']*(1-out['interventions']['calpain_block_fraction'])
    td=simulate_phospho_tdp_from_activity(time_s,active,parameters=tp,phosphorylation_parameters=phosphorylation_parameters,
        initial_TDP=ini['pools'],synthesis_per_s=ini['synthesis_per_s'],activity_provenance='Same Ca/CAST-gated activity as prior integrated model')
    out['TDP']=td
    old_init.update(TDP_phospho=ini['pools'],TDP=np.array([1.,ini['pools'][2],ini['pools'][3]]),
        TDP_synthesis_per_s=ini['synthesis_per_s'],phosphorylated_fraction=ini['phosphorylated_fraction'])
    f=lp.nuclear_fraction
    out['localization']=partition_intact_tdp(time_s,td['intact_TDP'],import_per_s=lp.exchange_per_s*f,
       export_per_s=lp.exchange_per_s*(1-f),synthesis_per_s=ini['synthesis_per_s'],synthesis_nuclear_fraction=f,
       initial_nuclear_fraction=f,provenance=lp.provenance+'; identical U/P partition assumed')
    st=drive_stmn2_from_chain(out,functional_fraction=stmn2_functional_fraction,
        splice_rescue_fraction=splice_rescue_fraction,stmn2_rna_retention=stmn2_rna_retention,
        parameters=kwargs['stmn2_parameters'],drive_provenance=stmn2_drive_provenance or
        'Equal splicing activity per intact U/P molecule assumed; not measured by Yamashita')
    out['STMN2']=st
    out['summary'].update(intact_TDP_end=float(td['intact_TDP'][-1]),
       phosphorylated_intact_end=float(td['phosphorylated_intact'][-1]),
       primary_cleavage_cumulative=float(td['formed_large_cumulative'][-1]),
       nuclear_TDP_relative_end=float(out['localization']['nuclear_intact'][-1]/old_init['nuclear_reference']))
    for key,channel in [('STMN2_mRNA_end','full_length_mrna'),('STMN2_mRNA_target_end','full_length_mrna_target'),('axonal_STMN2_proxy_end','axonal_stmn2_proxy'),('regrowth_support_end','regrowth_support_index')]:out['summary'][key]=float(st[channel][-1])
    out['checks'].update(TDP_lineage_balance=td['lineage_balance_max_error'],nuclear_partition_balance=out['localization']['partition_balance_error'])
    out['provenance']['TDP_phosphorylation']=asdict(phosphorylation_parameters)
    def describe(x):return asdict(x) if isinstance(x,FractionSchedule) else float(x)
    out['interventions'].update(stmn2_functional_fraction=describe(stmn2_functional_fraction),
        stmn2_rna_retention=describe(stmn2_rna_retention),splice_rescue_fraction=describe(splice_rescue_fraction))
    out['limitations']+=['U/P rates and multiplier assumed, not human calibration',
       'N fragment cleavage does not inherit full-length phospho multiplier',
       'Equal U/P basal clearance, localization and function are assumptions; aggregation not modeled']
    return out


def chain_from_electrical_trace(trace,*,basal_ca_uM,enzyme_parameters,tdp_parameters,
        localization_parameters,stmn2_parameters,compartment,calcium_sample_ms=.1,slow_dt_s=.05,**interventions):
    if compartment not in ('local','bulk'):raise ValueError('Choose local shell or bulk')
    if not np.isfinite(calcium_sample_ms) or calcium_sample_ms<=0:raise ValueError('Positive sampling interval required')
    t=np.asarray(trace['t']);dt=float(np.median(np.diff(t)))
    stride=max(1,int(round(calcium_sample_ms/dt)));ix=np.unique(np.r_[np.arange(0,len(t),stride),len(t)-1]).astype(int)
    r=chain_from_calcium(t[ix]/1000,np.asarray(trace['cs' if compartment=='local' else 'cb'])[ix],
        basal_ca_uM=basal_ca_uM,enzyme_parameters=enzyme_parameters,tdp_parameters=tdp_parameters,
        localization_parameters=localization_parameters,stmn2_parameters=stmn2_parameters,compartment=compartment,
        ca_provenance='Native NEURON Ca/SK/Vm; explicitly sampled',slow_dt_s=slow_dt_s,**interventions)
    r['sampling']=dict(source_dt_ms=dt,retained_Ca_dt_ms=stride*dt,STMN2_grid_s=slow_dt_s)
    return r


def run_integrated_chain(profile,*,adar2_activity,editing_parameters,preconditioning_h,mapping_family,
        editing_provenance,coupling_parameters,enzyme_parameters,tdp_parameters,localization_parameters,
        stmn2_parameters,compartment,protocol,calcium_sample_ms=.1,slow_dt_s=.05,**interventions):
    receptor=prepare_receptor_state(adar2_activity=adar2_activity,editing_parameters=editing_parameters,
        preconditioning_h=preconditioning_h,mapping_family=mapping_family,editing_provenance=editing_provenance)
    duration=protocol.get('pre_ms',200.)+protocol.get('duration_ms',500. if profile=='Burley' else 1000.)+protocol.get('post_ms',1800.)
    if duration>.01*min(editing_parameters.tau_rna_h,editing_parameters.tau_surface_h)*3600000:raise ValueError('Frozen receptor approximation invalid')
    electrical=run_coupled(profile,surface_Q=receptor['surface_Q'],parameters=coupling_parameters,**protocol)
    out=chain_from_electrical_trace(electrical,basal_ca_uM=coupling_parameters.basal_ca_uM,
        enzyme_parameters=enzyme_parameters,tdp_parameters=tdp_parameters,localization_parameters=localization_parameters,
        stmn2_parameters=stmn2_parameters,compartment=compartment,calcium_sample_ms=calcium_sample_ms,
        slow_dt_s=slow_dt_s,**interventions)
    out.update(receptor=receptor,electrical=electrical)
    return out


def source_training_data():
    result=_prior.source_training_data()
    result['Yamashita2016_full_length_phosphorylation']=json.loads((_HERE/'source-evidence.json').read_text(encoding='utf-8'))
    return result
