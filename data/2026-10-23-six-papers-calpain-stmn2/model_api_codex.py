"""[codex] 2026-10-23. Functional TDP/STMN2 interventions and R110 observation.

The upstream electrical/Ca/calpain/TDP dynamics are inherited unchanged.
Baughn supports a distinction between TDP amount and splicing function and a
downstream rescue; it does not measure our unknown human kinetic constants.
Gitler's dye transport is an Aplysia assay reference, not human calcium removal.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util,sys,json
import numpy as np
from scipy.fft import dct,idct

_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_before_six_sources_20261023',_HERE.parent/'2026-10-23-chou-cascella-supplement/model_api_codex.py')
_prior=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_prior;_spec.loader.exec_module(_prior)
for _name in dir(_prior):
    if not _name.startswith('_') and _name not in globals(): globals()[_name]=getattr(_prior,_name)

def _source(x):
    if not isinstance(x,str) or not x.strip():raise ValueError('Explicit provenance required')

@dataclass(frozen=True)
class FractionSchedule:
    """Declared left-constant input, including a coverage endpoint. No inferred history.

    Values are fractions [0,1]. The final value has no duration unless a later
    endpoint is supplied. Initial molecular pools remain those of the caller.
    """
    times_s: tuple
    values: tuple
    provenance: str
    def arrays(self):
        _source(self.provenance)
        t=np.asarray(self.times_s,float);v=np.asarray(self.values,float)
        if t.ndim!=1 or len(t)<2 or v.shape!=t.shape or not np.isfinite(t).all() or np.any(np.diff(t)<=0) or not np.isfinite(v).all() or np.any((v<0)|(v>1)):
            raise ValueError('Finite increasing schedule times and matching fractions [0,1] required')
        return t,v

def _fraction(x,t):
    if isinstance(x,FractionSchedule):
        ts,v=x.arrays()
        if ts[0]>t[0] or ts[-1]<t[-1]:raise ValueError('Schedule must cover the entire supplied time domain')
        return v[np.searchsorted(ts,t,side='right')-1]
    a=np.asarray(x,float)
    if a.ndim!=0 or not np.isfinite(a) or not 0<=a<=1:raise ValueError('A scalar fraction or FractionSchedule is required')
    return np.full(t.shape,float(a))

def drive_stmn2_from_chain(chain,*,functional_fraction,splice_rescue_fraction,
                          stmn2_rna_retention,parameters,drive_provenance):
    """Replace only the downstream functional drive, preserving upstream trajectories.

    Fractional function is independent of intact nuclear amount. It is not
    inferred from a mutation name, a TDP band, or immunofluorescence. Rescue is
    correction of splicing, NOT an ASO concentration or a generic drug effect.
    RNA retention permits direct STMN2 knockdown as a distinct intervention.
    All schedule change points are inserted exactly into the slow grid.
    """
    _source(drive_provenance)
    original=np.asarray(chain['STMN2']['time_s'],float)
    schedules=(functional_fraction,splice_rescue_fraction,stmn2_rna_retention)
    extra=[]
    for s in schedules:
        if isinstance(s,FractionSchedule):
            ts,_=s.arrays();extra.extend(ts[(ts>=original[0])&(ts<=original[-1])])
    t=np.unique(np.r_[original,extra])
    loc=chain['localization']
    n=np.interp(t,loc['time_s'],loc['nuclear_intact'])/chain['initial_state']['nuclear_reference']
    if chain['interventions']['freeze_tdp_input_to_stmn2']:n=np.ones_like(n)
    f,s,r=(_fraction(x,t) for x in schedules)
    st=simulate_stmn2_regrowth(t,n,functional_fraction=f,splice_rescue_fraction=s,
        stmn2_rna_retention=r,parameters=parameters,initial_state=chain['initial_state']['STMN2'],drive_provenance=drive_provenance)
    return st

def chain_from_calcium(time_s,ca_uM,*,stmn2_functional_fraction=1.,
        stmn2_rna_retention=1.,splice_rescue_fraction=0.,stmn2_drive_provenance=None,**kwargs):
    """Acute cascade with separately declared STMN2 function and intervention schedules.

    Default reproduces the previous chain exactly. All altered conditions begin
    from the same untreated basal state. This is not chronic preconditioning.
    """
    extended=(isinstance(stmn2_functional_fraction,FractionSchedule) or
              isinstance(stmn2_rna_retention,FractionSchedule) or
              isinstance(splice_rescue_fraction,FractionSchedule) or
              np.ndim(stmn2_functional_fraction)!=0 or np.ndim(stmn2_rna_retention)!=0 or
              stmn2_functional_fraction!=1 or stmn2_rna_retention!=1)
    if not extended:
        return _prior.chain_from_calcium(time_s,ca_uM,splice_rescue_fraction=splice_rescue_fraction,**kwargs)
    _source(stmn2_drive_provenance)
    # Validate before expensive upstream calculations.
    for x in (stmn2_functional_fraction,stmn2_rna_retention,splice_rescue_fraction):_fraction(x,np.asarray(time_s,float))
    out=_prior.chain_from_calcium(time_s,ca_uM,splice_rescue_fraction=0.,**kwargs)
    st=drive_stmn2_from_chain(out,functional_fraction=stmn2_functional_fraction,
        splice_rescue_fraction=splice_rescue_fraction,stmn2_rna_retention=stmn2_rna_retention,
        parameters=kwargs['stmn2_parameters'],drive_provenance=stmn2_drive_provenance)
    out['STMN2']=st
    for key,channel in [('STMN2_mRNA_end','full_length_mrna'),('STMN2_mRNA_target_end','full_length_mrna_target'),('axonal_STMN2_proxy_end','axonal_stmn2_proxy'),('regrowth_support_end','regrowth_support_index')]:
        out['summary'][key]=float(st[channel][-1])
    def describe(x):return asdict(x) if isinstance(x,FractionSchedule) else float(x)
    out['interventions'].update(stmn2_functional_fraction=describe(stmn2_functional_fraction),stmn2_rna_retention=describe(stmn2_rna_retention),splice_rescue_fraction=describe(splice_rescue_fraction))
    out['provenance']['STMN2_functional_drive']=stmn2_drive_provenance
    out['limitations'].append('Baughn supports intervention topology, not calibrated mutation-to-function or human time constants')
    return out

def chain_from_electrical_trace(trace,*,basal_ca_uM,enzyme_parameters,tdp_parameters,
        localization_parameters,stmn2_parameters,compartment,calcium_sample_ms=.1,slow_dt_s=.05,**interventions):
    if compartment not in ('local','bulk'):raise ValueError('Choose local shell or bulk')
    if not np.isfinite(calcium_sample_ms) or calcium_sample_ms<=0:raise ValueError('Positive sampling interval required')
    t=np.asarray(trace['t']);dt=float(np.median(np.diff(t)))
    stride=max(1,int(round(calcium_sample_ms/dt)));ix=np.unique(np.r_[np.arange(0,len(t),stride),len(t)-1]).astype(int)
    r=chain_from_calcium(t[ix]/1000,np.asarray(trace['cs' if compartment=='local' else 'cb'])[ix],basal_ca_uM=basal_ca_uM,
       enzyme_parameters=enzyme_parameters,tdp_parameters=tdp_parameters,localization_parameters=localization_parameters,
       stmn2_parameters=stmn2_parameters,ca_provenance='Native NEURON Ca/SK/Vm trace; sample '+str(calcium_sample_ms)+' ms',
       compartment=compartment,slow_dt_s=slow_dt_s,**interventions)
    r['sampling']=dict(source_dt_ms=dt,retained_Ca_dt_ms=stride*dt,STMN2_grid_s=slow_dt_s)
    return r

def run_integrated_chain(profile,*,adar2_activity,editing_parameters,preconditioning_h,mapping_family,
        editing_provenance,coupling_parameters,enzyme_parameters,tdp_parameters,
        localization_parameters,stmn2_parameters,compartment,protocol,calcium_sample_ms=.1,slow_dt_s=.05,
        inactivation_parameters=None,**stmn2_interventions):
    receptor=prepare_receptor_state(adar2_activity=adar2_activity,editing_parameters=editing_parameters,
       preconditioning_h=preconditioning_h,mapping_family=mapping_family,editing_provenance=editing_provenance)
    total_ms=protocol.get('pre_ms',200.)+protocol.get('duration_ms',500. if profile=='Burley' else 1000.)+protocol.get('post_ms',1800.)
    if total_ms>0.01*min(editing_parameters.tau_rna_h,editing_parameters.tau_surface_h)*3600000:
        raise ValueError('Frozen surface approximation invalid: explicitly simulate receptor turnover')
    electrical=run_coupled(profile,surface_Q=receptor['surface_Q'],parameters=coupling_parameters,**protocol)
    result=chain_from_electrical_trace(electrical,basal_ca_uM=coupling_parameters.basal_ca_uM,
       enzyme_parameters=enzyme_parameters,tdp_parameters=tdp_parameters,localization_parameters=localization_parameters,
       stmn2_parameters=stmn2_parameters,compartment=compartment,calcium_sample_ms=calcium_sample_ms,slow_dt_s=slow_dt_s,
       inactivation_parameters=inactivation_parameters,**stmn2_interventions)
    result.update(receptor=receptor,electrical=electrical)
    return result

@dataclass(frozen=True)
class R110ObservationParameters:
    diffusion_um2_per_s: float
    leak_tau_s: float
    production_gain: float
    system: str
    provenance: str
    def validate(self):
        _source(self.system);_source(self.provenance)
        a=np.array([self.diffusion_um2_per_s,self.leak_tau_s,self.production_gain])
        if not np.isfinite(a).all() or a[0]<0 or a[1]<=0 or a[2]<0:raise ValueError('Nonnegative D/gain and positive dye leakage time required')

def gitler_r110_reference(*,system):
    """Measured dye D and leakage tau, not a calcium pump/activation fit."""
    if system!='Aplysia_B1_B2_axotomy':raise ValueError('Gitler values are restricted to their Aplysia R110 assay')
    return R110ObservationParameters(68.8,211.,1.,system,
        'Gitler & Spira 1998 Fig8, pp1132-1133: D=68.8 +/-15 um2/s n=8; tau=211 +/-66 s n=9; gain is arbitrary, NOT measured')

def simulate_r110_observation(time_s,x_um,proteolytic_drive,*,parameters,initial_product,drive_provenance):
    """P_t = D P_xx - P/tau + gain*drive; no-flux finite-volume boundary.

    Constant cross section, uniform cell centres and left-constant source per
    interval. Exact spectral propagation of the spatial discretization. Source
    represents substrate cleavage under excess substrate, not specific CAPN2.
    No inverse identification of enzyme activity and no human parameter transfer.
    Outputs are corrected dye-product units, not raw R490/F350 or FRET channels.
    One effective R110 pool; monoamide intermediates and substrate transport are
    not separately resolved. D/tau measured for R110 do not identify those steps.
    """
    _source(drive_provenance);parameters.validate()
    t=np.asarray(time_s,float);x=np.asarray(x_um,float);q=np.asarray(proteolytic_drive,float);y=np.asarray(initial_product,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Increasing finite time in seconds required')
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all() or np.any(np.diff(x)<=0) or not np.allclose(np.diff(x),np.diff(x)[0]):raise ValueError('Uniform increasing cell centres required')
    if q.shape!=(len(t),len(x)) or y.shape!=(len(x),) or not np.isfinite(q).all() or not np.isfinite(y).all() or np.any(q<0) or np.any(y<0):raise ValueError('Nonnegative finite source and initial product of correct shape required')
    dx=x[1]-x[0];p=parameters
    rate=4*p.diffusion_um2_per_s*np.sin(np.arange(len(x))*np.pi/(2*len(x)))**2/dx**2+1/p.leak_tau_s
    out=np.empty_like(q);out[0]=y;source_total=np.zeros(len(t));lost=np.zeros(len(t))
    for i,dt in enumerate(np.diff(t)):
        a=np.exp(-rate*dt);b=-np.expm1(-rate*dt)/rate
        out[i+1]=idct(a*dct(out[i],norm='ortho')+b*dct(p.production_gain*q[i],norm='ortho'),norm='ortho')
        if out[i+1].min()<-1e-10:raise ArithmeticError('Dye positivity failure')
        production=p.production_gain*q[i].sum()*dx*dt
        source_total[i+1]=source_total[i]+production
        lost[i+1]=lost[i]+production-(out[i+1].sum()-out[i].sum())*dx
    mass=out.sum(axis=1)*dx
    return dict(time_s=t.copy(),x_um=x.copy(),product=out,total_product=mass,
      produced_cumulative=source_total,leaked_cumulative=lost,
      balance_max_error=float(np.max(abs(mass-mass[0]-source_total+lost))),
      parameters=asdict(p),drive_provenance=drive_provenance,
      human_calcium_calibration=False,enzyme_activity_inferred=False,
      limitation='Transport and leakage of R110 reporter; cannot equate fluorescence to instant calpain activity or TDP cleavage')

def six_source_evidence():
    return json.loads((_HERE/'source-evidence_codex.json').read_text(encoding='utf-8'))

def source_training_data():
    out=_prior.source_training_data();out['six_source_20261023']=six_source_evidence();return out
