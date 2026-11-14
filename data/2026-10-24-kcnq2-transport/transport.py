"""[codex] 2026-10-24. Conserved, cargo-specific transport and KCNQ2 turnover.

34 normalized states, not molecule counts. Human KCNQ2 source gives endpoint
fold changes, not splice fractions, nuclear activity, or kinetic rates.
Lysosome motility, RNA attachment, and channel delivery are independent gates.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

@dataclass(frozen=True)
class TransportParameters:
    baseline_delta_fraction:float=.01
    delta_fold_reference:float=36.39
    assumed_reference_TDP_function:float=.15
    splice_power:float=1.
    rna_half_h:float=4.
    channel_half_h:float=24.
    er_export_h:float=2.
    golgi_delivery_h:float=1.
    mutant_export_fraction:float=.1
    wt_retention_gain:float=1.
    membrane_recycle_h:float=12.
    channel_axon_share:float=.3
    channel_transit_h:float=4.
    mutant_residual_function:float=.1
    electrical_transfer_gain:float=1.
    soma_dwell_h:float=4.
    axon_dwell_h:float=4.
    lysosome_transit_h:float=1.
    pause_h:float=.5
    resume_h:float=.5
    cargo_rna_half_h:float=4.
    cargo_load_h:float=4.
    cargo_transit_h:float=1.
    local_protein_half_h:float=4.
    stmn2_motor_gain:float=0.
    provenance:str='Human KCNQ2 2025 endpoints; all time scales, basal PSI and source-to-model transfer are hypotheses. BORC/ANXA11 gates independent; no inferred Fe/ATP/TDP loss.'

    def validate(self):
        v=asdict(self);v.pop('provenance')
        if not np.isfinite(list(v.values())).all() or not self.provenance.strip():raise ValueError('Finite parameters and provenance required')
        for n,x in v.items():
            if n.endswith('_h') and x<=0:raise ValueError('Time scales must be positive')
        for n in ['baseline_delta_fraction','assumed_reference_TDP_function','mutant_export_fraction','channel_axon_share']:
            if not 0<getattr(self,n)<1:raise ValueError(n+' must lie in (0,1)')
        for n in ['mutant_residual_function','electrical_transfer_gain','stmn2_motor_gain']:
            if not 0<=getattr(self,n)<=1:raise ValueError(n+' must lie in [0,1]')
        if self.splice_power<=0 or self.delta_fold_reference<1 or self.wt_retention_gain<0:raise ValueError('Invalid splice/retention parameters')

RANGES=dict(baseline_delta_fraction=[.001,.01,.02],assumed_reference_TDP_function=[.05,.15,.5],
    splice_power=[.5,1.,2.],rna_half_h=[.5,4.,24.],channel_half_h=[4.,24.,72.],
    er_export_h=[.5,2.,8.],golgi_delivery_h=[.25,1.,4.],mutant_export_fraction=[.03,.1,.3],
    wt_retention_gain=[0.,1.,3.],channel_transit_h=[.5,4.,24.],lysosome_transit_h=[.5,1.,4.],
    electrical_transfer_gain=[0.,.5,1.],stmn2_motor_gain=[0.,.5,1.],
    cargo_load_h=[.5,4.,24.],cargo_transit_h=[.5,1.,4.],pause_h=[.1,.5,4.],resume_h=[.1,.5,4.],
    membrane_recycle_h=[2.,12.,48.],channel_axon_share=[.05,.3,.7],mutant_residual_function=[0.,.1,.3],
    soma_dwell_h=[.5,4.,24.],axon_dwell_h=[.5,4.,24.],cargo_rna_half_h=[.5,4.,24.],local_protein_half_h=[.5,4.,24.])

NAMES=['KCNQ_WT_RNA','KCNQ_delta_RNA']
NAMES += [typ+'_'+loc for typ in ['WT','delta'] for loc in ['ER','Golgi','soma_surface','anterograde','axon_surface','retrograde']]
NAMES += ['lys_soma','lys_anterograde','lys_distal','lys_retrograde','lys_pause_antero','lys_pause_retro']
NAMES += ['cargo_RNA_soma','cargo_RNA_anterograde','cargo_RNA_distal','local_protein',
          'KCNQ_RNA_produced','KCNQ_RNA_removed','WT_produced','WT_removed','delta_produced','delta_removed',
          'cargo_RNA_produced','cargo_RNA_removed','local_protein_produced','local_protein_removed']
DRIVERS=['start_s','end_s','nuclear_TDP_function_relative','splice_correction_fraction','channel_delivery_gate',
         'lysosome_anterograde_gate','lysosome_retrograde_gate','RNA_attachment_gate','translation_gate','STMN2_relative']

def _rhs(y,drive,p,cargo_source,protein_source):
    """All rates per hour; driver dimensions are separate, not interchangeable."""
    tdp,aso,ch,motor,retro,tether,tr,st=drive
    motor*=max(0.,1-p.stmn2_motor_gain*max(0.,1-st))
    retro*=max(0.,1-p.stmn2_motor_gain*max(0.,1-st))
    deficit=min(1.,max(0.,(1-tdp)/(1-p.assumed_reference_TDP_function)))
    fold=1+(p.delta_fold_reference-1)*deficit**p.splice_power*(1-aso)
    f=min(.95,p.baseline_delta_fraction*fold)
    d=np.zeros(34);kr=np.log(2)/p.rna_half_h;kp=np.log(2)/p.channel_half_h
    d[:2]=kr*(np.array([1-f,f])-y[:2]);d[24]=kr;d[25]=kr*sum(y[:2])
    actual_f=y[1]/sum(y[:2]) if sum(y[:2])>0 else p.baseline_delta_fraction
    for n,off in enumerate([2,8]):
        q=y[off:off+6];dq=d[off:off+6]
        export=1/p.er_export_h*(1/(1+p.wt_retention_gain*actual_f) if n==0 else p.mutant_export_fraction)
        synth=kp*y[n];loss=kp*q;dq[:]= -loss;dq[0]+=synth
        def flux(a,b,rate):
            z=rate*q[a];dq[a]-=z;dq[b]+=z
        flux(0,1,export)
        flux(1,2,(1-p.channel_axon_share)*ch/p.golgi_delivery_h)
        flux(1,3,p.channel_axon_share*ch/p.golgi_delivery_h)
        flux(3,4,ch/p.channel_transit_h)
        flux(2,1,1/p.membrane_recycle_h)
        flux(4,5,1/p.membrane_recycle_h)
        flux(5,1,ch/p.channel_transit_h)
        d[26+2*n]=synth;d[27+2*n]=sum(loss)
    q=y[14:20];dq=d[14:20]
    def lf(a,b,k):
        z=k*q[a];dq[a]-=z;dq[b]+=z
    lf(0,1,motor/p.soma_dwell_h);lf(1,2,motor/p.lysosome_transit_h)
    lf(2,3,retro/p.axon_dwell_h);lf(3,0,retro/p.lysosome_transit_h)
    lf(1,4,1/p.pause_h);lf(4,1,motor/p.resume_h)
    lf(3,5,1/p.pause_h);lf(5,3,retro/p.resume_h)
    # RNA attaches to lysosomal carriers; changing attachment does not alter
    # lysosome movement. This represents a cargo class, not all cellular RNAs.
    r=y[20:23];kd=np.log(2)/p.cargo_rna_half_h
    load=tether*motor/p.cargo_load_h*r[0];arrive=motor/p.cargo_transit_h*r[1]
    d[20:23]=[cargo_source-load-kd*r[0],load-arrive-kd*r[1],arrive-kd*r[2]]
    d[30]=cargo_source;d[31]=kd*sum(r)
    prod=protein_source*tr*r[2];loss=np.log(2)/p.local_protein_half_h*y[23]
    d[23]=prod-loss;d[32]=prod;d[33]=loss
    return d

def baseline(p):
    p.validate();drive=np.array([1.,0.,1.,1.,1.,1.,1.,1.])
    # Linear steady state in amounts at a fixed splice fraction; conserved
    # lysosome number supplies one normalization equation.
    b=_rhs(np.zeros(34),drive,p,1.,1.)[:24]
    A=np.column_stack([_rhs(np.eye(34)[i],drive,p,1.,1.)[:24]-b for i in range(24)])
    A[19,:]=0;A[19,14:20]=1;b[19]=-1
    y=np.zeros(34);y[:24]=np.linalg.solve(A,-b)
    # Normalize total cargo RNA and local protein separately; no common mass
    # scale is assumed between unrelated lineages.
    cargo_source=1/sum(y[20:23]);y[20:23]*=cargo_source
    protein_source=(np.log(2)/p.local_protein_half_h)/y[22];y[23]=1.
    assert np.min(y)>-1e-12 and np.max(abs(_rhs(y,drive,p,cargo_source,protein_source)[:24]))<1e-9
    return y,cargo_source,protein_source

def simulate_transport(segments,*,parameters=None,checkpoint=None,output_step_s=300.,rtol=1e-9,atol=1e-11):
    p=TransportParameters() if parameters is None else parameters;p.validate()
    s=np.asarray(segments,float)
    if s.ndim!=2 or s.shape[1]!=10 or not len(s) or not np.isfinite(s).all() or np.any(s<0) or np.any(s[:,1]<=s[:,0]) or np.any(s[1:,0]!=s[:-1,1]):raise ValueError('Contiguous finite ten-column nonnegative schedule required')
    if np.any(s[:,2:]>1) or not np.isfinite([output_step_s,rtol,atol]).all() or min(output_step_s,rtol,atol)<=0:raise ValueError('Drivers are fractions; positive solver settings required')
    ref,rs,ps=baseline(p);y=ref.copy()
    if checkpoint is not None:
        if checkpoint['parameters']!=asdict(p) or checkpoint['time_s']!=s[0,0]:raise ValueError('Checkpoint parameters/time mismatch')
        y=np.asarray(checkpoint['state'],float)
        if y.shape!=(34,) or not np.isfinite(y).all() or min(y)<-1e-10:raise ValueError('Invalid checkpoint state')
    initial=y.copy();tt=[s[0,0]];zz=[y.copy()]
    for row in s:
        a,b=row[:2];grid=np.unique(np.r_[np.arange(a+output_step_s,b,output_step_s),b])
        sol=solve_ivp(lambda t,x:_rhs(x,row[2:],p,rs,ps)/3600,(a,b),y,t_eval=grid,method='BDF',rtol=rtol,atol=atol)
        if not sol.success:raise RuntimeError(sol.message)
        y=sol.y[:,-1];tt.extend(sol.t);zz.extend(sol.y.T)
    t=np.asarray(tt);z=np.asarray(zz)
    errors={}
    for name,sl,produced,removed in [('KCNQ_RNA',slice(0,2),24,25),('WT',slice(2,8),26,27),('delta',slice(8,14),28,29),('cargo_RNA',slice(20,23),30,31),('local_protein',slice(23,24),32,33)]:
        balance=z[:,sl].sum(axis=1)-z[:,produced]+z[:,removed]
        errors[name]=float(np.max(abs(balance-balance[0])))
    errors['lysosome_number']=float(np.max(abs(z[:,14:20].sum(axis=1)-initial[14:20].sum())))
    if min(z.flat)<-1e-8 or max(errors.values())>1e-6 or not np.isfinite(z).all():raise ArithmeticError('Transport positivity/balance failed')
    f=z[:,1]/(z[:,0]+z[:,1]);soma=(z[:,4]+p.mutant_residual_function*z[:,10])/(ref[4]+p.mutant_residual_function*ref[10])
    axon=(z[:,6]+p.mutant_residual_function*z[:,12])/(ref[6]+p.mutant_residual_function*ref[12])
    return dict(time_s=t,state=z,columns=NAMES,baseline_state=ref,parameters=asdict(p),segments=s,
        delta_fraction=f,soma_functional_channel_ratio=soma,axon_functional_channel_ratio=axon,
        effective_M_ratio=1+p.electrical_transfer_gain*(soma-1),balance_errors=errors,
        checkpoint=dict(time_s=float(t[-1]),state=y.tolist(),parameters=asdict(p)),
        limitation='Normalized hypothetical kinetics; nuclear TDP function is not TARDBP RNA. No continuous long-voltage feedback, no Fe duplication or inferred ATP deficit.')
