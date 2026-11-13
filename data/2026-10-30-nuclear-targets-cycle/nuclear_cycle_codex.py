"""[codex] 2026-10-30. Optional API0.26; source architecture, unknown human rates.

One nuclear TDP signal drives TARDBP/STMN2, PKN cryptic RNA and KCNQ2 splice
choice. Only TARDBP/STMN2 use the selected ribosome pool. No rRNA biogenesis,
universal stall-rescue claim, or absolute human channel-density calibration.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'2026-10-24-kcnq2-transport'))
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'2026-10-30-nucleus-ribosome'))
from transport_codex import TransportParameters
import numpy as np
from numba import njit
from nucleus_ribosome_codex import nucleus_fluxes

CYCLE_NAMES=('free_40S_selected','free_60S_selected','initiating_TARDBP_40S','initiating_STMN2_40S',
 'stalled_TARDBP_80S','stalled_STMN2_80S','post_termination_TARDBP_80S','post_termination_STMN2_80S',
 'aborted_TARDBP_complexes','aborted_STMN2_complexes','ribosome_recycling_ATP_uM','ribosome_replaced_background_ATP_uM')
K_NAMES=('KCNQ_WT_RNA','KCNQ_delta_RNA')+tuple(typ+'_'+loc for typ in ('KCNQ_WT','KCNQ_delta') for loc in
 ('ER','Golgi','soma_surface','anterograde','process_surface','retrograde'))+('KCNQ_RNA_produced','KCNQ_RNA_removed',
 'KCNQ_WT_produced','KCNQ_WT_removed','KCNQ_delta_produced','KCNQ_delta_removed')+('KCNQ_soma_gate','KCNQ_process_gate')
EXTRA_NAMES=CYCLE_NAMES+K_NAMES

@dataclass(frozen=True)
class RibosomeCycleParameters:
    joining_s:float=5.
    recycling_s:float=10.
    rescue_s:float=300.
    stress_stall_per_s:float=.001
    selected_ribosome_capacity_uM:float=.01
    recycling_ATP_per_complex:float=2.
    provenance:str='Pisarev2010/Pisareva2011 reconstituted mammalian recycling architecture. All stage rates, effective ATP stoichiometry and selected-pool concentration are assumptions, not human MN fits. Stalled pool means rescue-competent substrates only.'
    def vector(self):
        a=np.array([self.joining_s,self.recycling_s,self.rescue_s,self.stress_stall_per_s,self.selected_ribosome_capacity_uM,self.recycling_ATP_per_complex])
        if not np.isfinite(a).all() or np.any(a<0) or min(a[:3])<=0 or not self.provenance.strip():raise ValueError('Invalid ribosome cycle parameters')
        return a

@dataclass(frozen=True)
class KCNQCouplingParameters:
    g_reference_nS:float=.5
    vhalf_mV:float=-27.3966
    slope_mV:float=11.7788
    gate_tau_s:float=.08
    provenance:str='Joseph2025 CHO gate reference; gM and time constant are assumptions. Differential current relative to implicit healthy fitted potassium current, not independent absolute M-current validation.'
    def vector(self):
        a=np.array([self.g_reference_nS,self.vhalf_mV,self.slope_mV,self.gate_tau_s])
        if not np.isfinite(a).all() or a[0]<0 or min(a[2:])<=0 or not self.provenance.strip():raise ValueError('Invalid KCNQ coupling')
        return a

CYCLE_RANGES=dict(joining_s=[1.,5.,20.],recycling_s=[1.,10.,60.],rescue_s=[30.,300.,3000.],
 stress_stall_per_s=[0.,.001,.01],selected_ribosome_capacity_uM=[.001,.01,.1],recycling_ATP_per_complex=[1.,2.,4.],
 joining_support=[0.,.25,1.],recycling_support=[0.,.25,1.],rescue_support=[0.,.25,1.],stall_support=[0.,.5,1.])
KCNQ_COUPLING_RANGES=dict(g_reference_nS=[0.,.5,2.,8.],gate_tau_s=[.02,.08,.3])

def cycle_schedule(segments,supplied=None):
    a=np.c_[segments[:,:2],np.ones((len(segments),3)),np.zeros(len(segments))] if supplied is None else np.asarray(supplied,float)
    if a.shape!=(len(segments),6) or not np.array_equal(a[:,:2],segments[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>1):raise ValueError('Cycle schedule: matching start/end, joining/recycling/rescue/stall supports [0,1]')
    return a

def initial_cycle(v,rv):
    completion=v[5]/np.array([414.,v[10]])*v[6:8]
    init=completion*rv[0];post=completion*rv[1]
    free60=1-v[6:8].sum()-post.sum();free40=free60-init.sum()
    if min(free40,free60)<=0:raise ValueError('Cycle initialization exceeds selected subunit capacity; reduce occupied slots or residence times')
    return np.r_[free40,free60,init,[0.,0.],post,np.zeros(4)]

def kcnq_vector(parameters,coupling):
    parameters.validate();d=asdict(parameters)
    names=['baseline_delta_fraction','delta_fold_reference','assumed_reference_TDP_function','splice_power','rna_half_h','channel_half_h',
       'er_export_h','golgi_delivery_h','mutant_export_fraction','wt_retention_gain','membrane_recycle_h','channel_axon_share','channel_transit_h','mutant_residual_function','electrical_transfer_gain']
    a=np.array([d[n]*(3600. if n.endswith('_h') else 1.) for n in names])
    return np.r_[a,coupling.vector()]

def initial_kcnq(parameters,coupling,vm):
    from transport_codex import baseline
    b,_,_=baseline(parameters)
    gates=1/(1+np.exp(-(np.asarray(vm)-coupling.vhalf_mV)/coupling.slope_mV))
    return np.r_[b[:14],np.zeros(6),gates]

@njit
def cycle_fluxes(x,elong,mT,mS,ATP,ATP0,v,rv,controls,nuccontrols,initial):
    """Conserved selected 40S/60S equivalents; stalled substrates are eligible for rescue.
    Termination/peptide release is lumped into elongation exit; post-state awaits
    recycling. No added rRNA, automatic NMD, collisions or universal PELO action.
    """
    x=np.maximum(x,0.);elong=np.maximum(elong,0.)
    energy=max(ATP,0.)/(v[8]+max(ATP,0.))*(v[8]+ATP0)/ATP0
    k=v[5]/np.array([414.,v[10]]);c0=k*v[6:8]
    load=c0*np.array([max(mT,0.),max(mS,0.)])*x[0]/initial[0]*energy*nuccontrols[1]
    join=x[2:4]/rv[0]*x[1]/initial[1]*energy*controls[0]
    complete=k*elong*energy*nuccontrols[2]
    stall=rv[3]*controls[3]*elong
    rescue=x[4:6]/rv[2]*energy*controls[2]
    recycle=x[6:8]/rv[1]*energy*controls[1]
    dx=np.zeros(12)
    dx[0]=sum(rescue+recycle-load);dx[1]=sum(rescue+recycle-join)
    dx[2:4]=load-join;dx[4:6]=stall-rescue;dx[6:8]=complete-recycle
    dx[8:10]=rescue
    dx[10]=sum(rescue+recycle)*rv[4]*rv[5];dx[11]=sum(c0)*rv[4]*rv[5]
    return dx,join-complete-stall,complete/c0

@njit
def replace_cycle_targets(z,old,p,hv,nv,v,rv,f0,ATP0,nuccontrol,cyclecontrol,cyclebase,row):
    out=old.copy()
    dx,dmT,dmS,_,_,nuc=nucleus_fluxes(z[320:330],z[6]+z[7],z[50],z[10],z[120],p,hv,v,f0,ATP0,nuccontrol)
    cy,de,fac=cycle_fluxes(z[330:342],z[326:328],z[50],z[10],z[120],ATP0,v,rv,cyclecontrol,nuccontrol,cyclebase)
    syn=p[20]*fac[0];delta=syn-old[18]
    out[6]+=delta;out[18]+=delta;out[50]=dmT;out[10]=dmS
    out[11]=np.log(2)/v[11]*(fac[1]-z[11]);dx[6:8]=de
    energy=max(z[120],0.)/(v[8]+max(z[120],0.))*(v[8]+ATP0)/ATP0
    f=min(1.,max(0.,z[320]));T=max(z[6]+z[7],1e-20)
    dx[0]=f0/(1-f0)*(v[0]+p[20])*nuccontrol[0]*energy*(1-f)-v[0]*f-f*syn/T
    dx[8]=np.log(2)/v[11]*v[9]*v[10]*4*fac[1]
    # PKN RNA proxy gets exactly the same nuclear signal, not total TDP.
    target=nv[1]*max(1-nuc/nv[0],0.)
    out[64]=(target-z[64])/nv[2]
    # Optional inherited TDP-loss->ER link also receives the same signal.
    # PKN207 and caspase-generated AF3 remain separate, with old kinetics.
    oldloss=max(1-(z[6]+z[7]),0.);newloss=max(1-nuc,0.)
    before=min(1.,row[7]/(1+nv[11]*oldloss+nv[23]*max(z[65],0.)))*z[69]/nv[12]
    after=min(1.,row[7]/(1+nv[11]*newloss+nv[23]*max(z[65],0.)))*z[69]/nv[12]
    out[69]-=after-before;out[70]+=after-before
    return out,dx,cy

@njit
def kcnq_rhs(y,nuclear,p):
    """First 14 cargo states + 6 counters from the established transport model.
    No duplicated lysosomes or RNA-transport classes. Rates per second.
    """
    d=np.zeros(20);y=np.maximum(y,0.)
    deficit=min(1.,max(0.,(1-nuclear)/(1-p[2])))
    f=min(.95,p[0]*(1+(p[1]-1)*deficit**p[3]))
    kr=np.log(2)/p[4];kp=np.log(2)/p[5]
    d[0]=kr*(1-f-y[0]);d[1]=kr*(f-y[1]);d[14]=kr;d[15]=kr*(y[0]+y[1])
    frac=y[1]/max(y[0]+y[1],1e-30)
    for n in range(2):
        off=2+6*n;q=y[off:off+6]
        synth=kp*y[n];loss=kp*q
        export=q[0]/p[6]*(1/(1+p[9]*frac) if n==0 else p[8])
        soma=q[1]*(1-p[11])/p[7];forward=q[1]*p[11]/p[7]
        arrive=q[3]/p[12];rs=q[2]/p[10];rp=q[4]/p[10];back=q[5]/p[12]
        d[off:off+6]=np.array([synth-export,export+rs+back-soma-forward,soma-rs,forward-arrive,arrive-rp,rp-back])-loss
        d[16+2*n]=synth;d[17+2*n]=sum(loss)
    return d

@njit
def kcnq_ratios(y,baseline,p):
    a=np.zeros(2)
    for j,i in enumerate((4,6)):
        ratio=(max(y[i],0.)+p[13]*max(y[i+6],0.))/(baseline[i]+p[13]*baseline[i+6])
        a[j]=max(0.,1+p[14]*(ratio-1))
    return a

@njit
def kcnq_current_extension(z,old,ctx,iv,p,baseline):
    out=old.copy();ratios=kcnq_ratios(z[342:362],baseline,p)
    for j in range(2):
        b=199+20*j;vi=185+7*j;vm=z[vi];gv=ctx[38];cap=ctx[37][0]*gv[j]/(gv[0]+gv[1])
        ek=iv[1]*np.log(max(z[b+3],1e-12)/max(z[b+1],1e-12))
        current=(ratios[j]-1)*p[15]*gv[j]/(gv[0]+gv[1])*z[362+j]*(vm-ek)
        F=10364.26965661773/(ctx[27] if j==0 else ctx[28])
        out[vi]-=1000*current/cap;out[b+1]-=current*F;out[b+3]+=current*F/iv[10];out[b+6]+=current*F
    return out

@njit
def append_cycle(z,old,cy,rv,kp,kbase,f0,other0,underlying):
    out=np.zeros(364);out[:330]=old;out[330:342]=cy
    replaced=cy[11]*max(underlying,0.)/other0 if other0>0 else 0.
    out[341]=replaced;delta=cy[10]-replaced
    out[120]-=delta;out[121]+=delta;out[125]+=delta
    nuclear=max(z[6]+z[7],0.)*min(1.,max(z[320],0.))/f0
    out[342:362]=kcnq_rhs(z[342:362],nuclear,kp)
    for j in range(2):
        vm=z[185+7*j];m=1/(1+np.exp(-max(-700.,min(700.,(vm-kp[16])/kp[17]))))
        out[362+j]=(m-z[362+j])/kp[18]
    return out

def cycle_checks(z,base):
    b=np.asarray(base);r=z[:,330:342]
    c40=r[:,0]+r[:,2:8].sum(1)+z[:,326:328].sum(1)
    c60=r[:,1]+r[:,4:8].sum(1)+z[:,326:328].sum(1)
    checks=dict(selected_40S_balance=float(abs(c40-1).max()),selected_60S_balance=float(abs(c60-1).max()),
      cycle_KCNQ_nonnegative=float(max(0.,-z[:,330:].min())),KCNQ_gate_bounds=float(max(0.,z[:,362:364].max()-1)),
      KCNQ_RNA_balance=float(abs(z[:,342:344].sum(1)-z[:,356]+z[:,357]-b[342:344].sum()).max()))
    for n,label in enumerate(('WT','delta')):
        off=344+6*n;prod=358+2*n
        checks['KCNQ_'+label+'_balance']=float(abs(z[:,off:off+6].sum(1)-z[:,prod]+z[:,prod+1]-b[off:off+6].sum()).max())
    return checks
