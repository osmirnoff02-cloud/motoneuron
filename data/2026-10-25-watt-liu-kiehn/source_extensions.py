"""[codex] 2026-10-25. Source-qualified experimental and observation modules.

Liu/Liman: mouse TRPM5 in heterologous cells, not a human MN kinetic fit.
Kiehn/Eken: adult human motor units; firing is not a direct plateau recording.
These functions never alter the default electrical or ATP backend.
"""
from dataclasses import dataclass, asdict
import numpy as np

LIU_ENDPOINTS={'fresh':(21.,2.4,1.05),'desensitized':(77.,2.4,.23)}
TRPM5_SCENARIO_RANGES=dict(desensitization_per_uM_s=[0.,.001,.003,.005],
    pip2_recovery_tau_s=[1.,3.,10.],pip2_target_fraction=[0.,.1,.25,.5])

def _finite_nonnegative(x,name):
    a=np.asarray(x,float)
    if not np.isfinite(a).all() or np.any(a<0):raise ValueError(name+' must be finite and nonnegative')
    return a

def liu_trpm5_response(calcium_uM,*,fresh_fraction=1.):
    """Fig1d endpoints and Fig3c illustrative mixture; current, not open probability.

    25% recovery in Fig3c is an author's illustration, not an identified channel
    count/fraction. Fresh Vmax=1.05 is kept on the original normalization scale.
    """
    c=_finite_nonnegative(calcium_uM,'Ca');a=np.asarray(fresh_fraction,float)
    if not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('fresh_fraction outside [0,1]')
    def hill(k,n,v):
        q=np.zeros_like(c);mask=c>0
        q[mask]=v/(1+np.exp(np.clip(n*(np.log(k)-np.log(c[mask])),-700,700)))
        return q
    fresh=hill(*LIU_ENDPOINTS['fresh']);des=hill(*LIU_ENDPOINTS['desensitized'])
    return dict(relative_current=a*fresh+(1-a)*des,fresh_current=fresh,
        desensitized_current=des,fresh_fraction=a.copy(),
        source='10.1073/pnas.2334159100 Fig1d/3c; mouse construct CHO-K1 excised patch -80mV room temperature',
        human_MN_calibrated=False,direct_calcium_current=False,
        recovery_fraction_identified=False,normalization='Published normalized current; fresh fitted Vmax 1.05; not probability')

@dataclass(frozen=True)
class TRPM5HistoryParameters:
    # Approximate scale motivated by Fig2c; values are declared hypotheses.
    desensitization_per_uM_s:float=.003
    pip2_recovery_tau_s:float=3.
    pip2_target_fraction:float=.25
    provenance:str='Scenario: approximate Fig2 scale, not digitized kinetic fit; Fig3c illustrative 25% recovery; tau unknown.'
    def validate(self):
        v=[self.desensitization_per_uM_s,self.pip2_recovery_tau_s,self.pip2_target_fraction]
        if not np.isfinite(v).all() or v[0]<0 or v[1]<=0 or not 0<=v[2]<=1 or not self.provenance.strip():raise ValueError('Invalid TRPM5 history parameters')

def _advance(a,dt,d,r,target):
    """Exact bounded solution a'=-d*a+r*max(target-a,0)."""
    if r==0:return a*np.exp(-d*dt)
    if a>target:
        if d==0 or target==0:return a*np.exp(-d*dt)
        cross=np.log(a/target)/d
        if dt<=cross:return a*np.exp(-d*dt)
        a=target;dt-=cross
    eq=r*target/(d+r)
    return eq+(a-eq)*np.exp(-(d+r)*dt)

def run_trpm5_source_protocol(segments,*,parameters=None,checkpoint=None,output_step_s=.1):
    """Rows: absolute start/end(s), clamped Ca(uM), PIP2 exposure switch 0/1.

    One-way source experiment only: no Ca input, no ATP use, no membrane feedback.
    PIP2 switch is not a measured cellular lipid concentration. Extrapolation
    below the Fig2 measured 40–500uM range is flagged, not human calibration.
    """
    p=parameters or TRPM5HistoryParameters();p.validate();sg=np.asarray(segments,float)
    if sg.ndim!=2 or sg.shape[1]!=4 or not len(sg) or not np.isfinite(sg).all() or np.any(sg<0) or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[1:,0]!=sg[:-1,1]) or np.any(~np.isin(sg[:,3],[0.,1.])):raise ValueError('Contiguous source-protocol segments required')
    if not np.isfinite(output_step_s) or output_step_s<=0:raise ValueError('Positive output step required')
    if checkpoint is None:
        a=1.;origin=float(sg[0,0])
    else:
        if checkpoint['parameters']!=asdict(p) or checkpoint['time_s']!=sg[0,0]:raise ValueError('Checkpoint/time/parameters mismatch')
        a=float(checkpoint['fresh_fraction']);origin=float(checkpoint['origin_s'])
        if not np.isfinite(a) or not 0<=a<=1:raise ValueError('Invalid checkpoint fraction')
    ts=[];avs=[];cs=[];switches=[]
    for start,end,c,pip in sg:
        d=p.desensitization_per_uM_s*c;r=pip/p.pip2_recovery_tau_s
        first=origin+np.ceil((start-origin)/output_step_s)*output_step_s
        grid=np.unique(np.r_[start,np.arange(first,end,output_step_s),end])
        for t in grid:
            if ts and t<=ts[-1]:
                if t==ts[-1]:
                    avs[-1]=a;cs[-1]=c;switches[-1]=pip
                continue
            ts.append(float(t));avs.append(_advance(a,float(t-start),d,r,p.pip2_target_fraction));cs.append(c);switches.append(pip)
        a=_advance(a,float(end-start),d,r,p.pip2_target_fraction)
    avs=np.array(avs);cs=np.array(cs)
    if np.any(avs<0) or np.any(avs>1):raise ArithmeticError('Availability escaped bounds')
    response=liu_trpm5_response(cs,fresh_fraction=avs)
    return dict(time_s=np.array(ts),fresh_fraction=avs,calcium_uM=cs,PIP2_exposure=np.array(switches),
        relative_current=response['relative_current'],parameters=asdict(p),
        checkpoint=dict(time_s=float(sg[-1,1]),fresh_fraction=float(a),origin_s=origin,parameters=asdict(p)),
        kinetic_extrapolation=bool(np.any((sg[:,2]>0)&((sg[:,2]<40)|(sg[:,2]>500)))),
        human_MN_calibrated=False,feeds_back_to_voltage=False,direct_Ca_input_uM=0.,
        limitation='Source-experiment phenomenology; PIP2 is a prescribed switch, not cellular lipid turnover. Recovery rates and state interpretation not identified.')

def kiehn_firing_evidence(spike_times_s,*,pre_window_s,post_window_s,
        minimum_spikes=5,control_rate_change_Hz=None,control_tolerance_Hz=None,
        control_sensitivity_verified=False,shared_drive_verified=False,
        matched_adult_human_protocol=False):
    """Separate silence-to-firing and rate shift from proof of a plateau.

    Windows/minimum count and control tolerance are caller analysis choices,
    not universal Kiehn1997 cutoffs. The paper excludes first post-stimulus
    second for the Fig2 stable-rate comparison; physiological proof is absent.
    """
    t=_finite_nonnegative(spike_times_s,'Spikes');pre=np.asarray(pre_window_s,float);post=np.asarray(post_window_s,float)
    if t.ndim!=1 or np.any(np.diff(t)<=0) or pre.shape!=(2,) or post.shape!=(2,) or not np.isfinite(np.r_[pre,post]).all() or pre[0]<0 or pre[1]<=pre[0] or post[0]<pre[1] or post[1]<=post[0] or not isinstance(minimum_spikes,int) or minimum_spikes<2:raise ValueError('Sorted spike times and disjoint positive windows required')
    def stats(w):
        x=t[(t>=w[0])&(t<w[1])];isi=np.diff(x)
        return dict(count=len(x),count_rate_Hz=len(x)/(w[1]-w[0]),
            mean_instantaneous_Hz=float(np.mean(1/isi)) if len(isi) else None,
            isi_CV=float(np.std(isi)/np.mean(isi)) if len(isi) else None,
            enough_for_analysis=len(x)>=minimum_spikes)
    a=stats(pre);b=stats(post);stable_control=None
    if control_rate_change_Hz is not None:
        if control_tolerance_Hz is None or not np.isfinite([control_rate_change_Hz,control_tolerance_Hz]).all() or control_tolerance_Hz<0:raise ValueError('Explicit finite control tolerance required')
        stable_control=abs(control_rate_change_Hz)<=control_tolerance_Hz
    behavior='insufficient_post_window_spikes'
    if b['enough_for_analysis']:behavior='silence_to_firing' if a['count']==0 else 'firing_before_and_after'
    qualified=bool(stable_control and control_sensitivity_verified and shared_drive_verified and matched_adult_human_protocol)
    return dict(pre=a,post=b,phenotype=behavior,paired_protocol_qualified=qualified,
        plateau_proven=False,two_stable_states_proven=False,
        mean_rate_difference_Hz=(b['mean_instantaneous_Hz']-a['mean_instantaneous_Hz']) if a['enough_for_analysis'] and b['enough_for_analysis'] else None,
        limitation='Prolonged firing is compatible with intrinsic support, not direct voltage/Ca plateau evidence. Adult in-vivo motor units are not iPSC neurons.',
        source='10.1152/jn.1997.78.6.3061 Methods/Results/Discussion, pp3062-3067')

def watt_complex_reference():
    return dict(source='10.1073/pnas.1011099107 main + SI FigS1-S6',
        system='Purified bovine heart F1-c-ring structural complex; not human MN flux measurement',
        partial_complex=True,c_ring_copies=8,ATP_per_turn=3,
        synthase_H_per_ATP=8/3,cytoplasmic_H_equivalent_per_ATP=11/3,
        c_subunit_human_gene_candidates=['ATP5MC1','ATP5MC2','ATP5MC3'],
        c_isoform_composition_in_MN=None,complex_abundance=None,ATP_production_rate=None,
        figures=dict(S1='purification/subunit content',S2='c8 versus c9 refinement',S3_S4='bovine/yeast comparison',S5_S6='sequence comparison; species ring counts differ'),
        rate_identified=False,human_MN_calibrated=False)
