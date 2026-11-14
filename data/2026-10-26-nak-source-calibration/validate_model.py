"""[codex] Source-aware regression, conservation and conditional scenarios."""
from pathlib import Path
import sys,json,time,hashlib
from dataclasses import replace,asdict
import numpy as np
from energetic_neuron import run_recovery_neuron
from extension_parameters import RecoveryParameters,BudgetParameters
from connected_parameters import inputs,schedule,load
from mito_ion_model import run_mito_ion_neuron
from nmj import simulate_nmj,NMJParameters
B=Path(__file__).resolve().parent
def save(name,obj):
    (B/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf8')
def run(sg,profile='Burley',m_scale=1.,**kwargs):
    kw=inputs(profile);kw['profile_p']=kw['profile_p'].copy();kw['profile_p'][12]*=m_scale
    return run_recovery_neuron(sg,**kw,**kwargs)
def features(r,on,off,profile):
    tt=r['time_s'];vm=r['soma_voltage_mV'];grid=np.arange(tt[0],tt[-1],.0001);v=np.interp(grid,tt,vm)
    det=load(B.parent/'2026-10-16-selvaraj-waveform/waveform_fit.py','_source_calibration_detector26')
    if profile=='Burley':n=sum(on<=x<off for x in r['voltage_crossing_times_s']['soma'])
    # Detector indexes its sampled vector from zero, even for a checkpoint
    # trace whose absolute timestamps start after hours of simulation.
    else:n=int(det.features(v,.1,(on-tt[0])*1000,(off-on)*1000,inputs(profile)['profile_c'][2])[0])
    rest=np.mean(v[(grid>=max(tt[0],on-.1))&(grid<on)]);post=(grid>=off)&(grid<=tt[-1]);early=post&(grid<off+.1)
    return dict(spikes=int(n),rest_mV=float(rest),peak_mV=float(np.max(v[(grid>=on)&(grid<off)])),post_min_mV=float(v[post].min()-rest),post_100ms_min_mV=float(v[early].min()-rest),post_area_mVs=float(np.trapezoid(v[post]-rest,grid[post])),tail_delta_mV=float(np.mean(v[-100:])-rest))
def numerical():
    sg=schedule([(0,.2,0,1,0),(.2,.7,.001,1,0),(.7,2,0,1,0)]);es=np.c_[sg[:,:2],[0,25,0],[0,0,0]]
    kw=dict(electrical_schedule=es,output_step_s=.001,max_step_s=.005)
    r=run(sg,**kw);strict=run(sg,**dict(kw,rtol=1e-10,atol=1e-12,max_step_s=.0025))
    a=run(sg[:2],**dict(kw,electrical_schedule=es[:2]));b=run(sg[2:],checkpoint=a['checkpoint'],**dict(kw,electrical_schedule=es[2:]))
    neutral=run(sg,recovery_parameters=RecoveryParameters(enabled=False),budget_parameters=BudgetParameters(enabled=False),**kw)
    old=run_mito_ion_neuron(sg,**inputs(),**kw)
    rest=run(schedule([(0,30,0,1,0)]),output_step_s=.1,max_step_s=.5)
    err=lambda a,b:float(np.max(abs(a-b)/(1+abs(b))))
    out=dict(strict=err(strict['state'][-1],r['state'][-1]),restart=err(b['state'][-1],r['state'][-1]),disabled=err(neutral['state'][-1,:247],old['state'][-1]),rest_drift=float(np.max(abs(rest['state'][:,[199,219,120,136,247,248,257,258]]-rest['baseline_state'][[199,219,120,136,247,248,257,258]]))),checks=r['checks'])
    save('numerical.json',out);print('numerical',out,flush=True)
    assert max(out[k] for k in ('strict','restart','disabled'))<3e-4 and out['rest_drift']<1e-4
def electrical():
    old=json.loads((B.parent/'2026-10-26-nak-energy-neuron/electrical.json').read_text());rows=[]
    for q in old['rows']:
        profile=q['profile'];cur=q['current_pA'];dur=.5 if profile=='Burley' else 1.
        sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)]);es=np.c_[sg[:,:2],[0,cur,0],[0,0,0]]
        r=run(sg,profile,electrical_schedule=es,output_step_s=.0001,max_step_s=.005)
        row=dict(profile=profile,current_pA=cur,target=q['native_counts']['soma'],previous=q['counts']['soma'],features=features(r,.2,.2+dur,profile),checks=r['checks'])
        rows.append(row);save('electrical.json',rows);print('electrical',profile,cur,row['features']['spikes'],row['target'],flush=True)
def scenarios():
    rows=[];traces={}
    # Short repeated pulse trains. Same actual electrical driver, no forced AP count.
    pulses=[(1+b*10+k*.05,1.005+b*10+k*.05,100.) for b in range(2) for k in range(10)]
    edges=sorted(set([0.,60.]+[x for a,b,_ in pulses for x in (a,b)]));sg=schedule([(a,b,0,1,0) for a,b in zip(edges[:-1],edges[1:])]);es=np.c_[sg[:,:2],[sum(i for x,y,i in pulses if x<=a<y) for a in sg[:,0]],np.zeros(len(sg))]
    cases=[('reference',{}),('alpha3_25',dict(alpha3_support=.25)),('respiration_50',dict(respiratory_support=.5)),('small_reserve',dict(budget_parameters=replace(BudgetParameters(),proton_reserve_uM=100))),('large_reserve',dict(budget_parameters=replace(BudgetParameters(),proton_reserve_uM=2000))),('TDP_scale_01',dict(budget_parameters=replace(BudgetParameters(),tdp_uM_per_lineage_unit=.1))),('TDP_scale_10',dict(budget_parameters=replace(BudgetParameters(),tdp_uM_per_lineage_unit=10))),('alpha3_high',dict(recovery_parameters=replace(RecoveryParameters(),alpha3_max_fraction=.9)))]
    for name,kw in cases:
        r=run(sg,electrical_schedule=es,output_step_s=.002,max_step_s=.2,**kw);z=r['state'];t=r['time_s']
        m=dict(name=name,Na_peak_uM=r['sodium_in_uM'].max(0),Na_end_uM=r['sodium_in_uM'][-1],ATP_min_uM=[z[:,120].min(),z[:,136].min()],Ca_peak_uM=[r['calcium_uM'].max(),r['process_calcium_uM'].max()],reserve_min=r['proton_reserve_relative'].min(0),O2_min_uM=r['oxygen_uM'].min(0),protein_ATP_uM=r['protein_ATP_uM'][-1],counts={s:len(x) for s,x in r['spike_times_s'].items()},post=features(r,1.,11.455,'Burley'),checks=r['checks'])
        rows.append(m);save('scenarios.json',rows);print('scenario',name,m['counts'],m['ATP_min_uM'],flush=True)
        ix=np.unique(np.r_[np.arange(0,len(t),5),len(t)-1]);traces[name+'_t']=t[ix];traces[name+'_z']=z[ix]
        if name=='reference':
            nmj=[]
            for label,params,atp in [('reference',NMJParameters(),r['process_ATP_uM']),('low_ATP',NMJParameters(),r['process_ATP_uM']*.1),('slow_replenishment',replace(NMJParameters(),replenishment_s=10),r['process_ATP_uM'])]:
                n=simulate_nmj(r['spike_times_s']['process'],t,atp,parameters=params,step_s=.002)
                nmj.append(dict(arm=label,events=n['events'],endplate_crossings=n['endplate_threshold_crossings_s'],balance=n['vesicle_balance'],ATP_recycling_molecules=n['state'][-1,6],limitation=n['limitations']))
            save('nmj_results.json',nmj)
    np.savez_compressed(B/'scenario_traces.npz',**traces)
if __name__=='__main__':
    for task in sys.argv[1:] or ['numerical','electrical','scenarios']:globals()[task]()
