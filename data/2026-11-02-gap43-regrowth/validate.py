"""[codex]2026-11-02. Full-day resting controls and conditional recovery assay."""
from pathlib import Path
from dataclasses import replace,asdict
import sys,importlib.util,json,argparse,time
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
threads=threadpool_limits(1);B=Path(__file__).resolve().parent
sys.path.insert(0,str(B));sys.path.insert(0,str(B.parent/'2026-11-02-local-mrna'))
spec=importlib.util.spec_from_file_location('gap36_prior',B.parent/'2026-11-02-local-mrna/validate.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
api=m.api
from gap43_regrowth import *

def save(name,obj):
 (B/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n',encoding='utf8')

def unit():
 cs,base,*_=m.setup();v=GAP43RegrowthParameters().vector();z,ref=initialize(base,v)
 d=extension(z,np.zeros(617),v,np.array([0.,1.,1.]),ref)
 assert len(d)==620 and not d.any()
 z[617]=.25;d=extension(z,np.zeros(617),v,np.array([0.,1.,1.]),ref)
 assert abs(d[617]-.75/(24*3600))<1e-16 and not d[:617].any()
 zz=z.copy();zz[583]=0;d0=extension(zz,np.zeros(617),v,np.array([0.,1.,1.]),ref)
 assert np.isclose(d0[617],.5*d[617])
 null=v.copy();null[1]=0
 assert extension(zz,np.zeros(617),null,np.array([0.,1.,1.]),ref)[617]==d[617]
 zz[136]=0;assert extension(zz,np.zeros(617),v,np.array([0.,1.,1.]),ref)[619]==0
 for x in [0.,1.]:
  zz=z.copy();zz[617]=x;dd=extension(zz,np.zeros(617),v,np.array([1.,1.,1.]),ref)
  assert dd[617]>=0 if x==0 else dd[617]<=0
 for bad in [replace(GAP43RegrowthParameters(),gap43_dependence=2),replace(GAP43RegrowthParameters(),repair_time_h=0),replace(GAP43RegrowthParameters(),initial_deficit=-1)]:
  try:bad.vector();raise AssertionError('Bad parameter accepted')
  except ValueError:pass
 sol=solve_ivp(lambda t,y:extension(np.r_[base,y],np.zeros(617),v,np.array([0.,1.,1.]),ref)[617:],(0,86400),[.25,0,0],method='LSODA',rtol=1e-11,atol=1e-13)
 exact=1-.75*np.exp(-1);assert abs(sol.y[0,-1]-exact)<1e-9
 ch=checks(np.c_[np.tile(base,(len(sol.t),1)),sol.y.T],z);assert max(ch.values())<1e-9,ch
 save('unit.json',dict(passed=True,analytic_recovery_error=float(abs(sol.y[0,-1]-exact)),null_hypothesis_exact=True,zero_ATP_stops_recovery=True,readout_does_not_change_upstream=True,checks=ch))
 print('unit passed',flush=True)

def full():
 sg=m.schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)])
 es=np.c_[sg[:,:2],[0,25,0],[0,0,0]];gc=controls(sg);gc[1,2]=1
 opts=dict(**m.m.hist.options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
 run=api.run_gap43_regrowth_neuron;r=run(sg,gap43_regrowth_schedule=gc,**opts)
 assert r['state'].shape[1]==len(r['columns'])==620
 a=run(sg[:2],gap43_regrowth_schedule=gc[:2],**dict(opts,electrical_schedule=es[:2]));b=run(sg[2:],gap43_regrowth_schedule=gc[2:],checkpoint=a['checkpoint'],**dict(opts,electrical_schedule=es[2:]))
 strict=run(sg,gap43_regrowth_schedule=gc,**dict(opts,rtol=3e-10,atol=3e-12,max_step_s=.00125))
 # Absolute clock shift must preserve state and hybrid events. No pool reset
 # is involved in the separate postday test; this checks time-coordinate bias.
 shifted=sg.copy();shifted[:,:2]+=86400.;shifted_gc=gc.copy();shifted_gc[:,:2]+=86400.
 shifted_es=es.copy();shifted_es[:,:2]+=86400.
 shifted_result=run(shifted,gap43_regrowth_schedule=shifted_gc,**dict(opts,electrical_schedule=shifted_es))
 err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
 restart=err(b['state'][-1],r['state'][-1]);conv=err(strict['state'][-1],r['state'][-1]);assert max(restart,conv)<1e-5
 clock_error=err(shifted_result['state'][-1],r['state'][-1]);assert clock_error<1e-5
 for compartment in ('soma','process'):
  assert np.allclose(np.asarray(shifted_result['voltage_crossing_times_s'][compartment])-86400,r['voltage_crossing_times_s'][compartment],rtol=0,atol=1e-7)
 bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][617]=1.1
 try:run(sg[2:],checkpoint=bad,gap43_regrowth_schedule=gc[2:],**dict(opts,electrical_schedule=es[2:]));raise AssertionError('Bad recovery checkpoint accepted')
 except ValueError:pass
 save('integrated.json',dict(passed=True,states=620,restart_error=restart,strict_error=conv,clock_shift_error=clock_error,checks=r['checks'],config=r['config']))
 save('columns.json',list(r['columns']))
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical.json').read_text())['rows']
 metrics=m.module('gap36_features',B.parent/'2026-10-26-nak-source-calibration/validate_model.py');rows=[]
 for target in targets:
  p=target['profile'];cur=target['current_pA'];dur=.5 if p=='Burley' else 1.
  sg=m.schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  r=run(sg,**m.m.hist.options(p),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  f=metrics.features(r,.2,.2+dur,p);assert f['spikes']==target['target']
  delta=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.02
  rows.append(dict(profile=p,current_pA=cur,features=f,max_change=delta))
  save('electrical.json',dict(complete=len(rows)==9,rows=rows));print('electrical',p,cur,delta,flush=True)

def resting_day(profile='Burley',strict=False):
 opts=m.m.hist.options(profile);run=api.run_gap43_regrowth_neuron
 r=run(m.schedule([(0,86400,0,1,0)]),**opts,output_step_s=300,max_step_s=15 if strict else 30,rtol=3e-9 if strict else 1e-8,atol=3e-11 if strict else 1e-10)
 z=r['state'];b=r['baseline_state']
 # Pulse-chase labels must decay/redistribute even when total physical pools
 # are stationary. They are distinct observers, not a baseline instability.
 label_indices=[j for j,n in enumerate(r['columns']) if n.startswith('labelled_') or n.startswith('ferritin_label_')]
 nonzero=np.setdiff1d(np.flatnonzero(abs(b)>1e-12),label_indices)
 drift=float(np.max(abs(z[:,nonzero]-b[nonzero])/abs(b[nonzero])))
 # A conservation-based initial near-equilibrium, not a fitted healthy state.
 assert drift<.02 and np.max(abs(z[:,[185,192]]-b[[185,192]]))<.1
 assert not len(r['voltage_crossing_times_s']['soma']) and not len(r['voltage_crossing_times_s']['process'])
 assert np.max(abs(z[:,617]-1))<1e-9
 summary=dict(profile=profile,strict=strict,completed=True,duration_s=86400,
  baseline_nonzero_states_checked=len(nonzero),excluded_pulse_chase_labels=label_indices,maximum_relative_drift=drift,
  voltage_max_change_mV=float(np.max(abs(z[:,[185,192]]-b[[185,192]]))),
  ATP_max_relative_change=float(np.max(abs(z[:,[120,136,282]]/b[[120,136,282]]-1))),
  PCr_max_relative_change=float(np.max(abs(z[:,315:318]/b[315:318]-1))),
  Na_max_relative_change=float(np.max(abs(z[:,[199,219]]/b[[199,219]]-1))),
  Ca_max_absolute_change_uM=float(np.max(abs(z[:,[0,134]]-b[[0,134]]))),
  GAP43_max_relative_change=float(np.max(abs(z[:,583]/b[583]-1))),
  maximum_ledger_error=max(x for k,x in r['checks'].items() if k!='minimum_state'),checks=r['checks'],nfev=r['nfev'])
 suffix=profile+('_strict' if strict else '')
 save('rest_'+suffix+'.json',summary)
 np.savez_compressed(B/('rest_'+suffix+'.npz'),time_s=r['time_s'],state=z)
 if not strict:
  # Verify excitability on the same physical state after24h, without resetting.
  cur=25 if profile=='Burley' else 15;dur=.5 if profile=='Burley' else 1.;t=86400.
  sg=m.schedule([(t,t+.2,0,1,0),(t+.2,t+.2+dur,0,1,0),(t+.2+dur,t+1.2+dur,0,1,0)])
  rr=run(sg,checkpoint=r['checkpoint'],**opts,electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  metrics=m.module('day_'+profile+'_features',B.parent/'2026-10-26-nak-source-calibration/validate_model.py')
  f=metrics.features(rr,t+.2,t+.2+dur,profile)
  targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical.json').read_text())['rows'];target=next(a for a in targets if a['profile']==profile and a['current_pA']==cur)
  assert f['spikes']==target['target'],(f,target)
  save('postday_'+profile+'.json',dict(passed=True,profile=profile,current_pA=cur,features=f,target_spikes=target['target'],same_state_no_reset=True,checks=rr['checks']))
 print('day',profile,strict,'maxdrift',drift,flush=True)
 return r

def trace(support):
 edges=np.array([0,2,3,8,12,13,18,24])*3600.
 sg=m.schedule([(l,r,0,1,0) for l,r in zip(edges[:-1],edges[1:])]);mc=m.controls(sg);gc=controls(sg)
 for i,(l,r) in enumerate(zip(edges[:-1]/3600,edges[1:]/3600)):
  if 2<=l<8 or 12<=l<18:mc[i,3]=support
  if l in (2,12):gc[i,2]=1.
 r=api.run_gap43_regrowth_neuron(sg,**m.m.hist.options(),local_mrna_schedule=mc,gap43_regrowth_schedule=gc,output_step_s=120,max_step_s=30,rtol=1e-8,atol=1e-10)
 np.savez_compressed(B/('full_departure_'+str(support)+'.npz'),time_s=r['time_s'],state=r['state'],segments=gc)
 save('full_departure_'+str(support)+'.json',dict(completed=True,checks=r['checks'],config=r['config'],nfev=r['nfev'],scope='All620 states, virtual lesion is downstream only and does not sever the anatomical process'))
 print('full trace',support,'integrity',r['state'][-1,617],flush=True)
 return r

def replay(r,parameters,step=120.):
 v=parameters.vector();base=r['baseline_state'];ref=np.array([base[583],base[136]])
 t=r['time_s'];z=r['state'];gc=r['gap43_regrowth_segments'];y=np.array([1-v[5],0.,0.]);tt=[t[0]];yy=[y.copy()]
 for row in gc:
  l,h=row[:2];ctl=row[2:]
  def rhs(ti,yi):
   zz=np.r_[base[:617],yi];zz[583]=np.interp(ti,t,z[:,583]);zz[136]=np.interp(ti,t,z[:,136])
   return extension(zz,np.zeros(617),v,ctl,ref)[617:]
  grid=np.r_[np.arange(l+step,h,step),h]
  sol=solve_ivp(rhs,(l,h),y,method='LSODA',t_eval=grid,rtol=1e-10,atol=1e-12,max_step=step)
  assert sol.success;y=sol.y[:,-1];tt.extend(sol.t);yy.extend(sol.y.T)
 return np.array(tt),np.array(yy)

def scenarios():
 traces={s:trace(s) for s in (0.,.25,1.)};rows=[];err=0.
 for s,r in traces.items():
  for gain in (0.,.5,1.):
   for tau in (6.,24.,72.):
    t,y=replay(r,replace(GAP43RegrowthParameters(),gap43_dependence=gain,repair_time_h=tau))
    assert y.min()>-1e-8 and y[:,0].max()<=1+1e-8
    residual=float(np.max(abs(y[:,0]-1+y[:,1]-y[:,2])));assert residual<1e-8
    rows.append(dict(departure_support=s,gap43_dependence=gain,repair_time_h=tau,integrity24h=float(y[-1,0]),recovered_cumulative=float(y[-1,2]),ledger_error=residual))
    if gain==.5 and tau==24:
     e=float(np.max(abs(y[:,0]-np.interp(t,r['time_s'],r['state'][:,617]))));err=max(err,e)
  # Recovery history is retained, not normalized away after each insult.
 assert err<1e-4,err
 baseline=next(x for x in rows if x['departure_support']==1 and x['gap43_dependence']==.5 and x['repair_time_h']==24)
 affected=next(x for x in rows if x['departure_support']==0 and x['gap43_dependence']==.5 and x['repair_time_h']==24)
 # Null dependence leaves only tiny indirect ATP effects; do not force equality.
 nulls=[x for x in rows if x['gap43_dependence']==0 and x['repair_time_h']==24]
 assert max(x['integrity24h'] for x in nulls)-min(x['integrity24h'] for x in nulls)<1e-4
 t,y=replay(traces[0.],GAP43RegrowthParameters(),step=60);coarse_t,coarse_y=replay(traces[0.],GAP43RegrowthParameters());strict_error=float(np.max(abs(coarse_y-np.column_stack([np.interp(coarse_t,t,y[:,j]) for j in range(3)]))));assert strict_error<1e-6
 save('scenarios.json',dict(completed=True,full_neuron_trajectories=3,downstream_assays=27,rows=rows,
  nominal_control=baseline,nominal_delivery_block=affected,integrated_replay_error=err,replay_strict_error=strict_error,
  limitations='No human GAP43 dose response or regeneration speed; downstream normalized virtual-lesion readout, not new anatomical tissue. Same full molecular traces reused because downstream coupling has no feedback;27 assays are not27 independent full-neuron simulations.'))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(10,6))
 for s,r in traces.items():
  t=r['time_s']/3600;z=r['state']
  for a,yy in zip(axs.flat,[z[:,607]/z[0,607],z[:,583]/z[0,583],z[:,617],z[:,136]]):a.plot(t,yy,label=str(s))
 for a,title in zip(axs.flat,['Local free GAP43 mRNA / initial','Local GAP43 protein / initial','Virtual-lesion recovery index (not axon length)','Process ATP (uM)']):
  a.set_title(title);a.set_xlabel('Time (h)');a.grid(alpha=.2)
  for l,h in [(2,3),(12,13)]:a.axvspan(l,h,color='red',alpha=.1)
 axs[0,0].legend(title='mRNA departure support');fig.suptitle('Full-neuron inputs; conditional GAP43 recovery readout');fig.tight_layout();fig.savefig(B/'GAP43_recovery.png',dpi=140);plt.close(fig)
 print('27 downstream assays passed',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['unit','full','day','scenarios','resume','all'],default='all');q=p.parse_args();begin=time.perf_counter()
 if q.mode in ('unit','all'):unit()
 if q.mode in ('full','all'):full()
 if q.mode in ('day','all'):
  a=resting_day();b=resting_day(strict=True);c=resting_day('Selvaraj-Con1')
  error=float(np.max(abs(a['state']-b['state'])/(1+abs(b['state']))));assert error<1e-5
  save('whole_day.json',dict(passed=True,profiles=['Burley','Selvaraj-Con1'],strict_error=error,scope='All620 states at rest; source and parameter assumptions retained; not validation under arbitrary stimulation'))
 if q.mode=='resume':
  c=resting_day('Selvaraj-Con1')
  with np.load(B/'rest_Burley.npz') as aa, np.load(B/'rest_Burley_strict.npz') as bb:
   assert np.array_equal(aa['time_s'],bb['time_s']);error=float(np.max(abs(aa['state']-bb['state'])/(1+abs(bb['state']))));assert error<1e-5
  save('whole_day.json',dict(passed=True,profiles=['Burley','Selvaraj-Con1'],strict_error=error,scope='All620 states at rest; source and parameter assumptions retained; not validation under arbitrary stimulation'))
 if q.mode in ('scenarios','resume','all'):scenarios()
 print('completed seconds',time.perf_counter()-begin,flush=True)
