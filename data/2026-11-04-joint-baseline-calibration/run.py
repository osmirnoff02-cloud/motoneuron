"""[codex] Full4413-state joint baseline and no-reset protocol validation."""
from pathlib import Path
import sys,json,time,copy,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
_limit=threadpool_limits(limits=1)
B=Path(__file__).resolve().parent;D=B.parent
if '--adaptive' in sys.argv:raise ValueError('Experimental colored solver was not accepted; use --sparse for full-column BDF')
SPARSE='--sparse' in sys.argv
PREFIX='joint_v2_sparse_' if SPARSE else 'joint_v2_'
sys.path.insert(0,str(D/'neuron-model'))
from model_contracts import load_module
api=load_module(D/'neuron-model/model_api.py','joint04_api')
setup=load_module(D/'2026-11-04-integrated-neuron/validate.py','joint04_setup')
mod=load_module(B/'profile.py','joint04_profile')
features_mod=load_module(D/'2026-10-26-nak-source-calibration/validate_model.py','joint04_features')
schedule=setup.schedule
def save(name,obj):
 (B/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n',encoding='utf8')
def options(profile='Burley'):
 return mod.joint_inputs(api,profile)
def summary(r):
 ch={k:float(v) for k,v in r['checks'].items() if k!='minimum_state'}
 z=r['state'];idx=[6,7,8,9,10,11,50,120,121,122,136,137,138,185,192,199,200,201,202,282,283,284,315,316,317,322,324]
 return dict(states=z.shape[1],nfev=r['nfev'],worst_balance=max(ch.values()),checks=ch,
     selected={r['columns'][i]:dict(index=i,initial=float(z[0,i]),final=float(z[-1,i]),min=float(z[:,i].min()),max=float(z[:,i].max())) for i in idx},
     all_finite=bool(np.isfinite(z).all()),duration_s=float(r['time_s'][-1]-r['time_s'][0]))
def run(name,sg,opts,**kw):
 name=PREFIX+name
 if SPARSE:kw.update(solver_method='BDF',solver_sparse_jacobian=True)
 started=time.monotonic();print(name,'start',flush=True)
 def observer(cp,n):
  save(name+'_progress_checkpoint',cp)
  if sg[-1,1]-sg[0,0]>3600:save(name+'_segment_'+str(int(cp['time_s']))+'_checkpoint',cp)
  save(name+'_progress',dict(time_s=cp['time_s'],elapsed_s=time.monotonic()-started,nfev=n))
  print(name,cp['time_s'],n,round(time.monotonic()-started,1),flush=True)
 try:r=api.run_integrated_neuron(sg,**opts,solver_observer=observer,**kw)
 except (ArithmeticError,ValueError) as exc:
  tb=exc.__traceback__
  while tb:
   local=tb.tb_frame.f_locals
   if 'checks' in local and 'z' in local and 't' in local:
    save(name+'_FAILED_checks',dict(error=str(exc),checks=local['checks'],accepted=False))
    np.savez_compressed(B/(name+'_FAILED_trace.npz'),time_s=local['t'],state=local['z'],baseline=local['base'])
   if 'checked_recycling' in local and 'y' in local and 'base' in local:
    save(name+'_FAILED_recycling',dict(error=str(exc),checks=local['checked_recycling'](local['y'][None,:],local['base']),accepted=False))
   tb=tb.tb_next
  raise
 s=summary(r);s['solver_settings']=r.get('solver_settings');s['elapsed_s']=time.monotonic()-started
 save(name,s)
 if 'rate_diagnostics' in r:save(name+'_diagnostics',r['rate_diagnostics'])
 save(name+'_checkpoint',r['checkpoint']);save(name+'_config',r['config'])
 np.savez_compressed(B/(name+'_trace.npz'),time_s=r['time_s'],state=r['state'],baseline=r['baseline_state'])
 save('columns',r['columns']);print(name,'DONE',s['worst_balance'],s['elapsed_s'],flush=True)
 return r
def pilot():
 o,p=options();save('joint_v2_candidate',p)
 r=run('pilot',schedule([(0,1.,0,1,0)]),o,output_step_s=.1,max_step_s=.1,rtol=1e-8,atol=1e-10)
 assert summary(r)['worst_balance']<3e-5
def electrical():
 targets=json.loads((D/'2026-10-29-joint-recovery/electrical.json').read_text())['rows'];rows=[]
 for ref in targets:
  profile=ref['profile'];amp=ref['current_pA'];on=.2;off=.7 if profile=='Burley' else 1.2
  sg=schedule([(0,on,0,1,0),(on,off,0,1,0),(off,off+1,0,1,0)])
  o,p=options(profile)
  r=run('electric_'+profile+'_'+str(amp),sg,o,electrical_schedule=np.c_[sg[:,:2],[0,amp,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-8,atol=1e-10)
  feat=features_mod.features(r,on,off,profile)
  deltas={k:float(feat[k])-v for k,v in ref['features'].items() if isinstance(v,(int,float)) and np.isfinite(v) and np.isfinite(feat[k])}
  rows.append(dict(profile=profile,current_pA=amp,features=feat,old_features=ref['features'],deltas=deltas,target=ref['target'],spike_pass=feat['spikes']==ref['target']))
  save(PREFIX+'electrical',dict(complete=len(rows)==len(targets),rows=rows,all_spike_counts_pass=all(x['spike_pass'] for x in rows)))
def rest(hours=72.):
 o,p=options();save('joint_v2_candidate',p)
 edges=np.unique(np.r_[0,1,10,60,600,3600,np.arange(6*3600,hours*3600,6*3600),hours*3600]);edges=edges[edges<=hours*3600]
 sg=schedule([(a,b,0,1,0) for a,b in zip(edges[:-1],edges[1:])])
 run('rest_'+str(hours)+'h',sg,o,output_step_s=300,max_step_s=300,rtol=1e-8,atol=1e-10,center_solver_coordinates=True,retain_dense_solver_output=False,diagnostic_rates=True)
def sparse_check():
 o,_=options()
 old=np.load(B/'pilot_trace.npz')
 # The earlier pilot used v1 CK interpolation: reconstruct that exact baseline
 # for a solver comparison, not for the accepted v2 scientific profile.
 from dataclasses import replace
 ck=json.loads((B/'candidate.json').read_text());e=ck['energy']
 keys=('ATP0_uM','ADP0_uM','AMP0_uM')
 o['energy_parameters']=replace(o['energy_parameters'],**{k:e[k] for k in keys},gly_half_ADP_uM=e['ADP0_uM'])
 o['creatine_parameters']=replace(o['creatine_parameters'],reverse_equilibrium_factor=1/ck['thermodynamics']['Kobs'])
 r=run('sparse_check',schedule([(0,1.,0,1,0)]),o,output_step_s=.1,max_step_s=.1,rtol=1e-8,atol=1e-10,solver_method='BDF',solver_sparse_jacobian=True)
 assert np.array_equal(r['baseline_state'],old['baseline'])
 a=np.array([np.interp(old['time_s'],r['time_s'],r['state'][:,i]) for i in range(4413)]).T
 delta=abs(a-old['state']);rel=delta/(1e-8+np.maximum(abs(a),abs(old['state'])))
 save('solver_comparison',dict(max_absolute=float(delta.max()),max_scaled=float(rel.max()),states=4413,
  voltage_max_mV=float(delta[:,[185,192]].max()),energy_max_uM=float(delta[:,[120,121,122,315,316,317]].max()),
  passed=bool(np.max(rel)<2e-3),scope='matched v1 one-second full state trajectory; v1 chemical candidate was superseded after source table check'))
 assert rel.max()<2e-3

def loads(branch):
 # One physical trajectory per protocol from the same common rest checkpoint.
 # Input is AMPA mean opening; no conversion from arbitrary applied Ca(t).
 if branch not in ('control','continuous','repeated'):raise ValueError(branch)
 cp=json.loads((B/'joint_v2_sparse_rest_72.0h_checkpoint.json').read_text());start=cp['time_s'];end=start+24*3600
 o,record=options();edges=np.unique(np.r_[start+np.array([0,.01,.1,1,10,60,600]),np.arange(start,end+1,3600.)])
 rows=[]
 for a,b in zip(edges[:-1],edges[1:]):
  h=int((a-start)/3600)
  active=(branch=='continuous' and h<3) or (branch=='repeated' and h in (0,8,16))
  rows.append((a,b,.001 if active else 0,1,0))
 sg=schedule(rows)
 r=run('load_'+branch,sg,o,checkpoint=cp,output_step_s=300,max_step_s=300,rtol=1e-8,atol=1e-10,
   center_solver_coordinates=True,retain_dense_solver_output=False,diagnostic_rates=True)
 assert np.array_equal(r['state'][0],cp['state']) and r['checkpoint']['origin_time_s']==cp['origin_time_s']
 save(PREFIX+'load_'+branch+'_protocol',dict(start_checkpoint='joint_v2_sparse_rest_72.0h_checkpoint.json',
  baseline_healthy_validated=False,AMPA_mean_open=.001,active_hours=0 if branch=='control' else 3,
  input_opening_integral_s=0 if branch=='control' else 10.8,reset=False,
  equal_opening_time_is_not_equal_calcium_entry=True,conditions=record))

def load_suite():
 for branch in ('continuous','repeated'):loads(branch)

def electrical_convergence():
 o,_=options();sg=schedule([(0,.2,0,1,0),(.2,.7,0,1,0),(.7,1.7,0,1,0)])
 run('electric_Burley_15_tight',sg,o,electrical_schedule=np.c_[sg[:,:2],[0,15,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)

def warm_electrical():
 cp=json.loads((B/'joint_v2_sparse_rest_72.0h_checkpoint.json').read_text());a=cp['time_s'];o,_=options()
 sg=schedule([(a,a+.2,0,1,0),(a+.2,a+.7,0,1,0),(a+.7,a+1.7,0,1,0)])
 r=run('electric_Burley_25_after72h',sg,o,checkpoint=cp,electrical_schedule=np.c_[sg[:,:2],[0,25,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-8,atol=1e-10)
 feat=features_mod.features(r,a+.2,a+.7,'Burley')
 save('warm_electrical',dict(profile='Burley',current_pA=25,start_h=72,reset=False,features=feat,initial_target=4,spike_pass=feat['spikes']==4,scope='One post-preparation diagnostic; Selvaraj after72h not tested',healthy_baseline_validated=False))

def inspect_rest_checkpoint():
 o,_=options();cp=json.loads((B/'joint_v2_sparse_rest_72.0h_progress_checkpoint.json').read_text());a=cp['time_s']
 r=run('rest_checkpoint_check',schedule([(a,a+.01,0,1,0)]),o,checkpoint=cp,output_step_s=.01,max_step_s=.01,rtol=1e-8,atol=1e-10)
 assert np.array_equal(r['state'][0],cp['state'])
 save('joint_v2_sparse_rest_72.0h_checkpoint',cp)
 save('recovered_rest_checkpoint_validation',dict(accepted=True,scope='72h original checkpoint validated at restart and after10ms; RHS unchanged by positivity-view correction',start_time_s=a,checks=r['checks']))

if __name__=='__main__':
 action=sys.argv[1] if len(sys.argv)>1 else 'pilot'
 if action=='loads':loads(sys.argv[2])
 elif action=='rest':rest(float(sys.argv[2]) if len(sys.argv)>2 else 72.)
 else:globals()[action]()
