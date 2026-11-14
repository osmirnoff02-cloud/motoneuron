"""[codex]2026-11-02. Full-neuron maintenance/supply experiments, no reset."""
from pathlib import Path
import sys,importlib.util,json,argparse
from dataclasses import asdict,replace
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(1);B=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('renewal_prior_validation',B.parent/'2026-11-02-ribosome-maintenance/validate.py');q=importlib.util.module_from_spec(sp);sys.modules[sp.name]=q;sp.loader.exec_module(q)
api=q.api;m=q.m
def save(n,x):(B/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a),allow_nan=False)+'\n',encoding='utf8')
def run(sg,**kw):return api.run_ribosome_renewal_neuron(sg,**kw)
def assess(r):
 z=r['state'];b=r['baseline_state'];ch=r['checks'];assert max(ch.values())<3e-5,ch
 # Independent global ATP+ADP+AMP: individual compartments exchange adenylates.
 cfg=r['config'];g=cfg['geometry'];vs=g['soma_volume_um3'];vp=np.pi*g['process_radius_um']**2*g['process_length_um']*g['process_accessible_fraction']
 # Terminal volume is explicit in config; original branch also checks each flux.
 va=cfg['terminal']['volume_um3']
 aden=(z[:,120:123]-b[120:123]).sum(1)*vs+(z[:,136:139]-b[136:139]).sum(1)*vp+(z[:,282:285]-b[282:285]).sum(1)*va
 assert max(abs(aden))/(vs+vp+va)<3e-5
 assert z[:,675:].min()>=0 and z[:,675:].max()<=r['config']['ribosome_renewal']['maximum_capacity_factor']+1e-7
 return dict(completed=True,full_neuron=True,states=z.shape[1],checks=ch,nfev=r['nfev'],
  ATP_min_uM=z[:,[120,136]].min(0),soma_free_end_relative=z[-1,330:332]/b[330:332],
  local_free_end_relative=z[-1,564:566]/b[564:566],local_free_min_relative=(z[:,564:566]/b[564:566]).min(0),
  GAP43_end_relative=z[-1,583]/b[583],RPS26_end_relative=z[-1,643]/b[643],
  retired_Ceq=z[-1,652:654],RP_exchanged_Ceq=z[-1,651],RP_made_Ceq=z[-1,658],
  delivered_Ceq=z[-1,566:568],matured_Ceq=z[-1,396:398],soma_retired_Ceq=z[-1,398:400],
  precursor_RNA_end_Ceq=z[-1,384:388],precursor_RNA_initial_Ceq=b[384:388],
  capacity_end=z[-1,675:678],reference=r['config']['renewal_reference'])
def quick():
 sg=m.schedule([(0,60,0,1,0)])
 r=run(sg,**m.m.hist.options(),output_step_s=10,max_step_s=2,rtol=1e-8,atol=1e-10)
 save('quick.json',assess(r));print('quick',r['state'].shape,flush=True)
def rest():
 for profile in ['Burley','Selvaraj-Con1']:
  sg=m.schedule([(0,86400,0,1,0)]);r=run(sg,**m.m.hist.options(profile),output_step_s=300,max_step_s=60,rtol=1e-8,atol=1e-10)
  row=assess(r);save('rest_'+profile+'.json',row);np.savez_compressed(B/('rest_'+profile+'.npz'),time_s=r['time_s'],state=r['state'])
  print('rest',profile,row['local_free_end_relative'],row['GAP43_end_relative'],flush=True)
def challenges():
 sg=m.schedule([(0,21600,0,1,0),(21600,32400,0,1,0),(32400,43200,0,1,0),(43200,54000,0,1,0),(54000,86400,0,1,0)])
 for label in ['delivery_stop','production_stop','unadapted']:
  opts={}
  if label=='delivery_stop':
   c=np.c_[sg[:,:2],np.ones((5,3))];c[[1,3],2]=0;opts['local_ribosome_schedule']=c
  if label=='production_stop':
   c=np.c_[sg[:,:2],np.ones((5,7))];c[[1,3],2:4]=0;opts['biogenesis_schedule']=c
   sc=np.c_[sg[:,:2],np.ones((5,7))];sc[[1,3],8]=0;opts['shared_translation_schedule']=sc
  if label=='unadapted':opts['ribosome_renewal_schedule']=np.c_[sg[:,:2],np.zeros((5,3))]
  r=run(sg,**opts,**m.m.hist.options(),output_step_s=300,max_step_s=60,rtol=1e-8,atol=1e-10)
  row=assess(r);save(label+'.json',row);np.savez_compressed(B/(label+'.npz'),time_s=r['time_s'],state=r['state'])
  print(label,row['local_free_end_relative'],row['GAP43_end_relative'],flush=True)
def electrical():
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical.json').read_text())['rows'];metrics=m.module('renewal_features',B.parent/'2026-10-26-nak-source-calibration/validate_model.py');rows=[]
 for target in targets:
  p=target['profile'];cur=target['current_pA'];dur=.5 if p=='Burley' else 1.;sg=m.schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  r=run(sg,**m.m.hist.options(p),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  f=metrics.features(r,.2,.2+dur,p);assert f['spikes']==target['target'];delta=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.02
  rows.append(dict(profile=p,current_pA=cur,features=f,max_change=delta));save('electrical.json',dict(complete=len(rows)==9,rows=rows));print('electrical',p,cur,delta,flush=True)
def restart():
 sg=m.schedule([(0,120,0,1,0),(120,240,0,1,0)])
 opts=dict(**m.m.hist.options(),output_step_s=30,max_step_s=5,rtol=1e-9,atol=1e-11)
 r=run(sg,**opts);a=run(sg[:1],**opts);b=run(sg[1:],checkpoint=a['checkpoint'],**opts)
 strict=run(sg,**dict(opts,max_step_s=2,rtol=3e-10,atol=3e-12));err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
 er=err(b['state'][-1],r['state'][-1]);ec=err(strict['state'][-1],r['state'][-1]);assert max(er,ec)<3e-5
 save('restart.json',dict(passed=True,restart_error=er,strict_error=ec,checks=assess(r)))
 print('restart',er,ec,flush=True)
def all_tests():
 quick();restart();electrical();rest();challenges()

def postday():
 # Check firing after an actual unreset day of continuous maintenance.
 metrics=m.module('renewal_postday_features',B.parent/'2026-10-26-nak-source-calibration/validate_model.py')
 targets=json.loads((B/'electrical.json').read_text())['rows'];rows=[]
 for p in ['Burley','Selvaraj-Con1']:
  a=run(m.schedule([(0,86400,0,1,0)]),**m.m.hist.options(p),output_step_s=600,max_step_s=60,rtol=1e-8,atol=1e-10)
  assess(a);cp=a['checkpoint'];save('checkpoint_'+p+'.json',cp)
  t=86400.;dur=.5 if p=='Burley' else 1.;sg=m.schedule([(t,t+.2,0,1,0),(t+.2,t+.2+dur,0,1,0),(t+.2+dur,t+1.2+dur,0,1,0)])
  r=run(sg,**m.m.hist.options(p),checkpoint=cp,electrical_schedule=np.c_[sg[:,:2],[0,25,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  f=metrics.features(r,t+.2,t+.2+dur,p);old=next(x['features'] for x in targets if x['profile']==p and x['current_pA']==25)
  assert f['spikes']==old['spikes']
  delta=max(abs(float(f[k])-float(x)) for k,x in old.items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.05
  rows.append(dict(profile=p,features=f,initial_features=old,max_feature_change=delta,checkpoint_without_reset=True));save('postday.json',dict(passed=True,complete=len(rows)==2,rows=rows))
  print('postday',p,f['spikes'],delta,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['quick','rest','challenges','electrical','restart','all_tests','postday']);globals()[p.parse_args().mode]()
