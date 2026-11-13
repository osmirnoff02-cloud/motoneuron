"""[codex]2026-11-02. Source and connected-neuron regression experiments."""
from pathlib import Path
import sys,importlib.util,json
import numpy as np
from dataclasses import replace
from threadpoolctl import threadpool_limits
threadpool_limits(1);B=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('surplus_previous_validation',B.parent/'2026-11-02-ribosome-renewal/validate_codex.py');q=importlib.util.module_from_spec(s);sys.modules[s.name]=q;s.loader.exec_module(q)
api=q.api;m=q.m
def save(n,x):(B/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a),allow_nan=False)+'\n',encoding='utf8')
def run(sg,**kw):return api.run_ribosome_surplus_neuron(sg,**kw)
def assess(r,require_strict=True):
 z=r['state'];b=r['baseline_state'];ch=r['checks']
 if require_strict:assert max(ch.values())<3e-5,ch
 assert len(r['columns'])==692 and z.shape[1]==692
 g=r['config']['geometry'];vs=g['soma_volume_um3'];vp=np.pi*g['process_radius_um']**2*g['process_length_um']*g['process_accessible_fraction'];va=r['config']['terminal']['volume_um3']
 aden=(z[:,120:123]-b[120:123]).sum(1)*vs+(z[:,136:139]-b[136:139]).sum(1)*vp+(z[:,282:285]-b[282:285]).sum(1)*va
 assert max(abs(aden))/(vs+vp+va)<3e-5
 return dict(completed=True,strict_balance_pass=max(ch.values())<3e-5,states=692,checks=ch,nfev=r['nfev'],ATP_min_uM=z[:,[120,136]].min(0),
  local_free_end_relative=z[-1,564:566]/b[564:566],GAP43_end_relative=z[-1,583]/b[583],
  precursor_end_relative=z[-1,384:392]/b[384:392],precursor_peak_relative=(z[:,384:392]/b[384:392]).max(0),
  disposed_Ceq=z[-1,678:686],coproduct_residues_uM=z[-1,686:688],coproduct_returned_uM=z[-1,688:690],
  extra_disposal_ATP_uM=z[-1,690:692],RP_completed_kits_Ceq=z[-1,394:396],
  rp_synthesis_high_energy_uM=4*r['config']['biogenesis']['selected_mature_pool_uM']*z[-1,[414,429]],
  scope='Chosen parameter scenario, no human kinetic fit')
def quick():
 r=run(m.schedule([(0,60,0,1,0)]),**m.m.hist.options(),output_step_s=10,max_step_s=2,rtol=1e-8,atol=1e-10)
 save('quick_codex.json',assess(r));print('quick',r['state'].shape,flush=True)
 return r
def electrical():
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical_codex.json').read_text())['rows'];metrics=m.module('surplus_features',B.parent/'2026-10-26-nak-source-calibration/validate_model_codex.py');rows=[]
 for target in targets:
  p=target['profile'];cur=target['current_pA'];dur=.5 if p=='Burley' else 1.;sg=m.schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  r=run(sg,**m.m.hist.options(p),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  assess(r);f=metrics.features(r,.2,.2+dur,p);assert f['spikes']==target['target'];delta=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.02
  rows.append(dict(profile=p,current_pA=cur,features=f,max_change=delta));save('electrical_codex.json',dict(complete=len(rows)==9,rows=rows));print('electrical',p,cur,delta,flush=True)
def restart():
 sg=m.schedule([(0,120,0,1,0),(120,240,0,1,0)]);opts=dict(**m.m.hist.options(),output_step_s=30,max_step_s=5,rtol=1e-9,atol=1e-11)
 r=run(sg,**opts);a=run(sg[:1],**opts);b=run(sg[1:],checkpoint=a['checkpoint'],**opts);strict=run(sg,**dict(opts,max_step_s=2,rtol=1e-8,atol=1e-10))
 err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
 er=err(b['state'][-1],r['state'][-1]);ec=err(strict['state'][-1],r['state'][-1]);assert max(er,ec)<3e-5
 save('restart_codex.json',dict(passed=True,restart_error=er,tolerance_comparison_error=ec,tolerances_compared=[[1e-9,1e-11],[1e-8,1e-10]],extreme_tolerance_attempt='3e-10/3e-12 interrupted due tiny LSODA steps, not a passed check',checks=assess(r)));print('restart',er,ec,flush=True)
def long_runs():
 for profile in ['Burley','Selvaraj-Con1']:
  r=run(m.schedule([(0,259200,0,1,0)]),**m.m.hist.options(profile),output_step_s=600,max_step_s=120,rtol=1e-8,atol=1e-10)
  save('rest72h_'+profile+'_codex.json',assess(r));np.savez_compressed(B/('rest72h_'+profile+'_codex.npz'),time_s=r['time_s'],state=r['state']);save('checkpoint72h_'+profile+'_codex.json',r['checkpoint']);print('rest72h',profile,assess(r)['precursor_end_relative'],flush=True)
def stress():
 sg=m.schedule([(0,21600,0,1,0),(21600,43200,0,1,0),(43200,86400,0,1,0),(86400,108000,0,1,0),(108000,172800,0,1,0)])
 bc=np.c_[sg[:,:2],np.ones((5,7))];bc[[1,3],4]=0.
 for name,p,c in [('balanced',api.RibosomeSurplusParameters(),[1,1]),('without_surveillance',api.RibosomeSurplusParameters(),[0,0]),('slow_clearance',replace(api.RibosomeSurplusParameters(),clearance_s=86400),[1,1])]:
  r=run(sg,**m.m.hist.options(),biogenesis_schedule=bc,ribosome_surplus_parameters=p,ribosome_surplus_schedule=np.c_[sg[:,:2],np.tile(c,(5,1))],output_step_s=600,max_step_s=120,rtol=1e-8,atol=1e-10)
  save('stress_'+name+'_codex.json',assess(r));np.savez_compressed(B/('stress_'+name+'_codex.npz'),time_s=r['time_s'],state=r['state']);print('stress',name,assess(r)['precursor_peak_relative'],flush=True)
def source_tests():
 v=api._sp40.RECIPE_VECTOR;assert np.array_equal(v[:,0],v[:,1]+v[:,2]);assert int(v[:,3].sum())==82
 h=api.observe_RP_heavy_fraction([1,3],[3,1]);assert np.allclose(h,[.25,.75])
 t=np.array([0.,1.,2.,4.,8.]);o=api.ribosomal_pulse_chase_observer(t);assert o['bound_heavy_fraction'][3]>o['bound_heavy_fraction'][1]
 s=api.ribosome_surplus_source_evidence()['source_data']['sheets'];r=s['Figure3']['RPS26'][0]['values']
 assert r[3]>r[2];save('source_checks_codex.json',dict(passed=True,observed_RPS26_pulse=r[2],observed_after_chase=r[3],observer_bound_H=o['bound_heavy_fraction'],source_fit=False,
  note='Qualitative tracer-delay check, no fitted human rate. Fig5b RPS26 effects fail multiple-testing correction.'))
 print('source checks',flush=True)
def post72h(profiles=('Burley','Selvaraj-Con1'),rows=None):
 metrics=m.module('surplus_post72_features',B.parent/'2026-10-26-nak-source-calibration/validate_model_codex.py');targets=json.loads((B/'electrical_codex.json').read_text())['rows'];rows=[] if rows is None else list(rows)
 for p in profiles:
  cp=json.loads((B/('checkpoint72h_'+p+'_codex.json')).read_text());t=cp['time_s'];dur=.5 if p=='Burley' else 1.;sg=m.schedule([(t,t+.2,0,1,0),(t+.2,t+.2+dur,0,1,0),(t+.2+dur,t+1.2+dur,0,1,0)])
  r=run(sg,**m.m.hist.options(p),checkpoint=cp,electrical_schedule=np.c_[sg[:,:2],[0,25,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  assess(r);f=metrics.features(r,t+.2,t+.2+dur,p);old=next(x['features'] for x in targets if x['profile']==p and x['current_pA']==25)
  differences={k:float(f[k])-float(x) for k,x in old.items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k])};delta=max(abs(x) for x in differences.values())
  rows.append(dict(profile=p,features=f,differences=differences,max_feature_change=delta,spike_count_preserved=f['spikes']==old['spikes'],strict_shape_drift_pass=delta<.05,unreset=True));save('post72h_codex.json',dict(complete=len(rows)==2,rows=rows,strict_shape_drift_threshold=.05));print('post72h',p,f['spikes'],delta,flush=True)
def continuation14d():
 cp=json.loads((B/'checkpoint72h_Burley_codex.json').read_text());sg=m.schedule([(cp['time_s'],14*86400,0,1,0)])
 r=run(sg,**m.m.hist.options(),checkpoint=cp,output_step_s=3600,max_step_s=300,rtol=1e-8,atol=1e-10)
 row=assess(r,require_strict=False);z=r['state'];b=r['baseline_state'];t=r['time_s'];i=np.argmin(abs(t-(t[-1]-86400)))
 row['last_day_precursor_change_in_initial_units']=(z[-1,384:392]-z[i,384:392])/b[384:392]
 row['soma_process_Ki_end_relative']=z[-1,[200,220]]/b[[200,220]]
 save('continuation14d_codex.json',row);np.savez_compressed(B/'continuation14d_codex.npz',time_s=t,state=z)
 print('14days',row['precursor_end_relative'],row['last_day_precursor_change_in_initial_units'],flush=True)
if __name__=='__main__':
 source_tests();quick()
