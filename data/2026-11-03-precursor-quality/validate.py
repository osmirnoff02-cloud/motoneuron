"""[codex] 2026-11-03. Explicit tests; no human parameter fitting."""
from pathlib import Path
import sys,importlib.util,json
import numpy as np
from dataclasses import replace
from scipy.integrate import solve_ivp
B=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('quality_previous_validation',B.parent/'2026-11-02-ribosome-surplus/validate.py');q=importlib.util.module_from_spec(s);sys.modules[s.name]=q;s.loader.exec_module(q)
api=q.api;m=q.m;pq=api._pq41;sp=api._sp40

def save(name,x):
 (B/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a),allow_nan=False)+'\n',encoding='utf8')
def run(sg,**kw):return api.run_precursor_quality_neuron(sg,**kw)
def assess(r):
 z=r['state'];b=r['baseline_state'];checks=r['checks'];assert z.shape[1]==len(r['columns'])==726
 assert max(checks.values())<3e-5,sorted(checks.items(),key=lambda kv:-kv[1])[:8]
 o=r['precursor_quality_observation'];H=z[:,384:392];D=z[:,692:700]+z[:,700:708]
 assert np.max(abs(o['healthy_waiting_Ceq']+o['healthy_excess_Ceq']-H))<1e-10
 return dict(completed=True,states=726,checks=checks,nfev=r['nfev'],
  healthy_end_relative=H[-1]/b[384:392],healthy_peak_relative=(H/b[384:392]).max(0),
  defect_peak_Ceq=D.max(0),defect_end_Ceq=D[-1],damage_cumulative_Ceq=z[-1,708:716],
  defect_disposed_Ceq=z[-1,716:724],healthy_disposed_Ceq=z[-1,678:686]-z[-1,716:724],
  local_free_end_relative=z[-1,564:566]/b[564:566],GAP43_end_relative=z[-1,583]/b[583],
  ATP_min_uM=z[:,[120,136]].min(0),recognition_ATP_uM=z[-1,724:726],
  scope='Conditional model scenario; no human kinetic calibration; no correction to known long ionic drift')
def basic():
 r=run(m.schedule([(0,60,0,1,0)]),**m.m.hist.options(),output_step_s=10,max_step_s=2,rtol=1e-8,atol=1e-10)
 save('quick.json',assess(r));np.savez_compressed(B/'baseline.npz',state=r['baseline_state'])
 print('quick',r['state'].shape,flush=True)
 return r
def electrical():
 oldrun,oldassess,oldB=q.run,q.assess,q.B
 q.run,q.assess,q.B=run,assess,B
 try:q.electrical()
 finally:q.run,q.assess,q.B=oldrun,oldassess,oldB
def restart():
 sg=m.schedule([(0,120,0,1,0),(120,240,0,1,0)])
 ct=np.c_[sg[:,:2],np.tile([1,1,1,1,1,1,1.],(2,1))]
 opts=dict(**m.m.hist.options(),output_step_s=20,max_step_s=5,rtol=1e-8,atol=1e-10)
 r=run(sg,precursor_quality_schedule=ct,**opts)
 a=run(sg[:1],precursor_quality_schedule=ct[:1],**opts)
 b=run(sg[1:],precursor_quality_schedule=ct[1:],checkpoint=a['checkpoint'],**opts)
 tight=run(sg,precursor_quality_schedule=ct,**dict(opts,rtol=1e-9,atol=1e-11))
 er=lambda a,b:float(np.max(abs(a-b)/(1+abs(b))))
 re=er(b['state'][-1],r['state'][-1]);te=er(tight['state'][-1],r['state'][-1]);assert max(re,te)<3e-5
 for x in (r,a,b,tight):assess(x)
 save('restart.json',dict(passed=True,restart_error=re,tolerance_error=te,damage_active=True));print('restart',re,te,flush=True)
def boundaries():
 base=np.ones(726);base[120]=1000.;base[678:]=0.;z=base.copy();z[384:392]=5.;v=pq.PrecursorQualityParameters().vector();sv=sp.RibosomeSurplusParameters().vector();recipe=sp.RECIPE_VECTOR;C=.01
 ctl=np.array([0,0,0,1,1,1,1.]);sc=np.ones(2)
 def f(x=z,c=ctl,prot=1.):return pq.finish(x,np.zeros(692),base,v,c,sv,sc,recipe,C,100.,prot)
 a=f();assert np.all(a[692:724]==0);assert np.all(a[[384,385,386,387,390,391]]==0);assert np.all(a[388:390]<0)
 d=z.copy();d[692:700]=1;d[700:708]=1
 noATP=d.copy();noATP[120]=0;assert np.all(f(noATP)==0)
 ctl_injury=ctl.copy();ctl_injury[:3]=1
 depleted=f(noATP,ctl_injury);assert np.all(depleted[692:700]>0) and np.all(depleted[700:708]==0)
 no_rec=ctl.copy();no_rec[3]=0;d0=d.copy();d0[700:708]=0;assert np.all(f(d0,no_rec)[692:708]==0)
 no_prot=f(d,ctl,0);assert np.all(no_prot[716+4:724]==0) and np.all(no_prot[716:720]>0)
 no_RNA=ctl.copy();no_RNA[4]=0;nr=f(d,no_RNA);assert np.all(nr[716:720]==0) and np.all(nr[722:724]==0)
 a=f(d);assert a[513]>0 and a[497]==0 and a[499]==0
 assert abs(a[120]+a[121])<1e-12
 assert np.allclose(a[692:700]+a[700:708]+a[716:724],a[708:716])
 # Check malformed schedules and negative kinetic parameters fail.
 try:pq.controls(np.array([[0,1.]]),np.zeros((1,9)))
 except ValueError:pass
 else:raise AssertionError('Bad schedule accepted')
 try:replace(pq.PrecursorQualityParameters(),recognition_s=-1).vector()
 except ValueError:pass
 else:raise AssertionError('Negative rate accepted')
 save('boundaries.json',dict(passed=True,checks=['delay_not_damage','healthy_RNA_RNP_retained','excess_RP_disposed','zero_ATP_no_handling','damage_can_occur_without_ATP','recognition_off_retains_unrecognized','protein_gate','RNA_gate','no_instant_AA_GTP','adenylate_conservation','defect_material_conservation','bad_schedule_rejected','negative_parameters_rejected']))
 print('boundaries passed',flush=True)
def isolated_grid():
 # Closed precursor subsystem: no new synthesis/maturation. Independent analytic
 # damage survival validates effective hazard; no claim of full-neuron robustness.
 rows=[];recipe=sp.RECIPE_VECTOR;C=.01;sv=sp.RibosomeSurplusParameters().vector();sc=np.array([0.,0.])
 for rate in [.001,.03,.3]:
  for recog in [300.,3600.,21600.]:
   for clear in [300.,7200.,86400.]:
    v=replace(pq.PrecursorQualityParameters(),damage_per_h=rate,recognition_s=recog,RNA_clearance_s=clear,protein_clearance_s=clear).vector()
    base=np.zeros(726);base[384:392]=1.;base[120]=1e7;y=base.copy()
    for start,end,inj in [(0,3600,1),(3600,7200,0),(7200,10800,1),(10800,86400,0)]:
     ctl=np.array([inj,inj,inj,1.,1.,1.,0.])
     def rhs(t,y):return pq.finish(y,np.zeros(692),base,v,ctl,sv,sc,recipe,C,100.,1.)
     r=solve_ivp(rhs,(start,end),y,method='LSODA',rtol=1e-9,atol=1e-11);assert r.success;y=r.y[:,-1]
    checks=pq.checks(y[None,:],base,v,C,recipe);assert max(checks.values())<1e-7
    assert np.allclose(y[384:392],np.exp(-rate*2),rtol=1e-7)
    rows.append(dict(damage_per_h=rate,recognition_s=recog,clearance_s=clear,remaining_defect_Ceq=(y[692:700]+y[700:708]).sum(),disposed_Ceq=y[716:724].sum(),max_balance=max(checks.values())))
 save('isolated_grid.json',dict(passed=True,rows=rows,scope='27 closed precursor-only scenarios; ATP initially1e7uM, artificial supply stress excluded, not neuronal concentration/whole-model ensemble'))
 print('isolated grid27 passed',flush=True)
def scenarios(names=None):
 sg=m.schedule([(0,7200,0,1,0),(7200,14400,0,1,0),(14400,28800,0,1,0),(28800,36000,0,1,0),(36000,86400,0,1,0)])
 names=names or ['control','delay_only','damage','recognition_off','slow_disposal']
 for name in names:
  bc=np.c_[sg[:,:2],np.ones((5,7))];ct=pq.controls(sg)
  if name!='control':bc[[1,3],4]=0.
  if name not in ('control','delay_only'):ct[np.ix_([1,3],[2,3,4])]=1.
  if name=='recognition_off':ct[:,5]=0.
  p=pq.PrecursorQualityParameters()
  if name=='slow_disposal':p=replace(p,RNA_clearance_s=86400,protein_clearance_s=86400)
  r=run(sg,**m.m.hist.options(),biogenesis_schedule=bc,precursor_quality_schedule=ct,precursor_quality_parameters=p,output_step_s=600,max_step_s=120,rtol=1e-8,atol=1e-10)
  row=assess(r);save(name+'.json',row);save(name+'_checkpoint.json',r['checkpoint']);np.savez_compressed(B/(name+'.npz'),time_s=r['time_s'],state=r['state'])
  print(name,'max defect',max(row['defect_peak_Ceq']),'max check',max(row['checks'].values()),flush=True)
if __name__=='__main__':
 action=sys.argv[1] if len(sys.argv)>1 else 'basic'
 if action=='basic':boundaries();isolated_grid();basic()
 elif action=='electrical':electrical()
 elif action=='restart':restart()
 elif action=='scenarios':scenarios(sys.argv[2:] or None)
 elif action=='integrated':
  restart();electrical();scenarios()
