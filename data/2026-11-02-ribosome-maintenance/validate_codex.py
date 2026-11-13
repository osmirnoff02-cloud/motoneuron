"""[codex]2026-11-02. Independent material/cost tests and coupled regressions."""
from pathlib import Path
import sys,importlib.util,json,argparse
from dataclasses import replace,asdict
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
threads=threadpool_limits(1);B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
sp=importlib.util.spec_from_file_location('maintenance_prior',B.parent/'2026-11-02-local-rqc/validate_codex.py');q=importlib.util.module_from_spec(sp);sys.modules[sp.name]=q;sp.loader.exec_module(q)
from maintenance_codex import *
api=q.api;m=q.m
def save(n,x):(B/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a))+'\n',encoding='utf8')

def setup(par=None):
 z,old,ch,tv,C,K,rq=q.setup();v=(par or RibosomeMaintenanceParameters()).vector();z=initialize(z,v)
 def component(y,ctl):return extension(y,old(y,np.array([0.,1.,1.,1.])),v,ctl,tv,C,4.,K,7500.,12500.)
 def check(y,b):
  yy,bb=adjusted_particles(y,b)
  return revise_checks(ch(yy,bb),y,b,v,tv,C,4.,7500.,12500.)
 return z,component,check,tv,C,K,v

def unit():
 z,comp,check,tv,C,K,v=setup();z[640]=.02;z[673]=.03;z[641:643]=[.01,.02];z[644:648]=.001;z[648]=.002;z[649]=1
 d=extension(z,np.zeros(640),v,np.ones(7),tv,C,4.,K,7500.,12500.);scale=C*4
 # Independent conservation of actual 40/60 particles plus disposed history.
 assert abs(d[564]+d[640]+d[641]+sum(d[644:649])+d[654])<1e-14
 assert abs(d[565]+d[673]+d[642]+sum(d[644:649])+d[655])<1e-14
 nas=d[644:648]@np.arange(4)*115/4*scale
 rp_mass=(d[564]+d[640]+d[641]+sum(d[644:649]))*7500*scale+(d[565]+d[673]+d[642]+sum(d[644:649]))*12500*scale
 assert abs(d[574]+d[575]+nas+115*d[643]+d[584]+d[649]+rp_mass)<1e-10
 assert abs(d[150]-sum(d[663:668])-d[672])<1e-14 and d[136]==-d[150] and d[137]==d[150]
 assert abs(d[576]+d[577])<1e-14
 for name,idx in [('ATP',136),('GTP',576),('charged',575),('spare',643)]:
  zz=z.copy();zz[idx]=0;dd=extension(zz,np.zeros(640),v,np.ones(7),tv,C,4.,K,7500.,12500.)
  if name=='ATP':assert dd[[651,652,653,654,655,656,659,671]].max()==0
  if name in ('GTP','charged'):assert dd[660]==0
  if name=='spare':assert dd[651]==0
 zz=z.copy();dd=extension(zz,np.zeros(640),v,np.array([1.,1.,1.,1.,0.,1.,1.]),tv,C,4.,K,7500.,12500.)
 assert dd[656]==0 and dd[649]>0 # digestion alone cannot refill cytosol
 z,comp,ch,*_=setup();err=ch(np.array([z,z+comp(z,np.ones(7))*1e-6]),z);assert max(err.values())<1e-7,err
 save('unit_codex.json',dict(passed=True,actual_particle_and_RP_residue_conservation=True,no_RNA_to_AA=True,no_double_GTP_ATP=True,finite_resource_gates=True,checks=err))
 print('unit passed',flush=True)

def molecular(par=None,repair=1.,retire=1.,synthesis=1.,strict=False):
 base,comp,ch,tv,C,K,v=setup(par);origin=base.copy()
 idx=np.unique(np.r_[m.m.fusco.INDEX,[136,137,148,150],np.arange(562,675)])
 def cost(z,d):
  bg=.5*max(z[136],0)/(K+max(z[136],0))*(K+base[136])/base[136]
  return np.array([m.m.fusco.demand(d)+d[572]+d[614]+d[615],bg+d[573]+d[594]+d[595]+d[598]+d[592]-d[593]+d[616]+sum(d[634:638:2])+d[637]+sum(d[663:668])+d[672]])
 rate=cost(base,comp(base,np.array([0.,1.,0.,1.,1.,0.,0.])))
 state=base[idx].copy();times=[0.];states=[state.copy()]
 for lo,hi in [(0,2),(2,8),(8,12),(12,18),(18,24)]:
  ctl=np.array([1. if lo in (2,12) else 0.,repair,retire if lo else 0.,1.,1.,synthesis if lo else 0.,1. if lo else 0.])
  def rhs(t,y):
   z=base.copy();z[idx]=y;d=comp(z,ctl);spent=cost(z,d)
   for j,(ai,di,pi,ci) in enumerate([(120,121,123,125),(136,137,148,150)]):
    adp=max(z[di],0);prod=rate[j]*2*adp/(origin[di]+adp);d[ai]=prod-spent[j];d[di]=-d[ai];d[pi]=prod;d[ci]=spent[j]
   return d[idx]
  l,r=lo*3600,hi*3600;sol=solve_ivp(rhs,(l,r),state,t_eval=np.r_[np.arange(l+300,r,300),r],method='LSODA',rtol=2e-10 if strict else 1e-8,atol=2e-12 if strict else 1e-10,max_step=30 if strict else 180)
  assert sol.success;state=sol.y[:,-1];times.extend(sol.t);states.extend(sol.y.T)
 t=np.array(times);z=np.tile(base,(len(t),1));z[:,idx]=states;checks=ch(z,base)
 for ai,di,pi,ci,n in [(120,121,123,125,'soma'),(136,137,148,150,'process')]:
  checks[n+'_ATP']=float(abs(z[:,ai]-base[ai]-z[:,pi]+z[:,ci]).max());checks[n+'_adenylate']=float(abs(z[:,ai]+z[:,di]-base[ai]-base[di]).max())
 assert max(checks.values())<3e-5 and z[:,idx].min()>-1e-7,checks
 row=dict(parameters=asdict(par or RibosomeMaintenanceParameters()),repair=repair,retire=retire,synthesis=synthesis,checks=checks,protein24h_relative=float(z[-1,583]/base[583]),ATP_min_uM=float(z[:,136].min()),RPS26_end_uM=float(z[-1,643]),exchanges_uM=float(z[-1,651]*C*4),AA_returned_uM=float(z[-1,656]),retired_Ceq=z[-1,652:654],impaired_end_Ceq=z[-1,[640,673]],extra_ATP_uM=float(sum(z[-1,663:668])+z[-1,672]))
 return t,z,row

def scenarios():
 rows=[];chosen={}
 for damage in [0.,1e-5,1e-4]:
  for repair,retire,synthesis in [(0.,0.,0.),(1.,0.,0.),(1.,0.,1.),(0.,1.,0.),(1.,1.,1.)]:
   t,z,r=molecular(replace(RibosomeMaintenanceParameters(),damage_per_s=damage),repair,retire,synthesis);rows.append(r)
   if damage==1e-5:chosen[str((repair,retire,synthesis))]=(t,z)
   save('scenarios_codex.json',dict(completed=False,scope='Coupled molecular subsystem;not full-neuron24h',rows=rows));print('scenario',damage,repair,retire,synthesis,r['protein24h_relative'],flush=True)
 t,strict,_=molecular(strict=True);err=float(np.max(abs(strict-chosen['(1.0, 1.0, 1.0)'][1])/(1+abs(strict))));assert err<3e-5
 save('scenarios_codex.json',dict(completed=True,runs=len(rows),strict_error=err,scope='Coupled molecular subsystem;not full-neuron24h',rows=rows))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(11,7))
 for name,(t,z) in chosen.items():
  for a,y in zip(axs.flat,[z[:,583]/z[0,583],z[:,640],z[:,643],z[:,656]]):a.plot(t/3600,y,label=name)
 for a,title in zip(axs.flat,['GAP43 / initial','Impaired40S (Ceq)','Spare RPS26 (uM)','AA exported from lysosome (uM)']):a.set_title(title);a.set_xlabel('Hours');a.grid(alpha=.2)
 axs[0,0].legend(title='Repair, retirement, synthesis',fontsize=7);fig.tight_layout();fig.savefig(B/'maintenance_codex.png',dpi=140);plt.close(fig)

def full():
 run=api.run_ribosome_maintenance_neuron
 sg=m.schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)]);es=np.c_[sg[:,:2],[0,25,0],[0,0,0]];ctl=controls(sg);ctl[1:,2:]=1
 opts=dict(**m.m.hist.options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
 r=run(sg,ribosome_maintenance_schedule=ctl,**opts);assert r['state'].shape[1]==len(r['columns'])==675 and r['state'][-1,650]>0
 a=run(sg[:2],ribosome_maintenance_schedule=ctl[:2],**dict(opts,electrical_schedule=es[:2]));b=run(sg[2:],ribosome_maintenance_schedule=ctl[2:],checkpoint=a['checkpoint'],**dict(opts,electrical_schedule=es[2:]))
 strict=run(sg,ribosome_maintenance_schedule=ctl,**dict(opts,rtol=3e-10,atol=3e-12,max_step_s=.00125));err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
 restart=err(b['state'][-1],r['state'][-1]);conv=err(strict['state'][-1],r['state'][-1]);assert max(restart,conv)<1e-5
 bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][640]+=.1
 try:run(sg[2:],checkpoint=bad,ribosome_maintenance_schedule=ctl[2:],**dict(opts,electrical_schedule=es[2:]));raise AssertionError('Corrupted checkpoint accepted')
 except ValueError:pass
 save('integrated_codex.json',dict(passed=True,states=675,restart_error=restart,strict_error=conv,checks=r['checks'],config=r['config']))
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical_codex.json').read_text())['rows'];metrics=m.module('maintenance_features',B.parent/'2026-10-26-nak-source-calibration/validate_model_codex.py');rows=[]
 for target in targets:
  p=target['profile'];cur=target['current_pA'];dur=.5 if p=='Burley' else 1.;sg=m.schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  r=run(sg,**m.m.hist.options(p),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  f=metrics.features(r,.2,.2+dur,p);assert f['spikes']==target['target'];delta=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.02
  rows.append(dict(profile=p,current_pA=cur,features=f,max_change=delta));save('electrical_codex.json',dict(complete=len(rows)==9,rows=rows));print('electrical',p,cur,delta,flush=True)

def long():
 run=api.run_ribosome_maintenance_neuron
 for profile in ['Burley','Selvaraj-Con1']:
  r=run(m.schedule([(0,86400,0,1,0)]),**m.m.hist.options(profile),output_step_s=300,max_step_s=30,rtol=1e-8,atol=1e-10)
  z=r['state'];b=r['baseline_state'];labels=[j for j,n in enumerate(r['columns']) if n.startswith(('labelled_','ferritin_label_'))];idx=np.setdiff1d(np.flatnonzero(abs(b)>1e-12),labels);drift=float(np.max(abs(z[:,idx]/b[idx]-1)));assert drift<.02
  save('rest_'+profile+'_codex.json',dict(completed=True,states=675,duration_s=86400,maximum_relative_drift=drift,checks=r['checks'],nfev=r['nfev']));print('rest',profile,drift,flush=True)
 sg=m.schedule([(0,7200,0,1,0),(7200,28800,0,1,0),(28800,43200,0,1,0),(43200,64800,0,1,0),(64800,86400,0,1,0)])
 for label,repair,retire,syn in [('disabled',0.,0.,0.),('exchange',1.,0.,1.),('both',1.,1.,1.)]:
  ctl=controls(sg);ctl[[1,3],2]=1;ctl[:,3]=repair;ctl[1:,4]=retire;ctl[1:,7]=syn;ctl[1:,8]=1
  r=run(sg,ribosome_maintenance_schedule=ctl,**m.m.hist.options(),output_step_s=300,max_step_s=30,rtol=1e-8,atol=1e-10);z=r['state']
  save('full_stress_'+label+'_codex.json',dict(completed=True,full_neuron=True,label=label,states=675,checks=r['checks'],GAP43_relative=float(z[-1,583]/z[0,583]),ATP_min=float(z[:,136].min()),exchanges_Ceq=float(z[-1,651]),AA_returned_uM=float(z[-1,656]),nfev=r['nfev']));np.savez_compressed(B/('full_stress_'+label+'_codex.npz'),time_s=r['time_s'],state=z)
  print('full stress',label,z[-1,583]/z[0,583],flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['unit','scenarios','full','long']);globals()[a.parse_args().mode]()
