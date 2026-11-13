"""[codex]2026-11-02. Stable molecular baseline, transport recovery,617 integration.
Long assay is the selected molecular subsystem, not a24h electrical simulation.
"""
from pathlib import Path
import importlib.util,sys,json,argparse
from dataclasses import replace,asdict
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
_threads=threadpool_limits(1);B=Path(__file__).resolve().parent
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=module('mrna_previous_validation',B.parent/'2026-11-02-local-translation/validate_codex.py');api=m.api
from local_mrna_codex import *
from connected_parameters_codex import schedule
def save(n,x):(B/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a))+'\n',encoding='utf8')

def setup(par=None,translation=None,ratio=4.):
 cs,z,v,rv,C,K=m.constants(translation or cargo_parameters());w=(par or LocalMRNAParameters()).vector()
 old=z.copy();z,v,ref=initialize(z,v,rv,cs[0][5],w,ratio,K)
 def component(z,mc,tc=None):
  d=m.ribo_extension(z,m.fusco.component(z[:562],cs,np.ones(2)),rv,np.ones(3),C,ratio,ref[1],ref[2],K)
  return extension(z,d,v,np.ones(6) if tc is None else tc,w,mc,ref,C,ratio,K)
 return cs,z,v,rv,C,K,w,ref,component,old

def allchecks(z,b,cs,v,rv,w,C,ratio):
 ch=m.all_checks(z,b,cs,v,rv,C,ratio);ch.update(checks(z,b,w,C,ratio));return ch

def unit():
 cs,z,v,rv,C,K,w,ref,component,old=setup();d=component(z,np.ones(4))
 original=m.fusco.component(old[:562],cs,np.ones(2))
 idx=np.r_[np.setdiff1d(np.flatnonzero(abs(original)<1e-13),[120,121,123,125,136,137,148,150]),np.arange(562,566),np.arange(574,585),np.arange(605,608)]
 residual=float(abs(d[idx]).max());assert residual<1e-12,(residual,[(int(i),d[i]) for i in idx if abs(d[i])>1e-12])
 ch=allchecks(np.array([z,z+1e-6*d]),z,cs,v,rv,w,C,4.);assert max(ch.values())<1e-9,ch
 for j in [605,606,607]:
  q=z.copy();q[j]=0;assert component(q,np.ones(4))[j]>=-1e-15
 q=z.copy();q[607]=0;assert component(q,np.ones(4))[585]==0
 q=z.copy();q[578:582]=0;assert component(q,np.ones(4))[586]==0
 # Engaged message is conserved through run-off; no future synthesis is free.
 no_new=np.ones(6);no_new[0]=0
 dd=component(z,np.ones(4),no_new);assert dd[585]==0 and dd[586]>0
 assert len(z)==617 and abs(d[614]-w[6]*d[608])<1e-14
 for bad in [replace(LocalMRNAParameters(),local_half_h=0),replace(LocalMRNAParameters(),arrival_s=-1)]:
  try:bad.vector();raise AssertionError('Bad parameter accepted')
  except ValueError:pass
 save('unit_codex.json',dict(passed=True,stationary_molecular_derivative=residual,checked_indices=idx,
  baseline_state=z,effective_local_translation_vector=v,mRNA_reference=ref,
  note='Excludes cumulative counters and ATP supply, which is balanced separately in long assay; not full-neuron steady state'))
 print('617 unit passed, residual',residual,flush=True)

def full():
 sg=schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)])
 es=np.c_[sg[:,:2],[0,25,0],[0,0,0]];mc=np.c_[sg[:,:2],np.ones((3,4))];mc[1,3]=0
 opts=dict(**m.hist.options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
 run=api.run_local_mrna_neuron;r=run(sg,local_mrna_schedule=mc,**opts)
 assert r['state'].shape[1]==len(r['columns'])==617 and r['columns'][583]=='local_GAP43_protein_uM'
 a=run(sg[:2],local_mrna_schedule=mc[:2],**dict(opts,electrical_schedule=es[:2]));c=run(sg[2:],checkpoint=a['checkpoint'],local_mrna_schedule=mc[2:],**dict(opts,electrical_schedule=es[2:]))
 strict=run(sg,local_mrna_schedule=mc,**dict(opts,rtol=3e-10,atol=3e-12,max_step_s=.00125))
 err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
 restart=err(c['state'][-1],r['state'][-1]);conv=err(strict['state'][-1],r['state'][-1]);assert max(restart,conv)<1e-5
 bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][607]+=.1
 try:run(sg[2:],checkpoint=bad,local_mrna_schedule=mc[2:],**dict(opts,electrical_schedule=es[2:]));raise AssertionError('Corrupt RNA accepted')
 except ValueError:pass
 save('integrated_codex.json',dict(passed=True,states=617,restart_error=restart,strict_error=conv,checks=r['checks'],config=r['config']))
 print('617 integrated passed',flush=True)
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical_codex.json').read_text())['rows']
 metrics=module('mrna_features',B.parent/'2026-10-26-nak-source-calibration/validate_model_codex.py');rows=[]
 for target in targets:
  p=target['profile'];cur=target['current_pA'];dur=.5 if p=='Burley' else 1.
  sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  opts=dict(**m.hist.options(p),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  r=run(sg,**opts);f=metrics.features(r,.2,.2+dur,p);assert f['spikes']==target['target']
  delta=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.02
  rows.append(dict(profile=p,current_pA=cur,features=f,max_change=delta))
  save('electrical_codex.json',dict(complete=len(rows)==9,rows=rows));print(p,cur,delta,flush=True)

ASSAY_RANGES={'process_background_ATP_uM_s':[.1,.5,1.], 'ADP_supply_reserve':[1.5,2.,3.], 'volume_ratio':[1.,4.,16.]}

def molecular(half_h=6.,support=1.,mode='departure',strict=False,perturb=False):
 par=replace(LocalMRNAParameters(),local_half_h=half_h)
 cs,base,v,rv,C,K,w,ref,component,_=setup(par);ratio=4.
 idx=np.unique(np.r_[m.fusco.INDEX,[136,137,148,150],np.arange(562,617)])
 def costs(z,d):
  bg=.5*max(z[136],0)/(K+max(z[136],0))*(K+base[136])/base[136]
  return m.fusco.demand(d)+d[572]+d[614]+d[615],bg+d[573]+d[594]+d[595]+d[598]+d[592]-d[593]+d[616]
 d0=component(base,np.ones(4));rate=np.array(costs(base,d0));origin=base.copy()
 if perturb:
  # Ten percent free local RNA perturbation, conserving RNA amount in soma.
  delta=.1*base[607];base[607]+=delta;base[605]-=delta/ratio
  if base[605]<0:raise ValueError('Invalid perturbation')
 state=base[idx].copy();tt=[0.];yy=[state.copy()];dd=[d0]
 for l,r in zip([0,2,8,12,18],[2,8,12,18,24]):
  mc=np.ones(4);sup=1.
  if l in (2,12):
   if mode=='departure':mc[1]=support
   elif mode=='production':mc[0]=support
   elif mode=='ATP':sup=support
  def rhs(t,y):
   z=base.copy();z[idx]=y;d=component(z,mc);cost=costs(z,d)
   for j,(ai,di,pi,ci) in enumerate([(120,121,123,125),(136,137,148,150)]):
    adp=max(z[di],0);prod=rate[j]*2*adp/(origin[di]+adp)*(sup if j else 1.)
    d[ai]=prod-cost[j];d[di]=cost[j]-prod;d[pi]=prod;d[ci]=cost[j]
   return d[idx]
  start=l*3600;end=r*3600;grid=np.r_[np.arange(start+120,end,120),end]
  sol=solve_ivp(rhs,(start,end),state,method='LSODA',t_eval=grid,rtol=2e-10 if strict else 1e-8,atol=2e-12 if strict else 1e-10,max_step=30 if strict else 180)
  assert sol.success;state=sol.y[:,-1];tt.extend(sol.t);yy.extend(sol.y.T)
  for y in sol.y.T:
   z=base.copy();z[idx]=y;dd.append(component(z,mc))
 t=np.array(tt);z=np.tile(base,(len(t),1));z[:,idx]=np.array(yy);d=np.array(dd)
 ch=allchecks(z,base,cs,v,rv,w,C,ratio)
 for ai,di,pi,ci,label in [(120,121,123,125,'soma'),(136,137,148,150,'process')]:
  ch[label+'_ATP_balance']=float(abs(z[:,ai]-base[ai]-z[:,pi]+z[:,ci]).max())
  ch[label+'_adenylate_balance']=float(abs(z[:,ai]+z[:,di]-base[ai]-base[di]).max())
 assert max(ch.values())<3e-5 and z[:,idx].min()>-1e-7,ch
 phys=np.r_[120,121,136,137,330,331,390,391,np.arange(497,501),np.arange(562,566),np.arange(574,585),np.arange(605,608)]
 drift=float((abs(z[:,phys]-origin[phys])/np.maximum(abs(origin[phys]),1e-8)).max())
 at=lambda h:int(np.argmin(abs(t-h*3600)))
 out=dict(local_half_h=half_h,support=support,mode=mode,perturb=perturb,checks=ch,reference_source_uM_s=rate,
  initial_protein_uM=float(base[583]),protein24h_fraction=float(z[-1,583]/origin[583]),
  synthesis8h_fraction=float(d[at(8),586]/d0[586]),synthesis24h_fraction=float(d[-1,586]/d0[586]),
  protein_min_fraction=float(z[:,583].min()/origin[583]),ATP_min_uM=float(z[:,136].min()),
  molecular_drift_relative=drift,freeRNA_final_deviation=float(abs(z[-1,607]/origin[607]-1)),
  effective_NDPK_rate=float(v[17]),local_free_initial_uM=float(base[607]),bound_initial_uM=float(base[578:582].sum()*C*ratio))
 return t,z,d,out

def scenarios():
 rows=[];chosen={}
 for h in [1.,6.,24.]:
  for support in [0.,.25,1.]:
   t,z,d,r=molecular(h,support);rows.append(r);print(h,support,r['synthesis8h_fraction'],flush=True)
   if support==1:assert r['molecular_drift_relative']<1e-5,r
   if h==6:
    chosen[str(support)]=(t,z,d);np.savez_compressed(B/f'mRNA_departure_{support}_codex.npz',time_s=t,state=z,rate=d[:,586]*.4)
 for mode in ['production','ATP']:
  t,z,d,r=molecular(6,.25,mode);rows.append(r);chosen[mode]=(t,z,d)
 t,z,d,r=molecular(6,1,perturb=True);rows.append(r);assert r['freeRNA_final_deviation']<.1,r
 t,z,d,r=molecular(6,0,strict=True);err=float(np.max(abs(z-chosen['0.0'][1])/(1+abs(z))));assert err<3e-5
 save('scenarios_codex.json',dict(complete=True,runs=len(rows),strict_error=err,protocol='2-8h and12-18h, no reset',
  scope='Selected soma/process molecular subsystem; dynamic ATP with explicit demand-matched baseline and assumed ADP feedback, not full electrical24h or human kinetic calibration',rows=rows))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(10,6))
 for name,(t,z,d) in chosen.items():
  for a,y in zip(axs.flat,[z[:,607]/z[0,607],d[:,586]/d[0,586],z[:,583]/z[0,583],z[:,136]]):a.plot(t/3600,y,label=name)
 for a,title in zip(axs.flat,['Free local GAP43 mRNA / baseline','GAP43 completion / baseline','Local GAP43 protein / baseline','Process ATP (uM)']):
  a.set_title(title);a.set_xlabel('Time (h)');a.grid(alpha=.2)
  for l,r in [(2,8),(12,18)]:a.axvspan(l,r,color='grey',alpha=.12)
 axs[0,0].legend(title='Departure support / other intervention',fontsize=7);fig.suptitle('Conditional mRNA delivery and local production; human rates unknown');fig.tight_layout();fig.savefig(B/'local_mrna_codex.png',dpi=140);plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['unit','full','scenarios','all']);a=p.parse_args()
 if a.mode=='all':unit();scenarios();full()
 else:globals()[a.mode]()
