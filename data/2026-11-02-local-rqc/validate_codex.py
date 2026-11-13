"""[codex]2026-11-02. Local RQC material/energy, repeated stress and regression."""
from pathlib import Path
from dataclasses import replace,asdict
import sys,importlib.util,json,argparse
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
threads=threadpool_limits(1);B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
spec=importlib.util.spec_from_file_location('rqc_prior',B.parent/'2026-11-02-gap43-regrowth/validate_codex.py');g=importlib.util.module_from_spec(spec);sys.modules[spec.name]=g;spec.loader.exec_module(g)
m=g.m;api=m.api
from local_rqc_codex import *

def save(n,x):(B/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a))+'\n',encoding='utf8')

def setup(par=None):
 cs,z,v,rv,C,K,w,ref,comp,old=m.setup();gv=g.GAP43RegrowthParameters().vector();z,gr=g.initialize(z,gv)
 z=np.r_[z,np.zeros(20)];rq=(par or LocalRQCParameters()).vector()
 def component(q,ctl):
  d=comp(q,np.ones(4));d=g.extension(q,d,gv,np.array([0.,1.,1.]),gr)
  return extension(q,d,rq,ctl,v[0],C,4.,K,v[14])
 def checks(q,b):
  # Existing particle checks see bound40S/60S as unavailable-but-conserved.
  qq,bb=adjusted_particles(q,b)
  ch=m.allchecks(qq,bb,cs,v,rv,w,C,4.)
  return revise_checks(ch,q,b,rq,v[0],C,4.)
 return z,component,checks,v,C,K,rq

def unit():
 z,comp,ch,v,C,K,rq=setup();ctl=np.array([1.,1.,1.,1.]);old=g.extension(z,np.zeros(617),g.GAP43RegrowthParameters().vector(),np.array([0.,1.,1.]),np.array([z[583],z[136]]))
 # Independent manufactured states: equal cohorts at each quarter-length.
 y=z.copy();y[620:624]=[0.,.1,.2,.3];y[624:628]=[0.,.3,.2,.1];y[628]=.7
 for gp,reuse in [(0.,1.),(1.,0.),(1.,.5)]:
  pr=replace(LocalRQCParameters(),rescue_GTP_per_chain=gp,reusable_message_fraction=reuse).vector()
  d=extension(y,old,pr,ctl,v[0],C,4.,K,v[14]);scale=C*4
  assert abs(d[564]-d[630])<1e-14 and abs(d[565]-d[631])<1e-14
  assert d[583]==0 and d[574]==0 #no instant mature protein/AA
  assert abs(d[150]-(d[634]+d[636]+d[637]))<1e-14
  assert d[136]==-d[150] and d[137]==d[150] and d[576]==-d[635] and d[577]==d[635]
  assert abs(d[607]+d[639]-d[630]*scale)<1e-14
  nas_indices=np.r_[np.arange(578,582),np.arange(620,624),np.arange(624,628)]
  weighted=d[nas_indices]@np.tile(np.arange(4)*v[0]/4*scale,3)
  assert abs(weighted+d[628]+d[584])<1e-14
  zero=y.copy();zero[136]=0;dz=extension(zero,old,pr,ctl,v[0],C,4.,K,v[14])
  assert not dz[[630,631,632,633,634,635,636,637]].any()
  zero=y.copy();zero[576]=0;dz=extension(zero,old,pr,ctl,v[0],C,4.,K,v[14])
  assert dz[630]==0 if gp else dz[630]>0
 d=comp(z,ctl);errors=ch(np.array([z,z+1e-6*d]),z);assert max(errors.values())<1e-8,errors
 save('unit_codex.json',dict(passed=True,checks=errors,partial_chain_mass_conserved=True,zero_ATP_stops_processing=True,GTP_route_distinguished=True,no_ATP_refund=True))
 print('unit passed',flush=True)

def full():
 sg=m.schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)]);es=np.c_[sg[:,:2],[0,25,0],[0,0,0]]
 rc=controls(sg);rc[1,2]=1
 opts=dict(**m.m.hist.options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
 run=api.run_local_rqc_neuron;r=run(sg,local_rqc_schedule=rc,**opts)
 assert r['state'].shape[1]==len(r['columns'])==640 and r['state'][-1,629]>0
 a=run(sg[:2],local_rqc_schedule=rc[:2],**dict(opts,electrical_schedule=es[:2]));b=run(sg[2:],local_rqc_schedule=rc[2:],checkpoint=a['checkpoint'],**dict(opts,electrical_schedule=es[2:]))
 strict=run(sg,local_rqc_schedule=rc,**dict(opts,rtol=3e-10,atol=3e-12,max_step_s=.00125))
 err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
 restart=err(b['state'][-1],r['state'][-1]);conv=err(strict['state'][-1],r['state'][-1]);assert max(restart,conv)<1e-5
 bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][620]+=.1
 try:run(sg[2:],checkpoint=bad,local_rqc_schedule=rc[2:],**dict(opts,electrical_schedule=es[2:]));raise AssertionError('Bad RQC checkpoint accepted')
 except ValueError:pass
 save('integrated_codex.json',dict(passed=True,states=640,restart_error=restart,strict_error=conv,checks=r['checks'],config=r['config']))
 np.savez_compressed(B/'integrated_codex.npz',time_s=r['time_s'],state=r['state'])
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical_codex.json').read_text())['rows'];metrics=m.module('rqc_features',B.parent/'2026-10-26-nak-source-calibration/validate_model_codex.py');rows=[]
 for target in targets:
  p=target['profile'];cur=target['current_pA'];dur=.5 if p=='Burley' else 1.
  sg=m.schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  r=run(sg,**m.m.hist.options(p),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  f=metrics.features(r,.2,.2+dur,p);assert f['spikes']==target['target']
  delta=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.02
  rows.append(dict(profile=p,current_pA=cur,features=f,max_change=delta));save('electrical_codex.json',dict(complete=len(rows)==9,rows=rows));print('electrical',p,cur,delta,flush=True)

def molecular(par=None,rescue=1.,strict=False):
 base,component,check,v,C,K,rq=setup(par);origin=base.copy()
 idx=np.unique(np.r_[m.m.fusco.INDEX,[136,137,148,150],np.arange(562,640)])
 def cost(z,d):
  bg=.5*max(z[136],0)/(K+max(z[136],0))*(K+base[136])/base[136]
  soma=m.m.fusco.demand(d)+d[572]+d[614]+d[615]
  process=bg+d[573]+d[594]+d[595]+d[598]+d[592]-d[593]+d[616]+d[634]+d[636]+d[637]
  return np.array([soma,process]) #optional rescue GTP replenishment already in NDPK
 d0=component(base,np.array([0.,1.,1.,1.]));rate=cost(base,d0)
 state=base[idx].copy();times=[0.];states=[state.copy()]
 for l,r in [(0,2),(2,8),(8,12),(12,18),(18,24)]:
  ctl=np.array([1. if l in (2,12) else 0.,rescue,1.,1.])
  def rhs(t,y):
   z=base.copy();z[idx]=y;d=component(z,ctl);spent=cost(z,d)
   for j,(ai,di,pi,ci) in enumerate([(120,121,123,125),(136,137,148,150)]):
    adp=max(z[di],0);prod=rate[j]*2*adp/(origin[di]+adp)
    d[ai]=prod-spent[j];d[di]=spent[j]-prod;d[pi]=prod;d[ci]=spent[j]
   return d[idx]
  lo,hi=l*3600,r*3600;grid=np.r_[np.arange(lo+120,hi,120),hi]
  sol=solve_ivp(rhs,(lo,hi),state,t_eval=grid,method='LSODA',rtol=2e-10 if strict else 1e-8,atol=2e-12 if strict else 1e-10,max_step=30 if strict else 180)
  assert sol.success;state=sol.y[:,-1];times.extend(sol.t);states.extend(sol.y.T)
 t=np.array(times);z=np.tile(base,(len(t),1));z[:,idx]=states;ch=check(z,base)
 for ai,di,pi,ci,n in [(120,121,123,125,'soma'),(136,137,148,150,'process')]:
  ch[n+'_ATP']=float(abs(z[:,ai]-base[ai]-z[:,pi]+z[:,ci]).max());ch[n+'_adenylate']=float(abs(z[:,ai]+z[:,di]-base[ai]-base[di]).max())
 assert max(ch.values())<3e-5 and z[:,idx].min()>-1e-7,ch
 end=z[-1];out=dict(parameters=asdict(par or LocalRQCParameters()),split_support=rescue,checks=ch,
  protein24h_relative=float(end[583]/base[583]),ATP_min_uM=float(z[:,136].min()),
  stalled_peak_Ceq=float(z[:,620:624].sum(1).max()),stalled24h_Ceq=float(end[620:624].sum()),
  AA_proteolyzed_uM=float(end[633]),extra_direct_ATP_uM=float(end[634]+end[636]+end[637]),
  GTP_rescue_uM=float(end[635]),completion24h_relative=float(component(end,np.array([0.,rescue,1.,1.]))[586]/d0[586]))
 return t,z,out

def scenarios():
 rows=[];chosen={}
 for hazard in [.0001,.001,.01]:
  for tau in [30.,300.,1800.]:
   for support in [0.,.1,1.]:
    p=replace(LocalRQCParameters(),stall_per_s=hazard,split_s=tau)
    t,z,r=molecular(p,support);rows.append(r)
    if hazard==.001 and tau==300:chosen[str(support)]=(t,z)
    print('scenario',hazard,tau,support,r['protein24h_relative'],flush=True)
 for p in [replace(LocalRQCParameters(),stall_per_s=0),replace(LocalRQCParameters(),rescue_GTP_per_chain=1.,reusable_message_fraction=0.)]:
  t,z,r=molecular(p);rows.append(r)
 t,strict,_=molecular(strict=True);err=float(np.max(abs(strict-chosen['1.0'][1])/(1+abs(strict))));assert err<3e-5
 save('scenarios_codex.json',dict(complete=True,runs=len(rows),strict_error=err,protocol='2-8h and12-18h eligible stalls;state never reset',scope='Coupled molecular subsystem with dynamicATP/ADP feedback;not full-neuron24h',rows=rows))
 for name,(t,z) in chosen.items():np.savez_compressed(B/f'repeated_{name}_codex.npz',time_s=t,state=z)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(10,6))
 for name,(t,z) in chosen.items():
  for a,y in zip(axs.flat,[z[:,620:624].sum(1),z[:,583]/z[0,583],z[:,633],z[:,136]]):a.plot(t/3600,y,label=name)
 for a,title in zip(axs.flat,['Stalled ribosomes (Ceq)','GAP43 / initial','Aborted-chain AA sent to hydrolysis (uM)','Process ATP (uM)']):
  a.set_title(title);a.set_xlabel('Hours');a.grid(alpha=.2)
  for lo,hi in [(2,8),(12,18)]:a.axvspan(lo,hi,color='grey',alpha=.12)
 axs[0,0].legend(title='Rescue support');fig.tight_layout();fig.savefig(B/'local_rqc_codex.png',dpi=140);plt.close(fig)

def long():
 run=api.run_local_rqc_neuron
 for profile in ['Burley','Selvaraj-Con1']:
  r=run(m.schedule([(0,86400,0,1,0)]),**m.m.hist.options(profile),output_step_s=300,max_step_s=30,rtol=1e-8,atol=1e-10)
  z=r['state'];b=r['baseline_state'];labels=[j for j,n in enumerate(r['columns']) if n.startswith(('labelled_','ferritin_label_'))]
  idx=np.setdiff1d(np.flatnonzero(abs(b)>1e-12),labels);drift=float(np.max(abs(z[:,idx]/b[idx]-1)))
  assert drift<.02 and not z[:,620:].any() and not r['voltage_crossing_times_s']['soma']
  save('rest_'+profile+'_codex.json',dict(completed=True,duration_s=86400,physical_states=640,maximum_relative_drift=drift,checks=r['checks'],nfev=r['nfev']))
  cur=25 if profile=='Burley' else 15;dur=.5 if profile=='Burley' else 1.;t=86400.
  sg=m.schedule([(t,t+.2,0,1,0),(t+.2,t+.2+dur,0,1,0),(t+.2+dur,t+1.2+dur,0,1,0)])
  rr=run(sg,checkpoint=r['checkpoint'],**m.m.hist.options(profile),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  metrics=m.module('rqc_postday_'+profile,B.parent/'2026-10-26-nak-source-calibration/validate_model_codex.py');f=metrics.features(rr,t+.2,t+.2+dur,profile)
  targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical_codex.json').read_text())['rows'];target=next(a for a in targets if a['profile']==profile and a['current_pA']==cur)
  assert f['spikes']==target['target'];save('postday_'+profile+'_codex.json',dict(passed=True,same_state_no_reset=True,features=f))
  print('day',profile,drift,flush=True)
 sg=m.schedule([(0,7200,0,1,0),(7200,28800,0,1,0),(28800,43200,0,1,0),(43200,64800,0,1,0),(64800,86400,0,1,0)])
 rc=controls(sg);rc[[1,3],2]=1
 for support in [1.,0.]:
  rc[:,3]=support
  r=run(sg,local_rqc_schedule=rc,**m.m.hist.options(),output_step_s=300,max_step_s=30,rtol=1e-8,atol=1e-10)
  z=r['state'];save('full_stress_'+str(support)+'_codex.json',dict(completed=True,split_support=support,full_neuron=True,states=640,checks=r['checks'],protein24h_relative=float(z[-1,583]/z[0,583]),ATP_min_uM=float(z[:,136].min()),stalled24h_Ceq=float(z[-1,620:624].sum()),AA_proteolyzed_uM=float(z[-1,633]),extra_direct_ATP_uM=float(z[-1,634]+z[-1,636]+z[-1,637]),nfev=r['nfev']))
  np.savez_compressed(B/('full_stress_'+str(support)+'_codex.npz'),time_s=r['time_s'],state=z)
  print('full stress',support,z[-1,583]/z[0,583],flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['unit','full','scenarios','long']);args=a.parse_args();globals()[args.mode]()
