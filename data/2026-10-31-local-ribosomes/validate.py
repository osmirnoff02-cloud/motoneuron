"""[codex]2026-10-31. Spatial lineage, energy, restart and electrical checks."""
from pathlib import Path
import sys,json,importlib.util,argparse
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import replace
from threadpoolctl import threadpool_limits
_threads=threadpool_limits(1);B=Path(__file__).resolve().parent
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
sys.path.insert(0,str(B.parent/'2026-10-30-fusco-supplement'))
prior=module('local_prior',B.parent/'2026-10-30-fusco-supplement/validate.py');api=prior.api
sys.path.insert(0,str(B))
from local_ribosomes import LocalRibosomeParameters,extension,adjusted,checks,controls
from connected_parameters import schedule
def save(name,x):(B/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda q:q.tolist() if hasattr(q,'tolist') else str(q)),encoding='utf8')
options=prior.prior.hist.prior.prior.hist.v29.options
def unit():
 cs=prior.prior.constants();z=np.r_[cs[-1],np.zeros(12)];v=LocalRibosomeParameters().vector();d=prior.component(z[:562],cs,np.ones(2));C=cs[0][5][0]
 out=extension(z,d,v,np.ones(3),C,4.,2700.,2700.,100.)
 assert np.max(abs(out[390:392]-d[390:392]+out[562:564]))<1e-16
 assert abs((out[120]-d[120])+out[572])<1e-16
 assert abs((out[121]-d[121])-out[572])<1e-16
 off=extension(z,d,v,np.array([0.,1.,1.]),C,4.,2700.,2700.,100.)
 assert np.array_equal(off[:562],d) and np.all(off[562:]==0)
 z[562:566]=[.1,.2,.3,.4];out=extension(z,np.zeros(562),v,np.ones(3),C,4.,2700.,2700.,100.)
 for j in range(2):assert abs(out[390+j]+out[330+j]+out[562+j]+out[564+j])<1e-16
 zz=z.copy();zz[136]=0;low=extension(zz,np.zeros(562),v,np.ones(3),C,4.,2700.,2700.,100.);assert np.all(low[568:572]==0)
 assert max(checks(z[None,:],z,v,C,4.).values())<1e-12
 for p in [replace(LocalRibosomeParameters(),delivery_time_s=0),replace(LocalRibosomeParameters(),ATP_per_return=-1)]:
  try:p.vector();raise AssertionError('Invalid parameter accepted')
  except ValueError:pass
 save('unit.json',dict(passed=True,off_exact=True,lineage=True,ATP_gate=True,ATP_charged_once=True))
 print('Unit passed',flush=True)

def full():
 sg=schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)])
 es=np.c_[sg[:,:2],[0,25,0],[0,0,0]];ctl=np.c_[sg[:,:2],[[1,1,1],[1,.25,0],[1,1,1]]]
 opts=dict(**options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
 run=api.run_local_ribosome_neuron;r=run(sg,local_ribosome_schedule=ctl,**opts)
 assert r['state'].shape[1]==len(r['columns'])==574
 a=run(sg[:2],local_ribosome_schedule=ctl[:2],**dict(opts,electrical_schedule=es[:2]))
 c=run(sg[2:],local_ribosome_schedule=ctl[2:],checkpoint=a['checkpoint'],**dict(opts,electrical_schedule=es[2:]))
 strict=run(sg,local_ribosome_schedule=ctl,**dict(opts,rtol=3e-10,atol=3e-12,max_step_s=.00125))
 err=lambda a,b:float(np.max(abs(a-b)/(1+abs(b))))
 er=err(c['state'][-1],r['state'][-1]);ec=err(strict['state'][-1],r['state'][-1]);assert max(er,ec)<1e-5
 save('integrated.json',dict(passed=True,states=574,restart_error=er,strict_error=ec,checks=r['checks'],config=r['config']))
 bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][562]+=.1
 try:run(sg[2:],local_ribosome_schedule=ctl[2:],checkpoint=bad,**dict(opts,electrical_schedule=es[2:]));raise AssertionError('Corrupt particle accepted')
 except ValueError:pass
 print('Integrated574 states passed',flush=True)
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical.json').read_text())['rows']
 metrics=module('local_features',B.parent/'2026-10-26-nak-source-calibration/validate_model.py');rows=[]
 for target in targets:
  p=target['profile'];cur=target['current_pA'];dur=.5 if p=='Burley' else 1.
  sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  opts=dict(**options(p),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  r=run(sg,**opts);f=metrics.features(r,.2,.2+dur,p);assert f['spikes']==target['target']
  delta=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.02
  rows.append(dict(profile=p,current_pA=cur,features=f,max_change=delta));save('electrical.json',dict(complete=len(rows)==9,rows=rows));print(p,cur,delta,flush=True)

def molecular(tau,support,strict=False):
 cs=prior.prior.constants();base=np.r_[cs[-1],np.zeros(12)];ctx=cs[0];C=ctx[5][0];base[136]=base[120] # isolated assay: same clamped ATP in both compartments
 # Fixed ATP availability for this isolated trafficking experiment; ledgers
 # quantify its demand. Integrated electrical test above uses dynamic ATP.
 v=replace(LocalRibosomeParameters(),delivery_time_s=tau).vector();idx=np.unique(np.r_[prior.INDEX,np.arange(562,574)])
 idx=idx[~np.isin(idx,[120,121,123,125])]
 def comp(z,ctl):return extension(z,prior.component(z[:562],cs,np.ones(2)),v,ctl,C,1.,base[120],base[136],ctx[6][8])
 state=base[idx].copy();tt=[0.];yy=[state.copy()];dd=[comp(base,np.ones(3))]
 for l,r in zip([0,2,8,12,18],[2,8,12,18,24]):
  ctl=np.array([1.,support,support]) if l in [2,12] else np.ones(3)
  def rhs(t,y):
   z=base.copy();z[idx]=y;return comp(z,ctl)[idx]
  start=l*3600;end=r*3600;grid=np.r_[np.arange(start+120,end,120),end]
  sol=solve_ivp(rhs,(start,end),state,method='LSODA',t_eval=grid,rtol=2e-10 if strict else 1e-8,atol=2e-12 if strict else 1e-10,max_step=30 if strict else 180);assert sol.success
  state=sol.y[:,-1];tt.extend(sol.t);yy.extend(sol.y.T)
  for y in sol.y.T:
   z=base.copy();z[idx]=y;dd.append(comp(z,ctl))
 t=np.array(tt);z=np.tile(base,(len(t),1));z[:,idx]=np.array(yy);d=np.array(dd)
 az,ab=adjusted(z,base);cs2=(*cs[:-1],ab[:562]);ch=prior.prior.check(az[:,:562],cs2);ch.update(checks(z,base,v,C,1.))
 assert max(ch.values())<3e-5 and z[:,idx].min()>-1e-7,ch
 g=(d[:,254]+d[:,328]+d[:,493])/(dd[0][254]+dd[0][328]+dd[0][493])
 out=dict(delivery_time_s=tau,local_maturation_support=support,checks=ch,local_pre_max_Ceq=z[:,562:564].max(0),local_mature_24h_Ceq=z[-1,564:566],matured_Ceq=z[-1,568:570],soma_synthesis_min=float(g.min()),soma_synthesis_24h=float(g[-1]),transport_ATP_soma_uM=float(z[-1,572]),return_ATP_equal_volume_uM=float(z[-1,573]))
 return t,z,g,out

def scenarios():
 rows=[];selected={}
 for tau in [1800.,7200.,28800.]:
  for support in [0.,.25,1.]:
   t,z,g,r=molecular(tau,support);rows.append(r);print(tau,support,r['soma_synthesis_min'],flush=True)
   if tau==7200:
    selected[support]=(t,z,g);np.savez_compressed(B/f'local_{support}_24h.npz',time_s=t,state=z,synthesis=g)
 t,z,g,r=molecular(7200.,0.,True);e=float(np.max(abs(z-selected[0.][1])/(1+abs(z))));assert e<3e-5
 save('scenarios.json',dict(complete=True,runs=9,strict_error=e,protocol='Local maturation modified2-8h and12-18h, no reset',scope='Selected molecular module, ATP clamped, equal-volume energy reporting, not full electrical24h',rows=rows))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(2,2,figsize=(10,6))
 for support,(t,z,g) in selected.items():
  for a,y in zip(ax.flat,[z[:,562:564].sum(1),z[:,564:566].sum(1),g,z[:,572]+z[:,573]]):a.plot(t/3600,y,label=f'local maturation {support:g}')
 for a,title in zip(ax.flat,['Local immature particles (Ceq)','Local mature reserve (Ceq)','Somatic synthesis / starting value','Transport ATP (uM, equal volumes)']):
  a.set_title(title);a.set_xlabel('Time (h)');a.grid(alpha=.2)
  for l,r in [(2,8),(12,18)]:a.axvspan(l,r,color='grey',alpha=.15)
 ax[0,0].legend(fontsize=8);fig.suptitle('Conditional transport and local maturation; human rates unknown');fig.tight_layout();fig.savefig(B/'local_recovery.png',dpi=120);plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['unit','full','scenarios'],required=True);a=p.parse_args();globals()[a.mode]()
