"""[codex]2026-10-30. Maturation invariants, continuation, electrical preservation.
Long runs are selected molecular subsystem, not whole electrical neuron24h.
"""
from pathlib import Path
import importlib.util,sys,json,argparse,numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
_threads=threadpool_limits(1);B=Path(__file__).resolve().parent
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
sys.path.insert(0,str(B.parent/'2026-10-30-isr-selective'))
prior=module('fusco_prior_validate',B.parent/'2026-10-30-isr-selective/validate.py');api=prior.api
from maturation import controls,apply_maturation,factor_evidence
from connected_parameters import schedule
def save(name,obj):
 (B/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf8')
def component(z,cs,ctl):return apply_maturation(prior.component(z,cs,np.array([0.,1.,1.])),ctl)
def unit():
 cs=prior.constants();z=cs[-1];d=prior.component(z,cs,np.array([0.,1.,1.]))
 assert np.array_equal(d,apply_maturation(d.copy(),np.ones(2)))
 for j in range(2):
  ctl=np.ones(2);ctl[j]=0.;x=apply_maturation(d.copy(),ctl)
  assert x[396+j]==0. and x[396+1-j]==d[396+1-j]
  assert abs((x[330+j]+x[390+j])-(d[330+j]+d[390+j]))<1e-14
  assert set(np.flatnonzero(x!=d)).issubset({330+j,390+j,396+j})
 sg=np.array([[0.,1.],[1.,2.]])
 for bad in [np.array([[0,1,-1,1],[1,2,1,1]]),np.array([[0,1,1,2],[1,2,1,1]]),np.array([[0,1,np.nan,1],[1,2,1,1]])]:
  try:controls(sg,bad);raise AssertionError('Accepted invalid control')
  except ValueError:pass
 ev=factor_evidence('Dimt1')['groups'];assert len(ev)==1 and ev[0]['n_detected']['lysate_neurites']==0 and ev[0]['n_detected']['cushion_neurites']==5 and ev[0]['relative_enrichment'] is None
 assert len(factor_evidence('Gtpbp4')['groups'])==2
 save('unit.json',dict(passed=True,default_derivative_exact=True,lineage_conserved=True,ATP_derivative_unchanged=True,missing_never_zero=True,duplicate_gene_groups_preserved=True))
 print('Unit and source boundaries passed',flush=True)

def integrated():
 sg=schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)])
 es=np.c_[sg[:,:2],[0,25,0],[0,0,0]];mc=np.c_[sg[:,:2],[[1,1],[.25,0],[1,1]]]
 opts=dict(**prior.hist.prior.prior.hist.v29.options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
 run=api.run_ribosome_maturation_neuron
 r=run(sg,maturation_schedule=mc,**opts);assert r['state'].shape[1]==562
 a=run(sg[:2],maturation_schedule=mc[:2],**dict(opts,electrical_schedule=es[:2]))
 c=run(sg[2:],maturation_schedule=mc[2:],checkpoint=a['checkpoint'],**dict(opts,electrical_schedule=es[2:]))
 strict=run(sg,maturation_schedule=mc,**dict(opts,rtol=3e-10,atol=3e-12,max_step_s=.00125))
 err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
 restart=err(c['state'][-1],r['state'][-1]);conv=err(strict['state'][-1],r['state'][-1]);assert max(restart,conv)<1e-5
 save('integrated.json',dict(passed=True,restart_error=restart,strict_error=conv,checks=r['checks'],states=562))
 np.savez_compressed(B/'integrated.npz',time_s=r['time_s'],state=r['state']);print('Integrated branch passed',flush=True)

def electrical():
 metrics=module('fusco_electric_metrics',B.parent/'2026-10-26-nak-source-calibration/validate_model.py')
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical.json').read_text())['rows'];rows=[]
 for target in targets:
  profile=target['profile'];cur=target['current_pA'];dur=.5 if profile=='Burley' else 1.
  sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  opts=dict(**prior.hist.prior.prior.hist.v29.options(profile),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  r=api.run_ribosome_maturation_neuron(sg,**opts);f=metrics.features(r,.2,.2+dur,profile);assert f['spikes']==target['target']
  delta=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.02
  rows.append(dict(profile=profile,current_pA=cur,features=f,max_change=delta));save('electrical.json',dict(complete=len(rows)==9,rows=rows));print(profile,cur,flush=True)

INDEX=np.unique(np.r_[prior.hist.prior.prior.INDEX,[120,121,123,125],np.arange(497,562)])
def demand(d):return sum(d[[254,328,493,340,400,401,494,495,559,560]])+d[509]+d[510]-d[511]-d[512]
def molecular(a,b,strict=False):
 cs=prior.constants();base=cs[-1];d0=component(base,cs,np.ones(2));source0=demand(d0)
 state=base[INDEX].copy();tt=[0.];yy=[state.copy()];dd=[d0]
 for l,r in zip([0,2,8,12,18],[2,8,12,18,24]):
  ctl=np.array([a,b]) if l in (2,12) else np.ones(2)
  def rhs(t,y):
   z=base.copy();z[INDEX]=y;d=component(z,cs,ctl);cost=demand(d);adp=max(z[121],0);source=source0*2*adp/(base[121]+adp)
   d[120]=source-cost;d[121]=cost-source;d[123]=source;d[125]=cost
   return d[INDEX]
  start=l*3600;end=r*3600;grid=np.r_[np.arange(start+120,end,120),end]
  sol=solve_ivp(rhs,(start,end),state,method='LSODA',t_eval=grid,rtol=2e-10 if strict else 1e-8,atol=2e-12 if strict else 1e-10,max_step=30 if strict else 180)
  assert sol.success;state=sol.y[:,-1];tt.extend(sol.t);yy.extend(sol.y.T)
  for y in sol.y.T:
   z=base.copy();z[INDEX]=y;dd.append(component(z,cs,ctl))
 t=np.array(tt);z=np.tile(base,(len(t),1));z[:,INDEX]=np.array(yy);d=np.array(dd)
 checks=prior.check(z,cs);checks['ATP_balance']=float(abs(z[:,120]-base[120]-z[:,123]+z[:,125]).max());checks['adenylate_balance']=float(abs(z[:,120:123].sum(1)-base[120:123].sum()).max())
 assert max(checks.values())<3e-5 and z[:,INDEX].min()>-1e-7,checks
 g=(d[:,254]+d[:,328]+d[:,493])/(d0[254]+d0[328]+d0[493]);at=lambda h:int(np.argmin(abs(t-h*3600)))
 out=dict(late40_support=a,late60_support=b,checks=checks,precursor40_max_fold=float(z[:,390].max()/base[390]),precursor60_max_fold=float(z[:,391].max()/base[391]),synthesis_min=float(g.min()),synthesis_8h=float(g[at(8)]),synthesis_24h=float(g[-1]),ATP_min_uM=float(z[:,120].min()),ATP_final_uM=float(z[-1,120]),precursor40_final_fold=float(z[-1,390]/base[390]),precursor60_final_fold=float(z[-1,391]/base[391]))
 return t,z,g,out
def scenarios():
 rows=[];selected={}
 for a in [0.,.25,1.]:
  for b in [0.,.25,1.]:
   t,z,g,r=molecular(a,b);rows.append(r);print(a,b,r['synthesis_min'],flush=True)
   if (a,b) in [(0.,1.),(1.,0.),(0.,0.),(1.,1.)]:
    key=f'{a}_{b}';np.savez_compressed(B/('maturation_'+key+'_24h.npz'),time_s=t,state=z,translation=g,integrated_indices=INDEX);selected[key]=(t,z,g)
 t,z,g,r=molecular(0.,0.,strict=True);err=float(np.max(abs(z-selected['0.0_0.0'][1])/(1+abs(z))));assert err<3e-5
 save('scenarios.json',dict(complete=True,runs=9,strict_error=err,protocol='24h; supports modified2-8h and12-18h, no state reset',scope='Selected somatic molecular subsystem, effective ATP supply; not human calibration or whole electrical neuron24h',rows=rows))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(10,6))
 for key,(t,z,g) in selected.items():
  for ax,y in zip(axs.flat,[z[:,390]/z[0,390],z[:,391]/z[0,391],g,z[:,120]]):ax.plot(t/3600,y,label='40S/60S '+key.replace('_','/'))
 for ax,title in zip(axs.flat,['Premature40S / baseline','Premature60S / baseline','Selected translation / baseline','ATP (uM)']):
  ax.set_title(title);ax.set_xlabel('Time (h)');ax.grid(alpha=.2)
  for l,r in [(2,8),(12,18)]:ax.axvspan(l,r,color='grey',alpha=.1)
 axs[0,0].legend(fontsize=7);fig.suptitle('Conditional maturation/recovery: unknown human kinetics');fig.tight_layout();fig.savefig(B/'maturation_recovery.png',dpi=130);plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['unit','integrated','electrical','scenarios']);a=p.parse_args();globals()[a.mode]()
