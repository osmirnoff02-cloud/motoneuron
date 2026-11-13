"""[codex] 2026-10-30. New coupling, conservation and unchanged electrical targets."""
from pathlib import Path
from dataclasses import replace,asdict
import sys,importlib.util,json,argparse
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
_threads=threadpool_limits(limits=1)
B=Path(__file__).resolve().parent
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
prior=module('prior_nucleus_validation_026',B.parent/'2026-10-30-nucleus-ribosome/validate_codex.py');api=prior.api
from nuclear_cycle_codex import *
from nucleus_ribosome_codex import NucleusRibosomeParameters,initial_nucleus
from connected_parameters_codex import schedule
from transport_codex import _rhs as old_transport,baseline as old_baseline

def save(name,x):
 (B/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a)),encoding='utf8')

def unit():
 p,hv,f0,atp=prior.constants();v=NucleusRibosomeParameters().vector();rv=RibosomeCycleParameters().vector();cb=initial_cycle(v,rv)
 x=initial_nucleus(p,hv,v,f0);cy,de,fac=cycle_fluxes(cb,x[6:8],1.,1.,atp,atp,v,rv,np.array([1.,1.,1.,0.]),np.ones(4),cb)
 assert max(abs(cy[:10]))<1e-12 and max(abs(de))<1e-12 and max(abs(fac-1))<1e-12
 rng=np.random.default_rng(26)
 for i in range(12):
  cr=cb*rng.uniform(.5,1.5,len(cb));e=x[6:8]*.7
  d,ee,ff=cycle_fluxes(cr,e,.8,.6,atp*.8,atp,v,rv,np.array([.3,.4,.2,1.]),np.ones(4),cb)
  assert abs(d[0]+sum(d[2:8])+sum(ee))<1e-14
  assert abs(d[1]+sum(d[4:8])+sum(ee))<1e-14
 tp=TransportParameters();kc=KCNQCouplingParameters();kp=kcnq_vector(tp,kc);bt,rs,ps=old_baseline(tp)
 for n in [0.,.15,.7,1.,1.2]:
  trial=bt*rng.uniform(.7,1.3,len(bt));old=old_transport(trial,np.array([n,0,1,1,1,1,1,1]),tp,rs,ps)/3600
  new=kcnq_rhs(np.r_[trial[:14],trial[24:30]],n,kp)
  assert np.max(abs(new-np.r_[old[:14],old[24:30]]))<1e-14
 opts=api.energy_terminal_inputs('Burley');nv=opts['organelle_parameters'].vector()
 z=np.zeros(364);z[6]=1;z[50]=z[10]=z[11]=1;z[120]=atp;z[320:330]=x;z[330:342]=cb;z[69]=1
 old=np.zeros(247);old[18]=p[20];row=schedule([(0,1,0,1,0)])[0]
 normal=replace_cycle_targets(z,old,p,hv,nv,v,rv,f0,atp,np.ones(4),np.array([1.,1.,1.,0.]),cb,row)
 z[320]*=.15
 low=replace_cycle_targets(z,old,p,hv,nv,v,rv,f0,atp,np.ones(4),np.array([1.,1.,1.,0.]),cb,row)
 assert normal[0][64]==0 and low[0][64]>0 and z[6]==1
 bk=initial_kcnq(tp,kc,[-65,-65]);dy=kcnq_rhs(bk[:20],.15,kp);assert dy[1]>0 and abs(dy[:2].sum())<1e-14
 assert abs(sum(dy[2:8])-dy[16]+dy[17])<1e-14
 # Negative or oversized checkpoints must not hide resource creation.
 bad=np.zeros((1,364));bad[0,326:328]=v[6:8];bad[0,330:342]=cb;bad[0,342:364]=bk
 assert max(cycle_checks(bad,bad[0]).values())<1e-12
 bad[0,330]+=.1;assert cycle_checks(bad,bad[0])['selected_40S_balance']>.09
 save('unit_checks_codex.json',dict(passed=True,checks=['40S and 60S independent conservation','matched baseline','KCNQ equivalence to inherited cargo equations','constant total TDP but reduced nucleus drives PKN and KCNQ splicing','no instantaneous channel change','invalid subunit pool rejected']))
 print('Unit checks passed',flush=True)

def integrated():
 sg=schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)]);es=np.c_[sg[:,:2],[0,25,0],[0,0,0]]
 opts=dict(**prior.v29.options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
 r=api.run_nuclear_cycle_neuron(sg,**opts);assert r['state'].shape[1]==len(r['columns'])==364
 a=api.run_nuclear_cycle_neuron(sg[:2],**dict(opts,electrical_schedule=es[:2]))
 b=api.run_nuclear_cycle_neuron(sg[2:],checkpoint=a['checkpoint'],**dict(opts,electrical_schedule=es[2:]))
 restart=float(max(abs(r['state'][-1]-b['state'][-1])/(1+abs(r['state'][-1]))));assert restart<1e-5
 strict=api.run_nuclear_cycle_neuron(sg,**dict(opts,rtol=3e-10,atol=3e-12,max_step_s=.00125))
 conv=float(max(abs(r['state'][-1]-strict['state'][-1])/(1+abs(strict['state'][-1]))));assert conv<1e-4
 # Real intervention in full model: short block of joining/recycling/rescue, stalls on.
 rib=np.c_[sg[:,:2],[[1,1,1,0],[0,0,0,1],[1,1,1,0]]]
 block=api.run_nuclear_cycle_neuron(sg,**dict(opts,ribosome_schedule=rib))
 assert block['state'][:,334:336].max()>0
 bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][330]+=.1
 try:api.run_nuclear_cycle_neuron(sg[2:],checkpoint=bad,**dict(opts,electrical_schedule=es[2:]));raise AssertionError('Invalid resource accepted')
 except ValueError:pass
 save('integrated_checks_codex.json',dict(complete=True,states=364,checks=r['checks'],blockade_checks=block['checks'],restart_scaled_error=restart,convergence_scaled_error=conv,config=r['config']))
 np.savez_compressed(B/'integrated_trace_codex.npz',time_s=r['time_s'],state=r['state'])
 print('Integrated, restart, convergence and blockade passed',flush=True)

def electrical():
 metrics=prior.module('cycle_features26',B.parent/'2026-10-26-nak-source-calibration/validate_model_codex.py')
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical_codex.json').read_text())['rows'];rows=[]
 for target in targets:
  profile=target['profile'];cur=target['current_pA'];dur=.5 if profile=='Burley' else 1.
  sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  opts=dict(**prior.v29.options(profile),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  entries={}
  for name,runner in [('legacy_025',api.run_nucleus_ribosome_neuron),('cycle_026',api.run_nuclear_cycle_neuron)]:
   r=runner(sg,**opts);f=metrics.features(r,.2,.2+dur,profile)
   assert f['spikes']==target['target'],(name,profile,cur,f)
   error=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]))
   assert error<.02,(name,profile,cur,error)
   entries[name]=dict(features=f,max_feature_change=error,checks=r['checks'])
  rows.append(dict(profile=profile,current_pA=cur,target=target['target'],branches=entries));save('electrical_codex.json',dict(complete=len(rows)==9,rows=rows));print(profile,cur,target['target'],flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['unit','integrated','electrical'],required=True);args=parser.parse_args();globals()[args.mode]()
