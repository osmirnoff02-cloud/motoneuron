"""[codex] 2026-10-30. Independent stock/flow, pulse and electrical checks."""
from pathlib import Path
import sys,json,argparse,importlib.util,numpy as np
from dataclasses import replace,asdict
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
_threads=threadpool_limits(limits=1)
B=Path(__file__).resolve().parent
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
prior=module('cycle_validation_for_bio',B.parent/'2026-10-30-nuclear-targets-cycle/validate.py');api=prior.api
from biogenesis import *
from nucleus_ribosome import NucleusRibosomeParameters,initial_nucleus
from nuclear_cycle import RibosomeCycleParameters,TransportParameters,KCNQCouplingParameters,initial_kcnq
from connected_parameters import schedule
def save(name,x):(B/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a)),encoding='utf8')
def constants(parameters=None):
 p,hv,f0,atp=prior.prior.constants();nv=api.energy_terminal_inputs('Burley')['organelle_parameters'].vector()
 b=(parameters or BiogenesisParameters()).vector();budget=api.BudgetParameters().vector()
 v,rv=prepare(NucleusRibosomeParameters().vector(),RibosomeCycleParameters().vector(),p,budget,b)
 cybase=initial_cycle(v,rv);biobase=initial(v,rv,b)
 z=np.zeros(403);z[[6,10,11,50,63,69]]=1.;z[120]=atp;z[320:330]=initial_nucleus(p,hv,v,f0)
 z[330:342]=cybase;z[342:364]=initial_kcnq(TransportParameters(),KCNQCouplingParameters(),[-65,-65]);z[364:]=biobase
 return p,hv,f0,atp,nv,b,v,rv,cybase,biobase,z
def unit():
 p,hv,f0,atp,nv,b,v,rv,cy,bb,z=constants()
 dq,dc,de,fac,cost=bio_fluxes(z,v,rv,b,atp,np.ones(4),np.array([1.,1.,1.,0.]),np.ones(7),cy,bb)
 assert max(abs(dc[:8]))<1e-12 and max(abs(dq[:14]))<1e-12 and max(abs(dq[20:28]))<1e-12
 assert np.allclose(fac,1) and np.allclose(cost,[p[20]*414*4,np.log(2)/v[11]*v[9]*v[10]*4])
 assert max(checks(z[None,:],z,v,b).values())<1e-12
 # One partially synthesized TDP cohort, all production/inflow off. Energy
 # in its past is initial stock; future elongation charges before completion.
 q=z.copy();q[326:328]=0;q[330:342]=0;q[330:332]=1;q[364:]=0
 q[326]=.1;q[330:332]-=.1 # cohort initially at start of translation
 qbase=q.copy();cc=np.array([0.,1.,1.,1.]);nc=np.array([1.,0.,1.,1.]);bc=np.zeros(7);bc[4:6]=1
 def rhs(t,part):
  zz=qbase.copy();zz[326:328]=part[:2];zz[330:342]=part[2:14];zz[364:]=part[14:]
  a,c,e,f,en=bio_fluxes(zz,v,rv,b,atp,nc,cc,bc,cy,bb)
  return np.r_[e,c,a]
 y0=np.r_[q[326:328],q[330:342],q[364:]]
 sol=solve_ivp(rhs,(0,12000),y0,method='LSODA',rtol=1e-9,atol=1e-12,t_eval=np.r_[np.arange(0,121,1),np.arange(150,12001,30)])
 assert sol.success
 out=np.tile(qbase,(len(sol.t),1));out[:,326:328]=sol.y[:2].T;out[:,330:342]=sol.y[2:14].T;out[:,364:]=sol.y[14:].T
 # Translation ledger normally populated by final ATP extension.
 out[:,254]=4*b[0]*out[:,378];out[:,328]=4*b[0]*out[:,379]
 ck=checks(out,qbase,v,b);assert max(ck.values())<1e-7,ck
 assert out[1,378]>100*out[1,380] # costs precede finished protein
 assert out[:,372].max()>0 and out[-1,382]>0 # retained60S and discarded residues
 assert out[-1,378]>out[-1,380] # irrecoverable synthesis cost of aborted chains
 assert abs(mature_counts(out[-1])[0]-1)<1e-7 and abs(mature_counts(out[-1])[1]-1)<1e-7
 try:prepare(v,rv,p,api.BudgetParameters().vector(),replace(BiogenesisParameters(),selected_mature_pool_uM=1e-8).vector());raise AssertionError('Capacity accepted')
 except ValueError:pass
 save('unit.json',dict(passed=True,effective_slots=v[6:8],partial_chain_checks=ck,energy_equivalents_per_completed_aa=4,
  one_second_incorporated_aa=out[1,378],one_second_completed_aa=out[1,380],cohort_incorporated_aa=out[-1,378],cohort_completed_aa=out[-1,380],cohort_cleared_aa=out[-1,382],RQC60_peak=out[:,372].max()))
 np.savez_compressed(B/'single_cohort.npz',time_s=sol.t,state=out)
 print('Unit stock/flow and nascent pulse passed',flush=True)
def integrated():
 sg=schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)]);es=np.c_[sg[:,:2],[0,25,0],[0,0,0]]
 opts=dict(**prior.prior.v29.options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
 r=api.run_biogenesis_neuron(sg,**opts);assert r['state'].shape[1]==len(r['columns'])==403
 a=api.run_biogenesis_neuron(sg[:2],**dict(opts,electrical_schedule=es[:2]))
 bb=api.run_biogenesis_neuron(sg[2:],checkpoint=a['checkpoint'],**dict(opts,electrical_schedule=es[2:]))
 strict=api.run_biogenesis_neuron(sg,**dict(opts,rtol=3e-10,atol=3e-12,max_step_s=.00125))
 err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
 restart=err(bb['state'][-1],r['state'][-1]);convergence=err(strict['state'][-1],r['state'][-1]);assert max(restart,convergence)<1e-5
 rib=np.c_[sg[:,:2],[[1,1,1,0],[1,1,1,1],[1,1,1,0]]]
 bio=np.c_[sg[:,:2],[[1]*7,[0,0,0,0,0,0,1],[1]*7]]
 block=api.run_biogenesis_neuron(sg,**dict(opts,ribosome_schedule=rib,biogenesis_schedule=bio))
 assert block['state'][:,370:372].max()>0
 bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][330]+=.1
 try:api.run_biogenesis_neuron(sg[2:],checkpoint=bad,**dict(opts,electrical_schedule=es[2:]));raise AssertionError('Invalid checkpoint accepted')
 except ValueError:pass
 save('integrated.json',dict(complete=True,states=403,restart_error=restart,convergence_error=convergence,checks=r['checks'],blockade_checks=block['checks'],config=r['config']))
 np.savez_compressed(B/'integrated_trace.npz',time_s=r['time_s'],state=r['state'])
 print('403-state integration/restart/convergence/blocks passed',flush=True)
def electrical():
 metrics=prior.prior.module('bio_electrical_features',B.parent/'2026-10-26-nak-source-calibration/validate_model.py')
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical.json').read_text())['rows'];rows=[]
 for target in targets:
  profile=target['profile'];cur=target['current_pA'];dur=.5 if profile=='Burley' else 1.
  sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  opts=dict(**prior.prior.v29.options(profile),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  entries={}
  for label,runner in [('legacy026',api.run_nuclear_cycle_neuron),('biogenesis027',api.run_biogenesis_neuron)]:
   r=runner(sg,**opts);f=metrics.features(r,.2,.2+dur,profile);assert f['spikes']==target['target']
   error=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert error<.02,(profile,error)
   entries[label]=dict(features=f,max_change=error,checks=r['checks'])
  rows.append(dict(profile=profile,current_pA=cur,branches=entries));save('electrical.json',dict(complete=len(rows)==9,rows=rows));print(profile,cur,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=['unit','integrated','electrical']);args=parser.parse_args();globals()[args.mode]()
