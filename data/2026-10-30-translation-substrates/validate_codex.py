"""[codex]2026-10-30. Finite substrates: mass, energy, boundaries, integration."""
from pathlib import Path
import sys,json,numpy as np,importlib.util,argparse
from dataclasses import asdict,replace
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
_threads=threadpool_limits(1);B=Path(__file__).resolve().parent
sys.path.insert(0,str(B.parent/'2026-10-30-shared-translation'))
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
prior=module('shared_substrate_validation',B.parent/'2026-10-30-shared-translation/validate_codex.py');api=prior.api
from substrates_codex import *
from connected_parameters_codex import schedule
def save(name,x):(B/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a)),encoding='utf8')
def constants(params=None,bp=None,sp=None):
 ctx=list(prior.constants(bp,sp));base=ctx[10].copy();base[121]=api.EnergyParameters().ADP0_uM;base[122]=api.EnergyParameters().AMP0_uM;ctx[10]=base
 v=(params or TranslationSubstrateParameters()).vector();q,ref=initial(v,ctx[5],ctx[6],ctx[13],base[120],base[121]);z=np.r_[base,q]
 return tuple(ctx),v,ref,z
def rhs_component(z,ctx,v,ref,nc,cc,bc,sc,control):
 nc=nc.copy();nc[2]*=gate(z,v)
 old=prior.component_rhs(z[:497],ctx,nc,cc,bc,sc)
 return append_substrates(z,old,v,ref,ctx[5],ctx[6],control)
def unit():
 ctx,v,ref,z=constants();nc=np.ones(4);cc=np.array([1.,1.,1.,0.]);bc=np.ones(7);sc=np.ones(7);ctl=np.ones(3)
 d=rhs_component(z,ctx,v,ref,nc,cc,bc,sc,ctl)
 assert max(abs(d[497:501]))<1e-12 and abs(d[125])<1e-12
 assert max(checks(z[None,:],z,v).values())<1e-12
 # No energy: charging and forward NDPK stop; reverse consumes GTP and gives ATP.
 q=z.copy();q[120]=0;raw=fluxes(q,d[:497],v,ref,ctx[5],ctx[6],ctl)
 assert raw[7]==0 and raw[10]==0 and raw[11]>0
 for k in [498,499]:
  q=z.copy();q[k]=0;assert gate(q,v)==0
 q=z.copy();q[498]=v[1];raw=fluxes(q,d[:497],v,ref,ctx[5],ctx[6],ctl);assert raw[7]==0
 q=z.copy();q[497]=0;raw=fluxes(q,d[:497],v,ref,ctx[5],ctx[6],ctl);assert raw[7]==0 and raw[0]>=0
 # Recovered orphan mass alone returns to AA, never creates new GTP or charge.
 raw0=np.zeros(497);raw0[382]=1;raw=fluxes(z,raw0,v,ref,ctx[5],ctx[6],np.zeros(3))
 assert abs(raw[6]-ctx[5][0]*v[10])<1e-12
 try:constants(replace(TranslationSubstrateParameters(),GTP0_uM=1e6));raise AssertionError('Impossible basal NDPK accepted')
 except ValueError:pass
 save('unit_codex.json',dict(passed=True,basal_residue_flux_uM_s=ref[0],derived_NDPK_coefficient=ref[1],baseline_pool_drift=float(abs(d[497:501]).max()),baseline_energy_difference=float(d[125]),charged_tRNA_baseline=z[498],stoichiometry='2charging ATP equivalents +2GTP per residue; deltaATP=2deltaCharged+deltaGTP'))
 print('Substrate unit and energy boundary checks passed',flush=True)
def integrated():
 sg=schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)]);es=np.c_[sg[:,:2],[0,25,0],[0,0,0]]
 opts=dict(**prior.hist.v29.options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
 run=api.run_translation_substrates_neuron;r=run(sg,**opts);assert r['state'].shape[1]==len(r['columns'])==513
 a=run(sg[:2],**dict(opts,electrical_schedule=es[:2]));c=run(sg[2:],checkpoint=a['checkpoint'],**dict(opts,electrical_schedule=es[2:]))
 strict=run(sg,**dict(opts,rtol=3e-10,atol=3e-12,max_step_s=.00125));err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
 restart=err(c['state'][-1],r['state'][-1]);conv=err(strict['state'][-1],r['state'][-1]);assert max(restart,conv)<1e-5
 ctl=np.c_[sg[:,:2],[[1,1,1],[0,0,0],[1,1,1]]];block=run(sg,**dict(opts,substrate_schedule_input=ctl))
 bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][499]+=1
 try:run(sg[2:],checkpoint=bad,**dict(opts,electrical_schedule=es[2:]));raise AssertionError('Corrupt guanylate checkpoint accepted')
 except ValueError:pass
 save('integrated_codex.json',dict(complete=True,states=513,restart_error=restart,strict_error=conv,checks=r['checks'],block_checks=block['checks'],config=r['config']))
 np.savez_compressed(B/'integrated_codex.npz',time_s=r['time_s'],state=r['state']);print('513-state integration passed',flush=True)
def electrical():
 metrics=prior.hist.module('substrates_electrical_features',B.parent/'2026-10-26-nak-source-calibration/validate_model_codex.py')
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical_codex.json').read_text())['rows'];rows=[]
 for target in targets:
  profile=target['profile'];cur=target['current_pA'];dur=.5 if profile=='Burley' else 1.
  sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  opts=dict(**prior.hist.v29.options(profile),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  entries={}
  for label,run in [('legacy028',api.run_shared_translation_neuron),('substrates029',api.run_translation_substrates_neuron)]:
   r=run(sg,**opts);f=metrics.features(r,.2,.2+dur,profile);assert f['spikes']==target['target']
   delta=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.02
   entries[label]=dict(features=f,max_change=delta)
  rows.append(dict(profile=profile,current_pA=cur,branches=entries));save('electrical_codex.json',dict(complete=len(rows)==9,rows=rows));print(profile,cur,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['unit','integrated','electrical'],required=True);a=p.parse_args();globals()[a.mode]()
