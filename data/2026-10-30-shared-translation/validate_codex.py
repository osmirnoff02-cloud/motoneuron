"""[codex] 2026-10-30. Shared-resource balance, baseline and integration tests."""
from pathlib import Path
import sys,json,importlib.util,argparse,numpy as np
from dataclasses import replace,asdict
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
_threads=threadpool_limits(1);B=Path(__file__).resolve().parent
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
prior=module('bio_validation_shared',B.parent/'2026-10-30-ribosome-biogenesis/validate_codex.py');api=prior.api;hist=prior.prior.prior
from shared_translation_codex import *
from biogenesis_codex import replace_targets,BiogenesisParameters
from nuclear_cycle_codex import kcnq_vector,kcnq_rhs,kcnq_ratios,TransportParameters,KCNQCouplingParameters
from connected_parameters_codex import schedule
def save(name,data):(B/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a)),encoding='utf8')

def constants(bp=None,sp=None):
 p,hv,f0,atp,nv,b,v,rv,cy,bb,z=prior.constants(bp)
 sv=(sp or SharedTranslationParameters()).vector();kp=kcnq_vector(TransportParameters(),KCNQCouplingParameters())
 sh,cy,ref=setup(cy,v,rv,b,nv,kp,z[342:364],sv);z[330:342]=cy;z=np.r_[z,sh]
 return p,hv,f0,atp,nv,b,v,rv,cy,bb,z,sv,kp,ref

def component_rhs(z,ctx,nc,cc,bc,sc):
 p,hv,f0,atp,nv,b,v,rv,cy,bb,base,sv,kp,ref=ctx
 old=np.zeros(247);old[18]=p[20]*max(z[50],0);old[6]=old[18]-p[20]*z[6]
 prod=np.log(2)/nv[3];dfl=prod*z[63];daf=np.log(2)/nv[5]*z[66];p207=prod*z[64];d207=np.log(2)/nv[4]*z[65]
 old[63]=prod*(1-z[64])-dfl;old[65]=p207-d207;old[66]=-daf
 old[75]=prod*(1-z[64]);old[76]=dfl+daf;old[77]=p207;old[78]=d207
 row=schedule([(0,1,0,1,0)])[0]
 out,nx,dc,dq,en=replace_targets(z,old,p,hv,nv,v,rv,b,f0,atp,nc,cc,bc,cy,bb,row)
 out,dc,dq,dx,complete,cost=replace_shared(z,out,nx,dc,dq,p,v,rv,b,sv,ref,kp,nv,atp,nc,cc,bc,sc,cy)
 rates=np.zeros(497);rates[:247]=out;rates[320:330]=nx;rates[330:342]=dc;rates[364:403]=dq
 rates[342:362]=kcnq_rhs(z[342:362],max(z[6],0)*z[320]/f0,kp)
 for i in range(2):
  syn=complete[2+i]*b[0]/sv[0];oldsrc=np.log(2)/kp[5]*max(z[342+i],0)
  rates[344+6*i]+=syn-oldsrc;rates[358+2*i]=syn
 rates[254]=en[0];rates[328]=en[1];rates[403:493]=dx.ravel();rates[493:497]=cost
 return rates

INDEX=np.r_[np.array([6,10,11,18,50,63,64,65,66,75,76,77,78,254]),np.arange(320,497)]

def unit():
 ctx=constants();p,hv,f0,atp,nv,b,v,rv,cy,bb,z,sv,kp,ref=ctx
 nc=np.ones(4);cc=np.array([1.,1.,1.,0.]);bc=np.ones(7);sc=np.ones(7)
 d=component_rhs(z,ctx,nc,cc,bc,sc)
 active=np.r_[np.array([6,10,11,50,63,64,65,66]),np.arange(320,328),np.arange(330,338),np.arange(342,356),np.arange(364,378),np.arange(384,392),np.concatenate([np.arange(403+15*i,414+15*i) for i in range(6)])]
 assert abs(d[active]).max()<1e-10,(active[np.argmax(abs(d[active]))],d[active].max())
 assert max(checks(z[None,:],z,v,b,sv,ref).values())<1e-12
 total40=cy[0]+sum(cy[2:8])+sum(z[326:328])+bound_counts(z[403:493].reshape(6,15))[0]
 assert abs(total40-1)<1e-12
 # Match old allocation exactly at baseline: RP debit moved, never duplicated.
 birth=np.log(2)/(b[1]*86400);oldbio=b[0]*birth*(sum(b[11:15])+2*b[15])
 assert abs(d[400]+d[493]-oldbio-(d[496]-d[494]))<1e-10
 blocked=bc.copy();blocked[0]=0;db=component_rhs(z,ctx,nc,cc,blocked,sc)
 assert db[392]==0 and db[393]>0 # 47S off does not force5S off
 try:constants(sp=replace(SharedTranslationParameters(),PKN_unit_uM=1e5));raise AssertionError('Capacity accepted')
 except ValueError:pass
 # Synthesis of mutantPKN is distinct from caspase-generated AF3.
 q=z.copy();q[64]=.4;dv=component_rhs(q,ctx,nc,cc,bc,sc)
 assert dv[403+5*15]>0 and dv[66]==0
 save('unit_codex.json',dict(passed=True,baseline_max_derivative=float(abs(d[active]).max()),free40=float(cy[0]),free60=float(cy[1]),shared_chain_fluxes=ref[:6],extra_background_ATP_uM_s=ref[14],fiveS_independent=True))
 print('Shared baseline/mass/allocation/5S/PKN checks passed',flush=True)

def integrated():
 sg=schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)]);es=np.c_[sg[:,:2],[0,25,0],[0,0,0]]
 opts=dict(**hist.v29.options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
 run=api.run_shared_translation_neuron;r=run(sg,**opts);assert r['state'].shape[1]==len(r['columns'])==497
 a=run(sg[:2],**dict(opts,electrical_schedule=es[:2]));c=run(sg[2:],checkpoint=a['checkpoint'],**dict(opts,electrical_schedule=es[2:]))
 strict=run(sg,**dict(opts,rtol=3e-10,atol=3e-12,max_step_s=.00125))
 err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
 restart=err(c['state'][-1],r['state'][-1]);conv=err(strict['state'][-1],r['state'][-1]);assert max(restart,conv)<1e-5
 sh=np.c_[sg[:,:2],np.ones((3,7))];sh[1,2:4]=0;sh[1,4:6]=10
 rib=np.c_[sg[:,:2],[[1,1,1,0],[1,1,1,1],[1,1,1,0]]]
 block=run(sg,**dict(opts,shared_translation_schedule=sh,ribosome_schedule=rib))
 bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][403]+=.1
 try:run(sg[2:],checkpoint=bad,**dict(opts,electrical_schedule=es[2:]));raise AssertionError('Invalid state accepted')
 except ValueError:pass
 save('integrated_codex.json',dict(complete=True,states=497,restart_error=restart,convergence_error=conv,checks=r['checks'],block_checks=block['checks'],config=r['config']))
 np.savez_compressed(B/'integrated_trace_codex.npz',time_s=r['time_s'],state=r['state']);print('497-state integration passed',flush=True)

def electrical():
 metrics=hist.module('shared_electrical_features',B.parent/'2026-10-26-nak-source-calibration/validate_model_codex.py')
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical_codex.json').read_text())['rows'];rows=[]
 for target in targets:
  profile=target['profile'];cur=target['current_pA'];dur=.5 if profile=='Burley' else 1.
  sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  opts=dict(**hist.v29.options(profile),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  entries={}
  for label,runner in [('legacy027',api.run_biogenesis_neuron),('shared028',api.run_shared_translation_neuron)]:
   r=runner(sg,**opts);f=metrics.features(r,.2,.2+dur,profile);assert f['spikes']==target['target']
   delta=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.02
   entries[label]=dict(features=f,max_change=delta)
  rows.append(dict(profile=profile,current_pA=cur,branches=entries));save('electrical_codex.json',dict(complete=len(rows)==9,rows=rows));print(profile,cur,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['unit','integrated','electrical'],required=True);a=p.parse_args();globals()[a.mode]()
