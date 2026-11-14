"""[codex] 2026-10-30. Mass ownership, response, replay and electrical controls."""
from pathlib import Path
import sys,json,importlib.util,argparse,numpy as np
from dataclasses import replace,asdict
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
_threads=threadpool_limits(1);B=Path(__file__).resolve().parent
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
prior=module('substrates_recycling_validation',B.parent/'2026-10-30-translation-substrates/validate.py');api=prior.api
from recycling import *
from substrates import TranslationSubstrateParameters
from connected_parameters import schedule

def save(name,x):(B/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a)),encoding='utf8')
def constants(params=None,sp=None):
 ctx,sub,subref,base=prior.constants(sp);v=(params or NuclearRecyclingParameters()).vector()
 n=np.ones(4);cc=np.array([1.,1.,1.,0.]);ones=np.ones(7)
 old=prior.rhs_component(base,ctx,sub,subref,n,cc,ones,ones,np.ones(3))
 src=removal_sources(base,old,ctx[0][20]*base[6]*414*api.BudgetParameters().tdp_uM_per_lineage_unit,ctx[5],ctx[6],ctx[4],ctx[11],ctx[12])
 q,ref=initial(src,v,subref)
 return ctx,sub,subref,v,ref,np.r_[base,q],src

def component(z,ctx,sub,subref,v,ref,nc,cc,bc,sc,sctl,rctl):
 nc,cc,bc,sc=modulate(z,v,sub,ctx[7],nc,cc,bc,sc)
 old=prior.rhs_component(z[:513],ctx,sub,subref,nc,cc,bc,sc,sctl)
 src=removal_sources(z,old,ctx[0][20]*max(z[6],0.)*414*api.BudgetParameters().tdp_uM_per_lineage_unit,ctx[5],ctx[6],ctx[4],ctx[11],ctx[12])
 return append_recycling(z,old,src,v,ref,sub,ctx[6],ctx[10][120],rctl,sctl[0],1.)

def unit():
 ctx,sub,sref,v,ref,z,src=constants();nc=np.ones(4);cc=np.array([1.,1.,1.,0.]);bc=np.ones(7);sc=np.ones(7)
 d=component(z,ctx,sub,sref,v,ref,nc,cc,bc,sc,np.ones(3),np.ones(6))
 assert np.max(abs(d[np.r_[497:501,513:516,532:537]]))<1e-12
 assert d[501]>=0 and d[521]>0 and d[125]==0
 assert max(checks(z[None,:],z,v,sub).values())<1e-12
 # Released material not counted twice and proteolysis does not mint ATP/GTP.
 none=np.zeros(513);q=z.copy();q[513:516]=[2.,3.,4.]
 dx=append_recycling(q,none,np.zeros(8),v,np.zeros(2),sub,ctx[6],z[120],np.ones(6),0.,1.)
 assert abs(dx[513:516].sum()+dx[521]+dx[522])<1e-12
 assert dx[120]==0 and np.max(abs(dx[498:501]))==0
 blocked=np.ones(6);blocked[2]=0
 dx=append_recycling(q,none,np.zeros(8),v,np.zeros(2),sub,ctx[6],z[120],blocked,0.,1.)
 assert dx[520]==0 and dx[515]>=0
 # Cleavage and generic autophagic counter alone cannot create AA.
 fake=np.zeros(513);fake[82]=100.;fake[19]=100.
 sr=removal_sources(np.zeros(537),fake,0.,ctx[5],ctx[6],ctx[4],ctx[11],ctx[12]);assert sr.sum()==0
 # Source conservation has distinct counters for RP retirement and somatic cargo.
 fake[398]=1.;fake[399]=2.;sr=removal_sources(np.zeros(537),fake,0.,ctx[5],ctx[6],ctx[4],ctx[11],ctx[12])
 assert np.isclose(sr.sum(),ctx[5][0]*(ctx[5][13]+2*ctx[5][14])/4)
 q=z.copy();q[498]=0.;q[532]=1.;q[533]=.5
 dd=component(q,ctx,sub,sref,v,ref,nc,cc,bc,sc,np.ones(3),np.ones(6));assert dd[534]<1e-14
 try:initial(src*1e6,v,sref);raise AssertionError('Excess basal return accepted')
 except ValueError:pass
 save('unit.json',dict(passed=True,baseline_max_drift=float(abs(d[np.r_[497:501,513:516,532:537]]).max()),basal_sources=dict(zip(SOURCES,src)),basal_return_uM_s=ref[0],old_external_AA_uM_s=sref[0],new_external_AA_uM_s=d[501],initial_recycling_pools_uM=z[513:516],activity_proxy_not_absolute_expression=True))
 print('Recycling unit checks passed',flush=True)

def integrated():
 sg=schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)]);es=np.c_[sg[:,:2],[0,25,0],[0,0,0]]
 opts=dict(**prior.prior.hist.v29.options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
 run=api.run_nuclear_recycling_neuron;r=run(sg,**opts);assert r['state'].shape[1]==len(r['columns'])==537
 a=run(sg[:2],**dict(opts,electrical_schedule=es[:2]));c=run(sg[2:],checkpoint=a['checkpoint'],**dict(opts,electrical_schedule=es[2:]))
 strict=run(sg,**dict(opts,rtol=3e-10,atol=3e-12,max_step_s=.00125));err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
 restart=err(c['state'][-1],r['state'][-1]);conv=err(strict['state'][-1],r['state'][-1]);assert max(restart,conv)<1e-5
 controls=np.c_[sg[:,:2],np.ones((3,6))];controls[1,4]=0
 block=run(sg,**dict(opts,nuclear_recycling_schedule=controls))
 bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][515]+=1
 try:run(sg[2:],checkpoint=bad,**dict(opts,electrical_schedule=es[2:]));raise AssertionError('Unbalanced recycled matter accepted')
 except ValueError:pass
 save('integrated.json',dict(complete=True,states=537,restart_error=restart,strict_error=conv,checks=r['checks'],block_checks=block['checks'],config=r['config']))
 np.savez_compressed(B/'integrated.npz',time_s=r['time_s'],state=r['state']);print('537-state integration passed',flush=True)

def electrical():
 metrics=prior.prior.hist.module('recycling_electrical_features',B.parent/'2026-10-26-nak-source-calibration/validate_model.py')
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical.json').read_text())['rows'];rows=[]
 for target in targets:
  profile=target['profile'];cur=target['current_pA'];dur=.5 if profile=='Burley' else 1.
  sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  opts=dict(**prior.prior.hist.v29.options(profile),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  entries={}
  for label,run in [('legacy029',api.run_translation_substrates_neuron),('recycling030',api.run_nuclear_recycling_neuron)]:
   r=run(sg,**opts);f=metrics.features(r,.2,.2+dur,profile);assert f['spikes']==target['target']
   delta=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.02
   entries[label]=dict(features=f,max_change=delta)
  rows.append(dict(profile=profile,current_pA=cur,branches=entries));save('electrical.json',dict(complete=len(rows)==9,rows=rows));print(profile,cur,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['unit','integrated','electrical'],required=True);a=p.parse_args();globals()[a.mode]()
