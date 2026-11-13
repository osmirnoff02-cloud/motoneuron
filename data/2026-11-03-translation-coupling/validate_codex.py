"""[codex]2026-11-03. Conservation and whole-neuron coupling tests."""
from pathlib import Path
from dataclasses import replace
import sys,importlib.util,json,time
import numpy as np
B=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('coupling_prior_validation',B.parent/'2026-11-03-rna-salvage/validate_codex.py');q=importlib.util.module_from_spec(s);sys.modules[s.name]=q;s.loader.exec_module(q)
api=q.api;m=q.m;ct=api._ct44
def save(n,x):
 (B/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n',encoding='utf8')
def run(sg,**kw):return api.run_coupled_translation_neuron(sg,**kw)
def assess(r):
 z=r['state'];ch=r['checks'];assert z.shape[1]==len(r['columns'])==ct.N
 assert max(v for k,v in ch.items() if k!='minimum_state')<3e-5,sorted(ch.items(),key=lambda kv:-kv[1])[:10]
 return dict(passed=True,states=ct.N,checks=ch,nfev=r['nfev'],ATP_end=z[-1,[120,136]],AA_end=z[-1,ct.A],
  tRNA_damage_end=(z[-1,ct.D]+z[-1,ct.H]).sum(),GAP43_end=z[-1,583],counters=z[-1,1589:],handoff=r['coupled_handoff'])
def basic(duration=10.):
 start=time.monotonic();r=run(m.schedule([(0,duration,0,1,0)]),**m.m.hist.options(),output_step_s=duration/10,max_step_s=2,rtol=1e-8,atol=1e-10)
 save('quick_codex.json',dict(**assess(r),elapsed_s=time.monotonic()-start))
 np.savez_compressed(B/'baseline_codex.npz',state=r['baseline_state']);print('basic passed',r['nfev'],time.monotonic()-start,flush=True)
def boundaries():
 # A synthetic local rest isolates ownership from electrical background demand.
 z=np.zeros(ct.N);z[136]=2700;z[137]=270;z[574]=1000;z[575]=25;z[576]=400;z[577]=100;z[731]=100
 z[ct.A]=50;z[ct.Q]=25/62;z[ct.U]=25/62;z[564:566]=1;z[607]=.001
 tv=api.LocalTranslationParameters(length_aa=238).vector();v=api.CoupledTranslationParameters().vector();ctl=np.ones(6)
 def f(x=z,p=v,old=None):return ct.finish(x,np.zeros(745) if old is None else old,p,tv,ctl,.1,100.,100.,100.)
 a=f();assert abs(a[136]+a[137])<1e-12
 assert abs(-a[136]-(a[594]+a[595]+a[ct.K['split_ATP']]+a[ct.K['release_ATP']]))<1e-12
 assert abs(a[575]-a[589]+a[588])<1e-12
 assert abs(a[576]+a[590]+a[591])<1e-12
 zz=z.copy();zz[ct.A][ct.recipe.AI['K']]=0
 b=f(zz);lys=np.flatnonzero(ct.DA==ct.recipe.AI['K']);assert np.all(b[ct.Q][lys]<=0)
 zz=z.copy();zz[136]=0;b=f(zz);assert b[594]==b[ct.K['split_ATP']]==0
 zz=z.copy();zz[ct.H][0]=1;zz[ct.U][0]-=1;zz[731]=0;assert f(zz)[ct.K['repaired']]==0
 zz[731]=100;b=f(zz);assert b[ct.K['repaired']]>0
 assert abs(b[731]+3*b[ct.K['repaired']]/100)<1e-12
 # Repair uses an already activated reserve; it does not add another ATP debit.
 no_repair=v.copy();no_repair[6]=0
 assert abs(b[136]-f(zz,no_repair)[136])<1e-12
 zz=z.copy();zz[ct.R][12]=.001;b=f(zz)
 assert b[ct.K['cleaved']]>0 and abs(b[727]-3*b[ct.K['cleaved']]/100)<1e-12
 assert abs(b[ct.PEP].sum()-b[ct.K['released_aa']])<1e-12
 o=np.zeros(745);o[575]=-.01;b=f(z,v,o);assert abs(b[ct.Q].sum()-(a[ct.Q].sum()-.01))<1e-12
 assert abs((b[ct.U]-a[ct.U]).sum()-.01)<1e-12
 save('boundaries_codex.json',dict(passed=True,checks=['single_ATP_owner','adenylate','GTP_owner','charged_tRNA_consumption','no_lysine_substitution','no_ATP_no_charging','no_RNA_reserve_no_CCA','CCA_3nt_transfer_no_second_ATP','cleaved_CCA_to_debris','exact_RQC_prefix','other_cargo_competes_for_same_tRNA']))
 print('boundary checks passed',flush=True)

def electrical():
 prev=q.q.q;prev.B=B;prev.run=run;prev.assess=assess;prev.save=save;prev.electrical()

def contracts():
 rows=[]
 cases=[dict(local_rqc_schedule=[[0,1,1,1]]),dict(local_translation_parameters=replace(api._lm35.cargo_parameters(),initiation_GTP_per_chain=4))]
 for kw in cases:
  try:api.run_coupled_translation_neuron([],**kw)
  except ValueError as e:rows.append(str(e))
  else:raise AssertionError('Unsupported legacy override silently accepted')
 save('api_contract_codex.json',dict(passed=True,rejections=rows));print('API contracts passed',flush=True)

def scenarios():
 rows=[];sg=m.schedule([(0,300,0,1,0)]);opts=dict(**m.m.hist.options(),output_step_s=10,max_step_s=10,rtol=1e-8,atol=1e-10)
 for name,p in [('control',api.CoupledTranslationParameters()),('pause',replace(api.CoupledTranslationParameters(),pause_multiplier=0.)),('pause_no_repair',replace(api.CoupledTranslationParameters(),pause_multiplier=0.,elac1_per_s=0.,trnt1_per_s=0.)),('charging_off',replace(api.CoupledTranslationParameters(),charging_support=0.))]:
  t=time.monotonic();r=run(sg,coupled_translation_parameters=p,**opts)
  row=dict(name=name,parameters=p.__dict__,**assess(r),elapsed_s=time.monotonic()-t);rows.append(row);save('scenarios_codex.json',dict(complete=len(rows)==4,rows=rows))
  np.savez_compressed(B/(name+'_trace_codex.npz'),time_s=r['time_s'],state=r['state'],baseline=r['baseline_state'])
  print(name,row['elapsed_s'],row['counters'][:7],flush=True)

def restart():
 sg=m.schedule([(0,30,0,1,0),(30,60,0,1,0)]);opts=dict(**m.m.hist.options(),output_step_s=10,max_step_s=5,rtol=1e-8,atol=1e-10)
 r=run(sg,**opts);a=run(sg[:1],**opts);b=run(sg[1:],checkpoint=a['checkpoint'],**opts)
 tight=run(sg,**dict(opts,rtol=2e-9,atol=2e-11,max_step_s=2))
 for x in (r,a,b,tight):assess(x)
 err=lambda a,b:float(np.max(abs(a-b)/(1+abs(b))))
 e=err(b['state'][-1],r['state'][-1]);t=err(tight['state'][-1],r['state'][-1]);assert max(e,t)<3e-5
 save('restart_codex.json',dict(passed=True,restart_scaled_error=e,tighter_tolerance_error=t));print('restart/tolerance passed',e,t,flush=True)

if __name__=='__main__':
 import faulthandler
 faulthandler.dump_traceback_later(300,repeat=True)
 for action in sys.argv[1:] or ['boundaries','basic']:globals()[action]()
