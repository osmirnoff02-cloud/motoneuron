"""[codex]2026-10-30. Resource conservation, ISR specificity, electrical/replay."""
from pathlib import Path
import sys,json,importlib.util,argparse,numpy as np
from dataclasses import replace
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
_threads=threadpool_limits(1);B=Path(__file__).resolve().parent
sys.path.insert(0,str(B.parent/'2026-10-30-nuclear-recycling'))
sys.path.insert(0,str(B.parent/'2026-10-30-translation-substrates'))
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
hist=module('isr_recycling_validation',B.parent/'2026-10-30-nuclear-recycling/validate_codex.py');api=hist.api
# Validation-process-only module reuse avoids recompiling the identical
# connected RHS for each current. All simulation state remains run-local.
_loader=api._module;_loaded={}
def cached_module(path,name):
 key=(str(path),name)
 if key not in _loaded:_loaded[key]=_loader(path,name)
 return _loaded[key]
api._module=cached_module
from selective_isr_codex import *
from substrates_codex import gate,append_substrates
from recycling_codex import modulate,append_recycling,removal_sources,checks as rec_checks
from shared_translation_codex import checks as sh_checks
from connected_parameters_codex import schedule
def save(name,x):(B/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a)),encoding='utf8')

def constants(ip=None,substrate_parameters=None):
 ctx,sub,sref,nrv,nref,z,src=hist.constants(sp=substrate_parameters);iv=(ip or SelectiveISRParameters()).vector()
 ctx=list(ctx);z,cy,sref,nref,ref=initialize(z,iv,ctx[5],ctx[6],ctx[7],ctx[8],sub,sref,nrv,nref)
 ctx[8]=cy;ctx[10]=z.copy()
 return tuple(ctx),sub,sref,nrv,nref,iv,ref,z

def component(z,cs,ctl,sctl=np.ones(3),rctl=np.ones(6)):
 ctx,sub,sref,nrv,nref,iv,ref,base=cs
 nc=np.ones(4);cc=np.array([1.,1.,1.,0.]);bc=np.ones(7);sc=np.ones(7)
 nc,cc,bc,sc=modulate(z,nrv,sub,ctx[7],nc,cc,bc,sc);nc=general_control(nc,z,nrv);nc[2]*=gate(z,sub)
 old=hist.prior.prior.component_rhs(z[:497],ctx,nc,cc,bc,sc)
 q,free,ph=fluxes(z,iv,ref,ctx[5],ctx[6],ctx[7],sub,nrv,np.ones(4),cc,ctl,rctl,gate(z,sub))
 old,credit=before_substrates(old,q,free,ref,1.,1.)
 old=append_substrates(z,old,sub,sref,ctx[5],ctx[6],sctl);old[493]-=q[22]
 src=removal_sources(z,old,ctx[0][20]*max(z[6],0)*414*api.BudgetParameters().tdp_uM_per_lineage_unit,ctx[5],ctx[6],ctx[4],ctx[11],ctx[12])
 old=append_recycling(z,old,src,nrv,nref,sub,ctx[6],base[120],rctl,sctl[0],1.)
 return finish(z,old,q,ph,credit,iv)

def check(z,cs):
 ctx,sub,sref,nrv,nref,iv,ref,base=cs
 c=revise_checks(rec_checks(z,base,nrv,sub),z,base,iv,ctx[5])
 zz,bb=adjusted_ribosomes(z,base);c.update(sh_checks(zz,bb,ctx[6],ctx[5],ctx[11],ctx[13]));return c

def unit():
 cs=constants();z=cs[-1];d=component(z,cs,np.array([0.,1.,1.]));active=np.r_[497:501,513:516,532:553]
 assert np.max(abs(d[active]))<1e-12,(active[np.argmax(abs(d[active]))],d[active])
 assert max(check(z[None,:],cs).values())<1e-12
 er=component(z,cs,np.array([1.,1.,1.]));assert er[537]>0 and er[532]==0
 deprived=z.copy();deprived[498]=0
 aa=component(deprived,cs,np.array([0.,1.,1.]));assert aa[532]>0 and aa[537]==0
 for idx in (120,498,499):
  zz=z.copy();zz[idx]=0;dd=component(zz,cs,np.array([1.,1.,1.]));assert abs(dd[553:555]).max()<1e-12
 xx=np.linspace(0,1,101);a=np.array([selective_factor(t,6.,2.) for t in xx]);assert a[0]==0 and np.max(a)>a[-1] and np.argmax(a) not in (0,100)
 bad=replace(SelectiveISRParameters(),ATF4_baseline_uM=1e6)
 try:constants(bad);raise AssertionError('Accepted impossible ribosome demand')
 except ValueError:pass
 save('unit_codex.json',dict(passed=True,baseline_drift=float(abs(d[active]).max()),baseline_ISR_residues_uM_s=cs[6][2],baseline_ISR_ATP_allocation_uM_s=cs[6][3],ATF4_selection_peak=float(a.max()),canonical_lengths=LENGTHS,source='UniProt P18848/O75807; kinetics scenarios'))
 print('ISR unit/mass/energy boundary checks passed',flush=True)

def integrated():
 sg=schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)]);es=np.c_[sg[:,:2],[0,25,0],[0,0,0]]
 opts=dict(**hist.prior.prior.hist.v29.options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
 run=api.run_selective_isr_neuron;ctl=np.c_[sg[:,:2],[[0,1,1],[1,1,1],[0,1,1]]];opts['isr_schedule']=ctl
 r=run(sg,**opts);assert r['state'].shape[1]==len(r['columns'])==562
 a=run(sg[:2],**dict(opts,electrical_schedule=es[:2],isr_schedule=ctl[:2]));c=run(sg[2:],checkpoint=a['checkpoint'],**dict(opts,electrical_schedule=es[2:],isr_schedule=ctl[2:]))
 strict=run(sg,**dict(opts,rtol=3e-10,atol=3e-12,max_step_s=.00125));err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
 restart=err(c['state'][-1],r['state'][-1]);conv=err(strict['state'][-1],r['state'][-1]);assert max(restart,conv)<1e-5
 bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][539]+=1
 try:run(sg[2:],checkpoint=bad,**dict(opts,electrical_schedule=es[2:],isr_schedule=ctl[2:]));raise AssertionError('Corrupt protein balance accepted')
 except ValueError:pass
 save('integrated_codex.json',dict(complete=True,states=562,restart_error=restart,strict_error=conv,checks=r['checks'],config=r['config']))
 np.savez_compressed(B/'integrated_codex.npz',time_s=r['time_s'],state=r['state']);print('562-state integration passed',flush=True)

def electrical():
 metrics=module('isr_electrical_features',B.parent/'2026-10-26-nak-source-calibration/validate_model_codex.py')
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical_codex.json').read_text())['rows'];rows=[]
 for target in targets:
  profile=target['profile'];cur=target['current_pA'];dur=.5 if profile=='Burley' else 1.
  sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  opts=dict(**hist.prior.prior.hist.v29.options(profile),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
  entries={}
  for label,run in [('legacy030',api.run_nuclear_recycling_neuron),('selective031',api.run_selective_isr_neuron)]:
   r=run(sg,**opts);f=metrics.features(r,.2,.2+dur,profile);assert f['spikes']==target['target']
   delta=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.02
   entries[label]=dict(features=f,max_change=delta)
  rows.append(dict(profile=profile,current_pA=cur,branches=entries));save('electrical_codex.json',dict(complete=len(rows)==9,rows=rows));print(profile,cur,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['unit','integrated','electrical','all'],required=True);a=p.parse_args()
 if a.mode=='all':unit();integrated();electrical()
 else:globals()[a.mode]()
