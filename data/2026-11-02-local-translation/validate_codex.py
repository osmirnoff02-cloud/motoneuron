"""[codex]2026-11-02. Conservation, resource depletion/recovery, electrical regressions."""
from pathlib import Path
import sys,json,importlib.util,argparse
from dataclasses import replace
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
_threads=threadpool_limits(1);B=Path(__file__).resolve().parent
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
hist=module('translation_hist',B.parent/'2026-10-31-local-ribosomes/validate_codex.py');api=hist.api
fusco=hist.prior;isr=fusco.prior
from local_ribosomes_codex import extension as ribo_extension,adjusted as ribo_adjust,checks as ribo_checks,LocalRibosomeParameters
from connected_parameters_codex import schedule
from local_translation_codex import *
def save(n,x):(B/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a))+'\n',encoding='utf8')

def constants(par=None):
 cs=isr.constants();base=np.r_[cs[-1],np.zeros(12)];base[136:138]=base[120:122]
 v=(par or LocalTranslationParameters()).vector();base=initialize(base,v)
 return cs,base,v,LocalRibosomeParameters().vector(),cs[0][5][0],cs[0][6][8]

def all_checks(z,base,cs,v,rv,C,ratio):
 az,ab=adjusted_particles(z,base);az,ab=ribo_adjust(az,ab)
 cs2=(*cs[:-1],ab[:562]);ch=isr.check(az[:,:562],cs2)
 ch=revise_somatic_checks(ch,z,base,ratio)
 zz,bb=adjusted_particles(z,base);ch.update(ribo_checks(zz,bb,rv,C,ratio));ch.update(checks(z,base,v,C,ratio));return ch

def unit():
 cs,z,v,rv,C,K=constants(replace(LocalTranslationParameters(),initial_local_fraction=.05));ratio=4.
 assert abs((z[330]+z[564])-cs[-1][330])<1e-12
 old=np.zeros(574);d=extension(z,old,v,np.ones(6),C,ratio,K)
 assert abs(d[564]+d[578])<1e-12 and abs(d[565]+d[578])<1e-12
 assert d[578]>0 and d[583]==0 and d[588]==0 # load does not create a completed protein
 assert abs(d[136]+d[137])<1e-12 and abs(d[136]+d[150])<1e-12
 for s,p,k in [(497,574,599),(499,576,601),(500,577,603)]:
  q=z.copy();q[p]=q[s]/2;dd=extension(q,old,v,np.ones(6),C,ratio,K)
  # Net exchange is independently recorded and the soma receives its opposite.
  assert abs(dd[s]+(dd[k]-dd[k+1])/ratio)<1e-12
 zz=z.copy();zz[578:582]=.0001;zz[583]=.001;zz[584]=.1
 d=extension(zz,old,v,np.ones(6),C,ratio,K)
 h=1e-5;traj=np.array([zz,zz+h*d]);ch=checks(traj,zz,v,C,ratio);assert max(ch.values())<1e-10,ch
 ob=observe(traj,C,ratio)
 assert np.allclose(ob['total_occupied_ribosomes_uM'],ob['busy_ribosomes_uM']+ob['posttermination_ribosomes_uM'],rtol=0,atol=1e-15)
 assert np.allclose(ob['posttermination_ribosomes_uM'],traj[:,582]*C*ratio,rtol=0,atol=1e-15)
 for idx in [574,575,576,577,136]:
  q=zz.copy();q[idx]=0;dd=extension(q,old,v,np.ones(6),C,ratio,K);assert dd[idx]>=-1e-12
  if idx==575:assert dd[588]==0
  if idx==576:assert dd[588]==0 and dd[585]==0
  if idx==136:assert dd[589]==0 and dd[587]==0 and dd[588]>0 # stored Q/G can still fuel elongation
 off=extension(z,old,v,np.zeros(6),C,ratio,K);assert np.all(off==0)
 for p in [replace(LocalTranslationParameters(),length_aa=0),replace(LocalTranslationParameters(),initial_local_fraction=.8),replace(LocalTranslationParameters(),NDPK_rate_per_uM_s=-1)]:
  try:p.vector();raise AssertionError('Invalid parameters accepted')
  except ValueError:pass
 save('unit_codex.json',dict(passed=True,particle_redistribution=True,load_not_completion=True,AA_GTP_GDP_amount_conserved=True,ATP_debited_once=True,stored_energy_can_finish_after_ATP_loss=True,off_exact=True))
 print('Unit605 passed',flush=True)

def full():
 sg=schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)])
 es=np.c_[sg[:,:2],[0,25,0],[0,0,0]];ctl=np.c_[sg[:,:2],np.ones((3,6))];ctl[1,3]=.25
 par=replace(LocalTranslationParameters(),initial_local_fraction=.05)
 opts=dict(**hist.options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11,local_translation_parameters=par)
 run=api.run_local_translation_neuron;r=run(sg,local_translation_schedule=ctl,**opts)
 assert r['state'].shape[1]==len(r['columns'])==605 and r['state'][-1,586]>0
 a=run(sg[:2],local_translation_schedule=ctl[:2],**dict(opts,electrical_schedule=es[:2]));c=run(sg[2:],checkpoint=a['checkpoint'],local_translation_schedule=ctl[2:],**dict(opts,electrical_schedule=es[2:]))
 strict=run(sg,local_translation_schedule=ctl,**dict(opts,rtol=3e-10,atol=3e-12,max_step_s=.00125))
 err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
 restart=err(c['state'][-1],r['state'][-1]);conv=err(strict['state'][-1],r['state'][-1]);assert max(restart,conv)<1e-5
 bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][583]+=.1
 try:run(sg[2:],checkpoint=bad,local_translation_schedule=ctl[2:],**dict(opts,electrical_schedule=es[2:]));raise AssertionError('Corrupt protein accepted')
 except ValueError:pass
 save('integrated_codex.json',dict(passed=True,states=605,restart_error=restart,strict_error=conv,checks=r['checks'],config=r['config'],completed_local_protein_uM=float(r['local_translation_observation']['completed_protein_uM'][-1]),initial_local_fraction=.05))
 print('Integrated605 passed',flush=True)
 targets=json.loads((B.parent/'2026-10-29-joint-recovery/electrical_codex.json').read_text())['rows']
 metrics=module('translation_features',B.parent/'2026-10-26-nak-source-calibration/validate_model_codex.py');rows=[]
 for target in targets:
  p=target['profile'];cur=target['current_pA'];dur=.5 if p=='Burley' else 1.
  sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
  opts=dict(**hist.options(p),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11,local_translation_parameters=par)
  r=run(sg,**opts);f=metrics.features(r,.2,.2+dur,p);assert f['spikes']==target['target']
  delta=max(abs(float(f[k])-float(x)) for k,x in target['features'].items() if isinstance(x,(int,float)) and np.isfinite(x) and np.isfinite(f[k]));assert delta<.02
  rows.append(dict(profile=p,current_pA=cur,features=f,max_change=delta,completed_local_protein_uM=float(r['local_translation_observation']['completed_protein_uM'][-1])))
  save('electrical_codex.json',dict(complete=len(rows)==9,initial_local_fraction=.05,rows=rows));print(p,cur,delta,flush=True)

# Explicitly assumed long-assay energy settings; not measured mitochondrial kinetics.
ASSAY_RANGES={'process_background_ATP_uM_s':[.1,.5,1.], 'ADP_supply_reserve':[1.5,2.,3.], 'volume_ratio':[1.,4.,16.]}

def molecular(slots,mode,strict=False):
 par=replace(LocalTranslationParameters(),initial_local_fraction=.05,message_slots_uM=slots,AA0_uM=100.,GTP0_uM=100.,GDP0_uM=50.)
 cs,base,v,rv,C,K=constants(par);ratio=4.;base[497]=100. # choose matched finite somaticAA too; no refit of existing kinetics
 idx=np.unique(np.r_[fusco.INDEX,np.array([136,137,148,150]),np.arange(562,605)])
 def component(z,ctl):
  d=ribo_extension(z,fusco.component(z[:562],cs,np.ones(2)),rv,np.ones(3),C,ratio,base[120],base[136],K)
  return extension(z,d,v,ctl,C,ratio,K)
 def costs(z,d):
  bg=.5*max(z[136],0)/(K+max(z[136],0))*(K+base[136])/base[136]
  return fusco.demand(d)+d[572], bg+d[573]+d[594]+d[595]+d[598]+d[592]-d[593]
 d0=component(base,np.ones(6));ref=np.array(costs(base,d0))
 state=base[idx].copy();tt=[0.];yy=[state.copy()];dd=[d0]
 for l,r in zip([0,2,8,12,18],[2,8,12,18,24]):
  ctl=np.ones(6);support=1.
  if l in (2,12):
   if mode=='AA_exchange_off':ctl[2]=0
   elif mode=='GTP_regeneration_off':ctl[[3,5]]=0
   elif mode=='ATP_supply_quarter':support=.25
  def rhs(t,y):
   z=base.copy();z[idx]=y;d=component(z,ctl);cs_,cp_=costs(z,d)
   for ai,di,pi,ci,cost,rate,sup in [(120,121,123,125,cs_,ref[0],1.),(136,137,148,150,cp_,ref[1],support)]:
    adp=max(z[di],0);prod=rate*2*adp/(base[di]+adp)*sup
    d[ai]=prod-cost;d[di]=cost-prod;d[pi]=prod;d[ci]=cost
   return d[idx]
  start=l*3600;end=r*3600;grid=np.r_[np.arange(start+120,end,120),end]
  sol=solve_ivp(rhs,(start,end),state,method='LSODA',t_eval=grid,rtol=2e-10 if strict else 1e-8,atol=2e-12 if strict else 1e-10,max_step=30 if strict else 180)
  assert sol.success;state=sol.y[:,-1];tt.extend(sol.t);yy.extend(sol.y.T)
  for y in sol.y.T:
   z=base.copy();z[idx]=y;dd.append(component(z,ctl))
 t=np.array(tt);z=np.tile(base,(len(t),1));z[:,idx]=np.array(yy);d=np.array(dd)
 ch=all_checks(z,base,cs,v,rv,C,ratio)
 for ai,di,pi,ci,label in [(120,121,123,125,'soma'),(136,137,148,150,'process')]:
  ch[label+'_ATP_balance']=float(abs(z[:,ai]-base[ai]-z[:,pi]+z[:,ci]).max())
  ch[label+'_adenylate_balance']=float(abs(z[:,ai]+z[:,di]-base[ai]-base[di]).max())
 assert max(ch.values())<3e-5 and z[:,idx].min()>-1e-7,ch
 at=lambda h:int(np.argmin(abs(t-h*3600)))
 out=dict(message_slots_uM=slots,mode=mode,checks=ch,initial_parameters=asdict(par),reference_source_uM_s=ref,
  protein24h_uM=float(z[-1,583]),completed24h_uM=float(z[-1,586]*C*ratio),incorporated_AA_uM=float(z[-1,588]),
  ATP_min_uM=float(z[:,136].min()),GTP_min_uM=float(z[:,576].min()),AA_min_uM=float(z[:,574].min()),
  rate_before_first_block_uM_s=float(d[at(2),586]*C*ratio),rate_end_first_block_uM_s=float(d[at(8),586]*C*ratio),rate_final_uM_s=float(d[-1,586]*C*ratio),
  net_local_translation_ATP_uM=float(z[-1,594]+z[-1,595]+z[-1,598]+z[-1,592]-z[-1,593]))
 return t,z,d,out

def scenarios():
 rows=[];selected={}
 for slots in [.0001,.001,.01]:
  for mode in ['control','AA_exchange_off','GTP_regeneration_off','ATP_supply_quarter']:
   t,z,d,r=molecular(slots,mode);rows.append(r);print(slots,mode,r['completed24h_uM'],r['ATP_min_uM'],flush=True)
   if slots==.01:
    selected[mode]=(t,z,d);np.savez_compressed(B/f'{mode}_24h_codex.npz',time_s=t,state=z,completed_rate_uM_s=d[:,586]*.4)
 t,z,d,r=molecular(.01,'GTP_regeneration_off',True);err=float(np.max(abs(z-selected['GTP_regeneration_off'][1])/(1+abs(z))));assert err<3e-5
 save('scenarios_codex.json',dict(complete=True,runs=12,strict_error=err,protocol='2-8h and12-18h, no reset',scope='Selected two-compartment molecular subsystem; dynamic ATP, effective ADP-controlled supplies. No full electrical24h or axial ATP exchange in this reduced assay.',volume_ratio=4,rows=rows))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,3,figsize=(12,6))
 for mode,(t,z,d) in selected.items():
  for a,y in zip(axs.flat,[z[:,583],d[:,586]*.4,z[:,136],z[:,574],z[:,576],z[:,578:583].sum(1)*.4]):a.plot(t/3600,y,label=mode.replace('_',' '))
 for a,title in zip(axs.flat,['Local protein (uM)','Completion rate (uM/s)','Process ATP (uM)','Local free AA (uM)','Local GTP (uM)','Occupied ribosomes (uM)']):
  a.set_title(title);a.set_xlabel('Time (h)');a.grid(alpha=.2)
  for l,r in [(2,8),(12,18)]:a.axvspan(l,r,color='grey',alpha=.15)
 axs[0,0].legend(fontsize=7);fig.suptitle('Conditional local synthesis; unknown human rates, message slots0.01uM');fig.tight_layout();fig.savefig(B/'local_translation_codex.png',dpi=130);plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['unit','full','scenarios','all'],required=True);args=p.parse_args()
 if args.mode=='all':unit();full();scenarios()
 else:globals()[args.mode]()
