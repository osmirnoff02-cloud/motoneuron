"""Codex. Full no-reset 600h branches; paired turnover interventions."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
B=Path(__file__).resolve().parent;D=B.parent;OLD=D/'2026-11-14-ribosome-gap43-balance'
sys.path.insert(0,str(D/'neuron-model'))
from model_contracts import load_module
prior=load_module(OLD/'run.py','_plg14_previous')
r=prior.r;r.B=B;r.PREFIX='PLG14_';r.SPARSE=True
read=lambda p:json.loads(p.read_bytes())
CP=OLD/'RGB14_rest_Burley_312to600h_checkpoint.json'
SCENARIOS=read(B/'protocol.json')['turnover']

def options(sg,scenario='h24',profile='Burley'):
 o=prior.options(sg,profile)
 o['local_GAP43_turnover_multiplier']=SCENARIOS[scenario]['multiplier']
 return o

def schedule(branch,start,hours=72):
 edges=set(np.arange(0,hours+0.1,6.).tolist()+[float(hours)])
 changes=[0,6] if branch=='continuous' else [x for h in (0,4,8,12,16,20) for x in (h,h+1)] if branch=='repeated' else [0]
 for h in changes:
  edges.update(h+x/3600 for x in (0,1,10,60) if h+x/3600<hours)
 edges=sorted(edges)
 rows=[]
 for lo,hi in zip(edges[:-1],edges[1:]):
  active=(branch=='continuous' and lo<6) or (branch=='repeated' and any(h<=lo<h+1 for h in (0,4,8,12,16,20)))
  rows.append((start+lo*3600,start+hi*3600,.001 if active else 0,1,0))
 return r.schedule(rows)

def load(scenario,branch,tight=False):
 tag=f'{scenario}_{branch}'+('_tight' if tight else '')
 if (B/f'{tag}_proof.json').exists():print('Already completed',tag,flush=True);return
 cp=read(CP);sg=schedule(branch,cp['time_s']);o=options(sg,scenario)
 result=r.run(tag,sg,o,checkpoint=cp,retain_dense_solver_output=False,center_solver_coordinates=True,
  output_step_s=300,max_step_s=150 if tight else 300,rtol=3e-8 if tight else 1e-7,
  atol=prior.a.atol()*(.3 if tight else 1))
 assert np.array_equal(result['state'][0],cp['state'])
 assert result['checkpoint']['origin_time_s']==cp['origin_time_s']
 assert np.array_equal(result['baseline_state'],cp['baseline_state'])
 r.save(tag+'_proof',dict(scenario=SCENARIOS[scenario],checkpoint=str(CP),
  checkpoint_sha256=hashlib.sha256(CP.read_bytes()).hexdigest(),all_states_preserved=True,
  fingerprint_unchanged=result['checkpoint']['fingerprint']==cp['fingerprint'],
  schedule=sg,AMPA_opening_integral_s=float(np.sum((sg[:,1]-sg[:,0])*sg[:,2])),
  equal_opening_not_equal_calcium=True,diagnostic_interventions=result['checkpoint']['diagnostic_interventions']))

def electric(profile,scenario='h24',cpfile=None):
 cp=None if cpfile is None else read(Path(cpfile));label='initial' if cp is None else f"post{cp['time_s']/3600:g}h_{scenario}_{Path(cpfile).stem}"
 for amp in (15,25,50):
  tag=f'electric_{label}_{profile}_{amp}'
  if (B/f'{tag}_features.json').exists():continue
  start=0 if cp is None else cp['time_s'];on=start+.2;off=on+(.5 if profile=='Burley' else 1.)
  sg=r.schedule([(start,on,0,1,0),(on,off,0,1,0),(off,off+1,0,1,0)])
  z=r.run(tag,sg,options(sg,scenario,profile),checkpoint=cp,retain_dense_solver_output=False,
   center_solver_coordinates=True,electrical_schedule=np.c_[sg[:,:2],[0,amp,0],[0,0,0]],
   output_step_s=.0002,max_step_s=.0025,rtol=1e-8,atol=1e-10)
  r.save(tag+'_features',dict(profile=profile,scenario=scenario,current_pA=amp,
   features=r.features_mod.features(z,on,off,profile),no_reset=cp is None or np.array_equal(z['state'][0],cp['state']),
   start_s=start,historical_regression_not_human_target=True))

if __name__=='__main__':
 if sys.argv[1]=='load':load(sys.argv[2],sys.argv[3],len(sys.argv)>4 and sys.argv[4]=='tight')
 elif sys.argv[1]=='batch':
  for branch in ('control','continuous','repeated'):load(sys.argv[2],branch)
 elif sys.argv[1]=='electric':electric(sys.argv[2])
 elif sys.argv[1]=='post':electric('Burley',sys.argv[2],sys.argv[3])
