"""Codex: factorial GAP43 mRNA supply screen with shared finite resources."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
B=Path(__file__).resolve().parent;D=B.parent
sys.path.insert(0,str(D/'neuron-model'))
from model_contracts import load_module
prior=load_module(D/'2026-11-14-prepared-loads-gap43/run.py','_gcc14_prior')
r=prior.r;r.B=B;r.PREFIX='GCC14_';r.SPARSE=True
read=lambda p:json.loads(Path(p).read_bytes())
CP=Path(read(B/'protocol.json')['checkpoint'])

def options(sg,p=1.,d=1.,turn='h6',profile='Burley'):
 o=prior.options(sg,turn,profile)
 o['local_mrna_schedule']=np.c_[sg[:,:2],np.tile([p,d,d,1.],(len(sg),1))]
 return o

def load(p,d,branch,turn='h6',tight=False):
 tag=f'{turn}_P{p:g}_D{d:g}_{branch}'+('_tight' if tight else '')
 if (B/f'{tag}_proof.json').exists():return
 cp=read(CP);sg=prior.schedule(branch,cp['time_s'])
 out=r.run(tag,sg,options(sg,p,d,turn),checkpoint=cp,retain_dense_solver_output=False,
  center_solver_coordinates=True,output_step_s=300,max_step_s=150 if tight else 300,
  rtol=3e-8 if tight else 1e-7,atol=prior.prior.a.atol()*(.3 if tight else 1))
 assert np.array_equal(out['state'][0],cp['state'])
 assert np.array_equal(out['baseline_state'],cp['baseline_state'])
 assert out['checkpoint']['fingerprint']==cp['fingerprint']
 assert out['checkpoint']['origin_time_s']==cp['origin_time_s']
 r.save(tag+'_proof',dict(no_reset=True,initial_states=4431,production=p,delivery=d,turnover=turn,
    checkpoint=str(CP),sha256=hashlib.sha256(CP.read_bytes()).hexdigest(),hours=72,
    schedule=sg,mRNA_schedule=options(sg,p,d,turn)['local_mrna_schedule']))

def electric(profile,p=1.,d=1.,cpfile=None,turn='h24',amps=(15,25,50)):
 cp=None if cpfile is None else read(Path(cpfile))
 label='initial' if cp is None else f'post{cp["time_s"]/3600:g}h_P{p:g}_D{d:g}'
 for amp in amps:
  tag=f'electric_{label}_{profile}_{amp}'
  if (B/f'{tag}_features.json').exists():continue
  start=0 if cp is None else cp['time_s'];on=start+.2;off=on+(.5 if profile=='Burley' else 1.)
  sg=r.schedule([(start,on,0,1,0),(on,off,0,1,0),(off,off+1,0,1,0)])
  z=r.run(tag,sg,options(sg,p,d,turn,profile),checkpoint=cp,retain_dense_solver_output=False,
    center_solver_coordinates=True,electrical_schedule=np.c_[sg[:,:2],[0,amp,0],[0,0,0]],
    output_step_s=.0002,max_step_s=.0025,rtol=1e-8,atol=1e-10)
  r.save(tag+'_features',dict(profile=profile,current_pA=amp,features=r.features_mod.features(z,on,off,profile),
    no_reset=cp is None or np.array_equal(z['state'][0],cp['state']),start_s=start,production=p,delivery=d,
    historical_regression_not_human_target=True))

if __name__=='__main__':
 if sys.argv[1]=='one':load(float(sys.argv[2]),float(sys.argv[3]),sys.argv[4],sys.argv[5] if len(sys.argv)>5 else 'h6',len(sys.argv)>6)
 elif sys.argv[1]=='electric':electric(sys.argv[2])
 elif sys.argv[1]=='post':electric('Burley',4,4,B/'GCC14_h6_P4_D4_repeated_checkpoint.json','h6',(25,))
 elif sys.argv[1]=='grid':
  for p in (1,2,4):
   for d in (1,2,4):
    for b in ('control','repeated'):load(p,d,b)
