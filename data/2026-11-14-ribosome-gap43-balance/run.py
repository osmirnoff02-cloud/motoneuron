"""Codex 2026-11-14. Bounded baseline continuation with all states/ledgers retained."""
from pathlib import Path
import sys,json,numpy as np
B=Path(__file__).resolve().parent;D=B.parent
sys.path.insert(0,str(D/'neuron-model'))
from model_contracts import load_module
a=load_module(D/'2026-11-14-acid-coupling/run.py','_rgb14_acid')
r=a.r;r.B=B;r.PREFIX='RGB14_';r.SPARSE=True
read=lambda p:json.loads(p.read_text(encoding='utf8'))
ACCEPTED=read(D/'2026-11-14-phase-aware-waveform/accepted.json')

def options(sg,profile):
    o,c=a.options(sg,profile);o['profile_p']=np.array(ACCEPTED['profiles'][profile]['parameters'])
    o['control_context']=c
    return o

def continuation(end_h=240.):
    cpfile=D/'2026-11-14-phase-aware-waveform/P14_load_Burley_control_checkpoint.json'
    cp=read(cpfile);start=cp['time_s'];end=end_h*3600
    edges=np.r_[np.arange(start,end,21600),end]
    sg=r.schedule([(x,y,0,1,0) for x,y in zip(edges[:-1],edges[1:])]);o=options(sg,'Burley')
    out=r.run(f'rest_Burley_108to{end_h:g}h',sg,o,checkpoint=cp,retain_dense_solver_output=False,
        center_solver_coordinates=True,output_step_s=600,max_step_s=300,rtol=1e-7,atol=a.atol())
    assert np.array_equal(out['state'][0],cp['state'])
    r.save('continuation_proof',dict(origin=str(cpfile),end_h=end_h,reset=False,states=out['state'].shape[1],
        state_schema=out['state_schema'],checks=out['checks'],parameter_change=False,human_calibrated=False))

def electric(profile,cpfile=None,amps=(15,25,50)):
    cp=None if cpfile is None else read(Path(cpfile))
    label='initial' if cp is None else f"post{cp['time_s']/3600:g}h"
    for amp in amps:
        start=0 if cp is None else cp['time_s'];on=start+.2;off=on+(.5 if profile=='Burley' else 1.)
        sg=r.schedule([(start,on,0,1,0),(on,off,0,1,0),(off,off+1,0,1,0)])
        tag=f'electric_{label}_{profile}_{amp:g}'
        out=r.run(tag,sg,options(sg,profile),checkpoint=cp,retain_dense_solver_output=False,
          center_solver_coordinates=True,electrical_schedule=np.c_[sg[:,:2],[0,amp,0],[0,0,0]],
          output_step_s=.0002,max_step_s=.0025,rtol=1e-8,atol=1e-10)
        r.save(tag+'_features',dict(profile=profile,current_pA=amp,
          features=r.features_mod.features(out,on,off,profile),
          start_s=start,checkpoint=None if cpfile is None else str(cpfile),
          no_reset=cp is None or np.array_equal(out['state'][0],cp['state']),
          historical_regression_not_human_target=True))

def validate(branch):
    cpfile=B/'RGB14_rest_Burley_108to240h_checkpoint.json';cp=read(cpfile);start=cp['time_s']
    sg=r.schedule([(start+h*3600,start+(h+1)*3600,
       .001 if branch=='repeated' and h in (0,4,8,12,16,20) else 0,1,0) for h in range(72)])
    out=r.run('validation_'+branch,sg,options(sg,'Burley'),checkpoint=cp,
        retain_dense_solver_output=False,center_solver_coordinates=True,
        output_step_s=600,max_step_s=300,rtol=1e-7,atol=a.atol())
    assert np.array_equal(out['state'][0],cp['state'])
    assert out['checkpoint']['origin_time_s']==cp['origin_time_s']
    r.save('validation_'+branch+'_proof',dict(checkpoint=str(cpfile),no_reset=True,
       schedule=sg,checks=out['checks'],states=out['state'].shape[1],parameter_change=False))

def extend_baseline():
    """Predeclared final preparation at600h, independent600-672h observation.
    No further search if this fails; preserve the measured residual drift.
    """
    cpfile=B/'RGB14_validation_control_checkpoint.json';cp=read(cpfile)
    for end_h,tag in ((600,'rest_Burley_312to600h'),(672,'validation_late_control')):
        start=cp['time_s'];end=end_h*3600;edges=np.r_[np.arange(start,end,21600),end]
        sg=r.schedule([(x,y,0,1,0) for x,y in zip(edges[:-1],edges[1:])])
        out=r.run(tag,sg,options(sg,'Burley'),checkpoint=cp,retain_dense_solver_output=False,
          center_solver_coordinates=True,output_step_s=600,max_step_s=300,rtol=1e-7,atol=a.atol())
        assert np.array_equal(out['state'][0],cp['state'])
        assert out['checkpoint']['origin_time_s']==cp['origin_time_s']
        cp=out['checkpoint']

if __name__=='__main__':
    cmd=sys.argv[1] if len(sys.argv)>1 else '240'
    if cmd=='electric':electric(sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
    elif cmd=='electric-one':electric(sys.argv[2],sys.argv[3],(float(sys.argv[4]),))
    elif cmd=='validate':validate(sys.argv[2])
    elif cmd=='extend':extend_baseline()
    else:continuation(float(cmd))
