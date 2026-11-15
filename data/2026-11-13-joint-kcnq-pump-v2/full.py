"""[codex] Full 4425-state referee and no-reset long protocols.

Historical electrical counts are regression targets, not nine independent
human observations. Candidate selection never changes the spike detector.
"""
from pathlib import Path
import sys,json,numpy as np
import capture as c
from options import options
from candidate_parameters import params
B=Path(__file__).resolve().parent
r=c.old.r;r.B=B;r.PREFIX='J13_';r.SPARSE=True

def capture(tag,sg,o,**kw):
    result=r.run(tag,sg,o,retain_dense_solver_output=False,center_solver_coordinates=True,**kw)
    z=result['state'];kb=c.old.ledger.observe(z,result['config'])
    info=dict(parameters=o['absolute_kcnq_parameters'].__dict__ if 'absolute_kcnq_parameters' in o else {},
      selected={k:dict(initial=float(z[0,i]),final=float(z[-1,i]),min=float(z[:,i].min()),max=float(z[:,i].max())) for k,i in c.old.KEYS.items()},
      all_finite=bool(np.isfinite(z).all()),states=z.shape[1],
      K_ledger_max_residual=max(float(abs(x['residual']).max()) for x in kb),
      K_terms=[{k:float(v[-1]) for k,v in x['terms'].items()} for x in kb],
      checks={k:float(v) for k,v in result['checks'].items() if k!='minimum_state'},
      diagnostic_interventions=result['checkpoint']['diagnostic_interventions'])
    r.save('J13_'+tag+'_metrics',info)
    return result

def electric(name,profile,amp,cp=None,suffix=''):
    tag=f'electric_{name}_{profile}_{amp:g}'+suffix
    if (B/('J13_'+tag+'_features.json')).exists():return
    start=0 if cp is None else cp['time_s'];on=start+.2;off=on+(.5 if profile=='Burley' else 1.)
    sg=r.schedule([(start,on,0,1,0),(on,off,0,1,0),(off,off+1,0,1,0)])
    o,rec=options(sg,profile,params(name))
    z=capture(tag,sg,o,checkpoint=cp,electrical_schedule=np.c_[sg[:,:2],[0,amp,0],[0,0,0]],
      output_step_s=.0002,max_step_s=.0025,rtol=1e-8,atol=1e-10)
    f=r.features_mod.features(z,on,off,profile)
    r.save('J13_'+tag+'_features',dict(profile=profile,current_pA=amp,features=f,parameters=params(name),
      start_s=start,no_reset=cp is None or np.array_equal(z['state'][0],cp['state']),
      historical_not_human_target=True,conditions=rec))

def screen(name):
    for active in (False,True):
        tag='screen_'+name+('_load' if active else '_rest')
        if (B/('J13_'+tag+'_metrics.json')).exists():continue
        sg=r.schedule([(0,10,0,1,0),(10,70,.001 if active else 0,1,0),(70,130,0,1,0),
           (130,190,.001 if active else 0,1,0),(190,600,0,1,0)])
        o,rec=options(sg,'Selvaraj-Con2',params(name))
        capture(tag,sg,o,output_step_s=2,max_step_s=10,rtol=1e-7,atol=c.old.legacy.long_atol())
        r.save('J13_'+tag+'_protocol',dict(parameters=params(name),schedule=sg,conditions=rec,reset=False))

def rest(name,hours=72.):
    tag=f'rest_{name}_{hours:g}h'
    if (B/('J13_'+tag+'_metrics.json')).exists():return
    end=hours*3600;edges=np.unique(np.r_[0,.01,.1,1,10,60,600,3600,np.arange(21600,end,21600),end])
    sg=r.schedule([(a,b,0,1,0) for a,b in zip(edges[:-1],edges[1:])]);o,_=options(sg,'Selvaraj-Con2',params(name))
    capture(tag,sg,o,output_step_s=300,max_step_s=300,rtol=1e-7,atol=c.old.legacy.long_atol())

def loads(name,branch,opening=.001):
    cpfile=B/f'J13_rest_{name}_72h_checkpoint.json';cp=json.loads(cpfile.read_text());start=cp['time_s']
    tag=f'load36_{name}_{branch}_{opening:g}'
    if (B/('J13_'+tag+'_metrics.json')).exists():return
    sg=r.schedule([(start+h*3600,start+(h+1)*3600,opening if ((branch=='continuous' and h<6) or
      (branch=='repeated' and h in (0,4,8,12,16,20))) else 0,1,0) for h in range(36)])
    o,_=options(sg,'Selvaraj-Con2',params(name))
    z=capture(tag,sg,o,checkpoint=cp,output_step_s=300,max_step_s=300,rtol=1e-7,atol=c.old.legacy.long_atol())
    assert np.array_equal(z['state'][0],cp['state']) and z['checkpoint']['origin_time_s']==cp['origin_time_s']
    r.save('J13_'+tag+'_protocol',dict(checkpoint=str(cpfile),schedule=sg,active_hours=0 if branch=='control' else 6,
      all_states_preserved=True,equal_opening_not_equal_calcium=True,reset=False))

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='electric':electric(sys.argv[2],sys.argv[3],float(sys.argv[4]))
    elif cmd=='suite':
        for amp in (15,25,50):electric(sys.argv[2],sys.argv[3],amp)
    elif cmd=='screen':screen(sys.argv[2])
    elif cmd=='screen_many':
        for name in sys.argv[2:]:screen(name)
    elif cmd=='rest':rest(sys.argv[2])
    elif cmd=='loads':loads(sys.argv[2],sys.argv[3],float(sys.argv[4]))
    elif cmd=='post':
        cp=json.loads((B/sys.argv[3]).read_text());electric(sys.argv[2],'Selvaraj-Con2',25,cp,'_'+sys.argv[4])
