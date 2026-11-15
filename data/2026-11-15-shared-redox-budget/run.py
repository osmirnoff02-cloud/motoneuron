"""Codex. Shared resource full-state runs; migration is explicit, never a reset."""
from pathlib import Path
import sys,json,inspect,hashlib,time
import numpy as np
B=Path(__file__).resolve().parent;D=B.parent
sys.path.insert(0,str(D/'neuron-model'))
from model_contracts import load_module
previous=load_module(D/'2026-11-14-gap43-compensation/run.py','_shared15_previous')
shared=load_module(B/'shared_budget.py','_stable_shared_redox15')
host=load_module(D/'2026-10-28-takeuchi-balaji/adaptive_recycling_neuron.py','_shared15_host')
r=previous.r;r.B=B;r.PREFIX='SR15_';r.SPARSE=True
read=lambda p:json.loads(Path(p).read_bytes())
def options(sg,profile='Burley',production=4.,delivery=4.,fraction=1.,enabled=True):
    o=previous.options(sg,production,delivery,'h6',profile)
    o['shared_redox_parameters']=shared.SharedRedoxParameters(active_water_fraction=fraction,coupling_enabled=enabled)
    return o

class Capture(Exception):pass
def migrate(sg,o,old):
    box={};saved=host.solve_ivp
    def stop(*args,**kw):
        f=inspect.currentframe().f_back.f_locals
        box.update({k:f[k] for k in ('base','fingerprint','schema','config')});raise Capture()
    host.solve_ivp=stop
    try:r.api.run_integrated_neuron(sg,**o,atol=1e-9,retain_dense_solver_output=False)
    except Capture:pass
    finally:host.solve_ivp=saved
    assert np.array_equal(box['base'][:len(old['state'])],old['baseline_state'])
    cp=dict(old);cp['state']=old['state']+box['base'][len(old['state']):].tolist();cp['baseline_state']=box['base'].tolist();cp['fingerprint']=box['fingerprint'];cp['state_schema']=box['schema']
    r.save('migration_f'+str(o['shared_redox_parameters'].active_water_fraction),dict(old_states=len(old['state']),new_states=len(cp['state']),old_state_exact=True,old_baseline_exact=True,new_redox_history='Declared pools at connection; no fabricated 600h history',origin_preserved=True,parameters=o['shared_redox_parameters'].__dict__))
    return cp

def pilot():
    sg=r.schedule([(0,.02,0,1,0)])
    o=options(sg,enabled=False);off=r.run('pilot_zero',sg,o,output_step_s=.005,max_step_s=.002,rtol=1e-8,atol=1e-10,retain_dense_solver_output=False)
    bare=previous.options(sg,4,4,'h6','Burley');old=r.run('pilot_legacy',sg,bare,output_step_s=.005,max_step_s=.002,rtol=1e-8,atol=1e-10,retain_dense_solver_output=False)
    assert np.max(abs(off['state'][:,:4431]-old['state']))<1e-6
    on=r.run('pilot_on',sg,options(sg),output_step_s=.005,max_step_s=.002,rtol=1e-8,atol=1e-10,retain_dense_solver_output=False)
    q=on['config']['shared_redox']['peroxide_reference'];expected=q[shared.P.Q_V0]*.02
    actual=on['state'][-1,4431+33]-on['state'][0,4431+33]
    assert abs(actual-expected)<1e-10,(actual,expected)
    o=options(sg);o['shared_redox_schedule']=np.c_[sg[:,:2],[[1.,3.,1.]]]
    tri=r.run('pilot_tripled',sg,o,output_step_s=.005,max_step_s=.002,rtol=1e-8,atol=1e-10,retain_dense_solver_output=False)
    tripled=tri['state'][-1,4431+33]-tri['state'][0,4431+33]
    assert abs(tripled-3*expected)<1e-10,(tripled,expected)
    r.save('pilot_checks',dict(zero_max_abs=float(np.max(abs(off['state'][:,:4431]-old['state']))),new_states=on['state'].shape[1],checks=on['checks'],supplied_H2O2=actual,expected_H2O2=expected,tripled_H2O2=tripled,controls_reach_RHS=True))

def load(branch='control',fraction=1.,oxidant=1.,nadph=1.,hours=72.,production=4.,tight=False):
    tag=f'{branch}_f{fraction:g}_ox{oxidant:g}_n{nadph:g}_p{production:g}_{hours:g}h'+('_tight' if tight else '')
    if (B/f'{tag}_done.json').exists():return
    old=read(previous.CP);sg=previous.prior.schedule(branch,old['time_s'],hours)
    o=options(sg,production=production,fraction=fraction)
    # Peroxide challenge follows the same active windows; control remains1.
    o['shared_redox_schedule']=np.c_[sg[:,:2],np.ones(len(sg)),np.where(sg[:,2]>0,oxidant,1.),np.full(len(sg),nadph)]
    cp=migrate(sg,o,old)
    atol=np.r_[previous.prior.prior.a.atol(),np.full(shared.NS,1e-9)]
    atol[4431+27]=1e-12 # subnanomolar H2O2
    out=r.run(tag,sg,o,checkpoint=cp,retain_dense_solver_output=False,center_solver_coordinates=True,output_step_s=600,max_step_s=150 if tight else 300,rtol=3e-8 if tight else 1e-7,atol=atol*(.3 if tight else 1))
    assert np.array_equal(out['state'][0],cp['state']);assert out['checkpoint']['origin_time_s']==cp['origin_time_s']
    r.save(tag+'_done',dict(no_reset=True,states=out['state'].shape[1],hours=hours,fraction=fraction,oxidant=oxidant,nadph=nadph,production=production,checks=out['checks'],redox=out['shared_redox_observation'],config=out['config']['shared_redox']))

def electric(profile,amps=(15,25,50)):
    for amp in amps:
        tag=f'electric_{profile}_{amp}'
        if (B/f'{tag}_features.json').exists():continue
        on=.2;off=on+(.5 if profile=='Burley' else 1.)
        sg=r.schedule([(0,on,0,1,0),(on,off,0,1,0),(off,off+1,0,1,0)])
        o=options(sg,profile,1,1);o['local_GAP43_turnover_multiplier']=1.
        z=r.run(tag,sg,o,electrical_schedule=np.c_[sg[:,:2],[0,amp,0],[0,0,0]],retain_dense_solver_output=False,center_solver_coordinates=True,output_step_s=.0002,max_step_s=.0025,rtol=1e-8,atol=1e-10)
        r.save(tag+'_features',dict(profile=profile,current_pA=amp,features=r.features_mod.features(z,on,off,profile)))
if __name__=='__main__':
    if sys.argv[1]=='pilot':pilot()
    elif sys.argv[1]=='electric':electric(sys.argv[2])
    else:load(sys.argv[2],*[float(x) for x in sys.argv[3:8]],tight='tight' in sys.argv[8:])
