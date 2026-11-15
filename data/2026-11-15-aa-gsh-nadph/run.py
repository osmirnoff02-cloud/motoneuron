from pathlib import Path
import sys,json
import numpy as np
import supply as shared
from model_contracts import load_module
B=Path(__file__).resolve().parent/'runs_v2';D=B.parent.parent
prev=load_module(D/'2026-11-15-shared-redox-budget/run.py','_agn15_previous')
r=prev.r;r.B=B;r.PREFIX='AGN15_';r.SPARSE=True
read=prev.read
def options(sg,profile='Burley',**overrides):
    o=prev.options(sg,profile)
    o['shared_redox_parameters']=shared.SupplyParameters(**overrides)
    return o
def pilot():
    sg=r.schedule([(0,.1,0,1,0)])
    out=r.run('pilot',sg,options(sg),output_step_s=.01,max_step_s=.01,rtol=1e-8,atol=1e-10,retain_dense_solver_output=False)
    r.save('pilot_done',dict(checks=out['checks'],config=out['config']['shared_redox']))
def load(branch='control',mode='middle',tight=False,hours=72.):
    grid={'middle':{},'low':dict(AA_delivery_capacity_uM_s=.0,GSH_recovery_fraction=0.,GSH_delivery_tau_s=36000.),
          'high':dict(AA_delivery_capacity_uM_s=.3,GSH_recovery_fraction=.9,GSH_delivery_tau_s=600.),
          'noPPP':dict(explicit_NADPH=False),'ki18':dict(G6PD_Ki_NADPH_uM=18.),
          'soma_high':dict(soma_GSH_synthesis_capacity_uM_s=.6)}
    tag=f'{branch}_{mode}_{hours:g}h'+('_tight' if tight else '')
    if (B/f'{tag}_done.json').exists():return
    old=read(prev.previous.CP);sg=prev.previous.prior.schedule(branch,old['time_s'],hours)
    o=options(sg,**grid[mode]);cp=prev.migrate(sg,o,old)
    atol=np.r_[prev.previous.prior.prior.a.atol(),np.full(shared.NS,1e-9)];atol[4458]=1e-12
    out=r.run(tag,sg,o,checkpoint=cp,retain_dense_solver_output=False,center_solver_coordinates=True,
        output_step_s=600,max_step_s=150 if tight else 300,rtol=3e-8 if tight else 1e-7,atol=atol*(.3 if tight else 1.))
    assert np.array_equal(out['state'][0],cp['state'])
    r.save(tag+'_done',dict(no_reset=True,checks=out['checks'],config=out['config']['shared_redox'],observation=out['shared_redox_observation'],branch=branch,mode=mode,hours=hours))
def electric(profile,amp):
    tag=f'electric_{profile}_{amp}'
    if (B/f'{tag}_features.json').exists():return
    on=.2;off=on+(.5 if profile=='Burley' else 1.)
    sg=r.schedule([(0,on,0,1,0),(on,off,0,1,0),(off,off+1,0,1,0)])
    o=options(sg,profile);o.update(local_GAP43_turnover_multiplier=1.)
    # Match the previous initial electrical protocols exactly.
    original=prev.options(sg,profile,1,1)
    original['shared_redox_parameters']=o['shared_redox_parameters'];original['local_GAP43_turnover_multiplier']=1.
    z=r.run(tag,sg,original,electrical_schedule=np.c_[sg[:,:2],[0,amp,0],[0,0,0]],retain_dense_solver_output=False,
        center_solver_coordinates=True,output_step_s=.0002,max_step_s=.0025,rtol=1e-8,atol=1e-10)
    r.save(tag+'_features',dict(profile=profile,current_pA=amp,features=r.features_mod.features(z,on,off,profile)))
if __name__=='__main__':
    if sys.argv[1]=='pilot':pilot()
    elif sys.argv[1]=='electric':electric(sys.argv[2],int(sys.argv[3]))
    else:load(*sys.argv[2:4],tight='tight' in sys.argv[4:],hours=1. if 'short' in sys.argv[4:] else 72.)
