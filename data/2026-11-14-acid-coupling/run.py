"""[codex] C08 + reviewed Claude acid closure. Full 4431 states, no reset."""
from pathlib import Path
import sys,json,numpy as np
B=Path(__file__).resolve().parent;D=B.parent;OLD=D/'2026-11-13-joint-kcnq-pump-v2'
sys.path.insert(0,str(OLD));sys.path.insert(0,str(D/'neuron-model'))
from model_contracts import load_module
f=load_module(OLD/'full.py','acid14_previous_full')
r=f.r;r.B=B;r.PREFIX='A14_';r.SPARSE=True
ac=r.api._acid13_claude
PARAM=f.params('C08')
def options(sg,profile='Selvaraj-Con2',tau=120.,hca=2.):
    o,rec=f.options(sg,profile,PARAM)
    if tau is not None:o['acid_extrusion_parameters']=r.api.AcidExtrusionParameters(recovery_tau_s=tau,H_per_Ca=hca)
    return o,rec
def atol(tau=120.):return f.c.old.legacy.long_atol() if tau is None else np.r_[f.c.old.legacy.long_atol(),np.full(6,1e-8)]
def run(tag,sg,o,**kw):
    out=r.run(tag,sg,o,retain_dense_solver_output=False,center_solver_coordinates=True,**kw)
    if out.get('acid_extrusion_observation') is not None:
        r.save('A14_'+tag+'_acid',out['acid_extrusion_observation'])
    r.save('A14_'+tag+'_metadata',dict(states=out['state'].shape[1],config=out['config'],state_schema=out['state_schema'],
        integrated_model=out['integrated_model'],checks=out['checks'],events=out.get('voltage_crossing_times_s'),
        no_reset=kw.get('checkpoint') is None or np.array_equal(out['state'][0],kw['checkpoint']['state'])))
    return out
def pilot():
    sg=r.schedule([(0,.02,0,1,0)])
    off,_=options(sg,tau=None);zero,_=options(sg,hca=0);on,_=options(sg)
    a=run('pilot_off',sg,off,output_step_s=.002,max_step_s=.002,rtol=1e-8,atol=1e-10)
    b=run('pilot_zero',sg,zero,output_step_s=.002,max_step_s=.002,rtol=1e-8,atol=1e-10)
    c=run('pilot_on',sg,on,output_step_s=.002,max_step_s=.002,rtol=1e-8,atol=1e-10)
    delta=float(np.max(abs(a['state']-b['state'][:,:4425])))
    assert delta<1e-6 and np.max(abs(b['state'][:,4425:]))==0,delta
    sg2=r.schedule([(.02,.04,0,1,0)]);o,_=options(sg2)
    q=run('pilot_restart',sg2,o,checkpoint=c['checkpoint'],output_step_s=.002,max_step_s=.002,rtol=1e-8,atol=1e-10)
    assert np.array_equal(q['state'][0],c['state'][-1])
    r.save('pilot_checks',dict(zero_coupling_max_abs=delta,states=4431,restart_exact=True,default_off=True))
def electric(profile,amp,cp=None,label='initial',tau=120.):
    tag=f'electric_{label}_{profile}_{amp:g}'
    if (B/f'A14_{tag}_features.json').exists():return
    start=0 if cp is None else cp['time_s'];on=start+.2;off=on+(.5 if profile=='Burley' else 1.)
    sg=r.schedule([(start,on,0,1,0),(on,off,0,1,0),(off,off+1,0,1,0)])
    o,rec=options(sg,profile,tau)
    out=run(tag,sg,o,checkpoint=cp,electrical_schedule=np.c_[sg[:,:2],[0,amp,0],[0,0,0]],output_step_s=.0002,max_step_s=.0025,rtol=1e-8,atol=1e-10)
    r.save('A14_'+tag+'_features',dict(profile=profile,current_pA=amp,features=r.features_mod.features(out,on,off,profile),
        start_s=start,no_reset=cp is None or np.array_equal(out['state'][0],cp['state']),historical_not_human_target=True,conditions=rec))
def rest(tau=120.):
    tag=f'rest_tau{tau:g}_72h'
    if (B/f'A14_{tag}_metadata.json').exists():return
    end=72*3600;edges=np.unique(np.r_[0,.01,.1,1,10,60,600,3600,np.arange(21600,end,21600),end])
    sg=r.schedule([(a,b,0,1,0) for a,b in zip(edges[:-1],edges[1:])]);o,_=options(sg,tau=tau)
    run(tag,sg,o,output_step_s=300,max_step_s=300,rtol=1e-7,atol=atol(tau))
def loads(tau,branch,opening=.001):
    cpfile=B/f'A14_rest_tau{tau:g}_72h_checkpoint.json';cp=json.loads(cpfile.read_text());start=cp['time_s']
    tag=f'load_tau{tau:g}_{branch}_{opening:g}'
    if (B/f'A14_{tag}_metadata.json').exists():return
    sg=r.schedule([(start+h*3600,start+(h+1)*3600,opening if ((branch=='continuous' and h<6) or (branch=='repeated' and h in (0,4,8,12,16,20))) else 0,1,0) for h in range(36)])
    o,_=options(sg,tau=tau)
    out=run(tag,sg,o,checkpoint=cp,output_step_s=300,max_step_s=300,rtol=1e-7,atol=atol(tau))
    assert np.array_equal(out['state'][0],cp['state']) and out['checkpoint']['origin_time_s']==cp['origin_time_s']
    r.save('A14_'+tag+'_protocol',dict(checkpoint=str(cpfile),schedule=sg,all_states_preserved=True,equal_opening_not_equal_calcium=True,reset=False))
if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='pilot':pilot()
    elif cmd=='suite':
        for amp in (15,25,50):electric(sys.argv[2],amp)
    elif cmd=='rest':rest(float(sys.argv[2]))
    elif cmd=='loads':loads(float(sys.argv[2]),sys.argv[3],float(sys.argv[4]))
    elif cmd=='post_one':
        branch=sys.argv[2];amp=float(sys.argv[3])
        cp=json.loads((B/f'A14_load_tau120_{branch}_{0 if branch=="control" else 0.001:g}_checkpoint.json').read_text())
        electric('Selvaraj-Con2',amp,cp,'post108_'+branch)
    elif cmd=='post':
        # Independent probes branch from the same full checkpoint. Processes
        # avoid shared module globals; no state reset within any trajectory.
        import subprocess,time,msvcrt
        from concurrent.futures import ThreadPoolExecutor
        def task(item):
            branch,amp=item
            if (B/f'A14_electric_post108_{branch}_Selvaraj-Con2_{amp:g}_features.json').exists():return
            with (B/f'post_{branch}_{amp}_log.txt').open('w') as log:
                subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'post_one',branch,str(amp)],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('post completed',branch,amp,flush=True)
        # A second queue can safely reuse completed probes while the first is
        # still running. Windows releases the byte lock on close/crash; the
        # small lock file is retained, never deleted.
        with (B/'post_queue_codex.lock').open('a+b') as lock:
            if lock.tell()==0:lock.write(b'0');lock.flush()
            deadline=time.monotonic()+10800
            while True:
                lock.seek(0)
                try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError('Post-probe queue lock')
                    time.sleep(2)
            try:
                with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(task,[(b,a) for b in ('control','repeated') for a in (15,25,50)]))
            finally:
                lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
