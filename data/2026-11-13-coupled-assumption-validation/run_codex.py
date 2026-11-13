"""[codex] Full 4425-state runs, fixed declared candidate inputs, no ion reset."""
from pathlib import Path
import sys,json,time,traceback,hashlib
from dataclasses import replace
import numpy as np
B=Path(__file__).resolve().parent; D=B.parent
sys.path.insert(0,str(B)); sys.path.insert(0,str(D/'neuron-model'))
from scenarios_codex import SCENARIOS
from model_contracts_codex import load_module
legacy=load_module(D/'2026-11-11-kcnq-passive/run_codex.py','sensitivity13_legacy')
r=legacy.r;r.B=B;r.PREFIX='S13_';r.SPARSE=True
ledger=load_module(D/'2026-11-12-potassium-flux-audit/ledger_audit_codex.py','sensitivity13_ledger')
KEYS={'soma_V':185,'process_V':192,'soma_K':200,'process_K':220,'soma_Na':199,'process_Na':219,
      'soma_ATP':120,'process_ATP':136,'soma_PCr':315,'process_PCr':316,'soma_Ca_increment':0,
      'process_Ca_increment':134,'intact_TDP':6,'soma_ADP':121,'process_ADP':137}

def options(sg,name,profile='Selvaraj-Con2'):
    p=SCENARIOS[name];o,rec=legacy.options(sg,profile,p['leak'])
    o['water_volume_parameters']=r.api.WaterVolumeParameters(process_water_fraction=p['water'])
    if p.get('absolute',True):
        o['absolute_kcnq_parameters']=r.api.AbsoluteKCNQParameters(g_total_nS=p['g'],replace_slow_fraction=p['replacement'])
        o['kcnq_coupling_parameters']=replace(o.get('kcnq_coupling_parameters',r.api.KCNQCouplingParameters()),gate_tau_s=p['tau'],vhalf_mV=p['vhalf'])
    ck=o['creatine_parameters'];o['creatine_parameters']=replace(ck,total_creatine_uM=tuple(x*p['creatine'] for x in ck.total_creatine_uM))
    # CK initialization remains the host equilibrium. Supply changes actual flux,
    # not ATP pools or an equilibrium-restoring controller.
    o['energy_schedule']=np.c_[sg[:,:2],np.full((len(sg),2),p['supply']),np.ones(len(sg))]
    o['pump_capacity_multiplier']=p['pump']
    return o,rec

def capture(name,sg,o,**kw):
    result=r.run(name,sg,o,retain_dense_solver_output=False,center_solver_coordinates=True,**kw)
    t=result['time_s'];z=result['state'];kb=ledger.observe(z,result['config'])
    selected={k:dict(initial=float(z[0,i]),final=float(z[-1,i]),min=float(z[:,i].min()),max=float(z[:,i].max())) for k,i in KEYS.items()}
    info=dict(selected=selected,all_finite=bool(np.isfinite(z).all()),states=z.shape[1],
              K_ledger_max_residual=max(float(np.max(abs(x['residual']))) for x in kb),
              K_terms=[{k:float(v[-1]) for k,v in x['terms'].items()} for x in kb],
              checks={k:float(v) for k,v in result['checks'].items() if k!='minimum_state'},
              diagnostic_interventions=result['checkpoint']['diagnostic_interventions'])
    r.save('S13_'+name+'_metrics',info)
    return result

def screen(name):
    for active in (False,True):
        label='screen_'+name+('_load' if active else '_rest')
        if (B/('S13_'+label+'_metrics_codex.json')).exists():continue
        sg=r.schedule([(0,10,0,1,0),(10,70,.001 if active else 0,1,0),(70,130,0,1,0),
                       (130,190,.001 if active else 0,1,0),(190,600,0,1,0)])
        o,rec=options(sg,name)
        z=capture(label,sg,o,output_step_s=2,max_step_s=10,rtol=1e-7,atol=legacy.long_atol())
        r.save('S13_'+label+'_protocol',dict(scenario=name,parameters=SCENARIOS[name],profile='Selvaraj-Con2',
          schedule=sg,conditions=rec,seconds=600,equal_opening_not_equal_calcium=True,reset=False))
        del z

def electrical(name,profile,amp,cp=None,suffix=''):
    tag=f'electric_{name}_{profile}_{amp:g}'+suffix
    if (B/('S13_'+tag+'_features_codex.json')).exists():return
    start=0. if cp is None else cp['time_s'];on=start+.2;off=on+(.5 if profile=='Burley' else 1.)
    sg=r.schedule([(start,on,0,1,0),(on,off,0,1,0),(off,off+1,0,1,0)])
    o,_=options(sg,name,profile)
    z=capture(tag,sg,o,checkpoint=cp,electrical_schedule=np.c_[sg[:,:2],[0,amp,0],[0,0,0]],
      output_step_s=.0002,max_step_s=.0025,rtol=1e-8,atol=1e-10)
    f=r.features_mod.features(z,on,off,profile)
    r.save('S13_'+tag+'_features',dict(profile=profile,current_pA=amp,scenario=name,features=f,
      start_s=start,no_reset=cp is None or np.array_equal(z['state'][0],cp['state']),
      historical_not_human_target=True))
    del z

def rest(name,profile='Selvaraj-Con2',hours=72.):
    tag=f'rest_{name}_{profile}_{hours:g}h'
    if (B/('S13_'+tag+'_metrics_codex.json')).exists():return
    end=hours*3600;edges=np.unique(np.r_[0,.01,.1,1,10,60,600,3600,np.arange(21600,end,21600),end])
    sg=r.schedule([(a,b,0,1,0) for a,b in zip(edges[:-1],edges[1:])]);o,_=options(sg,name,profile)
    capture(tag,sg,o,output_step_s=300,max_step_s=300,rtol=1e-7,atol=legacy.long_atol())

def loads(name,branch,opening=.001):
    profile='Selvaraj-Con2';cpfile=B/f'S13_rest_{name}_{profile}_72h_checkpoint_codex.json'
    cp=json.loads(cpfile.read_text());start=cp['time_s'];tag=f'load36_{name}_{branch}_{opening:g}'
    if (B/('S13_'+tag+'_metrics_codex.json')).exists():return
    sg=r.schedule([(start+h*3600,start+(h+1)*3600,opening if ((branch=='continuous' and h<6) or (branch=='repeated' and h in (0,4,8,12,16,20))) else 0,1,0) for h in range(36)])
    o,_=options(sg,name,profile)
    z=capture(tag,sg,o,checkpoint=cp,output_step_s=300,max_step_s=300,rtol=1e-7,atol=legacy.long_atol())
    assert np.array_equal(z['state'][0],cp['state']) and z['checkpoint']['origin_time_s']==cp['origin_time_s']
    r.save('S13_'+tag+'_protocol',dict(checkpoint=str(cpfile),schedule=sg,active_hours=0 if branch=='control' else 6,
      all_states_preserved=True,equal_opening_not_equal_calcium=True,reset=False))

def pilot():
    # Same inputs/settings as stored P13 pilot; compare all states at identity.
    sg=r.schedule([(0,.02,0,1,0)]);o,_=options(sg,'legacy','Burley')
    o.pop('energy_schedule');o.pop('pump_capacity_multiplier')
    a=capture('pilot_legacy',sg,o,output_step_s=.002,max_step_s=.002,rtol=1e-8,atol=1e-10)
    old=np.load(D/'2026-11-13-pump-response-test/P13_pilot_current_trace_codex.npz')
    delta=float(np.max(abs(a['state']-old['state'])))
    r.save('identity_check',dict(all_state_max_difference=delta,states=a['state'].shape[1],baseline_equal=bool(np.array_equal(a['baseline_state'],old['baseline']))))
    assert delta<1e-6
    # A restart inherits the multiplier; test common flux ledgers through host.
    o['pump_capacity_multiplier']=2.
    b=capture('pilot_pump2',sg,o,output_step_s=.002,max_step_s=.002,rtol=1e-8,atol=1e-10)
    o.pop('pump_capacity_multiplier');sg2=r.schedule([(.02,.04,0,1,0)])
    c=capture('pilot_pump2_restart',sg2,o,checkpoint=b['checkpoint'],output_step_s=.002,max_step_s=.002,rtol=1e-8,atol=1e-10)
    assert c['checkpoint']['diagnostic_interventions']['pump_capacity_multiplier']==2.
    assert np.array_equal(c['state'][0],b['state'][-1])
    r.save('pump_hook_check',dict(identity_delta=delta,common_ledgers_checked=True,restart_inherits_multiplier=True))

def batch(worker,total):
    for i,name in enumerate(SCENARIOS):
        if i%total!=worker:continue
        try:screen(name)
        except Exception as e:
            r.save('S13_failed_'+name,dict(error=str(e),traceback=traceback.format_exc(),parameters=SCENARIOS[name],accepted=False))
            print('FAILED',name,str(e),flush=True)

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='batch':batch(int(sys.argv[2]),int(sys.argv[3]))
    elif cmd=='screen':screen(sys.argv[2])
    elif cmd=='electric':electrical(sys.argv[2],sys.argv[3],float(sys.argv[4]))
    elif cmd=='rest':rest(sys.argv[2],sys.argv[3] if len(sys.argv)>3 else 'Selvaraj-Con2')
    elif cmd=='loads':loads(sys.argv[2],sys.argv[3],float(sys.argv[4]) if len(sys.argv)>4 else .001)
    elif cmd=='pilot':pilot()
