"""[codex] Joint composition checks, preserving the reviewed numerical backend."""
from pathlib import Path
import sys,json,time,hashlib,inspect,copy
from unittest.mock import patch
import numpy as np

B=Path(__file__).resolve().parent;D=B.parent
sys.path.insert(0,str(D/'neuron-model'))
from model_contracts_codex import load_module
api=load_module(D/'neuron-model/model_api_codex.py','integration04_api')
host=load_module(D/'2026-10-28-takeuchi-balaji/adaptive_recycling_neuron_codex.py','integration04_host')
setup=load_module(D/'2026-11-03-translation-coupling/validate_codex.py','integration04_setup')
schedule=setup.m.schedule;options=setup.m.m.hist.options

def save(name,obj):
    (B/(name+'_codex.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,
        default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n',encoding='utf8')

def audit(x):
    check={k:v for k,v in x['checks'].items() if k!='minimum_state'}
    assert max(check.values())<3e-5
    assert np.isfinite(x['state']).all()
    return dict(states=len(x['columns']),worst_check=max(check.values()),checks=check,
                nfev=x['nfev'],state_schema=x['state_schema'])

def contract():
    class Prepared(Exception):pass
    found={}
    def stop(*a,**kw):
        f=inspect.currentframe().f_back
        found.update(config=copy.deepcopy(f.f_locals['config']),schema=f.f_locals['schema'])
        raise Prepared
    sg=schedule([(0,.01,0,1,0)])
    with patch.object(host,'solve_ivp',stop):
        try:api.run_integrated_neuron(sg,**options())
        except Prepared:pass
        else:raise AssertionError('Expected the shared solver')
    for k in ('individual_RP','RP_production','nuclear_transport','RNA_export','RP_mRNA_kinetics'):
        assert k in found['config'],(k,list(found['config']))
    assert found['schema']['count']==4413
    for kw in ({'include_individual_rp':False},{'nuclear_transport_parameters':None}):
        try:api.run_integrated_neuron(sg,**options(),**kw)
        except ValueError:pass
        else:raise AssertionError('Incomplete integrated topology accepted')
    save('contract',dict(passed=True,components=api.integrated_model_components(),**found))
    print('PASS common configuration, all five components, 4413 states',flush=True)

def smoke():
    started=time.monotonic();sg=schedule([(0,.02,0,1,0)])
    opts=dict(**options(),output_step_s=.01,max_step_s=.01,rtol=1e-8,atol=1e-10)
    # Run the existing backend first so diagnostic arrays survive any metadata bug.
    a=api.run_rp_mrna_kinetics_neuron(sg,**opts)
    save('actual_keys',dict(result=list(a),config=list(a['config']),
        observations={k:list(a[k]) for k in ('individual_RP_observation','RP_production_observation','RP_mRNA_kinetics_observation')}))
    b=api.run_integrated_neuron(sg,**opts)
    assert np.array_equal(a['state'],b['state']) and a['checkpoint']==b['checkpoint']
    assert a['checks']==b['checks'] and a['config']==b['config']
    assert a['columns']==b['columns'] and len(set(b['integrated_columns']))==4413
    for gene in api.RP_PRODUCTION_GENES:
        for field in ('nuclear_free','cytoplasmic_free'):
            assert b['integrated_columns'].index('nrt_RNA_'+gene+'_'+field+'_uM')!=b['integrated_columns'].index('nrt_protein_'+gene+'_'+field+'_uM')
    assert b['integrated_model']['new_energy_debits']==0
    assert not b['integrated_model']['human_motor_neuron_calibrated']
    external=b['integrated_model']['RP_mRNA_external_transfer_validation']
    assert external['genes']==80 and external['qualifying_genes']==0
    # Observer projections do not alter solver state or account for another cost.
    for k in ('UAP56_ATP','DDX19_ATP','ribosome_RanGTP_hydrolyzed'):
        names=b['RNA_export_observation']['counter_names']
        assert np.array_equal(b['integrated_transport_readout']['existing_energy_counters'][k],
                              b['RNA_export_observation']['counters'][:,names.index(k)])
    cp=copy.deepcopy(b['checkpoint']);tail=schedule([(.02,.03,0,1,0)])
    c=api.run_integrated_neuron(tail,checkpoint=cp,include_u41_readout=False,**opts)
    assert np.array_equal(c['state'][0],b['state'][-1])
    assert 'integrated_transport_readout' not in c and c['checkpoint_schema_status']=='verified'
    save('smoke',dict(passed=True,elapsed_s=time.monotonic()-started,
        same_state_checkpoint_config_checks=True,observer_costs_not_redebited=True,restart_continuity=True,
        integration=b['integrated_model'],results=[audit(x) for x in (a,b,c)]))
    np.savez_compressed(B/'short_trace_codex.npz',time_s=b['time_s'],state=b['state'])
    print('PASS integrated identity, balances, shared energy, restart',flush=True)

def electrical(profile='Selvaraj-Con1'):
    started=time.monotonic();dur=.5 if profile=='Burley' else 1.;cur=25
    sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
    def progress(cp,nfev):
        save('electrical_progress',dict(profile=profile,time_s=cp['time_s'],nfev=nfev,elapsed_s=time.monotonic()-started))
        print('Full model',profile,'t=',cp['time_s'],'nfev=',nfev,flush=True)
    x=api.run_integrated_neuron(sg,**options(profile),
        electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],
        output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11,solver_observer=progress)
    metrics=load_module(D/'2026-10-26-nak-source-calibration/validate_model_codex.py','integration04_metrics')
    features=metrics.features(x,.2,.2+dur,profile)
    old=json.loads((D/'2026-11-03-ion-ahp-mechanisms/electrical_regression_codex.json').read_text())[profile]['features']
    deltas={k:float(features[k])-float(v) for k,v in old.items() if k in features and isinstance(v,(int,float)) and np.isfinite(v)}
    row=dict(profile=profile,current_pA=cur,features=features,baseline_API047=old,feature_deltas=deltas,
        elapsed_s=time.monotonic()-started,**audit(x))
    row['passed']=features['spikes']==old['spikes'] and max(abs(v) for v in deltas.values())<.02
    save('electrical_'+profile,row)
    np.savez_compressed(B/('electrical_'+profile+'_codex.npz'),time_s=x['time_s'],voltage=x['state'][:,185])
    assert row['passed'],row
    print('PASS full integrated',profile,features['spikes'],'spikes',flush=True)

if __name__=='__main__':
    for task in sys.argv[1:] or ['contract','smoke']:
        if task.startswith('electrical:'):electrical(task.split(':',1)[1])
        else:globals()[task]()
