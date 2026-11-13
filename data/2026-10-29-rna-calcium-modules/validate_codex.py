"""[codex] 2026-10-29. Conditional mechanisms, provenance and regression tests."""
from pathlib import Path
from dataclasses import replace,asdict
import importlib.util,sys,json,time,argparse
import numpy as np
from threadpoolctl import threadpool_limits
_limit=threadpool_limits(limits=1)
B=Path(__file__).resolve().parent
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
api=module('rna_api29',B.parent/'neuron-model/model_api_codex.py')
from rna_calcium_codex import receptor_capacity,surface_rates,fxyd_ratio,GENES
from connected_parameters_codex import schedule,ConnectedParameters
def save(name,obj):
    (B/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a)),encoding='utf8')
def factors(**kw):return tuple(kw.get(k,1.) for k in GENES)
def options(profile='Burley'):
    return dict(**api.energy_terminal_inputs(profile),adaptive_parameters=api.AdaptiveRecyclingParameters(enabled=True,calcium_capacity_enabled=True),recycling_parameters=api.RecyclingParameters(enabled=True),pathway_parameters=api.TerminalPathwayParameters(enabled=True),activity_parameters=api.ActivityEnergyParameters(gain=60.))
def source_tests():
    p=api.RNACalciumParameters();v=p.vector()
    assert abs(surface_rates(.5,.0804,.0804,v))<1e-15
    assert receptor_capacity(v)==(1.,1.)
    assert receptor_capacity(replace(p,protein_factors=factors(GABRB2=2)).vector())[0]==1 # partners limit
    assert receptor_capacity(replace(p,protein_factors=factors(GABRB2=.5)).vector())[0]==.5
    # Every surface stays in [0,1] by vector field pointing inward.
    for protein in (0.,.5,1.,2.):
        for ca in (0.,.08,1.,10.):
            q=replace(p,protein_factors=factors(EIF3E=protein,DNM1=protein,CLTC=protein,ACTN1=protein),calcium_removal_gain=5).vector()
            assert surface_rates(0.,ca,.08,q)>=0 and surface_rates(1.,ca,.08,q)<=0
    for na in (1000.,10000.,30000.):
        assert fxyd_ratio(na,4000,-70,v)==1.
    for name in ('CLU','PRNP'):
        try:replace(p,protein_factors=factors(**{name:.5})).vector();raise AssertionError('Unresolved gene silently activated')
        except ValueError:pass
    save('mechanism_checks_codex.json',dict(passed=True,checks=['finite CaV surface boundaries','limiting GABA partners','neutral FXYD ratio','no unsupported CLU/PRNP kinetics'],source_scope='Conditional architecture, not biological calibration'))
def electrical_tests():
    metrics=module('rna_features29',B.parent/'2026-10-26-nak-source-calibration/validate_model_codex.py')
    prior=json.loads((B.parent/'2026-10-29-joint-recovery/electrical_codex.json').read_text())['rows'];rows=[]
    for row in prior:
        profile=row['profile'];cur=row['current_pA'];dur=.5 if profile=='Burley' else 1.
        sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
        r=api.run_rna_calcium_neuron(sg,**options(profile),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
        f=metrics.features(r,.2,.2+dur,profile);assert f['spikes']==row['target'],(profile,cur,f)
        change=max(abs(float(f[k])-float(a)) for k,a in row['features'].items() if isinstance(a,(int,float)) and np.isfinite(a) and np.isfinite(f[k]))
        assert change<.02,(profile,cur,change)
        rows.append(dict(profile=profile,current_pA=cur,target=row['target'],features=f,max_feature_change=change,checks=r['checks']))
        save('electrical_codex.json',dict(rows=rows,complete=len(rows)==9));print(profile,cur,f['spikes'],change,flush=True)
def run_case(name,params,*,end=12.,strict=False,checkpoint=None,tail=False,ligands=True):
    sg=schedule([(0,2,0,1,0),(2,4,.01,1,0),(4,end,0,1,0)])
    es=np.c_[sg[:,:2],[0,25,0],[0,0,0]];ls=np.c_[sg[:,:2],[0,1 if ligands else 0,0],[0,1 if ligands else 0,0]]
    if tail:sg=sg[2:];es=es[2:];ls=ls[2:]
    kw=dict(output_step_s=.005,max_step_s=.0025 if strict else .01,rtol=1e-9 if strict else 1e-8,atol=1e-11 if strict else 1e-10)
    r=api.run_rna_calcium_neuron(sg,**options(),rna_calcium_parameters=params,rna_ligand_schedule=ls,electrical_schedule=es,connected_parameters=ConnectedParameters(g_vgcc_nS=.03,g_sk_nS=.015),checkpoint=checkpoint,**kw)
    np.savez_compressed(B/(name+'_codex.npz'),time_s=r['time_s'],state=r['state'])
    spikes=[t for t in r['voltage_crossing_times_s']['soma'] if 2<=t<4]
    summary=dict(name=name,params=asdict(params),spikes_during_load=len(spikes),Ca_peak_uM=float(r['calcium_uM'].max()),Ca_excess_AUC_uM_s=float(np.trapz(np.maximum(r['calcium_uM']-r['config']['basal_ca_uM'],0),r['time_s'])),
        final_Ca_uM=float(r['calcium_uM'][-1]),Na_final_uM=float(r['sodium_in_uM'][-1,0]),ATP_final_uM=float(r['ATP_uM'][-1]),surface_final=r['CaV12_surface_fraction'][-1],checks=r['checks'],config=r['config'],checkpoint=r['checkpoint'],segments=sg,ligands=ls,electrical=es,solver=kw)
    save(name+'_codex.json',summary);print(name,summary['spikes_during_load'],summary['Ca_peak_uM'],flush=True)
    return r,summary
def scenarios():
    p=api.RNACalciumParameters(gaba_nS=.1,nmda_nS=.003,cav12_fraction=.5,calcium_removal_gain=1.,fxyd7_strength=.5)
    cases={'reference':p,'GABA_beta_half':replace(p,protein_factors=factors(GABRB2=.5)),'GABA_partners_double':replace(p,protein_factors=factors(GABRB2=2,GABRA1=2,GABRA2=2),receptor_partner_ceiling=2),
        'GABA_depolarizing':replace(p,gaba_reversal_mV=-40),'GRIN2A_half':replace(p,protein_factors=factors(GRIN2A=.5)),
        'FXYD7_half':replace(p,protein_factors=factors(FXYD7=.5)),'DNM1_half':replace(p,protein_factors=factors(DNM1=.5)),
        'ACTN4_half':replace(p,protein_factors=factors(ACTN4=.5)), 'EIF3E_double_removal':replace(p,protein_factors=factors(EIF3E=2)),
        'EIF3E_double_both':replace(p,protein_factors=factors(EIF3E=2),eif3e_insertion_gain=1.),'magnesium_zero':replace(p,magnesium_mM=0.)}
    summaries=[]
    for name,params in cases.items():
        r,s=run_case(name,params);summaries.append(s)
        if name=='reference':ref=r
    strict,_=run_case('reference_strict',p,strict=True)
    head,_=run_case('reference_head',p,end=4.0001)
    # Use an exact split at a segment boundary (a tiny extra segment is not a restart).
    sg=schedule([(0,2,0,1,0),(2,4,.01,1,0)]);es=np.c_[sg[:,:2],[0,25],[0,0]];ls=np.c_[sg[:,:2],[0,1],[0,1]]
    a=api.run_rna_calcium_neuron(sg,**options(),rna_calcium_parameters=p,rna_ligand_schedule=ls,electrical_schedule=es,connected_parameters=ConnectedParameters(g_vgcc_nS=.03,g_sk_nS=.015),output_step_s=.005,max_step_s=.01,rtol=1e-8,atol=1e-10)
    resumed,_=run_case('reference_restart',p,checkpoint=a['checkpoint'],tail=True)
    err=lambda a,b:float(np.max(abs(a-b)/(1+abs(b))))
    n=dict(restart_scaled_error=err(ref['state'][-1],resumed['state'][-1]),convergence_scaled_error=err(ref['state'][-1],strict['state'][-1]),columns=len(ref['columns']),states=ref['state'].shape[1])
    assert n['columns']==n['states']==320
    assert max(n['restart_scaled_error'],n['convergence_scaled_error'])<1e-4,n
    bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][318]=1.1
    try:run_case('invalid_checkpoint',p,checkpoint=bad,tail=True);raise AssertionError('Invalid surface accepted')
    except ValueError:pass
    save('numerical_codex.json',n);save('scenarios_codex.json',summaries)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('pilot','electrical','scenarios','unit'),default='pilot');args=parser.parse_args()
    source_tests()
    if args.mode=='pilot':run_case('pilot',api.RNACalciumParameters())
    elif args.mode=='electrical':electrical_tests()
    elif args.mode=='scenarios':scenarios()
