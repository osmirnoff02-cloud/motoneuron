"""[codex] 2026-10-30. Mechanistic checks, source separation, electrical regression."""
from pathlib import Path
from dataclasses import replace,asdict
import importlib.util,sys,json,argparse
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
_limit=threadpool_limits(limits=1)
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B.parent/'2026-10-29-rna-calcium-modules'))
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
v29=module('validate_rna30',B.parent/'2026-10-29-rna-calcium-modules/validate.py');api=v29.api
from nucleus_ribosome import *
from connected_parameters import schedule
def save(name,obj):
    (B/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a)),encoding='utf8')
def constants():
    import long_pmca as c
    kw=api.energy_terminal_inputs('Burley')
    required={'basal_ca_uM','calcium_tau_s','enzyme_parameters','tdp_parameters','localization_parameters','stmn2_parameters','inactivation_parameters','phosphorylation_parameters','reactivation_parameters','stability_parameters'}
    _,p,_,_,_=c._prior._prior._agg._baseline({k:kw[k] for k in required},kw['aggregation_parameters'].vector())
    hv=np.r_[kw['autoregulation_parameters'].vector(),kw['rna_feedback_parameters'].vector()]
    return p,hv,kw['localization_parameters'].nuclear_fraction,kw['energy_parameters'].ATP0_uM
def unit():
    p,hv,f0,atp=constants();pv=NucleusRibosomeParameters().vector();x=initial_nucleus(p,hv,pv,f0)
    out=nucleus_fluxes(x,1.,1.,1.,atp,p,hv,pv,f0,atp,np.ones(4))
    assert abs(out[0][:8]).max()<1e-12 and abs(out[1])<1e-12 and abs(out[2])<1e-12
    assert abs(out[3]-1)<1e-12 and abs(out[4]-1)<1e-12
    # Complete block cannot produce completed protein. At full capacity,
    # initiation cannot push total occupancy outside the simplex.
    off=nucleus_fluxes(x,1.,1.,1.,atp,p,hv,pv,f0,atp,np.array([1.,1.,0.,1.]))
    assert off[3]==0 and off[4]==0
    xx=x.copy();xx[6:8]=[.6,.4]
    full=nucleus_fluxes(xx,1.,1.,1.,atp,p,hv,pv,f0,atp,np.ones(4))
    assert full[0][6:8].sum()<0
    # RNA balances in normalized amounts: precursor + nuclear + cytoplasmic
    # (plus cryptic for STMN2) = transcription minus actual degradation.
    xx=x.copy()*1.2;xx[0]=f0*.7
    trial=nucleus_fluxes(xx,.8,.7,.6,atp,p,hv,pv,f0,atp,np.ones(4))
    nuc=.8*xx[0]/f0;target=hv[1]+(hv[2]-hv[1])/(1+(hv[2]-1)/(1-hv[1])*nuc**hv[3])
    assert abs(trial[0][1]+trial[0][2]+trial[1]-(target**(1-hv[5])-.7/target**hv[5])/hv[4])<1e-12
    assert abs(sum(trial[0][3:6])+trial[2]-(1/(p[23]*pv[4])-.6/p[23]-np.log(2)*xx[5]/pv[3]))<1e-12
    z=np.zeros(330);z[6]=1;z[50]=z[10]=1;z[120]=atp;z[320:]=x
    old=np.zeros(247);old[18]=p[20];new,dx=replace_biosynthesis(z,old,p,hv,pv,f0,atp,np.array([1.,.2,1.,1.]))
    assert abs(new[6]-old[6]-(new[18]-old[18]))<1e-15
    pulse=hachoumi_pulse_train();assert sum(pulse[:,2]>0)==400
    assert abs(np.sum((pulse[:,1]-pulse[:,0])*pulse[:,2])-32)<1e-10
    assert np.allclose(pulse[1:,0],pulse[:-1,1])
    save('unit_checks.json',dict(passed=True,checks=['matched basal biosynthesis','finite ribosome simplex','zero elongation no completed protein','RNA mass balance','TDP synthesis ownership','400 pulses 20Hz 2ms 40pA'],hachoumi=hachoumi2022_reference(),Harley=nuclear_recovery_reference(),Tomuro=translation_source_reference()))
def integrated():
    sg=schedule([(0,.2,0,1,0),(.2,.7,.01,1,0),(.7,2,0,1,0)]);es=np.c_[sg[:,:2],[0,25,0],[0,0,0]]
    opts=dict(**v29.options(),electrical_schedule=es,output_step_s=.001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
    r=api.run_nucleus_ribosome_neuron(sg,**opts)
    assert r['state'].shape[1]==len(r['columns'])==330
    a=api.run_nucleus_ribosome_neuron(sg[:2],**dict(opts,electrical_schedule=es[:2]))
    b=api.run_nucleus_ribosome_neuron(sg[2:],checkpoint=a['checkpoint'],**dict(opts,electrical_schedule=es[2:]))
    er=float(np.max(abs(r['state'][-1]-b['state'][-1])/(1+abs(r['state'][-1]))));assert er<1e-5
    strict=api.run_nucleus_ribosome_neuron(sg,**dict(opts,max_step_s=.00125,rtol=3e-10,atol=3e-12))
    conv=float(np.max(abs(r['state'][-1]-strict['state'][-1])/(1+abs(strict['state'][-1]))));assert conv<1e-4
    save('integrated_checks.json',dict(complete=True,states=330,checks=r['checks'],restart_scaled_error=er,convergence_scaled_error=conv,config=r['config']))
    np.savez_compressed(B/'integrated_trace.npz',time_s=r['time_s'],state=r['state'])
    bad=dict(a['checkpoint']);bad['state']=list(bad['state']);bad['state'][326]=1.2
    try:api.run_nucleus_ribosome_neuron(sg[2:],checkpoint=bad,**dict(opts,electrical_schedule=es[2:]));raise AssertionError('Invalid capacity accepted')
    except ValueError:pass
    print('Integrated/restart/convergence passed',er,conv,flush=True)
def electrical():
    metrics=module('nuclear_features30',B.parent/'2026-10-26-nak-source-calibration/validate_model.py')
    prior=json.loads((B.parent/'2026-10-29-joint-recovery/electrical.json').read_text())['rows'];rows=[]
    for row in prior:
        profile=row['profile'];cur=row['current_pA'];dur=.5 if profile=='Burley' else 1.
        sg=schedule([(0,.2,0,1,0),(.2,.2+dur,0,1,0),(.2+dur,1.2+dur,0,1,0)])
        kw=dict(**v29.options(profile),electrical_schedule=np.c_[sg[:,:2],[0,cur,0],[0,0,0]],output_step_s=.0001,max_step_s=.0025,rtol=1e-9,atol=1e-11)
        # Both old topology and enabled nuclear branch are checked.
        entries={}
        for name,runner in [('legacy',api.run_rna_calcium_neuron),('nucleus',api.run_nucleus_ribosome_neuron)]:
            r=runner(sg,**kw);f=metrics.features(r,.2,.2+dur,profile)
            assert f['spikes']==row['target'],(name,profile,cur,f)
            change=max(abs(float(f[k])-float(a)) for k,a in row['features'].items() if isinstance(a,(int,float)) and np.isfinite(a) and np.isfinite(f[k]));assert change<.02,(name,profile,cur,change)
            entries[name]=dict(features=f,max_feature_change=change,checks=r['checks'])
        rows.append(dict(profile=profile,current_pA=cur,target=row['target'],branches=entries))
        save('electrical.json',dict(rows=rows,complete=len(rows)==9));print(profile,cur,row['target'],flush=True)
def component_run(parameters,segments,*,step_s=120.,max_step_s=60.,rtol=1e-9):
    """Isolated biosynthesis test, not a long-duration spiking neuron.

    ATP is clamped. TDP uses matched basal gross first-order turnover; no
    additional CAPN or aggregation stimulus. This isolates import vs translation.
    """
    p,hv,f0,atp=constants();pv=parameters.vector();x=initial_nucleus(p,hv,pv,f0)
    # Total intact TDP, cytoplasmic TARDBP/STMN2 RNA, STMN2 protein; 10 new states.
    y=np.r_[1.,1.,1.,1.,x];times=[0.];states=[y.copy()]
    def fun(t,z,controls):
        dx,mt,ms,tf,sf,nuc=nucleus_fluxes(z[4:],z[0],z[1],z[2],atp,p,hv,pv,f0,atp,controls)
        return np.r_[p[20]*(tf-z[0]),mt,ms,np.log(2)/pv[11]*(sf-z[3]),dx]
    for start,end,controls in segments:
        grid=np.unique(np.r_[np.arange(start+step_s,end,step_s),end])
        s=solve_ivp(lambda t,z:fun(t,z,np.array(controls)),(start,end),y,method='LSODA',t_eval=grid,rtol=rtol,atol=rtol*.01,max_step=max_step_s)
        assert s.success;sx=s.y.T;times.extend(s.t);states.extend(sx);y=sx[-1]
    z=np.asarray(states);t=np.asarray(times);assert z.min()>-1e-8 and max(z[:,10]+z[:,11])<=1+1e-8
    ncr=(z[:,4]/(1-z[:,4]))/(f0/(1-f0));nuclear=z[:,0]*z[:,4]/f0
    return t,z,ncr,nuclear
def scenarios():
    rows=[];traces={};p=NucleusRibosomeParameters();H=3600.
    for export in NUCLEUS_RANGES['passive_export_per_s']:
      for elong in NUCLEUS_RANGES['elongation_aa_s']:
       for mode in ['import','translation','combined']:
        controls=[0. if mode in ['import','combined'] else 1.,.2 if mode in ['translation','combined'] else 1.,1.,1.]
        sg=[(0,H,[1,1,1,1]),(H,2*H,controls),(2*H,8*H,[1,1,1,1]),(8*H,9*H,controls),(9*H,24*H,[1,1,1,1])]
        q=replace(p,passive_export_per_s=export,elongation_aa_s=elong);t,z,ncr,nuc=component_run(q,sg)
        i2=np.searchsorted(t,4*H);i6=np.searchsorted(t,8*H)
        rows.append(dict(export_per_s=export,elongation_aa_s=elong,mode=mode,nuclear_function_min=float(nuc.min()),nuclear_function_end=float(nuc[-1]),N_C_ratio_after_2h_recovery=float(ncr[i2]),N_C_ratio_after_6h_recovery=float(ncr[i6]),TARDBP_RNA_end=float(z[-1,1]),STMN2_RNA_min=float(z[:,2].min()),STMN2_protein_end=float(z[-1,3]),intact_TDP_end=float(z[-1,0])))
        if export==p.passive_export_per_s and elong==p.elongation_aa_s:traces[mode]=(t,z,ncr,nuc)
    save('component_scenarios.json',dict(rows=rows,n=27,scope='Isolated biosynthesis at clamped ATP, matched basal TDP turnover; not 24h of the full spiking/CAPN model',human_fit=False,parameter_grid=NUCLEUS_RANGES,baseline=asdict(p)))
    for name,(t,z,ncr,nuc) in traces.items():np.savez_compressed(B/(name+'_component.npz'),time_s=t,state=z,nuclear_ratio=ncr,nuclear_function=nuc)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(10,7))
    for name,(t,z,ncr,nuc) in traces.items():
        for a,values in zip(ax.flat,[nuc,z[:,1],z[:,2],z[:,3]]):a.plot(t/H,values,label=name)
    for a,title in zip(ax.flat,['Nuclear functional TDP-43','Cytoplasmic TARDBP RNA','Productive STMN2 RNA','STMN2 body protein']):
        a.set_title(title);a.set_xlabel('hours');a.set_ylabel('relative to baseline');a.grid(alpha=.2)
        a.axvspan(1,2,color='grey',alpha=.15);a.axvspan(8,9,color='grey',alpha=.15)
    ax[0,0].legend();fig.suptitle('Isolated biosynthesis scenarios — unknown rates; ATP clamped');fig.tight_layout();fig.savefig(B/'biosynthesis_scenarios.png',dpi=150)
    print('27 isolated scenarios completed',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['unit','integrated','electrical','scenarios'],required=True);args=ap.parse_args();globals()[args.mode]()
