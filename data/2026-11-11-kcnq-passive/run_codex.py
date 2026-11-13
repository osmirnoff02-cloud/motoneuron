"""[codex]2026-11-11. Finite Cl and differential KCNQ; explicit candidates only."""
from pathlib import Path
import sys,json,numpy as np
B=Path(__file__).resolve().parent;D=B.parent
sys.path.insert(0,str(D/'neuron-model'))
from model_contracts_codex import load_module
r=load_module(D/'2026-11-04-joint-baseline-calibration/run_codex.py','passive11_runner')
r.B=B;r.PREFIX='CL11_';r.SPARSE=True

def options(sg,profile,fraction=.5,chloride=True):
    o,record=r.api.q3_baseline_inputs(sg,profile)
    o['pump_leak_closure_fraction']=fraction
    if chloride:o['passive_chloride_parameters']=r.api.PassiveChlorideParameters()
    return o,record

def long_atol():
    # Non-feedback cumulative Cl flux ledgers can cross zero flux. Resolving
    # their positive/negative split to 1e-9 uM is unnecessary; concentrations,
    # voltage and every original state retain the original absolute tolerance.
    a=np.full(4425,1e-9)
    for j in range(2):a[4413+j*6+2:4413+j*6+6]=1e-5
    return a

def pilot():
    sg=r.schedule([(0,.02,0,1,0)])
    o,_=options(sg,'Burley',1.,False)
    a=r.run('neutral_old',sg,o,output_step_s=.002,max_step_s=.002,rtol=1e-8,atol=1e-10)
    o['passive_chloride_parameters']=r.api.PassiveChlorideParameters()
    b=r.run('neutral_Cl_zero_g',sg,o,output_step_s=.002,max_step_s=.002,rtol=1e-8,atol=1e-10)
    d=float(np.max(abs(a['state']-b['state'][:,:4413])))
    r.save('neutral_comparison',dict(max_absolute=d,old_states=4413,new_states=4425,
        zero_chloride_drift=float(np.max(abs(b['state'][:,4413:]-b['state'][0,4413:]))),passed=d<1e-6))
    assert d<1e-6
    o,_=options(sg,'Burley',.5)
    return r.run('pilot',sg,o,output_step_s=.002,max_step_s=.002,rtol=1e-8,atol=1e-10)

def rest(profile='Selvaraj-Con2',fraction=.5,hours=72.):
    end=hours*3600.;edges=np.unique(np.r_[0,.01,.1,1,10,60,600,3600,np.arange(21600,end,21600),end])
    edges=edges[(edges>=0)&(edges<=end)]
    sg=r.schedule([(a,b,0,1,0) for a,b in zip(edges[:-1],edges[1:])]);o,record=options(sg,profile,fraction)
    z=r.run(f'rest_{profile}_f{fraction}_{hours:g}h',sg,o,output_step_s=300,max_step_s=300,
        rtol=1e-7,atol=long_atol(),center_solver_coordinates=True,retain_dense_solver_output=False,diagnostic_rates=True)
    r.save(f'rest_{profile}_f{fraction}_{hours:g}h_provenance',record)
    return z

def checkpoint(profile,fraction,hours):
    return json.loads((B/f'CL11_rest_{profile}_f{fraction}_{hours:g}h_checkpoint_codex.json').read_text())

def electrical(profile,amp,fraction=.5,hours=0.,multiplier=1.):
    from scipy.signal import find_peaks
    cp=checkpoint(profile,fraction,hours) if hours else None
    start=0. if cp is None else cp['time_s'];on=start+.2;off=on+(.5 if profile=='Burley' else 1.)
    sg=r.schedule([(start,on,0,1,0),(on,off,0,1,0),(off,off+1.,0,1,0)]);o,_=options(sg,profile,fraction)
    name=f'electric_{profile}_{amp:g}pA_f{fraction}_{hours:g}h_m{multiplier:g}'
    z=r.run(name,sg,o,checkpoint=cp,differential_KCNQ_activity_multiplier=multiplier,
        electrical_schedule=np.c_[sg[:,:2],[0,amp,0],[0,0,0]],output_step_s=.0002,max_step_s=.00125,
        rtol=1e-9,atol=1e-11,center_solver_coordinates=True)
    t=z['time_s'];v=z['state'][:,185];f=r.features_mod.features(z,on,off,profile)
    peaks,_=find_peaks(v,height=0 if profile=='Burley' else -20,prominence=20)
    result=dict(profile=profile,current_pA=amp,NaK_share=fraction,prepared_h=hours,
      differential_KCNQ_multiplier=multiplier,features=f,after=int(sum(t[peaks]>=off)),
      post_min_relative_mV=float(v[t>=off].min()-v[0]),post_end_relative_mV=float(v[-1]-v[0]),
      no_reset=bool(cp is None or np.array_equal(z['state'][0],cp['state'])),
      scope='50pA is historical model regression, not Selvaraj measurement. Multiplier changes ONLY differential RNA/surface KCNQ; not complete M-block.')
    r.save('CL11_'+name+'_features',result);return z

def loads(branch,fraction=.5,hours=72.,multiplier=1.):
    if branch not in ('control','continuous','repeated'):raise ValueError(branch)
    profile='Selvaraj-Con2';cp=checkpoint(profile,fraction,hours);start=cp['time_s']
    edges=np.unique(np.r_[np.arange(0,36001,3600),1800,5400,9000]);rows=[]
    for a,b in zip(edges[:-1],edges[1:]):
        active=(branch=='continuous' and a<10800) or (branch=='repeated' and any(s<=a<s+3600 for s in (0,7200,14400)))
        rows.append((start+a,start+b,.0001 if active else 0.,1.,0.))
    sg=r.schedule(rows);o,_=options(sg,profile,fraction)
    z=r.run(f'load_{branch}_f{fraction}_m{multiplier:g}',sg,o,checkpoint=cp,
      differential_KCNQ_activity_multiplier=multiplier,output_step_s=300,max_step_s=300,
      rtol=1e-7,atol=long_atol(),center_solver_coordinates=True,retain_dense_solver_output=False)
    assert np.array_equal(z['state'][0],cp['state'])
    r.save(f'load_{branch}_f{fraction}_m{multiplier:g}_protocol',dict(no_reset=True,schedule=sg,
      active_s=0 if branch=='control' else 10800,equal_opening_not_equal_Ca=True,
      differential_only_not_XE991_block=True))
    return z

def suite(hours=0.,fraction=.5):
    refs=json.loads((D/'2026-10-29-joint-recovery/electrical_codex.json').read_text())['rows']
    for ref in refs:electrical(ref['profile'],ref['current_pA'],fraction,hours)

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='rest':rest(sys.argv[2],float(sys.argv[3]),float(sys.argv[4]))
    elif cmd=='suite':suite(float(sys.argv[2]),float(sys.argv[3]))
    elif cmd=='electrical':electrical(sys.argv[2],float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6]))
    elif cmd=='loads':loads(sys.argv[2],float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]))
    else:globals()[cmd]()
