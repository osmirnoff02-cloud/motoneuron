"""[codex] Capture the unchanged full RHS for explicitly reduced screening only."""
from pathlib import Path
import sys,inspect,json,time
import numpy as np
from scipy.integrate import solve_ivp
B=Path(__file__).resolve().parent;D=B.parent
sys.path.insert(0,str(D/'neuron-model'))
from model_contracts import load_module
old=load_module(D/'2026-11-13-coupled-assumption-validation/run.py','joint13_previous')
host=load_module(D/'2026-10-28-takeuchi-balaji/adaptive_recycling_neuron.py','joint13_host')
ARGS=('row','target','erow','srow','elecrow','ligand','nuccontrol','cyclecontrol','biocontrol','shcontrol','subcontrol','nrcontrol','isrcontrol','latecontrol','lrcontrol','ltcontrol')
class Captured(Exception):pass
def capture(opts,sg):
    saved=host.solve_ivp; box={}
    def intercept(*args,**kw):
        f=inspect.currentframe().f_back.f_locals
        box.update({k:f[k] for k in ('rhs','base','config','ctx','current_kp','ionbase','ionvec','ioncv','background_ref','chloride_ref','absolute_delta0','kbase','mv','rpv','rnav','bpv','budgetref','ckv','water_scale')+ARGS})
        box['connection_factor']=f['g'].connection_factor_um
        raise Captured()
    host.solve_ivp=intercept
    try:old.r.api.run_integrated_neuron(sg,**opts,retain_dense_solver_output=False)
    except Captured:pass
    finally:host.solve_ivp=saved
    assert box
    return box

# Ion/voltage/Ca/energy fast physical variables; all other states frozen only
# in the screening approximation. No claim of 4425-state conservation here.
LIVE=np.array(sorted(set([0,105,106,113,114,120,121,122,134,135,136,137,138,239,243,247,248,257,258,282,283,284,315,316,317,318,319,362,363,4413,4414,4419,4420]+list(range(185,203))+list(range(219,223)))))
def rhs_of(b):
    def f(t,y,amp):
        z=b['base'].copy();z[LIVE]=y
        return b['rhs'](z,b['row'],b['target'],b['erow'][2:],b['srow'],np.array([amp,0.]),b['ligand'],b['nuccontrol'],b['cyclecontrol'],b['biocontrol'],b['shcontrol'],b['subcontrol'],b['nrcontrol'],b['isrcontrol'],b['latecontrol'],b['lrcontrol'],b['ltcontrol'])[LIVE]
    return f
def run(b,profile,amp):
    f=rhs_of(b);dur=.5 if profile=='Burley' else 1.;ends=[0,.2,.2+dur,1.2+dur]
    y=b['base'][LIVE].copy();ts=[];zs=[]
    for a,c,current in zip(ends[:-1],ends[1:],[0,amp,0]):
        grid=np.unique(np.r_[np.arange(a,c,.0002),c])
        sol=solve_ivp(lambda t,y:f(t,y,current),(a,c),y,method='LSODA',rtol=1e-8,atol=1e-9,max_step=.0025,t_eval=grid)
        assert sol.success,sol.message
        y=sol.y[:,-1];ts.extend(sol.t);zs.extend(sol.y.T)
    t=np.array(ts);small=np.array(zs)
    result=dict(time_s=t,soma_voltage_mV=small[:,np.where(LIVE==185)[0][0]],config=b['config'])
    voltage=result['soma_voltage_mV'];ix=np.flatnonzero((voltage[:-1]<0)&(voltage[1:]>=0))
    result['voltage_crossing_times_s']={'soma':t[ix]+(-voltage[ix])/(voltage[ix+1]-voltage[ix])*(t[ix+1]-t[ix])}
    feat=old.r.features_mod.features(result,.2,.2+dur,profile)
    return t,small,feat
if __name__=='__main__':
    profile='Selvaraj-Con2';sg=old.r.schedule([(0,.2,0,1,0),(.2,1.2,0,1,0),(1.2,2.2,0,1,0)])
    o,_=old.options(sg,'tau_0p02',profile);started=time.monotonic();b=capture(o,sg)
    print('Captured RHS',round(time.monotonic()-started,2),flush=True)
    t,z,f=run(b,profile,25);p=D/'2026-11-13-coupled-assumption-validation/S13_electric_tau_0p02_Selvaraj-Con2_25_trace.npz'
    ref=np.load(p);diff={}
    for i in [185,192,200,220,199,219,120,136]:
        diff[str(i)]=float(np.max(abs(z[:,np.where(LIVE==i)[0][0]]-np.interp(t,ref['time_s'],ref['state'][:,i]))))
    report=dict(author='codex',live_states=LIVE.tolist(),features=f,max_differences=diff,elapsed_s=time.monotonic()-started,scope='Reduced ranking only; full physiological acceptance requires unmodified full integration.')
    (B/'reduced_pilot.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
