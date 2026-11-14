"""Codex 2026-10-20: opt-in bidirectional Ca/spikes; immutable prior API.

Unknown conductances, geometry of accessible calcium and SK kinetics remain
scenario parameters. Surface Q is NOT inferred from ADAR2 RNA abundance.
"""
from pathlib import Path
from dataclasses import dataclass,asdict
import importlib.util
import sys
from functools import lru_cache
import numpy as np
from scipy.linalg import expm
from scipy.signal import find_peaks

B=Path(__file__).resolve().parent
sys.path.insert(0,str(B.parent/'neuron-model'))
from model_contracts import mechanism_path
spec=importlib.util.spec_from_file_location("_ampa_prior_coupled",
    B.parent/"2026-10-20-ampa-current-permeability/model_api.py")
prior=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=prior
spec.loader.exec_module(prior)
for _name in dir(prior):
    if not _name.startswith("_") and _name not in globals():
        globals()[_name]=getattr(prior,_name)
from recovery_model import nr,h,recovery_features,spike_indices
from electrical_model import derived,spike_features
from waveform_fit import features,FEATURE_NAMES

def _ensure_coupled_neuron():
    DLL=mechanism_path('ipsc-coupled-2026-10-20')
    if not hasattr(h,"CoupledCa20261020"):
        h.nrn_load_dll(str(DLL))

@dataclass(frozen=True)
class CouplingParameters:
    g_ampa_nS:float=.3
    surface_glua2_fraction:float=.5
    cp_conductance_ratio:float=3.
    pf_cp:float=.035
    pf_ci:float=.001
    eca_mV:float=120.
    g_vgcc_nS:float=.03
    g_sk_nS:float=.15
    replace_adaptation:float=0.
    basal_ca_uM:float=.08
    kappa:float=50.
    gamma_per_s:float=5.1
    cytosol_fraction:float=.1
    shell_fraction:float=.05
    exchange_ms:float=50.
    sk_half_uM:float=.42
    sk_hill:float=5.2
    sk_tau_ms:float=10.
    vg_half_mV:float=-20.
    vg_slope_mV:float=6.
    vg_tau_ms:float=2.

    def validate(self):
        if not np.isfinite(list(asdict(self).values())).all():raise ValueError("Finite parameters required")
        for key in ("g_ampa_nS","g_vgcc_nS","g_sk_nS","kappa"):
            if getattr(self,key)<0:raise ValueError(key+" must be nonnegative")
        for key in ("surface_glua2_fraction","replace_adaptation","pf_cp","pf_ci"):
            if not 0<=getattr(self,key)<=1:raise ValueError(key+" requires a fraction")
        for key in ("cytosol_fraction","shell_fraction"):
            if not 0<getattr(self,key)<1:raise ValueError(key+" must lie strictly between 0 and 1")
        for key in ("gamma_per_s","basal_ca_uM","cp_conductance_ratio","exchange_ms",
                    "sk_half_uM","sk_hill","sk_tau_ms","vg_slope_mV","vg_tau_ms"):
            if getattr(self,key)<=0:raise ValueError(key+" must be positive")
        if self.eca_mV<=0:raise ValueError("Positive calcium reversal required")

@lru_cache(maxsize=32)
def ampa_opening(dt,pre,duration,post,pulse_ms,period_ms):
    """Exact C/O/D propagator with left-held input; shared by CP and CI.

    Rates (/s): opening 1000 when stimulated; closing 200, desensitization
    100, recovery 10. These are inherited scenario rates, not human fits.
    Returns breakpoints for NEURON stepwise Vector.play.
    """
    n=round((pre+duration+post)/dt)
    t=np.arange(n+1,dtype=float)*dt
    drive=(t>=pre-1e-9)&(t<pre+duration-1e-9)&(((t-pre)%period_ms)<pulse_ms-1e-9)
    # Avoid modulo round-off at an intended period boundary.
    phase=np.round((t-pre)/dt).astype(np.int64)%round(period_ms/dt)
    drive=(t>=pre-1e-9)&(t<pre+duration-1e-9)&(phase<round(pulse_ms/dt))
    mats=[]
    for a in (0.,1000.):
        mats.append(expm(np.array([[-a,200.,10.],[a,-300.,0.],[0.,100.,-10.]])*dt/1000))
    state=np.array([1.,0.,0.]);o=np.zeros(n+1)
    for k in range(n):
        state=mats[int(drive[k])]@state
        o[k+1]=state[1]
    if np.min(o)<-1e-12 or np.max(o)>1+1e-12:raise RuntimeError("Invalid AMPA occupancy")
    return t,o

def run_coupled(profile="Burley",surface_Q=.001,parameters=None,*,current_pA=25.,
                duration_ms=None,pre_ms=200.,post_ms=1800.,dt_ms=.005,
                ampa=False,feedback=True,pulse_ms=5.,period_ms=100.,
                disable_extension=False):
    """Native NEURON replay with simultaneous AMPA/VGCC/SK feedback.

    Default duration is the original source-specific electrical protocol.
    The comparative Q sweep explicitly requests the same 1s stimulus for
    all three profiles. No parameter fitting occurs here.
    """
    par=CouplingParameters() if parameters is None else parameters
    par.validate()
    if not np.isfinite(surface_Q) or not 0<=surface_Q<=1:raise ValueError("Q fraction must be in [0,1]")
    p,c=refined_profile(profile)
    bur=profile=="Burley"
    duration_ms=(500. if bur else 1000.) if duration_ms is None else duration_ms
    if (not np.isfinite([current_pA,duration_ms,pre_ms,post_ms,dt_ms,pulse_ms,period_ms]).all()
        or duration_ms<=0 or pre_ms<100 or post_ms<100 or dt_ms<=0
        or pulse_ms<=0 or period_ms<=pulse_ms):raise ValueError("Invalid protocol")
    init=float(c[5] if bur else c[2])
    _ensure_coupled_neuron()
    h.celsius=23
    s=h.Section(name="ipsc_ca_coupled")
    s.L=s.diam=float(np.sqrt(c[0]*100/np.pi))
    s.cm=1;s.nseg=1;s.insert("ipscmulti")
    k=s(.5).ipscmulti
    for name,value in zip(["gna","vm","km","gk","vn","kn","vh","kh","tm","thmin",
                           "threc","tn","ga","ta","tmfast","tndeact"],p):
        setattr(k,name,value/1000 if name in ("gna","gk","ga") else value)
    gl,el,hold=derived(p,c)
    k.gl=gl/1000;k.el=el;k.ena0=c[3];k.ek0=c[4]
    stim=h.IClamp(s(.5));stim.delay=pre_ms;stim.dur=duration_ms;stim.amp=current_pA/1000
    bias=h.IClamp(s(.5));bias.delay=0;bias.dur=1e9;bias.amp=hold/1000 if bur else 0
    bridge=h.CoupledCa20261020(s(.5))
    h.setpointer(k._ref_w,"w_old",bridge)
    cp=(1-par.surface_glua2_fraction*(1-surface_Q))**4
    gcp=par.g_ampa_nS*cp*par.cp_conductance_ratio if ampa else 0.
    gci=par.g_ampa_nS*(1-cp) if ampa else 0.
    total=gcp+gci
    gca=(par.pf_cp*gcp+par.pf_ci*gci)*60/(60+par.eca_mV)
    mono=total-gca
    geometric_volume=float(np.pi*s.diam*s.diam*s.L/4)
    volume=geometric_volume*par.cytosol_fraction
    mapping=dict(gmono=mono/1000,gca=gca/1000,
        emono=-gca*par.eca_mV/mono if mono else 0,eca=par.eca_mV,
        gvg=par.g_vgcc_nS/1000,gsk=par.g_sk_nS/1000,ek=float(c[4]),
        g_adapt=float(p[12]*c[0]/1000),replace_adapt=par.replace_adaptation,
        feedback=float(feedback),c0=par.basal_ca_uM,kappa=par.kappa,
        gamma=par.gamma_per_s,volume=volume,shell_fraction=par.shell_fraction,
        exchange_ms=par.exchange_ms,sk_half=par.sk_half_uM,sk_hill=par.sk_hill,
        sk_tau=par.sk_tau_ms,vg_half=par.vg_half_mV,vg_slope=par.vg_slope_mV,
        vg_tau=par.vg_tau_ms,vref=init,w0=float(1/(1+np.exp(-(init+20)/5))))
    if disable_extension:
        mapping.update(gmono=0.,gca=0.,gvg=0.,gsk=0.,replace_adapt=0.)
    for name,value in mapping.items():setattr(bridge,name,value)
    players=[]
    if ampa and not disable_extension:
        tt,oo=ampa_opening(dt_ms,pre_ms,duration_ms,post_ms,pulse_ms,period_ms)
        tv=h.Vector(tt);ov=h.Vector(oo);ov.play(bridge._ref_opening,tv,0)
        players=[tv,ov]
    vectors={"t":h.Vector().record(h._ref_t),"v":h.Vector().record(s(.5)._ref_v)}
    for name in ("cs","cb","z","m","iampa","icaampa","ivg","isk","iadapt","qampa","qvg","qout","opening"):
        vectors[name]=h.Vector().record(getattr(bridge,"_ref_"+name))
    h.dt=dt_ms;h.steps_per_ms=1/dt_ms;h.secondorder=0
    h.finitialize(init);h.continuerun(pre_ms+duration_ms+post_ms)
    out={name:np.array(vec) for name,vec in vectors.items()}
    for v in players[1:]:v.play_remove()
    h.delete_section(sec=s)
    if not all(np.isfinite(x).all() for x in out.values()):raise RuntimeError("Nonfinite simulation")
    if min(np.min(out["cs"]),np.min(out["cb"]))<-1e-6:raise RuntimeError("Negative free Ca")
    sf=par.shell_fraction
    stored=(1+par.kappa)*(sf*(out["cs"]-par.basal_ca_uM)+(1-sf)*(out["cb"]-par.basal_ca_uM))
    balance=stored-(out["qampa"]+out["qvg"]-out["qout"])
    sourcefeatures=spike_features(out["v"],dt_ms,pre_ms,duration_ms)
    count=int(sourcefeatures["zero_crossings"]) if bur else int(features(out["v"],dt_ms,pre_ms,duration_ms,c[2])[0])
    # The Selvaraj helper order is verified explicitly instead of assuming count at index 0.
    if not bur:
        sourcefeatures=dict(zip(FEATURE_NAMES,features(out["v"],dt_ms,pre_ms,duration_ms,c[2]).tolist()))
        count=int(sourcefeatures["count"])
    rec=recovery_features(out["t"],out["v"],onset_ms=pre_ms,offset_ms=pre_ms+duration_ms)
    peakidx=spike_indices(out["t"],out["v"],rec["baseline_mV"],pre_ms)
    post_ap=int(np.count_nonzero(out["t"][peakidx]>=pre_ms+duration_ms))
    out["summary"]=dict(profile=profile,surface_Q=surface_Q,cp_fraction=cp,
        source_count=count,all_ap_count=len(peakidx),post_ap_count=post_ap,
        spike_times_ms=out["t"][peakidx].tolist(),
        peak_local_ca_uM=float(out["cs"].max()),peak_bulk_ca_uM=float(out["cb"].max()),
        bulk_auc_uM_s=float(np.trapezoid(out["cb"]-par.basal_ca_uM,out["t"])/1000),
        ca_ampa_entry_uM_unbuffered=float(out["qampa"][-1]),
        ca_vgcc_entry_uM_unbuffered=float(out["qvg"][-1]),
        max_balance_error_uM=float(np.max(np.abs(balance))),
        min_ca_uM=float(min(out["cs"].min(),out["cb"].min())),
        peak_SK_pA=float(1000*np.max(out["isk"])),
        geometric_volume_um3=geometric_volume,accessible_volume_um3=volume,
        max_voltage_mV=float(np.max(out["v"])),features=sourcefeatures,recovery=rec)
    out["parameters"]=asdict(par)
    out["protocol"]=dict(profile=profile,current_pA=current_pA,duration_ms=duration_ms,
        pre_ms=pre_ms,post_ms=post_ms,dt_ms=dt_ms,ampa=ampa,feedback=feedback,
        pulse_ms=pulse_ms,period_ms=period_ms,disable_extension=disable_extension,
        temperature_C=23,baseline="held -70" if bur else "free rest",
        measured_human_calcium_fit=False)
    return out

def compatible_scenarios(profile=None):
    """Return source-compatible scenarios with exact counts at 15/25/50pA.

    Compatibility is conditional on the tested source protocols; it is not
    acceptance of calcium parameters as measured human values.
    """
    import json
    final=B/"final-compatibility.json"
    if final.exists():
        table=json.loads(final.read_text(encoding="utf-8"))
        return [dict(draw=r["draw"],profile=r["profile"],counts_exact=True,
                     verification="refined dt .01; main .01/.005",source_counts=r["source_counts"])
                for r in table if r["accepted"] and (profile is None or r["profile"]==profile)]
    table=json.loads((B/"broad-compatibility.json").read_text(encoding="utf-8"))
    return [dict(r) for r in table if r["counts_exact"] and (profile is None or r["profile"]==profile)]

def scenario_parameters(scenario_id):
    import json
    doc=json.loads((B/"design.json").read_text(encoding="utf-8"))
    row=next((r for r in doc["draws"] if r["id"]==scenario_id),None)
    if row is None:raise ValueError("Unknown scenario: "+scenario_id)
    return CouplingParameters(**{k:v for k,v in row.items() if k!="id"})

def run_compatible(profile,scenario_id="nominal",surface_Q=.001,**protocol):
    """Explicitly select an engineering-compatible coupled scenario.

    For an AMPA challenge pass ampa=True and define its duration. Surface_Q
    is a fraction of surface GluA2 subunits, not a receptor count or ADAR2 RNA.
    """
    if not any(r["draw"]==scenario_id for r in compatible_scenarios(profile)):
        raise ValueError("Scenario has not passed the declared electrical compatibility checks")
    return run_coupled(profile,surface_Q,scenario_parameters(scenario_id),**protocol)
