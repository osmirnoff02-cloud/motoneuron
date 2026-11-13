"""[codex] Source-specific model observations, not universal MN constants."""
from pathlib import Path
import json,numpy as np
B=Path(__file__).resolve().parent

def mn_pump_source_constraints():
    return json.loads((B/'constraints_codex.json').read_text(encoding='utf8'))

def safronov1996_open_probability(Na_local_mM,*,temperature_C):
    """Rat P2-P6 inside-out KNa fit; SE is NOT a physiological interval."""
    if not np.isfinite(temperature_C) or not 21<=temperature_C<=24:
        raise ValueError('Measured at 21-24C; no measured Q10 supplied')
    na=np.asarray(Na_local_mM,dtype=float)
    if np.any(~np.isfinite(na)) or np.any(na<0):raise ValueError('Finite nonnegative local Na required')
    with np.errstate(divide='ignore'):
        p=.54*np.exp(-np.logaddexp(0,3.7*(np.log(39.9)-np.log(na))))
    return p

def safronov1996_candidate(api,*,channel_count,tau_s,temperature_C,Ko_mM,Ki_mM):
    """Source-derived Hill/gmax for existing ONE-WAY replay; count/tau unknown.

    Existing gate is normalized [0,1]. Absorb measured Pmax in gmax exactly
    once: count * 139.2pS * 0.54. Use the high-K inside-out context of the Na
    response. The separate44.8pS Ko5.6/Ki106 assay does not share this context.
    Never infer channel count from patch detection.
    """
    safronov1996_open_probability(0.,temperature_C=temperature_C)
    if not(np.isclose(Ko_mM,155.) and np.isclose(Ki_mM,85.)):
        raise ValueError('Na-response source assay: Ko155/Ki85mM; transferring Po to physiological K requires a new explicit assumption')
    if not np.isfinite(channel_count) or channel_count<0 or not np.isfinite(tau_s) or tau_s<=0:
        raise ValueError('Explicit nonnegative channel count and positive unknown tau required')
    return api.KNaCandidateParameters(Na_half_mM=39.9,hill=3.7,tau_s=tau_s,
        gmax_nS=channel_count*.1392*.54,
        provenance='Safronov1996 rat P2-P6: EC50 39.9mM,Hill3.7,Pmax0.54,gamma139.2pS,Ko155/Ki85; count/tau are caller assumptions; one-way replay only')

def ck_inferred_free_nucleotides(ATP_uM,PCr_uM,total_creatine_uM,*,Kobs_CK,Kobs_AK,conditions):
    """CK: ATP*Cr/(ADP*PCr); AK: ATP*AMP/ADP^2. Not total ADP/AMP.

    Requires a caller-supplied validated chemical/speciation context and CK
    near-equilibrium. Ronner1999 used Kobs110/1 at pH7.2,Mg1mM,38C.
    None of those conditions is silently installed as a model default.
    """
    q=np.array([ATP_uM,PCr_uM,total_creatine_uM,Kobs_CK,Kobs_AK],dtype=float)
    if np.any(~np.isfinite(q)) or np.any(q<=0) or total_creatine_uM<PCr_uM:
        raise ValueError('Positive finite pools/constants and total creatine>=PCr required')
    needed=('pH','free_Mg_mM','temperature_C','chemical_forms','CK_near_equilibrium')
    if not isinstance(conditions,dict) or any(k not in conditions for k in needed) or conditions['CK_near_equilibrium'] is not True:
        raise ValueError('Explicit pH/Mg/T/speciation and CK near-equilibrium required')
    for key in needed[:3]:
        if not np.isfinite(conditions[key]):raise ValueError('Finite assay conditions required')
    if not str(conditions['chemical_forms']).strip():raise ValueError('Specify chemical forms')
    Cr=total_creatine_uM-PCr_uM;ADP=ATP_uM*Cr/(Kobs_CK*PCr_uM)
    return dict(free_ADP_uM=float(ADP),free_AMP_uM=float(Kobs_AK*ADP**2/ATP_uM),
        free_creatine_uM=float(Cr),conditions=conditions,total_ADP_measured=False,
        limitation='Equilibrium inference, not direct ADP assay; total nucleotide pools cannot be substituted without speciation')

def simkin2021_postburst_observer(time_s,voltage_mV,*,last_pulse_end_s,holding_mV,temperature_C,AP_count,frequency_Hz):
    """Compare like-defined postburst endpoints; do not fit MN to cortical cells."""
    if not np.isclose(holding_mV,-65) or not 32<=temperature_C<=35 or AP_count!=25 or frequency_Hz!=50:
        raise ValueError('Source protocol: -65mV,32-35C,25AP at50Hz')
    t=np.asarray(time_s,dtype=float);v=np.asarray(voltage_mV,dtype=float)
    if t.ndim!=1 or t.shape!=v.shape or len(t)<2 or np.any(~np.isfinite(t)) or np.any(~np.isfinite(v)) or np.any(np.diff(t)<=0):
        raise ValueError('Finite matched increasing trace required')
    end=last_pulse_end_s
    if not np.isfinite(end) or end<t[0] or end+1>t[-1]:raise ValueError('Need full1s after train')
    post=(t>=end)&(t<=end+1)
    sample=np.r_[v[post],np.interp([end,end+1],t,v)]
    return dict(mAHP_relative_mV=float(np.min(sample)-holding_mV),
        sAHP_1s_relative_mV=float(np.interp(end+1,t,v)-holding_mV),
        source_cell='human iPSC NGN2 excitatory neurons, not motor neurons',human_MN_calibrated=False,
        limitation='Negative voltage changes; source peak window not fully specified, here first1s explicitly. Main text1.4nA vs Methods/caption1.2nA conflict retained.')

def torregrosa2025_energy_reference(control_line):
    if control_line not in ('0%(1)','0%(2)'):raise ValueError('Choose one of two isogenic controls explicitly')
    q=json.loads((B/'metabolic_reference_codex.json').read_text(encoding='utf8'))
    return dict(reference=q['groups'][control_line],source=q['doi'],normalization=q['normalization'],
        limitation=q['uncertainty']+'; '+q['experiments'],rate_per_cell=None,calibrated_full_model=False)

def compare_torregrosa_pathway_fraction(oxidative_ATP_rate,glycolytic_ATP_rate,*,control_line):
    a=np.array([oxidative_ATP_rate,glycolytic_ATP_rate],dtype=float)
    if np.any(~np.isfinite(a)) or np.any(a<0) or a.sum()<=0:raise ValueError('Nonnegative production rates in same units required')
    r=torregrosa2025_energy_reference(control_line)['reference']
    return dict(oxidative_fraction=float(a[0]/a.sum()),
        source_mean_of_well_fractions=r['OxPhos_percent']['mean']/100,
        source_ratio_of_mean_rates=r['ATP_OxPhos']['mean']/(r['ATP_OxPhos']['mean']+r['ATP_glyco']['mean']),
        human_MN_calibrated=False,scope='Context-specific observation; FCCP maximal OCR is not a measured ATP-production ceiling')
