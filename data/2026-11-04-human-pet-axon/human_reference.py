"""[codex] Assay-specific human PET and myelinated-axon reference components.

No automatic transfer to the immature iPSC soma or extra ATP debit. These
observers/kinetics are accessible from the common API without changing its RHS.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from scipy.special import expit
B=Path(__file__).resolve().parent

def human_pet_axon_sources():
    return {name:json.loads((B/(name+'.json')).read_text(encoding='utf8')) for name in ['Bishu2008_tables','Howells2018_targets']}

def bishu_leucine_reference(region,study=1):
    if study not in (1,2):raise ValueError('Study must be 1 or 2')
    data=human_pet_axon_sources()['Bishu2008_tables']
    hit=[r for r in data['rows'] if r['region']==region]
    if not hit:raise ValueError('Unknown source region')
    return dict(**hit[0]['study'][study-1],region=region,n_subjects=hit[0]['n_subjects'],context=data['context'])

def bishu_leucine_flux(K1_mL_g_min,k23_per_min,k4_per_min,plasma_leucine_nmol_mL):
    v=np.array([K1_mL_g_min,k23_per_min,k4_per_min,plasma_leucine_nmol_mL],dtype=float)
    if not np.isfinite(v).all() or np.any(v<0) or k23_per_min<=0:raise ValueError('Finite nonnegative inputs, k23>0 required')
    K1,k23,k4,Cp=v
    lam=k23/(k23+k4);K=K1*k4/(k23+k4)
    return dict(lambda_plasma=float(lam),tissue_proteolysis_precursor_share=float(1-lam),
        net_label_uptake_mL_g_min=float(K),rCPS_nmol_leucine_g_min=float(K*Cp/lam),
        lysosomal_recycling_yield=None,whole_neuron_calibrated=False)

def bishu_label_observer(time_min,plasma_label,*,K1_mL_g_min,k23_per_min,k4_per_min,
                        blood_total_label,blood_CO2_label,blood_fraction,CO2_partition=.41):
    """Source eq1-2. Stepwise-constant arterial input; tracer starts at zero.

    Units: plasma/blood label per mL, tissue tracer per g. Signals must already
    be consistently decay-corrected. No labeled protein breakdown over90min.
    Output is an observation model, not an ATP/AA synthesis flux owner.
    """
    t=np.asarray(time_min,float);u=np.asarray(plasma_label,float)
    blood=np.asarray(blood_total_label,float);co2=np.asarray(blood_CO2_label,float)
    if t.ndim!=1 or len(t)<2 or u.shape!=t.shape or blood.shape!=t.shape or co2.shape!=t.shape:raise ValueError('Equal 1D time/signal arrays required')
    if not all(np.isfinite(x).all() for x in (t,u,blood,co2)) or np.any(np.diff(t)<=0) or t[0]!=0 or t[-1]>90:raise ValueError('Strict time0..90min and finite signals required')
    if np.any(u<0) or np.any(blood<0) or np.any(co2<0):raise ValueError('Negative tracer activity')
    scalars=np.asarray([K1_mL_g_min,k23_per_min,k4_per_min,blood_fraction,CO2_partition],float)
    if not np.isfinite(scalars).all() or np.any(scalars<0) or not 0<=blood_fraction<1:raise ValueError('Invalid parameters')
    # free label, protein label, cumulative combined loss, cumulative input, constant input.
    A=np.zeros((5,5));A[0,0]=-(k23_per_min+k4_per_min);A[0,4]=K1_mL_g_min
    A[1,0]=k4_per_min;A[2,0]=k23_per_min;A[3,4]=K1_mL_g_min
    state=np.zeros((len(t),5))
    for i,dt in enumerate(np.diff(t)):
        state[i,4]=u[i];state[i+1]=expm(A*dt)@state[i]
    total=(1-blood_fraction)*(state[:,0]+state[:,1]+CO2_partition*co2)+blood_fraction*blood
    return dict(time_min=t,free_label=state[:,0],protein_label=state[:,1],PET_total=total,
        cumulative_input=state[:,3],cumulative_k23_loss=state[:,2],
        conservation_residual=state[:,0]+state[:,1]+state[:,2]-state[:,3],
        assumptions='Bishu2008 eq1-2; piecewise constant arterial samples; supplied decay-corrected tracer; no new energetic debit',
        measured_data=False,human_MN_calibrated=False)

def howells_hcn_kinetics(voltage_mV,*,source_type='motor',temperature_C=36.):
    """Howells2012 Appendix Table2; alpha=A*exp(x), beta=A*exp(-x).

    C=-12.2 is a rate parameter: equilibrium Boltzmann width is abs(C)/2,
    not abs(C). No unsupported Q10 extrapolation is applied.
    """
    if source_type not in ('motor','sensory'):raise ValueError('Unknown source type')
    if temperature_C!=36.:raise ValueError('Source rates are at36C; a new HCN temperature calibration is required')
    v=np.asarray(voltage_mV,float)
    if not np.isfinite(v).all():raise ValueError('Finite voltage required')
    half=-107.3 if source_type=='motor' else -94.2
    x=(v-half)/-12.2;A_per_s=.00522*1000
    q=expit(2*x);z=np.exp(-np.abs(x));tau=z/(A_per_s*(1+z*z))
    return dict(q_inf=q,tau_s=tau,half_activation_mV=half,rate_C_mV=-12.2,
        context='Source-fitted mature human myelinated peripheral axon, not iPSC soma',
        source='Howells2012 PDFpp1637-1638 Appendix/Table2',measured_single_channel_kinetics=False)

def howells_hcn_voltage_steps(time_s,voltage_mV,*,source_type='motor'):
    t=np.asarray(time_s,float);v=np.asarray(voltage_mV,float)
    if t.ndim!=1 or t.shape!=v.shape or len(t)<2 or not np.isfinite(t).all() or t[0]!=0 or np.any(np.diff(t)<=0):raise ValueError('Strict time from0 and matching voltage array required')
    k=howells_hcn_kinetics(v,source_type=source_type);q=np.empty_like(t);q[0]=k['q_inf'][0]
    for i,dt in enumerate(np.diff(t)):
        tau=k['tau_s'][i]
        q[i+1]=k['q_inf'][i]+(q[i]-k['q_inf'][i])*(np.exp(-dt/tau) if tau>0 else 0.)
    # Source Table1 and Appendix ion concentrations. Na selectivity=.097.
    eh=8.314462618*309.15/96485.33212*1000*np.log((.903*4.5+.097*144.2)/(.903*155+.097*9))
    gh=2.95 if source_type=='motor' else 4.1
    return dict(time_s=t,q=q,Ih_pA=gh*q*(v-eh),reversal_mV=float(eh),
        scope='Isolated source HCN gate, ideal imposed voltage; not a full threshold-tracking reproduction',ATP_debit=0.)

def howells_axon_fit(group='normal',*,fit='independent'):
    d=human_pet_axon_sources()['Howells2018_targets']
    if group not in d['group_order']:raise ValueError('Unknown cohort')
    if fit=='independent':v=d['inferred_fit_multipliers'][group]
    elif fit=='equal' and group!='normal':v=d['equal_channel_scaling'][group]
    else:raise ValueError('Unsupported fit/group combination')
    return dict(parameters=v,context=d['context'],measured_protein_abundance=False,
        limitation='Original inverse fit; membrane and paranodal leak differ. Not applied to iPSC conductances or ATP usage.')

def compare_howells_axon_targets(values,*,protocol,group='normal'):
    d=human_pet_axon_sources()['Howells2018_targets']
    if protocol!=d['protocol']:raise ValueError('Protocol mismatch: soma AP/AHP is not extracellular axonal threshold recovery')
    if group not in d['group_order']:raise ValueError('Unknown cohort')
    i=d['group_order'].index(group);out={}
    for key,v in values.items():
        if key not in d['targets'] or not np.isfinite(v):raise ValueError('Unknown endpoint or nonfinite value')
        r=d['targets'][key];target=r['values'][i]
        out[key]=dict(model=float(v),source_center=target[0],difference=float(v-target[0]),source_statistic=r['statistic'],source_values=target,calibration_pass=None)
    return out
