"""[codex]2026-11-03. Source-assay checks, NOT fitted human conductances."""
import numpy as np

def perinatal_K_reference():
    return dict(title='Development of Potassium Conductances in Perinatal Rat Phrenic Motoneurons',
      doi='10.1152/jn.2000.83.6.3497',system='Perinatal rat phrenic motoneurons, whole-cell slices, 27 +/- 1 C',
      source='Table 2, p3503; Methods pp3498-3499; Figs7-9',uncertainty='SEM; paired percentage changes are NOT ratios of group means',
      AHP_internal='0.1 mM BAPTA, no added Ca, 126 mM K-gluconate, 10 mM phosphocreatine; distinct from 10 mM BAPTA voltage-clamp solution',
      bath_K_mM=3.,bath_Ca_mM=1.5,
      rows=[dict(blocker='apamin',dose_uM=1.,n=7,AP_halfwidth_ms=[2.2,2.2],AP_halfwidth_SEM=[.2,.2],mAHP_mV=[2.2,.2],mAHP_SEM=[.2,.1],frequency_Hz=[15.8,26.2],frequency_SEM=[1.1,3.],paired_frequency_change_percent=68.,paired_change_SEM=29.),
            dict(blocker='omega-conotoxin GVIA',dose_uM=.3,n=4,AP_halfwidth_ms=[2.3,2.4],AP_halfwidth_SEM=[.2,.2],mAHP_mV=[1.8,.1],mAHP_SEM=[.3,.1],frequency_Hz=[15.7,20.],frequency_SEM=[1.,1.8],paired_frequency_change_percent=35.,paired_change_SEM=10.),
            dict(blocker='iberiotoxin',dose_uM=.3,n=6,AP_halfwidth_ms=[1.8,2.2],AP_halfwidth_SEM=[.1,.1],mAHP_mV=[2.6,2.6],mAHP_SEM=[.6,.7],frequency_Hz=[14.7,15.7],frequency_SEM=[1.1,1.],paired_frequency_change_percent=11.,paired_change_SEM=4.)],
      interpretation='SK/N-type-Ca relation to mAHP differs from BK contribution to repolarization. Minimum current supporting repetitive firing; not matched 25 pA protocol.',
      limitations=['No raw traces or individual paired data in supplied PDF','Voltage-clamp space-clamp limitations; authors could not reliably record KCa currents','No human kinetic fit; no assignment of these currents to KCNQ2','mAHP duration is not an exponential time constant','fAHP apparent reversal -50 mV is not EK'])

def luthi_slow_K_reference():
    return dict(title='A Slowly Inactivating Potassium Current in CA3 Pyramidal Cells of Rat Hippocampus In Vitro',
      doi='10.1523/JNEUROSCI.16-02-00586.1996',system='Rat P4-6, organotypic hippocampal culture 3-6 weeks, CA3, 33 C, sharp-electrode clamp',
      source='Figs3,4,6,7 and Results pp589-591',uncertainty='SEM; no raw individual traces',
      decay_tau_s=7.5,decay_tau_SEM_s=.5,decay_n=35,individual_tau_range_s=[2.,15.],
      recovery_tau_s=9.,recovery_n=3,recovery_protocol='-120 mV prepulse, normalized to response after 30 s; readout -60 mV',
      abstract_half_recovery_s=5.5,
      activation_Vhalf_mV=-65.1,activation_SEM_mV=2.6,activation_k_per_mV=.20,
      inactivation_Vhalf_mV=-84.3,inactivation_SEM_mV=1.,inactivation_k_per_mV=-.12,
      voltage_n=7,activation_valid_mV=[-90.,-55.],
      reversal=[dict(Kout_mM=2.7,EK_mV=-93.7,SEM_text_mV=1.9,SEM_caption_mV=1.7,n=3),dict(Kout_mM=8.,EK_mV=-69.6,SEM_text_mV=1.,n=4)],
      limitations=['Different cell type; molecular identity not established; not identified KCNQ/M current',
       'Tentative activation fit; no extrapolation above -55 mV',
       '5.5 s half-recovery is not the 9 s fitted time constant',
       'Strophanthidin 2-7 uM had no effect in n=4; this current is not an electrogenic pump',
       'TEA block is modest at 10 mM and substantial at 50 mM despite broad wording in abstract'])

def luthi_slow_K_protocol(prepulse_s,decay_s,voltage_mV=-60.):
    """Normalized source-protocol observer, not a human channel current.

    Recovery and decay were measured in different voltage phases. Their tau
    values cannot become a universal voltage-independent gating equation.
    """
    p=np.asarray(prepulse_s,dtype=float);t=np.asarray(decay_s,dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(t).all() or np.any(p<0) or np.any(p>30) or np.any(t<0) or np.any(t>30):
        raise ValueError('Source observer restricted to nonnegative 0-30 s phases')
    if not np.isfinite(voltage_mV) or not -90<=voltage_mV<=-55:raise ValueError('No justified activation extrapolation outside -90..-55 mV')
    return dict(recovered_relative_to_30s=(1-np.exp(-p/9.))/(1-np.exp(-30/9.)),
      current_decay_relative=np.exp(-t/7.5),activation=1/(1+np.exp(-.20*(voltage_mV+65.1))),
      steady_inactivation=1/(1+np.exp(.12*(voltage_mV+84.3))),
      source=luthi_slow_K_reference(),human_kinetics_calibrated=False)

def compare_perinatal_K_response(blocker,control,treated):
    """Descriptive comparison only: group SEM cannot yield a paired fit score."""
    rows=[x for x in perinatal_K_reference()['rows'] if x['blocker']==blocker]
    if not rows:raise ValueError('Unknown blocker; use an exact source-assay label')
    row=rows[0]
    keys=['AP_halfwidth_ms','mAHP_mV','frequency_Hz']
    if any(k not in control or k not in treated for k in keys):raise ValueError('Width, mAHP and frequency required, with matched source protocol')
    for sample in (control,treated):
        if not all(np.isfinite(sample[k]) and sample[k]>=0 for k in keys) or sample['AP_halfwidth_ms']==0:
            raise ValueError('Finite positive AP width and nonnegative AHP magnitude/frequency required')
    return dict(source=row,model_difference={k:float(treated[k]-control[k]) for k in keys},
      observed_group_difference={k:row[k][1]-row[k][0] for k in keys},
      fitted=False,warning='Comparison requires the rat assay conditions; no inference of human channel density or statistical pass from aggregate SEM.')
