"""[codex] Source-specific constraints; no cross-cell transfer of fitted rates."""
from copy import deepcopy
import math

SOURCES={
 'Telezhkin2018':dict(doi='10.1007/s00424-018-2155-7',
  title='Kv7 channels are upregulated during striatal neuron development and promote maturation of human iPSC-derived neurons',
  path='papers/424_2018_Article_2155.pdf',system='human iPSC striatal/forebrain neurons, NOT spinal motor neurons',
  temperature_C=[21.5,22.5],patch_hold_mV=-70.,
  endpoint='Table 2, week 3 CTR33Qn1, sham versus forced Kv7.2/7.3',
  sham=dict(Vm_mV=[-38.5,1.7,16],Rin_GOhm=[.79,.06,12],Cm_pF=[17.1,2.8,11],
            overshoot_mV=[34.19,5.1,12],halfwidth_ms=[3.,.6,12]),
  kv7=dict(Vm_mV=[-45.5,1.7,11],Rin_GOhm=[.61,.05,10],Cm_pF=[16.3,2.5,10],
           overshoot_mV=[42.2,3.8,10],halfwidth_ms=[2.9,.6,10]),
  format='mean, SEM, n; separate samples, not paired cells',
  inference='Rest hyperpolarization may enable overshooting APs through Na-channel availability. Not a universal monotonic frequency inhibition rule.',
  prohibited=['Using Na-channel activation/inactivation fits as Kv7 gating',
              'Assigning the entire change in 1/Rin to isolated KCNQ conductance',
              'Treating overexpression as normal human-MN channel density']),
 'Yang2023':dict(doi='10.1016/j.stemcr.2023.07.004',
  title='A novel protocol to derive cervical motor neurons from induced pluripotent stem cells for amyotrophic lateral sclerosis',
  path='papers/A-novel-protocol-to-derive-cervical-motor-neurons-from-induced-pluripotent-stem-cells-for.pdf',
  system='human iPSC cervical motor neurons; cohort-specific controls',
  patch_temperature_C=[18.,20.],MEA_temperature_C=37.,
  bath_NaCl_mM=135.,bath_NaHEPES_mM=10.,bath_named_Na_mM=145.,bath_K_mM=5.,
  pipette_named_Na_mM=20.6,pipette_KCl_mM=150.,pipette_ATP_mM=4.,pipette_PCr_mM=10.,
  uncertainty='ATP counterions, pH titration and ion activities not specified here; 20.6mM Na is the named Na2GTP+Na2PCr subtotal, not final free sodium.',
  day28=dict(Rin_MOhm=[546.,62.],rheobase_pA=[10.7,2.3],threshold_mV=[-33.5,1.1],reported_AP_amplitude_mV=[29.,2.5]),
  prohibited=['Combining 37C MEA with 18-20C patch as one protocol','Treating 30mM TEA-sensitive total K as isolated KCNQ',
              'Reusing comparative Table S3 errors instead of each primary source']),
 'ComellaBolla2020':dict(doi='10.1007/s12035-020-01907-4',
  title='Human Pluripotent Stem Cell-Derived Neurons Are Functionally Mature In Vitro and Integrate into the Mouse Striatum Following Transplantation',
  path='papers/s12035-020-01907-4.pdf',system='human striatal hPSC neurons, not MN',
  age_DIV=37,indicator='Fluo-4 AM',loading_uM=2.,sampling_Hz=20.,duration_s=600.,temperature='room temperature',
  averaging_window_s=.25,drift_spline_block_s=25.,optical_inference_tau_s=1.,optical_event_threshold_dFF_percent=.8,
  supplement_path='data/2026-11-12-kcnq-joint-calibration/ComellaBolla2020_ESM1.pdf',
  supplement_burst_ISI_threshold_s=1.,
  supplement_expression=dict(figure='S17a, PDF p21',genes=['KCNQ2','KCNQ3','KCNQ4'],
       status='Expressed at all sampled DIVs, no significant change across differentiation in Ctr-Q33; not evidence of constant current or protein abundance'),
  absolute_calcium_calibrated=False,
  prohibited=['Using the 1s peeling-kernel decay as measured calcium removal','Converting inferred optical events directly into measured voltage AP counts']),
 'Workman2023':dict(doi='10.1016/j.neuron.2023.01.010',
  title='Large-scale differentiation of iPSC-derived motor neurons from ALS and control subjects',
  path='papers/Large-scale-differentiation-of-iPSC-derived-motor-neurons-from-ALS-and-control-subjects.pdf',
  system='433 iPSC lines, ALS/control, day32 mixed MN cultures',
  key_covariates=['sex','batch','cell composition / S100B staining','disease','genotype'],
  raw_data_portal='https://dataportal.answerals.org',access='Publication describes DUA; not evidence of current access',
  prohibited=['RNA fold-change equals channel current fold-change','Inferring pure-MN expression from uncorrected bulk culture counts']),
}

def source_constraints():return deepcopy(SOURCES)

def yang_named_salt_nernst_reference(temperature_C=19.):
    """Illustrative Nernst from named salts only, NOT free-ion calibration."""
    if not 18<=temperature_C<=20:raise ValueError('Yang patch range is 18-20C; do not substitute MEA temperature')
    theta=8.314462618*(temperature_C+273.15)/96485.33212*1000
    return dict(ENa_mV=theta*math.log(145/20.6),EK_mV=theta*math.log(5/150),
       ENa_shift_from_135mM_mV=theta*math.log(145/135),
       scope='Named salt subtotal; unknown ATP counterions/titration/activity corrections remain.')

def telezhkin_reference_comparison(rest_mV,Rin_GOhm,condition='sham'):
    if condition not in ('sham','kv7'):raise ValueError(condition)
    if not math.isfinite(rest_mV) or not math.isfinite(Rin_GOhm) or Rin_GOhm<=0:raise ValueError('Finite Vm and positive Rin required')
    ref=SOURCES['Telezhkin2018'][condition]
    return dict(rest_delta_mV=rest_mV-ref['Vm_mV'][0],Rin_delta_GOhm=Rin_GOhm-ref['Rin_GOhm'][0],
       inference='Descriptive cross-system comparison only; no human-MN fit or pass/fail.')
