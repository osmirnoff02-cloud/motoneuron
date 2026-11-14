"""[codex] 2026-11-11. Four primary sources, explicit assay/species boundaries."""
import numpy as np
from pathlib import Path
import json

RANGES={'pump_leak_closure_fraction':[.25,.5,.75,1.]}

def source_constraints():
    return dict(
      Krishnan2015=dict(title='Electrogenic properties of the Na+/K+ ATPase controls transitions between normal and pathological brain states',
        doi='10.1152/jn.00460.2014',file='papers/krishnan2015.pdf',pages='PDF5–10, equations1–7; References',
        scope='Cortical network model and rat hippocampal slices; pump/passive-current equilibrium architecture, not human MN calibration',
        application='Optional fixed pump capacity derived with passive Na/K conductance; 3Na out/2K in/1ATP, same current and material owners',
        not_transferred='Absolute Imax, network seizure kinetics, glial clearance and extracellular geometry'),
      Hamada2003=dict(title='Properties of the Na+/K+ pump current in small neurons from adult rat dorsal root ganglia',
        doi='10.1038/sj.bjp.0705170',file='papers/hamada2003.pdf',pages='Methods; Fig3–5; discussion; References',
        scope='Adult rat small DRG; whole-cell, -40mV, Na pipette5–100mM; K currents/NCX suppressed; ATP5mM/PCr2.5mM supplied by pipette',
        application='Independent two-component Na-response benchmark, not a human kinetic replacement',
        half_Na_mM=[6.7,67.6],Hill_fixed=3.,Imax_pA_pF=[1.6222,.9998],recording_temperature_C=None,
        caveat='Pipette Na is not directly measured cytosolic Na; functional components do not uniquely measure isoform abundance'),
      Baranauskas2010=dict(title='Na/K ATPase activity is coordinated with the persistent sodium current amplitude',
        doi='10.1097/WNR.0b013e32833904dd',file='papers/baranauskas2010.pdf',pages='Fig1–3, discussion and References',
        scope='Rat prefrontal layerV–VI pyramidal cells; n13 for correlation',
        R_squared=.62,P_less_than=.002,partial_R_squared=.32,recording_temperature_C=[22.,24.],
        application='Joint pump/channel calibration requirement',
        caveat='Correlation does not identify a dynamic feedback law, KCNQ coupling or a fixed pump/NaP ratio'),
      Wei2023=dict(title='Aerobic glycolysis is the predominant means of glucose metabolism in neuronal somata, which protects against oxidative damage',
        doi='10.1038/s41593-023-01476-4',file='papers/Aerobic-glycolysis-is-the-predominant-means-of-glucose-metabolism-in-neuronal-somata-which.pdf',
        pages='Fig1–4, Methods, ExtendedData2–4; References',
        scope='Mouse brain somata vs terminals, cortical imaging and Pkm2 manipulation; not human motor neurons',
        application='Separate compartment-specific assay observer; retain uncertain metabolic capacities',
        caveat='Sequential inhibitor-induced ATP-sensor drops are not steady ATP-production fractions; no matched AMPA/volume/ATP-ADP-PCr calibration',
        repository='https://data.mendeley.com/datasets/x844x4ky8x/2'))

def hamada_pump_density(pipette_Na_mM):
    x=np.asarray(pipette_Na_mM,dtype=float)
    if not np.isfinite(x).all() or np.any(x<0):raise ValueError('Nonnegative finite pipette Na required')
    n=x**3
    return dict(current_pA_pF=1.6222*n/(6.7**3+n)+.9998*n/(67.6**3+n),
        measured_range_mM=[5.,100.],extrapolation=(x<5)|(x>100),human_calibrated=False,
        conditions=source_constraints()['Hamada2003']['scope'])

def krishnan_pump_activation(Na_i_mM,K_o_mM):
    na,ko=np.broadcast_arrays(np.asarray(Na_i_mM,float),np.asarray(K_o_mM,float))
    if not np.isfinite(na).all() or not np.isfinite(ko).all() or np.any(na<0) or np.any(ko<0):raise ValueError('Nonnegative finite concentrations required')
    return (ko/(2.5+ko))**2*(na/(20.+na))**3

def wei_inhibitor_index(baseline_signal,after_oligomycin_signal,after_combined_signal):
    """Assay-specific endpoint index, not flux partition or ATP concentration."""
    a,b,c=np.broadcast_arrays(np.asarray(baseline_signal,float),np.asarray(after_oligomycin_signal,float),np.asarray(after_combined_signal,float))
    if not all(np.isfinite(v).all() for v in (a,b,c)) or np.any(a<=c):raise ValueError('Finite signals with positive total inhibitor-induced drop required')
    idx=(a-b)/(a-c)
    return dict(OXPHOS_inhibitor_index=idx,monotone_inhibitor_drops=(a>=b)&(b>=c),
        clipped=False,ATP_flux_fraction=False,ATP_concentration=False,
        scope='Wei2023 sequential inhibitor assay; optical calibration, timing, compensatory metabolism and compartment must match')

def wei_mouse_reference_indices():
    """Original cell-level index summaries, never human metabolic parameters."""
    return dict(data=json.loads((Path(__file__).resolve().parent/'Wei_assay_summary.json').read_text(encoding='utf8')),
        source='https://data.mendeley.com/datasets/x844x4ky8x/2',licence='CC-BY-4.0',
        species='mouse',human_MN_calibrated=False,raw_time_series_available_in_these_files=False,
        caveats=['Missing animal identifiers prevent animal-level inference.',
        'ED_Fig3 group header ACSF+oligoA is ambiguous against the combined-inhibitor figure; do not silently relabel.',
        'Calibration trace ED_Fig2a and complete temporal traces are absent from these three XLS files.'])
