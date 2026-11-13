"""[codex] Source-specific benchmarks; none is a human MN parameter fit."""
from pathlib import Path
import json
import numpy as np

def source_constraints():
    result=dict(
      Gerkau2019=dict(doi='10.1113/JP278658',system='mouse hippocampal CA1',temperature_C=[20.,22.],
        protocol='10 afferent stimuli at50Hz',paired_same_cell=False,
        sodium_recovery_s={'soma':[60.,90.],'dendrite':[20.,30.]},ATP_recovery_s=[300.,360.],
        ATP_drop_uM={'soma':90.,'dendrite':130.},
        scope='Separate experimental cohorts/assays. Recovery windows are reported descriptive ranges, not confidence intervals or pump time constants.'),
      Lombardo2018=dict(doi='10.1371/journal.pone.0193948',system='P3–8 mouse spinal MN',temperature_C=34.,
        protocol='10mM K exposure >=30min; separate patch solutions and XE991 deactivation assay',
        scenarios_not_measurements={'gKCNQ_multiplier':6.,'Vhalf_shift_mV':-7.25},
        raw_data='Lombardo2018_S1_Table.xlsx',quarantined='Figure 2 G!D4 negative input resistance -63.31; published group mean also includes this value; do not fit without clarification',
        scope='Figure6 simulation curves are not independent experimental validation. Tail/deactivation current is not a steady conductance.'),
      Buskila2019=dict(doi='10.1038/s41419-019-1538-9',system='WT/SOD1G93A mouse cortical upper MN',temperature_C=22.,
        selection='Sag>10%; age-dependent Ih/M interaction',
        scope='RNA and spike counts alone cannot identify absolute M-current. Pipette ATP/PCr and selected sag population prevent direct intact human energy calibration.'),
      Yuan2025=dict(doi='10.1016/j.cell.2025.07.048',system='mouse DRG plus TRPV1-engineered K562/HEK293T',
        ATP_assay='PercevalHR in HEK293T; relative ATP/ADP optical response before lysis',
        scope='Not absolute neuronal ATP or AMPA input. ETC/ROS/NAD/trafficking perturbations must remain separate from human MN pump calibration.'))
    measured=json.loads((Path(__file__).resolve().parent/'Lombardo2018_extracted_assays_codex.json').read_text(encoding='utf8'))
    result['Lombardo2018']['measured_deactivation_pA']=[
        {k:r[k] for k in ('group','Vcommand_mV','n','mean_pA','sem_pA','source_cell_range')}
        for r in measured['rows']]
    result['Lombardo2018']['measurement_scope']='18 group-voltage summaries from individual cells; mouse assay reference, not human fitted conductances.'
    return result

def gerkau_recovery_comparison(na_recovery_s,atp_recovery_s,compartment='soma'):
    """Descriptive source ordering, not transfer of mouse rates into the model."""
    if compartment not in ('soma','dendrite'):raise ValueError('Unknown source compartment')
    if not np.isfinite([na_recovery_s,atp_recovery_s]).all() or min(na_recovery_s,atp_recovery_s)<0:
        raise ValueError('Finite nonnegative times required')
    source=source_constraints()['Gerkau2019']
    return dict(ATP_later_than_Na=bool(atp_recovery_s>na_recovery_s),
      observed_source_Na_window_s=source['sodium_recovery_s'][compartment],
      observed_source_ATP_window_s=source['ATP_recovery_s'],human_fit=False,
      scope='Only meaningful for explicitly matched recovery criterion and stimulation; distinct source cohorts.')
