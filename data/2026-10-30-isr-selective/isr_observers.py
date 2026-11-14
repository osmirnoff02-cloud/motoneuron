"""[codex]2026-10-30. Source-specific observations, never kinetic calibration.
Fusco2025 rat neurites: immature particles != translation-ready ribosomes.
Ziff2023 human iPSC-MNs: image concentration ratios != molecule fractions/TPM.
"""
import numpy as np

def nuclear_concentration_ratio(amount_fraction,nuclear_to_cytoplasmic_volume_ratio):
    f=np.asarray(amount_fraction,float);v=float(nuclear_to_cytoplasmic_volume_ratio)
    if not np.isfinite(f).all() or np.any(f<0) or np.any(f>=1) or not np.isfinite(v) or v<=0:raise ValueError('Require fraction[0,1) and independently specified positive volume ratio')
    return f/((1-f)*v)

def fraction_from_concentration_ratio(ratio,nuclear_to_cytoplasmic_volume_ratio):
    r=np.asarray(ratio,float);v=float(nuclear_to_cytoplasmic_volume_ratio)
    if not np.isfinite(r).all() or np.any(r<0) or not np.isfinite(v) or v<=0:raise ValueError('Require nonnegative measured ratio and positive volume ratio')
    return r*v/(1+r*v)

def ribosome_maturity_observer(state,selected_ribosome_uM,compartment='soma'):
    z=np.atleast_2d(np.asarray(state,float));C=float(selected_ribosome_uM)
    if z.shape[1]<403 or not np.isfinite(z).all() or not np.isfinite(C) or C<=0:raise ValueError('Explicit biogenesis state and positive selected concentration required')
    if compartment!='soma':raise ValueError('Current selected ribosome pools are somatic; Fusco localization does not justify relabeling them as axonal')
    return dict(premature40_uM=C*z[:,390],premature60_uM=C*z[:,391],
      free_mature40_uM=C*z[:,330],free_mature60_uM=C*z[:,331],
      limitation='Free mature and premature pools separately observed; this is not total RP fluorescence or all translating ribosomes. Rat neurite localization provides structural evidence, not human maturation rates.')

def validate_fraction_calibration_input(sample_ids,value_kind='TPM'):
    ids=set(sample_ids);reasons=[]
    if 'ctrl3_untreated_nucleus' in ids:reasons.append('CTRL3 untreated nucleus failed fractionation QC in paper and author code; exclude explicitly.')
    if value_kind.upper()=='TPM':reasons.append('TPM cannot calibrate absolute import/export or be used as raw DESeq2 counts.')
    if 'ncrme6_ml240_cytoplasm' in ids and 'ncrme6_ml240_nucleus' not in ids:reasons.append('Missing paired ncrme6 ML240 nuclear sample; do not impute.')
    return dict(eligible_for_absolute_kinetic_fit=False,reasons=reasons,
      required='Raw counts, donor/clone and fraction QC mapping for DE; fraction yields/spike-ins and time course for absolute rates.')
