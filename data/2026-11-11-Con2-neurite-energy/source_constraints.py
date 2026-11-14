"""[codex] 2026-11-11. Assay/chemistry constraints, not a human MN fit.

Source PDFs and verified pages are in sources_manifest.json. No silent
replacement of integrated-neuron rates; ATP/ADP totals and bare ionic species
are distinguished. Table2.2 constants are valid only at25C/I0.1M.
"""
import math

def source_constraints():
    return {
      'Vinnakota2009':dict(title='Multiple Ion Binding Equilibria, Reaction Kinetics, and Thermodynamics in Dynamic Models of Biochemical Pathways',
        doi='10.1016/S0076-6879(08)03802-0',pdf='papers/nihms133202.pdf',pages='17-18,35',
        CK_points=[dict(Kobs=166.2,temperature_C=37,pH_concentration=7,free_Mg_mM=1,free_K_mM=140,ionic_strength_M=.18),
                   dict(Kobs=99.3,temperature_C=37,pH_concentration=7,free_Mg_mM=1,free_K_mM=250,ionic_strength_M=.25)],
        kind='Published chemistry calculations, not measured neuron equilibria'),
      'Teague1992':dict(title='Effect of Temperature on the Creatine Kinase Equilibrium',
        pdf='papers/PIIS0021925819496828.pdf',pages='1-7; embedded supplement8-10',
        temperatures_C=[5,15,25,38],Kobs_mean=[307,255,217,177],Kobs_SE=[13,10,11,7],n_per_condition=8,
        pH=7,free_Mg_mM=1,ionic_strength_M=.25,delta_H_J_mol=-11930,
        kind='Solution apparent constants; historical ion-binding convention'),
      'Lei2003':dict(title='In Vivo 31P Magnetic Resonance Spectroscopy of Human Brain at 7 T: An Initial Experience',
        doi='10.1002/mrm.10379',pdf='papers/lei2003.pdf',pages='202-203',system='human visual cortex, mixed tissue',
        PCr_gammaATP_peak_area_ratio_mean=1.29,PCr_gammaATP_peak_area_ratio_reported_error=.22,ratio_n=12,
        pH_mean=6.97,pH_reported_error=.04,pH_n=12,kf_s_inv_mean=.24,kf_s_inv_reported_error=.03,kf_n=11,
        assumed_PCr_umol_g=3.43,derived_forward_exchange_umol_g_s=.24*3.43,
        kind='Peak-area ratio and gross forward CK exchange, not net ATP production; reported +/- retained, error type not explicit in these Results sentences'),
      'limits':['No joint human motor-neuron ATP/ADP/PCr/Cr/Mg/pH calibration',
        'FCCP maximal OCR is uncoupled respiration; not ATP-synthesis capacity',
        'ECAR needs buffer capacity, volume and respiratory acidification correction',
        'Tissue MRS cannot assign neurite concentrations or fluxes']}

def teague1992_temperature(temperature_C,*,pH=7.,free_Mg_mM=1.,ionic_strength_M=.25):
    if not math.isfinite(temperature_C) or not 5<=temperature_C<=38:
        raise ValueError('Only measured temperature span5..38C supported')
    if (pH,free_Mg_mM,ionic_strength_M)!=(7.,1.,.25):
        raise ValueError('Teague1992 temperature series only pH7/freeMg1mM/I0.25M')
    # Van't Hoff interpolation anchored to the observed38C mean, not new data.
    k=177*math.exp(11930/8.314462618*(1/(temperature_C+273.15)-1/311.15))
    return dict(Kobs=k,temperature_C=temperature_C,pH=pH,free_Mg_mM=free_Mg_mM,
        ionic_strength_M=ionic_strength_M,status='interpolation of solution experiment',human_MN_calibrated=False)

def vinnakota_binding_species(*,pH_concentration=7.,free_Mg_mM=1.,free_K_mM=140.,
        temperature_C=25.,ionic_strength_M=.1):
    if temperature_C!=25 or ionic_strength_M!=.1:
        raise ValueError('Table2.2 only25C/I0.1M; unknown binding enthalpies prevent silent extrapolation')
    vals=[pH_concentration,free_Mg_mM,free_K_mM]
    if not all(math.isfinite(x) for x in vals) or not 0<=pH_concentration<=14 or min(vals[1:])<0:
        raise ValueError('Finite physical free-ion concentrations required')
    H=10**(-pH_concentration);M=free_Mg_mM/1000;K=free_K_mM/1000
    # Dissociation pK from Table2.2, checked against rendered PDFp35.
    atp=dict(ATP4=1.,HATP=H*10**6.48,MgATP=M*10**4.19,
        MgHATP=H*M*10**(6.48+2.32),Mg2ATP=M*M*10**(4.19+1.7),KATP=K*10**1.17)
    adp=dict(ADP3=1.,HADP=H*10**6.38,MgADP=M*10**3.25,
        Mg2ADP=M*M*10**(3.25+1.),KADP=K*10**1.)
    pcr=dict(HPCr2=1.,H2PCr=H*10**4.5,MgHPCr=M*10**1.6,KHPCr=K*10**.31)
    cr=dict(HCr=1.,H2Cr=H*10**2.3)
    out={n:dict(binding_polynomial=sum(w.values()),fractions={s:v/sum(w.values()) for s,v in w.items()})
        for n,w in [('ATP',atp),('ADP',adp),('PCr',pcr),('Cr',cr)]}
    return dict(species=out,temperature_C=25.,ionic_strength_M=.1,pH_concentration=pH_concentration,
        free_Mg_mM=free_Mg_mM,free_K_mM=free_K_mM,
        status='Read-only species observer; not an integrated dynamic Mg/K/pH model',human_MN_calibrated=False)

def lei_forward_exchange(kf_s_inv=.24,pcr_umol_g=3.43):
    if not all(math.isfinite(x) and x>=0 for x in (kf_s_inv,pcr_umol_g)):
        raise ValueError('Nonnegative finite values required')
    return dict(forward_exchange_umol_g_s=kf_s_inv*pcr_umol_g,
        net_ATP_synthesis=None,net_ATP_synthesis_reason='Reverse exchange and pathway-specific net flux are not determined',
        PCr_status='Default3.43 is assumed from Lei ref10, not measured in that experiment')

def seahorse_ATP_assay(*,ATP_linked_OCR_pmol_min_ug,protein_pg_per_cell,
        accessible_volume_pL_per_cell,ATP_per_O2):
    """Explicit conditional unit conversion; caller must supply all normalizers.

    Input is oligomycin-sensitive OCR, NEVER FCCP maximal OCR. ATP/O2 is a
    declared effective coupling assumption, not a fixed empirical neuron value.
    """
    vals=(ATP_linked_OCR_pmol_min_ug,protein_pg_per_cell,accessible_volume_pL_per_cell,ATP_per_O2)
    if not all(math.isfinite(x) and x>0 for x in vals):raise ValueError('All explicit assay inputs must be positive')
    # 1ug=1e6pg; 1pmol/pL=1mol/L, then1e6uM/M; /60 for seconds.
    return dict(conditional_ATP_uM_s=ATP_linked_OCR_pmol_min_ug*protein_pg_per_cell*ATP_per_O2/(60*accessible_volume_pL_per_cell),
        source_signal='oligomycin-sensitive OCR',human_MN_calibrated=False,
        caution='Whole-cell average cannot assign supply to soma versus neurite')
