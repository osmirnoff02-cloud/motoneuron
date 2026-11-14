"""[codex] 2026-11-04. Explicit joint candidate, NOT a healthy human calibration.

The old draw13 remains reproducible. This factory changes all three first-order
turnover terms together and solves CK/AK/pool constraints together. No reset,
hidden flux offset, new energy debit or fitted electrical conductance is added.
"""
from pathlib import Path
from dataclasses import replace,asdict
import math,json
import numpy as np
from scipy.optimize import brentq

B=Path(__file__).resolve().parent
RANGES=dict(temperature_C=[23.,37.],pH=[6.8,7.,7.2],free_Mg_mM=[.5,1.,2.],
    AK_Kobs=[.391,1.05,1.62],TDP_reference_half_h=[12.,48.,143.55],
    assumed_basal_cleavage_share=[.05,.2,.5],TDP_Q10=[1.,2.,3.],RNA_Q10=[1.,2.,3.],
    RNA_rate_scale=[.5,1.,2.],respiratory_reserve=[2.,4.,8.])

def ck_equilibrium(*,temperature_C,pH,free_Mg_mM,ionic_strength_M=.25):
    """Wiseman2023 Eq12/13 + author beta table, ExcelFileS1.

    Proton stoichiometry uses the author-supplied outputs. Piecewise linear
    interpolation in pH, pMg and temperature is an explicit approximation.
    This is solution thermodynamics, not a neuron concentration measurement.
    """
    t=float(temperature_C);pmg=-math.log10(free_Mg_mM/1000) if free_Mg_mM>0 else float('inf')
    if not all(math.isfinite(x) for x in [t,pH,pmg,ionic_strength_M]) or not (10<=t<=40 and 6<=pH<=8 and 2<=pmg<=4) or abs(ionic_strength_M-.25)>1e-12:
        raise ValueError('Source domain:10..40C,pH6..8,Mg0.1..10mM,I=0.25M')
    tab=json.loads((B/'Wiseman2023_beta.json').read_text(encoding='utf8'))['rows']
    # Interpolate the author's actual beta outputs; Table A2 alone does not
    # reproduce the supplied calculator. Never silently substitute our inference.
    ts=[10.,20.,30.,40.];ms=[2.,3.,4.];by_t=[]
    for ti in ts:
        by_m=[]
        for mi in ms:
            rows=sorted((r['pH'],r['beta']) for r in tab if r['temperature_C']==ti and r['pMg']==mi)
            if not rows[0][0]<=pH<=rows[-1][0]:
                raise ValueError('Author beta grid does not cover this pH at all conditions; no extrapolation')
            by_m.append(float(np.interp(pH,[r[0] for r in rows],[r[1] for r in rows])))
        by_t.append(float(np.interp(pmg,ms,by_m)))
    beta=float(np.interp(t,ts,by_t))
    double=(-.667*pmg-.0307*t+4.287)*1e9 # ExcelFileS1 named coefficients
    H=10**(-pH);obs=double*H**beta
    if obs<=0 or not 0<beta<1.1:raise ValueError('Nonphysical CK estimate')
    return dict(Kobs=obs,Kdouble_M_inv=double,beta=beta,temperature_C=t,pH=pH,free_Mg_mM=free_Mg_mM,
        ionic_strength_M=ionic_strength_M,source='10.3390/ijms241713244 Eq12/13 ExcelFileS1 Figure5&A1 and named regression coefficients',
        solution_Na_mM=145.,solution_K_mM=6.5,beta_interpolation='pH then pMg then temperature, no extrapolation',
        interpretation='All chemical species sums in source solution; transfer to neuronal cytosol is conditional; not measured neuronal ADP')

def energy_pools(*,adenylate_total_uM,creatine_total_uM,PCr_uM,KCK,KAK):
    v=[adenylate_total_uM,creatine_total_uM,PCr_uM,KCK,KAK]
    if not all(math.isfinite(x) and x>0 for x in v) or PCr_uM>=creatine_total_uM:raise ValueError('Positive finite pools and constants required')
    ratio=KCK*PCr_uM/(creatine_total_uM-PCr_uM)
    d=adenylate_total_uM/(ratio+1+KAK/ratio)
    return dict(ATP0_uM=ratio*d,ADP0_uM=d,AMP0_uM=KAK*d/ratio)

def joint_inputs(api,profile='Burley',*,temperature_C=23.,pH=7.,free_Mg_mM=1.,AK_Kobs=1.05,
                 tdp_reference_half_h=48.,cleavage_share=.2,tdp_q10=1.,rna_q10=1.,rna_rate_scale=1.,
                 rna_source='retained_prior',respiratory_reserve=2.):
    """Return (ordinary run_integrated_neuron kwargs, explicit assumption record).

    Q10=1 is the temperature-neutral hypothesis, not a measured coefficient.
    Protein/RNA reference incubation37C; unknown fast rates are interpreted as
    target-temperature hypotheses. Electrical23C is retained for regression.
    """
    if not all(math.isfinite(x) and x>0 for x in [tdp_reference_half_h,tdp_q10,rna_q10,rna_rate_scale,AK_Kobs]) or not 0<cleavage_share<1:raise ValueError('Positive values and a cleavage share strictly0..1 required')
    o=dict(**api.energy_terminal_inputs(profile),adaptive_parameters=api.AdaptiveRecyclingParameters(enabled=True,calcium_capacity_enabled=True),
        recycling_parameters=api.RecyclingParameters(enabled=True),pathway_parameters=api.TerminalPathwayParameters(enabled=True),activity_parameters=api.ActivityEnergyParameters(gain=60.))
    if temperature_C!=23.:raise ValueError('Frozen electrical implementation uses23C; use ck_equilibrium for37C chemistry until channel Q10 is known. This is not a claim about recording temperature in all papers.')
    oldenergy=o['energy_parameters'];oldtdp=o['tdp_parameters'];ck=api.CreatineParameters()
    pc=float(api._ck28.equilibrium_PCr(oldenergy.ATP0_uM,oldenergy.ADP0_uM,ck.total_creatine_uM[0],ck.reverse_equilibrium_factor))
    thermo=ck_equilibrium(temperature_C=temperature_C,pH=pH,free_Mg_mM=free_Mg_mM)
    pools=energy_pools(adenylate_total_uM=oldenergy.ATP0_uM+oldenergy.ADP0_uM+oldenergy.AMP0_uM,
        creatine_total_uM=ck.total_creatine_uM[0],PCr_uM=pc,KCK=thermo['Kobs'],KAK=AK_Kobs)
    o['energy_parameters']=replace(oldenergy,**pools,gly_half_ADP_uM=pools['ADP0_uM'],
        provenance=oldenergy.provenance+'; Joint candidate: fixed total adenylate/Cr/PCr, CK source-solution equation, explicitly assumed AK and ADP response.')
    o['creatine_parameters']=replace(ck,reverse_equilibrium_factor=1/thermo['Kobs'],provenance='Wiseman2023 source-solution thermodynamics; donor forward CK capacity retained, conditional human transfer')
    o['ion_parameters']=replace(o.get('ion_parameters') or api.IonParameters(),temperature_C=temperature_C)
    o['respiratory_reserve']=respiratory_reserve
    # Solve the weighted intact first-order loss budget, not a label half-life.
    agg=api._module(B.parent/'2026-10-23-tdp-aggregation-chain/model_api.py','_joint_baseline_agg04')
    keys=('basal_ca_uM','calcium_tau_s','enzyme_parameters','tdp_parameters','localization_parameters','stmn2_parameters','inactivation_parameters','phosphorylation_parameters','reactivation_parameters','stability_parameters')
    h=tdp_reference_half_h*tdp_q10**((37.-temperature_C)/10)
    target=math.log(2)/(h*3600);intrinsic=target*(1-cleavage_share)
    def state(coef):
        tp=replace(oldtdp,intact_half_life_s=math.log(2)/intrinsic,cleavage1_per_s=coef)
        kw={k:o[k] for k in keys};kw['tdp_parameters']=tp
        y,p,kv,sv,act=agg._baseline(kw,o['aggregation_parameters'].vector())
        hazard=coef*act*(y[6]+p[19]*y[7])/(y[6]+y[7])
        return hazard,tp,y,p,act
    hi=oldtdp.cleavage1_per_s
    while state(hi)[0]<target*cleavage_share:hi*=2
    coefficient=brentq(lambda x:state(x)[0]-target*cleavage_share,0,hi,xtol=1e-15)
    hazard,tp,y,p,act=state(coefficient)
    o['tdp_parameters']=replace(tp,provenance='Joint candidate first-order intact budget; half-life target and cleavage share are assumptions, not an assay fit; synthesis reconstructed with aggregates')
    # Keep process TDP a distinct lineage but update its rate assignment too.
    if 'injury_parameters' in o:
        loc=dict(o['injury_parameters']);loc['substrates']=[replace(s,cleavage_per_activity_s=coefficient,intact_half_life_s=tp.intact_half_life_s) if s.name=='cytoplasmic_TDP43' else s for s in loc['substrates']];o['injury_parameters']=loc
    rp=api.rp_mrna_evidence_parameters();rates=np.array(rp.decay_s_inv);selected=[]
    if rna_source=='Ye_positive_interval_transfer':
        data=json.loads((B.parent/'2026-11-04-human-MN-rp-mrna/ye_chua_RP80_fits.json').read_text(encoding='utf8'))
        rows={row['gene']:row for row in data['genes']}
        for i,g in enumerate(api.RP_MRNA_GENES):
            row=rows[g];f=row['nonlinear_all9']
            if f['local_95_half_life']['positive_decay_supported']:
                rates[i]=f['k_s_inv_signed'];selected.append(g)
    elif rna_source!='retained_prior':raise ValueError('Unknown RNA source scenario')
    rates*=rna_rate_scale*rna_q10**((temperature_C-37.)/10)
    # Preserve original declared total abundance; scale production with decay.
    amounts=np.array(rp.initial_total_uM)
    o['rp_mrna_kinetics_parameters']=replace(rp,decay_s_inv=tuple(rates),production_uM_s=tuple(rates*amounts),
        evidence_profile='joint_'+rna_source,provenance=rp.provenance+'; Joint scenario: production=k*declared total; no NTR-as-decay conversion. Ye is ActD i3Neurons, not basal human MN; Q10/transfer are assumptions.')
    record=dict(author='codex',date='2026-11-04',profile=profile,healthy_human_calibrated=False,
        status='conditional joint candidate; full-network stationary state not imposed',thermodynamics=thermo,
        energy=dict(**pools,PCr0_uM=pc,Cr0_uM=ck.total_creatine_uM[0]-pc,AK_Kobs=AK_Kobs,
            AK_source='1.05 measured38C Lawson; at other temperatures an explicit transfer hypothesis',respiratory_reserve=respiratory_reserve,
            adenylate_total_preserved=True,Cr_total_preserved=True,initial_energy_charge_not_conserved_between_distinct_baseline_candidates=True),
        TDP=dict(reference_half_h=tdp_reference_half_h,target_temperature_half_h=h,Q10=tdp_q10,
            assumed_cleavage_share=cleavage_share,cleavage1_per_s=coefficient,intrinsic_half_h=tp.intact_half_life_s/3600,
            effective_cleavage_hazard_s_inv=hazard,intrinsic_hazard_s_inv=intrinsic,
            reconstructed_synthesis_per_s=float(p[20]),calpain_activity=float(act),
            interpretation='Two first-order intact loss paths only; NOT total dSILAC/Dendra2 half-life or total turnover including aggregate exchange'),
        RNA=dict(source=rna_source,source_changed_genes=selected,source_changed_count=len(selected),Q10=rna_q10,rate_scale=rna_rate_scale,
            synthesis_rule='k*original declared total abundance; conditional stationarity budget, not measured RNA synthesis',
            source_reference_temperature_C=37.,median_half_h=float(np.median(np.log(2)/rates/3600))),
        temperatures=dict(electrical_reference_C=23.,target_C=temperature_C,biochemical_reference_C=37.,
            electrical_reference_scope='Implementation reference only; experimental bath temperature must be checked per publication, not inferred from this number',
            TDP_Q10=tdp_q10,RNA_Q10=rna_q10,unknown_fast_rates='Effective hypotheses assigned to target temperature; their Q10 remains unmeasured',
            electrical_kinetic_temperature_correction=False,global_temperature_calibration=False),
        changed_conductances=False,source_scenario_13_residual_parameters_retained=True)
    return o,record
