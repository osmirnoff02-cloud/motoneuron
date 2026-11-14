"""[codex] Source-specific turnover and conditional loss-budget diagnostics.

Neither function edits the model or claims a human basal calibration.
"""
from pathlib import Path
import json,math
B=Path(__file__).resolve().parent

def hasan_cortical_turnover(gene, *, cohort):
    sheets={'WT_global':'WT iNeuron protein half-lives',
            'WT_KO_comparison':'Protein half-lives KOvsWT'}
    if cohort not in sheets:raise ValueError('Explicit reviewed cohort required')
    d=json.loads((B/'hasan_selected_turnover.json').read_text(encoding='utf8'))
    hits=[r for r in d['rows'] if r['gene']==gene and r['sheet']==sheets[cohort]]
    if len(hits)!=1:raise ValueError('No unique measured value for this gene/cohort; do not impute')
    return dict(**hits[0],doi=d['doi'],source=d['source'],system=d['system'],
        human_MN_calibrated=False,enzyme_off_rate=False,
        limitation='Global protein-label estimate; WT panels are distinct selections, not interval endpoints or independent experiments. No conversion to active calpain lifetime or fragment clearance.')

def tdp_first_order_loss_budget(*,half_life_h,max_cleavage_fraction,fixed_other_hazard_s_inv):
    """Bounds for commensurate intact-protein first-order hazards.

half_life_h is a requested LOWER bound, not a fitted exact half-life. The
separate exact_target_possible flag tests equality with the supplied share cap.

Caller must supply an ASSUMED cleavage share; no study here measures a50% cap.
Not generally a dSILAC/Dendra2/FRET forward model: fragments and flux recovery
may retain label and invalidate this single-pool approximation.
"""
    x=[half_life_h,max_cleavage_fraction,fixed_other_hazard_s_inv]
    if not all(math.isfinite(v) for v in x) or half_life_h<=0 or not 0<=max_cleavage_fraction<=1 or fixed_other_hazard_s_inv<0:
        raise ValueError('Finite positive half-life, fraction0..1 and nonnegative other hazard required')
    target=math.log(2)/(half_life_h*3600)
    remaining=target-fixed_other_hazard_s_inv
    feasible=remaining>=0
    # A fraction of actual total loss is not just a fraction of the target cap:
    # cleavage/(cleavage+other)<=f implies cleavage<=f*other/(1-f).
    fixed_share_cap=(max_cleavage_fraction*fixed_other_hazard_s_inv/(1-max_cleavage_fraction)
                     if max_cleavage_fraction<1 else None)
    return dict(target_total_hazard_s_inv=target,
        half_life_constraint='total half-life at least the supplied hours',
        assumed_cleavage_share_cap=max_cleavage_fraction,
        share_only_cleavage_cap_s_inv=max_cleavage_fraction*target,
        share_cap_with_fixed_other_loss_s_inv=fixed_share_cap,
        remaining_hazard_s_inv=remaining,
        joint_cleavage_cap_s_inv=min(remaining,fixed_share_cap) if feasible and fixed_share_cap is not None else (remaining if feasible else None),
        feasible_with_fixed_other_loss=feasible,
        exact_target_possible_with_share_cap=feasible and remaining<=max_cleavage_fraction*target,
        maximum_half_life_without_cleavage_h=math.log(2)/(fixed_other_hazard_s_inv*3600) if fixed_other_hazard_s_inv else None,
        scope='Conditional first-order budget, not a fitted biochemical parameter or a treatment recommendation; intrinsic cleavage1 coefficient additionally depends on activity and substrate state.')
