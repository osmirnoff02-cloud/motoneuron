"""[codex] 2026-11-04. Read-only CK/AK and calibration-readiness audit.

Lawson & Veech (1979), PMID36398, abstract/definitions verified via Entrez
and PubMed; full PDF unavailable (HTTP403). Constants refer to summed
chemical species under the stated assay conditions, not unbound ADP.
No ODE constants, temperature scaling or starting pools are overwritten.
"""
from copy import deepcopy
import math

_REFERENCE = dict(
    title='Effects of pH and free Mg2+ on the Keq of the creatine kinase reaction and other phosphate hydrolyses and phosphate transfer reactions',
    url='https://pubmed.ncbi.nlm.nih.gov/36398/',
    scope='Biochemical equilibrium assay, not a human motor-neuron measurement',
    read='Abstract and definitions; full PDF/References unavailable (HTTP403)',
    temperature_C=38., pH=7., ionic_strength_M=.25,
    chemical_species='sums of nucleotide/creatine forms at specified free Mg2+',
    rows=[dict(free_Mg_mM=0., CK_Kobs=37.8, AK_Kobs=.391),
          dict(free_Mg_mM=1., CK_Kobs=166., AK_Kobs=1.05)],
)


def ck_equilibrium_reference():
    return deepcopy(_REFERENCE)


def integrated_baseline_requirements():
    """Readiness of the current model, not a test of arbitrary new parameters."""
    return dict(validated_healthy_human_MN_baseline=False,
                default_interpretation='Exploratory parameter scenario; not a healthy reference cell',
                unresolved=[
                    'Joint TDP synthesis, cleavage, clearance and assay-specific fragments',
                    'Human-MN RP-mRNA kinetics: recorded conditional transfer screen passed 0/80 genes',
                    'Common steady state of all 4413 states and prolonged recovery',
                    'CK/AK equilibrium, nucleotide species, pH, free Mg and temperature',
                    'Matched electrical and biochemical temperatures; mechanism-specific kinetics',
                    'Complete Burley/Selvaraj panel on the current common baseline',
                ])


def conditional_ck_adp(*, atp_uM, cr_uM, pcr_uM, free_Mg_mM,
                       temperature_C, pH, ionic_strength_M):
    """Conditional SUM-species ADP, only at the two measured conditions.

    Not a measurement of cytosolic or Mg-unbound ADP. No extrapolation in
    temperature/pH/Mg/ionic strength is justified by the verified abstract.
    """
    vals=(atp_uM, cr_uM, pcr_uM, free_Mg_mM, temperature_C, pH, ionic_strength_M)
    if not all(math.isfinite(float(x)) for x in vals) or min(atp_uM,cr_uM,pcr_uM)<=0:
        raise ValueError('Finite conditions and positive pools required')
    for name,value in [('temperature_C',temperature_C),('pH',pH),('ionic_strength_M',ionic_strength_M)]:
        if not math.isclose(value,_REFERENCE[name],rel_tol=0,abs_tol=1e-12):
            raise ValueError('No validated extrapolation outside Lawson/Veech assay conditions: '+name)
    row=next((r for r in _REFERENCE['rows'] if math.isclose(free_Mg_mM,r['free_Mg_mM'],rel_tol=0,abs_tol=1e-12)),None)
    if row is None:
        raise ValueError('Only measured free Mg2+ conditions 0 and 1 mM are supported')
    return dict(ADP_sum_forms_uM=atp_uM*cr_uM/(row['CK_Kobs']*pcr_uM),
                CK_Kobs=row['CK_Kobs'], measured_human_ADP=False,
                assumption='Input pools use the same summed-species convention and compartment',
                source=ck_equilibrium_reference())


def energy_baseline_audit(result):
    """Attachable audit of actual first state, including checkpoint starts.

    Uses named state slots, not the original ATP0 when a run resumes.
    Returns conditional source comparisons, never an automatic calibration.
    """
    config=result['config']; cols=list(result['columns']); z=result['state'][0]
    cp=config['creatine']; en=config['energy']; factor=cp['reverse_equilibrium_factor']
    if not math.isfinite(factor) or factor<=0:
        raise ValueError('Positive finite CK reverse/forward ratio required')
    rows=[]
    for j,comp in enumerate(('soma','process','terminal')):
        prefix='' if comp=='soma' else comp+'_'
        a,d,m=(float(z[cols.index(prefix+n+'_uM')]) for n in ('ATP','ADP','AMP'))
        pc=float(z[cols.index(comp+'_PCr_uM')]); total=cp['total_creatine_uM'][j]; cr=total-pc
        q=a*cr/(d*pc) if d>0 and pc>0 else None
        hypothetical=(conditional_ck_adp(atp_uM=a,cr_uM=cr,pcr_uM=pc,free_Mg_mM=1.,
                        temperature_C=38.,pH=7.,ionic_strength_M=.25)['ADP_sum_forms_uM']
                      if min(a,cr,pc)>0 else None)
        rows.append(dict(compartment=comp,ATP_uM=a,ADP_uM=d,AMP_uM=m,PCr_uM=pc,Cr_uM=cr,
                         CK_reaction_quotient=q,AK_reaction_quotient=a*m/d**2 if d>0 else None,
                         CK_net_flux_uM_s=result['creatine_support']*cp['rate_scale'][j]*cp['forward_per_uM_s']*(pc*d-factor*cr*a),
                         conditional_ADP_at_Lawson_conditions_uM=hypothetical))
    return dict(
        scope='Read-only starting-state audit; conditional source comparison, not healthy baseline validation',
        start_time_s=float(result['time_s'][0]), compartments=rows,
        active_CK=dict(**cp,Kobs=1./factor,convention='reverse_equilibrium_factor = k_reverse/k_forward = 1/Kobs'),
        active_AK_Kobs=en['ATP0_uM']*en['AMP0_uM']/en['ADP0_uM']**2,
        active_ox_half_ADP_uM=en['ox_half_ADP_uM'],
        measured_to_model_CK_ratio=166.*factor,
        temperature=dict(ion_Nernst_C=config['ions']['temperature_C'],
                         native_NEURON_reference_C=23.,
                         unified_biochemical_temperature_C=None,
                         shared_Q10_correction=False,
                         note='Culture/incubation temperatures are source metadata, not an implemented second global thermal bath'),
        matched_reference_conditions_verified=False,
        missing_conditions=['Model pH','Free Mg2+','Ionic strength','Nucleotide species/compartment mapping','Rate-specific reference temperatures'],
        CK_reference=ck_equilibrium_reference(),
        calibration=integrated_baseline_requirements(),
    )
