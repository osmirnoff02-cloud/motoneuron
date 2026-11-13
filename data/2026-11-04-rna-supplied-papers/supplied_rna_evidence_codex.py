"""[codex] 2026-11-04. Optional source-cell scenarios and assay-aware RNA readout.

Does not replace the neuron default, add ODE states, or charge energy twice.
Mouse whole-cell rates are transferred only on explicit profile selection.
"""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
from scipy.stats import binom

B = Path(__file__).resolve().parent

def reference_panel():
    """A fresh source record, including censoring, quality and compartment flags."""
    return json.loads((B/'loedige_rp80_codex.json').read_text(encoding='utf8'))

def supplied_paper_evidence():
    return json.loads((B/'paper_evidence_codex.json').read_text(encoding='utf8'))

def loedige_parameters(base, profile='PCN_GFP_strict'):
    """Replace selected decay rates; retain declared synthesis and initial pools.

    Profiles: PCN_GFP_strict (29 independently strict control estimates),
    PCN_GFP_filtered (all finite author-filtered control estimates),
    PCN_GFP_paired and PCN_dnCAF1_paired (same paired gene set).
    No '>16' values are treated as point estimates. Missing genes retain base.
    Production is held fixed as a scenario, not inferred from a new equilibrium.
    """
    d=reference_panel();genes=d['genes']
    vectors=base.vector()
    rows={c:{r['model_gene']:r for r in d['whole_cell_rows'] if r['condition']==c and r['decay_s_inv'] is not None}
          for c in ('GFP','dnCaf1')}
    if profile=='PCN_GFP_strict':
        audit=json.loads((B/'loedige_audit_codex.json').read_text(encoding='utf8'))
        if audit['sha256']!=d['source_sha256']:
            raise ValueError('Strict audit and source table hashes disagree')
        selected=set(audit['whole_summary']['GFP']['stringent_genes'])
        if not selected.issubset(rows['GFP']):
            raise ValueError('Strict audit must be a subset of finite control estimates')
        condition='GFP'
    elif profile=='PCN_GFP_filtered':
        selected=set(rows['GFP']);condition='GFP'
    elif profile in ('PCN_GFP_paired','PCN_dnCAF1_paired'):
        selected=set(rows['GFP'])&set(rows['dnCaf1'])
        condition='GFP' if profile=='PCN_GFP_paired' else 'dnCaf1'
    else: raise ValueError('Unknown Loedige source-cell profile: '+str(profile))
    rates=vectors[1].copy()
    for i,g in enumerate(genes):
        if g in selected: rates[i]=rows[condition][g]['decay_s_inv']
    provenance=(f'Loedige2023 mouse primary cortical neuron {condition}; {len(selected)}/80 positive finite author hl_filtered rates; '
      f'{80-len(selected)} rates retained from {base.evidence_profile}. Uniform whole-cell loss across existing five pools remains an assumption. '
      f'Not human MN calibration; source stringent flag unresolved; >16 censored values excluded. Strict criterion independently reapplied: {profile == "PCN_GFP_strict"}. '
      'Production and initial RNA remain exactly the supplied base values; fixed synthesis under dnCAF1 is a scenario, not measured. '
      f'Source SHA256 {d["source_sha256"]}. DOI10.1016/j.molcel.2023.06.021. Base provenance: {base.provenance}')
    result=replace(base,decay_s_inv=tuple(rates),evidence_profile='Loedige2023_'+profile+'__base_'+base.evidence_profile,provenance=provenance)
    result.vector()
    return result

def _probability(name,value):
    v=np.asarray(value,dtype=float)
    if not np.isfinite(v).all() or np.any((v<0)|(v>1)):
        raise ValueError(name+' must be finite in [0,1]')
    return v

def labeling_measurement(new_fraction, *, p_old, p_new, recovery_new_to_old):
    """Assay expectation; all chemistry/recovery values must be supplied.

    Schofield2018 motivates separate old/new mutation probabilities. The
    normalized relative recovery formula is our observation model, not a
    universal dropout correction fitted by that paper. No default 3%/5% is used.
    """
    f=_probability('new_fraction',new_fraction)
    old=_probability('p_old',p_old);new=_probability('p_new',p_new)
    if np.any(new<=old):raise ValueError('p_new must exceed p_old for identifiable old/new components')
    q=np.asarray(recovery_new_to_old,dtype=float)
    if not np.isfinite(q).all() or np.any(q<=0):raise ValueError('Relative new/old recovery must be positive finite')
    theta=q*f/(1-f+q*f)
    return dict(true_new_fraction=f,observed_new_fraction=theta,
                expected_TC_fraction=(1-theta)*old+theta*new,
                scope='Assay measurement, not synthesis or degradation rate; chemistry probabilities require experiment-specific calibration')

def labeling_mixture_pmf(mutations, n_uridines, new_fraction, *, p_old,p_new,recovery_new_to_old):
    """Old/new binomial mixture; not a single binomial at the mean probability."""
    n=np.asarray(n_uridines,dtype=float);x=np.asarray(mutations,dtype=float)
    if not np.isfinite(n).all() or not np.isfinite(x).all() or np.any(n<0) or np.any(n!=np.floor(n)) or np.any(x<0) or np.any(x!=np.floor(x)) or np.any(x>n):
        raise ValueError('Mutation and uridine counts must be finite integers, 0 <= mutations <= n_uridines')
    obs=labeling_measurement(new_fraction,p_old=p_old,p_new=p_new,recovery_new_to_old=recovery_new_to_old)
    theta=obs['observed_new_fraction']
    return (1-theta)*binom.pmf(x,n,p_old)+theta*binom.pmf(x,n,p_new)

def compare_compartments(parameters):
    """Read-only comparison. No nuclear-export rate or axonal pool is invented."""
    d=reference_panel();k=parameters.vector()[1]
    rows=[]
    for r in d['compartment_rows']:
        v=dict(r);i=d['genes'].index(r['model_gene'])
        v['model_effective_whole_cell_decay_s_inv']=float(k[i])
        v['automatic_rate_assignment']=False
        v['comparison_scope']='Different species/assay/compartment; no fit or equality constraint'
        rows.append(v)
    return dict(rows=rows,model_profile=parameters.evidence_profile,human_motor_neuron_calibrated=False)
