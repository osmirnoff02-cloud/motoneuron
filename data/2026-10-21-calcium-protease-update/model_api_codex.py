"""[codex] 2026-10-21. Competing TDP cleavage and site-specific spectrin gating.

De Marco 2014 supports a competing caspase route, NOT a human Ca->caspase fit.
Nicolas 2002 supports reduced Y1176 susceptibility, NOT human rate constants.
All electrical APIs and previous scenarios remain available and unchanged.
"""
from pathlib import Path
from dataclasses import dataclass, asdict, replace
import importlib.util
import sys
import json
import math
import numpy as np
from numba import njit

_HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_previous = _load('_before_competing20261021', _HERE.parent/'2026-10-21-spectrin-tdp-state-update/model_api_codex.py')
_core = _load('_calpain_core_competing20261021', _HERE.parent/'2026-10-19-calpain-tdp-scenarios/calpain_tdp_codex.py')
for _name in dir(_previous):
    if not _name.startswith('_') and _name not in globals():
        globals()[_name] = getattr(_previous, _name)


def _array(value, t, label, upper=None):
    x = np.full(t.shape, float(value)) if np.ndim(value) == 0 else np.asarray(value, float)
    if x.shape != t.shape or not np.isfinite(x).all() or np.any(x < 0) or (upper is not None and np.any(x > upper)):
        raise ValueError(f'{label}: finite nonnegative scalar or matching time required')
    return x


@njit
def _rhs_competing(y, ca, block, hazard, synthesis, p):
    # Index 9 is a cumulative exit into unresolved caspase products. It is NOT
    # the abundance of the 35/25/15-kDa C-terminal/internal bands.
    old = _core._rhs(y[:9], ca, block, synthesis, p)
    q = hazard*y[1]
    result = np.empty(10)
    result[:9] = old
    result[1] -= q
    result[9] = q
    return result


@njit
def _integrate_competing(t, ca, blocks, hazards, y0, synthesis, p, max_step):
    out = np.empty((len(t), 10)); out[0] = y0
    rate = max(1/min(p[2], p[3]), p[5]+p[7]+np.max(hazards), p[6]+p[8], p[9])
    safe = min(max_step, .15/rate)
    for i in range(len(t)-1):
        local = min(safe, .1) if abs(ca[i+1]-ca[i]) > 1e-12 else safe
        n = max(1, int(math.ceil((t[i+1]-t[i])/local)))
        h = (t[i+1]-t[i])/n; y = out[i].copy()
        for j in range(n):
            c0 = ca[i]+(ca[i+1]-ca[i])*j/n
            cm = ca[i]+(ca[i+1]-ca[i])*(j+.5)/n
            c1 = ca[i]+(ca[i+1]-ca[i])*(j+1)/n
            k1 = _rhs_competing(y,c0,blocks[i],hazards[i],synthesis,p)
            k2 = _rhs_competing(y+h*k1/2,cm,blocks[i],hazards[i],synthesis,p)
            k3 = _rhs_competing(y+h*k2/2,cm,blocks[i],hazards[i],synthesis,p)
            k4 = _rhs_competing(y+h*k3,c1,blocks[i],hazards[i],synthesis,p)
            y += h*(k1+2*k2+2*k3+k4)/6
        out[i+1] = y
    return out


def coupled_trace_to_competing_tdp(trace, *, parameters, compartment, basal_ca_uM,
                                   caspase_hazard_per_s, caspase_provenance,
                                   inhibition=0., max_step_s=.01):
    """Same Ca->calpain dynamics; shared intact TDP can exit by either protease.

    Caspase hazard = supplied activity x susceptibility, in s^-1. It is not
    inferred from Ca, Ni/Co exposure, RNA, ordinal + signs, or antibody signal.
    Left-constant on each interval; switches must be included among time knots.
    All arms share the previous unblocked, caspase-free baseline and synthesis.
    This is an acute added-caspase scenario, not a chronic equilibrium.
    Caspase cleavage of existing calpain fragments and enzyme cross-activation
    are unresolved and excluded. The extra counter preserves N-lineage
    accounting but predicts no C-terminal fragment abundance or phospho signal.
    """
    if not isinstance(caspase_provenance,str) or not caspase_provenance.strip():
        raise ValueError('Explicit caspase source/assumption required')
    ref = _previous.coupled_trace_to_tdp(trace, parameters=parameters, compartment=compartment,
             basal_ca_uM=basal_ca_uM, inhibition=inhibition, max_step_s=max_step_s)
    t = ref['time_s']; hazard = _array(caspase_hazard_per_s,t,'Caspase hazard')
    if np.any(hazard):
        base = _previous.baseline_equilibrium(basal_ca_uM,parameters)
        y = _integrate_competing(t,ref['ca_uM'],ref['inhibition'],hazard,
            np.r_[base['initial_state'],0.],base['synthesis_per_s'],parameters.vector(),max_step_s)
        residual = y[:,1:5].sum(axis=1)+y[:,9]-y[:,8]
        error = float(np.max(abs(residual-residual[0])))
        if not np.isfinite(y).all() or y.min() < -1e-9 or y[:,0].max() > 1+1e-9 or error > 1e-8*max(1.,float(y[:,1:5].sum(axis=1).max())):
            raise ArithmeticError('Competing-route trajectory or lineage balance invalid')
        ref.update({name:y[:,j] for j,name in enumerate(_core.STATE_NAMES)})
        p = parameters.vector(); effective = y[:,0]*(1-ref['inhibition'])/(1+p[4])
        ref.update(effective_activity=effective,formation_rate_per_s=p[5]*effective*y[:,1],
            secondary_rate_per_s=p[6]*effective*y[:,2],large_removal_rate_per_s=p[8]*y[:,2],
            lineage_balance_max_error=error,caspase_diverted_lineages=y[:,9])
    else:
        # Exactly preserve every legacy numerical value in the zero-route limit.
        ref['caspase_diverted_lineages'] = np.zeros_like(t)
    ref.update(caspase_hazard_per_s=hazard,caspase_provenance=caspase_provenance,
        caspase_exit_rate_per_s=hazard*ref['intact_TDP'],caspase_product_abundances=None,
        human_kinetic_fit=False,calcium_to_caspase_calibrated=False,
        status='Conditional competing intact-TDP cleavage; unresolved caspase products',
        omitted_interactions=['caspase cleavage of calpain fragments','protease cross-activation',
                             'phosphorylation dynamics','aggregation/nuclear function/viability'])
    return ref


@dataclass(frozen=True)
class Y1176Gate:
    initial_phosphorylated_fraction: float
    phosphorylated_first_cut_relative_rate: float
    provenance: str

    def validate(self):
        v = [self.initial_phosphorylated_fraction,self.phosphorylated_first_cut_relative_rate]
        if not np.isfinite(v).all() or min(v)<0 or max(v)>1 or not isinstance(self.provenance,str) or not self.provenance.strip():
            raise ValueError('Explicit Y1176 fraction/rate in [0,1] and provenance required')


def simulate_y1176_spectrin(time_s, activity, *, accessibility, parameters, baseline,
                            gate, activity_multiplier=1., max_step_s=.01):
    """Two fixed-label precursor cohorts, with different first-cut susceptibility.

    The same pre-intervention baseline is split by the requested starting
    fraction, as is synthesis. No kinase/phosphatase interconversion is inferred.
    Selective depletion changes the fraction among surviving intact substrate.
    Only alphaII Y1176 first cleavage is gated; downstream S1230 cleavage is
    unchanged. Not applicable to betaII or TDP-43 phosphorylation.
    """
    if not isinstance(gate,Y1176Gate): raise ValueError('Explicit Y1176Gate required')
    gate.validate()
    phi = gate.initial_phosphorylated_fraction
    rho = gate.phosphorylated_first_cut_relative_rate
    weighted = []
    # Use a COMMON baseline before changing susceptibility, not a separately
    # rebalanced pool that would erase the intervention.
    for fraction, factor in ((1-phi,1.),(phi,rho)):
        p = replace(parameters,first_basal_per_activity_s=parameters.first_basal_per_activity_s*factor,
                     first_accessible_per_activity_s=parameters.first_accessible_per_activity_s*factor)
        r = _previous.simulate_sequential_spectrin(time_s,activity,accessibility=accessibility,
            lineage='alphaII_C',parameters=p,baseline=baseline,
            activity_multiplier=activity_multiplier,max_step_s=max_step_s)
        weighted.append((fraction,r))
    keys = ('intact','first_fragment','second_fragment','removed','synthesized',
            'first_cut_events','second_cut_events','first_cut_neoepitope_amount','retained_fragment_total')
    result = dict(weighted[0][1])
    for key in keys: result[key] = sum(f*r[key] for f,r in weighted)
    residual = result['intact']+result['first_fragment']+result['second_fragment']+result['removed']-result['synthesized']
    intact_phos = phi*weighted[1][1]['intact']
    surviving_fraction = np.divide(intact_phos,result['intact'],out=np.zeros_like(intact_phos),where=result['intact']>0)
    result.update(gate=asdict(gate),intact_phosphorylated=intact_phos,
        phosphorylated_fraction_among_intact=surviving_fraction,
        lineage_balance_error=float(np.max(abs(residual-residual[0]))),
        parameters=asdict(parameters),kinase_phosphatase_dynamics=False)
    return result


def coupled_trace_to_regulated_proteolysis(trace, *, parameters, compartment, basal_ca_uM,
        caspase_hazard_per_s, caspase_provenance, spectrin_parameters, accessibility,
        basal_accessibility, y1176_gate, inhibition=0., max_step_s=.01):
    """Connected conditional TDP and alphaII reporter; no feedback to Vm/Ca."""
    tdp = coupled_trace_to_competing_tdp(trace,parameters=parameters,compartment=compartment,
        basal_ca_uM=basal_ca_uM,caspase_hazard_per_s=caspase_hazard_per_s,
        caspase_provenance=caspase_provenance,inhibition=inhibition,max_step_s=max_step_s)
    basal = _previous.baseline_equilibrium(basal_ca_uM,parameters)['initial_state'][0]/(1+parameters.cast_over_ki)
    baseline = _previous.spectrin_equilibrium(basal,basal_accessibility,lineage='alphaII_C',parameters=spectrin_parameters)
    sp = simulate_y1176_spectrin(tdp['time_s'],tdp['active_fraction']/(1+parameters.cast_over_ki),
        accessibility=accessibility,parameters=spectrin_parameters,baseline=baseline,gate=y1176_gate,
        activity_multiplier=1-tdp['inhibition'],max_step_s=max_step_s)
    return dict(TDP=tdp,alphaII_spectrin=sp,human_kinetic_fit=False,
        pTDP_signal=None,neuronal_viability=None,nuclear_TDP_function=None,NMJ_function=None,
        biochemical_feedback_to_voltage=False)


def regulated_source_evidence():
    return json.loads((_HERE/'evidence_codex.json').read_text(encoding='utf-8'))


def source_training_data():
    result = _previous.source_training_data()
    result['competing_proteolysis_and_site_specific_phosphorylation'] = regulated_source_evidence()
    return result
