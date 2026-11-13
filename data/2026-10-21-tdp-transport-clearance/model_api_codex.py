"""[codex] 2026-10-21. Conditional TDP localization and assay-specific turnover.

Park 2020 motivates separating localization from cleavage, not human import rates.
Zhang 2010 and Li 2011 motivate a separate C-fragment phosphorylation assay.
McCartney 2018 defines a two-channel reporter; fluorescence is not enzyme activity.
No new human kinetic fit or change to the inherited electrical/Ca/proteolysis model.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util
import sys
import math
import numpy as np
from scipy.linalg import expm
from numba import njit

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_before_localization20261021', _HERE.parent/'2026-10-21-calcium-protease-update/model_api_codex.py')
_previous = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _previous
_spec.loader.exec_module(_previous)
for _name in dir(_previous):
    if not _name.startswith('_') and _name not in globals():
        globals()[_name] = getattr(_previous, _name)


def _time(values):
    t = np.asarray(values, float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('At least two finite, strictly increasing time points in seconds required')
    return t


def _values(values, t, name, upper=None):
    a = np.full(t.shape, float(values)) if np.ndim(values) == 0 else np.asarray(values, float)
    if a.shape != t.shape or not np.isfinite(a).all() or np.any(a < 0) or (upper is not None and np.any(a > upper)):
        raise ValueError(f'{name}: nonnegative finite scalar or matching array required')
    return a


def _provenance(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Explicit source or assumption required')


@njit
def _partition(t, total, kin, kout, synthesis, birth_fraction, f0, max_step):
    f = np.empty(len(t)); f[0] = f0
    rate = np.max(kin+kout)+synthesis/np.min(total)
    safe = min(max_step, .1/rate) if rate > 0 else max_step
    for i in range(len(t)-1):
        n = max(1, int(math.ceil((t[i+1]-t[i])/safe)))
        h = (t[i+1]-t[i])/n; y = f[i]
        for j in range(n):
            p0 = total[i]+(total[i+1]-total[i])*j/n
            pm = total[i]+(total[i+1]-total[i])*(j+.5)/n
            p1 = total[i]+(total[i+1]-total[i])*(j+1)/n
            a, b = kin[i], kin[i]+kout[i]
            k1 = a-b*y+synthesis/p0*(birth_fraction-y)
            z = y+h*k1/2; k2 = a-b*z+synthesis/pm*(birth_fraction-z)
            z = y+h*k2/2; k3 = a-b*z+synthesis/pm*(birth_fraction-z)
            z = y+h*k3; k4 = a-b*z+synthesis/p1*(birth_fraction-z)
            y += h*(k1+2*k2+2*k3+k4)/6
        f[i+1] = y
    return f


def partition_intact_tdp(time_s, intact_total, *, import_per_s, export_per_s,
                         synthesis_per_s, synthesis_nuclear_fraction,
                         initial_nuclear_fraction, provenance, max_step_s=.01):
    """Conditional N/C amounts, not concentrations or RNA function.

    Equal per-molecule losses in both compartments leave the previous whole-cell
    trajectory intact. f=N/P obeys f'=kin(1-f)-kout*f+(s/P)(phi_birth-f).
    P is linearly interpolated; supplied import/export are left-constant.
    This allocation is invalid for compartment-specific cleavage/clearance.
    Fractions and transport rates must be supplied; Park FRAP is not an import fit.
    Export is a lumped return rate, potentially passive diffusion (Pinarbasi 2018),
    not an XPO1/NES-specific process. Synthesis allocation is phenomenological;
    it does not represent nuclear translation or a ribosome compartment.
    """
    _provenance(provenance); t = _time(time_s)
    total = _values(intact_total, t, 'Total intact TDP')
    if np.any(total <= 0):
        raise ValueError('Strictly positive total needed for fraction dynamics')
    kin = _values(import_per_s, t, 'Import'); kout = _values(export_per_s, t, 'Export')
    v = [synthesis_per_s, synthesis_nuclear_fraction, initial_nuclear_fraction, max_step_s]
    if not np.isfinite(v).all() or synthesis_per_s < 0 or max_step_s <= 0 or not 0 <= synthesis_nuclear_fraction <= 1 or not 0 <= initial_nuclear_fraction <= 1:
        raise ValueError('Invalid synthesis, fraction, or integration step')
    f = _partition(t,total,kin,kout,synthesis_per_s,synthesis_nuclear_fraction,initial_nuclear_fraction,max_step_s)
    if not np.isfinite(f).all() or f.min() < -1e-12 or f.max() > 1+1e-12:
        raise ArithmeticError('Invalid compartment fraction')
    n = f*total; c = (1-f)*total
    return dict(time_s=t, nuclear_intact=n, cytoplasmic_intact=c,
        nuclear_fraction=f, total_intact=total, provenance=provenance,
        partition_balance_error=float(np.max(abs(n+c-total))),
        human_kinetic_fit=False, calcium_to_import_calibrated=False,
        nuclear_RNA_function=None, geometry_or_nuclear_pores_modeled=False,
        assumption='Equal per-molecule loss; explicit independent transport and synthesis allocation')


def coupled_trace_to_localized_tdp(trace, *, parameters, compartment, basal_ca_uM,
        import_per_s, export_per_s, initial_nuclear_fraction,
        synthesis_nuclear_fraction, transport_provenance,
        caspase_hazard_per_s, caspase_provenance, inhibition=0., max_step_s=.01):
    """Inherited proteolysis plus a conditional partition of intact protein only."""
    r = _previous.coupled_trace_to_competing_tdp(trace,parameters=parameters,
        compartment=compartment,basal_ca_uM=basal_ca_uM,
        caspase_hazard_per_s=caspase_hazard_per_s,caspase_provenance=caspase_provenance,
        inhibition=inhibition,max_step_s=max_step_s)
    s = baseline_equilibrium(basal_ca_uM,parameters)['synthesis_per_s']
    loc = partition_intact_tdp(r['time_s'],r['intact_TDP'],import_per_s=import_per_s,
        export_per_s=export_per_s,synthesis_per_s=s,
        synthesis_nuclear_fraction=synthesis_nuclear_fraction,
        initial_nuclear_fraction=initial_nuclear_fraction,
        provenance=transport_provenance,max_step_s=max_step_s)
    return dict(TDP=r,localization=loc,feedback_to_electrical_model=False)


@dataclass(frozen=True)
class CFragmentAssayParameters:
    phosphorylation_per_s: float
    dephosphorylation_per_s: float
    unphosphorylated_clearance_per_s: float
    phosphorylated_clearance_per_s: float
    provenance: str


def simulate_c_fragment_chase(time_s, *, initial_unphosphorylated,
        initial_phosphorylated, parameters, synthesis_per_s=0.,
        synthesis_phosphorylated_fraction=0.):
    """Separate C220-414-like assay U <-> P, not the neuron's N-fragment lineage.

    After shutoff, total=U+P and phospho abundance=P. Antibody gain/recovery must
    be modeled separately. Zhang total half-life 14.2h is NOT U's half-life.
    Parameters are explicit assumptions; no Li aggregation/protection law added.
    """
    _provenance(parameters.provenance); t = _time(time_s)
    kp,kdp,du,dp = (parameters.phosphorylation_per_s,parameters.dephosphorylation_per_s,
                    parameters.unphosphorylated_clearance_per_s,parameters.phosphorylated_clearance_per_s)
    v = [kp,kdp,du,dp,initial_unphosphorylated,initial_phosphorylated,synthesis_per_s,synthesis_phosphorylated_fraction]
    if not np.isfinite(v).all() or min(v)<0 or synthesis_phosphorylated_fraction>1:
        raise ValueError('Finite nonnegative assay rates/amounts and valid fraction required')
    # U, P, removed, synthesized, constant; exact propagation for constant rates.
    M = np.zeros((5,5)); M[0,0]=-(kp+du); M[0,1]=kdp
    M[1,0]=kp; M[1,1]=-(kdp+dp); M[2,:2]=[du,dp]
    M[0,4]=synthesis_per_s*(1-synthesis_phosphorylated_fraction)
    M[1,4]=synthesis_per_s*synthesis_phosphorylated_fraction; M[3,4]=synthesis_per_s
    y = np.array([initial_unphosphorylated,initial_phosphorylated,0.,0.,1.])
    out = [y.copy()]; propagators = {}
    for dt in np.diff(t):
        if dt not in propagators: propagators[dt] = expm(M*dt)
        y = propagators[dt]@y; out.append(y.copy())
    z = np.asarray(out); total=z[:,0]+z[:,1]
    balance=total+z[:,2]-z[:,3]-(initial_unphosphorylated+initial_phosphorylated)
    if not np.isfinite(z).all() or z.min() < -1e-11 or np.max(abs(balance))>1e-9*max(1.,float(total.max())):
        raise ArithmeticError('Assay amount balance failed')
    return dict(time_s=t,unphosphorylated=z[:,0],phosphorylated=z[:,1],total=total,
        removed=z[:,2],synthesized=z[:,3],balance_error=float(np.max(abs(balance))),
        parameters=asdict(parameters),human_kinetic_fit=False,
        species='Separate C-terminal fragment assay; not endogenous whole TDP or calpain N fragments',
        aggregation_or_viability=None)


def observe_mccartney_fret(time_s, *, intact_reporter, cleaved_reporter,
        channel_coefficients, backgrounds, bleaching_per_s, provenance):
    """[CFP485,YFP528] under 434-nm excitation; YFP/CFP ratio, not efficiency.

    Rows of coefficients are CFP,YFP; columns are intact,cleaved amounts.
    The four brightness coefficients require endpoint/channel calibration.
    A common channel bleaching rate is assumed for both substrate states.
    Proteolysis history and off-target proteases cannot be inverted to instantaneous
    calpain activity from this observer alone. No Table1 absolute rate is imported.
    """
    _provenance(provenance);t=_time(time_s)
    u=_values(intact_reporter,t,'Intact reporter');f=_values(cleaved_reporter,t,'Cleaved reporter')
    A=np.asarray(channel_coefficients,float);b=np.asarray(backgrounds,float);k=np.asarray(bleaching_per_s,float)
    if A.shape!=(2,2) or b.shape!=(2,) or k.shape!=(2,) or not all(np.isfinite(x).all() and np.all(x>=0) for x in (A,b,k)):
        raise ValueError('Nonnegative 2x2 brightness matrix and two backgrounds/bleaching rates required')
    intensity=(A@np.stack([u,f]))*np.exp(-k[:,None]*(t-t[0]))+b[:,None]
    if np.any(intensity[0]<=0):raise ValueError('Positive CFP denominator required')
    return dict(time_s=t,CFP485=intensity[0],YFP528=intensity[1],
        YFP_over_CFP=intensity[1]/intensity[0],provenance=provenance,
        inferred_calpain_activity=None,calpain_exclusive=False,
        absolute_TDP_cleavage_rate=None,brightness_matrix_rank=int(np.linalg.matrix_rank(A)))


def source_training_data():
    result=dict(_previous.source_training_data())
    result['tdp_localization_and_assay_scope'] = dict(
        Park2020=dict(doi='10.7554/eLife.60132',system='Drosophila C4da sensory neurons; CalpA',
            FRAP_half_recovery_s={'ATR_minus':46.7,'ATR_plus':24.9},
            import_rate_identified=False,ATR_plus_time_column_unresolved=True,
            CalpA_is_not_CAPN1_or_CAPN2=True),
        Zhang2010=dict(doi='10.1186/1750-1326-5-33',system='M17D3 GFP-TDP220-414 pulse-chase',
            apparent_total_half_life_h=14.2,apparent_phospho_half_life_h=22.1,
            total_is_not_unphosphorylated=True,human_motor_neuron_fit=False),
        Li2011=dict(doi='10.1371/journal.pone.0023075',universal_phospho_protection_law=False),
        Pinarbasi2018=dict(doi='10.1038/s41598-018-25008-4',
            system='HeLa/3T3 shuttling and rat hippocampal cultures',
            export_is_not_automatically_XPO1=True,human_motor_neuron_rates_identified=False),
        McCartney2018=dict(doi='10.1016/j.bbamcr.2017.10.013',readout='YFP528/CFP485, excitation434',
            PLFAAR_over_alpha_spectrin={'calpain1':224/48,'calpain2':593/195},
            absolute_Table1_rate_not_used='Printed scale inconsistent with concentration/time; unresolved',
            cell_experiment='Lysed MDA-MB-231 breast cancer cells, 5mM added Ca'))
    return result
