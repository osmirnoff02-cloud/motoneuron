"""Codex 2026-11-15: selected Blue Brain reactions, NOT a full NGV replica.

GLUT/HK adapted from BlueBrain/metabolism-in-aging commit 1204a605a8d2397f2b78952f74d163df217ed2f9,
Copyright 2024 Blue Brain Project/EPFL, Apache-2.0 (LICENSE_BlueBrain.txt).
Changes: explicit uM conversion, finite extracellular pool, explicit ATP/ADP,
carbon/energy ledgers. The reduced glycolytic closure and boundary are Codex
assumptions. It does NOT modify the full neuron or add a second PPP there.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import json
import sys
import numpy as np
from scipy.integrate import solve_ivp

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parent / 'neuron-model'))
from model_contracts import load_module
N = load_module(B.parent / '2026-11-14-nadph-supply/nadph_supply.py', '_donor_carbon15_ppp')

NAMES = ('extracellular_glucose_uM', 'glucose_uM', 'G6P_uM', 'pentose_uM',
         'ATP_uM', 'ADP_uM', 'NADPH_uM', 'NADP_uM',
         'boundary_glucose_net_uM', 'transport_glucose_net_uM', 'HK_glucose_uM',
         'glycolysis_C6_out_uM', 'oxidative_PPP_uM', 'pentose_return_uM',
         'NADPH_used_uM', 'ATP_load_used_uM', 'ATP_glycolysis_produced_uM', 'ATP_HK_used_uM')
RANGES = dict(external_volume_ratio=[0.2, 1., 5.], perfusion_tau_s=[60., 600., 3600.],
              glycolysis_ADP_half_uM=[20., 100., 500.], ATP_load_capacity_uM_s=[0., 0.25, 1.],
              NADPH_demand_capacity_uM_s=[0.1, 0.24, 1.], transport_activity=[0., 0.5, 1.],
              HK_activity=[0., 0.5, 1.], G6PD_activity=[0., 1., 2.],
              ATP_load_half_uM=[100., 500., 1000.], NADPH_demand_half_uM=[1., 10., 30.])

@dataclass(frozen=True)
class DonorCarbonParameters:
    # Paired donor law + coefficients. Do not call these human MN measurements.
    HK_rate_per_s: float = 0.51617647
    HK_glucose_half_uM: float = 50.
    HK_G6P_inhibition_uM: float = 10.21
    HK_ATP_inhibition_uM: float = 554.
    HK_ATP_hill: float = 4.
    GLUT_capacity_uM_s: float = 400.
    GLUT_half_uM: float = 2800.
    GLUT_outer_factor: float = 1.
    GLUT_inner_factor: float = 0.72
    GLUT_cross_factor: float = 0.72
    GLUT_neuronal_factor: float = 0.41
    glucose_initial_uM: float = 911.7230433328604
    external_glucose_initial_uM: float = 1003.4276408975709
    G6P_initial_uM: float = 45.1803110921908
    ATP_initial_uM: float = 1384.6374147608125
    # New / inherited prototype assumptions, never copied as measured values.
    ADP_initial_uM: float = 200.
    external_volume_ratio: float = 1.
    perfusion_tau_s: float = 600.
    glycolysis_ADP_half_uM: float = 100.
    ATP_load_capacity_uM_s: float = 0.25
    ATP_load_half_uM: float = 500.
    NADPH_demand_capacity_uM_s: float = 0.24
    NADPH_demand_half_uM: float = 10.
    transport_activity: float = 1.
    HK_activity: float = 1.
    G6PD_activity: float = 1.
    boundary_open: bool = True

    def validate(self):
        values = {k: v for k, v in asdict(self).items() if k != 'boundary_open'}
        if any(not np.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError('Finite nonnegative parameters required')
        positive = ('HK_glucose_half_uM', 'HK_G6P_inhibition_uM', 'HK_ATP_inhibition_uM',
                    'HK_ATP_hill', 'GLUT_half_uM', 'external_volume_ratio', 'perfusion_tau_s',
                    'glycolysis_ADP_half_uM', 'ATP_load_half_uM', 'NADPH_demand_half_uM')
        if any(values[k] <= 0 for k in positive) or not isinstance(self.boundary_open, bool):
            raise ValueError('Invalid denominators or boundary flag')

def bluebrain_hexokinase(glucose_uM, ATP_uM, G6P_uM, parameters=None):
    """Source HEX1_n; uM/s. Source coefficient multiplies dimensional ATP.

    Its dimensional interpretation is s^-1 for this exact law, even though
    its source name starts with Vmax. It must NOT be multiplied by 1000.
    """
    p = parameters or DonorCarbonParameters()
    g, a, q = np.maximum(np.array([glucose_uM, ATP_uM, G6P_uM], dtype=float), 0.)
    return float(p.HK_activity * p.HK_rate_per_s * g / (g + p.HK_glucose_half_uM)
                 * a / (1. + (a / p.HK_ATP_inhibition_uM) ** p.HK_ATP_hill)
                 / (1. + q / p.HK_G6P_inhibition_uM))

def bluebrain_glucose_transport(external_uM, internal_uM, parameters=None):
    """Source JGlc_ecsAN2n including donor dGlc_n coefficient 0.41; uM/s.
    Signed inward. Equal concentrations give zero; reversal is preserved.
    """
    p = parameters or DonorCarbonParameters()
    e, g = max(float(external_uM), 0.), max(float(internal_uM), 0.)
    k = p.GLUT_half_uM
    denominator = k*k + k*p.GLUT_outer_factor*e + k*p.GLUT_inner_factor*g + p.GLUT_cross_factor*e*g
    return p.transport_activity * p.GLUT_neuronal_factor * p.GLUT_capacity_uM_s * k * (e-g) / denominator

def prepare(parameters=None):
    p = parameters or DonorCarbonParameters(); p.validate()
    # Reuse current PPP equation and human recombinant G6PD constants.
    # All pentoses return to G6P equivalents for this reduced carbon closure.
    np_ = N.NADPHSupplyParameters(G6PD_Ki_NADPH_uM=9., other_source_share=0., recycle_fraction=1.)
    ny, q, report = N.prepare(np_, 0.24, 200.)
    y = np.zeros(len(NAMES))
    y[:8] = [p.external_glucose_initial_uM, p.glucose_initial_uM, p.G6P_initial_uM,
             ny[1], p.ATP_initial_uM, p.ADP_initial_uM, report['resting_NADPH_uM'], report['resting_NADP_uM']]
    report.update(parameters=asdict(p), PPP_parameters=asdict(np_), PPP_reference=q.tolist(),
        initialization='Declared mixed-source initial pools; not a stationary whole-neuron state or source NGV replica.',
        closure='G6P to two C3 exported products: 3 ATP after the separately charged HK step; rapid Pi/NAD recycling implicit, no TCA/ETC.',
        boundary='Open perfusion is an explicit signed carbon boundary; closed mode has no external replacement.',
        ownership='Standalone ATP/ADP and NADP(H), no host pools or second host PPP. Full-neuron coupling remains unimplemented.')
    return y, q, report

def fluxes(y, p, q):
    e, g, g6p, p5, a, d, nh, nn = np.maximum(y[:8], 0.)
    transport = bluebrain_glucose_transport(e, g, p)
    hk = bluebrain_hexokinase(g, a, g6p, p)
    ny = np.array([g6p, p5])
    nf = N.fluxes(ny, np.array([nh, 0., p.G6PD_activity, 0.]), q)
    # The old constant f0 is deliberately zero. HK is the sole G6P input.
    gly = nf[1] * d / (p.glycolysis_ADP_half_uM + d)
    ox, ret = nf[2], nf[3]
    nadph_use = p.NADPH_demand_capacity_uM_s * nh / (p.NADPH_demand_half_uM + nh)
    atp_use = p.ATP_load_capacity_uM_s * a / (p.ATP_load_half_uM + a)
    boundary = p.external_volume_ratio * (p.external_glucose_initial_uM-e)/p.perfusion_tau_s if p.boundary_open else 0.
    return np.array([boundary, transport, hk, gly, ox, ret, nadph_use, atp_use])

def rhs(t, y, p, q):
    boundary, jt, hk, gly, ox, ret, nh_use, a_use = fluxes(y, p, q)
    z = np.zeros(len(NAMES))
    z[0] = (boundary-jt)/p.external_volume_ratio
    z[1] = jt-hk
    z[2] = hk-gly-ox+5./6.*ret
    z[3] = ox-ret
    z[4] = 3.*gly-hk-a_use; z[5] = -z[4]
    z[6] = 2.*ox-nh_use; z[7] = -z[6]
    z[8:] = [boundary, jt, hk, gly, ox, ret, nh_use, a_use, 3.*gly, hk]
    return z

def checks(states, base, p):
    d = np.atleast_2d(states)-base
    err = lambda v: float(np.max(np.abs(v)))
    carbon = 6*p.external_volume_ratio*d[:,0]+6*d[:,1]+6*d[:,2]+5*d[:,3]+6*d[:,11]+d[:,12]-6*d[:,8]
    return dict(carbon_uM_C=err(carbon), adenylates_uM=err(d[:,4]+d[:,5]),
        NADP_total_uM=err(d[:,6]+d[:,7]), ATP_ledger_uM=err(d[:,4]+d[:,17]+d[:,15]-d[:,16]),
        NADPH_ledger_uM=err(d[:,6]-2*d[:,12]+d[:,14]), glucose_uM=err(d[:,1]-d[:,9]+d[:,10]),
        G6P_uM=err(d[:,2]-d[:,10]+d[:,11]+d[:,12]-5./6.*d[:,13]),
        pentose_uM=err(d[:,3]-d[:,12]+d[:,13]), HK_cost_uM=err(d[:,17]-d[:,10]),
        negative_physical_uM=float(max(0., -np.atleast_2d(states)[:,:8].min())))

def run_donor_carbon_module(parameters=None, *, duration_s=3600., sample_s=30., checkpoint=None,
                            rtol=1e-8, atol=1e-10):
    """Validated separate module; continuation retains all pools and ledgers."""
    p = parameters or DonorCarbonParameters()
    y, q, report = prepare(p)
    if duration_s <= 0 or sample_s <= 0 or not np.isfinite(duration_s+sample_s):
        raise ValueError('Positive finite duration and sample step required')
    start = 0.
    if checkpoint is not None:
        if checkpoint['parameters'] != asdict(p):
            raise ValueError('Restart requires identical parameters; interventions need a separate explicit protocol')
        y = np.array(checkpoint['state'], dtype=float); start = float(checkpoint['time_s'])
        if y.shape != (len(NAMES),) or not np.isfinite(y).all() or min(y[:8]) < -1e-7:
            raise ValueError('Invalid checkpoint')
        if not np.isclose(y[6]+y[7],q[0],rtol=1e-10):raise ValueError('NADP pool mismatch')
    t = start + np.linspace(0., duration_s, int(np.ceil(duration_s/sample_s))+1)
    sol = solve_ivp(lambda t,y:rhs(t,y,p,q), (start,start+duration_s), y, method='BDF',
                    rtol=rtol, atol=atol, t_eval=t, max_step=min(60.,duration_s))
    if not sol.success:raise ArithmeticError(sol.message)
    z = sol.y.T; c = checks(z,y,p)
    if max(c.values()) > 1e-5:raise ArithmeticError('Conservation/positivity failed: '+repr(c))
    return dict(time_s=t, state=z, names=list(NAMES), checks=c, config=report,
                checkpoint=dict(time_s=float(t[-1]),state=z[-1].tolist(),parameters=asdict(p)))

def donor_parameter_catalog():
    return json.loads((B/'donor_parameters.json').read_text('utf-8'))
