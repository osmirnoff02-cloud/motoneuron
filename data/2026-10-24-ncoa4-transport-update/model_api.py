"""[codex] 2026-10-24. Iron-dependent NCOA4 synthesis/turnover extension, 115 states.

This is an explicit fixed-voltage, mean-open-fraction experiment, not a
days-long spike simulation. No short NEURON trace is repeated or stretched.
All previous APIs are reexported; human kinetic parameters remain unknown.
"""
from pathlib import Path
from dataclasses import asdict, dataclass
import importlib.util, sys, json, hashlib
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

B = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_iron_before_ncoa4_extension', B.parent/'2026-10-23-iron-neuron-network/model_api.py')
_prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _prior
spec.loader.exec_module(_prior)
for name in dir(_prior):
    if not name.startswith('_') and name not in globals():
        globals()[name] = getattr(_prior, name)
UNIFIED_NAMES = IRON_NAMES + ('GluA2_RNA_R_fraction', 'surface_GluA2_R_fraction', 'ferritin_label_soma', 'ferritin_label_axon', 'ferritin_label_lysosome', 'ferritin_label_degraded', 'NCOA4_available', 'NCOA4_synthesized', 'NCOA4_proteasomal_loss', 'NCOA4_autophagic_loss')
UNIFIED_DRIVERS = IRON_DRIVERS[:2] + ('AMPA_mean_open_fraction',) + IRON_DRIVERS[3:] + ('ADAR2_activity', 'AMPA_capacity_scale', 'lysosomal_acid_support', 'ferritin_proteolysis_support', 'endosomal_acid_support', 'NCOA4_synthesis_scale', 'HERC2_support')
_iron_rhs = _prior._iron_rhs

@dataclass(frozen=True)
class IronReleaseParameters:
    # f=0: proteolysis-associated; f=1: acid-supported extraction independent of proteolysis.
    acid_only_fraction: float
    soma_position_coupling: float = 0.
    provenance: str = 'Asano2011 Fig3 vs Kidane2006 different cells/assays; fraction unknown in human MN. Farias2017 rat hippocampal axonal transport, not global hydrolysis.'
    def vector(self):
        v=np.array([self.acid_only_fraction,self.soma_position_coupling],float)
        if not np.isfinite(v).all() or np.any(v<0) or np.any(v>1) or not self.provenance.strip():
            raise ValueError('Fractions in [0,1] and source provenance required')
        return v

def trpml_pi35_assay(lipid_nM):
    """Dong2010 Fig1d: diC8 PI(3,5)P2, recombinant channel/endolysosomal patch.
    Normalized ligand-dependent activation, not absolute current/open probability
    and never automatically assigned to human Fe or Ca flux.
    """
    x=np.asarray(lipid_nM,float)
    if not np.isfinite(x).all() or np.any(x<0):raise ValueError('Finite nonnegative nM required')
    return x**1.9/(48.**1.9+x**1.9)

def ferritin_assay_reference(cell, hours):
    """Apparent exponentials using labels printed in Kidane2006 Fig1, NOT a new fit.
    Experimental whole-cell apparent t1/2 cannot identify intracellular rate constants.
    Background/plateaus not reconstructed; this is a reference operator only.
    """
    half={'rat_hepatoma':(24.,22.5),'human_Caco2':(17.,14.5),'human_K562':(12.5,10.)}
    if cell not in half:raise ValueError('Only source cell types allowed; no human MN calibration')
    t=np.asarray(hours,float)
    if not np.isfinite(t).all() or np.any(t<0):raise ValueError('Nonnegative hours required')
    a,b=half[cell]
    return dict(protein_relative=2.**(-t/a),ferritin_Fe59_relative=2.**(-t/b),
                protein_half_life_h=a,Fe59_half_life_h=b,human_motor_neuron=False,
                source='Kidane2006 Fig1 labels, PDF C446; DFO 100uM after overnight 180uM FAC. Rounded abstract values differ.')



@dataclass(frozen=True)
class NCOA4Parameters:
    # Human motor-neuron values unknown; no fit to U2OS is silently transferred.
    baseline_half_life_h: float
    herc2_loss_fraction: float
    iron_half_response_relative: float = 1.
    iron_hill: float = 1.
    enabled: bool = True
    provenance: str = 'Mancias2015 Fig3 and Fig3-S1: parallel HERC2/proteasome and autophagic turnover; rates and sensing curve are hypotheses for human MN'
    def vector(self):
        v=np.array([self.baseline_half_life_h*3600.,self.herc2_loss_fraction,
                    self.iron_half_response_relative,self.iron_hill,float(self.enabled)],float)
        if not np.isfinite(v).all() or min(v[0],v[2],v[3])<=0 or not 0<=v[1]<=1 or not isinstance(self.enabled,bool) or not self.provenance.strip():
            raise ValueError('Positive finite times/sensor scales, HERC2 fraction [0,1], bool enabled and provenance required')
        return v

@njit
def _ncoa4_rates(z,row,qv,base_iron):
    if qv[4]==0.:return 0.,0.,0.
    n=max(z[111],0.)
    x=max(z[83],0.)/base_iron
    # Normalized at baseline iron x=1; a phenomenological response, NOT measured Kd.
    xn=x**qv[3];kn=qv[2]**qv[3]
    sensor=(1.+kn)*xn/(kn+xn)
    k=np.log(2.)/qv[0]
    production=k*row[27]
    ups=k*qv[1]*row[28]*row[10]*sensor*n
    auto=k*(1.-qv[1])*max(z[72],0.)*row[24]*row[12]*n
    return production,ups,auto

@njit
def _flux(surface_R, opening, capacity, ep):
    # Existing conditional assembly: any edited R dominates, weights n=0..4.
    r = min(max(surface_R, 0.), 1.)
    cp = 0.
    for n in range(5):
        cp += ep[3+n]*(1.-r)**n
    ci_current = ep[8]*ep[9]*(1.-cp)*opening*capacity
    cp_current = ep[8]*ep[9]*cp*ep[10]*ep[11]*opening*capacity
    raw_flux = (ci_current*ep[12]+cp_current*ep[13])*1e9/(2.*96485.33212*ep[14])
    return raw_flux/(1.+ep[15]), cp, ci_current+cp_current

@njit
def _rhs(z, p, kv, sv, av, hv, nv, ag0, iv, base_n, row, ep, target, rv, qv, base_iron):
    row=row.copy()
    row[16]*=max(z[111],0.) if qv[4]>0. else 1.
    j, _, _ = _flux(z[106], row[2], row[23], ep)
    dz = np.zeros(115)
    dz[:105] = _iron_rhs(z[:105], j, p, kv, sv, av, hv, row[3], nv, row[4:13], ag0, iv, row[13:22], base_n)
    # Conservative replacement of bound-ferritin Fe -> lysosomal labile Fe.
    acid,protease,endo_acid=row[24:27]
    lyso=max(z[72],0.)
    old_unload=lyso*max(z[103],0.)/iv[1]
    new_unload=acid*(rv[0]+(1.-rv[0])*protease*lyso)*max(z[103],0.)/iv[1]
    delta=new_unload-old_unload
    dz[85]+=delta;dz[103]-=delta
    # Bafilomycin-like loss of acidification can also affect endosomal Fe uptake.
    old_endo=row[14]*max(z[104],0.)/iv[3]
    delta_endo=(endo_acid-1.)*old_endo
    dz[83]+=delta_endo;dz[104]-=delta_endo
    encounter=(1.-nv[21]+nv[21]*max(z[68],0.))*row[12]
    soma_encounter=(1.-rv[1]+rv[1]*(1.-nv[21]+nv[21]*max(z[68],0.)))*row[12]
    delta_delivery=row[16]*(soma_encounter-encounter)*max(z[84],0.)/iv[0]
    dz[84]-=delta_delivery;dz[103]+=delta_delivery;dz[99]+=delta_delivery
    # Passive pulse label: no inference of total ferritin protein from Fe content.
    ps=row[16]*soma_encounter*max(z[107],0.)/iv[0]
    pa=row[16]*encounter*max(z[108],0.)/iv[0]
    pd=acid*protease*lyso*max(z[109],0.)/iv[1]
    dz[107]=-ps;dz[108]=-pa;dz[109]=ps+pa-pd;dz[110]=pd
    m = min(max(z[105], 0.), 1.)
    incorporation = ep[2]*m/(ep[2]*m+1.-m)
    dz[105] = (target-z[105])/ep[0]
    dz[106] = (incorporation-z[106])/ep[1]
    production,ups,auto=_ncoa4_rates(z,row,qv,base_iron)
    dz[111]=production-ups-auto
    dz[112]=production;dz[113]=ups;dz[114]=auto
    return dz

def run_ncoa4_iron(segments, *, ncoa4_parameters, iron_release_parameters, editing_parameters, mapping_family,
                   editing_provenance, iron_parameters, checkpoint=None,
                   output_step_s=1800., max_step_s=1800., rtol=1e-9, atol=1e-11, **kw):
    """29-column contiguous schedule, seconds; ADAR2 affects newly made RNA.

    All states start in their normal basal equilibria at the first time point.
    Receptor and biochemical history then evolve together, including rest.
    Calcium uses buffered effective flux J/(1+kappa), with tau=(1+kappa)/gamma.
    The supplied AMPA opening is a declared constant within each segment;
    neither agonist concentration nor spike frequency is inferred from it.
    """
    sg = np.asarray(segments, float)
    if sg.ndim != 2 or sg.shape[1] != 29 or not len(sg) or not np.isfinite(sg).all() or np.any(sg < 0) or np.any(sg[:,1] <= sg[:,0]) or np.any(sg[1:,0] != sg[:-1,1]):
        raise ValueError('Contiguous finite nonnegative 29-column schedule required')
    if np.any(sg[:,[2,3,4,5,6,8,9,10,11,12,22,24,25,26]] > 1):
        raise ValueError('Fraction drivers must be in [0,1]')
    if not isinstance(editing_provenance, str) or not editing_provenance.strip():
        raise ValueError('Explicit editing/source provenance required')
    if not np.isfinite([output_step_s,max_step_s,rtol,atol]).all() or min(output_step_s,max_step_s,rtol,atol) <= 0:
        raise ValueError('Positive solver settings required')
    rv=iron_release_parameters.vector()
    qv=ncoa4_parameters.vector()
    e = editing_parameters
    # Calls the original comprehensive AMPA validation, without a preparation run.
    e0 = editing_trajectory(1., e, hours=0., family=mapping_family)
    if not np.isclose(e.basal_ca_uM, kw['basal_ca_uM'], rtol=1e-12, atol=0.) or not np.isclose((1.+e.kappa)/e.gamma_per_s, kw['calcium_tau_s'], rtol=1e-12, atol=0.):
        raise ValueError('Calcium baseline and buffer/removal parameters must agree across blocks')
    targets = np.array([editing_target(row[22], e.normal_editing, mapping_family) for row in sg])
    ep = np.array([e.tau_rna_h*3600., e.tau_surface_h*3600., e.incorporation_R_over_Q,
                   *e.weights, e.g_reference_nS, e.reversal_mV-e.voltage_mV,
                   e.cp_conductance_ratio, e.cp_unblocked_fraction, e.pf_ci, e.pf_cp,
                   e.volume_um3, e.kappa], float)
    base_row = np.r_[0., 1e-4, 0., 0., 0., 0., 0., np.ones(15)]
    initial = _prior.calcium_iron_chain([base_row], iron_parameters=iron_parameters, **kw)
    label=np.array([4.,.8,4.8*iron_parameters.lysosomal_digestion_s/iron_parameters.ferritin_turnover_s]); label/=label.sum()
    base = np.r_[initial['baseline_state'], e0['rna_editing'][0], e0['surface_R'][0], label, 0., 1., 0., 0., 0.]
    iv = iron_parameters.vector(); nv = kw['organelle_parameters'].vector()
    av = kw['aggregation_parameters'].vector()
    hv = np.r_[kw['autoregulation_parameters'].vector(), kw['rna_feedback_parameters'].vector()]
    required = {'basal_ca_uM','calcium_tau_s','enzyme_parameters','tdp_parameters','localization_parameters','stmn2_parameters','inactivation_parameters','phosphorylation_parameters','reactivation_parameters','stability_parameters'}
    _, p, kv, sv, _ = _prior._prior._agg._baseline({k:kw[k] for k in required}, av)
    ag0 = base[37:45].sum(); base_n = base[6]+base[7]
    config = dict(initial['config'], topology='continuous_editing_ncoa4_iron115_v1', ncoa4=asdict(ncoa4_parameters), iron_release=asdict(iron_release_parameters), ferritin_label_initial='Assumed steady delivery weights 4:0.8:(4.8*tdigest/tferritin), normalized; no human measurement', editing=asdict(e),
                  mapping_family=mapping_family, editing_provenance=editing_provenance,
                  stimulus='Fixed voltage and explicitly supplied mean open fraction; no action potentials simulated')
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if checkpoint is None:
        y = base.copy(); origin = float(sg[0,0])
    else:
        if checkpoint['fingerprint'] != fingerprint or checkpoint['time_s'] != sg[0,0] or not np.array_equal(checkpoint['baseline_state'], base):
            raise ValueError('Checkpoint parameters, time or baseline differ')
        y = np.asarray(checkpoint['state'], float).copy(); origin = checkpoint['origin_time_s']
        if y.shape != (115,) or not np.isfinite(y).all() or y.min() < -1e-7 or np.any(y[105:107] > 1.+1e-7):
            raise ValueError('Invalid checkpoint state')
    ts = [sg[0,:1].copy()]; zs = [y[None,:]]; nfev = 0
    for row, target in zip(sg, targets):
        start, end = row[:2]
        near = start+np.geomspace(1e-4, min(end-start,3600.), 70)
        grid = np.unique(np.r_[np.arange(start+output_step_s,end,output_step_s), near[near<end], end])
        sol = solve_ivp(lambda t,z:_rhs(z,p,kv,sv,av,hv,nv,ag0,iv,base_n,row,ep,target,rv,qv,base[83]),
                        (start,end),y,t_eval=grid,method='LSODA',rtol=rtol,atol=atol,max_step=max_step_s)
        if not sol.success: raise RuntimeError(sol.message)
        y = sol.y[:,-1].copy(); ts.append(sol.t); zs.append(sol.y.T); nfev += sol.nfev
    t = np.concatenate(ts); z = np.concatenate(zs)
    idx = np.clip(np.searchsorted(sg[:,0], t, side='right')-1, 0, len(sg)-1)
    obs = np.array([_flux(r,sg[i,2],sg[i,23],ep) for r,i in zip(z[:,106],idx)])
    ag = z[:,37:45].sum(1)
    enzyme = z[:,1:6].sum(1)+z[:,31]+z[:,33]
    enzyme0 = base[1:6].sum()+base[31]+base[33]
    td = z[:,6:10].sum(1)+ag; td0 = base[6:10].sum()+ag0
    err = lambda a:float(abs(a).max())
    checks = dict(enzyme_balance=err(enzyme+z[:,15]-z[:,14]-enzyme0),
        TDP_lineage_balance=err(td+z[:,19]-z[:,18]-td0),
        aggregate_balance=err(ag-ag0-z[:,45]+z[:,46]+z[:,47]+z[:,49]),
        calcium_balance_uM=err(z[:,0]+z[:,23]-z[:,22]),
        calcium_AUC_identity=err(z[:,24]-p[1]*z[:,23]),
        PKN1_AF3_balance=err(z[:,63]+z[:,66]+z[:,76]-z[:,75]-1.),
        PKN207_balance=err(z[:,65]+z[:,78]-z[:,77]),
        secretory_balance=err(z[:,69:72].sum(1)+z[:,80]-z[:,79]-base[69:72].sum()),
        autophagic_balance=err(z[:,73:75].sum(1)+z[:,82]-z[:,81]-base[73:75].sum()),
        iron_mass_balance=err(z[:,FE_INDEX].sum(1)+z[:,97]-z[:,96]-base[FE_INDEX].sum()),
        ferritin_bound_balance=err(z[:,84]+z[:,87]-base[84]-base[87]-z[:,98]+z[:,99]),
        lipid_oxidation_balance=err(z[:,93:96].sum(1)-base[93:96].sum()-z[:,101]+z[:,102]),
        editing_bounds_error=float(max(0., -z[:,105:107].min(), z[:,105:107].max()-1.)),
        ferritin_label_balance=err(z[:,107:111].sum(1)-1.),
        NCOA4_balance=err(z[:,111]+z[:,113]+z[:,114]-z[:,112]-1.),
        minimum_state=float(min(z.min(),(z[:,1]-z[:,28]).min(),(z[:,2]-z[:,29]).min(),(z[:,31]-z[:,32]).min())),
        tracer_excess=float(max(0.,np.max(z[:,51:63]-z[:,_prior._prior._prior.LABEL_INDEX]))))
    if not np.isfinite(z).all() or checks['minimum_state'] < -1e-7 or max(v for k,v in checks.items() if k!='minimum_state') > 5e-5:
        raise ArithmeticError(str(checks))
    cp = dict(time_s=float(t[-1]),state=y.tolist(),baseline_state=base.tolist(),origin_time_s=origin,fingerprint=fingerprint)
    return dict(time_s=t,state=z,columns=UNIFIED_NAMES,segments=sg,config=config,checks=checks,checkpoint=cp,
        baseline_state=base,nfev=nfev,calcium_uM=p[0]+z[:,0],effective_ca_flux_uM_s=obs[:,0],
        raw_ca_flux_uM_s=obs[:,0]*(1.+e.kappa),cp_receptor_fraction=obs[:,1],AMPA_inward_pA=obs[:,2],
        activity=(z[:,28]+z[:,29]+z[:,32])*(1.-sg[idx,3]),aggregate_total=ag,
        nuclear_function_proxy=z[:,6]+z[:,7],iron_total=z[:,FE_INDEX].sum(1),
        ferritin_label_remaining=z[:,107:110].sum(1),ncoa4_available=z[:,111],human_kinetic_fit=False,electrical_feedback=False,absolute_TDP_per_cell=None,
        limitations=initial['limitations']+[
            'NCOA4 is an effective shared available protein pool, not separate nuclear/somal/axonal protein concentrations',
            'HERC2 iron sensing is a normalized phenomenological curve, not iron-binding Kd or a calibrated human response',
            'NCOA4 synthesis, binding competence (driver16), HERC2 support, and global UPS/autophagy support are distinct',
            'NCOA4-dependent cargo delivery is inherited; residual NCOA4-independent ferritin routes are not resolved',
            'NCOA4 autophagic loss is an effective lysosome/acid/fusion-dependent term; not measured stoichiometry with ferritin',
            'Acid-supported Fe extraction and ferritin proteolysis are separate; acid_only_fraction [0,1] is not identified in human MN',
            'Acid drivers are normalized support, NOT pH; drug concentration is not inferred. Global lysosome support must be separately specified for non-ferritin substrates',
            'Ferritin pulse label is passive, normalized, separate from Fe and total ferritin capacity; no protein synthesis or iron occupancy inferred from it',
            'Soma/axon delivery distinction is informed by rat hippocampal BORC perturbations; remaining bulk TDP clearance still uses inherited effective encounter factor',
            'Two uncalibrated ADAR2 maps and assumed RNA/surface turnover; activity is not ADAR2 RNA abundance',
            'Mean open fraction at fixed voltage, not a pulse train or freely firing neuron; fast calcium peaks unresolved',
            'No short electrical trace extrapolation; normal receptor and biochemical state at schedule origin',
            'Uniform cytosolic buffer and fixed receptor assembly, conductance and fractional calcium charge remain assumptions'])

def pulse_chase_readout(result, **kw):
    if np.asarray(result['state']).shape[1] in (107,111,115):
        result = dict(result,state=result['state'][:,:105],baseline_state=result['baseline_state'][:105])
    return _prior.pulse_chase_readout(result, **kw)

def run_adar2_iron(*args, **kwargs):
    """Compatibility entry point: the preserved 107-state model, without new drivers."""
    name='_preserved_unified107_api'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,B.parent/'2026-10-24-adar2-iron-unified/model_api.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name].run_adar2_iron(*args,**kwargs)


def run_lysosomal_iron(*args, **kwargs):
    """Compatibility: unchanged preserved 111-state model."""
    name='_preserved_lysosomal111_api'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,B.parent/'2026-10-24-lysosomal-iron-papers/model_api.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name].run_lysosomal_iron(*args,**kwargs)
