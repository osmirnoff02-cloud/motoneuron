"""[codex] 2026-10-24. Same-cell long ER/Ca/ADAR2/proteostasis/iron network.
134 states: preserved 120-state ER network, ATP/ADP/AMP, 10 ledgers and PMCA2 memory.
Fast membrane mode is separate; this mode does not extrapolate short spike traces.
"""
from pathlib import Path
from dataclasses import asdict
import importlib.util,sys,json,hashlib
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'2026-10-24-atp-calcium-feedback'))
from pmca_memory import PMCAMemoryParameters,memory_rates,memory_equilibrium,pmca2_flux
from energy_parameters import EnergyParameters,energy_fluxes,validate_schedule
B=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('_er_long_previous',B.parent/'2026-10-24-ncoa4-transport-update/model_api.py')
previous=importlib.util.module_from_spec(sp);sys.modules[sp.name]=previous;sp.loader.exec_module(previous)
for n in dir(previous):
    if not n.startswith('_') and n not in globals():globals()[n]=getattr(previous,n)
_prior=previous._prior
_old_rhs=previous._rhs
_flux=previous._flux
ER_NETWORK_NAMES=previous.UNIFIED_NAMES+('ER_total_Ca_per_cytosol_uM','SERCA_Ca_cumulative_uM','ER_release_cumulative_uM','ER_leak_net_cumulative_uM','SOCE_Ca_cumulative_uM')

@njit
def _rhs_er(z,p,kv,sv,av,hv,nv,ag0,iv,base_n,row,ep,target,rv,qv,base_iron,ev,voltage):
    r=row[:29].copy()
    r[19]*=row[29];r[17]*=row[30]
    out=np.zeros(120)
    out[:115]=_old_rhs(z[:115],p,kv,sv,av,hv,nv,ag0,iv,base_n,r,ep,target,rv,qv,base_iron)
    if ev[0]==0.:return out
    c=max(p[0]+z[0],0.);e=z[115]/ev[1];c0=p[0]
    vmax=ev[3];k=ev[4]
    basal=vmax*c0*c0/(k*k+c0*c0)*.5
    serca=row[32]*vmax*c*c/(k*k+c*c)*max(1-e/(2*ev[2]),0.)
    leak=basal*(e-c)/(ev[2]-c0)
    h=ev[7]
    signal=row[31]+ev[6]*max(c*c/(h*h+c*c)-c0*c0/(h*h+c0*c0),0.)
    release=ev[5]*signal*ev[1]*max(e-c,0.)
    # g(nS)*dV(mV) = pA; pA -> uM/s by 5182.1348283 / volume_um3.
    soce=row[33]*ev[8]*max(1-e/ev[2],0.)*(120.-voltage)*5182.1348283/ep[14]
    net=leak+release-serca
    out[0]+=(net+soce)/(1+ep[15])
    out[22]+=soce/(1+ep[15])
    out[115]=-net;out[116]=serca;out[117]=release;out[118]=leak;out[119]=soce
    return out

ENERGY_NAMES=('ATP_uM','ADP_uM','AMP_uM','OXPHOS_ATP_uM','glycolysis_ATP_uM','other_ATP_uM','PMCA_ATP_uM','SERCA_ATP_uM','NCX_gradient_ATP_equivalent_uM','AK_net_ATP_uM','PMCA_gross_Ca_uM','NCX_gross_Ca_uM','basal_Ca_in_uM')

@njit
def _rhs_energy(z,p,kv,sv,av,hv,nv,ag0,iv,base_n,row,ep,target,rv,qv,base_iron,ev,voltage,en,drive):
    out=np.zeros(133)
    out[:120]=_rhs_er(z[:120],p,kv,sv,av,hv,nv,ag0,iv,base_n,row,ep,target,rv,qv,base_iron,ev,voltage)
    gamma=(1+ep[15])/p[1];c0=p[0];c=max(c0+z[0],0.)
    s0=ev[0]*ev[3]*c0*c0/(ev[4]*ev[4]+c0*c0)*.5
    old_serca=out[116]
    pm,nc,se,other,ox,gly,ak=energy_fluxes(z[120],z[121],z[122],c,old_serca,gamma,c0,s0,en,drive)
    net=pm+nc-gamma*c0
    out[0]+=z[0]/p[1]-net/(1+ep[15])+(old_serca-se)/(1+ep[15])
    out[23]=net/(1+ep[15])
    out[115]+=se-old_serca;out[116]=se
    cost=other+pm+se/2+nc*en[10];prod=ox+gly
    out[120]=prod-cost+ak;out[121]=cost-prod-2*ak;out[122]=ak
    out[123:133]=np.array([ox,gly,other,pm,se/2,nc*en[10],ak,pm,nc,gamma*c0])
    return out

def run_pmca_network(segments, *, pmca_parameters, energy_parameters, energy_schedule=None, er_parameters, protein_input_provenance, ncoa4_parameters, iron_release_parameters, editing_parameters, mapping_family,
                   editing_provenance, iron_parameters, checkpoint=None,
                   output_step_s=1800., max_step_s=1800., rtol=1e-9, atol=1e-11, **kw):
    """34-column contiguous schedule, seconds; ADAR2 affects newly made RNA.

    All states start in their normal basal equilibria at the first time point.
    Receptor and biochemical history then evolve together, including rest.
    Calcium uses buffered effective flux J/(1+kappa), with tau=(1+kappa)/gamma.
    The supplied AMPA opening is a declared constant within each segment;
    neither agonist concentration nor spike frequency is inferred from it.
    """
    sg = np.asarray(segments, float)
    if sg.ndim != 2 or sg.shape[1] != 34 or not len(sg) or not np.isfinite(sg).all() or np.any(sg < 0) or np.any(sg[:,1] <= sg[:,0]) or np.any(sg[1:,0] != sg[:-1,1]):
        raise ValueError('Contiguous finite nonnegative 34-column schedule required')
    if np.any(sg[:,[2,3,4,5,6,8,9,10,11,12,22,24,25,26]] > 1):
        raise ValueError('Fraction drivers must be in [0,1]')
    if not isinstance(editing_provenance, str) or not editing_provenance.strip():
        raise ValueError('Explicit editing/source provenance required')
    if not np.isfinite([output_step_s,max_step_s,rtol,atol]).all() or min(output_step_s,max_step_s,rtol,atol) <= 0:
        raise ValueError('Positive solver settings required')
    er_parameters.validate(kw['basal_ca_uM'])
    if er_parameters.trpm_nS!=0:raise ValueError('TRPM current requires fast voltage mode; not supported in long fixed-voltage mode')
    if not isinstance(protein_input_provenance,str) or not protein_input_provenance.strip():raise ValueError('Protein-input provenance required')
    if iron_parameters.assumed_low_lysosome_GPX4_gain!=0:raise ValueError('Disable autonomous GPX4 gain for independent protein clamps')
    pv=pmca_parameters.vector()
    en=energy_parameters.vector()
    energy_sg=validate_schedule(energy_schedule,sg)
    ev=er_parameters.vector()
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
    base=np.r_[base,er_parameters.initial_amount,0.,0.,0.,0.,en[:3],np.zeros(10),memory_equilibrium(kw['basal_ca_uM'],pv)]
    iv = iron_parameters.vector(); nv = kw['organelle_parameters'].vector()
    av = kw['aggregation_parameters'].vector()
    hv = np.r_[kw['autoregulation_parameters'].vector(), kw['rna_feedback_parameters'].vector()]
    required = {'basal_ca_uM','calcium_tau_s','enzyme_parameters','tdp_parameters','localization_parameters','stmn2_parameters','inactivation_parameters','phosphorylation_parameters','reactivation_parameters','stability_parameters'}
    _, p, kv, sv, _ = _prior._prior._agg._baseline({k:kw[k] for k in required}, av)
    ag0 = base[37:45].sum(); base_n = base[6]+base[7]
    config = dict(initial['config'], topology='PMCA_memory_ATP_ER_134_v1',pmca_memory=asdict(pmca_parameters),energy=asdict(energy_parameters), ER=asdict(er_parameters), protein_input_provenance=protein_input_provenance, ncoa4=asdict(ncoa4_parameters), iron_release=asdict(iron_release_parameters), ferritin_label_initial='Assumed steady delivery weights 4:0.8:(4.8*tdigest/tferritin), normalized; no human measurement', editing=asdict(e),
                  mapping_family=mapping_family, editing_provenance=editing_provenance,
                  stimulus='Fixed voltage and explicitly supplied mean open fraction; no action potentials simulated')
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if checkpoint is None:
        y = base.copy(); origin = float(sg[0,0])
    else:
        if checkpoint['fingerprint'] != fingerprint or checkpoint['time_s'] != sg[0,0] or not np.array_equal(checkpoint['baseline_state'], base):
            raise ValueError('Checkpoint parameters, time or baseline differ')
        y = np.asarray(checkpoint['state'], float).copy(); origin = checkpoint['origin_time_s']
        if y.shape != (134,) or not np.isfinite(y).all() or min(y[np.setdiff1d(np.arange(134),[0,23,24,118,129])].min(),y[0]+kw['basal_ca_uM']) < -1e-7 or np.any(y[105:107] > 1.+1e-7) or y[133]>1.+1e-7:
            raise ValueError('Invalid checkpoint state')
    ts = [sg[0,:1].copy()]; zs = [y[None,:]]; nfev = 0
    for row, target, energy_row in zip(sg, targets, energy_sg):
        start, end = row[:2]
        near = start+np.geomspace(1e-4, min(end-start,3600.), 70)
        grid = np.unique(np.r_[np.arange(start+output_step_s,end,output_step_s), near[near<end], end])
        def depletion(t,z):return min(z[120:123])+1e-7
        depletion.terminal=True
        sol = solve_ivp(lambda t,z:rhs_memory(z,p,kv,sv,av,hv,nv,ag0,iv,base_n,row,ep,target,rv,qv,base[83],ev,e.voltage_mV,en,energy_row[2:],pv),
                        (start,end),y,t_eval=grid,method='LSODA',rtol=rtol,atol=atol,max_step=max_step_s,events=depletion)
        if sol.status==1:raise ArithmeticError('Negative adenylate pool: no-feedback counterfactual exhausted its ATP budget')
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
    checks = dict(PMCA2_memory_bounds_error=float(max(0.,-z[:,133].min(),z[:,133].max()-1.)),enzyme_balance=err(enzyme+z[:,15]-z[:,14]-enzyme0),
        TDP_lineage_balance=err(td+z[:,19]-z[:,18]-td0),
        aggregate_balance=err(ag-ag0-z[:,45]+z[:,46]+z[:,47]+z[:,49]),
        calcium_balance_uM=err((1+e.kappa)*(z[:,0]+z[:,23]-z[:,22])+z[:,115]-base[115]),
        ER_balance_uM=err(z[:,115]-base[115]-z[:,116]+z[:,117]+z[:,118]),
        adenylate_balance_uM=err(z[:,120:123].sum(1)-base[120:123].sum()),
        ATP_balance_uM=err(z[:,120]-base[120]-z[:,123]-z[:,124]+z[:,125:129].sum(1)-z[:,129]),
        high_energy_bond_balance_uM=err(2*(z[:,120]-base[120])+z[:,121]-base[121]-z[:,123]-z[:,124]+z[:,125:129].sum(1)),
        gross_clearance_balance_uM=err((1+e.kappa)*z[:,23]-z[:,130]-z[:,131]+z[:,132]),
        SERCA_ATP_balance_uM=err(z[:,127]-z[:,116]/2),
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
        minimum_state=float(min(z[:,np.setdiff1d(np.arange(134),[0,23,24,118,129])].min(),(z[:,0]+p[0]).min(),(z[:,1]-z[:,28]).min(),(z[:,2]-z[:,29]).min(),(z[:,31]-z[:,32]).min())),
        tracer_excess=float(max(0.,np.max(z[:,51:63]-z[:,_prior._prior._prior.LABEL_INDEX]))))
    if not np.isfinite(z).all() or checks['minimum_state'] < -1e-7 or max(v for k,v in checks.items() if k!='minimum_state') > 5e-5:
        raise ArithmeticError(str(checks))
    cp = dict(time_s=float(t[-1]),state=y.tolist(),baseline_state=base.tolist(),origin_time_s=origin,fingerprint=fingerprint)
    return dict(time_s=t,state=z,columns=ER_NETWORK_NAMES+ENERGY_NAMES+('PMCA2_memory_activation',),PMCA2_memory=z[:,133],energy_segments=energy_sg,ATP_uM=z[:,120],ADP_uM=z[:,121],AMP_uM=z[:,122],segments=sg,config=config,checks=checks,checkpoint=cp,
        baseline_state=base,nfev=nfev,
        er_fraction=z[:,115]/base[115],er_free_uM=z[:,115]/ev[1],
        serca_total_uM=z[:,116],er_release_total_uM=z[:,117],er_leak_net_uM=z[:,118],
        soce_total_uM=z[:,119],serca_ATP_equivalent_uM=z[:,116]/2,
        protein_input_segments=sg.copy(),
        GPX4_protein_relative=z[:,91]*sg[idx,29],FPN_protein_relative=sg[idx,30],calcium_uM=p[0]+z[:,0],effective_ca_flux_uM_s=obs[:,0],
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
            'Uniform cytosolic buffer and fixed receptor assembly, conductance and fractional calcium charge remain assumptions',
            'ER exchange conserves total calcium; constant differential ER buffer, volume, SERCA/release/SOCE are unknown hypotheses',
            'ER secretory cargo and ER calcium are distinct states; no unsupported cargo-to-pump mapping',
            'This long mode has fixed-voltage AMPA/SOCE. TRPM/SK and individual spikes are computed only in the separate native fast mode',
            'ATP/ADP/AMP explicit. Effective ADP-driven supply and ATP-sensitive forward pumps are hypotheses; no MgATP/Pi/pH/Na/mitochondrial membrane potential or thermodynamic pump reversal',
            'NCX gradient cost is an instantaneous NaK proxy, not direct NCX ATP hydrolysis; other demand excludes separately counted Ca costs',
            'Ca AUC remains signed state24; AUC=tau*clearance does not hold with energy feedback; no-feedback runs with negative ATP are rejected' ])

# One-state reduction affects only PMCA2 flux and the same ATP/Ca ledgers.
def rhs_memory(z,p,kv,sv,av,hv,nv,ag0,iv,base_n,row,ep,target,rv,qv,base_iron,ev,voltage,en,drive,pv):
 out=np.zeros(134)
 out[:133]=_rhs_energy(z[:133],p,kv,sv,av,hv,nv,ag0,iv,base_n,row,ep,target,rv,qv,base_iron,ev,voltage,en,drive)
 c=max(p[0]+z[0],0.);gamma=(1+ep[15])/p[1]
 pf=(z[120]/(en[4]+z[120])*(en[4]+en[0])/en[0]) if en[12]>0 else 1.
 delta=pmca2_flux(c,z[133],p[0],gamma,en[9],pf,pv)-out[130]
 out[0]-=delta/(1+ep[15]);out[23]+=delta/(1+ep[15])
 out[120]-=delta;out[121]+=delta;out[126]+=delta;out[130]+=delta
 on,off=memory_rates(c,pv);out[133]=on*(1-z[133])-off*z[133]
 return out
