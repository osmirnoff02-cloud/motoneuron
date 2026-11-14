"""[codex] 2026-10-26. Derived NaK 239-state model (source retained).

Opt-in continuous two-compartment voltage and bulk-Ca model. Inherited
somatic biochemistry plus distinct local CAPN/substrate pools; no duplicate Ca.
All spatial/traffic parameters below are hypotheses, not a human fit.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import sys, json, hashlib
import numpy as np
from scipy.integrate import solve_ivp
B = Path(__file__).resolve().parent
sys.path.insert(0,str(B.parent/'2026-10-26-connected-spatial-neuron'))
from ion_parameters import IonParameters, ION_NAMES, prepare_ions, ion_extension
from mito_parameters import MitoParameters, MITO_NAMES, mito_extension
sys.path.insert(0, str(B.parent/'2026-10-25-pmca-geometry-model'))
from surface_pmca import PMCASurfaceParameters, map_surface_pmca, IONS_PER_UM3_PER_UM
import long_pmca as core
from pmca_memory import PMCAMemoryParameters, memory_equilibrium, memory_rates, pmca2_flux
from energy_parameters import EnergyParameters, energy_fluxes, validate_schedule
from connected_parameters import ConnectedParameters, local, gates, derived, membrane_rhs
compiled_memory = core.rhs_memory  # Its 133-state biochemical RHS is already JIT compiled.

from connected_parameters import spatial as spatial_source
EXTRA_NAMES=spatial_source.EXTRA_NAMES
passive_exchange_rate=spatial_source.passive_exchange_rate
traffic_rates=spatial_source.traffic_rates
sat_norm=spatial_source.sat_norm

from numba import njit
from compiled_kernels import memory_jit,pmca_flux_jit,rates_jit,sat_jit,traffic_jit
from connected_parameters import injury_rhs

@njit
def connected_rhs_kernel(z,row,target,drive,srow,injection,ctx):
    p,kv,sv,av,hv,nv,ag0,iv,bn,ep,rv,qv,base,ev,en,pv,unit_ratio,ppv,punit,delta_s,cost0s,delta_p,cost0p,pmp0,ncp0,tf,trev,vs,vp,sgamma,skappa,pgamma,pkappa,ca_diffusion,aden_diffusion,trvec,profile_p,profile_c,geomvec,couplevec,fbvec,c0,lev,liv,lkv,lsv,lpv,lsyn=ctx
    out=np.zeros(199);localpv=pv.copy();localpv[8]=unit_ratio*max(z[139],0)
    # Original biochemical AMPA flux is disabled: membrane current below owns it once.
    norow=row.copy();norow[2]=0.
    damage=min(1.,max(0.,1.-z[180]));support=1-fbvec[1]*damage
    out[:134]=memory_jit(z[:134],p,kv,sv,av,hv,nv,ag0,iv,bn,norow,ep,target,rv,qv,base[83],ev,z[185],en,drive,localpv)
    cs=max(c0+z[0],0);cp=max(c0+z[134],0)
    fs=sat_jit(z[120],en[4],en[0]) if en[12] else 1.
    fp=sat_jit(z[136],en[4],en[0]) if en[12] else 1.
    # Transport ATP sensitivity is independent of pump_feedback switch.
    rates,af,ar=traffic_jit(z[139:143],trvec,tf,trev,srow[3]*support,srow[4]*support,sat_jit(z[120],en[4],en[0]),sat_jit(z[136],en[4],en[0]))
    out[139:143]=rates;out[159:161]=[af,ar]
    costs=af*trvec[2]/(IONS_PER_UM3_PER_UM*vs)
    costp=ar*trvec[2]/(IONS_PER_UM3_PER_UM*vp)
    out[154:156]=[costs,costp]
    out[0]+=delta_s/(1+skappa);out[23]-=delta_s/(1+skappa);out[132]+=delta_s
    oxextra=(delta_s+cost0s)*(1-en[8])*drive[0]*sat_jit(z[121],en[6],en[1])
    glyextra=(delta_s+cost0s)*en[8]*drive[1]*sat_jit(z[121],en[7],en[1])
    out[120]+=oxextra+glyextra-costs;out[121]-=oxextra+glyextra-costs
    out[123]+=oxextra;out[124]+=glyextra
    pvec=ppv.copy();pvec[8]=punit*max(z[141],0)
    pm=pmca_flux_jit(cp,z[135],c0,pgamma,en[9],fp,pvec)
    nc=(1-en[9])*pgamma*cp*fp
    _,_,_,other,ox,gly,ak=energy_fluxes(z[136],z[137],z[138],cp,0.,pgamma,c0,0.,en,drive)
    ox+=(delta_p+cost0p)*(1-en[8])*drive[0]*sat_jit(z[137],en[6],en[1])
    gly+=(delta_p+cost0p)*en[8]*drive[1]*sat_jit(z[137],en[7],en[1])
    cost=other+pm+nc*en[10]+costp;prod=ox+gly
    out[136:139]=[prod-cost+ak,cost-prod-2*ak,ak]
    out[148:154]=[ox,gly,other,pm,nc*en[10],ak]
    on,off=rates_jit(cp,pvec);out[135]=on*(1-z[135])-off*z[135]
    out[144:147]=[pm,nc,pmp0+ncp0]
    cv=couplevec.copy();cv[6]=srow[2]
    dv,ca_current=membrane_rhs(z,profile_p,profile_c,geomvec,cv,fbvec,row[2],row[23],z[106],ep[3:8],ep,injection,c0)
    # SOCE was already included in the somatic Ca ledger by the ER source.
    # Return the SAME charge to voltage; do not add its Ca twice.
    dv[0]+=1000*out[119]*vs/5182.1348283/(geomvec[0]/100.)
    out[185:199]=dv
    soma_input=ca_current[0]*5182.1348283/vs
    proc_input=ca_current[1]*5182.1348283/vp
    out[0]+=soma_input/(1+skappa);out[22]+=soma_input/(1+skappa);out[143]=proc_input
    out[162:185]=injury_rhs(z[162:185],max(c0+z[134],0.),row[3],lev,liv,lkv,lsv,lpv,lsyn)
    f=ca_diffusion*(cs-cp);out[147]=f
    out[0]-=f/(vs*(1+skappa))
    out[134]=(proc_input+pmp0+ncp0-pm-nc+f/vp)/(1+pkappa)
    out[161]=z[134]
    fa=aden_diffusion*(z[120:123]-z[136:139])
    out[120:123]-=fa/vs;out[136:139]+=fa/vp;out[156:159]=fa
    return out


def run_mito_ion_neuron(segments, *, mito_parameters=None, uptake_support=1., ion_parameters=None, pump_support=1., geometry, traffic_parameters, pump_kinetics,
        profile,profile_p,profile_c,coupling,injury_parameters,connected_parameters=None,electrical_schedule=None,window_s=None,
        spatial_schedule=None, memory_parameters, energy_parameters, checkpoint=None,
        output_step_s=300., max_step_s=300., rtol=1e-8, atol=1e-10, **kw):
    """Spatial schedule columns: start,end,AMPA fraction to process,forward,reverse.

    Continuous local voltage determines AMPA/VGCC Ca current; shared bulk Ca
    drives SK and local CAPN. Somatic ER SOCE returns its charge to voltage.
    pump_kinetics is PMCASurfaceParameters with density_per_um2=1 (unit marker);
    ACTUAL densities are calculated from traffic counts / membrane areas.
    Basal Ca influx and ADP-controlled supply are constructed ONCE at origin
    for matched rest, then held fixed through all stimulation/traffic changes.
    """
    mp=mito_parameters or MitoParameters();mv=mp.vector()
    if not np.isfinite(uptake_support) or not 0<=uptake_support<=1:raise ValueError('Uptake support outside [0,1]')
    ip=ion_parameters or IonParameters();ip.validate()
    if not np.isfinite(pump_support) or not 0<=pump_support<=1:raise ValueError('Pump support in [0,1] required')
    geometry.validate();traffic_parameters.validate();pump_kinetics.validate()
    feedback_parameters=connected_parameters or ConnectedParameters();feedback_parameters.validate()
    if geometry.process_area_um2<=0 or geometry.axial_conductance_nS<=0:raise ValueError('Shared electrical geometry required')
    if not np.isclose(geometry.soma_area_um2+geometry.process_area_um2,profile_c[0]*100,rtol=1e-12):raise ValueError('Total membrane area must preserve frozen profile capacitance')
    if window_s is not None and (not np.isfinite(window_s) or window_s<=0):raise ValueError('Positive window_s required')
    if pump_kinetics.density_per_um2!=1:
        raise ValueError('Use density_per_um2=1 unit kinetics; actual counts come only from traffic')
    e=kw['editing_parameters'];c0=kw['basal_ca_uM'];g=geometry;tr=traffic_parameters
    if not np.isclose(e.volume_um3,g.soma_volume_um3,rtol=1e-12,atol=0):
        raise ValueError('Editing input volume must equal soma accessible volume')
    if not np.isfinite([output_step_s,max_step_s,rtol,atol]).all() or min(output_step_s,max_step_s,rtol,atol)<=0:
        raise ValueError('Positive solver settings required')
    sg=np.asarray(segments,float)
    if sg.ndim!=2 or sg.shape[1]!=34 or not len(sg) or not np.isfinite(sg).all() or np.any(sg<0) or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[1:,0]!=sg[:-1,1]):
        raise ValueError('Contiguous finite 34-column schedule required')
    if np.any(sg[:,[2,3,4,5,6,8,9,10,11,12,22,24,25,26]]>1):raise ValueError('Driver fractions outside [0,1]')
    ss=np.c_[sg[:,:2],np.full(len(sg),.9),np.ones((len(sg),2))] if spatial_schedule is None else np.asarray(spatial_schedule,float)
    if ss.shape!=(len(sg),5) or not np.isfinite(ss).all() or not np.array_equal(ss[:,:2],sg[:,:2]) or np.any(ss[:,2:]<0) or np.any(ss[:,2:]>1):
        raise ValueError('Spatial rows must match source times; fraction/support in [0,1]')
    es=validate_schedule(kw.pop('energy_schedule',None),sg)
    electrical=np.c_[sg[:,:2],np.zeros((len(sg),2))] if electrical_schedule is None else np.asarray(electrical_schedule,float)
    if electrical.shape!=(len(sg),4) or not np.isfinite(electrical).all() or not np.array_equal(electrical[:,:2],sg[:,:2]):raise ValueError('Electrical schedule must match segment boundaries')
    if window_s is not None:
        expanded=[];ee=[];spat=[];energy=[]
        for row,erow,srow,energyrow in zip(sg,electrical,ss,es):
            edges=np.r_[np.arange(row[0],row[1],window_s),row[1]]
            for a,b in zip(edges[:-1],edges[1:]):
                for target,src in ((expanded,row),(ee,erow),(spat,srow),(energy,energyrow)):
                    v=src.copy();v[:2]=[a,b];target.append(v)
        sg,ss,es,electrical=map(np.asarray,(expanded,spat,energy,ee))
    # The original runner performs its full parameter validation and supplies
    # the ORIGINAL baseline, not the endpoint of its tiny validation interval.
    ghost=sg[:1].copy();ghost[0,:2]=[0,1e-4];ghost[0,2]=0
    old=core.run_pmca_network(ghost,pmca_parameters=memory_parameters,energy_parameters=energy_parameters,**kw)
    en=energy_parameters.vector();ev=kw['er_parameters'].vector();rv=kw['iron_release_parameters'].vector();qv=kw['ncoa4_parameters'].vector()
    base=np.r_[old['baseline_state'],np.zeros(28)];base[135]=base[133];base[136:139]=en[:3]
    base[139:143],tf,trev=tr.initial(g)
    vs,vp=g.soma_volume_um3,g.process_volume_um3
    ep=np.array([e.tau_rna_h*3600,e.tau_surface_h*3600,e.incorporation_R_over_Q,*e.weights,
        e.g_reference_nS,e.reversal_mV-e.voltage_mV,e.cp_conductance_ratio,e.cp_unblocked_fraction,e.pf_ci,e.pf_cp,e.volume_um3,e.kappa])
    iv=kw['iron_parameters'].vector();nv=kw['organelle_parameters'].vector();av=kw['aggregation_parameters'].vector()
    hv=np.r_[kw['autoregulation_parameters'].vector(),kw['rna_feedback_parameters'].vector()]
    required={'basal_ca_uM','calcium_tau_s','enzyme_parameters','tdp_parameters','localization_parameters','stmn2_parameters','inactivation_parameters','phosphorylation_parameters','reactivation_parameters','stability_parameters'}
    _,p,kv,sv,_=core._prior._prior._agg._baseline({k:kw[k] for k in required},av)
    ag0=base[37:45].sum();bn=base[6]+base[7]
    from dataclasses import replace
    unit=replace(pump_kinetics,density_per_um2=1/g.soma_area_um2)
    mapped,_=map_surface_pmca(unit,area_um2=g.soma_area_um2,accessible_volume_um3=vs,basal_ca_uM=c0,
        gamma_per_s=e.gamma_per_s,memory_parameters=memory_parameters,energy_parameters=energy_parameters)
    pv=mapped.vector();unit_ratio=pv[8];pv[8]*=base[139]
    pp=replace(pump_kinetics,density_per_um2=1/g.process_area_um2)
    mappedp,_=map_surface_pmca(pp,area_um2=g.process_area_um2,accessible_volume_um3=vp,basal_ca_uM=c0,
        gamma_per_s=g.process_gamma_per_s,memory_parameters=memory_parameters,energy_parameters=energy_parameters)
    ppv=mappedp.vector();punit=ppv[8];ppv[8]*=base[141]
    pm0=pmca2_flux(c0,base[133],c0,e.gamma_per_s,en[9],1.,pv)
    delta_s=pm0-en[9]*e.gamma_per_s*c0
    pmp0=pmca2_flux(c0,base[135],c0,g.process_gamma_per_s,en[9],1.,ppv)
    ncp0=(1-en[9])*g.process_gamma_per_s*c0
    _,af0,ar0=traffic_rates(base[139:143],tr,tf,trev,1.,1.)
    cost0s=af0*tr.ATP_molecules_per_delivery/(IONS_PER_UM3_PER_UM*vs)
    cost0p=ar0*tr.ATP_molecules_per_delivery/(IONS_PER_UM3_PER_UM*vp)
    delta_p=pmp0-en[9]*g.process_gamma_per_s*c0
    config=dict(old['config'],topology='spatial_PMCA_162_v1',geometry=asdict(g),traffic=asdict(tr),pump_kinetics=asdict(pump_kinetics),
        initial_pump_pools=base[139:143].tolist(),initial_PMCA_soma_uM_s=float(pm0),initial_PMCA_process_uM_s=float(pmp0),
        matched_rest='Basal Ca input and supply set once from initial pumps, never reset after perturbation',
        process_biochemistry='Ca, ATP/ADP/AMP and pumps only; inherited CAPN/TDP/STMN2/ER/iron network is somatic')
    local_options=dict(injury_parameters)
    if local_options['basal_ca_uM']!=c0:raise ValueError('Local baseline must equal common calcium baseline')
    if len(local_options['substrates'])!=2 or local_options['substrates'][1].name!='betaII_spectrin' or local_options['substrates'][1].initial_intact_amount!=1.:raise ValueError('Explicit two local lineages and unit spectrin reference required')
    li=local.simulate_local_injury([0.,1e-4],[c0,c0],**local_options)
    lev=local_options['enzyme_parameters'].vector().copy();lev[7]*=local_options.get('enzyme_pool_scale',1.)
    liv=local_options['inactivation_parameters'].vector();lkv=local_options['reactivation_parameters'].vector();lsv=local_options['stability_parameters'].vector()
    lpv=np.array([q.vector() for q in local_options['substrates']]);la=li['baseline_activity']
    lsyn=np.array([(v[1]+v[0]*la)*q.initial_intact_amount for v,q in zip(lpv,local_options['substrates'])])
    gl,el,hold=derived(profile_p,profile_c);init=profile_c[5] if profile=='Burley' else profile_c[2]
    if profile!='Burley':hold=0.
    vg0=1/(1+np.exp(-(init-coupling['vg_half_mV'])/coupling['vg_slope_mV']))
    sk0=c0**coupling['sk_hill']/(coupling['sk_half_uM']**coupling['sk_hill']+c0**coupling['sk_hill'])
    electrical_base=np.tile(np.r_[init,gates(init,profile_p),vg0,sk0],2)
    base=np.r_[base,li['baseline_state'],electrical_base]
    geomvec=np.array([g.soma_area_um2,g.process_area_um2,g.axial_conductance_nS,init,hold,gl,el])
    couplevec=np.array([coupling[k] for k in ('vg_half_mV','vg_slope_mV','vg_tau_ms','sk_half_uM','sk_hill','sk_tau_ms')]+[.9])
    fbvec=np.array([feedback_parameters.spectrin_to_Nav_loss_gain,feedback_parameters.spectrin_to_delivery_loss_gain,feedback_parameters.g_vgcc_nS,feedback_parameters.g_sk_nS])
    config.update(topology='connected_spatial_199_v1',profile=profile,HH_parameters=profile_p.tolist(),HH_context=profile_c.tolist(),
        bulk_coupling=coupling,feedback=asdict(feedback_parameters),local_injury=li['config'],
        calcium_owner='One bulk cytosolic Ca pool in each compartment; somatic ER. No independent shell pool in this reduced driver.',
        geometry_owner='Shared native cylinder geometry; lateral membrane area, actual accessible volumes; series half-length diffusion resistance',
        electrical_solver='Continuous HH/biochemistry LSODA, native NEURON reference tested separately; no repeated short traces',
        process_biochemistry='Local CAPN and TDP/spectrin lineages plus Ca/ATP and conserved PMCA traffic',
        protein_synthesis_ATP_cost=None)
    base,ionvec,ioncv,ionbase=prepare_ions(base,profile_p,profile_c,geomvec,couplevec,fbvec,ep,c0,np.array([vs,vp]),np.array([(1-en[9])*e.gamma_per_s*c0,ncp0]),ip)
    if ip.enabled and min(pm0+ionbase[0,8],pmp0+ionbase[1,8]) < -1e-9:
        raise ValueError('Resting Ca balance would require unowned active Ca export; recalibrate PMCA/NCX before accepting this combination')
    config.update(topology='ion_energy_239_v1',ions=asdict(ip),ion_baseline=ionbase.tolist(),
      resting_ion_background='Fixed fluxes constructed once to preserve calibrated rest; not measured leaks. Explicit reservoir ledgers.',
      NCX='Reversible bounded mass-action flux with independent uncertain capacity; basal external Ca input matched once. No singular normalization at reversal.',
      ion_energy='NaK actual cycles consume ATP; prior indirect NCX ATP proxy removed; Na-entry/3 is a separate demand metric, never another ATP debit')
    base=np.r_[base,np.tile([mv[1]*(1+mv[2])*mv[3],0.,0.,0.],2)]
    config.update(topology='matrix_ion_energy_247_v1',mitochondria=asdict(mp),
      matrix_model='Finite effective buffered Ca store in each compartment; effective exit is not a resolved NCLX flux. No matrix Na, proton gradient, oxygen/substrate or MCU energy debit.',
      oxidative_modulation='Optional bounded matrix-Ca modifier of existing OXPHOS production, with matching ATP/ADP and production ledgers; not a new independent ATP source')
    # External pump_support may change on restart, like applied current; biological capacities may not.
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    signed=[0,22,23,24,118,129,131,134,143,145,147,153,156,157,158,161]
    def minimum(z):
        return float(min(z[:,np.setdiff1d(np.arange(162),signed)].min(),(z[:,0]+c0).min(),(z[:,134]+c0).min(),
            (z[:,1]-z[:,28]).min(),(z[:,2]-z[:,29]).min(),(z[:,31]-z[:,32]).min(),z[:,162:185].min()))
    if checkpoint is None:y=base.copy();origin=float(sg[0,0])
    else:
        if checkpoint['fingerprint']!=fingerprint or checkpoint['time_s']!=sg[0,0] or not np.array_equal(checkpoint['baseline_state'],base):
            raise ValueError('Checkpoint geometry/parameters/baseline/time changed')
        y=np.asarray(checkpoint['state'],float);origin=checkpoint['origin_time_s']
        if y.shape!=(247,) or not np.isfinite(y).all() or minimum(y[None,:]) < -1e-7 or np.any(y[[105,106,133,135]]>1+1e-7):raise ValueError('Invalid checkpoint state')

    def rhs_reference(z,row,target,drive,srow,injection):
        out=np.zeros(199);localpv=pv.copy();localpv[8]=unit_ratio*max(z[139],0)
        # Original biochemical AMPA flux is disabled: membrane current below owns it once.
        norow=row.copy();norow[2]=0.
        damage=min(1.,max(0.,1.-z[180]));support=1-feedback_parameters.spectrin_to_delivery_loss_gain*damage
        out[:134]=compiled_memory(z[:134],p,kv,sv,av,hv,nv,ag0,iv,bn,norow,ep,target,rv,qv,base[83],ev,z[185],en,drive,localpv)
        cs=max(c0+z[0],0);cp=max(c0+z[134],0)
        fs=sat_norm(z[120],en[4],en[0]) if en[12] else 1.
        fp=sat_norm(z[136],en[4],en[0]) if en[12] else 1.
        # Transport ATP sensitivity is independent of pump_feedback switch.
        rates,af,ar=traffic_rates(z[139:143],tr,tf,trev,srow[3]*support,srow[4]*support,sat_norm(z[120],en[4],en[0]),sat_norm(z[136],en[4],en[0]))
        out[139:143]=rates;out[159:161]=[af,ar]
        costs=af*tr.ATP_molecules_per_delivery/(IONS_PER_UM3_PER_UM*vs)
        costp=ar*tr.ATP_molecules_per_delivery/(IONS_PER_UM3_PER_UM*vp)
        out[154:156]=[costs,costp]
        out[0]+=delta_s/(1+e.kappa);out[23]-=delta_s/(1+e.kappa);out[132]+=delta_s
        oxextra=(delta_s+cost0s)*(1-en[8])*drive[0]*sat_norm(z[121],en[6],en[1])
        glyextra=(delta_s+cost0s)*en[8]*drive[1]*sat_norm(z[121],en[7],en[1])
        out[120]+=oxextra+glyextra-costs;out[121]-=oxextra+glyextra-costs
        out[123]+=oxextra;out[124]+=glyextra
        pvec=ppv.copy();pvec[8]=punit*max(z[141],0)
        pm=pmca2_flux(cp,z[135],c0,g.process_gamma_per_s,en[9],fp,pvec)
        nc=(1-en[9])*g.process_gamma_per_s*cp*fp
        _,_,_,other,ox,gly,ak=energy_fluxes(z[136],z[137],z[138],cp,0.,g.process_gamma_per_s,c0,0.,en,drive)
        ox+=(delta_p+cost0p)*(1-en[8])*drive[0]*sat_norm(z[137],en[6],en[1])
        gly+=(delta_p+cost0p)*en[8]*drive[1]*sat_norm(z[137],en[7],en[1])
        cost=other+pm+nc*en[10]+costp;prod=ox+gly
        out[136:139]=[prod-cost+ak,cost-prod-2*ak,ak]
        out[148:154]=[ox,gly,other,pm,nc*en[10],ak]
        on,off=memory_rates(cp,pvec);out[135]=on*(1-z[135])-off*z[135]
        out[144:147]=[pm,nc,pmp0+ncp0]
        cv=couplevec.copy();cv[6]=srow[2]
        dv,ca_current=membrane_rhs(z,profile_p,profile_c,geomvec,cv,fbvec,row[2],row[23],z[106],ep[3:8],ep,injection,c0)
        # SOCE was already included in the somatic Ca ledger by the ER source.
        # Return the SAME charge to voltage; do not add its Ca twice.
        dv[0]+=1000*out[119]*vs/5182.1348283/(g.soma_area_um2/100.)
        out[185:199]=dv
        soma_input=ca_current[0]*5182.1348283/vs
        proc_input=ca_current[1]*5182.1348283/vp
        out[0]+=soma_input/(1+e.kappa);out[22]+=soma_input/(1+e.kappa);out[143]=proc_input
        out[162:185]=local._rhs(z[162:185],max(c0+z[134],0.),row[3],lev,liv,lkv,lsv,lpv,lsyn)
        f=passive_exchange_rate(cs,cp,g);out[147]=f
        out[0]-=f/(vs*(1+e.kappa))
        out[134]=(proc_input+pmp0+ncp0-pm-nc+f/vp)/(1+g.process_kappa)
        out[161]=z[134]
        fa=g.adenylate_diffusion_um2_s*g.connection_factor_um*(z[120:123]-z[136:139])
        out[120:123]-=fa/vs;out[136:139]+=fa/vp;out[156:159]=fa
        return out

    sgamma=e.gamma_per_s;skappa=e.kappa;pgamma=g.process_gamma_per_s;pkappa=g.process_kappa
    ca_diffusion=g.free_ca_diffusion_um2_s*g.connection_factor_um;aden_diffusion=g.adenylate_diffusion_um2_s*g.connection_factor_um
    trvec=np.array([tr.departure_soma_per_s,tr.departure_process_per_s,tr.ATP_molecules_per_delivery])
    ctx=(p,kv,sv,av,hv,nv,ag0,iv,bn,ep,rv,qv,base,ev,en,pv,unit_ratio,ppv,punit,delta_s,cost0s,delta_p,cost0p,pmp0,ncp0,tf,trev,vs,vp,sgamma,skappa,pgamma,pkappa,ca_diffusion,aden_diffusion,trvec,profile_p,profile_c,geomvec,couplevec,fbvec,c0,lev,liv,lkv,lsv,lpv,lsyn)
    # One explicit equivalence check before integration protects scalar rewrites.
    checkrow=sg[0].copy();checkrow[2]=.003
    probe=base.copy();probe[0]+=.1;probe[134]+=.2;probe[180]*=.8;probe[185]+=5.;probe[192]-=3.
    target=core.editing_target(checkrow[22],e.normal_editing,kw['mapping_family'])
    reference=rhs_reference(probe,checkrow,target,es[0,2:],ss[0],electrical[0,2:])
    fast=connected_rhs_kernel(probe,checkrow,target,es[0,2:],ss[0],electrical[0,2:],ctx)
    kernel_error=float(np.max(abs(reference-fast)/(1+abs(reference))))
    if kernel_error>1e-11:raise ArithmeticError('Compiled/reference RHS mismatch: '+str(kernel_error))
    def rhs(z,row,target,drive,srow,injection):
        old=connected_rhs_kernel(z,row,target,drive,srow,injection,ctx)
        ions=ion_extension(z,old,row,drive,srow,injection,ctx,ionvec,ionbase,ioncv,pump_support,g.connection_factor_um)
        return mito_extension(z,ions,ctx,mv,uptake_support)
    ie=ion_extension(probe,fast,checkrow,es[0,2:],ss[0],electrical[0,2:],ctx,ionvec,ionbase,ioncv,pump_support,g.connection_factor_um)
    ir=ion_extension.py_func(probe,reference,checkrow,es[0,2:],ss[0],electrical[0,2:],ctx,ionvec,ionbase,ioncv,pump_support,g.connection_factor_um)
    ion_kernel_error=float(np.max(abs(ie-ir)/(1+abs(ir))))
    if ion_kernel_error>1e-11:raise ArithmeticError('Ion compiled/reference mismatch')

    mi=mito_extension(probe,ie,ctx,mv,uptake_support)
    mr=mito_extension.py_func(probe,ir,ctx,mv,uptake_support)
    mito_error=float(np.max(abs(mi-mr)/(1+abs(mr))))
    if mito_error>1e-11:raise ArithmeticError('Mito compiled/reference mismatch')
    ts=[sg[0,:1]];zs=[y[None,:]];nfev=0;events_s=[];events_p=[]
    for row,srow,erow,elecrow in zip(sg,ss,es,electrical):
        start,end=row[:2];target=core.editing_target(row[22],e.normal_editing,kw['mapping_family'])
        near=start+np.geomspace(1e-4,min(end-start,3600),70)
        grid=np.unique(np.r_[np.arange(start+output_step_s,end,output_step_s),near[near<end],end])
        grid=grid[(grid>start)&(grid<=end)]
        def depletion(t,z):return min(np.min(z[120:123]),np.min(z[136:139]),np.min(z[199:203]),np.min(z[219:223]),z[239],z[243])+1e-7
        depletion.terminal=True
        def spike_s(t,z):return z[185]-(0. if profile=='Burley' else -10.)
        def spike_p(t,z):return z[192]-(0. if profile=='Burley' else -10.)
        spike_s.direction=1;spike_p.direction=1
        sol=solve_ivp(lambda t,z:rhs(z,row,target,erow[2:],srow,elecrow[2:]),(start,end),y,t_eval=grid,method='LSODA',rtol=rtol,atol=atol,max_step=max_step_s,events=[depletion,spike_s,spike_p])
        events_s.extend(sol.t_events[1]);events_p.extend(sol.t_events[2])
        if sol.status==1:raise ArithmeticError('Adenylate depletion; nonphysical negative pool rejected')
        if not sol.success:raise RuntimeError(sol.message)
        y=sol.y[:,-1].copy();ts.append(sol.t);zs.append(sol.y.T);nfev+=sol.nfev
    t=np.concatenate(ts);z=np.concatenate(zs);err=lambda x:float(np.max(np.abs(x)))
    ag=z[:,37:45].sum(1);enzyme=z[:,1:6].sum(1)+z[:,31]+z[:,33];enzyme0=base[1:6].sum()+base[31]+base[33]
    ca_s=(1+e.kappa)*vs*(z[:,0]+z[:,23]-z[:,22])+vs*(z[:,115]-base[115])+z[:,147]+vs*(z[:,239]-base[239])
    ca_p=(1+g.process_kappa)*vp*(z[:,134]-base[134])-vp*(z[:,143]+z[:,146]-z[:,144]-z[:,145])-z[:,147]+vp*(z[:,243]-base[243])
    atps=z[:,120]-base[120]-z[:,123]-z[:,124]+z[:,125:129].sum(1)-z[:,129]+z[:,154]+z[:,156]/vs+z[:,203]
    atpp=z[:,136]-base[136]-z[:,148]-z[:,149]+z[:,150:153].sum(1)-z[:,153]+z[:,155]-z[:,156]/vp+z[:,223]
    adens=z[:,120:123].sum(1)-base[120:123].sum()+z[:,156:159].sum(1)/vs
    adenp=z[:,136:139].sum(1)-base[136:139].sum()-z[:,156:159].sum(1)/vp
    checks=dict(soma_Ca_balance_uM=err(ca_s/vs),process_Ca_balance_uM=err(ca_p/vp),global_Ca_balance_uM=err((ca_s+ca_p)/(vs+vp)),
        soma_ATP_balance_uM=err(atps),process_ATP_balance_uM=err(atpp),global_ATP_balance_uM=err((vs*atps+vp*atpp)/(vs+vp)),
        soma_adenylate_balance_uM=err(adens),process_adenylate_balance_uM=err(adenp),pump_count_relative_error=err(z[:,139:143].sum(1)-tr.total_pumps)/max(1,tr.total_pumps),
        enzyme_balance=err(enzyme+z[:,15]-z[:,14]-enzyme0),TDP_lineage_balance=err(z[:,6:10].sum(1)+ag+z[:,19]-z[:,18]-base[6:10].sum()-ag0),
        aggregate_balance=err(ag-ag0-z[:,45]+z[:,46]+z[:,47]+z[:,49]),ER_balance_uM=err(z[:,115]-base[115]-z[:,116]+z[:,117]+z[:,118]),
        soma_PMCA_ATP=err(z[:,126]-z[:,130]),process_PMCA_ATP=err(z[:,151]-z[:,144]),SERCA_ATP=err(z[:,127]-z[:,116]/2),
        gross_soma_clearance=err((1+e.kappa)*z[:,23]-z[:,130]-z[:,131]+z[:,132]),
        iron_mass_balance=err(z[:,core.FE_INDEX].sum(1)+z[:,97]-z[:,96]-base[core.FE_INDEX].sum()),
        memory_editing_bounds=float(max(0,-z[:,[105,106,133,135]].min(),z[:,[105,106,133,135]].max()-1)),minimum_state=minimum(z))
    checks['compiled_rhs_scaled_error']=kernel_error
    checks['local_enzyme_lineage']=err(z[:,162:172].sum(1)+z[:,173]-z[:,172]-base[162:172].sum())
    for j,name in enumerate(('process_TDP','process_spectrin')):
        k=175+5*j;checks[name+'_lineage']=err(z[:,k]+z[:,k+1]+z[:,k+2]-z[:,k+3]-base[k]-base[k+1])
    gi=[186,187,188,189,190,191,193,194,195,196,197,198]
    checks['electrical_gate_bounds']=float(max(0.,-z[:,gi].min(),z[:,gi].max()-1.))
    checks['ion_compiled_rhs_error']=ion_kernel_error
    checks['mito_compiled_rhs_error']=mito_error
    for j,side in enumerate(('soma','process')):
        q=239+4*j
        checks[side+'_matrix_Ca_balance_uM']=err(z[:,q]-base[q]-z[:,q+1]+z[:,q+2])
        if np.min(z[:,q]) < -1e-8 or np.max(z[:,q]) > mv[1]*(1+mv[2])*mv[4]+1e-7:raise ArithmeticError('Matrix Ca bounds')
    for j,label in enumerate(('soma','process')):
        b=199+20*j;vol=vs if j==0 else vp
        checks[label+'_Na_balance_uM']=err(z[:,b]-base[b]+z[:,b+5]+3*z[:,b+4]-z[:,b+7]-z[:,b+8]-z[:,b+12]/vol)
        checks[label+'_K_balance_uM']=err(z[:,b+1]-base[b+1]+z[:,b+6]-2*z[:,b+4]-z[:,b+9]-z[:,b+13]/vol)
        checks[label+'_Na_cell_bath_uM']=err(z[:,b]-base[b]+ionvec[10]*(z[:,b+2]-base[b+2])-z[:,b+10]-z[:,b+12]/vol)
        checks[label+'_K_cell_bath_uM']=err(z[:,b+1]-base[b+1]+ionvec[10]*(z[:,b+3]-base[b+3])-z[:,b+11]-z[:,b+13]/vol)
        checks[label+'_NaK_charge_ATP_uM']=err(z[:,b+14]*10364.26965661773/vol-z[:,b+4])
        if np.min(z[:,b:b+4])<=0:raise ArithmeticError('Nonpositive ionic concentration')
    checks['Na_axial_amount']=err(z[:,211]+z[:,231])
    checks['K_axial_amount']=err(z[:,212]+z[:,232])
    if not np.isfinite(z).all() or checks['minimum_state'] < -1e-7 or max(v for k,v in checks.items() if k!='minimum_state')>5e-5:raise ArithmeticError(str(checks))
    idx=np.clip(np.searchsorted(sg[:,0],t,side='right')-1,0,len(sg)-1)
    cpdict=dict(time_s=float(t[-1]),state=y.tolist(),baseline_state=base.tolist(),origin_time_s=origin,fingerprint=fingerprint)
    return dict(time_s=t,state=z,columns=old['columns']+EXTRA_NAMES+tuple('process_'+q for q in li['columns'])+tuple(k+'_'+q for k in ('soma','process') for q in ('voltage_mV','m','h','n','w','VG_gate','SK_gate'))+ION_NAMES+MITO_NAMES,
        matrix_calcium_uM=z[:,[239,243]]/(mv[1]*(1+mv[2])),
        sodium_in_uM=z[:,[199,219]],potassium_in_uM=z[:,[200,220]],sodium_out_uM=z[:,[201,221]],potassium_out_uM=z[:,[202,222]],NaK_ATP_uM=z[:,[203,223]],
        soma_voltage_mV=z[:,185],process_voltage_mV=z[:,192],spike_times_s=dict(soma=events_s,process=events_p) if profile=='Burley' else None,
        voltage_crossing_times_s=dict(soma=events_s,process=events_p),
        event_definition='Upward Vm crossing: Burley0mV, Selvaraj-10mV. Selvaraj spike_times_s=None because crossings are NOT its source peak/dV/dt detector; use the original source detector for frequency validation.',
        process_spectrin_intact_fraction=z[:,180],process_TDP_intact=z[:,175],process_TDP_fragment=z[:,176],
        process_Nav_remaining=1-feedback_parameters.spectrin_to_Nav_loss_gain*np.clip(1-z[:,180],0,1),
        delivery_remaining=1-feedback_parameters.spectrin_to_delivery_loss_gain*np.clip(1-z[:,180],0,1),
        electrical_segments=electrical,window_s=window_s,baseline_state=base,checkpoint=cpdict,config=config,checks=checks,nfev=nfev,
        segments=sg,spatial_segments=ss,energy_segments=es,calcium_uM=c0+z[:,0],process_calcium_uM=c0+z[:,134],ATP_uM=z[:,120],process_ATP_uM=z[:,136],
        activity=(z[:,28]+z[:,29]+z[:,32])*(1-sg[idx,3]),aggregate_total=ag,nuclear_function_proxy=z[:,6]+z[:,7],
        pump_surface_soma=z[:,139],pump_surface_process=z[:,141],human_kinetic_fit=False,electrical_feedback=True,
        inherited_source_limitations=old['limitations'],limitations=[
        'Reduced bulk-Ca coupling is distinct from native shell-Ca mode; no claim of matching submembrane microdomains.',
        'Dynamic Nernst Na/K, finite perfused bath and NaK stoichiometry implemented. Resting background fluxes and pump kinetics are assumptions; residual leak/holding charge has no full Cl/osmotic model.',
        'Somatic ER and two finite effective matrix Ca stores; no process ER, matrix Na, NCLX stoichiometry, membrane potential, oxygen/substrate or MCU energetic cost. Not a complete mitochondrion.',
        'Spectrin-to-Nav and delivery gains are bounded hypotheses, not observed human transfer functions.',
        'TDP local lineage does not automatically impair nuclear TDP or STMN2. Protein turnover ATP cost is uncalibrated and omitted explicitly.',
        'Continuous stiff solver, not an accelerated multiday simulation. No independent human validation.',
        'Source Ca/energy balances cover represented pumps/traffic, not a whole-cell energy budget.',
        'Closed four-pool PMCA traffic. Exponential delivery, no synthesis/degradation. Actual geometry unmeasured.',
        ],source_spatial_limitations=['Two well-mixed volumes; connector is an effective resistance without its own stored mass.',
        'Free-Ca exchange assumes stationary buffers; mobile Ca-buffer transport and actual branched morphology absent.',
        'Closed pump pool, no new PMCA synthesis/degradation. Transit pumps cannot extrude Ca. Transit is exponential, not deterministic travel.',
        'Matched basal influx and ATP supply are constructed assumptions, not measured homeostasis or fitted rest.',
        'Only soma has the inherited ER/CAPN/TDP/STMN2/iron network. Process Ca does not directly cleave local TDP here.',
        'AMPA input is partitioned at fixed voltage. Separate native electrical controls are not continuous spatial spike coupling.',
        'PMCA delivery energy and ATP sensitivity are effective uncertain assumptions, not measured molecular motor stoichiometry.'])
