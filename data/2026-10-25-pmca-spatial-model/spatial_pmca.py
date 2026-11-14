"""[codex] 2026-10-25. Two accessible volumes, conserved PMCA2 traffic.

Opt-in long fixed-voltage mode. The inherited biochemical network is somatic;
the process has Ca, adenylates and surface pumps, not a second TDP network.
All spatial/traffic parameters below are hypotheses, not a human fit.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import sys, json, hashlib
import numpy as np
from scipy.integrate import solve_ivp
B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parent/'2026-10-25-pmca-geometry-model'))
from surface_pmca import PMCASurfaceParameters, map_surface_pmca, IONS_PER_UM3_PER_UM
import long_pmca as core
from pmca_memory import PMCAMemoryParameters, memory_equilibrium, memory_rates, pmca2_flux
from energy_parameters import EnergyParameters, energy_fluxes, validate_schedule
compiled_memory = core.rhs_memory  # Its 133-state biochemical RHS is already JIT compiled.

@dataclass(frozen=True)
class SpatialGeometry:
    soma_volume_um3: float = 1000.
    soma_area_um2: float = 2500.
    process_length_um: float = 100.
    process_radius_um: float = .5
    process_accessible_fraction: float = .8
    connection_length_um: float = 10.
    connection_radius_um: float = .5
    free_ca_diffusion_um2_s: float = 30.
    adenylate_diffusion_um2_s: float = 0.
    process_kappa: float = 50.
    process_gamma_per_s: float = 5.1
    provenance: str = 'Scenario geometry and stationary buffers; not measured human MN'
    @property
    def process_volume_um3(self):
        return np.pi*self.process_radius_um**2*self.process_length_um*self.process_accessible_fraction
    @property
    def process_area_um2(self):
        return 2*np.pi*self.process_radius_um*self.process_length_um + np.pi*self.process_radius_um**2
    @property
    def connection_factor_um(self):
        return np.pi*self.connection_radius_um**2/self.connection_length_um
    def validate(self):
        d=asdict(self);v=[x for k,x in d.items() if k!='provenance']
        if not np.isfinite(v).all() or min(v)<0 or min(v[:7])<=0 or self.process_accessible_fraction>1 or self.process_gamma_per_s<=0 or not self.provenance.strip():
            raise ValueError('Positive finite geometry, nonnegative diffusion/buffer and provenance required')

@dataclass(frozen=True)
class PumpTrafficParameters:
    total_pumps: float = 25000.
    departure_soma_per_s: float = .1/3600
    departure_process_per_s: float = .5/3600
    forward_speed_um_s: float = .1
    reverse_speed_um_s: float = .1
    insertion_delay_s: float = 180.
    initial_process_fraction_if_static: float = .2
    ATP_molecules_per_delivery: float = 100.
    provenance: str = 'Closed pump pool; effective first-order transport, uncertain delivery energy'
    def validate(self):
        v=[x for k,x in asdict(self).items() if k!='provenance']
        if not np.isfinite(v).all() or min(v)<0 or min(self.forward_speed_um_s,self.reverse_speed_um_s)<=0 or not 0<=self.initial_process_fraction_if_static<=1 or not self.provenance.strip():
            raise ValueError('Invalid pump traffic parameters')
        if (self.departure_soma_per_s==0)!=(self.departure_process_per_s==0):
            raise ValueError('Use both zero departure rates for static pools; dynamic steady origin needs both positive')
    def initial(self, geometry):
        self.validate();geometry.validate()
        tf=geometry.process_length_um/self.forward_speed_um_s+self.insertion_delay_s
        tr=geometry.process_length_um/self.reverse_speed_um_s+self.insertion_delay_s
        ks,kp=self.departure_soma_per_s,self.departure_process_per_s
        if ks==0:
            f=self.initial_process_fraction_if_static
            return np.array([self.total_pumps*(1-f),0,self.total_pumps*f,0]),tf,tr
        r=ks/kp; ns=self.total_pumps/(1+r+ks*tf+r*kp*tr)
        return np.array([ns,ks*ns*tf,r*ns,r*kp*ns*tr]),tf,tr

SPATIAL_RANGES = dict(free_ca_diffusion_um2_s=[0.,30.,300.], adenylate_diffusion_um2_s=[0.,10.,100.],
    process_length_um=[30.,100.,1000.],process_radius_um=[.2,.5,1.],process_kappa=[10.,50.,200.],
    process_gamma_per_s=[.5,5.1,50.],total_pumps=[1000.,25000.,100000.],forward_speed_um_s=[.01,.1,1.],
    reverse_speed_um_s=[.01,.1,1.],departure_soma_per_hour=[.01,.1,1.],departure_process_per_hour=[.05,.5,5.],
    ATP_molecules_per_delivery=[0.,100.,1000.],forward_support=[0.,.25,1.],reverse_support=[0.,.25,1.],
    soma_volume_um3=[500.,1000.,2000.],soma_area_um2=[1000.,2500.,5000.],process_accessible_fraction=[.2,.5,.8],
    connection_length_um=[1.,10.,100.],connection_radius_um=[.1,.5,1.],insertion_delay_s=[0.,180.,1800.],
    initial_process_fraction_if_static=[0.,.2,.8],process_AMPA_fraction=[0.,.5,.9,1.],memory_gain=[0.,1.,3.])
EXTRA_NAMES = ('process_excess_Ca_uM','process_PMCA2_memory','process_ATP_uM','process_ADP_uM','process_AMP_uM',
    'PMCA2_soma_surface','PMCA2_forward_transit','PMCA2_process_surface','PMCA2_reverse_transit',
    'process_external_Ca_uM','process_PMCA_Ca_uM','process_NCX_Ca_uM','process_basal_Ca_uM','Ca_exchange_amount_uM_um3',
    'process_OXPHOS_ATP_uM','process_glycolysis_ATP_uM','process_other_ATP_uM','process_PMCA_ATP_uM',
    'process_NCX_ATP_equivalent_uM','process_AK_net_ATP_uM','transport_ATP_soma_uM','transport_ATP_process_uM',
    'ATP_exchange_amount_uM_um3','ADP_exchange_amount_uM_um3','AMP_exchange_amount_uM_um3',
    'PMCA2_forward_arrivals','PMCA2_reverse_arrivals','process_Ca_signed_AUC_uM_s')

def passive_exchange_rate(cs, cp, geometry):
    """uM*um3/s, positive soma -> process. D is FREE-ion D, not buffered D."""
    return geometry.free_ca_diffusion_um2_s*geometry.connection_factor_um*(cs-cp)

def traffic_rates(pools, traffic, tf, tr, forward_support, reverse_support, atp_s_factor=1., atp_p_factor=1.):
    ns,nf,np_,nr=np.maximum(pools,0)
    ds=traffic.departure_soma_per_s*ns;dp=traffic.departure_process_per_s*np_
    af=nf/tf*forward_support*atp_s_factor;ar=nr/tr*reverse_support*atp_p_factor
    return np.array([-ds+ar,ds-af,af-dp,dp-ar]),af,ar

def sat_norm(x,k,x0):
    return max(x,0)/(k+max(x,0))*(k+x0)/x0

def run_spatial_pmca_network(segments, *, geometry, traffic_parameters, pump_kinetics,
        spatial_schedule=None, memory_parameters, energy_parameters, checkpoint=None,
        output_step_s=300., max_step_s=300., rtol=1e-8, atol=1e-10, **kw):
    """Spatial schedule columns: start,end,AMPA fraction to process,forward,reverse.

    Same source channels/opening/current as the inherited model, with its
    absolute AMPA Ca input divided between compartments. No local Vm solved.
    pump_kinetics is PMCASurfaceParameters with density_per_um2=1 (unit marker);
    ACTUAL densities are calculated from traffic counts / membrane areas.
    Basal Ca influx and ADP-controlled supply are constructed ONCE at origin
    for matched rest, then held fixed through all stimulation/traffic changes.
    """
    geometry.validate();traffic_parameters.validate();pump_kinetics.validate()
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
    ss=np.c_[sg[:,:2],np.zeros(len(sg)),np.ones((len(sg),2))] if spatial_schedule is None else np.asarray(spatial_schedule,float)
    if ss.shape!=(len(sg),5) or not np.isfinite(ss).all() or not np.array_equal(ss[:,:2],sg[:,:2]) or np.any(ss[:,2:]<0) or np.any(ss[:,2:]>1):
        raise ValueError('Spatial rows must match source times; fraction/support in [0,1]')
    es=validate_schedule(kw.pop('energy_schedule',None),sg)
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
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    signed=[0,23,24,118,129,134,147,153,156,157,158,161]
    def minimum(z):
        return float(min(z[:,np.setdiff1d(np.arange(162),signed)].min(),(z[:,0]+c0).min(),(z[:,134]+c0).min(),
            (z[:,1]-z[:,28]).min(),(z[:,2]-z[:,29]).min(),(z[:,31]-z[:,32]).min()))
    if checkpoint is None:y=base.copy();origin=float(sg[0,0])
    else:
        if checkpoint['fingerprint']!=fingerprint or checkpoint['time_s']!=sg[0,0] or not np.array_equal(checkpoint['baseline_state'],base):
            raise ValueError('Checkpoint geometry/parameters/baseline/time changed')
        y=np.asarray(checkpoint['state'],float);origin=checkpoint['origin_time_s']
        if y.shape!=(162,) or not np.isfinite(y).all() or minimum(y[None,:]) < -1e-7 or np.any(y[[105,106,133,135]]>1+1e-7):raise ValueError('Invalid checkpoint state')

    def rhs(z,row,target,drive,srow):
        out=np.zeros(162);localpv=pv.copy();localpv[8]=unit_ratio*max(z[139],0)
        out[:134]=compiled_memory(z[:134],p,kv,sv,av,hv,nv,ag0,iv,bn,row,ep,target,rv,qv,base[83],ev,e.voltage_mV,en,drive,localpv)
        cs=max(c0+z[0],0);cp=max(c0+z[134],0)
        fs=sat_norm(z[120],en[4],en[0]) if en[12] else 1.
        fp=sat_norm(z[136],en[4],en[0]) if en[12] else 1.
        # Transport ATP sensitivity is independent of pump_feedback switch.
        rates,af,ar=traffic_rates(z[139:143],tr,tf,trev,srow[3],srow[4],sat_norm(z[120],en[4],en[0]),sat_norm(z[136],en[4],en[0]))
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
        ampa=core._flux(z[106],row[2],row[23],ep)[0]
        proc_input=srow[2]*ampa*(1+e.kappa)*vs/vp
        out[0]-=srow[2]*ampa;out[22]-=srow[2]*ampa;out[143]=proc_input
        f=passive_exchange_rate(cs,cp,g);out[147]=f
        out[0]-=f/(vs*(1+e.kappa))
        out[134]=(proc_input+pmp0+ncp0-pm-nc+f/vp)/(1+g.process_kappa)
        out[161]=z[134]
        fa=g.adenylate_diffusion_um2_s*g.connection_factor_um*(z[120:123]-z[136:139])
        out[120:123]-=fa/vs;out[136:139]+=fa/vp;out[156:159]=fa
        return out

    ts=[sg[0,:1]];zs=[y[None,:]];nfev=0
    for row,srow,erow in zip(sg,ss,es):
        start,end=row[:2];target=core.editing_target(row[22],e.normal_editing,kw['mapping_family'])
        near=start+np.geomspace(1e-4,min(end-start,3600),70)
        grid=np.unique(np.r_[np.arange(start+output_step_s,end,output_step_s),near[near<end],end])
        def depletion(t,z):return min(np.min(z[120:123]),np.min(z[136:139]))+1e-7
        depletion.terminal=True
        sol=solve_ivp(lambda t,z:rhs(z,row,target,erow[2:],srow),(start,end),y,t_eval=grid,method='LSODA',rtol=rtol,atol=atol,max_step=max_step_s,events=depletion)
        if sol.status==1:raise ArithmeticError('Adenylate depletion; nonphysical negative pool rejected')
        if not sol.success:raise RuntimeError(sol.message)
        y=sol.y[:,-1].copy();ts.append(sol.t);zs.append(sol.y.T);nfev+=sol.nfev
    t=np.concatenate(ts);z=np.concatenate(zs);err=lambda x:float(np.max(np.abs(x)))
    ag=z[:,37:45].sum(1);enzyme=z[:,1:6].sum(1)+z[:,31]+z[:,33];enzyme0=base[1:6].sum()+base[31]+base[33]
    ca_s=(1+e.kappa)*vs*(z[:,0]+z[:,23]-z[:,22])+vs*(z[:,115]-base[115])+z[:,147]
    ca_p=(1+g.process_kappa)*vp*(z[:,134]-base[134])-vp*(z[:,143]+z[:,146]-z[:,144]-z[:,145])-z[:,147]
    atps=z[:,120]-base[120]-z[:,123]-z[:,124]+z[:,125:129].sum(1)-z[:,129]+z[:,154]+z[:,156]/vs
    atpp=z[:,136]-base[136]-z[:,148]-z[:,149]+z[:,150:153].sum(1)-z[:,153]+z[:,155]-z[:,156]/vp
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
    if not np.isfinite(z).all() or checks['minimum_state'] < -1e-7 or max(v for k,v in checks.items() if k!='minimum_state')>5e-5:raise ArithmeticError(str(checks))
    idx=np.clip(np.searchsorted(sg[:,0],t,side='right')-1,0,len(sg)-1)
    cpdict=dict(time_s=float(t[-1]),state=y.tolist(),baseline_state=base.tolist(),origin_time_s=origin,fingerprint=fingerprint)
    return dict(time_s=t,state=z,columns=old['columns']+EXTRA_NAMES,baseline_state=base,checkpoint=cpdict,config=config,checks=checks,nfev=nfev,
        segments=sg,spatial_segments=ss,energy_segments=es,calcium_uM=c0+z[:,0],process_calcium_uM=c0+z[:,134],ATP_uM=z[:,120],process_ATP_uM=z[:,136],
        activity=(z[:,28]+z[:,29]+z[:,32])*(1-sg[idx,3]),aggregate_total=ag,nuclear_function_proxy=z[:,6]+z[:,7],
        pump_surface_soma=z[:,139],pump_surface_process=z[:,141],human_kinetic_fit=False,electrical_feedback=False,
        limitations=old['limitations']+['Two well-mixed volumes; connector is an effective resistance without its own stored mass.',
        'Free-Ca exchange assumes stationary buffers; mobile Ca-buffer transport and actual branched morphology absent.',
        'Closed pump pool, no new PMCA synthesis/degradation. Transit pumps cannot extrude Ca. Transit is exponential, not deterministic travel.',
        'Matched basal influx and ATP supply are constructed assumptions, not measured homeostasis or fitted rest.',
        'Only soma has the inherited ER/CAPN/TDP/STMN2/iron network. Process Ca does not directly cleave local TDP here.',
        'AMPA input is partitioned at fixed voltage. Separate native electrical controls are not continuous spatial spike coupling.',
        'PMCA delivery energy and ATP sensitivity are effective uncertain assumptions, not measured molecular motor stoichiometry.'])
