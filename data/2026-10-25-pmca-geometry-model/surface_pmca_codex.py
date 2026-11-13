"""[codex] 2026-10-25. Explicit PMCA2 surface capacity and accessible volume.

Opt-in adapter to the existing mass-conserving PMCA/Ca/ATP backends. No
electrical profile or old simulation is refitted. Parameters are hypotheses.
"""
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import sys,json,hashlib
import numpy as np
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B.parent/'2026-10-25-pmca-memory-model'))
from pmca_memory_codex import PMCAMemoryParameters,memory_equilibrium
sys.path.insert(0,str(B.parent/'2026-10-24-atp-calcium-feedback'))
from energy_parameters_codex import EnergyParameters
IONS_PER_UM3_PER_UM=602.214076

@dataclass(frozen=True)
class PMCASurfaceParameters:
    density_per_um2: float
    active_fraction: float
    cycles_per_s: float
    calcium_half_uM: float
    calcium_hill: float
    provenance: str
    def validate(self):
        v=[self.density_per_um2,self.active_fraction,self.cycles_per_s,self.calcium_half_uM,self.calcium_hill]
        if not np.isfinite(v).all() or min(v)<0 or min(v[2:])<=0 or not 0<=self.active_fraction<=1 or not self.provenance.strip():
            raise ValueError('Explicit finite surface parameters/provenance required')

PMCA_SURFACE_RANGES=dict(density_per_um2=[.1,1.,10.,20.],active_fraction=[.01,.1,1.],
    cycles_per_s=[100.,1000.,18000.],calcium_half_uM=[.1,.3,1.],calcium_hill=[1.,2.,3.],
    accessible_volume_um3=[500.,1000.,2000.],area_um2=[1000.,2500.,5000.])

def hill(c,k,n):
    c=np.maximum(c,0.);return c**n/(c**n+k**n)

def surface_capacity(surface,*,area_um2,accessible_volume_um3):
    surface.validate()
    if not np.isfinite([area_um2,accessible_volume_um3]).all() or min(area_um2,accessible_volume_um3)<=0:
        raise ValueError('Positive membrane area and accessible cytosolic volume required')
    count=surface.density_per_um2*area_um2
    ions=count*surface.active_fraction*surface.cycles_per_s
    return dict(total_pumps=count,active_pumps=count*surface.active_fraction,maximum_Ca_ions_s=ions,
        maximum_Ca_uM_s=ions/(IONS_PER_UM3_PER_UM*accessible_volume_um3),
        maximum_ATP_uM_s=ions/(IONS_PER_UM3_PER_UM*accessible_volume_um3),
        area_um2=float(area_um2),accessible_volume_um3=float(accessible_volume_um3))

def surface_pmca_flux(c,activation,atp_uM,*,surface,capacity,memory,energy):
    """Absolute PMCA2 flux: bounded by active pumps * cycles, 1 Ca per ATP.

    Calcium, ATP and memory occupancies multiply this capacity. Membrane
    support is inherited. With pump_feedback=False ATP is clamped to ATP0.
    """
    a=np.asarray(activation)
    if not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):raise ValueError('Activation outside [0,1]')
    atp=np.maximum(atp_uM,0.) if energy.pump_feedback else energy.ATP0_uM
    atp_fraction=atp/(energy.pump_half_ATP_uM+atp)
    return (capacity['maximum_Ca_uM_s']*hill(c,surface.calcium_half_uM,surface.calcium_hill)
            *(1+memory.memory_gain*a)/(1+memory.memory_gain)*memory.membrane_support*atp_fraction)

def map_surface_pmca(surface,*,area_um2,accessible_volume_um3,basal_ca_uM,gamma_per_s,
                     memory_parameters,energy_parameters):
    """Exact algebraic mapping into existing PMCA2 replacement, not an extra pump.

    The old 0.43-uM clearance-shape default is not used. K and Hill must be
    explicit. Bautista Fig9 fits FREE-Ca decay; Fig11 fits buffer-corrected flux.
    A changed pump capacity need not preserve the old basal steady state.
    """
    mp=memory_parameters;en=energy_parameters
    mp.vector();en.validate();cap=surface_capacity(surface,area_um2=area_um2,accessible_volume_um3=accessible_volume_um3)
    if not np.isfinite([basal_ca_uM,gamma_per_s]).all() or min(basal_ca_uM,gamma_per_s,en.pmca_fraction,mp.pmca2_share)<=0:
        raise ValueError('Positive reference PMCA2 share, gamma and basal Ca required')
    if mp.pmca2_ratio!=1. or mp.saturation_weight!=0.:
        raise ValueError('Surface capacity replaces ratio/shape: do not also apply RNA ratio or another shape')
    a0=memory_equilibrium(basal_ca_uM,mp.vector())
    f0=en.ATP0_uM/(en.pump_half_ATP_uM+en.ATP0_uM)
    ratio=(cap['maximum_Ca_uM_s']*hill(basal_ca_uM,surface.calcium_half_uM,surface.calcium_hill)
           *(1+mp.memory_gain*a0)/(1+mp.memory_gain)*f0
           /(en.pmca_fraction*gamma_per_s*mp.pmca2_share*basal_ca_uM))
    mapped=replace(mp,pmca2_ratio=float(ratio),saturation_weight=1.,saturation_half_uM=surface.calcium_half_uM,
        saturation_hill=surface.calcium_hill,provenance=mp.provenance+'; explicit surface capacity: '+surface.provenance)
    meta=dict(surface=asdict(surface),capacity=cap,baseline_reference=dict(basal_ca_uM=basal_ca_uM,gamma_per_s=gamma_per_s),
        mapped_parameters=asdict(mapped),human_kinetic_fit=False,
        limitations=['Density, active fraction, turnover, accessible volume and half-saturation are not independently calibrated in human MN.',
                    'One effective accessible volume; no spatial pump clustering or 3D diffusion.',
                    'Only the PMCA2 share is replaced. Other PMCA/NCX/SERCA are inherited.',
                    'Basal influx and ATP production are inherited, not retuned to force an equilibrium after capacity changes.',
                    'Density times active fraction times turnover is degenerate; equal products give equal flux.',
                    'PMCA2 moves one Ca per ATP. Other ion gradients and thermodynamic reversal are not resolved.'])
    meta['fingerprint']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    return mapped,meta

def run_surface_pmca_probe(*,surface_parameters=None,memory_parameters=None,energy_parameters=None,**kwargs):
    from electrical_pmca_codex import run_pmca_coupled,CouplingParameters,refined_profile
    mp=memory_parameters or PMCAMemoryParameters();en=energy_parameters or EnergyParameters()
    meta=None
    if surface_parameters is not None:
        par=kwargs.get('parameters') or CouplingParameters()
        _,c=refined_profile(kwargs.get('profile','Burley'))
        area=float(c[0]*100.);diam=np.sqrt(area/np.pi);volume=np.pi*diam**3/4*par.cytosol_fraction
        mp,meta=map_surface_pmca(surface_parameters,area_um2=area,accessible_volume_um3=volume,
            basal_ca_uM=par.basal_ca_uM,gamma_per_s=par.gamma_per_s,memory_parameters=mp,energy_parameters=en)
    r=run_pmca_coupled(pmca_parameters=mp,energy_parameters=en,**kwargs)
    r['surface_PMCA2']=meta
    if meta:
        for key,act in [('cs','mems'),('cb','memb')]:
            r['PMCA2_'+key+'_uM_s']=surface_pmca_flux(r[key],r[act],r['atp'],surface=surface_parameters,
                capacity=meta['capacity'],memory=mp,energy=en)
        sf=r['parameters']['shell_fraction']
        r['PMCA2_total_uM_s']=sf*r['PMCA2_cs_uM_s']+(1-sf)*r['PMCA2_cb_uM_s']
    return r

def run_surface_pmca_network(segments,*,surface_parameters,area_um2,memory_parameters,energy_parameters,**kwargs):
    from long_pmca_codex import run_pmca_network
    e=kwargs['editing_parameters']
    mapped,meta=map_surface_pmca(surface_parameters,area_um2=area_um2,accessible_volume_um3=e.volume_um3,
        basal_ca_uM=e.basal_ca_uM,gamma_per_s=e.gamma_per_s,memory_parameters=memory_parameters,energy_parameters=energy_parameters)
    cp=kwargs.get('checkpoint')
    if cp is not None and cp.get('surface_fingerprint')!=meta['fingerprint']:
        raise ValueError('Physical surface/volume configuration changed across checkpoint')
    r=run_pmca_network(segments,pmca_parameters=mapped,energy_parameters=energy_parameters,**kwargs)
    r['surface_PMCA2']=meta;r['checkpoint']['surface_fingerprint']=meta['fingerprint']
    r['PMCA2_uM_s']=surface_pmca_flux(r['calcium_uM'],r['PMCA2_memory'],r['ATP_uM'],surface=surface_parameters,
        capacity=meta['capacity'],memory=mapped,energy=energy_parameters)
    r['PMCA2_ions_s']=r['PMCA2_uM_s']*IONS_PER_UM3_PER_UM*e.volume_um3
    return r

def bautista_buffer_factor(ca_uM,*,indicator_uM=111.,indicator_Kd_uM=.123,endogenous_kappa=79.):
    """Fig10 example: author's k_cyto=80 ALREADY INCLUDES free calcium.
    Thus project kappa=79, not 80. Concentration-dependent dye buffering.
    """
    v=np.asarray(ca_uM,float)
    if not np.isfinite(v).all() or np.any(v<0) or not np.isfinite([indicator_uM,indicator_Kd_uM,endogenous_kappa]).all() or min(indicator_uM,endogenous_kappa)<0 or indicator_Kd_uM<=0:
        raise ValueError('Invalid buffer/indicator parameters')
    return 1+endogenous_kappa+indicator_uM*indicator_Kd_uM/(indicator_Kd_uM+v)**2

def bautista_reference_curve(ca_uM,*,modulated=False):
    """One Jurkat example, Fig11; reconstruction of published fits, not raw data."""
    c=np.asarray(ca_uM,float);factor=bautista_buffer_factor(c)
    vmax,k,n=(4e6,.140,2.1) if modulated else (2.8e6,.303,1.8)
    ions=vmax*hill(c,k,n);flux=ions/(IONS_PER_UM3_PER_UM*2000.)
    apparent=(.050*hill(c,.430,2)) if modulated else (.029*hill(c,.480,2))
    return dict(calcium_uM=c,buffer_factor=factor,buffer_corrected_flux_ions_s=ions,
        buffer_corrected_flux_uM_s=flux,free_Ca_decay_uM_s=flux/factor,
        independent_Fig9_apparent_fit_uM_s=apparent,reference_volume_um3=2000.,
        system='Jurkat human leukemic T cell, 22-25 C; nonunique example fit',
        source_memory_reversal_tau_s=240.,human_MN_fit=False)

def paired_effective_volume(*,calcium_current_step_pA,free_ca_slope_change_uM_s):
    """Returns V*(1+kappa_total) from matched inward Ca current / slope magnitudes.
    Identifies their product, not V and kappa separately. Ca-selective current only.
    """
    if not np.isfinite([calcium_current_step_pA,free_ca_slope_change_uM_s]).all() or min(calcium_current_step_pA,free_ca_slope_change_uM_s)<=0:raise ValueError('Positive matched magnitudes required')
    return 5182.1348283*calcium_current_step_pA/free_ca_slope_change_uM_s
