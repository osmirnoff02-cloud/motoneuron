"""[codex] 2026-10-26. Explicit shared geometry and uncertain feedback."""
from pathlib import Path
from dataclasses import dataclass,asdict,replace
import importlib.util,sys,json
import numpy as np
from numba import njit
B=Path(__file__).resolve().parent
def load(path,name):
    if name in sys.modules:return sys.modules[name]
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
spatial=load(B.parent/'2026-10-25-pmca-spatial-model/spatial_pmca_codex.py','_connected_spatial26')
local=load(B.parent/'2026-10-25-spatial-electrical-injury/local_injury_codex.py','_connected_local26')
hh=load(B.parent/'2026-10-14-ipsc-electrical-multisource/electrical_model_codex.py','_connected_hh26')
gates,taus,derived=hh.gates,hh.taus,hh.derived
injury_rhs=local._rhs
@dataclass(frozen=True)
class SharedGeometry(spatial.SpatialGeometry):
    """No process endcap area: same nseg=1 cylinder convention as NEURON."""
    soma_length_um: float=1.
    soma_diameter_um: float=1.
    axial_resistivity_ohm_cm: float=100.
    @property
    def process_area_um2(self):return 2*np.pi*self.process_radius_um*self.process_length_um
    @property
    def connection_factor_um(self):
        return 1/(self.soma_length_um/2/(np.pi*self.soma_diameter_um**2/4)+self.process_length_um/2/(np.pi*self.process_radius_um**2))
    @property
    def axial_conductance_nS(self):return 100000*self.connection_factor_um/self.axial_resistivity_ohm_cm
    def validate(self):
        super().validate()
        if min(self.soma_length_um,self.soma_diameter_um,self.axial_resistivity_ohm_cm)<=0:raise ValueError('Positive cylinder dimensions and resistivity required')
        expected_area=np.pi*self.soma_diameter_um*self.soma_length_um
        expected_volume=np.pi*self.soma_diameter_um**2*self.soma_length_um/4*self.process_accessible_fraction
        if not np.isclose(self.soma_area_um2,expected_area,rtol=1e-12) or not np.isclose(self.soma_volume_um3,expected_volume,rtol=1e-12):raise ValueError('Shared cylinder geometry and accessible volume disagree')

CONNECTED_RANGES=dict(spectrin_to_Nav_loss_gain=[0.,.5,1.],spectrin_to_delivery_loss_gain=[0.,.5,1.],
    g_vgcc_nS=[0.,.003,.03],g_sk_nS=[0.,.015,.15],free_ca_diffusion_um2_s=[0.,30.,300.])
@dataclass(frozen=True)
class ConnectedParameters:
    spectrin_to_Nav_loss_gain:float=0.
    spectrin_to_delivery_loss_gain:float=0.
    g_vgcc_nS:float=0.
    g_sk_nS:float=0.
    provenance:str='Hypothesis controls; bulk Ca gates, no fitted human nanodomain or injury-to-Nav/traffic mapping'
    def validate(self):
        a=np.array([self.spectrin_to_Nav_loss_gain,self.spectrin_to_delivery_loss_gain,self.g_vgcc_nS,self.g_sk_nS])
        if not np.isfinite(a).all() or a.min()<0 or max(a[:2])>1 or not self.provenance.strip():raise ValueError('Invalid connected feedback')

def inputs(profile='Burley',*,process_fraction=.1,process_length_um=100.,axial_resistivity=100.,diffusion=30.):
    frozen=json.loads((B/'inherited_inputs_codex.json').read_text(encoding='utf-8'))
    def decode(v):
        if isinstance(v,list):return [decode(x) for x in v]
        if isinstance(v,dict) and 'class' in v:
            name=v['class'];typ=getattr(local,name,None) or getattr(spatial,name,None) or getattr(spatial.core,name,None)
            if typ is None:
                sys.path.insert(0,str(B.parent/'2026-10-24-er-unified-neuron'))
                from er_parameters_codex import ERParameters
                typ=ERParameters if name=='ERParameters' else None
            if typ is None:raise ValueError(name)
            return typ(**v['values'])
        return v
    opts={k:decode(v) for k,v in frozen['biochemical'].items()};inj={k:decode(v) for k,v in frozen['injury'].items()}
    pr=frozen['profiles'][profile];p=np.array(pr['p']);c=np.array(pr['c']);coupling=pr['coupling']
    if not np.isfinite([process_fraction,process_length_um,axial_resistivity,diffusion]).all() or not 0<process_fraction<1 or min(process_length_um,axial_resistivity)<=0 or diffusion<0:raise ValueError('Invalid geometry')
    f=process_fraction;area=c[0]*100;diam=np.sqrt(area*(1-f)/np.pi);radius=area*f/(2*np.pi*process_length_um);accessible=coupling['cytosol_fraction']
    geom=SharedGeometry(soma_volume_um3=np.pi*diam**3/4*accessible,soma_area_um2=area*(1-f),
        process_length_um=process_length_um,process_radius_um=radius,process_accessible_fraction=accessible,
        free_ca_diffusion_um2_s=diffusion,process_kappa=coupling['kappa'],process_gamma_per_s=coupling['gamma_per_s'],
        soma_length_um=diam,soma_diameter_um=diam,axial_resistivity_ohm_cm=axial_resistivity,
        provenance='Derived from frozen profile total capacitance, cm=1, native two-cylinder geometry; assumed cytosol fraction and diffusion')
    opts['geometry']=geom;c0=coupling['basal_ca_uM'];k=coupling['kappa'];gamma=coupling['gamma_per_s']
    opts['basal_ca_uM']=c0;opts['calcium_tau_s']=(1+k)/gamma
    opts['editing_parameters']=replace(opts['editing_parameters'],basal_ca_uM=c0,kappa=k,gamma_per_s=gamma,volume_um3=geom.soma_volume_um3)
    inj['basal_ca_uM']=c0
    opts.update(profile=profile,profile_p=p,profile_c=c,coupling=coupling,injury_parameters=inj)
    return opts

def schedule(rows):
    """rows=(start,end,mean_AMPA_open,ADAR2_activity,substrate_block)."""
    template=np.array(json.loads((B/'inherited_inputs_codex.json').read_text())['template'])
    out=np.tile(template,(len(rows),1))
    for r,values in zip(out,rows):r[0],r[1],r[2],r[22],r[3]=values
    return out

@njit
def membrane_rhs(z,p,c,geometry_vector,coupling_vector,feedback_vector,opening,capacity,surface_R,weights,ep,injection,basalCa):
    # Four frozen HH gates plus VGCC and SK gate, each compartment; seconds.
    out=np.zeros(14);currents=np.zeros(2);cs=np.array([basalCa+z[0],basalCa+z[134]])
    area_s,area_p,axial,init,hold,gl,el=geometry_vector
    fractions=np.array([area_s,area_p])/(area_s+area_p)
    damage=min(1.,max(0.,1.-z[180])) # betaII-spectrin initial amount = 1; verified by preparation
    cp=0.
    for n in range(5):cp+=weights[n]*(1-min(1.,max(0.,surface_R)))**n
    gcp=ep[8]*cp*ep[10]*ep[11];gci=ep[8]*(1-cp)
    gca=(gci*ep[12]+gcp*ep[13])*60./180.;gmono=gcp+gci-gca
    emono=-gca*120./gmono if gmono>0 else 0.
    vg_half,vg_slope,vg_tau,sk_half,sk_hill,sk_tau,ampa_fraction=coupling_vector
    vg0=1/(1+np.exp(-(init-vg_half)/vg_slope));sk0=basalCa**sk_hill/(sk_half**sk_hill+basalCa**sk_hill)
    for j in range(2):
        b=185+7*j;v=z[b];m,h,n,w=z[b+1:b+5];gv,zsk=z[b+5:b+7];frac=fractions[j];cap=c[0]*frac
        loss=(1-feedback_vector[0]*damage) if j else 1.
        ionic=p[0]*loss*m**3*h*(v-c[3])+(p[3]*n**4+p[12]*w)*(v-c[4])+gl*(v-el)
        af=ampa_fraction if j else 1-ampa_fraction
        ca_current=gca*af*opening*capacity*(v-120.)
        ampa_current=ca_current+gmono*af*opening*capacity*(v-emono)
        vg_current=feedback_vector[2]*frac*(gv**2*(v-120.)-vg0**2*(init-120.))
        sk_current=feedback_vector[3]*frac*(zsk-sk0)*(v-c[4])
        axial_current=axial*(z[192 if j==0 else 185]-v)
        out[7*j]=1000*((injection[j]+hold*frac+axial_current-ampa_current-vg_current-sk_current)/cap-ionic)
        gg=gates(v,p);tt=taus(v,p)
        for k in range(4):out[7*j+1+k]=1000*(gg[k]-z[b+1+k])/tt[k]
        out[7*j+5]=1000*(1/(1+np.exp(-max(-700.,min(700.,(v-vg_half)/vg_slope))))-gv)/vg_tau
        ca=max(cs[j],0.);out[7*j+6]=1000*(ca**sk_hill/(sk_half**sk_hill+ca**sk_hill)-zsk)/sk_tau
        currents[j]=-(ca_current+vg_current) # inward pA
    return out,currents
