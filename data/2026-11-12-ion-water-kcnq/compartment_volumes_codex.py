"""[codex] Explicit mobile-pool water volume, separate from calcium/reference volume.

Reference-normalized reaction amounts/counters retain their previous volumes.
Only Na/K/Cl, ATP/ADP/AMP and PCr concentrations use the new mobile water volume.
This is a fixed-capacity experiment, NOT swelling, new synthesis capacity or a
claim that every biochemical state is an actual water concentration.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np

RANGES={'soma_water_to_reference_ratio':[1.,2.,4.],
        'process_water_fraction':[.2,.5,.8]}

@dataclass(frozen=True)
class WaterVolumeParameters:
    soma_water_to_reference_ratio:float=1.
    process_water_fraction:float=.5
    provenance:str='Unmeasured scenario; cylinder water capacity independent of effective Ca/reference volume. No human MN volume fit.'
    def validate(self):
        if not np.isfinite([self.soma_water_to_reference_ratio,self.process_water_fraction]).all() or self.soma_water_to_reference_ratio<=0 or not 0<self.process_water_fraction<=1 or not self.provenance.strip():
            raise ValueError('Positive finite soma ratio, process fraction in (0,1], provenance required')

def prepare(p,geometry,nstates,chloride_offset=None):
    p.validate()
    reference=np.array([geometry.soma_volume_um3,geometry.process_volume_um3])
    total=np.pi*geometry.process_radius_um**2*geometry.process_length_um
    water=np.array([reference[0]*p.soma_water_to_reference_ratio,total*p.process_water_fraction])
    if hasattr(geometry,'soma_diameter_um'):
        soma_total=np.pi*geometry.soma_diameter_um**2*geometry.soma_length_um/4
        if water[0]>soma_total*(1+1e-12):raise ValueError('Somatic mobile water exceeds geometric cylinder volume')
    ratios=water/reference
    scale=np.ones(nstates)
    owned=[]
    for j,indices in enumerate(([120,121,122,199,200],[136,137,138,219,220])):
        indices=list(indices)
        if nstates>=318:indices.append(315+j)
        if chloride_offset is not None:indices.append(chloride_offset+6*j)
        scale[indices]=1/ratios[j];owned.append(indices)
    passport=dict(parameters=asdict(p),reference_volume_um3=reference.tolist(),
        calcium_effective_volume_um3=reference.tolist(),mobile_water_volume_um3=water.tolist(),
        process_geometric_volume_um3=float(total),water_to_reference_ratio=ratios.tolist(),
        physical_concentration_indices=owned,
        convention='Mobile concentration derivative = reference amount density derivative * Vref/Vwater. All cumulative ledgers, protein/RNA pools, reaction capacities and Ca remain reference-normalized. Exterior reservoirs and terminal unchanged; no swelling or restoration term.',
        limitations=['Equal initial concentrations imply different initial mobile molecule amounts.',
                     'This isolates capacity at fixed enzyme/pump counts; does not calibrate concentrations of all macromolecules.',
                     'Extracellular water, terminal volume and axial cross sections are separate unchanged assumptions.'])
    return scale,passport

def reference_delta(z,base,scale):
    """Conserved changes as amount/Vref, never used as kinetic concentrations."""
    return (z-base)/scale

def background_observation(reference_observation,passport):
    """Expose physical concentration rates; retain explicitly named reference rates."""
    if reference_observation is None:return None
    out=dict(reference_observation);out['compartments']=[]
    for j,q in enumerate(reference_observation['compartments']):
        q=dict(q);ratio=passport['water_to_reference_ratio'][j]
        for name in ('Na_background_in_uM_s','K_background_in_uM_s'):
            q[name.replace('_uM_s','_reference_uM_s')]=q[name]
            q[name]=q[name]/ratio
        out['compartments'].append(q)
    out['rate_volume_convention']='Reported Na/K rates use mobile water; explicitly marked reference rates and derived conversion remain amount/Vref. Voltage correction unchanged.'
    return out

def source_constraints():
    return json.loads((Path(__file__).parent/'source_constraints_codex.json').read_text(encoding='utf8'))

