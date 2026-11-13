"""[codex] 2026-10-25. Pure parameter schema, no native NEURON import."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class ElectricalGeometry:
    process_area_fraction: float = .1
    process_length_um: float = 100.
    axial_resistivity_ohm_cm: float = 100.
    ampa_process_fraction: float = .9
    process_sodium_density_ratio: float = 1.
    provenance: str = 'Assumed two-section geometry; membrane area/capacitance/channel budgets preserved'
    def validate(self):
        v=[v for k,v in asdict(self).items() if k!='provenance']
        if not np.isfinite(v).all() or not 0<self.process_area_fraction<1 or min(self.process_length_um,self.axial_resistivity_ohm_cm)<=0 or not 0<=self.ampa_process_fraction<=1 or self.process_sodium_density_ratio<0 or not self.provenance.strip():raise ValueError('Invalid electrical geometry/provenance')

ELECTRICAL_GEOMETRY_RANGES=dict(process_area_fraction=[.05,.1,.3],process_length_um=[30.,100.,300.],
    axial_resistivity_ohm_cm=[50.,100.,500.],ampa_process_fraction=[0.,.5,.9,1.],process_sodium_density_ratio=[0.,1.,3.])
