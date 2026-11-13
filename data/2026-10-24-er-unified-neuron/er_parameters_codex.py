"""[codex] 2026-10-24. Explicit hypotheses for ER/cytosol exchange, not human fits."""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class ERParameters:
    enabled: bool = True
    volume_ratio: float = .06  # V_ER / accessible V_cytosol, not whole-cell fraction
    er_baseline_uM: float = 200.
    er_buffer_ratio: float = 10.  # constant differential buffer approximation
    serca_vmax_uM_s: float = 3.  # TOTAL Ca per accessible cytosolic volume / second
    serca_half_uM: float = .2
    release_per_s: float = .1
    cicr_gain: float = .1
    cicr_half_uM: float = .5
    soce_nS: float = 0.
    trpm_nS: float = 0.  # mouse TRPM5 hypothesis, zero in reference human profiles
    trpm_half_uM: float = 8.  # Bos2021 model / taste-cell source, not human MN fit
    trpm_tau_ms: float = 150.  # hypothetical gate smoothing; Bos tauCa is NOT a measured gate tau
    trpm_hill: float = 3.2
    provenance: str = 'Hypothesis ranges; Davi2026 human cortical ER refill and Burley/Dafinca MN release constrain interpretation, not these kinetic constants. Mouse TRPM5 is optional.'

    def validate(self, basal_ca_uM=.08):
        d=asdict(self); d.pop('provenance'); d.pop('enabled')
        if not isinstance(self.enabled,bool) or not self.provenance.strip() or not np.isfinite(list(d.values())).all():
            raise ValueError('Finite ER parameters, bool enabled, explicit provenance required')
        if min(d.values()) < 0 or min(self.volume_ratio,self.er_baseline_uM,self.serca_half_uM,self.cicr_half_uM,self.trpm_half_uM,self.trpm_tau_ms,self.trpm_hill)<=0:
            raise ValueError('Positive volumes/concentrations, nonnegative rates required')
        if self.er_baseline_uM<=basal_ca_uM: raise ValueError('ER baseline must exceed cytosol baseline')

    @property
    def capacity(self):
        return self.volume_ratio*(1.+self.er_buffer_ratio)

    @property
    def initial_amount(self):
        return self.capacity*self.er_baseline_uM

    def vector(self):
        self.validate()
        return np.array([float(self.enabled), self.capacity,self.er_baseline_uM,
            self.serca_vmax_uM_s,self.serca_half_uM,self.release_per_s,
            self.cicr_gain,self.cicr_half_uM,self.soce_nS],float)

RANGES = {
 'volume_ratio':[.02,.06,.1], 'er_baseline_uM':[100.,200.,500.],
 'er_buffer_ratio':[1.,10.,100.], 'serca_vmax_uM_s':[.3,3.,30.],
 'serca_half_uM':[.1,.2,.5], 'release_per_s':[.01,.1,1.],
 'cicr_gain':[0.,.1,1.], 'cicr_half_uM':[.2,.5,1.],
 'soce_nS':[0.,.005,.02], 'trpm_nS':[0.,.02,.1],
 'trpm_half_uM':[2.,8.,20.], 'trpm_tau_ms':[50.,150.,500.], 'trpm_hill':[2.,3.2,4.]
}
# Values are sensitivity settings, NOT confidence intervals or measured motor-neuron ranges.
