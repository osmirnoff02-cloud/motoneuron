"""[codex] 2026-11-04. U41 assay model and source-cell benchmarks.

No implicit K562-to-motor-neuron transfer. Label tracers describe an assay;
they are not extra physical RNA and never charge the Ran/ATP ledger again.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm

B = Path(__file__).resolve().parent

@dataclass(frozen=True)
class RNACompartmentRates:
    chromatin_release_s_inv: float
    nuclear_export_s_inv: float
    cytoplasmic_decay_s_inv: float
    nuclear_decay_s_inv: float
    cell_context: str
    provenance: str

    def validate(self):
        a=np.array([self.chromatin_release_s_inv,self.nuclear_export_s_inv,
                    self.cytoplasmic_decay_s_inv,self.nuclear_decay_s_inv],float)
        if not np.isfinite(a).all() or np.any(a[:3]<=0) or a[3]<0:
            raise ValueError('Positive release/export/cytoplasmic loss and nonnegative nuclear loss required')
        if not self.cell_context.strip() or not self.provenance.strip():
            raise ValueError('Cell context and rate provenance are mandatory')
        return a

def u41_reference_panel():
    return json.loads((B/'K562_rate_panel.json').read_text(encoding='utf8'))

def u41_ribosome_reference(*, cell_context):
    """HeLa imaging observations only; no ODE rates or energy costs are set."""
    if cell_context != 'HeLa-S3':
        raise ValueError('Ruland2021 observations are HeLa-S3, not human motor-neuron calibration')
    path = B.parent/'2026-11-04-U41-supplements/ribosome_reference.json'
    return json.loads(path.read_text(encoding='utf8'))


def u41_reference_rates(symbol, replicate, *, cell_context):
    """Explicitly select one author replicate; do not combine marginal CIs."""
    if cell_context != 'K562':
        raise ValueError('These source rates are K562 only; no human motor-neuron export calibration exists in this panel')
    row=next((r for r in u41_reference_panel()['rows'] if r['symbol']==symbol and r['replicate']==replicate),None)
    if row is None or not row['usable']:
        raise ValueError('No usable classified source profile for this gene and replicate')
    r=row['rates']
    return RNACompartmentRates(r['release']['MAP_per_s'],r['export']['MAP_per_s'],
        r['cytoplasmic_decay']['MAP_per_s'],r['nuclear_decay']['MAP_per_s'],cell_context,
        f"Ietswaart2024 Table S1 {row['sheet']} row {row['excel_row']}; marginal MAP projection, PUND={row['PUND']}; not a joint posterior fit")

def predict_fractionated_rna_labeling(times_s, rates, *, pulse_duration_s=None):
    """Steady pre-labeling totals, immediate perfect labeling, optional chase.

    CH'=p-(kR+kND)CH; NP'=kR CH-(kE+kND)NP; CY'=kE NP-kC CY.
    Nuclear observations include CH+NP. NTR is a fraction in [0,1].
    p=1 is an arbitrary normalization that cancels, NOT an absolute flux.
    Independent marginal MAPs do not recreate the published joint posterior.
    """
    kr,ke,kc,knd=rates.validate()
    t=np.asarray(times_s,dtype=float)
    if t.ndim!=1 or t.size==0 or not np.isfinite(t).all() or np.any(t<0):
        raise ValueError('Provide a nonempty vector of finite nonnegative seconds')
    if pulse_duration_s is not None and (not np.isfinite(pulse_duration_s) or pulse_duration_s<0):
        raise ValueError('Pulse duration must be finite nonnegative seconds')
    A=np.array([[-kr-knd,0.,0.],[kr,-ke-knd,0.],[0.,ke,-kc]])
    ss=np.linalg.solve(-A,np.array([1.,0.,0.]))
    # Augmented matrix evaluates the source integral without subtracting nearly
    # equal steady pools at short times, and handles coincident rates exactly.
    aug=np.zeros((4,4));aug[:3,:3]=A;aug[0,3]=1.
    labeled=[]
    for tm in t:
        during=tm if pulse_duration_s is None else min(tm,pulse_duration_s)
        x=expm(aug*during)[:3,3]
        if pulse_duration_s is not None and tm>pulse_duration_s:
            x=expm(A*(tm-pulse_duration_s))@x
        labeled.append(x)
    labeled=np.asarray(labeled)
    weights={'chromatin':np.array([1,0,0]),'nucleoplasm':np.array([0,1,0]),
             'nuclear':np.array([1,1,0]),'cytoplasm':np.array([0,0,1]),'whole_cell':np.ones(3)}
    fractions={name:(labeled@w)/float(ss@w) for name,w in weights.items()}
    return dict(times_s=t,NTR=fractions,rates=asdict(rates),
        steady_pool_fractions=ss/ss.sum(),nuclear_export_survival_fraction=kr/(kr+knd)*ke/(ke+knd),
        observation_model='three-compartment steady-state pulse/chase; chromatin included in nuclear',
        human_motor_neuron_calibrated=False,
        limitations=['Instantaneous complete labeling and steady state are assumptions; verify labeling lag, recovery and fraction purity',
                     'No absolute RNA synthesis, NPC dwell, Ran or ATP/GTP stoichiometry is inferred',
                     'K562 marginal MAP projection is not a refit of published NTR or an automatic override of API0.50/0.51'])

def u41_labeling_observations(source='Ietswaart2024_K562'):
    if source=='Ietswaart2024_K562':
        return json.loads((B/'K562_NTR_observations.json').read_text(encoding='utf8'))
    if source=='Vargas2026_human_i3_LMN':
        return json.loads((B/'human_MN_NTR_observations.json').read_text(encoding='utf8'))
    raise ValueError('Unknown source; choose Ietswaart2024_K562 or Vargas2026_human_i3_LMN')

def observe_u41_transport(result):
    """Project actual API0.50/0.51 pools and cumulative export, not pore dwell.

    Counts/concentrations use the SAME cytoplasmic reference volume; N/C is an
    amount ratio, not a concentration ratio between different compartment volumes.
    """
    obs=result.get('RNA_export_observation')
    if obs is None:raise ValueError('Result requires RNA_export_observation from API0.50/0.51')
    expected=('nuclear_free','UAP56_bound','NXF_nuclear','NXF_cytoplasmic','cytoplasmic_free',
              'produced','removed_or_delivered_downstream','export_released')
    if tuple(obs['RNA_fields'])!=expected:raise ValueError('Unrecognized RNA state contract')
    x=np.asarray(obs['RNA_pools_counters']);rib=np.asarray(obs['pre_ribosome_pools'])
    if x.ndim!=3 or x.shape[1:]!=(len(obs['genes']),8) or rib.shape!=(len(x),2,4):
        raise ValueError('RNA/pre-ribosome observation dimensions differ from state contract')
    nuc=x[:,:,:3].sum(2);cyt=x[:,:,3:5].sum(2)
    names=tuple(obs['counter_names']);c=np.asarray(obs['counters'])
    return dict(genes=tuple(obs['genes']),nuclear_RNA_uM_reference=nuc,cytoplasmic_RNA_uM_reference=cyt,
        RNA_export_released_cumulative_uM=x[:,:,7].copy(),
        pre_subunits=('40S','60S'),nuclear_pre_subunits_uM_reference=rib[:,:,:2].sum(2),
        cytoplasmic_pre_subunits_uM_reference=rib[:,:,2:].sum(2),
        pre_subunit_export_released_cumulative_uM=c[:,[names.index('pre40_export_released'),names.index('pre60_export_released')]].copy(),
        existing_energy_counters={k:c[:,names.index(k)].copy() for k in ('UAP56_ATP','DDX19_ATP','ribosome_RanGTP_hydrolyzed')},
        unit='uM per somatic cytoplasmic reference volume; not each compartment concentration',
        accounting='read-only projection; counters are not physical pools; no new ATP/GTP debit',
        unresolved='API0.50/0.51 have no explicit chromatin retention or label tracer states; source assay model is separate',
        human_motor_neuron_calibrated=False)
