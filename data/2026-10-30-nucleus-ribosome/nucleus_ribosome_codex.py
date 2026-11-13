"""[codex] 2026-10-30. Optional finite biosynthesis branch, no human kinetic fit.

TDP U/P remain the ONLY intact protein owner. A nuclear fraction partitions
that pool; it does not add protein. RNA amounts are relative to each basal
cytoplasmic productive RNA pool. Ribosome slots are a selected allocatable
capacity, NOT all ribosomes/cell, and are not converted from RNA fold change.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit

NUCLEUS_NAMES=("nuclear_intact_TDP_fraction", "TARDBP_precursor_RNA", "TARDBP_nuclear_mature_RNA",
    "STMN2_precursor_RNA", "STMN2_nuclear_productive_RNA", "STMN2_cryptic_RNA",
    "ribosome_TARDBP_slots", "ribosome_STMN2_slots", "STMN2_translation_ATP_uM", "STMN2_replaced_background_ATP_uM")

@dataclass(frozen=True)
class NucleusRibosomeParameters:
    passive_export_per_s: float = 1e-5
    precursor_processing_s: float = 300.
    mature_RNA_export_s: float = 900.
    cryptic_RNA_half_life_s: float = 3600.
    basal_STMN2_productive_fraction: float = .99
    elongation_aa_s: float = 4.08
    TARDBP_bound_slots: float = .1
    STMN2_bound_slots: float = .1
    ATP_half_uM: float = 500.
    STMN2_baseline_protein_uM: float = .1
    STMN2_length_aa: float = 179.
    STMN2_protein_half_life_s: float = 7200.
    provenance: str = 'Pinarbasi2018 architecture; Harley2020 qualitative MN recovery; Tomuro2024 HEK elongation reference; Ellis2026 Fig5F ~2h STMN2 CHX chase in KOLF2.1J human MN, 1/2/4h sensitivity not CI. Other human MN capacities, RNA times, ATP coupling and nuclear exchange are hypotheses.'
    def vector(self):
        v=np.array([self.passive_export_per_s,self.precursor_processing_s,self.mature_RNA_export_s,
            self.cryptic_RNA_half_life_s,self.basal_STMN2_productive_fraction,self.elongation_aa_s,
            self.TARDBP_bound_slots,self.STMN2_bound_slots,self.ATP_half_uM,self.STMN2_baseline_protein_uM,self.STMN2_length_aa,self.STMN2_protein_half_life_s])
        if not np.isfinite(v).all() or np.any(v<=0) or not 0<v[4]<=1 or v[6]+v[7]>=1 or not self.provenance.strip():raise ValueError('Positive finite parameters, productive fraction <=1 and spare ribosome capacity required')
        return v

# Explicit scenario grids, not confidence intervals or fitted human values.
NUCLEUS_RANGES=dict(passive_export_per_s=[1e-6,1e-5,1e-4],precursor_processing_s=[60.,300.,1800.],
    mature_RNA_export_s=[300.,900.,3600.],cryptic_RNA_half_life_s=[600.,3600.,21600.],
    basal_STMN2_productive_fraction=[.95,.99,1.],elongation_aa_s=[1.,4.08,10.],
    TARDBP_bound_slots=[.01,.1,.4],STMN2_bound_slots=[.01,.1,.4],ATP_half_uM=[100.,500.,2000.],
    STMN2_baseline_protein_uM=[.01,.1,1.],import_support=[0.,.25,1.],initiation_support=[0.,.25,1.],
    elongation_support=[0.,.25,1.],splicing_support=[0.,.5,1.],STMN2_protein_half_life_s=[3600.,7200.,14400.])

def nucleus_schedule(segments, supplied=None):
    a=np.c_[segments[:,:2],np.ones((len(segments),4))] if supplied is None else np.asarray(supplied,float)
    if a.shape!=(len(segments),6) or not np.array_equal(a[:,:2],segments[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0) or np.any(a[:,2:]>1):raise ValueError('Nuclear schedule: matching start/end, import/initiation/elongation/splicing support in [0,1]')
    return a

def initial_nucleus(p,hv,v,nuclear_fraction):
    f=float(nuclear_fraction)
    if not 0<f<1:raise ValueError('Nuclear fraction strictly between 0 and 1')
    return np.array([f,v[1]/hv[4],v[2]/hv[4],v[1]/(p[23]*v[4]),v[2]/p[23],
        (1-v[4])/(p[23]*v[4])*v[3]/np.log(2),v[6],v[7],0.,0.])

@njit
def nucleus_fluxes(x,tdp_total,mT,mS,ATP,p,hv,v,f0,ATP0,controls):
    """Returns new state rates, RNA rates and *completed* protein flux factors.

    Uniform loss/phosphorylation/cleavage across locations is an explicit
    unresolved assumption. All new full-length TDP enters the cytoplasm.
    Import is effective Ran-cycle support, not direct ATP per imported cargo.
    """
    x=np.maximum(x,0.);T=max(tdp_total,1e-20);f=min(x[0],1.)
    energy=max(ATP,0.)/(v[8]+max(ATP,0.))*(v[8]+ATP0)/ATP0
    nuc=T*f/f0
    target=1. if hv[0]==0 else hv[1]+(hv[2]-hv[1])/(1+(hv[2]-1)/(1-hv[1])*nuc**hv[3])
    free=max(0.,1-x[6]-x[7]);free0=1-v[6]-v[7]
    # Completion delay: L/v; fixed canonical lengths, no ribosome collisions.
    kT=v[5]/414.;kS=v[5]/v[10]
    iT=kT*v[6]/free0*max(mT,0.)*free*energy*controls[1]
    iS=kS*v[7]/free0*max(mS,0.)*free*energy*controls[1]
    cT=kT*x[6]*energy*controls[2];cS=kS*x[7]*energy*controls[2]
    t_factor=cT/(kT*v[6]);s_factor=cS/(kS*v[7])
    syn=p[20]*t_factor
    # Basal import matches the EXISTING nuclear fraction and gross synthesis.
    # This matching is initialization, not a fitted human import measurement.
    kin=f0/(1-f0)*(v[0]+p[20])
    dx=np.zeros(10)
    dx[0]=kin*controls[0]*energy*(1-f)-v[0]*f-f*syn/T
    transT=target**(1-hv[5])/hv[4]
    spliceT=x[1]/v[1];exportT=x[2]/v[2];decayT=max(mT,0.)/target**hv[5]/hv[4]
    dx[1]=transT-spliceT;dx[2]=spliceT-exportT
    nh=nuc**p[22];kh=p[21]**p[22]
    function=nh*(1+kh)/(nh+kh) if nh+kh>0 else 0.
    q=min(1.,v[4]*function)*controls[3]
    transS=1/(p[23]*v[4]);splicedS=x[3]/v[1];exportS=x[4]/v[2]
    dx[3]=transS-splicedS;dx[4]=q*splicedS-exportS
    dx[5]=(1-q)*splicedS-np.log(2)*x[5]/v[3]
    dx[6]=iT-cT;dx[7]=iS-cS
    # Gross production, not net protein change. Basal allocation replaces
    # the same amount of existing other-demand. No second TDP ATP debit.
    stmn_cost=np.log(2)/v[11]*v[9]*v[10]*4.*s_factor
    baseline_cost=np.log(2)/v[11]*v[9]*v[10]*4.
    dx[8]=stmn_cost;dx[9]=baseline_cost
    return dx,exportT-decayT,exportS-max(mS,0.)/p[23],t_factor,s_factor,nuc

@njit
def replace_biosynthesis(z,old,p,hv,v,f0,ATP0,controls):
    out=old.copy()
    dx,dmT,dmS,tf,sf,nuc=nucleus_fluxes(z[320:330],z[6]+z[7],z[50],z[10],z[120],p,hv,v,f0,ATP0,controls)
    delta=p[20]*tf-old[18]
    out[6]+=delta;out[18]+=delta
    out[50]=dmT;out[10]=dmS
    out[11]=np.log(2)/v[11]*(sf-z[11])
    # Existing axonal STMN2 production/transport is retained, not reassigned
    # to somatic ribosomes. No fictional absolute protein amount for that pool.
    return out,dx

@njit
def append_nucleus(z,old,dx,other0,underlying_other):
    out=np.zeros(330);out[:320]=old;out[320:330]=dx
    # Replace a fractional basal background allocation, following its support.
    baseline=dx[9];replaced=baseline*max(underlying_other,0.)/other0 if other0>0 else 0.
    out[329]=replaced
    delta=dx[8]-replaced
    out[120]-=delta;out[121]+=delta;out[125]+=delta
    return out

def nucleus_checks(z):
    return dict(nuclear_fraction_bounds=float(max(0.,-z[:,320].min(),z[:,320].max()-1)),
        biosynthesis_pool_nonnegative=float(max(0.,-z[:,321:330].min())),
        ribosome_capacity_excess=float(max(0.,(z[:,326]+z[:,327]-1).max())))

def hachoumi2022_reference():
    return dict(source='10.1016/j.cub.2022.01.012',species='Xenopus laevis stage42',
        cells='mixed spinal CPG neurons selected for usAHP; no anatomical identification',
        stimulus=dict(duration_s=20.,frequency_Hz=20.,pulse_ms=2.,current_pA=40.),
        amplitude_mV=dict(mean=5.3,reported_plus_minus=1.2),duration_s=dict(mean=25.48,reported_plus_minus=4.4),n=6,
        error_type='not assigned from the numerical sentence; do not treat as a human confidence interval',
        ouabain_uM=.5,interpretation='Dynamic pump usAHP abolished without RMP shift; mixed-cell source benchmark only',
        human_motor_neuron_fit=False)

def nuclear_recovery_reference():
    return dict(source='10.3390/ijms21218346',cells='human iPSC spinal MN, 3 control lines',
        age='6 days terminal differentiation after day18; distinct from Burley day47',
        assay='fixed-cell nuclear/cytoplasmic immunofluorescence ratio; not live import tracking',
        osmotic=dict(sorbitol_M=.4,stress_h=1.,recovery_h=[1.,2.],observation='TDP43 ratio rises but remains below untreated after 2h'),
        heat=dict(temperature_C=42.,stress_h=1.,recovery_h=2.,observation='no significant TDP43 nuclear recovery after 2h'),
        usable_rate_constant=None,raw_per_well_values_obtained=False,
        caveat='Do not infer an import constant, calcium injury law or nuclear fraction directly from endpoint intensity ratios. Fig4 arsenite .05mM differs from Methods .5mM; no arsenite dose transfer.')

def translation_source_reference():
    from pathlib import Path
    import json
    rows=json.loads((Path(__file__).parent/'selected_ribosome_rows_codex.json').read_text(encoding='utf8'))
    tar=next(r for r in rows if r['Ensembl_transcript_ID']=='ENST00000240185.7')
    return dict(source='10.1038/s41467-024-51258-0',cells='HEK293 T-REx, not motor neurons',
        TARDBP=tar,STMN2_in_table=any(r['Gene']=='STMN2' for r in rows),
        global_elongation_codon_s=4.08,
        table_implied_elongation_codon_s=tar['ORF_length_[codon]']/(tar['Ribosome_number_mean']*tar['Initiation_rate_[s/event]']),
        caveat='ORF415 includes stop, protein414aa. Table-implied4.14 vs text4.08 retained as source discrepancy; no forced exact joint fit. Half_life is RNA, not TDP protein. Multiple isoforms are not biological replicates.')

def hachoumi_pulse_train(rest_s=1.,recovery_s=60.):
    if min(rest_s,recovery_s)<0:raise ValueError('Nonnegative rest/recovery')
    rows=[]
    if rest_s:rows.append([0.,rest_s,0.])
    for k in range(400):
        a=rest_s+k*.05;rows.extend([[a,a+.002,40.],[a+.002,rest_s+(k+1)*.05,0.]])
    if recovery_s:rows.append([rest_s+20,rest_s+20+recovery_s,0.])
    return np.asarray(rows)
