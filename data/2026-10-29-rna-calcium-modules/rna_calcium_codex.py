"""[codex] 2026-10-29. Conditional protein->function layer, not an RNA fit.

GABA reversal is clamped (no Cl/HCO3 homeostasis). NMDA carries Na/K/Ca;
CaV1.2 replaces a specified fraction of the EXISTING VGCC current. FXYD7
modulates only the alpha1 contribution. Every added ion/ATP flux uses the
existing ownership ledgers. Surface CaV pools are finite, not new synthesis.
CLU and PRNP are retained as evidence candidates, not invented rate laws.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit

GENES=('GABRA1','GABRA2','GABRB2','GRIN2A','FXYD7','EIF3E','DNM1','CLTC','ACTN1','ACTN2','ACTN4','CLU','PRNP')
RNA_CALCIUM_NAMES=('soma_CaV12_surface_fraction','process_CaV12_surface_fraction')
@dataclass(frozen=True)
class RNACalciumParameters:
    # Factors mean functional protein/support relative to a comparator, NOT RNA FC.
    # DNM1/CLTC slots are effective endocytic support; use the explicit transfer
    # adapter for protein perturbations, because gene-specific necessity is unknown.
    protein_factors:tuple=(1.,)*13
    gaba_nS:float=0.
    gaba_reversal_mV:float=-75.
    alpha1_fraction:float=.5
    receptor_partner_ceiling:float=1.
    nmda_nS:float=0.
    nr2a_fraction:float=.5
    magnesium_mM:float=1.
    nmda_Ca_conductance_fraction:float=.03
    cav12_fraction:float=0.
    surface0_fraction:float=.5
    trafficking_tau_s:float=300.
    calcium_removal_gain:float=0.
    calcium_removal_half_uM:float=.5
    eif3e_insertion_gain:float=0.
    eif3e_removal_gain:float=1.
    actinin_weights:tuple=(1/3,1/3,1/3)
    # Includes unknown eligible alpha1-beta1 fraction and source-transfer strength.
    fxyd7_strength:float=0.
    fxyd7_reference_occupancy:float=.5
    fxyd7_Na_transfer:float=1.
    fxyd7_K_transfer:float=1.
    # Source alternatives must not be pooled across species/assay configurations.
    fxyd7_kinetic_source:str='Meyer2020'
    fxyd7_surface_factor:float=1.
    fxyd7_association_factor:float=1.
    fxyd7_K_shift_factor:float=1.
    fxyd7_alpha1_beta1_fraction:float=1.
    provenance:str='Conditional scenarios. Meyer2020 human proteins in Xenopus; Green2007 neuronal/heterologous, Buda2013 INS-1, Hall2013 neuronal/HEK. No matched human MN protein/current calibration.'
    def vector(self):
        f=np.asarray(self.protein_factors,float);w=np.asarray(self.actinin_weights,float)
        if f.shape!=(13,) or w.shape!=(3,) or not np.isfinite(f).all() or np.any(f<0) or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1):raise ValueError('13 nonnegative protein factors and normalized actinin weights required')
        # Avoid silently pretending these genes have an established dose law.
        if np.any(f[11:]!=1):raise ValueError('CLU/PRNP are evidence-only here: form/localization and causal dose law unresolved')
        vals=np.array([self.gaba_nS,self.gaba_reversal_mV,self.alpha1_fraction,self.receptor_partner_ceiling,self.nmda_nS,self.nr2a_fraction,self.magnesium_mM,self.nmda_Ca_conductance_fraction,self.cav12_fraction,self.surface0_fraction,self.trafficking_tau_s,self.calcium_removal_gain,self.calcium_removal_half_uM,self.eif3e_insertion_gain,self.eif3e_removal_gain,*w,self.fxyd7_strength,self.fxyd7_reference_occupancy,self.fxyd7_Na_transfer,self.fxyd7_K_transfer])
        if not np.isfinite(vals).all() or min(self.gaba_nS,self.nmda_nS,self.magnesium_mM,self.calcium_removal_gain)<0 or self.receptor_partner_ceiling<1 or min(self.trafficking_tau_s,self.calcium_removal_half_uM)<=0:raise ValueError('Invalid conductance/traffic parameters')
        for x in (self.alpha1_fraction,self.nr2a_fraction,self.nmda_Ca_conductance_fraction,self.cav12_fraction,self.eif3e_insertion_gain,self.eif3e_removal_gain,self.fxyd7_strength,self.fxyd7_Na_transfer,self.fxyd7_K_transfer):
            if not 0<=x<=1:raise ValueError('Fraction outside [0,1]')
        if not 0<self.surface0_fraction<1 or not 0<self.fxyd7_reference_occupancy<1 or self.nmda_Ca_conductance_fraction>=.3:raise ValueError('Invalid surface/occupancy/Ca fraction')
        sources={'Meyer2020':0.,'Beguin2002':1.,'Crambert2004':2.}
        extra=(self.fxyd7_surface_factor,self.fxyd7_association_factor,self.fxyd7_K_shift_factor,self.fxyd7_alpha1_beta1_fraction)
        if self.fxyd7_kinetic_source not in sources or not np.isfinite(extra).all() or min(extra)<0 or self.fxyd7_K_shift_factor>2 or self.fxyd7_alpha1_beta1_fraction>1:
            raise ValueError('Invalid FXYD7 source, availability, K-shift [0,2], or eligible beta1 fraction [0,1]')
        return np.r_[f,vals,sources[self.fxyd7_kinetic_source],extra]

RNA_CALCIUM_RANGES=dict(protein_factor=[0.,.5,1.,2.],RNA_to_protein_elasticity=[0.,.5,1.],
    gaba_nS=[0.,.1,.5],gaba_reversal_mV=[-85.,-75.,-55.,-40.],alpha1_fraction=[0.,.5,1.],receptor_partner_ceiling=[1.,2.],
    nmda_nS=[0.,.003,.03],nr2a_fraction=[0.,.5,1.],magnesium_mM=[0.,1.,2.],nmda_Ca_conductance_fraction=[.01,.03,.1],
    cav12_fraction=[0.,.5,1.],surface0_fraction=[.2,.5,.8],trafficking_tau_s=[60.,300.,1800.],calcium_removal_gain=[0.,1.,5.],
    calcium_removal_half_uM=[.1,.5,1.],eif3e_insertion_gain=[0.,.5,1.],eif3e_removal_gain=[0.,.5,1.],
    actinin_weights=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1/3,1/3,1/3]],fxyd7_strength=[0.,.5,1.],fxyd7_reference_occupancy=[.1,.5,.9],fxyd7_Na_transfer=[0.,1.],fxyd7_K_transfer=[0.,1.],dnm1_to_endocytosis_gain=[0.,.5,1.],cltc_to_endocytosis_gain=[0.,.5,1.])

# Scenario grids, NOT confidence intervals of the source measurements.
FXYD7_PRIMARY_RANGES=dict(fxyd7_kinetic_source=['Meyer2020','Beguin2002','Crambert2004'],
    fxyd7_surface_factor=[0.,.5,1.,2.],fxyd7_association_factor=[0.,.5,1.,2.],
    fxyd7_K_shift_factor=[0.,.5,1.,2.],fxyd7_alpha1_beta1_fraction=[0.,.5,1.])

def endocytic_support_scenario(*,dnm1_protein_factor=1.,cltc_protein_factor=1.,dnm1_to_endocytosis_gain=0.,cltc_to_endocytosis_gain=0.):
    """Explicit uncertain transfer, with no effect as the conservative default.

    Green dominant-negative dyn1 is not DNM1-specific depletion. General
    dynamin compensation / alternate endocytosis do not quantify CaV cargo
    transfer. These gains are scenarios, NOT inferred from RNA or knockout.
    Feed returned support values into the DNM1/CLTC functional-input slots.
    """
    fs=(dnm1_protein_factor,cltc_protein_factor);gs=(dnm1_to_endocytosis_gain,cltc_to_endocytosis_gain)
    if not np.isfinite(fs+gs).all() or min(fs)<0 or min(gs)<0 or max(gs)>1:raise ValueError('Nonnegative protein and transfer gain [0,1] required')
    return dict(zip(('DNM1','CLTC'),(1+g*(f-1) for f,g in zip(fs,gs))))

def protein_scenario(rna_fold,elasticity):
    """Explicit hypothesis mapping; never used automatically by the driver."""
    if not 0<=elasticity<=1 or any(k not in GENES or not np.isfinite(v) or v<=0 for k,v in rna_fold.items()):raise ValueError('Positive known RNA folds and elasticity [0,1] required')
    return {k:float(v**elasticity) for k,v in rna_fold.items()}

def ligand_schedule(segments,values=None):
    """Rows [start,end,GABA_open,NMDA_open], identical edges to stimuli."""
    if values is None:return np.c_[segments[:,:2],np.zeros((len(segments),2))]
    x=np.asarray(values,float)
    if x.shape!=(len(segments),4) or not np.isfinite(x).all() or not np.array_equal(x[:,:2],segments[:,:2]) or np.any(x[:,2:]<0) or np.any(x[:,2:]>1):raise ValueError('Ligands [0,1], same schedule edges required')
    return x

@njit
def receptor_capacity(v):
    # Two effective alpha classes compete for the same beta/other partner pool.
    alpha=v[15]*v[0]+(1-v[15])*v[1]
    gaba=min(alpha,v[2],v[16])
    nmda=min(1-v[18]+v[18]*v[3],v[16])
    return gaba,nmda

@njit
def surface_rates(surface,ca,c0,v):
    activity=max(0.,ca-c0)/(v[25]+max(0.,ca-c0))
    eif_in=1+v[26]*(v[5]-1);eif_out=1+v[27]*(v[5]-1)
    retention=1+v[28]*v[8]+v[29]*v[9]+v[30]*v[10]
    # Independent necessary supports; not a demonstrated sequential complex.
    remove=eif_out*min(v[6],v[7])*2/retention*(1+v[24]*activity)
    insert=eif_in
    return v[22]/v[23]*insert*(1-surface)-(1-v[22])/v[23]*remove*surface

@njit
def fxyd_ratio(nai,ko,vm,v):
    """Source endpoint ratios; interpolate occupancy, not abundance=flux.

    Meyer2020 Tables1/3: K half 1.6->2.4mM(-100mV), .9->1.4(0mV);
    Na half 13.8->10.5mM, Hill2.8->2.2 in intracellular K. These assays
    were different configurations; their product is a transfer hypothesis.
    """
    source=v[35];ko=max(ko,0.)
    if source==0:
        a=min(1.,max(0.,(vm+100)/100.))
        k0=1600+(900-1600)*a;k1=2400+(1400-2400)*a;hill=1.
        x=max(nai,1e-9)/1000.
        nr=(x**2.2/(10.5**2.2+x**2.2))/(x**2.8/(13.8**2.8+x**2.8))
    else:
        # Rat alpha1/beta1 + mouse FXYD7 in oocytes, 90mM external Na.
        # Table means at -50mV; transfer to other voltages is a FLAT scenario,
        # not a fitted voltage law. No significant Na-affinity shift in 2002.
        k0=510. if source==1 else 580.
        k1=990. if source==1 else 910.;hill=1.6;nr=1.
    kr_ref=(k0**hill+ko**hill)/(k1**hill+ko**hill)
    k_changed=k0+v[38]*(k1-k0)
    kr=(k0**hill+ko**hill)/(k_changed**hill+ko**hill)
    na=1+v[33]*(nr-1)
    ratio=na*(1+v[34]*(kr-1));reference_ratio=na*(1+v[34]*(kr_ref-1))
    ref=v[32];half=(1-ref)/ref
    available=v[4]*v[36]*v[37];occ=available/(half+available)
    return (1+occ*(ratio-1))/(1+ref*(reference_ratio-1))

@njit
def gene_ion_extension(z,old,ctx,iv,rv,alpha3_support,v,ligand):
    """Called AFTER Na/K isoform replacement and BEFORE energy/mito consumers."""
    out=old.copy();p,c,gv,cv,fb,c0=ctx[36:42]
    gc,nc=receptor_capacity(v)
    for j in range(2):
        at=120 if j==0 else 136;b=199+20*j;ci=0 if j==0 else 134;vi=185+7*j
        vol=ctx[27] if j==0 else ctx[28];kappa=ctx[30] if j==0 else ctx[32];F=10364.26965661773/vol
        frac=gv[j]/(gv[0]+gv[1]);cap=c[0]*frac;vm=z[vi]
        # Cl/HCO3 reservoir is intentionally clamped; no hidden Ca or ATP debit.
        ig=v[13]*frac*gc*ligand[0]*(vm-v[14])
        block=1/(1+v[19]/3.57*np.exp(-.062*min(100.,max(-200.,vm))))
        gn=v[17]*frac*nc*ligand[1]*block
        gca=gn*v[20];gmono=gn*(1-v[20])
        emono=-v[20]*120/(1-v[20]);fna=(emono-c[4])/(c[3]-c[4])
        ena=iv[1]*np.log(max(z[b+2],1e-12)/max(z[b],1e-12));ek=iv[1]*np.log(max(z[b+3],1e-12)/max(z[b+1],1e-12))
        ina=gmono*fna*(vm-ena);ik=gmono*(1-fna)*(vm-ek);ica=gca*(vm-120.)
        # CaV1.2 owns only a declared fraction of the inherited VGCC current.
        vg0=1/(1+np.exp(-(gv[3]-cv[0])/cv[1]))
        priorvg=fb[2]*frac*(z[vi+5]**2*(vm-120)-vg0**2*(gv[3]-120))
        extra_vg=priorvg*v[21]*(z[318+j]/v[22]-1)
        ca_in=-(ica+extra_vg)*F/2
        out[vi]-=1000*(ig+ina+ik+ica+extra_vg)/cap
        out[ci]+=ca_in/(1+kappa)
        if j==0:out[22]+=ca_in/(1+kappa)
        else:out[143]+=ca_in
        out[b]-=ina*F;out[b+1]-=ik*F;out[b+2]+=ina*F/iv[10];out[b+3]+=ik*F/iv[10]
        out[b+5]+=ina*F;out[b+6]+=ik*F
        # Modify alpha1 only; existing current includes alpha1+alpha3 kinetics.
        share=1.
        if rv[0]>0:
            x=max(z[b],0);a1=(1-rv[1])*x**rv[4]/(rv[2]**rv[4]+x**rv[4]);a3=rv[1]*alpha3_support*x**rv[5]/(rv[3]**rv[5]+x**rv[5]);share=a1/max(a1+a3,1e-30)
        delta=out[b+4]*share*v[31]*v[39]*(fxyd_ratio(z[b],z[b+3],vm,v)-1)
        out[b]-=3*delta;out[b+1]+=2*delta;out[b+2]+=3*delta/iv[10];out[b+3]-=2*delta/iv[10]
        out[at]-=delta;out[at+1]+=delta;out[vi]-=1000*delta/F/cap
        out[b+4]+=delta;out[b+14]+=delta/F;out[b+15]+=delta
    return out

@njit
def append_surface(z,old,ctx,v):
    out=np.zeros(320);out[:318]=old
    out[318]=surface_rates(z[318],max(ctx[41]+z[0],0),ctx[41],v)
    out[319]=surface_rates(z[319],max(ctx[41]+z[134],0),ctx[41],v)
    return out
