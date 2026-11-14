"""[codex] 2026-10-26. Conditional TDP proteasome/energy coupling.

Peth2013 measures substrate-dependent ATP cost, NOT human TDP turnover or
an ATP concentration-response. The latter and pathway allocation stay uncertain.
One retained N-lineage unit is not one measured full-length fragment molecule.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from extension_parameters import budget_extension, protein_costs

@dataclass(frozen=True)
class ProteostasisParameters:
    enabled: bool = True
    ATP_half_uM: float = 300.
    ATP_hill: float = 1.
    process_TDP_UPS_fraction: float = .5
    UPS_ATP_per_lineage: float = 65.
    UPS_overhead_equivalents_per_lineage: float = 0.
    nonUPS_ATP_per_lineage: float = 100.
    provenance: str = 'Peth2013 Table1: purified mouse proteasomes, non-TDP substrates. 65 midpoint is an illustrative transfer, not a human TDP measurement; ATP response, pathway fractions, overhead and nonUPS cost unknown.'
    def vector(self):
        a=np.array([float(self.enabled),self.ATP_half_uM,self.ATP_hill,self.process_TDP_UPS_fraction,self.UPS_ATP_per_lineage,self.UPS_overhead_equivalents_per_lineage,self.nonUPS_ATP_per_lineage])
        if not isinstance(self.enabled,bool) or not np.isfinite(a).all() or min(a[1:3])<=0 or not 0<=a[3]<=1 or np.any(a[4:]<0) or not self.provenance.strip():
            raise ValueError('Invalid explicit proteostasis parameters')
        return a

PROTEOSTASIS_RANGES=dict(ATP_half_uM=[50.,300.,1000.],ATP_hill=[1.,2.],process_TDP_UPS_fraction=[0.,.5,1.],UPS_ATP_per_lineage=[50.,65.,160.,1000.,4000.],UPS_overhead_equivalents_per_lineage=[0.,50.,200.],nonUPS_ATP_per_lineage=[0.,100.,1000.],proteasome_support=[0.,.5,1.],folding_load=[1.,2.,4.])
PROTEOSTASIS_NAMES=tuple(s+'_'+k for s in ('soma','process') for k in ('TDP_UPS_cleared_lineage','TDP_nonUPS_cleared_lineage','TDP_UPS_ATP_uM','TDP_nonUPS_ATP_uM'))

@njit
def proteasome_factor(atp,atp0,v,support,folding):
    a=max(atp,0.)**v[2];a0=atp0**v[2];k=v[1]**v[2]
    return a/(k+a)*(k+a0)/a0*support/folding

@njit
def proteostasis_fluxes(z,old,ctx,row,v,support,folding):
    """Replace only the identifiable UPS part of existing TDP loss fluxes.

    Keep lysosomal modifiers, synthesis, cleavage, NCOA4 and other proteins.
    No reinterpretation of all fast aggregates as proteasomal substrates.
    """
    out=old.copy();ups=np.zeros(2)
    if v[0]==0:return out,ups
    p=ctx[0];av=ctx[3];nv=ctx[5];en=ctx[14];pv=ctx[46]
    f=proteasome_factor(z[120],en[0],v,support,folding)
    for i in range(4):
        k=p[14] if i<2 else p[13+i]
        q=k*z[6+i]*(1-nv[19])*row[10]
        delta=q*(f-1.)
        out[6+i]-=delta;out[19]+=delta;ups[0]+=q*f
        for j in range(2):
            b=37+4*j+i;q=av[4+4*j+i]*z[b]*(1-nv[20])*row[10]
            delta=q*(f-1.)
            out[b]-=delta;out[19]+=delta;out[46]+=delta;ups[0]+=q*f
    f=proteasome_factor(z[136],en[0],v,support,folding)
    for i in range(2):
        q=pv[0,1+i]*z[175+i]*v[3]
        delta=q*(f-1.);out[175+i]-=delta;out[177]+=delta;ups[1]+=q*f
    return out,ups

@njit
def split_costs(out,ups,bpv,v,folding,j):
    total=max(out[19 if j==0 else 177],0.)
    nonups=total-ups[j]
    # Allow roundoff only; validation checks actual component nonnegativity.
    return ups[j]*bpv[9]*(v[4]*folding+v[5]),max(nonups,0.)*bpv[9]*v[6]

@njit
def proteostasis_budget(z,bio,ups,ctx,bpv,ref,resp,ox,v,folding):
    legacy=budget_extension(z,bio,ctx,bpv,ref,resp,ox)
    out=np.zeros(275);out[:267]=legacy
    if v[0]==0:return out
    for j in range(2):
        at=120 if j==0 else 136;other=125 if j==0 else 150;b=247+10*j;c=267+4*j
        oldcost=protein_costs(bio,bpv,j)[1]
        u,n=split_costs(bio,ups,bpv,v,folding,j);delta=u+n-oldcost
        out[at]-=delta;out[at+1]+=delta;out[other]+=delta;out[b+8]=u+n
        total=bio[19 if j==0 else 177]
        out[c:c+4]=[ups[j],total-ups[j],u,n]
    return out

def peth_reference(substrate,relative_ATPase=1.,proteasome_nM=1.):
    """Source assay, never a human-neuron preset. Reported Table1 intervals.

    ATPase reduction was induced with ATPgammaS at 2mM ATP, not by reducing ATP.
    Km printed in Fig3 is substrate Km, not an ATP Km. Folded+folate used same
    ATPase rate but lower throughput. Approximate endpoints are not raw data.
    """
    rows={'Ub5-DHFR':(4.7,13.,50.,80.,40.),'Ub5-DHFR+folate':(2.7,23.,90.,140.,None),'Ubn-Sic1':(2.3,26.,100.,160.,71.)}
    if substrate not in rows or not np.isfinite([relative_ATPase,proteasome_nM]).all() or not 0<relative_ATPase<=1 or proteasome_nM<=0:raise ValueError('Source substrate and positive assay scales required')
    rate,t,lo,hi,km=rows[substrate]
    return dict(substrate=substrate,throughput_per_min_per_26S=rate*relative_ATPase,time_s=60/(rate*relative_ATPase),published_rounded_time_s=t,ATP_per_substrate_interval=[lo,hi],ATP_hydrolysis_nM_per_min_interval=[rate*relative_ATPase*proteasome_nM*lo,rate*relative_ATPase*proteasome_nM*hi],substrate_Km_nM=km,ATP_concentration_Km_uM=None,source='Peth2013 Table1/Fig2-4, purified mouse 26S; no human TDP calibration')
