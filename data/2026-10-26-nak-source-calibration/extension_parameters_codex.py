"""[codex] 2026-10-26. Source-aware optional recovery/energy extension.

No human fit. q is a lumped proton-motive reserve, NOT resolved mitochondrial
Na, delta-psi or pH. Protein costs use gross tracer fluxes and an uncertain
absolute conversion; 4 ATP->ADP/aa collapses charging, GTP regeneration and
paired AMP recycling. Original free adenylate kinase remains a separate flux.
"""
from dataclasses import dataclass,asdict
import numpy as np
from numba import njit

@dataclass(frozen=True)
class RecoveryParameters:
    enabled:bool=True
    alpha3_max_fraction:float=.5
    kinetic_family:str='fixed_hill3'
    provenance:str='Zahler1997 TableIII: paired Km/Hill families in HeLa with rat isoforms; abundance and transfer to human MN unknown'
    def vector(self):
        if self.kinetic_family not in ('fixed_hill3','variable_hill') or not 0<=self.alpha3_max_fraction<=1:raise ValueError('Invalid isoform family/fraction')
        return np.array([float(self.enabled),self.alpha3_max_fraction,*([12000.,44000.,3.,3.] if self.kinetic_family=='fixed_hill3' else [12000.,33000.,3.5,6.8])])

@dataclass(frozen=True)
class BudgetParameters:
    enabled:bool=True
    proton_reserve_uM:float=500.
    oxygen0_uM:float=200.
    oxygen_half_uM:float=10.
    oxygen_recovery_s:float=5.
    proton_leak_uM_s:float=5.
    protons_per_exported_ATP:float=11/3
    protons_per_O2:float=20.
    Ca_cycle_proton_equivalents:float=3.
    tdp_uM_per_lineage_unit:float=1.
    tdp_length_aa:float=414.
    translation_high_energy_per_aa:float=4.
    clearance_ATP_per_lineage:float=100.
    provenance:str='Watt c8 stoichiometry; all capacities/O2/absolute TDP quantities/Ca cycle costs unknown. Peth costs are non-TDP substrates, not a human clearance calibration'
    def vector(self):
        a=np.array([float(self.enabled),self.proton_reserve_uM,self.oxygen0_uM,self.oxygen_half_uM,self.oxygen_recovery_s,self.proton_leak_uM_s,self.protons_per_exported_ATP,self.protons_per_O2,self.Ca_cycle_proton_equivalents,self.tdp_uM_per_lineage_unit,self.tdp_length_aa,self.translation_high_energy_per_aa,self.clearance_ATP_per_lineage])
        if not np.isfinite(a).all() or np.any(a[1:5]<=0) or np.any(a[5:]<0) or min(a[6],a[7])<=0:raise ValueError('Invalid energy budget')
        return a

RECOVERY_RANGES=dict(alpha3_max_fraction=[0.,.5,.9],kinetic_family=['fixed_hill3','variable_hill'],alpha3_support=[0.,.25,1.])
BUDGET_RANGES=dict(proton_reserve_uM=[100.,500.,2000.],oxygen0_uM=[20.,200.,400.],oxygen_half_uM=[1.,10.,50.],oxygen_recovery_s=[.5,5.,50.],proton_leak_uM_s=[0.,5.,20.],protons_per_O2=[12.,16.,20.],Ca_cycle_proton_equivalents=[2.,3.,4.],tdp_uM_per_lineage_unit=[.1,1.,10.],clearance_ATP_per_lineage=[0.,25.,100.,1000.],respiratory_support=[0.,.5,1.])
BUDGET_NAMES=tuple(s+'_'+k for s in ('soma','process') for k in ('proton_reserve_uM','oxygen_uM','ETC_protons_uM','oxygen_consumed_uM','oxygen_supplied_uM','protons_leaked_uM','Ca_cycle_protons_uM','translation_ATP_uM','clearance_ATP_uM','replaced_background_ATP_uM'))

@njit
def shape(na,v,support):
    x=max(na,0.);h1=x**v[4]/(v[2]**v[4]+x**v[4]);h3=x**v[5]/(v[3]**v[5]+x**v[5])
    return (1-v[1])*h1+v[1]*h3*support

@njit
def recovery_extension(z,old,ctx,iv,ib,rv,support):
    out=old.copy()
    if rv[0]==0:return out
    for j in range(2):
        b=199+20*j;at=120 if j==0 else 136;v=ctx[27] if j==0 else ctx[28];cap=ctx[38][j]/100.;F=10364.26965661773/v
        na0=iv[2]*np.exp(-ctx[37][3]/iv[1]);x=max(z[b],0.)
        h=x**iv[7]/(iv[6]**iv[7]+x**iv[7]);h0=na0**iv[7]/(iv[6]**iv[7]+na0**iv[7])
        ratio=shape(x,rv,support)/max(h,1e-30)*h0/shape(na0,rv,1.)
        d=old[b+4]*(ratio-1.)
        out[b]-=3*d;out[b+1]+=2*d;out[b+2]+=3*d/iv[10];out[b+3]-=2*d/iv[10]
        out[at]-=d;out[at+1]+=d;out[185+7*j]-=1000*d/F/cap
        out[b+4]+=d;out[b+14]+=d/F;out[b+15]+=d
    return out

@njit
def protein_costs(old,v,j):
    # Soma gross synthesis/clearance counters18/19; local TDP178/177.
    s=max(old[18 if j==0 else 178],0.)*v[9]
    d=max(old[19 if j==0 else 177],0.)*v[9]
    return s*v[10]*v[11],d*v[12]

@njit
def budget_extension(z,old,ctx,v,ref,respiratory_support,ox_support):
    out=np.zeros(267);out[:247]=old
    if v[0]==0:return out
    for j in range(2):
        b=247+10*j;at=120 if j==0 else 136;ox=123 if j==0 else 148;other=125 if j==0 else 150;m=239+4*j;ci=0 if j==0 else 134
        q=max(z[b],0.)/v[1];O=max(z[b+1],0.);factor=2*q/(1+q)
        # Reduce the existing MCU-like entry and ATP production, not add second owners.
        entry=old[m+1]*min(1.,q);dc=entry-old[m+1]
        out[m]+=dc;out[m+1]=entry;out[ci]-=dc/(1+(ctx[30] if j==0 else ctx[32]))
        prod=old[ox]*factor;dp=prod-old[ox];out[ox]=prod;out[at]+=dp;out[at+1]-=dp
        sc,dcost=protein_costs(old,v,j)
        replaced=ref[j,2]*old[other]/ref[j,3] if ref[j,3]>0 else 0.
        cost=sc+dcost-replaced;out[other]+=cost;out[at]-=cost;out[at+1]+=cost
        # Preserve the historical operation exactly at default exponent 2.
        # Other exponents test response shape at the same basal/maximal ETC.
        power=q*q if len(v)<14 or v[13]==2. else q**v[13]
        etc=ref[j,0]*2/(1+power)*(O/(v[3]+O))/(v[2]/(v[3]+v[2]))*respiratory_support*ox_support
        leak=v[5]*q;ca_cost=v[8]*entry;Ouse=etc/v[7];Osupply=(ref[j,1]-O)/v[4]
        out[b:b+10]=[etc-v[6]*prod-leak-ca_cost,Osupply-Ouse,etc,Ouse,Osupply,leak,ca_cost,sc,dcost,replaced]
    return out
