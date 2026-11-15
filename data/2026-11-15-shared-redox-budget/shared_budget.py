"""Codex 2026-11-15. Optional common-resource redox extension.

Physical ATP/ADP and species AA belong to the host. GSH synthesis is not
ribosomal translation and consumes no GTP. Existing NDPK/translation keep
sole ownership of GTP and its ATP recharge. Redox NADPH supply is still an
explicit external scenario, not an implemented pentose-phosphate pathway.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import numpy as np
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'neuron-model'))
from model_contracts import load_module
B=Path(__file__).resolve().parent; D=B.parent
C=load_module(D/'2026-11-14-peroxide/coupled_redox.py','_shared15_claude_coupled')
L=C.L; P=C.P
NAMES=tuple('shared_redox_'+n for n in L.NAMES)+tuple('shared_peroxide_'+n for n in P.NAMES)+('shared_GSH_HNE_intracellular','shared_GSH_HNE_exported')
NS=len(NAMES)
# Canonical host AA order sorted one-letter amino-acid codes.
AA_INDICES=np.array([746,748,750])  # C, E, G in ACDEFGHIKLMNPQRSTVWY
RANGES=dict(active_water_fraction=[.1,.5,1.],AA_half_uM=[10.,100.,500.],conjugate_export_tau_s=[60.,600.,3600.],NADPH_supply_multiplier=[.1,.3,1.])

@dataclass(frozen=True)
class SharedRedoxParameters:
    active_water_fraction:float=1.
    AA_half_uM:float=100.
    conjugate_export_tau_s:float=600.
    coupling_enabled:bool=True
    provenance:str='Scenario, not human MN calibration; existing finite resources, no new nutrient or ATP source'
    def validate(self):
        v=[self.active_water_fraction,self.AA_half_uM,self.conjugate_export_tau_s]
        if not np.isfinite(v).all() or not 0< v[0]<=1 or min(v[1:])<=0 or not isinstance(self.coupling_enabled,bool) or not self.provenance.strip():
            raise ValueError('Positive finite parameters, fraction in (0,1], explicit boolean and provenance required')

def prepare(base,p,water_ratio):
    p.validate()
    y,model,report,params=C.prepare()
    ref,q,h0,share=model; lp=params[0]
    u=L.resting_inputs(lp)
    conjugate=L.fluxes(y[:27],u,ref)[6]*p.conjugate_export_tau_s
    red=np.r_[y,conjugate,0.]
    aa0=np.maximum(base[AA_INDICES],0.)/water_ratio
    gates=aa0/(p.AA_half_uM+aa0)
    if gates.min()<=0:raise ValueError('Positive precursor amino acids required')
    context=(ref,q,float(h0),float(share),np.array([lp.resting_ATP_uM,p.conjugate_export_tau_s,p.AA_half_uM,water_ratio,p.active_water_fraction*water_ratio,float(p.coupling_enabled),float(gates.min())]))
    return np.r_[base,red],context,dict(parameters=asdict(p),offset=len(base),count=NS,reference=ref.tolist(),peroxide_reference=q.tolist(),H2O2_reference=h0,initiation_share=share,resource_vector=context[-1].tolist(),preparation=report,ownership='Host ATP/ADP and Cys/Glu/Gly; host GTP unchanged directly. NADPH supply external, old relative iron/GSH block unchanged and not substituted.',initialization='New redox pools declared at connection time; not a simulated 600-hour redox history',volume='Redox pools physical uM in active water; host chemical/AA ledgers amount per reference volume')

@njit
def redox_rhs(y,ATP,aa,fe,oxidant,nadph_support,ref,q,h0,share,v):
    yl=y[:27];yp=y[27:45]
    up=np.ones(P.NU);up[0]=1.;up[1]=oxidant;up[2]=max(ATP,0.);up[3]=max(yl[9],0.);up[4]=max(yl[7],0.)
    fp=P.fluxes(yp,up,q);dp=P.rhs(yp,up,q)
    gate=1e30
    for a in aa:
        a=max(a,0.)/v[3];gate=min(gate,a/(v[2]+a))
    gate/=v[6]
    drive=1.+share*(max(yp[0],0.)/h0-1.)
    ul=np.ones(L.NU);ul[0]=fe;ul[1]=drive;ul[2]=max(ATP,0.);ul[3]=gate;ul[8]=nadph_support
    f=L.fluxes(yl,ul,ref);dl=L.rhs(yl,ul,ref)
    dl[7]-=2*fp[7];dl[8]+=fp[7];dl[9]-=fp[6];dl[23]+=fp[6]
    ka=ref[L.R_KATP];energy=(max(ATP,0.)/(ka+max(ATP,0.)))/(v[0]/(ka+v[0]))
    clearance=f[9]*energy;delta=clearance-f[9]
    dl[6]-=delta;dl[19]+=delta
    export=max(y[45],0.)/v[1]*energy
    dl[26]+=ref[L.R_CCLR]*delta+ref[L.R_CEXP]*(export-f[6])
    out=np.empty(NS);out[:27]=dl;out[27:45]=dp;out[45]=f[6]-export;out[46]=export
    return out

@njit
def finish(z,old,off,ref,q,h0,share,v,controls):
    # Legacy optional extensions can already allocate the full state length.
    # Preserve their prefix, and own the redox tail exactly once.
    out=np.zeros(len(z));out[:off]=old[:off]
    if v[5]==0:return out  # exact zero-coupling engineering control
    rd=redox_rhs(z[off:off+NS],z[136],z[AA_INDICES],controls[0],controls[1],controls[2],ref,q,h0,share,v)
    scale=v[4];cost=scale*(rd[26]+rd[44]);aa_cost=scale*rd[20]
    out[136]-=cost;out[137]+=cost;out[150]+=cost
    for i in AA_INDICES:out[i]-=aa_cost
    out[574]-=3*aa_cost
    out[off:]=rd
    return out

def check_view(z,base,off,v):
    """Credit only external-to-translation AA sink for legacy balance equations.
    Physical trajectory is never changed. New GSH/export ledgers close below.
    """
    zz=z.copy();bb=base.copy()
    for a in (zz,bb):
        cost=a[...,off+20]*v[4]
        a[...,574]+=3*cost
        for i in AA_INDICES:a[...,i]+=cost
    return zz,bb

def checks(z,base,off,context):
    ref,q,h0,share,v=context;y=z[:,off:off+NS];b=base[off:off+NS];d=y-b
    out=C.checks(y[:,:45],b[:45],ref,q)
    out['GSH_HNE_balance']=float(np.max(abs(d[:,45]-d[:,16]+d[:,46])))
    out['shared_glutathione_material']=float(np.max(abs(d[:,7]+2*d[:,8]+d[:,45]-d[:,20]+d[:,21]-d[:,16]+d[:,46])))
    out['nonnegative']=float(max(0.,-y.min()))
    out['physical_host_AA_nonnegative']=float(max(0.,-z[:,745:765].min()))
    out['ATP_stoichiometry']=float(np.max(abs(d[:,26]-2*d[:,20]-ref[L.R_CREM]*d[:,14]-ref[L.R_CEXP]*d[:,46]-ref[L.R_CCLR]*d[:,19])))
    out['sulfiredoxin_ATP_stoichiometry']=float(np.max(abs(d[:,44]-q[P.Q_CATP]*d[:,40])))
    return {'shared_redox_'+k:float(value) for k,value in out.items()}

def observe(z,base,off,context):
    v=context[-1];d=z[:,off:off+NS]-base[off:off+NS]
    return dict(parameters_vector=v.tolist(),ATP_spent_reference_uM=v[4]*(d[:,26]+d[:,44]),AA_spent_each_reference_uM=v[4]*d[:,20],direct_GTP_spent=0.,scope='Actual shared host debit; NADPH supply still external; enzyme abundance fixed; not whole-proteome or human calibration')
