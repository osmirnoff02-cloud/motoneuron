"""Codex 2026-11-15: finite carbon replaces existing ATP/PPP source owners.

GLUT/HK laws adapted from BlueBrain/metabolism-in-aging, commit
1204a605a8d2397f2b78952f74d163df217ed2f9, Copyright 2024 EPFL/Blue Brain,
Apache-2.0: ../2026-11-15-donor-carbon/LICENSE_BlueBrain.txt and NOTICE.txt.
New reduced glycolysis/C3/bath closure is an explicit scenario, not full NGV.
Reference capacities are supplied once from an unperturbed legacy RHS.
They are never fitted to load, nutrient support, interventions or late drift.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import numpy as np
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'neuron-model'))
from model_contracts import load_module
B=Path(__file__).resolve().parent;D=B.parent
A=load_module(D/'2026-11-15-aa-gsh-nadph/supply.py','_stable_shared_supply15')
C=load_module(D/'2026-11-15-donor-carbon/carbon.py','_stable_donor_carbon15')
S=A.S;N=A.N;L=A.L
SIDES=('soma','process','terminal')
EXTRA=tuple('carbon_'+s+'_'+n for n in ('external_glucose_uM','glucose_uM') for s in SIDES)
EXTRA+=('carbon_soma_G6P_uM','carbon_terminal_G6P_uM')
EXTRA+=tuple('carbon_'+s+'_C3_uM' for s in SIDES)
LEDGERS=('boundary_glucose_net','transport_glucose_net','HK_glucose','glycolysis_C6','oxidized_C3','exported_C3','HK_ATP','ETC_protons')
EXTRA+=tuple('carbon_'+s+'_'+n+'_reference_uM' for s in SIDES for n in LEDGERS)
NAMES=A.NAMES+EXTRA;NS=len(NAMES)
RANGES=dict(carbon_external_support=[0.,.25,1.],carbon_transport_activity=[.5,1.,2.],
 carbon_HK_activity=[.5,1.,2.],carbon_glycolysis_activity=[.5,1.,2.],carbon_oxidation_activity=[.5,1.,2.],
 carbon_G6P_half_uM=[30.,100.,300.],carbon_C3_half_uM=[30.,100.,300.],carbon_ATP_half_uM=[30.,100.,300.],
 carbon_external_ratio=[.2,1.,5.],carbon_perfusion_tau_s=[60.,600.,3600.],terminal_ATP_per_C3=[14.,15.,16.])

@dataclass(frozen=True)
class CarbonCouplingParameters(A.SupplyParameters):
    carbon_enabled:bool=True
    reference_gly_ATP_uM_s:tuple=(0.,0.,0.)
    reference_ETC_H_uM_s:tuple=(0.,0.)
    reference_terminal_OX_uM_s:float=0.
    reference_ATP_uM:tuple=(0.,0.,0.)
    carbon_external_support:float=1.
    carbon_transport_activity:float=1.
    carbon_HK_activity:float=1.
    carbon_glycolysis_activity:float=1.
    carbon_oxidation_activity:float=1.
    carbon_G6P_half_uM:float=100.
    carbon_C3_half_uM:float=100.
    carbon_ATP_half_uM:float=100.
    carbon_C3_initial_uM:float=100.
    carbon_external_ratio:float=1.
    carbon_perfusion_tau_s:float=600.
    carbon_boundary_open:bool=True
    carbon_H_per_O2:float=20.
    carbon_H_per_ATP:float=11/3
    terminal_ATP_per_C3:float=15.
    def validate(self):
        super().validate()
        if not self.explicit_NADPH or self.active_water_fraction!=1. or not self.coupling_enabled:
            raise ValueError('Carbon closure requires explicit PPP and whole active process water/shared coupling')
        if not isinstance(self.carbon_enabled,bool) or not isinstance(self.carbon_boundary_open,bool):raise ValueError('Boolean carbon flags required')
        if len(self.reference_gly_ATP_uM_s)!=3 or len(self.reference_ETC_H_uM_s)!=2 or len(self.reference_ATP_uM)!=3:raise ValueError('Reference dimensions')
        vals=np.r_[self.reference_gly_ATP_uM_s,self.reference_ETC_H_uM_s,self.reference_terminal_OX_uM_s,self.reference_ATP_uM]
        if not np.isfinite(vals).all() or min(vals)<=0:raise ValueError('Explicit positive legacy reference rates/pools required; use capture_reference.py')
        for k in ('carbon_G6P_half_uM','carbon_C3_half_uM','carbon_ATP_half_uM','carbon_C3_initial_uM','carbon_external_ratio','carbon_perfusion_tau_s','carbon_H_per_O2','carbon_H_per_ATP','terminal_ATP_per_C3'):
            if getattr(self,k)<=0:raise ValueError(k)

@njit
def hk(g,a,q):
    g=max(g,0.);a=max(a,0.);q=max(q,0.)
    return .51617647*g/(g+50.)*a/(1.+(a/554.)**4)/(1.+q/10.21)

@njit
def glut(e,g):
    e=max(e,0.);g=max(g,0.);k=2800.
    return .41*400.*k*(e-g)/(k*k+k*e+k*.72*g+.72*e*g)

@njit
def normsat(x,k,x0):
    x=max(x,0.)
    return x/(k+x)*(k+x0)/x0

def prepare(base,p,water_ratio,soma_process_ratio=1.,soma_water_ratio=1.):
    p.validate();z,ctx,report=A.prepare(base,p,water_ratio,soma_process_ratio,soma_water_ratio)
    ref,q,h0,share,nq,v=ctx;nq=nq.copy();off=len(base)
    # Transfer the OTHER NADPH source into the existing PPP capacity, at one
    # fixed reference only. No independent carbon-free NADPH donor remains.
    f=N.fluxes(z[off+47:off+56],np.array([z[off+9],0.,1.,1.]),nq)
    if p.carbon_enabled:
        nq[1]*=(2*f[2]+f[4])/(2*f[2]);nq[5]=0.;nq[10]=1.
    f=N.fluxes(z[off+47:off+56],np.array([z[off+9],0.,1.,1.]),nq)
    donor=C.DonorCarbonParameters();extra=np.zeros(len(EXTRA))
    extra[:3]=donor.external_glucose_initial_uM;extra[3:6]=donor.glucose_initial_uM
    extra[6:8]=z[off+47];extra[8:11]=p.carbon_C3_initial_uM
    cq=np.zeros((3,27))
    for j,w in enumerate((soma_water_ratio,water_ratio,1.)):
        gg=p.reference_gly_ATP_uM_s[j];ppp=f[2]*w if j==1 else 0.
        # Steady carbon cost of oxidative PPP is one C atom / oxidation.
        # 3 Jgly - JHK = legacy net glycolytic ATP at the reference.
        gly=(gg+ppp/6.)/2.;hkr=gly+ppp/6.
        ox=(p.reference_ETC_H_uM_s[j]/(3*p.carbon_H_per_O2) if j<2 else p.reference_terminal_OX_uM_s/p.terminal_ATP_per_C3)
        if 2*gly<ox:raise ValueError('Reference oxidation needs an additional C3 donor; cannot fabricate it')
        g0=z[off+47];at0=p.reference_ATP_uM[j]
        cq[j]=[w,hkr/w/hk(extra[3+j],at0,g0),hkr/w/glut(extra[j],extra[3+j]),gly,gg,at0,g0,p.carbon_C3_initial_uM,
          (2*gly-ox)/(w*p.carbon_C3_initial_uM),hkr,p.carbon_external_ratio,p.carbon_perfusion_tau_s,extra[j],
          p.carbon_external_support,p.carbon_transport_activity,p.carbon_HK_activity,p.carbon_glycolysis_activity,p.carbon_oxidation_activity,
          p.carbon_G6P_half_uM,p.carbon_C3_half_uM,p.carbon_ATP_half_uM,3*p.carbon_H_per_O2,p.carbon_H_per_O2,p.terminal_ATP_per_C3,
          float(p.carbon_enabled),float(p.carbon_boundary_open),ox]
    z=np.r_[z,extra]
    report.update(count=NS,schema_key='shared_carbon15',carbon_parameters=asdict(p),carbon_reference=cq.tolist(),NADPH_reference=nq.tolist(),
      carbon_columns=list(EXTRA),ownership='Finite carbon owns HK and glycolytic source; old gross source replaced, HK charged once to existing ATP/ADP and other-cost ledger. Soma/process ETC draws finite C3, existing proton reserve produces oxidative ATP; terminal oxidative ATP draws C3. No duplicate ATP/NADPH/G6P pools.',
      reference_scope='Matched fixed legacy capacities, not human data or a whole-cell steady state. No intervention-dependent refit.',
      activity_counter_scope='Legacy activity_produced_ATP counter retains the requested pre-carbon activity response, not actual supplied ATP.',
      limits='Lumped C3 (pyruvate/lactate reducing equivalents), fast Pi/NAD recycling; no explicit TCA, NADH shuttle or terminal oxygen pool; C3 export is an external sink; no axial glucose diffusion.')
    return z,(ref,q,h0,share,nq,cq,v),report

@njit
def finish(z,old,off,ref,q,h0,share,nq,cq,v,controls):
    out=np.zeros(len(z));out[:off+A.NS]=A.finish(z[:off+A.NS],old[:off+A.NS],off,ref,q,h0,share,nq,v,controls)
    if cq[0,24]==0:return out
    b=off+A.NS;u=np.array([max(z[off+9],0.),0.,controls[2],0.])
    nf=N.fluxes(z[off+47:off+56],u,nq);ppp=nf[2];ret=nf[3]
    for j in range(3):
        c=cq[j];w=c[0];at=(120 if j==0 else 136 if j==1 else 282)
        gi=(b+6 if j==0 else off+47 if j==1 else b+7)
        e=z[b+j];g=z[b+3+j];g6=z[gi];p3=z[b+8+j]
        jt=w*c[2]*glut(e,g)*c[14];hkr=w*c[1]*hk(g,z[at],g6)*c[15]
        legacy_gly=old[124 if j==0 else 149 if j==1 else 301]
        gly=c[3]*max(legacy_gly,0.)/c[4]*normsat(g6,c[18],c[6])*normsat(z[at],c[20],c[5])*c[16]
        boundary=(c[9]+w*c[10]*(c[12]-e)/c[11])*c[13]*c[25]
        export=w*c[8]*max(p3,0.)
        if j<2:
            ib=247+10*j;oldH=old[ib+2]
            newH=max(oldH,0.)*normsat(p3,c[19],c[7])*c[17]
            oxid=newH/c[21];dh=newH-oldH
            out[ib]+=dh;out[ib+1]-=dh/c[22];out[ib+2]=newH;out[ib+3]=newH/c[22]
            dg=3*gly-legacy_gly
            out[at]+=dg-hkr;out[at+1]-=dg-hkr
            out[124 if j==0 else 149]=3*gly
            out[125 if j==0 else 150]+=hkr
        else:
            newH=0.;oldox=old[302]
            newox=max(oldox,0.)*normsat(p3,c[19],c[7])*c[17];oxid=newox/c[23]
            dp=3*gly+newox-old[288]
            out[282]+=dp-hkr;out[283]-=dp-hkr
            out[288]=3*gly+newox;out[301]=3*gly;out[302]=newox;out[289]+=hkr
        out[b+j]=(boundary-jt)/(w*c[10]);out[b+3+j]=(jt-hkr)/w
        out[gi]=(hkr-gly)/w
        out[b+8+j]=(2*gly-oxid-export)/w
        k=b+11+8*j;out[k:k+8]=np.array([boundary,jt,hkr,gly,oxid,export,hkr,newH])
        if j==1:
            # Original PPP states/counters retain their identity and history.
            oldsource=out[off+22];source=2*ppp
            out[off+9]+=source-oldsource;out[off+22]=source
            out[off+47:off+56]=np.array([(hkr-gly)/w-ppp+5./6.*ret,ppp-ret,hkr/w,gly/w,ppp,ppp,2*ppp,0.,ret])
            out[off+69]=ppp*32./6. # retained diagnostic only, NEVER ATP hydrolysis
    return out

def check_view(z,base,off,v):return A.check_view(z,base,off,v)
def somatic_check_view(z,base,off):return A.somatic_check_view(z,base,off)

def checks(z,base,off,context):
    ref,q,h0,share,nq,cq,v=context
    out=A.checks(z,base,off,(ref,q,h0,share,nq,v))
    if cq[0,24]==0:return out
    d=z-base;b=off+A.NS;err=lambda x:float(np.max(abs(x)))
    for j,s in enumerate(SIDES):
        c=cq[j];w=c[0];gi=(b+6 if j==0 else off+47 if j==1 else b+7);k=b+11+8*j
        carbon=6*w*(c[10]*d[:,b+j]+d[:,b+3+j]+d[:,gi])+3*w*d[:,b+8+j]+3*(d[:,k+4]+d[:,k+5])-6*d[:,k]
        if j==1:carbon+=w*(5*d[:,off+48]+d[:,off+51])
        out['carbon_'+s+'_C_abs_uM']=err(carbon)
        out['carbon_'+s+'_HK_ATP']=err(d[:,k+6]-d[:,k+2])
        out['carbon_'+s+'_gly_ATP']=err(d[:,124 if j==0 else 149 if j==1 else 301]-3*d[:,k+3])
        out['carbon_'+s+'_C3']=err(w*d[:,b+8+j]-2*d[:,k+3]+d[:,k+4]+d[:,k+5])
        out['carbon_'+s+'_glucose']=err(w*d[:,b+3+j]-d[:,k+1]+d[:,k+2])
        if j<2:out['carbon_'+s+'_ETC_fuel']=err(d[:,249+10*j]-c[21]*d[:,k+4])
        else:out['carbon_terminal_OX_fuel']=err(d[:,302]-c[23]*d[:,k+4])
    out['carbon_nonnegative']=float(max(0.,-z[:,b:b+11].min()))
    out['carbon_other_NADPH_disabled']=err(d[:,off+54])
    return out

def observe(z,base,off,context):
    ref,q,h0,share,nq,cq,v=context
    out=A.observe(z,base,off,(ref,q,h0,share,nq,v));b=off+A.NS;d=z-base
    out['carbon']={s:dict(external_glucose_uM=z[:,b+j],glucose_uM=z[:,b+3+j],G6P_uM=z[:,b+6 if j==0 else off+47 if j==1 else b+7],C3_uM=z[:,b+8+j],
      glucose_input_reference_uM=d[:,b+11+8*j],HK_ATP_reference_uM=d[:,b+17+8*j],C3_oxidized_reference_uM=d[:,b+15+8*j]) for j,s in enumerate(SIDES)}
    out['scope']='Connected finite carbon/ATP/PPP scenario; prior opportunity counter is not charged, no duplicate ATP/NADPH; human calibration absent.'
    return out
