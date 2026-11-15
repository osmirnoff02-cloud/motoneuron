"""Codex 2026-11-15. Conservative opt-in extension of shared redox.

AA delivery is a declared open nutrient boundary, not inferred from RNA.
GSH export is distinguished from irreversible destruction and GS-HNE export.
Somatic AA remain a coarse pool: equal C/E/G availability is an assumption.
PPP replaces the prior NADPH source, never adds a second source. Carbon and
forgone ATP are reported; the host has no common glucose pool yet, so forgone
ATP is NOT falsely charged as ATP hydrolysis. This boundary remains open.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import sys
import numpy as np
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'neuron-model'))
from model_contracts import load_module
B=Path(__file__).resolve().parent;D=B.parent
S=load_module(D/'2026-11-15-shared-redox-budget/shared_budget.py','_stable_shared_redox15')
N=load_module(D/'2026-11-14-nadph-supply/nadph_supply.py','_agn15_nadph')
L=S.L;P=S.P
# 47..55 PPP; 56..58 new physical GSH/precursor compartments.
EXTRA=('soma_GSH_reference_uM','exported_GSH_reference_uM','recoverable_AA_triplets_reference_uM',
       'soma_GSH_synthesized','soma_GSH_exported','GSH_soma_to_axon_net',
       'exported_GSH_hydrolyzed','AA_triplets_returned','GSH_irrecoverable',
       'local_AA_each_external_net','soma_AA_each_external_net','extra_axon_ATP_spent','extra_soma_ATP_spent',
       'PPP_ATP_opportunity_only','AA_triplets_returned_axon','AA_triplets_returned_soma')
NAMES=S.NAMES+tuple('shared_supply_'+n for n in N.NAMES+EXTRA)
NS=len(NAMES)
RANGES=dict(AA_delivery_capacity_uM_s=[0.,.1,.3],AA_boundary_uM=[10.,50.,200.],
           GSH_recovery_fraction=[0.,.5,.9],GSH_hydrolysis_tau_s=[60.,600.,3600.],
           AA_return_tau_s=[60.,600.,3600.],AA_recovery_capacity_uM_s=[0.,.1,1.],GSH_delivery_tau_s=[600.,3600.,36000.],
           soma_GSH_synthesis_capacity_uM_s=[.03,.2,.6],soma_GSH_initial_uM=[1000.,2000.,5000.],
           soma_GSH_export_tau_s=[3600.,21600.,86400.],AA_delivery_ATP_equiv=[0.,1.,2.],
           G6PD_Ki_NADPH_uM=[9.,18.,36.],PPP_reserve_fold=[11.,30.,89.])
RANGES.update(GSH_loss_multiplier=[.5,1.,2.],GSH_synthesis_multiplier=[0.,.5,1.,3.])

@dataclass(frozen=True)
class SupplyParameters(S.SharedRedoxParameters):
    provenance:str='Conditional AA boundary/GSH recycling and finite somatic delivery; human purified G6PD Ki anchor, no human MN calibration; external glucose and other NADPH donors remain open.'
    AA_delivery_capacity_uM_s:float=.1
    AA_boundary_uM:float=50.
    GSH_recovery_fraction:float=.5
    GSH_hydrolysis_tau_s:float=600.
    AA_return_tau_s:float=600.
    AA_recovery_capacity_uM_s:float=.1
    GSH_delivery_tau_s:float=3600.
    soma_GSH_synthesis_capacity_uM_s:float=.2
    soma_GSH_initial_uM:float=2000.
    soma_GSH_export_tau_s:float=21600.
    AA_delivery_ATP_equiv:float=1.
    G6PD_Ki_NADPH_uM:float=9.
    PPP_reserve_fold:float=30.
    explicit_NADPH:bool=True
    GSH_loss_multiplier:float=1.
    GSH_synthesis_multiplier:float=1.
    fixed_local_AA_influx_uM_s:float|None=None
    fixed_soma_AA_influx_uM_s:float|None=None
    def validate(self):
        super().validate()
        vals=[v for k,v in asdict(self).items() if isinstance(v,(int,float)) and not isinstance(v,bool)]
        if not np.isfinite(vals).all() or min(vals)<0 or not 0<=self.GSH_recovery_fraction<=1:raise ValueError('Invalid supply range')
        for k in ('AA_boundary_uM','GSH_hydrolysis_tau_s','AA_return_tau_s','GSH_delivery_tau_s','soma_GSH_initial_uM','soma_GSH_export_tau_s','G6PD_Ki_NADPH_uM'):
            if getattr(self,k)<=0:raise ValueError(k+' must be positive')
        if self.PPP_reserve_fold<=1 or not isinstance(self.explicit_NADPH,bool):raise ValueError('Invalid PPP options')
        if self.GSH_loss_multiplier<=0:raise ValueError('Positive loss multiplier required')
        if (self.fixed_local_AA_influx_uM_s is None)!=(self.fixed_soma_AA_influx_uM_s is None):
            raise ValueError('Fixed-flux experiment requires both compartments')

def prepare(base,p,water_ratio,soma_process_ratio=1.,soma_water_ratio=1.):
    p.validate();y,ctx,report=S.prepare(base,p,water_ratio)
    ref,q,h0,share,v=ctx
    np_=N.NADPHSupplyParameters(G6PD_Ki_NADPH_uM=p.G6PD_Ki_NADPH_uM,PPP_reserve_fold=p.PPP_reserve_fold,
        provenance='Codex adapter: human recombinant Ki9uM (Wang2002); human recombinant Km (Hernandez-Ochoa2025), different Mg conditions. Rat-derived reserve scenario; glucose/other donor supply and enzyme amount are unknown. No human MN calibration.')
    # Demand includes thioredoxin regeneration. Old external supply is removed in RHS.
    rd=S.redox_rhs(y[len(base):],base[136],base[S.AA_INDICES],1.,1.,1.,ref,q,h0,share,v)
    demand=rd[23]
    ny,nq,nreport=N.prepare(np_,demand,ref[L.R_NTOT])
    # Independent interventions AFTER the original reference has been prepared.
    # Never solve synthesis again from the perturbed loss (that would hide drift).
    ref=ref.copy()
    ref[L.R_KLOSS]*=p.GSH_loss_multiplier
    ref[L.R_VSYN]*=p.GSH_synthesis_multiplier
    # Physical source capacities fixed at construction; no re-fit during loads.
    extras=np.zeros(len(EXTRA));extras[0]=p.soma_GSH_initial_uM*soma_water_ratio
    z=np.r_[y,ny,extras]
    vv=np.r_[v,soma_process_ratio,soma_water_ratio,p.AA_delivery_capacity_uM_s,p.AA_boundary_uM,
             p.GSH_recovery_fraction,p.GSH_hydrolysis_tau_s,p.AA_return_tau_s,p.GSH_delivery_tau_s,
             p.soma_GSH_synthesis_capacity_uM_s*p.GSH_synthesis_multiplier,
             p.soma_GSH_export_tau_s/p.GSH_loss_multiplier,p.AA_delivery_ATP_equiv,
             float(p.explicit_NADPH),np_.ATP_per_glucose,p.AA_recovery_capacity_uM_s,
             float(p.fixed_local_AA_influx_uM_s is not None),
             p.fixed_local_AA_influx_uM_s or 0.,p.fixed_soma_AA_influx_uM_s or 0.]
    report.update(count=NS,schema_key='shared_supply15',physical_states=23,bookkeeping_states=NS-23,
        supply_parameters=asdict(p),NADPH_parameters=asdict(np_),NADPH_reference=nq.tolist(),NADPH_preparation=nreport,
        resource_vector=vv.tolist(),ownership='Host ATP/AA; GTP unchanged directly. Finite soma GSH; explicit external AA boundary and export/recovery. PPP replaces old NADPH source. Glucose input and other NADPH donors remain external; opportunity cost diagnostic only.',
        initialization='Host and original redox pools preserved; PPP/soma/export pools declared at connection, not pre-equilibrated whole cell')
    return z,(ref,q,h0,share,nq,vv),report

@njit
def finish(z,old,off,ref,q,h0,share,nq,v,controls):
    out=np.zeros(len(z));out[:off+47]=S.finish(z[:off+47],old[:off+47],off,ref,q,h0,share,v[:7],controls)
    if v[5]==0:return out
    y=z[off:];ratio=v[7];ws=v[8];scale=v[4]
    # Explicit PPP supply supersedes, rather than augments, the former source.
    if v[18]:
        u=np.array([max(y[9],0.),1.,controls[2],controls[2]])
        nd=N.rhs(y[47:56],u,nq);out[off+47:off+56]=nd
        supply=nd[6]+nd[7]
        old_supply=out[off+22]
        out[off+9]+=supply-old_supply;out[off+22]=supply
        out[off+69]=nd[4]*v[19]/6.  # diagnostic, NOT hydrolysis
    ea=max(z[136],0.)/(500.+max(z[136],0.))
    es=max(z[120],0.)/(500.+max(z[120],0.))
    # Supplemental 1:1:1 nutrient packets from an explicit maintained boundary.
    # Limiting inward gradient gates the packet; not an identified transporter.
    local=np.zeros(3)
    for j in range(3):
        a=max(z[S.AA_INDICES[j]],0.)/v[3]
        local[j]=v[9]*max(v[10]-a,0.)/(v[10]+a)*scale*ea
        out[S.AA_INDICES[j]]+=local[j]
    # Ledger one per species (3 distinct states appended below would be required
    # if their gradients differ). Use the limiting precursor availability to
    # define a fixed 1:1:1 delivery packet, avoiding fictitious species equality.
    common=min(local[0],min(local[1],local[2]))
    # Diagnostic open-boundary experiment: exactly the same prescribed input
    # in all variants, not a fitted transporter or a fixed-medium assay. ATP
    # is still charged once below. Runs exhausting ATP must be rejected; this
    # mode deliberately does not compensate via a substrate/energy gate.
    if len(v)>21 and v[21]:common=v[22]*scale
    for j in range(3):out[S.AA_INDICES[j]]+=common-local[j]
    out[574]+=3*common;out[off+65]=common
    sa=max(z[497],0.)/(20.*ws)
    sin=v[9]*max(v[10]-sa,0.)/(v[10]+sa)*ws*es
    if len(v)>21 and v[21]:sin=v[23]*ws
    out[497]+=3*sin;out[off+66]=sin
    # Somatic GSH synthesis uses the same finite aggregate amino-acid/ATP pool.
    sg=max(y[56],0.);cs=sg/ws;cg=max(y[7],0.)
    syn=v[15]*es*sa/(v[2]+sa)/(1.+cs/ref[L.R_KI])*ws
    loss=sg/v[16]
    # Intracellular passive exchange, no claimed vesicle/motor ATP requirement.
    transport=scale*(cs-cg)/v[14]
    out[off+56]=syn-loss-transport/ratio
    out[off+7]+=transport/scale
    out[off+59]=syn;out[off+60]=loss;out[off+61]=transport
    out[497]-=3*syn
    local_loss=ref[L.R_KLOSS]*cg*scale
    hydro=max(y[57],0.)/v[12]
    # Shared extracellular precursor compartment, with declared volume equal to
    # combined active soma/axon water. Uptake has finite capacity and inward
    # concentration gradient; it cannot all be funnelled into the tiny axon.
    precursor=max(y[58],0.)/(ratio*ws+scale)
    caa=0.
    for j in range(3):caa=max(caa,max(z[S.AA_INDICES[j]],0.)/v[3])
    axon_gate=max(precursor-caa,0.)/(v[2]+precursor+caa)
    soma_gate=max(precursor-sa,0.)/(v[2]+precursor+sa)
    recovery_a=v[20]*scale*axon_gate*ea
    recovery_s=v[20]*ws*soma_gate*es  # soma-reference amount
    recovery=recovery_a+ratio*recovery_s
    ceiling=max(y[58],0.)/v[13]
    fac=min(1.,ceiling/max(recovery,1e-30))
    recovery_a*=fac;recovery_s*=fac;recovery*=fac
    out[off+57]=local_loss+ratio*loss-hydro
    out[off+58]=v[11]*hydro-recovery
    out[off+62]=hydro;out[off+63]=recovery;out[off+64]=(1-v[11])*hydro
    for j in range(3):out[S.AA_INDICES[j]]+=recovery_a
    out[574]+=3*recovery_a;out[497]+=3*recovery_s
    # Reuse cumulative ATP ledgers to recover each uptake amount exactly.
    # Dedicated per-compartment recovery ledgers appended below for zero-cost cases.
    out[off+70]=recovery_a;out[off+71]=recovery_s
    ca=v[17]*(3*max(common,0.)+3*recovery_a)
    cs_cost=2*syn+v[17]*3*(max(sin,0.)+recovery_s)
    out[136]-=ca;out[137]+=ca;out[150]+=ca;out[off+67]=ca
    out[120]-=cs_cost;out[121]+=cs_cost;out[125]+=cs_cost;out[off+68]=cs_cost
    return out

def check_view(z,base,off,v):
    zz,bb=S.check_view(z,base,off,v[:7])
    for a in (zz,bb):
        local=a[...,off+70]+a[...,off+65]
        for i in S.AA_INDICES:a[...,i]-=local
        a[...,574]-=3*local
        a[...,497]+=3*(a[...,off+59]-a[...,off+66]-a[...,off+71])
    return zz,bb

def somatic_check_view(z,base,off):
    zz=z.copy();bb=base.copy()
    for a in (zz,bb):a[...,497]+=3*(a[...,off+59]-a[...,off+66]-a[...,off+71])
    return zz,bb

def checks(z,base,off,context):
    ref,q,h0,share,nq,v=context;d=z[:,off:]-base[off:]
    adjusted=z.copy();b=base.copy()
    adjusted[:,off+7]-=z[:,off+61]/v[4];b[off+7]-=base[off+61]/v[4]
    out=S.checks(adjusted,b,off,(ref,q,h0,share,v[:7]))
    # Adjusted bookkeeping GSH is not a physical pool. Test physical pools only.
    out['shared_redox_nonnegative']=float(max(0.,-z[:,off:off+47].min()))
    out['shared_redox_lipid_negative_pool']=float(max(0.,-z[:,off:off+11].min()))
    err=lambda x:float(np.max(abs(x)))
    out.update(supply_soma_GSH=err(d[:,56]-d[:,59]+d[:,60]+d[:,61]/v[7]),
        supply_exported_GSH=err(d[:,57]-v[4]*(d[:,21]-d[:,16])-v[7]*d[:,60]+d[:,62]),
        supply_AA_recovery=err(d[:,58]-v[11]*d[:,62]+d[:,63]),
        supply_irrecoverable=err(d[:,64]-(1-v[11])*d[:,62]),
        supply_physical_nonnegative=float(max(0.,-z[:,off+47:off+49].min(),-z[:,off+56:off+59].min())),
        supply_global_GSH_material=err(v[7]*d[:,56]+v[4]*(d[:,7]+2*d[:,8]+d[:,45]+d[:,46])+d[:,57]+d[:,58]+d[:,64]-v[4]*d[:,20]-v[7]*d[:,59]+d[:,63]),
        supply_recovery_partition=err(d[:,63]-d[:,70]-v[7]*d[:,71]),
        supply_extra_axon_ATP=err(d[:,67]-3*v[17]*(d[:,65]+d[:,70])),
        supply_extra_soma_ATP=err(d[:,68]-2*d[:,59]-3*v[17]*(d[:,66]+d[:,71])))
    out.update({'supply_'+k:float(val) for k,val in N.checks(z[:,off+47:off+56],base[off+47:off+56]).items()})
    return out

def observe(z,base,off,context):
    v=context[-1];d=z[:,off:]-base[off:]
    out=S.observe(z,base,off,context)
    out.update(extra_ATP_axon_reference_uM=d[:,67],extra_ATP_soma_reference_uM=d[:,68],
        AA_returned_total_each_axon_reference_uM=d[:,63],AA_returned_axon_each_reference_uM=d[:,70],AA_returned_soma_each_reference_uM=d[:,71],AA_delivered_each_reference_uM=d[:,65],
        GSH_delivery_net_reference_uM=d[:,61],NADPH_from_PPP_uM=d[:,53],NADPH_from_other_uM=d[:,54],
        PPP_opportunity_ATP_uM=d[:,69],scope='Finite GSH recycling/delivery and common resources; external AA/glucose/donor boundaries declared; forgone glucose ATP is not hydrolysis or calibrated host energy coupling')
    return out
