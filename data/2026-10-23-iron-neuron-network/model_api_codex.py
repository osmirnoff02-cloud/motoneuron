"""[codex] 2026-10-23. Iron handling coupled to the 83-state prototype.

Iron pools are AMOUNTS in one arbitrary common unit, never uM. Ferritin
capacity, protein, RNA, lipid oxidation and Fe amount are separate variables.
All new numerical values are assumptions, not fitted human MN kinetics.
"""
from pathlib import Path
from dataclasses import dataclass,asdict,fields
import importlib.util,sys,json,hashlib
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_before_iron_network',_HERE.parent/'2026-10-23-pkn-organelle-network/model_api_codex.py')
_prior=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_prior;_spec.loader.exec_module(_prior)
for _name in dir(_prior):
    if not _name.startswith('_') and _name not in globals():globals()[_name]=getattr(_prior,_name)
_network_rhs=_prior._network_rhs
IRON_NAMES=NETWORK_NAMES+('somatic_labile_Fe','somatic_ferritin_Fe','lysosomal_labile_Fe',
 'axonal_labile_Fe','axonal_ferritin_Fe','somatic_ferritin_capacity','axonal_FTH1_RNA_capacity',
 'axonal_ferritin_capacity','GPX4_protein_capacity','GSH_capacity','somatic_lipid_oxidation',
 'axonal_lipid_oxidation','lysosomal_lipid_oxidation','Fe_imported','Fe_exported',
 'Fe_loaded_into_ferritin','Fe_ferritin_delivered_to_lysosome','Fe_released_from_lysosome',
 'lipid_oxidation_formed','lipid_oxidation_repaired','lysosomal_ferritin_bound_Fe','endosomal_Fe')
IRON_DRIVERS=DRIVERS+('Fe_import_support','endosomal_export_support','lysosomal_Fe_export_support',
 'NCOA4_delivery_support','ferroportin_export_support','axonal_FTH1_RNA_support',
 'GPX4_catalytic_support','GSH_supply_support','oxidant_support')
FE_INDEX=np.array([83,84,85,86,87,103,104])

@dataclass(frozen=True)
class IronParameters:
    ferritin_turnover_s:float=86400.                 # 0
    lysosomal_digestion_s:float=21600.              # 1
    lysosomal_Fe_export_s:float=7200.               # 2
    endosomal_Fe_export_s:float=1800.               # 3
    external_Fe_turnover_s:float=259200.            # 4
    soma_axon_exchange_s:float=21600.               # 5
    axonal_FTH1_RNA_tau_s:float=21600.              # 6
    axonal_ferritin_tau_s:float=43200.              # 7
    somatic_ferritin_tau_s:float=43200.             # 8
    GPX4_tau_s:float=86400.                         # 9
    GSH_tau_s:float=21600.                         # 10
    oxidation_formation_per_s:float=1.1574074074074074e-7 # 11
    oxidation_repair_per_s:float=1.1574074074074073e-5    # 12
    transferrin_import_fraction:float=.8           # 13
    assumed_TDP_to_axonal_FTH1_gain:float=.5        # 14
    assumed_iron_to_ferritin_gain:float=.5          # 15
    assumed_oxidation_to_lysosome_gain:float=.5     # 16
    assumed_low_lysosome_GPX4_gain:float=0.         # 17
    feedback_enabled:bool=True                   # 18
    provenance:str='Topology informed by Jinno2025, Mancias2014, Dong2008, Cañeque2025 and Bonte2026. Every new numerical value is an explicit sensitivity assumption, not a measured human MN parameter.'
    def vector(self):
        a=np.array([getattr(self,f.name) for f in fields(self) if f.name!='provenance'],float)
        if not np.isfinite(a).all() or min(a[:13])<=0 or np.any(a[13:]<0) or np.any(a[[13,14,15]]>1) or not isinstance(self.feedback_enabled,bool) or not self.provenance.strip():raise ValueError('Positive time scales/rates, bounded fractions and provenance required')
        return a

def _iron_baseline(v):
    z=np.zeros(22)
    z[:5]=[1.,4.,4.8*v[2]/v[0],.2,.8]
    z[5:10]=1.
    z[10:13]=v[11]/v[12]
    z[20]=4.8*v[1]/v[0]
    z[21]=v[13]*v[3]/v[4]
    return z

@njit
def _iron_rhs(y,j,p,kv,sv,av,hv,block,nv,d,ag0,iv,q,base_n):
    out=np.zeros(105)
    d2=d.copy();lp0=iv[11]/iv[12]
    if iv[18]>0:
        # Optional membrane-oxidation feedback, NOT an established human dose curve.
        excess=max(y[95]/lp0-1.,0.)
        d2[5]=d[5]/(1.+iv[16]*excess)
    out[:83]=_network_rhs(y[:83],j,p,kv,sv,av,hv,block,nv,d2,ag0)
    c=max(y[83],0.);f=max(y[84],0.);l=max(y[85],0.)
    a=max(y[86],0.);fa=max(y[87],0.);bound=max(y[103],0.);endo=max(y[104],0.)
    lyso=max(y[72],0.)
    encounter=((1.-nv[21])+nv[21]*max(y[68],0.))*d[8]
    # Import via transferrin/endosome and a separate non-transferrin route.
    incoming=q[0]/iv[4];into_endo=iv[13]*incoming;direct=incoming-into_endo
    endo_out=q[1]*endo/iv[3]
    export=q[4]*c/iv[4]
    load_s=c*max(8.*y[88]-f,0.)/iv[0]
    load_a=5.*a*max(1.6*y[90]-fa,0.)/iv[0]
    # NCOA4 delivery is distinct from digestion and release of labile iron.
    delivery_s=q[3]*encounter*f/iv[0];delivery_a=q[3]*encounter*fa/iv[0]
    digestion=lyso*bound/iv[1];release=q[2]*l/iv[2]
    forward=.2*c/iv[5];backward=a/iv[5]
    out[83]=direct+endo_out+release-export-load_s-forward+backward
    out[84]=load_s-delivery_s
    out[85]=digestion-release
    out[86]=forward-backward-load_a
    # Ferritinophagy moves Fe to the lysosome, never destroys Fe.
    out[87]=load_a-delivery_a
    out[103]=delivery_s+delivery_a-digestion
    out[104]=into_endo-endo_out
    # Baseline requires a supply of axonal labile iron equal to local trapping;
    # return of the lysosomal pool is somatic, so use a directed recycling route.
    recycling=.8/iv[0]*c
    out[83]-=recycling;out[86]+=recycling
    n=max((y[6]+y[7])/base_n,0.)
    RNA_target=q[5]*((1.-iv[14])+iv[14]*min(n,1.))
    out[89]=(RNA_target-y[89])/iv[6]
    out[90]=(y[89]-y[90])/iv[7]
    out[88]=((1.-iv[15])+iv[15]*c-y[88])/iv[8]
    # Bonte is a context sensitivity, not a universal effect of lysosome failure.
    gpx_target=1.+iv[17]*max(1.-min(lyso,1.),0.)
    out[91]=(gpx_target-y[91])/iv[9]
    out[92]=(q[7]-y[92])/iv[10]
    repair_support=max(y[91]*y[92],0.)*q[6]
    fe=np.array([c,a/.2,l/(4.8*iv[2]/iv[0])])
    for i in range(3):
        formed=iv[11]*q[8]*fe[i]
        repaired=iv[12]*repair_support*max(y[93+i],0.)
        out[93+i]=formed-repaired;out[101]+=formed;out[102]+=repaired
    out[96]=incoming;out[97]=export;out[98]=load_s+load_a
    out[99]=delivery_s+delivery_a;out[100]=release
    return out

def calcium_iron_chain(segments,*,iron_parameters,checkpoint=None,output_step_s=1800.,
        max_step_s=1800.,rtol=1e-9,atol=1e-11,**kw):
    """22-column contiguous protocol. Iron units are normalized AMOUNTS.

    kw is exactly the existing calcium_organelle_chain parameter interface.
    Existing entry points remain reexported and unchanged. No death prediction,
    no iron/Ca equivalence, no direct conversion of RNA fold change into rates.
    """
    sg=np.asarray(segments,float)
    if sg.ndim!=2 or sg.shape[1]!=22 or not len(sg) or not np.isfinite(sg).all() or np.any(sg<0) or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[1:,0]!=sg[:-1,1]):raise ValueError('Contiguous finite nonnegative 22-column protocol required')
    if np.any(sg[:,[3,4,5,6,8,9,10,11,12]]>1):raise ValueError('Original fraction drivers must be <=1')
    if not np.isfinite([output_step_s,max_step_s,rtol,atol]).all() or min(output_step_s,max_step_s,rtol,atol)<=0:raise ValueError('Positive integration scales required')
    iv=iron_parameters.vector()
    # Obtain the authoritative 83-state baseline and validate all old parameters.
    initial=_prior.calcium_organelle_chain([[0.,1e-4,0.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.]],**kw)
    base=np.r_[initial['baseline_state'],_iron_baseline(iv)]
    nv=kw['organelle_parameters'].vector();av=kw['aggregation_parameters'].vector()
    hv=np.r_[kw['autoregulation_parameters'].vector(),kw['rna_feedback_parameters'].vector()]
    required={'basal_ca_uM','calcium_tau_s','enzyme_parameters','tdp_parameters','localization_parameters','stmn2_parameters','inactivation_parameters','phosphorylation_parameters','reactivation_parameters','stability_parameters'}
    _,p,kv,sv,_=_prior._agg._baseline({k:kw[k] for k in required},av)
    ag0=base[37:45].sum();base_n=base[6]+base[7]
    config=dict(initial['config'],topology='organelle83_iron22_v1',iron=asdict(iron_parameters),iron_units='common arbitrary amount; never concentration')
    fp=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if checkpoint is None:y=base.copy();origin=float(sg[0,0])
    else:
        if checkpoint['fingerprint']!=fp or checkpoint['time_s']!=sg[0,0] or not np.array_equal(checkpoint['baseline_state'],base):raise ValueError('Checkpoint time, parameters or baseline mismatch')
        y=np.asarray(checkpoint['state'],float).copy();origin=checkpoint['origin_time_s']
        if y.shape!=(105,) or not np.isfinite(y).all() or y.min()<-1e-7:raise ValueError('Invalid checkpoint')
    ts=[sg[0,:1].copy()];zs=[y[None,:]];nfev=0
    for row in sg:
        start,end,j,block=row[:4];d=row[4:13];q=row[13:]
        near=start+np.geomspace(1e-4,min(end-start,3600.),70)
        grid=np.unique(np.r_[np.arange(start+output_step_s,end,output_step_s),near[near<end],end])
        sol=solve_ivp(lambda t,z:_iron_rhs(z,j,p,kv,sv,av,hv,block,nv,d,ag0,iv,q,base_n),(start,end),y,t_eval=grid,method='LSODA',rtol=rtol,atol=atol,max_step=max_step_s)
        if not sol.success:raise RuntimeError(sol.message)
        y=sol.y[:,-1].copy();ts.append(sol.t);zs.append(sol.y.T);nfev+=sol.nfev
    t=np.concatenate(ts);z=np.concatenate(zs);ag=z[:,37:45].sum(1)
    idx=np.clip(np.searchsorted(sg[:,0],t,side='right')-1,0,len(sg)-1)
    enzyme=z[:,1:6].sum(1)+z[:,31]+z[:,33];enzyme0=base[1:6].sum()+base[31]+base[33]
    td=z[:,6:10].sum(1)+ag;td0=base[6:10].sum()+ag0
    err=lambda x:float(abs(x).max())
    checks=dict(enzyme_balance=err(enzyme+z[:,15]-z[:,14]-enzyme0),TDP_lineage_balance=err(td+z[:,19]-z[:,18]-td0),
        aggregate_balance=err(ag-ag0-z[:,45]+z[:,46]+z[:,47]+z[:,49]),calcium_balance_uM=err(z[:,0]+z[:,23]-z[:,22]),calcium_AUC_identity=err(z[:,24]-p[1]*z[:,23]),
        PKN1_AF3_balance=err(z[:,63]+z[:,66]+z[:,76]-z[:,75]-1),PKN207_balance=err(z[:,65]+z[:,78]-z[:,77]),
        secretory_balance=err(z[:,69:72].sum(1)+z[:,80]-z[:,79]-base[69:72].sum()),autophagic_balance=err(z[:,73:75].sum(1)+z[:,82]-z[:,81]-base[73:75].sum()),
        iron_mass_balance=err(z[:,FE_INDEX].sum(1)+z[:,97]-z[:,96]-base[FE_INDEX].sum()),
        ferritin_bound_balance=err(z[:,84]+z[:,87]-base[84]-base[87]-z[:,98]+z[:,99]),
        lipid_oxidation_balance=err(z[:,93:96].sum(1)-base[93:96].sum()-z[:,101]+z[:,102]),
        minimum_state=float(min(z.min(),(z[:,1]-z[:,28]).min(),(z[:,2]-z[:,29]).min(),(z[:,31]-z[:,32]).min())),
        tracer_excess=float(max(0.,np.max(z[:,51:63]-z[:,_prior._prior.LABEL_INDEX]))))
    if not np.isfinite(z).all() or checks['minimum_state']<-1e-7 or max(v for k,v in checks.items() if k!='minimum_state')>5e-5:raise ArithmeticError(str(checks))
    cp=dict(time_s=float(t[-1]),state=y.tolist(),baseline_state=base.tolist(),origin_time_s=float(origin),fingerprint=fp)
    return dict(time_s=t,state=z,columns=IRON_NAMES,calcium_uM=p[0]+z[:,0],activity=(z[:,28]+z[:,29]+z[:,32])*(1-sg[idx,3]),
        aggregate_total=ag,soluble_intact_TDP=z[:,6]+z[:,7],total_intact_TDP=z[:,6]+z[:,7]+z[:,37]+z[:,38]+z[:,41]+z[:,42],nuclear_function_proxy=z[:,6]+z[:,7],
        synthesis_capacity=z[:,50],synthesis_per_s=p[20]*z[:,50]*(1.-nv[22]*(1.-np.minimum(z[:,68],1.)) if nv[25]>0 else 1.),
        iron_total=z[:,FE_INDEX].sum(1),iron_labile=z[:,[83,85,86]].sum(1),
        baseline_state=base,checkpoint=cp,config=config,checks=checks,segments=sg,nfev=nfev,provenance=kw['provenance'],
        human_kinetic_fit=False,electrical_feedback=False,absolute_TDP_per_cell=None,ferroptotic_death_probability=None,
        limitations=initial['limitations']+['Seven Fe pools use a common arbitrary amount; no absolute human baseline or compartment volumes',
        'Ferritin delivery, lysosomal digestion and labile Fe export are independent; Fe is conserved, not consumed by oxidation',
        'Jinno axonal FTH1 evidence is mouse cortical neurons, not human MN. Axonal return route is an effective transport assumption',
        'Lipid oxidation is a relative burden, not a calibrated injury/death probability; other antioxidants and detailed redox chemistry absent',
        'Low lysosome to GPX4 stabilization is optional: Bonte LUHMES long-treatment context, not general neuronal rule',
        'TDP to axonal ferritin and oxidation to lysosome gains require independent human measurements',
        'No mitochondria/heme/Fe-S/ATP, systemic hepcidin regulation or glial iron exchange; no new direct iron to electrical feedback'])

def pulse_chase_readout(result,**kw):
    if np.asarray(result['state']).shape[1]==105:
        result=dict(result,state=result['state'][:,:83],baseline_state=result['baseline_state'][:83])
    return _prior.pulse_chase_readout(result,**kw)
