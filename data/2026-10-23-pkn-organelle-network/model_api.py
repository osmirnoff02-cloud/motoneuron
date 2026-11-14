"""[codex] 2026-10-23. Conditional PKN / organelle network, NOT a human MN fit.

Prior 51 biochemical and 12 passive tracer states are retained. Twenty new
functional/accounting states describe processes, not complete organelles.
Secretory/autophagic cargo units are independent of TDP N-lineage units.
"""
from pathlib import Path
from dataclasses import dataclass,asdict,fields
import importlib.util,sys,json,hashlib
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_before_organelle_network',_HERE.parent/'2026-10-23-ling-ayala-supplement/model_api.py')
_prior=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_prior;_spec.loader.exec_module(_prior)
for _name in dir(_prior):
    if not _name.startswith('_') and _name not in globals():globals()[_name]=getattr(_prior,_name)
_old_rhs=_prior._pulse_chase_rhs
_agg=_prior._agg
NETWORK_NAMES=TRACER_NAMES+('PKN1_full_length','PKN_cryptic_RNA_fraction_proxy','PKN207','PKN_AF3',
    'neurofilament_transport_capacity','axonal_lysosome_position_capacity',
    'ER_secretory_cargo','ERGIC_secretory_cargo','Golgi_secretory_cargo','lysosome_degradation_capacity',
    'autophagosome_generic_cargo','autolysosome_generic_cargo',
    'PKN1_synthesized','PKN1_AF3_removed','PKN207_synthesized','PKN207_removed',
    'secretory_cargo_input','secretory_cargo_delivered','autophagic_cargo_input','autophagic_cargo_degraded')
DRIVERS=('start_s','end_s','Ca_flux_uM_s','calpain_block','external_caspase_activity',
    'selective_PKN207_production_block','PKN_cleavage_block','Rab1_support','BORC_support',
    'lysosome_activity_support','proteasome_support','global_PKN_synthesis_support','fusion_support')

@dataclass(frozen=True)
class OrganelleParameters:
    cryptic_function_threshold:float=.7
    maximum_cryptic_fraction:float=.5
    cryptic_RNA_tau_s:float=21600.
    PKN1_half_life_s:float=86400.
    PKN207_half_life_s:float=604800.
    AF3_half_life_s:float=21600.
    PKN_cleavage_per_s:float=3.209e-5
    AF3_neurofilament_gain:float=2.
    neurofilament_tau_s:float=21600.
    assumed_NF_to_lysosome_gain:float=0.
    lysosome_position_tau_s:float=21600.
    assumed_TDP_loss_to_ER_gain:float=0.
    ER_transit_s:float=3600.
    ERGIC_transit_s:float=1800.
    Golgi_transit_s:float=3600.
    lysosome_activity_tau_s:float=43200.
    assumed_Golgi_to_lysosome_gain:float=0.
    autophagosome_transit_s:float=14400.
    autolysosome_transit_s:float=28800.
    soluble_TDP_lysosomal_fraction:float=.5
    aggregate_TDP_lysosomal_fraction:float=.5
    assumed_position_clearance_gain:float=.5
    assumed_position_translation_gain:float=0.
    assumed_PKN207_to_ER_gain:float=0.
    assumed_aggregate_cargo_gain:float=1.
    feedback_enabled:bool=True
    provenance:str='Source-informed topology; ALL new numerical rates, thresholds, fractions and gains are sensitivity assumptions, not human motor-neuron measurements.'
    def vector(self):
        names=[f.name for f in fields(self) if f.name!='provenance']
        a=np.array([getattr(self,n) for n in names],float)
        if not np.isfinite(a).all() or np.any(a<0) or not self.provenance.strip() or not isinstance(self.feedback_enabled,bool):raise ValueError('Finite nonnegative parameters and explicit provenance required')
        if not 0<a[0]<1 or any(a[i]>1 for i in [1,16,19,20,21,22]) or min(a[[2,3,4,5,8,10,12,13,14,15,17,18]])<=0:raise ValueError('Invalid fraction, threshold or positive time scale')
        return a

@njit
def _network_rhs(y,j,p,kv,sv,av,hv,block,nv,d,base_aggregate):
    # d: caspase, selective207 block, cleavage block, Rab1, BORC, lyso, UPS, globalPKN, fusion.
    pos=max(y[68],0.);lyso=max(y[72],0.)
    encounter=((1.-nv[21])+nv[21]*pos)*d[8]
    pp=p.copy();aa=av.copy()
    if nv[25]>0:
        soluble=(1.-nv[19])*d[6]+nv[19]*encounter*lyso
        aggregate=(1.-nv[20])*d[6]+nv[20]*encounter*lyso
        pp[14:17]*=soluble;aa[4:12]*=aggregate
        pp[20]*=1.-nv[22]*(1.-min(pos,1.))
    out=np.zeros(83);out[:63]=_old_rhs(y[:63],j,pp,kv,sv,aa,hv,block)
    n=max(y[6]+y[7],0.);loss=max(1.-n,0.)
    target=nv[1]*max(1.-n/nv[0],0.)
    out[64]=(target-y[64])/nv[2]
    prod=np.log(2.)/nv[3]*d[7]
    cleavage=nv[6]*d[0]*(1.-d[2])*y[63]
    dfl=np.log(2.)/nv[3]*y[63];daf=np.log(2.)/nv[5]*y[66]
    p207=prod*y[64]*(1.-d[1]);d207=np.log(2.)/nv[4]*y[65]
    out[63]=prod*(1.-y[64])-cleavage-dfl
    out[65]=p207-d207;out[66]=cleavage-daf
    out[75]=prod*(1.-y[64]);out[76]=dfl+daf;out[77]=p207;out[78]=d207
    out[67]=(1./(1.+nv[7]*max(y[66],0.))-y[67])/nv[8]
    # NF and lysosome transport are distinct; their connection has zero as a tested option.
    pos_target=d[4]/(1.+nv[9]*max(1.-y[67],0.))
    out[68]=(pos_target-y[68])/nv[10]
    damage=nv[11]*loss+nv[23]*max(y[65],0.)
    er_eff=min(1.,d[3]/(1.+damage))
    incoming=1./nv[12];er_to_ergic=er_eff*y[69]/nv[12]
    ergic_to_golgi=y[70]/nv[13];delivered=y[71]/nv[14]
    out[69]=incoming-er_to_ergic;out[70]=er_to_ergic-ergic_to_golgi
    out[71]=ergic_to_golgi-delivered;out[79]=incoming;out[80]=delivered
    provision=(1.-nv[16])+nv[16]*min(max(delivered/incoming,0.),1.)
    out[72]=(d[5]*provision-y[72])/nv[15]
    # Generic cargo: abundance is not flux. It is NOT a second counted TDP pool.
    excess=max(y[37:45].sum()-base_aggregate,0.)
    cargo_in=(1.+nv[24]*excess/(1.+excess))/nv[17]
    fusion=encounter*y[73]/nv[17];degraded=lyso*y[74]/nv[18]
    out[73]=cargo_in-fusion;out[74]=fusion-degraded;out[81]=cargo_in;out[82]=degraded
    return out

def calcium_organelle_chain(segments,*,organelle_parameters,autoregulation_parameters,aggregation_parameters,
        rna_feedback_parameters,provenance,checkpoint=None,output_step_s=1800.,rtol=1e-9,atol=1e-11,
        max_step_s=1800.,pulse_duration_s=1800.,label_fraction=1.,**kw):
    """Contiguous 13-column rows named in DRIVERS. External caspase is NOT Ca.

    All histories and passive pulse labels continue through pauses/resume. New
    untreated capacities are 1 and exactly preserve prior biochemical baseline.
    Labels are initialized during a finite pulse at that untreated equilibrium.
    No production of label during the subsequent chase. No neuronal refit.
    """
    required={'basal_ca_uM','calcium_tau_s','enzyme_parameters','tdp_parameters','localization_parameters','stmn2_parameters','inactivation_parameters','phosphorylation_parameters','reactivation_parameters','stability_parameters'}
    if set(kw)!=required:raise ValueError('Unexpected/missing fields: '+str(set(kw)^required))
    sg=np.asarray(segments,float)
    if sg.ndim!=2 or sg.shape[1]!=13 or not len(sg) or not np.isfinite(sg).all() or np.any(sg<0) or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[1:,0]!=sg[:-1,1]):raise ValueError('Contiguous 13-column nonnegative finite protocol required')
    if np.any(sg[:,[3,4,5,6,8,9,10,11,12]]>1):raise ValueError('All fraction drivers except Rab1 must be <=1')
    if not np.isfinite([output_step_s,rtol,atol,max_step_s,pulse_duration_s,label_fraction]).all() or min(output_step_s,rtol,atol,max_step_s,pulse_duration_s,label_fraction)<=0 or label_fraction>1 or not isinstance(provenance,str) or not provenance.strip():raise ValueError('Positive scales and provenance required')
    nv=organelle_parameters.vector();av=aggregation_parameters.vector()
    if av[14]!=1:raise ValueError('Passive tracer currently requires first-order aggregation')
    hv=np.r_[autoregulation_parameters.vector(),rna_feedback_parameters.vector()]
    if hv[5]>0 and hv[1]==0:raise ValueError('Positive feedback floor required for regulated decay')
    b,p,kv,sv,a0=_agg._baseline(kw,av);b51=np.r_[b,1.]
    label=_prior._equilibrium_pulse(b51,p,kv,sv,av,pulse_duration_s,label_fraction)
    extra=np.zeros(20);extra[[0,4,5,6,9,10]]=1.
    extra[7]=nv[13]/nv[12];extra[8]=nv[14]/nv[12];extra[11]=nv[18]/nv[17]
    base=np.r_[b51,label,extra];ag0=b[37:45].sum()
    config={k:asdict(v) if hasattr(v,'__dataclass_fields__') else v for k,v in kw.items()}
    config.update(topology='RNA51_tracer12_organelle20_v1',organelle=asdict(organelle_parameters),aggregation=asdict(aggregation_parameters),autoregulation=asdict(autoregulation_parameters),rna_feedback=asdict(rna_feedback_parameters),pulse_duration_s=pulse_duration_s,label_fraction=label_fraction)
    fp=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if checkpoint is None:y=base.copy();origin=float(sg[0,0])
    else:
        if checkpoint['fingerprint']!=fp or checkpoint['time_s']!=sg[0,0] or not np.array_equal(checkpoint['baseline_state'],base):raise ValueError('Checkpoint time, parameters or baseline changed')
        y=np.asarray(checkpoint['state'],float).copy();origin=checkpoint['origin_time_s']
        if y.shape!=(83,) or not np.isfinite(y).all() or y.min()<-1e-7:raise ValueError('Invalid checkpoint state')
    times=[sg[0,:1].copy()];states=[y[None,:]];nfev=0
    for row in sg:
        start,end,j,block=row[:4];drivers=row[4:]
        near=start+np.geomspace(1e-4,min(end-start,3600.),70)
        grid=np.unique(np.r_[np.arange(start+output_step_s,end,output_step_s),near[near<end],end])
        sol=solve_ivp(lambda t,z:_network_rhs(z,j,p,kv,sv,av,hv,block,nv,drivers,ag0),(start,end),y,t_eval=grid,method='LSODA',rtol=rtol,atol=atol,max_step=max_step_s)
        if not sol.success:raise RuntimeError(sol.message)
        y=sol.y[:,-1].copy();times.append(sol.t);states.append(sol.y.T);nfev+=sol.nfev
    t=np.concatenate(times);z=np.concatenate(states);ag=z[:,37:45].sum(1)
    idx=np.clip(np.searchsorted(sg[:,0],t,side='right')-1,0,len(sg)-1);blocks=sg[idx,3]
    enzyme=z[:,1:6].sum(1)+z[:,31]+z[:,33];enzyme0=base[1:6].sum()+base[31]+base[33]
    td=z[:,6:10].sum(1)+ag;td0=base[6:10].sum()+ag0
    checks=dict(enzyme_balance=float(abs(enzyme+z[:,15]-z[:,14]-enzyme0).max()),
        TDP_lineage_balance=float(abs(td+z[:,19]-z[:,18]-td0).max()),
        aggregate_balance=float(abs(ag-ag0-z[:,45]+z[:,46]+z[:,47]+z[:,49]).max()),
        calcium_balance_uM=float(abs(z[:,0]+z[:,23]-z[:,22]).max()),calcium_AUC_identity=float(abs(z[:,24]-p[1]*z[:,23]).max()),
        PKN1_AF3_balance=float(abs(z[:,63]+z[:,66]+z[:,76]-z[:,75]-1.).max()),
        PKN207_balance=float(abs(z[:,65]+z[:,78]-z[:,77]).max()),
        secretory_cargo_balance=float(abs(z[:,69:72].sum(1)+z[:,80]-z[:,79]-base[69:72].sum()).max()),
        autophagic_cargo_balance=float(abs(z[:,73:75].sum(1)+z[:,82]-z[:,81]-base[73:75].sum()).max()),
        minimum_state=float(min(z.min(),(z[:,1]-z[:,28]).min(),(z[:,2]-z[:,29]).min(),(z[:,31]-z[:,32]).min())),
        tracer_excess=float(max(0.,np.max(z[:,51:63]-z[:,_prior.LABEL_INDEX]))))
    if not np.isfinite(z).all() or checks['minimum_state']<-1e-7 or max(v for k,v in checks.items() if k!='minimum_state')>5e-5:raise ArithmeticError(str(checks))
    if z[:,64].max()>nv[1]+1e-7 or z[:,50].min()<hv[1]-1e-7 or z[:,50].max()>hv[2]+1e-7:raise ArithmeticError('Regulated fraction escaped bounds')
    cp=dict(time_s=float(t[-1]),state=y.tolist(),baseline_state=base.tolist(),origin_time_s=float(origin),fingerprint=fp)
    encounter=(1.-nv[21]+nv[21]*z[:,68])*sg[idx,12]
    return dict(time_s=t,state=z,columns=NETWORK_NAMES,calcium_uM=p[0]+z[:,0],activity=(z[:,28]+z[:,29]+z[:,32])*(1-blocks),
        aggregate_total=ag,soluble_intact_TDP=z[:,6]+z[:,7],total_intact_TDP=z[:,6]+z[:,7]+z[:,37]+z[:,38]+z[:,41]+z[:,42],
        nuclear_function_proxy=z[:,6]+z[:,7],synthesis_capacity=z[:,50],
        synthesis_per_s=p[20]*z[:,50]*(1.-nv[22]*(1.-np.minimum(z[:,68],1.)) if nv[25]>0 else 1.),
        autophagic_degradation_flux= z[:,72]*z[:,74]/nv[18],autophagic_fusion_flux=encounter*z[:,73]/nv[17],
        EAAT3_support_hypothesis_marker=np.minimum(z[:,63],1.),
        baseline_state=base,checkpoint=cp,config=config,checks=checks,segments=sg,nfev=nfev,provenance=provenance,
        human_kinetic_fit=False,electrical_feedback=False,absolute_TDP_per_cell=None,
        limitations=['Functional organelle capacities and normalized cargo, not organelle counts, pH, ATP, morphology or absolute protein concentrations',
        'All new numerical parameters are assumptions; zero-gain controls are necessary',
        'PKN207 is a cryptic splicing product without kinase domain, AF3 is separate caspase-generated kinase fragment',
        'Caspase is an independent external driver; Ca/calpain does not automatically produce AF3',
        'PKN207-to-ER, NF-to-lysosome, TDP-loss-to-ER and position-to-translation are unvalidated causal connections',
        'No PKN1 Golgi claim from Golgi staining or non-neuronal PKN/KRT8 results',
        'Generic autophagic cargo is not a TDP sequestration compartment; TDP degradation uses an effective rate partition',
        'TDP nuclear function remains a soluble-protein proxy; STMN2 and biochemical pathology do not modify electrical currents',
        'Reduced clearance alone does not reproduce Ryan BORC knockdown with reduced TDP amount; no fit to that assay'])

def pulse_chase_readout(result,**kw):
    """Preserve the Ling observer for both old 63-state and new 83-state results."""
    z=np.asarray(result['state'])
    if z.ndim==2 and z.shape[1]==83:
        mapped=dict(result,state=z[:,:63],baseline_state=np.asarray(result['baseline_state'])[:63])
        return _prior.pulse_chase_readout(mapped,**kw)
    return _prior.pulse_chase_readout(result,**kw)
