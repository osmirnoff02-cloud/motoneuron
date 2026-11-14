"""[codex] 2026-10-23. Continuous long calcium challenges with restartable state.

Inherited electrical/acute APIs are unchanged. Long inputs are declared hypothetical
concentration-equivalent fluxes, NOT extrapolated NEURON traces or fitted human data.
"""
from pathlib import Path
from dataclasses import asdict
import importlib.util, sys, json, hashlib
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

_HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_before_memory20261023',_HERE.parent/'2026-10-23-library-functional-update/model_api.py')
_prior=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_prior;_spec.loader.exec_module(_prior)
for _name in dir(_prior):
    if not _name.startswith('_') and _name not in globals():globals()[_name]=getattr(_prior,_name)

MEMORY_NAMES=('calcium_excess_uM','native_free','processed_free','native_CAST','processed_CAST','inactive',
    'TDP_U','TDP_P','large_N_fragment','small_N_product','STMN2_mRNA','STMN2_body_proxy',
    'STMN2_axon_proxy','regrowth_support_index','enzyme_synthesized','enzyme_removed','autolysed_cumulative',
    'inactivated_cumulative','TDP_synthesized','TDP_removed','primary_cleavage_cumulative',
    'secondary_cleavage_cumulative','calcium_entered_uM','calcium_removed_uM','calcium_excess_AUC_uM_s',
    'activity_AUC_s','cleavage_from_U','cleavage_from_P')


@njit
def _memory_rhs(y,j,p):
    # p: Ca0,tau; enzyme[8]; inactivation[2]; TDP k1,k2,dU,dN,dS;
    # phospho[3]; synthesis; STMN2[9]; catalytic block.
    ca=max(0.,p[0]+y[0]);gu=1/(1+(p[2]/max(ca,1e-300))**p[4]);ga=1/(1+(p[3]/max(ca,1e-300))**p[4])
    aut=p[5]*gu*y[1];bu=p[6]*gu*y[1]-p[7]*y[3];ba=p[6]*ga*y[2]-p[7]*y[4]
    iu=p[10]*gu*y[1];ia=p[11]*ga*y[2];d=p[8]
    act=(gu*y[1]+ga*y[2])*(1-p[30])
    qU=p[12]*act*y[6];qP=p[12]*act*p[19]*y[7];qN=p[13]*act*y[8]
    du=p[14]*y[6];dp=p[14]*y[7];dn=p[15]*y[8];ds=p[16]*y[9]
    ph=p[17]*y[6]-p[18]*y[7]
    n=max(0.,y[6]+y[7]);nh=n**p[22];kh=p[21]**p[22];target=nh*(1+kh)/(nh+kh)
    ax=max(0.,y[12]);ah=ax**p[29];ak=p[28]**p[29];rg=ah*(1+ak)/(ah+ak)
    out=np.zeros(28)
    out[0]=j-y[0]/p[1]
    out[1]=p[9]-aut-bu-d*y[1]-iu;out[2]=aut-ba-d*y[2]-ia
    out[3]=bu-d*y[3];out[4]=ba-d*y[4];out[5]=iu+ia-d*y[5]
    out[6]=p[20]-du-qU-ph;out[7]=ph-dp-qP
    out[8]=qU+qP-qN-dn;out[9]=qN-ds
    out[10]=(target-y[10])/p[23];out[11]=np.log(2)*(y[10]-y[11])/p[24]
    out[12]=(p[27]*y[10]-y[12])/p[25];out[13]=(rg-y[13])/p[26]
    out[14]=p[9];out[15]=d*np.sum(y[1:6]);out[16]=aut;out[17]=iu+ia
    out[18]=p[20];out[19]=du+dp+dn+ds;out[20]=qU+qP;out[21]=qN
    out[22]=j;out[23]=y[0]/p[1];out[24]=y[0];out[25]=act
    out[26]=qU;out[27]=qP
    return out


def long_calcium_memory(segments,*,basal_ca_uM,calcium_tau_s,enzyme_parameters,tdp_parameters,
        localization_parameters,stmn2_parameters,inactivation_parameters,phosphorylation_parameters,
        provenance,checkpoint=None,output_step_s=300.,rtol=1e-8,atol=1e-10,max_step_s=3600.,
        calpain_block_fraction=0.):
    """Run contiguous (start_s,end_s,entry_uM_per_s) segments without resetting.

    dCex/dt=entry-Cex/tau is an EFFECTIVE buffered calcium relaxation, not gamma
    separately from kappa. All switches split integration exactly. A checkpoint
    preserves every physical state, accounting integral, basal reference and the
    original synthesis rate. Changing parameters during resume is rejected.

    Inherits free/native/processed/CAST/inactive calpain, U/P/N-lineage TDP and
    M/body/axon/regrowth STMN2 topology. Constant nuclear fraction is the exact
    solution under inherited equal N/C losses and matching birth/import fractions.
    Functional nuclear TDP is assumed proportional to intact amount. STMN2 body
    and axon remain separate phenomenological proxies, not conserved transport.
    No injury feedback to Vm, receptors, calcium removal or protein synthesis.
    """
    if not isinstance(provenance,str) or not provenance.strip():raise ValueError('Explicit hypothetical input provenance required')
    sg=np.asarray(segments,float)
    if sg.ndim!=2 or sg.shape[1]!=3 or not len(sg) or not np.isfinite(sg).all() or np.any(sg[:,0]<0) or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[:,2]<0) or np.any(sg[1:,0]!=sg[:-1,1]):
        raise ValueError('Contiguous positive-duration nonnegative-flux segments required')
    vals=[basal_ca_uM,calcium_tau_s,output_step_s,rtol,atol,max_step_s]
    if not np.isfinite(vals).all() or min(vals)<=0 or not np.isfinite(calpain_block_fraction) or not 0<=calpain_block_fraction<=1:raise ValueError('Positive finite scales and fractional block required')
    ep=enzyme_parameters;tp=tdp_parameters;sp=stmn2_parameters;lp=localization_parameters
    ev=ep.vector();tv=tp.vector();iv=inactivation_parameters.vector();pv=phosphorylation_parameters.vector();lp.validate()
    ini=healthy_chain_state(basal_ca_uM=basal_ca_uM,enzyme_parameters=ep,tdp_parameters=tp,
        localization_parameters=lp,stmn2_parameters=sp,inactivation_parameters=inactivation_parameters)
    pi=phospho_tdp_basal_state(ini['activity'],tp,phosphorylation_parameters)
    ybase=np.r_[0.,ini['enzyme'],pi['pools'],ini['STMN2'],np.zeros(14)]
    config=dict(basal_ca_uM=basal_ca_uM,calcium_tau_s=calcium_tau_s,enzyme=asdict(ep),TDP=asdict(tp),
        STMN2=asdict(sp),localization=asdict(lp),inactivation=asdict(inactivation_parameters),
        phosphorylation=asdict(phosphorylation_parameters),calpain_block_fraction=calpain_block_fraction,
        topology='memory_v1_28states_equal_partition')
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    syn=pi['synthesis_per_s']
    if checkpoint is None:
        y=ybase.copy();origin=float(sg[0,0])
    else:
        if checkpoint['fingerprint']!=fingerprint or checkpoint['time_s']!=sg[0,0]:raise ValueError('Resume requires identical parameters and exact continuation time')
        y=np.asarray(checkpoint['state'],float).copy();origin=checkpoint['origin_time_s']
        if y.shape!=(28,) or not np.isfinite(y).all() or y.min()<-1e-7:raise ValueError('Invalid checkpoint state')
        if checkpoint['synthesis_per_s']!=syn or not np.array_equal(checkpoint['baseline_state'],ybase):raise ValueError('Basal reference changed')
    st=np.array([sp.tdp_half_relative,sp.tdp_hill,sp.mrna_tau_s,sp.somatodendritic_protein_half_life_s,
        sp.axonal_availability_tau_s,sp.regrowth_response_tau_s,sp.axonal_gain,sp.regrowth_half_relative,sp.regrowth_hill])
    p=np.r_[basal_ca_uM,calcium_tau_s,ev,iv,tv[5:10],pv,syn,st,calpain_block_fraction]
    times=[np.array([sg[0,0]])];states=[y[None,:]];evaluation_count=0
    for start,end,j in sg:
        # Log-spaced near-switch samples resolve fast activity peaks and slow endpoints.
        near=start+np.geomspace(.001,min(end-start,3600.),100)
        grid=np.unique(np.r_[np.arange(start+output_step_s,end,output_step_s),near[near<end],end])
        sol=solve_ivp(lambda t,z:_memory_rhs(z,j,p),(start,end),y,t_eval=grid,
            method='LSODA',rtol=rtol,atol=atol,max_step=max_step_s)
        if not sol.success:raise RuntimeError(sol.message)
        y=sol.y[:,-1].copy();times.append(sol.t);states.append(sol.y.T);evaluation_count+=sol.nfev
    t=np.concatenate(times);z=np.concatenate(states)
    en=z[:,1:6].sum(axis=1)+z[:,15]-z[:,14]-ybase[1:6].sum()
    td=z[:,6:10].sum(axis=1)+z[:,19]-z[:,18]-ybase[6:10].sum()
    ca=z[:,0]+z[:,23]-z[:,22]
    auc=z[:,24]-calcium_tau_s*z[:,23]
    errors=dict(enzyme_balance=float(abs(en).max()),TDP_lineage_balance=float(abs(td).max()),
        calcium_balance_uM=float(abs(ca).max()),calcium_AUC_identity=float(abs(auc).max()),minimum_state=float(z.min()))
    if not np.isfinite(z).all() or z.min()<-1e-7 or max(errors[k] for k in errors if k!='minimum_state')>1e-5:raise ArithmeticError(str(errors))
    c=basal_ca_uM+z[:,0];activity=(z[:,1]/(1+(ev[0]/c)**ev[2])+z[:,2]/(1+(ev[1]/c)**ev[2]))*(1-calpain_block_fraction)
    cp=dict(time_s=float(t[-1]),state=y.tolist(),baseline_state=ybase.tolist(),origin_time_s=origin,
        synthesis_per_s=syn,fingerprint=fingerprint)
    return dict(time_s=t,state=z,columns=MEMORY_NAMES,calcium_uM=c,activity=activity,
        intact_TDP=z[:,6]+z[:,7],nuclear_TDP_relative=z[:,6]+z[:,7],
        nuclear_TDP=lp.nuclear_fraction*(z[:,6]+z[:,7]),baseline_state=ybase,
        baseline_activity=ini['activity'],checkpoint=cp,checks=errors,config=config,segments=sg,
        provenance=provenance,nfev=evaluation_count,human_kinetic_fit=False,
        limitations=['Synthetic input, not a long human recording or ADAR2-dose prediction',
            'Effective calcium tau does not identify buffer capacity and pump rate separately',
            'Dimensionless enzyme and N-terminal lineage pools; complementary C fragments omitted',
            'STMN2 function proportional to intact nuclear TDP is assumed',
            'Regrowth support is not measured axon length, NMJ function, viability or ALS risk',
            'No injury feedback to electrical model; inherited Burley/Selvaraj tests remain separate'])


def beccari_supplement_reference():
    """Published S4 endpoint thresholds; separate from day35 Figure7 rescue assay."""
    return json.loads((_HERE/'beccari-supplement-evidence.json').read_text(encoding='utf-8'))


def beccari_extended_culture_endpoint(*,genotype,threshold_um,readout_h=48):
    """Source group reference only; no inferred continuous STMN2 dose response."""
    ref=beccari_supplement_reference()['S4']
    if genotype not in ('WT','KO') or threshold_um not in ref['length_thresholds_um'] or readout_h!=48:
        raise ValueError('Only published WT/KO, 48h and 250/500/750/1000um endpoints available')
    i=ref['length_thresholds_um'].index(threshold_um)
    return dict(fraction_above_threshold=ref[genotype+'_percent_above_threshold'][i]/100,
        genotype=genotype,threshold_um=threshold_um,readout_h=readout_h,assay=ref['assay'],
        independent_validation=False,partial_STMN2_prediction=False,
        limitation='Three pooled technical replicates; distinct from Fig7 day35; dynamic history not calibrated')


def source_training_data():
    r=_prior.source_training_data();r['Beccari2025_supplement']=beccari_supplement_reference();return r
