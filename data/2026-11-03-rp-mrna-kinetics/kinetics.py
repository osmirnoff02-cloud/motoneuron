"""[codex] Individual RP-mRNA source and whole-cell effective loss, API0.51.

Replaces API0.50's common6h RP source/decay without duplicating RNA pools.
Published whole-cell rates are applied uniformly to the five RNA transport
states: an explicit effective approximation, not measured compartmental rates.
Absolute synthesis/abundance remain separately declared model quantities.
"""
from pathlib import Path
from dataclasses import dataclass,asdict
import json
import numpy as np
from numba import njit

B=Path(__file__).resolve().parent
GENES=tuple(p['gene'] for p in json.loads((B.parent/'2026-11-03-individual-ribosomal-proteins/protein_inputs.json').read_text())['proteins'])
SCOPE=('80 independent RP-mRNA production and effective whole-cell decay rates reuse API0.50 RNA transport pools. '
       'Whole-cell decay is distributed uniformly over free and carrier-bound nuclear/cytoplasmic RNA, releasing carriers; this is a model assumption. '
       'Published labeling-derived half-lives are source-cell measurements, not human motor-neuron calibration. '
       'Absolute RNA concentrations and synthesis fluxes must retain their own measured/inferred/assumed provenance.')

@dataclass(frozen=True)
class RPmRNAKineticsParameters:
    production_uM_s:tuple
    decay_s_inv:tuple
    initial_total_uM:tuple
    evidence_profile:str
    provenance:str

    def vector(self):
        v=np.asarray([self.production_uM_s,self.decay_s_inv,self.initial_total_uM],float)
        if v.shape!=(3,80) or not np.isfinite(v).all() or np.any(v<0):
            raise ValueError('RP-mRNA kinetics require three finite nonnegative80-gene vectors')
        if not self.evidence_profile.strip() or not self.provenance.strip():
            raise ValueError('Explicit evidence profile and provenance required')
        return v

def evidence_parameters(profile='human_neuron67_K56213'):
    catalog=json.loads((B/'model_profiles.json').read_text(encoding='utf8'))
    p=catalog['profiles'][profile]
    if tuple(catalog['genes'])!=GENES:raise ValueError('Kinetic catalog gene order differs from model')
    return RPmRNAKineticsParameters(tuple(p['production_uM_s']),tuple(p['decay_s_inv']),tuple(p['initial_total_uM']),profile,p['provenance'])

def controls(sg,supplied=None):
    a=np.c_[sg[:,:2],np.ones((len(sg),160))] if supplied is None else np.asarray(supplied,float)
    if a.shape!=(len(sg),162) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0):
        raise ValueError('RP-mRNA schedule: matching time bounds,80 production and80 decay supports')
    return a.copy()

def transfer_validation(parameters):
    """Read-only observation-model audit, never a fitted replacement rate."""
    record=json.loads((B/'human_MN_transfer_check.json').read_text(encoding='utf8'))
    tested=np.asarray([row['k_s_inv'] for row in record['rows']])
    if not np.allclose(np.asarray(parameters.decay_s_inv),tested,rtol=1e-13,atol=0.):
        return dict(status='not_evaluated_for_these_rates',human_motor_neuron_calibrated=False)
    return dict(status='does_not_pass_conditional_human_MN_transfer_screen',
        qualifying_genes=record['qualifying_genes'],genes=record['genes'],summary=record['summary'],
        assumptions=record['assumptions'],screen=record['screen'],source=record['source'],
        caution='This is an experimental source-cell prior, not a validated motor-neuron rate set. No automatic refit to the nonstationary/labeling-uncertain curves.')

def initialize(base,p,shape,off,rppv):
    if shape[1]!=80:raise ValueError('Individual RP-mRNA kinetics need all80 RP queues and RNA transport')
    out=base.copy()
    for i in range(80):
        k=off+8*(5+i)
        if np.any(out[k:k+4]):raise ValueError('RP RNA initializer expected empty nuclear/carrier pools')
        out[k+4]=p[2,i]*rppv[0,i]
    return out

@njit
def replace_rates(z,out,p,ctl,v,nrtctl,shape,off,rppv,rppcontrol,reference_ATP):
    """Replace only80 RP source/loss terms; conserve RNA lineage and carriers."""
    nr,np_,rp,rib,car,count,size=shape
    ATP=max(z[120],0.)
    e=ATP/(v[12]+ATP)
    eref=reference_ATP/(v[12]+reference_ATP)
    if eref<=0.:raise ValueError('Positive baseline ATP required to normalize RNA production')
    legacy_k=np.log(2)/(v[24]*3600.)
    cost_change=0.
    for i in range(80):
        k=off+8*(5+i)
        support=rppv[0,i]*rppcontrol[i]*nrtctl[8]
        oldsrc=legacy_k*v[1]*support*e
        newsrc=p[0,i]*support*ctl[i]*e/eref
        delta_source=newsrc-oldsrc
        out[k]+=delta_source;out[k+5]+=delta_source
        oldloss=0.;newloss=0.
        for j in range(5):
            amount=max(z[k+j],0.)
            old=legacy_k*amount if j==4 else (v[25]*amount if j<3 else 0.)
            new=p[1,i]*ctl[80+i]*amount
            delta_loss=new-old
            out[k+j]-=delta_loss;oldloss+=old;newloss+=new
            if j==1:out[off+car+2]+=delta_loss
            if j==2:out[off+car+1]+=delta_loss
            if j==3:out[off+car]+=delta_loss
        out[k+6]+=newloss-oldloss
        cost_change+=v[26]*delta_source
    out[off+count+2]+=cost_change
    out[120]-=cost_change;out[121]+=cost_change;out[125]+=cost_change
    return out

def observe(z,p,parameters,shape,off,v,reference_ATP):
    q=z[:,off:off+shape[2]].reshape(len(z),shape[0],8)[:,5:,:]
    h=np.full(80,np.inf);np.divide(np.log(2),p[1]*3600.,out=h,where=p[1]>0)
    return dict(genes=GENES,RNA_pools_counters=q,parameters=asdict(parameters),
        effective_half_life_h=h,reference_ATP_uM=float(reference_ATP),
        total_RNA_uM=q[:,:,:5].sum(2),cumulative_production_uM=q[:,:,5],cumulative_loss_uM=q[:,:,6],
        human_motor_neuron_calibrated=False,scope=SCOPE,
        legacy_common_half_life_replaced_h=float(v[24]),
        external_transfer_validation=transfer_validation(parameters),
        production='Declared uM/s at baseline ATP; independently controlled from decay; evidence profile records whether inferred')
