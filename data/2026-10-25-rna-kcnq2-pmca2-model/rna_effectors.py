"""[codex] 2026-10-25. RNA -> protein -> membrane capacity hypotheses.

Relative abundances, not molecules/cell. Lee endpoints constrain RNA only.
Fast probes are counterfactual perturbations of source electrical profiles,
not reconstructions of OMN identity or continuous 72-hour neuron recordings.
"""
from pathlib import Path
from dataclasses import dataclass,asdict
import json,sys,importlib.util
import numpy as np
from scipy.integrate import solve_ivp
B=Path(__file__).resolve().parent

@dataclass(frozen=True)
class RNAEffectorParameters:
    rna_half_h: float=4.
    protein_half_h: float=24.
    export_time_h: float=2.
    recycle_time_h: float=12.
    translation_exponent: float=.5
    functional_transfer: float=1.
    provenance: str='Hypothesis sensitivity parameters, not measured human MN kinetics'
    def validate(self):
        x=asdict(self);v=[v for k,v in x.items() if k!='provenance']
        if not np.isfinite(v).all() or min(v[:4])<=0:raise ValueError('Positive finite times required')
        if not 0<=self.translation_exponent<=1 or not 0<=self.functional_transfer<=1 or not self.provenance.strip():raise ValueError('Explicit bounded transfer and provenance required')

RNA_EFFECTOR_RANGES=dict(rna_half_h=[.5,4.,24.],protein_half_h=[4.,24.,72.],
 export_time_h=[.5,2.,12.],recycle_time_h=[4.,12.,48.],translation_exponent=[0.,.5,1.],
 functional_transfer=[0.,.5,1.],pmca2_share_of_pmca=[0.,.5,1.])

def rna_effector_reference():
    result=json.loads((B/'rna-source.json').read_text(encoding='utf-8'))
    result['functional_evidence']=json.loads((B/'functional_evidence.json').read_text(encoding='utf-8'))
    return result

def simulate_rna_effectors(*,line='SOD1_GO013',genes=('KCNQ2','ATP2B2'),hours=72.,
                          parameters=None,gene_parameters=None,checkpoint=None,dt_h=.25):
    p=RNAEffectorParameters() if parameters is None else parameters;p.validate()
    if not np.isfinite([hours,dt_h]).all() or min(hours,dt_h)<=0:raise ValueError('Positive duration required')
    if not set(genes)<=set(('KCNQ2','ATP2B2')):raise ValueError('Unknown gene')
    rows=[r for r in rna_effector_reference()['primary'] if r['line']==line]
    if len(rows)!=2:raise ValueError('Source line not found')
    targets=np.array([next(r['FC_OMN_SMN'] for r in rows if r['gene_symbol']==g) if g in genes else 1. for g in ('KCNQ2','ATP2B2')])
    gene_parameters={} if gene_parameters is None else gene_parameters
    if not set(gene_parameters)<=set(('KCNQ2','ATP2B2')):raise ValueError('Unknown gene parameters')
    ps=[gene_parameters.get(g,p) for g in ('KCNQ2','ATP2B2')]
    for pp in ps:pp.validate()
    kr=np.array([np.log(2)/pp.rna_half_h for pp in ps]);kd=np.array([np.log(2)/pp.protein_half_h for pp in ps])
    ke=np.array([1/pp.export_time_h for pp in ps]);krec=np.array([1/pp.recycle_time_h for pp in ps])
    fs=ke/(ke+krec+kd)
    initial=np.array([[1.,1-f,f,0.,0.] for f in fs]).ravel()
    signature=dict(parameters=[asdict(pp) for pp in ps],line=line,genes=list(genes),version=1)
    start=0.
    if checkpoint is not None:
        if checkpoint['signature']!=signature:raise ValueError('Checkpoint parameter/source mismatch')
        initial=np.array(checkpoint['state'],float);start=float(checkpoint['time_h'])
        if initial.shape!=(10,) or not np.isfinite(initial).all() or (initial<0).any():raise ValueError('Invalid checkpoint')
    def rhs(t,y):
        dy=np.zeros(10)
        for j,target in enumerate(targets):
            i=5*j;r,c,s=y[i:i+3];prod=kd[j]*max(r,0.)**ps[j].translation_exponent;loss=kd[j]*(c+s)
            dy[i:i+5]=[kr[j]*(target-r),prod-kd[j]*c-ke[j]*c+krec[j]*s,ke[j]*c-(krec[j]+kd[j])*s,prod,loss]
        return dy
    time=np.linspace(start,start+hours,max(2,int(np.ceil(hours/dt_h))+1))
    sol=solve_ivp(rhs,(start,start+hours),initial,t_eval=time,rtol=1e-9,atol=1e-11,method='LSODA')
    if not sol.success:raise RuntimeError(sol.message)
    z=sol.y.T
    balances=[float(np.max(abs(z[:,5*j+1]+z[:,5*j+2]-1-z[:,5*j+3]+z[:,5*j+4]))) for j in range(2)]
    if min(z.ravel()) < -1e-9 or max(balances)>1e-7:raise ArithmeticError('Protein balance failed')
    surface=z[:,[2,7]]/fs
    capacity=1+np.array([pp.functional_transfer for pp in ps])*(surface-1)
    return dict(time_h=time,state=z,RNA=z[:,[0,5]],total_protein=z[:,[1,6]]+z[:,[2,7]],
        surface_protein=surface,functional_capacity=capacity,genes=['KCNQ2','ATP2B2'],RNA_targets=targets,
        protein_balance_error=balances,parameters={g:asdict(pp) for g,pp in zip(('KCNQ2','ATP2B2'),ps)},line=line,
        checkpoint=dict(signature=signature,state=z[-1].tolist(),time_h=float(time[-1])),
        limitation='RNA endpoint measured; translation/turnover/trafficking/function are hypotheses. Existing TDP-splicing transport branch is not multiplied automatically.')

def run_rna_effector_probe(*,expression,pmca2_share=.5,**kwargs):
    """Snapshot experiment, seconds only. Explicit same Ca/ATP initial state.

    Select parameters=scenario_parameters(...) for a source profile as needed.
    Do not combine independently with another KCNQ expression multiplier.
    """
    ratios=np.asarray(expression['functional_capacity'],float)[-1]
    if ratios.shape!=(2,) or not np.isfinite(ratios).all() or (ratios<0).any():raise ValueError('Invalid effector state')
    if any(k in kwargs for k in ('channel_ratio','pmca2_ratio','pmca2_share')):raise ValueError('Effector multipliers set by expression state')
    sys.path.insert(0,str(B))
    from electrical_rna import run_rna_coupled
    out=run_rna_coupled(channel_ratio=float(ratios[0]),pmca2_ratio=float(ratios[1]),pmca2_share=pmca2_share,**kwargs)
    out['expression_snapshot']=dict(line=expression['line'],time_h=float(expression['time_h'][-1]),
        parameters=expression['parameters'],functional_capacity=ratios.tolist(),
        human_functional_fit=False,continuous_multiday_neuron=False)
    return out
