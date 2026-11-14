"""[codex] 2026-11-03. 80 independent, four-stage somatic RP translations.

Gene-resolved queues replace the two representative chains; old RP translation
states are residue-weighted diagnostic projections, never extra ribosomes.
Message availability is an external relative input, not an RNA birth/death ODE.
No codon-specific decoding, human kinetic fit or independent bound-RP exchange.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parent.parent
INPUT = json.loads((ROOT/'2026-11-03-individual-ribosomal-proteins/protein_inputs.json').read_text())
P = INPUT['proteins']
GENES = tuple(p['gene'] for p in P)
SUB = np.array([0 if p['subunit']=='40S' else 1 for p in P], dtype=np.int64)
COPY = np.array([p['copies'] for p in P], float)
L = np.array([sum(p['aa_translated']) for p in P], float)
LB = np.array([sum(p['aa_bound']) for p in P], float)
LC = L-LB
MASS = np.array([sum(COPY[SUB==s]*LB[SUB==s]) for s in range(2)])
FULL = np.array([sum(COPY[SUB==s]*L[SUB==s]) for s in range(2)])
CO = FULL-MASS
FIELDS = ('init40','elong0','elong1','elong2','elong3','stalled80','stalled_aa',
          'post80','RQC60','RQC_aa','orphan_aa','incorporated_aa','completed_chains','cleared_aa','aborted_chains')
OFFSET = 2160
N = OFFSET+80*15
X = slice(OFFSET,N)
NAMES = tuple('RP_translation_'+g+'_'+f+'_Ceq' for g in GENES for f in FIELDS)
RANGES = dict(message_support=[0.,.25,1.,4.], elongation_support=[0.,.25,1.,2.])

@dataclass(frozen=True)
class RPProductionParameters:
    message_support: tuple = (1.,)*80
    elongation_support: tuple = (1.,)*80

    def vector(self):
        v = np.asarray([self.message_support,self.elongation_support],float)
        if v.shape!=(2,80) or not np.isfinite(v).all() or np.any(v<0):
            raise ValueError('RP supports require two finite nonnegative 80-gene vectors')
        return v

def controls(sg,supplied=None):
    a=np.c_[sg[:,:2],np.ones((len(sg),160))] if supplied is None else np.asarray(supplied,float)
    if a.shape!=(len(sg),162) or not np.array_equal(a[:,:2],sg[:,:2]) or not np.isfinite(a).all() or np.any(a[:,2:]<0):
        raise ValueError('RP schedule: matching start,end,80 message supports,80 elongation supports')
    return a.copy()

@njit
def counts(x):
    return np.array([x[:,:6].sum()+x[:,7].sum(),x[:,1:6].sum()+x[:,7].sum()+x[:,8].sum()])

@njit
def project(x,representative):
    a=np.zeros((2,15))
    for i in range(80):
        s=SUB[i]
        for k in range(15):
            weight=1. if k in (6,9,10,11,13) else L[i]/representative[s]
            a[s,k]+=x[i,k]*weight
    return a

def initialize(base,ref):
    if len(base)!=OFFSET:raise ValueError('RP production requires the 2160-state RP inventory')
    z=np.r_[base,np.zeros(N-OFFSET)]
    old=base[403:433].reshape(2,15)
    x=z[X].reshape(80,15)
    for i in range(80):
        s=SUB[i]
        for k in range(15):
            weight=COPY[i]*(L[i] if k in (6,9,10,11,13) else ref[6+s])/FULL[s]
            x[i,k]=old[s,k]*weight
    # Preserve every inherited nascent-residue pool; redistribute the SAME
    # physical subunits between free and occupied fractions.
    z[330:332]+=counts(old)-counts(x)
    if min(z[330:332])<=0:raise ValueError('RP handoff exhausts free subunits')
    assert np.max(abs(project(x,ref[6:8])-old))<1e-12
    return z

@njit
def empty_legacy(z,sc):
    zz=z.copy();zz[403:433]=0.
    cc=sc.copy();cc[:2]=0.
    return zz,cc

@njit
def regulatory_view(z):
    """Nutrient stress reads physical occupied subunits, not mass projections."""
    zz=z.copy();x=np.maximum(z[X].reshape(80,15),0.)
    for s in range(2):
        for k in range(1,5):
            total=0.
            for i in range(80):
                if SUB[i]==s:total+=x[i,k]
            zz[403+15*s+k]=total
    return zz

@njit
def fluxes(z,v,rv,b,ref,atp0,nc,cc,bc,sc,cybase,pv,control):
    x=np.maximum(z[X].reshape((80,15)),0.)
    energy=max(z[120],0.)/(v[8]+max(z[120],0.))*(v[8]+atp0)/atp0
    birth=np.log(2)/(b[1]*86400)
    load=birth*COPY*bc[1]*sc[SUB]*pv[0]*control[:80]*max(z[330],0.)/cybase[0]*energy*nc[1]
    join=x[:,0]/rv[0]*max(z[331],0.)/cybase[1]*energy*cc[0]
    step=x[:,1:5]*(4*v[5]/L*pv[1]*control[80:])[:,None]*energy*nc[2]
    stall=rv[3]*cc[3]*x[:,1:5]
    rescue=x[:,5]/rv[2]*energy*cc[2];resc_aa=x[:,6]/rv[2]*energy*cc[2]
    rec=x[:,7]/rv[1]*energy*cc[1]
    release=x[:,8]/b[6]*energy*bc[5];releaseaa=x[:,9]/b[6]*energy*bc[5]
    clear=x[:,10]/b[8]*energy*bc[4]
    dx=np.zeros((80,15));dx[:,0]=load-join
    dx[:,1]=join-step[:,0]-stall[:,0];dx[:,2:5]=step[:,:3]-step[:,1:]-stall[:,1:]
    dx[:,5]=stall.sum(axis=1)-rescue
    dx[:,6]=L/4*(stall[:,1]+2*stall[:,2]+3*stall[:,3])-resc_aa
    dx[:,7]=step[:,3]-rec;dx[:,8]=(1-b[7])*rescue-release
    dx[:,9]=(1-b[7])*resc_aa-releaseaa;dx[:,10]=b[7]*resc_aa+releaseaa-clear
    dx[:,11]=L/4*step.sum(axis=1);dx[:,12]=step[:,3];dx[:,13]=clear;dx[:,14]=rescue
    free=np.array([(rescue+rec-load).sum(),(b[7]*rescue+rec+release-join).sum()])
    cost=np.array([4*b[0]*dx[:,11].sum(),b[0]*rv[5]*(rescue+rec).sum(),
                   b[0]*(b[9]*clear.sum()+b[10]*release.sum())])
    return dx,free,cost

@njit
def add_shared(dx,free,cost,cy,bio,shared,complete,paid,ref):
    projection=project(dx,ref[6:8])
    cy=cy.copy();bio=bio.copy();shared=shared.copy();complete=complete.copy();paid=paid.copy()
    cy[:2]+=free;shared[:2]=projection;complete[:2]=projection[:,12];paid[:3]+=cost
    for s in range(2):
        made=0.
        for i in range(80):
            if SUB[i]==s:made+=dx[i,12]*LB[i]/MASS[s]
        bio[24+s]+=made;bio[30+s]=made
    return cy,bio,shared,complete,paid

@njit
def finish(z,old,dx,C):
    out=np.zeros(N);out[:OFFSET]=old;out[X]=dx.ravel()
    # The inventory bridge supplied its old kit-based distribution. Replace
    # ONLY that source by actual gene completions; keep its disposal/assembly.
    for i in range(80):
        prior=C*COPY[i]*old[394+SUB[i]]
        actual=C*dx[i,12]
        out[1600+i]+=actual-prior;out[1840+i]=actual
    # Actual fusion output follows UBA52/FAU/RPS27A completion independently.
    for s in range(2):
        actual=0.
        for i in range(80):
            if SUB[i]==s:actual+=C*dx[i,12]*LC[i]
        out[686+s]+=actual-C*old[394+s]*CO[s]
    return out

def particle_view(z,base):
    zz=z.copy();bb=base.copy()
    for y in zz:
        y[330:332]+=counts(y[X].reshape(80,15))-counts(y[403:433].reshape(2,15))
    bb[330:332]+=counts(bb[X].reshape(80,15))-counts(bb[403:433].reshape(2,15))
    return zz,bb

def revise_shared(ch,z,base,C,ref):
    d=z-base;x=d[:,X].reshape(-1,80,15)
    for s in range(2):
        ch[('RP40','RP60')[s]+'_synthesis']=float(np.max(abs(d[:,394+s]-x[:,SUB==s,12]@LB[SUB==s]/MASS[s])))
    return ch

def revise_inventory(ch,z,base,C):
    d=z-base;x=d[:,X].reshape(-1,80,15)
    for s in range(2):
        ch['individual_RP_paid_synthesis_'+str(s)]=float(np.max(abs(d[:,1840:1920][:,SUB==s]-C*x[:,SUB==s,12])))
        copro=C*x[:,SUB==s,12]@LC[SUB==s]
        ch['fusion'+('40','60')[s]+'_residue_balance']=float(np.max(abs(d[:,686+s]+d[:,688+s]-copro)))
    return ch

def checks(z,base,ref,C):
    x=z[:,X].reshape(-1,80,15);b=base[X].reshape(80,15);d=x-b
    mass=L/4*(d[:,:,2]+2*d[:,:,3]+3*d[:,:,4])+d[:,:,6]+d[:,:,9]+d[:,:,10]+d[:,:,12]*L+d[:,:,13]-d[:,:,11]
    projected=np.array([project(a,ref[6:8]) for a in x])
    result=dict(RP_production_nonnegative=max(0.,float(-x.min())),
                RP_production_projection=float(np.max(abs(projected-z[:,403:433].reshape(-1,2,15)))),
                RP_production_chain_partition=float(np.max(abs(C*d[:,:,12]*L-C*d[:,:,12]*LB-C*d[:,:,12]*LC))))
    for i,g in enumerate(GENES):
        result[g+'_nascent_residue_balance']=float(np.max(abs(mass[:,i])))
        result[g+'_stalled_mass_bound']=max(0.,float(np.max(x[:,i,6]-.75*L[i]*x[:,i,5])),float(np.max(x[:,i,9]-.75*L[i]*x[:,i,8])))
    return result

def observe(z,C,parameters):
    x=z[:,X].reshape(-1,80,15)
    return dict(genes=GENES,fields=FIELDS,queues_Ceq=x,produced_uM=C*x[:,:,12],
                incorporated_residues_uM=C*x[:,:,11],translated_lengths_aa=L,bound_lengths_aa=LB,
                parameters=asdict(parameters),human_kinetics_calibrated=False,
                scope='80 independently controlled four-stage translations with sequence lengths, shared physical subunits and soma AA/charged-tRNA/GTP/ATP; exact fusion split. Message availability is relative external input; no RP-mRNA transcription/decay, codon decoding, amino-acid-specific soma pools or independent bound-RP exchange. Baseline demand is inherited stoichiometric renewal, not measured gene expression; initialization preserves inherited nascent mass and subunit count, not steady state.')
