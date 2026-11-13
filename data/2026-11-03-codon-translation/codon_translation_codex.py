"""[codex] 2026-11-03. Finite-resource, event-driven selected GAP43 translation.

Separate detailed module: one-way initial resource adapter, NOT an extra debit in
the legacy whole-neuron RHS. Counts are molecules in a specified local volume.
61 effective codon decoders plus initiator-Met are a resolution, not measured tRNAs;
wobble, modifications, errors, CAT tails and all-protein competition are absent.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import copy, json, math
import numpy as np
B=Path(__file__).resolve().parent
RECIPE=json.loads((B/'GAP43_recipe_codex.json').read_text(encoding='utf8'))
CODONS=sorted(RECIPE['standard_code'])
AA=sorted(set(RECIPE['standard_code'].values()))
CI={c:i for i,c in enumerate(CODONS)}; AI={a:i for i,a in enumerate(AA)}
DECODERS=CODONS+['initiator_Met'];NDEC=len(DECODERS)
DECODER_AA=np.array([AI[RECIPE['standard_code'][c]] for c in CODONS]+[AI['M']])
SEQ=np.array([CI[c] for c in RECIPE['codons']]); AS=np.array([AI[a] for a in RECIPE['protein']])
PC=SEQ.copy();PC[0]=61
PREFIX=np.vstack([np.zeros(20,dtype=int),np.cumsum(np.eye(20,dtype=int)[AS],axis=0)])
L=len(SEQ)

@dataclass(frozen=True)
class CodonTranslationParameters:
    messages: int=3
    ribosomes: int=30
    trna_per_decoder: int=20
    charged_fraction: float=.5
    aa_per_species: int=5000
    ATP_equiv0: float=1000000.
    GTP0: int=500000
    CTP0: int=10000
    initiation_per_message_s: float=.15
    elongation_per_s: float=5.
    charging_per_s: float=.3
    aa_half_molecules: float=100.
    decoding_half_molecules: float=2.
    footprint_codons: int=10
    collision_dwell_s: float=10.
    split_per_s: float=.05
    split_ATP_equiv: float=10.
    release_per_s: float=.1
    ankzf1_fraction: float=1.
    peptide_hydrolysis_per_s: float=.02
    proteolysis_ATP_per_aa: float=.2
    elac1_per_s: float=.02
    trnt1_per_s: float=.02
    termination_per_s: float=5.
    recycle_ATP_equiv: float=2.
    ndpk_per_s: float=0.

    def validate(self):
        for k,v in asdict(self).items():
            if not np.isfinite(v) or v<0: raise ValueError(k)
        for k in ('messages','ribosomes','trna_per_decoder','aa_per_species','GTP0','CTP0','footprint_codons'):
            if int(getattr(self,k))!=getattr(self,k): raise ValueError(k+' must be integer')
        if self.messages<1 or self.footprint_codons<1: raise ValueError('positive messages/footprint required')
        if not 0<=self.charged_fraction<=1 or not 0<=self.ankzf1_fraction<=1: raise ValueError('fraction')
        if min(self.aa_half_molecules,self.decoding_half_molecules)<=0: raise ValueError('half saturation')

RANGES={
 'initiation_per_message_s':[.03,.15,.5], 'elongation_per_s':[2.,5.,8.],
 'charging_per_s':[.03,.3,3.], 'aa_half_molecules':[10.,100.,1000.],
 'decoding_half_molecules':[.2,2.,20.], 'footprint_codons':[8,10,12],
 'collision_dwell_s':[5.,10.,30.], 'split_per_s':[.01,.05,.2],
 'split_ATP_equiv':[2.,10.,30.], 'release_per_s':[.01,.1,1.],
 'ankzf1_fraction':[0.,.5,1.], 'peptide_hydrolysis_per_s':[.002,.02,.2],
 'proteolysis_ATP_per_aa':[0.,.2,1.], 'elac1_per_s':[0.,1/300,1/60,.1],
 'trnt1_per_s':[0.,1/300,1/60,.1], 'termination_per_s':[1.,5.,10.],
 'recycle_ATP_equiv':[2.,4.,10.], 'ndpk_per_s':[0.,.01,.1]}

def initial(p,resources=None):
    p.validate(); n=p.trna_per_decoder; q=int(round(n*p.charged_fraction))
    s=dict(t=0.,aa=[p.aa_per_species]*20,q=[q]*NDEC,u=[n-q]*NDEC,d=[0]*NDEC,h=[0]*NDEC,
      ATP=p.ATP_equiv0,GTP=p.GTP0,GDP=0,CTP=p.CTP0,free40=p.ribosomes,free60=p.ribosomes,
      ribs=[],bound60=[],peptides=[],next_id=0,completed=0,initiated=0,elongated=0,
      split=0,released=0,cleaved=0,elac=0,repaired=0,charged=0,hydrolyzed_aa=0,
      ATP_charge=0.,ATP_split=0.,ATP_recycle=0.,ATP_proteolysis=0.,ATP_CCA=0.,ATP_ndpk=0.,
      GTP_used=0,CTP_CCA=0,ndpk=0,events=0)
    if resources:
        for k in ('aa','q','u','ATP','GTP','GDP','CTP','free40','free60'):
            if k in resources:s[k]=copy.deepcopy(resources[k])
    for k in ('aa','q','u'):
        expected=20 if k=='aa' else NDEC
        if len(s[k])!=expected or any(v<0 or int(v)!=v for v in s[k]):raise ValueError(k)
    for k in ('ATP','GTP','GDP','CTP','free40','free60'):
        if not np.isfinite(s[k]) or s[k]<0:raise ValueError('resource '+k)
        if k!='ATP' and int(s[k])!=s[k]:raise ValueError('integer resource '+k)
    s['origin']=inventory(s)
    return s

def inventory(s):
    aa=np.array(s['aa'],float); tr=np.array(s['q'])+s['u']+np.array(s['d'])+s['h']
    np.add.at(aa,DECODER_AA,s['q'])
    for r in s['ribs']+s['bound60']:
        aa+=PREFIX[r['pos']+1]; tr[PC[r['pos']]]+=1
    for pos in s['peptides']:aa+=PREFIX[pos+1]
    aa+=s['completed']*PREFIX[-1]
    return dict(aa=aa.tolist(),trna=tr.tolist(),r40=s['free40']+len(s['ribs']),
      r60=s['free60']+len(s['ribs'])+len(s['bound60']),
      ATP=s['ATP']+sum(s[k] for k in ('ATP_charge','ATP_split','ATP_recycle','ATP_proteolysis','ATP_CCA','ATP_ndpk')),
      G=s['GTP']+s['GDP'],C=s['CTP']+s['CTP_CCA'])

def checks(s,p):
    now=inventory(s); o=s['origin']; out={k:float(np.max(abs(np.asarray(now[k])-o[k]))) for k in now}
    out['GTP_ledger']=abs(s['GTP']+s['GTP_used']-s['ndpk']-(o['G']-s.get('initial_GDP',0)))
    # CCA addition incorporates ATP and 2CTP, not an ATP-generating reaction.
    out['CCA_stoichiometry']=abs(s['ATP_CCA']-s['repaired'])+abs(s['CTP_CCA']-2*s['repaired'])
    out['charging_cost']=abs(s['ATP_charge']-2*s['charged'])
    out['particle_ledger']=abs(s['initiated']-s['completed']-s['split']-len(s['ribs']))
    out['release_ledger']=abs(s['split']-s['released']-len(s['bound60']))
    out['repair_ledger']=abs(s['cleaved']-s['repaired']-sum(s['d'])-sum(s['h']))
    out['negative']=max(0.,-min(s['aa']+s['q']+s['u']+s['d']+s['h']+[s['ATP'],s['GTP'],s['GDP'],s['CTP'],s['free40'],s['free60']]))
    for m in range(p.messages):
        positions=sorted(r['pos'] for r in s['ribs'] if r['m']==m)
        if any(b-a<p.footprint_codons for a,b in zip(positions,positions[1:])):raise AssertionError('ribosome overlap')
    return out

def collision_pairs(s,p):
    pairs=[]
    for m in range(p.messages):
        rr=sorted((r for r in s['ribs'] if r['m']==m),key=lambda x:x['pos'])
        pairs += [(a,b) for a,b in zip(rr,rr[1:]) if b['pos']-a['pos']==p.footprint_codons]
    return pairs

def rates(s,p,challenge):
    ev=[]
    def add(rate,kind,index=0):
        if rate>0:ev.append((rate,kind,index))
    for c in range(NDEC):
        a=DECODER_AA[c]; x=s['aa'][a]
        if s['ATP']>=2 and x>0:
            factor=challenge.get('charging_multiplier',1.) if AA[a]==challenge.get('amino_acid') else 1.
            add(p.charging_per_s*s['u'][c]*x/(p.aa_half_molecules+x)*factor,'charge',c)
        add(p.elac1_per_s*s['d'][c],'elac',c)
        if s['ATP']>=1 and s['CTP']>=2:add(p.trnt1_per_s*s['h'][c],'repair',c)
    if s['ATP']>=1 and s['GDP']>0:add(p.ndpk_per_s*s['GDP'],'ndpk')
    if s['free40'] and s['free60'] and s['GTP']>=2 and s['q'][PC[0]]:
        for m in range(p.messages):
            if not any(r['m']==m and r['pos']<p.footprint_codons for r in s['ribs']):
                add(p.initiation_per_message_s,'init',m)
    pairs=collision_pairs(s,p); blocked={a['id'] for a,b in pairs}; leaders={b['id'] for a,b in pairs}
    for i,r in enumerate(s['ribs']):
        # Collision dwell starts when the trailing ribosome reaches the leader.
        if r['id'] in leaders:
            if r.get('collided_since') is None:r['collided_since']=s['t']
            if s['t']-r['collided_since']>=p.collision_dwell_s and s['ATP']>=p.split_ATP_equiv:
                add(p.split_per_s,'split',i)
        else:r['collided_since']=None
        if r['pos']==L-1:
            if s['GTP']>=1 and s['ATP']>=p.recycle_ATP_equiv:add(p.termination_per_s,'term',i)
        elif r['id'] not in blocked and s['GTP']>=2:
            c=SEQ[r['pos']+1]; q=s['q'][c]
            factor=challenge.get('pause_multiplier',1.) if r['pos']+1==challenge.get('pause_codon_index') else 1.
            add(p.elongation_per_s*q/(q+p.decoding_half_molecules)*factor,'elong',i)
    for i,r in enumerate(s['bound60']):add(p.release_per_s,'release',i)
    for i,pos in enumerate(s['peptides']):
        if s['ATP']>=p.proteolysis_ATP_per_aa*(pos+1):add(p.peptide_hydrolysis_per_s,'hydro',i)
    # Known deterministic age threshold is handled as a boundary, not ignored by SSA.
    deadlines=[r['collided_since']+p.collision_dwell_s for r in s['ribs'] if r.get('collided_since') is not None and r['collided_since']+p.collision_dwell_s>s['t']+1e-10]
    return ev,min(deadlines,default=math.inf)

def event(s,p,kind,i,rng):
    def atp(cost,ledger):s['ATP']-=cost;s[ledger]+=cost
    def gtp(n):s['GTP']-=n;s['GDP']+=n;s['GTP_used']+=n
    if kind=='charge':
        s['aa'][DECODER_AA[i]]-=1;s['u'][i]-=1;s['q'][i]+=1;s['charged']+=1;atp(2.,'ATP_charge')
    elif kind=='elac':s['d'][i]-=1;s['h'][i]+=1;s['elac']+=1
    elif kind=='repair':
        s['h'][i]-=1;s['u'][i]+=1;s['repaired']+=1;s['CTP']-=2;s['CTP_CCA']+=2;atp(1.,'ATP_CCA')
    elif kind=='ndpk':s['GDP']-=1;s['GTP']+=1;s['ndpk']+=1;atp(1.,'ATP_ndpk')
    elif kind=='init':
        s['q'][PC[0]]-=1;s['free40']-=1;s['free60']-=1;s['initiated']+=1;gtp(2)
        s['ribs'].append(dict(id=s['next_id'],m=i,pos=0,collided_since=None));s['next_id']+=1
    elif kind=='elong':
        r=s['ribs'][i];s['u'][PC[r['pos']]]+=1;r['pos']+=1;s['q'][PC[r['pos']]]-=1
        r['collided_since']=None;s['elongated']+=1;gtp(2)
    elif kind=='term':
        r=s['ribs'].pop(i);s['u'][PC[r['pos']]]+=1;s['completed']+=1;s['free40']+=1;s['free60']+=1
        gtp(1);atp(p.recycle_ATP_equiv,'ATP_recycle')
    elif kind=='split':
        r=s['ribs'].pop(i);s['free40']+=1;s['bound60'].append(r);s['split']+=1;atp(p.split_ATP_equiv,'ATP_split')
    elif kind=='release':
        r=s['bound60'].pop(i);s['free60']+=1;s['released']+=1;s['peptides'].append(r['pos']);c=PC[r['pos']]
        if rng.random()<p.ankzf1_fraction:s['d'][c]+=1;s['cleaved']+=1
        else:s['u'][c]+=1
    elif kind=='hydro':
        pos=s['peptides'].pop(i);s['aa']=(np.array(s['aa'])+PREFIX[pos+1]).tolist();s['hydrolyzed_aa']+=pos+1
        atp(p.proteolysis_ATP_per_aa*(pos+1),'ATP_proteolysis')
    s['events']+=1

def observation(s,p):
    return dict(time_s=s['t'],completed=s['completed'],active=len(s['ribs']),bound60=len(s['bound60']),
      collision_pairs=len(collision_pairs(s,p)),split=s['split'],damaged=sum(s['d']),repair_ready=sum(s['h']),repaired=s['repaired'],
      aa=s['aa'].copy(),charged=s['q'].copy(),ATP=s['ATP'],GTP=s['GTP'],CTP=s['CTP'],
      ATP_spent=sum(s[k] for k in ('ATP_charge','ATP_split','ATP_recycle','ATP_proteolysis','ATP_CCA','ATP_ndpk')))

def run(duration_s=600.,*,parameters=None,resources=None,seed=1729,challenge=None,checkpoint=None,output_step_s=10.):
    """Continue finite inventories; duration is additional time on restart.

    Challenge: {start_s,end_s,pause_codon_index,pause_multiplier} and/or
    {amino_acid,charging_multiplier}. No state reset at stimulus removal.
    output sampling changes neither event draws nor trajectories.
    """
    p=parameters or CodonTranslationParameters();p.validate()
    if duration_s<=0 or output_step_s<=0:raise ValueError('positive duration/output step')
    c=challenge or {}
    for key in ('pause_multiplier','charging_multiplier'):
        if c.get(key,1.)<0:raise ValueError(key)
    if 'pause_codon_index'in c and not 1<=c['pause_codon_index']<L:raise ValueError('codon index')
    if 'amino_acid'in c and c['amino_acid'] not in AI:raise ValueError('amino acid')
    if c and not 0<=c.get('start_s',0)<c.get('end_s',0):raise ValueError('challenge times')
    rng=np.random.default_rng(seed)
    if checkpoint:
        if checkpoint['recipe_sha256']!=RECIPE['sha256_genbank']:raise ValueError('recipe mismatch')
        s=copy.deepcopy(checkpoint['state']);rng.bit_generator.state=checkpoint['rng']
    else:s=initial(p,resources);s['initial_GDP']=s['GDP']
    end=s['t']+duration_s; obs=[observation(s,p)]; sample=s['t']+output_step_s
    maxerr=checks(s,p)
    while s['t']<end-1e-10:
        on=c.get('start_s',math.inf)<=s['t']<c.get('end_s',-math.inf)
        ev,deadline=rates(s,p,c if on else {})
        boundary=min([end,deadline]+[c[k] for k in ('start_s','end_s') if k in c and c[k]>s['t']+1e-10])
        total=sum(v[0] for v in ev)
        next_t=s['t']+rng.exponential(1/total) if total else math.inf
        target=min(next_t,boundary)
        while sample<target-1e-10:
            row=observation(s,p);row['time_s']=sample;obs.append(row);sample+=output_step_s
        s['t']=target
        if next_t<boundary:
            u=rng.random()*total;cum=0.
            for rate,kind,i in ev:
                cum+=rate
                if u<cum:event(s,p,kind,i,rng);break
        if s['events']%1000==0:
            for k,v in checks(s,p).items():maxerr[k]=max(maxerr[k],v)
        if s['t']>=sample-1e-10:obs.append(observation(s,p));sample+=output_step_s
    for k,v in checks(s,p).items():maxerr[k]=max(maxerr[k],v)
    if not obs or obs[-1]['time_s']<end-1e-8:obs.append(observation(s,p))
    return dict(interface_version='0.43-selected-cargo',parameters=asdict(p),challenge=c,seed=seed,
      recipe_accession=RECIPE['accession'],amino_acid_order=AA,decoder_order=DECODERS,observations=obs,
      checks=maxerr,checkpoint=dict(recipe_sha256=RECIPE['sha256_genbank'],state=s,rng=rng.bit_generator.state),
      limitations=['Selected GAP43 detailed branch; does not replace or double-charge legacy whole-neuron translation.',
       '61 independent effective codon decoder pools plus initiator-Met, no measured isoacceptor abundances/wobble/modifications.',
       '20 AA pools and finite ATP-equivalent/GTP/CTP budgets; charging ATP->AMP represented by2ATP-equivalent debit with rapid AMP recovery;not explicit adenylate chemistry.',
       'CCA repair consumes1ATP+2CTP and discards old3nt; no RNA salvage or ATP credit from those nucleotides.',
       'Rates/counts are scenarios, not measured human motor-neuron parameters; RQC requires persistent collision, Ptrh1-like alternative returns intact tRNA.',
       'No mRNA cleavage, CAT tails, ubiquitin kinetics, all-cargo competition or feedback onto GAP43 growth/electrical whole-neuron RHS.',
       'Restart retains inventories/ledgers; exponential waiting time is resampled at restart (distributionally valid, not pathwise identical).'])

def resources_from_neuron(state,*,volume_fL,CTP_uM,total_tRNA_uM,AA_fractions=None,decoder_fractions=None):
    """One-way snapshot, explicit assumed composition; caller supplies missing CTP.
    Do not run alongside the old flux and add costs: these are alternative branches.
    """
    z=np.asarray(state); factor=602.214076*volume_fL
    if len(z)<605 or volume_fL<=0 or CTP_uM<0 or total_tRNA_uM<z[575]:raise ValueError('snapshot/volume/CTP/tRNA')
    af=np.asarray(AA_fractions if AA_fractions is not None else np.ones(20)/20)
    tf=np.asarray(decoder_fractions if decoder_fractions is not None else np.ones(NDEC)/NDEC)
    if af.shape!=(20,) or tf.shape!=(NDEC,) or np.min(af)<0 or np.min(tf)<0 or not np.isclose(af.sum(),1) or not np.isclose(tf.sum(),1):raise ValueError('fractions')
    return dict(aa=np.floor(z[574]*factor*af).astype(int).tolist(),q=np.floor(z[575]*factor*tf).astype(int).tolist(),
      u=np.floor((total_tRNA_uM-z[575])*factor*tf).astype(int).tolist(),
      ATP=float(z[136]*factor),GTP=int(z[576]*factor),GDP=int(z[577]*factor),CTP=int(CTP_uM*factor))

def fibroblast_adenosine_deamination(adenosine_uM,*,Km_uM=37.):
    """Lindley&Pisoni1993 pH5.5,25C. Unit pmol/min/Hex unit, NOT neuron uM/s."""
    a=np.asarray(adenosine_uM,float)
    if np.any(a<0) or Km_uM<=0:raise ValueError('substrate/Km')
    return 14.8*a/(Km_uM+a)
