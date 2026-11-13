"""[codex]2026-11-03. Optional conservative GAP43 mean-field/whole-neuron bridge.

The exact finite-copy SSA remains in API0.43. Here densities of polysomes use a
mean-field exclusion closure, not single-cell trajectories. One physical set of
ATP/GTP/AA pools; legacy GAP43 initiation/elongation/charging/RQC are disabled.
Other local proteins retain pooled composition, explicitly an unresolved boundary.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import numpy as np
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'2026-11-03-codon-translation'))
import codon_translation_codex as recipe
AA,DECODERS=recipe.AA,recipe.DECODERS
DA,PC,PREFIX,L=recipe.DECODER_AA,recipe.PC,recipe.PREFIX.astype(float),recipe.L
A=slice(745,765);Q=slice(765,827);U=slice(827,889);D=slice(889,951);H=slice(951,1013)
B=slice(1013,1251);R=slice(1251,1489);PEP=slice(1489,1509)
AI=slice(1509,1529);AO=slice(1529,1549);PI=slice(1549,1569);CO=slice(1569,1589)
CN=('init','elong','term','split','release','cleaved','repaired','elac','split_ATP','released_aa','release_ATP')
K={n:1589+i for i,n in enumerate(CN)};N=1589+len(CN)
NAMES=tuple('coupled_'+prefix+'_'+str(x) for prefix,seq in
 [('AA',AA),('charged',DECODERS),('uncharged',DECODERS),('cleaved',DECODERS),('repair_ready',DECODERS),
  ('GAP43_pos',range(L)),('RQC60_pos',range(L)),('peptide_AA',AA),('AA_in',AA),('AA_out',AA),('peptide_in',AA),('other_cargo_AA_out',AA)] for x in seq)+tuple('coupled_'+x for x in CN)
assert len(NAMES)==N-745
RANGES=dict(footprint_codons=[8,10,12],collision_split_per_s=[0.,.01,.05],release_per_s=[.01,.1,1.],
 split_ATP_equiv=[2.,10.,30.],release_ATP_per_aa=[0.,.2,1.],elac1_per_s=[0.,1/300,1/60,.1],
 trnt1_per_s=[0.,1/300,1/60,.1],ankzf1_fraction=[0.,.5,1.],charging_support=[0.,.25,1.],
 decoder_pool_skew=[.25,1.,4.],pause_multiplier=[0.,.1,1.])

@dataclass(frozen=True)
class CoupledTranslationParameters:
 footprint_codons:int=10
 collision_split_per_s:float=.01
 release_per_s:float=.1
 split_ATP_equiv:float=10.
 release_ATP_per_aa:float=.2
 elac1_per_s:float=1/60
 trnt1_per_s:float=1/60
 ankzf1_fraction:float=1.
 charging_support:float=1.
 decoder_pool_skew:float=1.
 pause_codon_index:int=80
 pause_multiplier:float=1.
 balanced_coarse_AA:bool=False
 def vector(self):
  if not isinstance(self.balanced_coarse_AA,bool):raise ValueError('balanced_coarse_AA must be a boolean mode, not a fitted coefficient')
  v=np.array(list(asdict(self).values()),float)
  if not np.isfinite(v).all() or min(v)<0 or not 1<=v[0]<=L or int(v[0])!=v[0] or v[7]>1 or v[8]>1 or v[9]<=0 or int(v[10])!=v[10] or v[10]>=L or v[11]>1:raise ValueError('Invalid coupled translation parameters')
  return v

def initialize(base,v,tv,C,ratio):
 """Conservative handoff of old nascent inventory; no ATP/ion reinitialization."""
 z=np.r_[base,np.zeros(N-745)];scale=C*ratio
 af=np.ones(20)/20;tf=np.ones(62);tf[PC[int(v[10])]]*=v[9];tf/=tf.sum()
 # A quarter-bin may correspond to a half residue: distribute between neighbours.
 oldmass=base[578:582]@np.arange(4)*L/4*scale
 for j,n in enumerate(base[578:582]*scale):
  length=max(1.,j*L/4);lo=int(np.floor(length));w=length-lo
  z[B][lo-1]+=n*(1-w)
  if w:z[B][lo]+=n*w
 newmass=z[B]@np.arange(1,L+1)
 z[A]=base[574]*af-(newmass-oldmass)*af
 z[574]=z[A].sum();z[Q]=base[575]*tf
 bound=np.bincount(PC,weights=z[B],minlength=62)
 z[U]=tv[7]*tf-z[Q]-bound
 if z[A].min()<0 or z[U].min()<0:raise ValueError('Insufficient inventory for conservative handoff')
 z[PEP]=base[584]*af
 z[607]+=base[578:582].sum()*scale # total messages; no one-message-per-ribosome sequestration
 z[578:582]=0
 return z,dict(old_nascent_aa_uM=float(oldmass),new_nascent_aa_uM=float(newmass),
  AA_paid_for_first_residue_uM=float(newmass-oldmass),ATP_change=0.,RNA_change=0.,ribosome_change=0.,
  scope='Mean-field density handoff; initial old4bin prefix composition unavailable; uniform initial free/peptide AA, scenario decoder fractions.')

def particle_view(z,base,C,ratio):
 zz=z.copy();bb=base.copy();scale=C*ratio
 zz[:,564]+=z[:,B].sum(1)/scale;zz[:,565]+=(z[:,B].sum(1)+z[:,R].sum(1))/scale
 bb[564]+=base[B].sum()/scale;bb[565]+=(base[B].sum()+base[R].sum())/scale
 return zz,bb

@njit
def finish(z,old,v,tv,ctl,C,ratio,KATP,RNA0):
 out=np.zeros(N);out[:745]=old;scale=C*ratio
 a=np.maximum(z[A],0.);q=np.maximum(z[Q],0.);u=np.maximum(z[U],0.);d=np.maximum(z[D],0.);h=np.maximum(z[H],0.)
 b=np.maximum(z[B],0.);r=np.maximum(z[R],0.);pep=np.maximum(z[PEP],0.)
 ATP=max(z[136],0.);en=ATP/(KATP+ATP);G=max(z[576],0.);gg=G/(tv[14]+G)
 tf=np.ones(62);tf[PC[int(v[10])]]*=v[9];tf/=tf.sum()
 charge=np.zeros(62)
 for c in range(62):charge[c]=tv[11]*a[DA[c]]/(tv[12]/20+a[DA[c]])*u[c]/tv[7]*en*ctl[4]*v[8]
 M=max(z[607],1e-30);foot=int(v[0]);J=np.zeros(L);spl=np.zeros(L)
 start=max(M-b[:foot].sum(),0.)
 ini=min(max(min(z[564],z[565]),0.)*scale,start)*ctl[0]/tv[2]*gg*q[61]/(tv[13]*tf[61]+q[61])
 for i in range(L-1):
  middle=b[i+1:min(L,i+foot)].sum();occupied=b[i+1:min(L,i+foot+1)].sum()
  availability=min(1.,max(0.,(M-occupied)/max(M-middle,1e-30)))
  c=PC[i+1];factor=v[11] if i+1==int(v[10]) else 1.
  J[i]=tv[1]*b[i]*q[c]/(tv[13]*tf[c]+q[c])*gg*ctl[1]*availability*factor
  if i>=foot:spl[i]=v[1]*b[i]*min(1.,b[i-foot]/M)*en
 J[-1]=tv[1]*b[-1]*gg
 release=v[2]*r*en;cleave=v[7]*release
 repaired=v[6]*h*min(1.,max(z[731],0.)/RNA0)
 elac=v[5]*d
 db=-J-spl;db[0]+=ini;db[1:]+=J[:-1]
 out[B]=db;out[R]=spl-release
 dq=charge.copy();du=-charge.copy();dd=-elac;dh=elac-repaired
 dq[61]-=ini
 for i in range(L-1):du[PC[i]]+=J[i];dq[PC[i+1]]-=J[i]
 du[PC[-1]]+=J[-1]
 for i in range(L):
  dd[PC[i]]+=cleave[i];du[PC[i]]+=release[i]-cleave[i]
 du+=repaired
 # Legacy proportional-decoder consumption biases Leu/Arg/Ser solely by codon
 # multiplicity, while the same coarse cargo returns a uniform20-AA recipe.
 # Optional repair uses that SAME declared recipe for consumption and return.
 # Upstream maintenance is gated by the limiting AA; initiator Met is excluded
 # from internal-residue consumption. GAP43 retains its exact codon sequence.
 other=max(-old[575],0.);taken=other*q/max(q.sum(),1e-30)
 if len(v)>12 and v[12]:
  totals=np.zeros(20)
  for c in range(61):totals[DA[c]]+=q[c]
  taken=np.zeros(62)
  for c in range(61):taken[c]=other/20*q[c]/max(totals[DA[c]],1e-30)
 dq-=taken;du+=taken
 out[Q]=dq;out[U]=du;out[D]=dd;out[H]=dh
 da=np.zeros(20);dp=np.zeros(20);otheraa=np.zeros(20)
 for c in range(62):da[DA[c]]-=charge[c];otheraa[DA[c]]+=taken[c]
 released=release@PREFIX[1:];hydro=pep/tv[22]
 # These are net physical exchanges, NOT extra sources. Negative exchange uses
 # existing species composition; unknown imported/other-protein composition uniform.
 ext=old[574]-old[597];pin=old[584]-L*old[596]+old[597]
 incoming=max(ext,0.)*np.ones(20)/20;outgoing=max(-ext,0.)*a/max(a.sum(),1e-30)
 psource=max(pin,0.)*np.ones(20)/20
 da+=incoming-outgoing+hydro
 dp=old[596]*PREFIX[-1]+released+psource-hydro
 out[A]=da;out[PEP]=dp
 out[AI]=incoming;out[AO]=outgoing;out[PI]=psource;out[CO]=otheraa
 out[574]=da.sum();out[575]=dq.sum();out[584]=dp.sum()
 out[564]+=(max(z[582],0.)/tv[3]*en-ini/scale+spl.sum()/scale)
 out[565]+=(max(z[582],0.)/tv[3]*en-ini/scale+release.sum()/scale)
 recycle=max(z[582],0.)/tv[3]*en
 out[582]=J[-1]/scale-recycle;out[583]+=J[-1]
 out[585]=ini/scale;out[586]=J[-1]/scale;out[587]=recycle
 out[588]=ini+J[:-1].sum();out[589]=charge.sum()
 out[590]=2*J[:-1].sum()+J[-1];out[591]=2*ini
 out[594]=2*charge.sum();out[595]=tv[20]*recycle*scale
 spent=out[590]+out[591];out[576]-=spent;out[577]+=spent
 splitcost=v[3]*spl.sum();releasecost=v[4]*released.sum()
 cost=out[594]+out[595]+splitcost+releasecost
 out[136]-=cost;out[137]+=cost;out[150]+=cost
 out[1589:]=np.array([ini,J[:-1].sum(),J[-1],spl.sum(),release.sum(),cleave.sum(),repaired.sum(),elac.sum(),splitcost,released.sum(),releasecost])
 # CCA1A+2C uses a pre-activated RNA reserve, not free ATP a second time.
 # Old CCA enters RNA debris once. Generic lysosomal destination remains assumed.
 out[727]+=3*cleave.sum()/ratio;out[731]-=3*repaired.sum()/ratio
 return out

@njit
def coarse_charged_view(z):
 """Uniform20-AA coarse recipe: limiting elongator, never initiator Met.

 This only changes the substrate gate upstream; real physical Q/AA and all
 costs/products are still owned once by the original maintenance+finish pair.
 """
 view=z.copy();totals=np.zeros(20)
 for c in range(61):totals[DA[c]]+=max(z[765+c],0.)
 view[575]=20*totals.min()
 return view

def revise_checks(ch,z,base,v,tv,C,ratio,rp40aa,rp60aa):
 out=dict(ch);d=z-base;scale=C*ratio;err=lambda x:float(np.max(abs(x)))
 nas=(d[:,B]+d[:,R])@np.arange(1,L+1);active=d[:,B]@np.arange(1,L+1)
 busy=d[:,B].sum(1);bound=d[:,R].sum(1)
 species=d[:,A]+d[:,PEP]+d[:,583,None]*PREFIX[-1]+(d[:,B]+d[:,R])@PREFIX[1:]
 for c in range(62):species[:,DA[c]]+=d[:,Q][:,c]
 tr=d[:,Q]+d[:,U]+d[:,D]+d[:,H]
 for i,c in enumerate(PC):tr[:,c]+=d[:,B][:,i]+d[:,R][:,i]
 residues=(busy+bound)*rp60aa+busy*rp40aa
 nasRP=d[:,644:648]@np.arange(4)*115/4*scale
 oldbusy=d[:,582]+d[:,644:649].sum(1)
 p40=d[:,562]+d[:,564]+oldbusy+d[:,640]+d[:,641]+busy/scale
 p60=d[:,563]+d[:,565]+oldbusy+d[:,673]+d[:,642]+(busy+bound)/scale
 backbone=scale*(rp40aa*p40+rp60aa*p60)
 delivered=scale*(rp40aa*(d[:,566]-d[:,570])+rp60aa*(d[:,567]-d[:,571]))
 aaexchange=d[:,599]-d[:,600];gx=d[:,601]-d[:,602]
 out.update(coupled_species_AA=err(species-d[:,AI]+d[:,AO]-d[:,PI]+d[:,CO]),coupled_tRNA_each=err(tr),
  coupled_AA_projection=err(z[:,574]-z[:,A].sum(1)),coupled_Q_projection=err(z[:,575]-z[:,Q].sum(1)),
  coupled_peptide_projection=err(z[:,584]-z[:,PEP].sum(1)),
  coupled_nonnegative=max(0.,float(-z[:,745:].min())),
  coupled_split_energy=err(d[:,K['split_ATP']]-v[3]*d[:,K['split']]),
  coupled_release_energy=err(d[:,K['release_ATP']]-v[4]*d[:,K['released_aa']]),
  local_busy_balance=err(busy/scale-d[:,585]+d[:,586]+d[:,K['split']]/scale),
  local_nascent_balance=err(nas+L*scale*d[:,586]+d[:,K['released_aa']]-d[:,588]),
  local_peptide_balance=err(d[:,584]-L*d[:,596]-d[:,K['released_aa']]+d[:,597]-d[:,668]-115*d[:,671]),
  local_AA_balance=err(d[:,574]+d[:,575]+nas+L*d[:,583]+d[:,584]+nasRP+115*d[:,643]-aaexchange-d[:,656]),
  maintenance_total_local_residues=err(d[:,574]+d[:,575]+nas+L*d[:,583]+d[:,584]+nasRP+115*d[:,643]+d[:,649]+backbone-aaexchange-delivered),
  maintenance_actual_rRNA40=err(p40+d[:,669]-d[:,566]+d[:,570]),maintenance_actual_rRNA60=err(p60+d[:,670]-d[:,567]+d[:,571]),
  local_elongation_GTP=err(d[:,590]-2*d[:,K['elong']]-d[:,K['term']]),
  local_energy_reserve_identity=err(d[:,594]+d[:,592]-d[:,593]-2*d[:,588]-2*d[:,660]-d[:,590]-d[:,591]-d[:,661]-d[:,662]-2*d[:,575]-d[:,576]+gx),
  RNA_lyso_debris=err(d[:,727]-d[:,733]+d[:,735]-3*d[:,K['cleaved']]/ratio),
  RNA_NTP=err(d[:,731]-d[:,738]+d[:,741]+3*d[:,K['repaired']]/ratio),
  RNA_open_material_balance=err(d[:,726:732].sum(1)+d[:,741]+d[:,740]+d[:,744]-d[:,739]-d[:,732]-d[:,733]+3*(d[:,K['repaired']]-d[:,K['cleaved']])/ratio))
 # Four-bin RQC checks concern a disabled branch; new position inventories above
 # replace its active-chain identity. Zero old counters are retained for compatibility.
 out['local_RQC_abort_mass']=err(active+L*scale*d[:,586]+(d[:,R]@np.arange(1,L+1))+d[:,K['released_aa']]-d[:,588])
 return out

def observe(z):
 return dict(free_AA=z[:,A],charged=z[:,Q],damaged=z[:,D]+z[:,H],polysome_positions=z[:,B],RQC60_positions=z[:,R],
  counters=z[:,1589:],counter_names=CN,shared_ATP=z[:,136],shared_GTP=z[:,576],GAP43=z[:,583],
  scope='Coupled selected GAP43 mean-field densities; legacy GAP43 charge/init/elong/RQC disabled. No exact stochastic collision dwell. Total mRNA decay independent of occupancy; unknown other cargo AA composition pooled. CCA uses base-pooled activated RNA reserve, not separate CTP; no human calibration.')
