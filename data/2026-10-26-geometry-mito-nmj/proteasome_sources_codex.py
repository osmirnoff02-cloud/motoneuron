"""[codex] Primary source operators. No automatic human parameter transfer."""
import numpy as np
# Km uM, reported +/- (not declared CI), turnover/min/particle. Table I p32541.
HOFFMAN1996={
 '26S':{'ATP':(14.,5.,27.),'CTP':(16.,12.,46.),'GTP':(49.,31.,94.),'UTP':(107.,58.,109.)},
 'RC':{'ATP':(29.,7.,23.),'CTP':(87.,49.,28.),'GTP':(175.,48.,45.),'UTP':(490.,146.,59.)}}
# Thrower Table I, p97: Asp77 free K48 chains, n=3 mean+SD except Ub12 approximate.
THROWER_KI_NM={3:(1933.,219.),4:(171.,16.),6:(52.,4.),8:(26.,4.),12:(20.,None)}
def hoffman_reference(nucleotide_uM,particle='26S',nucleotide='ATP'):
 if not np.isfinite(nucleotide_uM) or nucleotide_uM<0:raise ValueError('Nonnegative concentration required')
 km,spread,kcat=HOFFMAN1996[particle][nucleotide]
 return dict(Km_uM=km,Km_reported_pm=spread,turnover_per_min=kcat*nucleotide_uM/(km+nucleotide_uM),saturating_turnover_per_min=kcat,source='10.1074/jbc.271.51.32538 TableI; rabbit reticulocytes37C; assumed26S1.4MDa(oneRC),RC0.7MDa; not comparable particle-for-particle with Henderson doubly capped26S',ADP_assay=dict(ATP_uM=200.,ADP_uM=1000.,relative_activity=.60 if particle=='26S' else .56,SD=.01 if particle=='26S' else .14,n=3),ADP_Ki_identified=False)
def thrower_competition_reference(substrate_nM=100.,free_chain_nM=0.,chain_length=4):
 if not np.isfinite([substrate_nM,free_chain_nM]).all() or min(substrate_nM,free_chain_nM)<0:raise ValueError('Nonnegative concentrations required')
 ki,spread=THROWER_KI_NM[chain_length];km=35.;kcat=.05
 return dict(rate_per_26S_min=kcat*substrate_nM/(km*(1+free_chain_nM/ki)+substrate_nM),Ki_nM=ki,Ki_SD_nM=spread,Km_nM=km,kcat_per_min=kcat,source='10.1093/emboj/19.1.94 TableI/Fig2; bovine/rabbit erythrocyte or rabbit reticulocyte purified26S37C,2mMATP+regeneration+1uMUbal; Ub5DHFR. Not human TDP or universal four-Ub requirement.')
TAKEUCHI_CONSTRAINTS=dict(source='10.1038/sj.emboj.7601476 Figures1-5; yeast/rat proteasomes, GFP-cODC/Rpn10, not TDP',requires_association=True,requires_accessible_initiation=True,extension_15aa='stable in tested construct',extension_22_27aa='intermediate, nonmonotonic',extension_32_37aa='faster in tested construct',universal_minimum_length_identified=False)
