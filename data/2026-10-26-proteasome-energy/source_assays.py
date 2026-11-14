"""[codex] Published assay operators, isolated from human-MN parameter sets."""
import numpy as np

ZAHLER_TABLE3={
 'human_a1_HeLa':{'variable_hill':(12.,.64,6.),'fixed_hill3':(17.,1.2,3.)},
 'human_a1_in_rat_a3_HeLa':{'variable_hill':(13.,.36,5.9),'fixed_hill3':(16.,.52,3.)},
 'rat_a1_HeLa':{'variable_hill':(12.,.25,3.5),'fixed_hill3':(12.,.27,3.)},
 'rat_a2_mutant_HeLa':{'variable_hill':(22.,1.8,4.8),'fixed_hill3':(33.,3.7,3.)},
 'rat_a3_mutant_HeLa':{'variable_hill':(33.,2.5,6.8),'fixed_hill3':(44.,3.8,3.)},
}
def zahler_assay(sodium_mM,cell='rat_a3_mutant_HeLa',family='variable_hill'):
    if cell not in ZAHLER_TABLE3 or family not in ZAHLER_TABLE3[cell]:raise ValueError('Source cell/paired fit required')
    x=np.asarray(sodium_mM,float)
    if not np.isfinite(x).all() or np.any(x<0):raise ValueError('Nonnegative finite intracellular Na in mM')
    k,v,h=ZAHLER_TABLE3[cell][family];flux=v*x**h/(k**h+x**h)
    return dict(Na_efflux_mM_per_min=flux,ATP_stoichiometric_uM_per_s=flux*1000/60/3,Km_mM=k,Vmax_mM_per_min=v,Hill=h,cell=cell,family=family,alpha3_above_observed_limit=(x>40) if 'rat_a3_mutant' in cell else None,source='Zahler1997 TableIII p207; apparent whole-cell kinetics in HeLa, not human motoneuron concentrations/capacity')
