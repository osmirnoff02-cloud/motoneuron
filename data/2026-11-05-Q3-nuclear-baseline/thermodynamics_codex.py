"""[codex] 2026-11-05. Teague/Golding/Dobson1996 Table1, solution chemistry.

Conditional comparator, not a measured human motoneuron concentration fit.
Retains the Table1 calculation even where the printed worked example differs.
"""
import math

# K(T=298.15K,I=0), enthalpy kJ/mol, sum(z_products**2)-sum(z_reactants**2).
TABLE={
 'HATP':(2.512e-8,-6.30,8),'HADP':(6.607e-8,-5.60,6),
 'HAMP':(1.862e-7,-5.40,4),'HPCr':(8.854e-6,2.66,4),
 'HPi':(6.026e-8,3.60,4),'MgATP':(1.514e6,22.90,-16),
 'MgHATP':(4.266e3,16.90,-12),'MgADP':(4.466e4,19.00,-12),
 'MgHADP':(3.163e2,12.50,-8),'MgAMP':(6.165e2,11.30,-8),
 'MgPCr':(2.320e2,8.19,-8),'MgPi':(5.128e2,12.20,-8),
 'CK':(2.580e8,-17.55,2),'AK':(.2248,-1.50,2),
 'ATP_hydrolysis':(.2946,-20.50,-2)}

def equilibrium(temperature_C=23.,pH=7.,free_Mg_mM=1.,ionic_strength_M=.25):
    vals=[temperature_C,pH,free_Mg_mM,ionic_strength_M]
    if not all(math.isfinite(v) for v in vals) or not (0<=temperature_C<=40 and 6<=pH<=8 and 0<=free_Mg_mM<=10 and 0<=ionic_strength_M<=.25):
        raise ValueError('Conservative calculation domain:0..40C,pH6..8,Mg0..10mM,I0..0.25M')
    T=273.15+temperature_C;I=ionic_strength_M;H=10**(-pH);M=free_Mg_mM/1000
    Am=3*(-16.390+261.337/T+3.369*math.log(T)-1.437*(T/100)+.112*(T/100)**2)
    phi=Am*math.sqrt(I)/(1+1.6*math.sqrt(I))
    k={name:K*math.exp(-1000*dH/8.3145*(1/T-1/298.15)+phi*dz2) for name,(K,dH,dz2) in TABLE.items()}
    P={name:1+H/k['H'+name]+M*k['Mg'+name] for name in ('ATP','ADP','AMP','PCr','Pi')}
    for name in ('ATP','ADP'):P[name]+=H*M*k['MgH'+name]/k['H'+name]
    return dict(CK_Kobs=k['CK']*H*P['ATP']/(P['ADP']*P['PCr']),
      AK_Kobs=k['AK']*P['ATP']*P['AMP']/P['ADP']**2,
      ATP_hydrolysis_Kobs_M=k['ATP_hydrolysis']/H*P['ADP']*P['Pi']/P['ATP'],
      reference_constants=k,binding_polynomials=P,temperature_C=temperature_C,pH=pH,
      free_Mg_mM=free_Mg_mM,ionic_strength_M=I,
      source='10.1242/jeb.199.2.509 Table1, equations6-10',
      limits='Only listed proton/Mg species; no K/Na binding, crowding or neuronal concentration measurement; constant enthalpy approximation')
