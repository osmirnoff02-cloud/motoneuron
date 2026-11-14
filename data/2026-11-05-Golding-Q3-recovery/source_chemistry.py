"""[codex] Golding1995 direct Table2/eq9,17,25 comparator and CK flux observer.

All pools are sums of the specified freely diffusible proton/Mg species, not
unbound ATP4-/ADP3- alone and not tissue-extract nucleotide totals.
This module does not change the physical baseline or infer human MN rates.
"""
import math

GOLDING38={
 'HATP':3.23e-7,'HADP':4.45e-7,'HAMP':6.28e-7,'HPCr':3.53e-5,'HPi':2.41e-7,
 'MgATP':9.90e3,'MgHATP':9.42e1,'MgADP':1.11e3,'MgHADP':2.62e1,
 'MgPi':4.34e1,'MgPCr':1.84e1,'MgAMP':5.47e1,
 'CK':3.77e8,'AK':.374,'ATP_hydrolysis':.0722}

def golding_equilibrium(pH=7.,free_Mg_mM=1.,temperature_C=38.,ionic_strength_M=.25):
 if not all(math.isfinite(v) for v in (pH,free_Mg_mM,temperature_C,ionic_strength_M)) or not (6.4<=pH<=7.4 and 0<=free_Mg_mM<=1.3):
  raise ValueError('Golding published grid: pH6.4..7.4, free Mg0..1.3mM')
 if temperature_C!=38 or ionic_strength_M!=.25:
  raise ValueError('Golding1995 constants only38C/I0.25M; use the separate Teague1996 temperature scenario')
 H=10**(-pH);M=free_Mg_mM/1000;k=GOLDING38
 species={}
 for n in ('ATP','ADP','AMP','PCr','Pi'):
  w={'unprotonated':1.,'protonated':H/k['H'+n],'Mg_bound':M*k['Mg'+n]}
  if n in ('ATP','ADP'):w['Mg_protonated']=M*H*k['MgH'+n]/k['H'+n]
  s=sum(w.values());species[n]={'binding_polynomial':s,'fractions':{a:b/s for a,b in w.items()}}
 P={n:v['binding_polynomial'] for n,v in species.items()}
 return dict(CK_Kobs=k['CK']*H*P['ATP']/(P['ADP']*P['PCr']),
  AK_Kobs=k['AK']*P['ATP']*P['AMP']/P['ADP']**2,
  ATP_hydrolysis_Kobs_M=k['ATP_hydrolysis']/H*P['ADP']*P['Pi']/P['ATP'],
  species=species,pH=pH,free_Mg_mM=free_Mg_mM,temperature_C=38.,ionic_strength_M=.25,
  source='10.1242/jeb.198.8.1775 Table2, eq9/17/25; compare Tables3/4/5, not introductory Lawson166',
  human_MN_calibrated=False)

def ck_exchange(atp_uM,adp_uM,pcr_uM,total_creatine_uM,forward_per_uM_s,reverse_equilibrium_factor,rate_scale=1.):
 vals=(atp_uM,adp_uM,pcr_uM,total_creatine_uM,forward_per_uM_s,reverse_equilibrium_factor,rate_scale)
 if not all(math.isfinite(x) for x in vals) or min(vals)<0 or adp_uM<=0 or pcr_uM<=0 or pcr_uM>=total_creatine_uM or reverse_equilibrium_factor<=0:
  raise ValueError('Finite interior physical pools and positive equilibrium factor required')
 cr=total_creatine_uM-pcr_uM;k=forward_per_uM_s*rate_scale
 f=k*pcr_uM*adp_uM;b=k*reverse_equilibrium_factor*cr*atp_uM
 return dict(forward_pseudo_first_order_s_inv=k*adp_uM,forward_uM_s=f,reverse_uM_s=b,net_ATP_uM_s=f-b,
  ATP_ADP=atp_uM/adp_uM,CK_quotient=atp_uM*cr/(adp_uM*pcr_uM),CK_Kobs=1/reverse_equilibrium_factor,
  quotient_over_K=atp_uM*cr*reverse_equilibrium_factor/(adp_uM*pcr_uM),
  meaning='Gross forward exchange is comparable in definition to saturation-transfer kf; net ATP production is forward minus reverse. Tissue/temperature/compartment transfer remains uncalibrated.')
