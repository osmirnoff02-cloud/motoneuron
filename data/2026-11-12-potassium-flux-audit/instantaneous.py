"""[codex] Pure diagnostic: instantaneous K owners on saved trajectories.

Positive means K added to cytosol. Differential KCNQ is not full Kv7.
This does not change the integrator and does not infer causal effects.
"""
import numpy as np
FCONV=1e9/96485.33212

def observe(state,baseline,config,opening,*,process_ampa_fraction,capacity=1.,pump_support=1.,alpha3_support=1.,kcnq_multiplier=1.):
 z=np.atleast_2d(np.asarray(state,dtype=float));base=np.asarray(baseline,dtype=float)
 if z.ndim!=2 or not np.isfinite(z).all() or z.shape[1]<364 or base.ndim!=1 or len(base)<364 or not np.isfinite(base).all():raise ValueError('Finite compatible states and one baseline required')
 op=np.asarray(opening,dtype=float)
 if op.ndim>1 or (op.ndim==1 and op.shape!=(len(z),)) or not np.isfinite(op).all() or np.any((op<0)|(op>1)):raise ValueError('One opening per sample, or scalar, in [0,1] required')
 if not 0<=process_ampa_fraction<=1 or not np.isfinite([capacity,pump_support,alpha3_support,kcnq_multiplier]).all() or min(capacity,pump_support,alpha3_support,kcnq_multiplier)<0:raise ValueError('Nonnegative explicit supports and AMPA spatial fraction [0,1] required')
 for key in ('ion_background_reference','ion_baseline','geometry','ions','HH_parameters','HH_context','editing','kcnq_transport','kcnq_coupling','bulk_coupling','basal_ca_uM','feedback','ion_background'):
  if key not in config:raise ValueError('Missing explicit configuration: '+key)
 p=np.asarray(config['HH_parameters']);c=np.asarray(config['HH_context']);ip=config['ions'];geo=config['geometry'];ref=np.asarray(config['ion_background_reference']);ib=np.asarray(config['ion_baseline'])
 rna=config.get('rna_calcium',{})
 if not ip['enabled']:raise ValueError('Dynamic ion branch required')
 if rna.get('fxyd7_strength',0) or rna.get('nmda_nS',0):raise ValueError('Active FXYD/NMDA needs an extended observer; refusing omitted K owners')
 theta=.08617333262*(273.15+ip['temperature_C']);ep=config['editing'];kp=config['kcnq_transport'];kc=config['kcnq_coupling']
 volumes=FCONV/ref[:,6];area=np.array([geo['soma_area_um2'],2*np.pi*geo['process_radius_um']*geo['process_length_um']]);frac=area/area.sum()
 if not np.isfinite(volumes).all() or np.any(volumes<=0):raise ValueError('Positive finite accessible volumes required')
 if not 0<=config['ion_background']['passive_fraction']<=1:raise ValueError('Passive fraction must be in [0,1]')
 if any(base[s]+kp['mutant_residual_function']*base[s+6]<=0 for s in (346,348)):raise ValueError('Positive KCNQ surface reference required')
 conn=1/(geo['soma_length_um']/2/(np.pi*geo['soma_diameter_um']**2/4)+geo['process_length_um']/2/(np.pi*geo['process_radius_um']**2))
 axial_g=100000*conn/geo['axial_resistivity_ohm_cm']
 # Chemical diffusion and conventional axial current are separately observable.
 diff=ip['ion_diffusion_um2_s']*conn*(z[:,220]-z[:,200])
 electrical=(1-ip['axial_Na_fraction'])*axial_g*(z[:,192]-z[:,185])*FCONV
 out=[]
 # AMPA fractions must match the original current partition.
 weights=np.asarray(ep['weights']) if 'weights' in ep else None
 if weights is None:raise ValueError('Explicit AMPA tetramer weights required')
 cp=sum(weights[n]*(1-np.clip(z[:,106],0,1))**n for n in range(5))
 gcp=ep['g_reference_nS']*cp*ep['cp_conductance_ratio']*ep['cp_unblocked_fraction']
 gci=ep['g_reference_nS']*(1-cp)
 gca=(gci*ep['pf_ci']+gcp*ep['pf_cp'])/3;gmono=gcp+gci-gca
 emono=np.divide(-gca*120,gmono,out=np.zeros_like(gca),where=gmono>0)
 naf=(emono-c[4])/(c[3]-c[4])
 for j in range(2):
  b=199+20*j;vi=185+7*j;v=z[:,vi];fac=ref[j,6];cap=ref[j,5]
  if np.any(z[:,b:b+4]<=0):raise ValueError('Positive ionic concentrations required')
  ek=theta*np.log(z[:,b+3]/z[:,b+1]);ena=theta*np.log(z[:,b+2]/z[:,b])
  s=346+2*j;ratio=np.maximum(0,1+kp['electrical_transfer_gain']*((np.maximum(z[:,s],0)+kp['mutant_residual_function']*np.maximum(z[:,s+6],0))/(base[s]+kp['mutant_residual_function']*base[s+6])-1))
  ikfast=cap*p[3]*z[:,vi+3]**4*(v-ek);ikslow=cap*p[12]*z[:,vi+4]*(v-ek)
  ikcnq=kcnq_multiplier*(ratio-1)*kc['g_reference_nS']*frac[j]*z[:,362+j]*(v-ek)
  bc=config['bulk_coupling'];ca0=config['basal_ca_uM'];sk0=ca0**bc['sk_hill']/(bc['sk_half_uM']**bc['sk_hill']+ca0**bc['sk_hill'])
  isk=config['feedback']['g_sk_nS']*frac[j]*(z[:,vi+6]-sk0)*(v-ek)
  # Spatial AMPA share is scheduled separately, not the area fraction.
  af=process_ampa_fraction if j else 1-process_ampa_fraction
  iampa=gmono*(1-naf)*af*op*capacity*(v-ek)
  bg=(1-config['ion_background']['passive_fraction'])*ib[j,4]-config['ion_background']['passive_fraction']*ref[j,4]*(v-ek)*fac
  na=z[:,b];ko=z[:,b+3];atp=z[:,120 if j==0 else 136]
  satNa=na**ip['pump_Na_hill']/(ip['pump_Na_half_uM']**ip['pump_Na_hill']+na**ip['pump_Na_hill']);satK=ko/(ip['pump_K_half_uM']+ko);satATP=atp/(ip['pump_ATP_half_uM']+atp)
  rp=config.get('recovery',{})
  if rp.get('enabled',False):
   if rp['kinetic_family'] not in ('fixed_hill3','variable_hill'):raise ValueError('Unsupported pump kinetic family')
   k1,k3,h1,h3=(12000.,44000.,3.,3.) if rp['kinetic_family']=='fixed_hill3' else (12000.,33000.,3.5,6.8)
   f3=rp['alpha3_max_fraction'];na0=ip['sodium_out0_uM']*np.exp(-c[3]/theta)
   shape=(1-f3)*na**h1/(k1**h1+na**h1)+f3*alpha3_support*na**h3/(k3**h3+na**h3)
   shape0=(1-f3)*na0**h1/(k1**h1+na0**h1)+f3*na0**h3/(k3**h3+na0**h3)
   satNa=shape/shape0*na0**ip['pump_Na_hill']/(ip['pump_Na_half_uM']**ip['pump_Na_hill']+na0**ip['pump_Na_hill'])
  vmax=ip['max_pump_cycles_uM_s']*(area[j]/volumes[j])/(area[0]/volumes[0])*(ib[j,9] if ib.shape[1]>9 else 1)
  cycles=vmax*satNa*satK*satATP*pump_support
  if 'pump_response' in config:
   from pathlib import Path
   from model_contracts import load_module
   pm=load_module(Path(__file__).resolve().parents[1]/'2026-11-13-pump-response-test/pump_response.py','_stable_pump_response13')
   pv=np.asarray(config['pump_response']['vector'],float)
   cycles=np.array([pm.cycles(n,k,a,pv,j,pump_support) for n,k,a in zip(na,ko,atp)])
   satNa=np.array([pm.substrate(n/1000.,pv[j,0],pv[j,1]) for n in na])
   satK=np.array([pm.substrate(k/1000.,pv[j,2],pv[j,3]) for k in ko])
   satATP=atp/(pv[j,5]+atp)
  terms=dict(HH_K_fast=-ikfast*fac,HH_slow_K_unidentified=-ikslow*fac,differential_KCNQ=-ikcnq*fac,SK_increment=-isk*fac,AMPA_K=-iampa*fac,passive_background_K=bg,pump_K=2*cycles,axial_diffusion=(1 if j==0 else -1)*diff/volumes[j],axial_electrical=(1 if j==0 else -1)*electrical/volumes[j])
  if 'absolute_kcnq' in config:
   ap=config['absolute_kcnq']['parameters']
   terms['HH_slow_K_unidentified']*=1-ap['replace_slow_fraction']
   terms['differential_KCNQ']=np.zeros(len(z))
   terms['absolute_KCNQ']=-ap['g_total_nS']*frac[j]*ratio*z[:,362+j]*(v-ek)*fac
  actual_volume=volumes[j]
  if 'mobile_water_volumes' in config:
   actual_volume=config['mobile_water_volumes']['mobile_water_volume_um3'][j]
   terms={key:value*volumes[j]/actual_volume for key,value in terms.items()}
  out.append(dict(terms_uM_s=terms,total_uM_s=sum(terms.values()),volume_um3=float(actual_volume),voltage_mV=v,EK_mV=ek,ENa_mV=ena,pump_pA=cycles/fac,pump_pA_pF=cycles/fac/cap,pump_saturation_Na=satNa,pump_saturation_K=satK,pump_saturation_ATP=satATP,KCNQ_surface_ratio=ratio))
 return out
