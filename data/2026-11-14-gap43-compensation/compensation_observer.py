"""GAP43 supply/remove/resource budgets; no modification of a physical state."""
import numpy as np

SUPPLY_RANGES={'production_multiplier':[1.,2.,4.],'departure_multiplier':[1.,2.,4.],
 'arrival_multiplier':[1.,2.,4.],'correctly_spliced_fraction':[0.,.5,1.]}

def source_constraints():
 return dict(
  human_function=dict(title='Cryptic Splicing of GAP43 mRNA is a Novel Hallmark of TDP-43-Associated ALS and AD',doi='10.1002/advs.202412054',
   system='Human iPSC motor neurons, lentiviral TDP-43/GAP43 knockdown and GAP43 rescue; Fig8',
   outcome='Axonal regrowth24/48h; qualitative rescue, not ATP/translation/transport flux calibration',
   caveat='Individual neurites are not independent cultures;3devices from2experiments; no single dosage-to-rate conversion'),
  transport=dict(doi='10.1111/jnc.12266',system='Rat DRG and sciatic-nerve axoplasm',RNA_half_h=[5.6,8.3],
   status='Two experimental conditions, not human range bounds; RNA half-life is not protein half-life'),
  decay=dict(doi='10.1093/nar/gkac337',system='Rodent axonal KHSRP and localized mRNA decay',
   caveat='Qualitative post-transcriptional mechanism, no fitted human degradation constant'),
  scope='Model supply denotes productive GAP43 mRNA. More total RNA does not ensure more normal protein.',
  missing=['Axonal nascent GAP43/turnover/ATP in same human MN culture','Absolute message arrival and shared HuD/ZBP1/ACTB competition',
           'Productive/cryptic GAP43 fractions with TDP-43 function and rescue dosage'])

def observe(time_s,state,config):
 t=np.asarray(time_s,float);z=np.asarray(state,float)
 if t.ndim!=1 or len(t)<2 or z.shape!=(len(t),4431) or not np.all(np.diff(t)>0) or not np.isfinite(z).all():raise ValueError('Physical4431-state trajectory required')
 v=np.asarray(config['mobile_water_volumes']['reference_volume_um3']);scale=config['biogenesis']['selected_mature_pool_uM']*v[0]/v[1]
 delta=z-z[0];d=delta[-1];bound=z[:,578:583].sum(1);free=np.minimum(z[:,564],z[:,565]);occupied=bound/(free+bound)
 last=t>=t[-1]-6*3600;drift=float(np.polyfit((t[last]-t[-1])/86400,z[last,583],1)[0]/z[0,583])
 pools={name:dict(start=float(z[0,i]),final=float(z[-1,i]),min_relative=float(z[:,i].min()/z[0,i])) for name,i in
  {'ATP_soma':120,'ATP_axon':136,'PCr_axon':316,'AA_soma':497,'AA_axon':574,'GTP_soma':499,'GTP_axon':576,'charged_tRNA_axon':575,'free40S_soma':330,'free60S_soma':331,'free40S_local':564,'free60S_local':565}.items()}
 # Separate cumulative costs. Do not add GTP expenditure on top of NDPK ATP.
 cost_local=d[594]+d[595]+d[598]+d[592]-d[593]+d[616]
 cost_soma=d[614]+d[615]
 message_bound=delta[:,578:582].sum(1)*scale
 ratio=v[0]/v[1]
 balances=dict(protein=float(np.max(np.abs(delta[:,583]-delta[:,586]*scale+delta[:,596]))),
  RNA_soma=float(np.max(np.abs(delta[:,605]-delta[:,608]+delta[:,609]+delta[:,610]))),
  RNA_transit=float(np.max(np.abs(delta[:,606]-delta[:,610]+delta[:,611]+delta[:,612]))),
  RNA_local=float(np.max(np.abs(delta[:,607]+message_bound-ratio*delta[:,612]+delta[:,613]))))
 resource_ok=min(pools[k]['min_relative'] for k in ('ATP_soma','ATP_axon'))>=.95 and min(pools[k]['min_relative'] for k in ('AA_soma','AA_axon','GTP_soma','GTP_axon'))>=.8
 imin=int(np.argmin(z[:,583]));below=np.flatnonzero(z[:,583]<.9*z[0,583])
 sustained_recovery=0. if not len(below) else (float((t[below[-1]+1]-t[0])/3600) if below[-1]+1<len(t) else None)
 return dict(GAP43_initial_uM=float(z[0,583]),GAP43_final_uM=float(z[-1,583]),GAP43_final_relative=float(z[-1,583]/z[0,583]),
  GAP43_min_relative=float(z[imin,583]/z[0,583]),GAP43_min_at_h=float((t[imin]-t[0])/3600),
  GAP43_sustained_recovery90_h=sustained_recovery,
  transient_metrics_status='Descriptive added metrics; predeclared endpoint compensation criterion unchanged',
  late_daily_drift_relative=drift,compensation_criterion=bool(z[-1,583]>=.9*z[0,583] and abs(drift)<=.05),
  resource_criterion=bool(resource_ok),criteria_are_scenario_choices=True,pools=pools,
  local_ribosome_occupied_fraction_start=float(occupied[0]),local_ribosome_occupied_fraction_max=float(occupied.max()),
  occupancy_denominator='Free limiting local subunit plus GAP43 elongation/posttermination particles; not all local proteome occupancy',
  free_RNA_mean_copies_start=float(z[0,607]*v[1]*602.214076),
  free_RNA_mean_copies_final=float(z[-1,607]*v[1]*602.214076),
  discrete_cargo_noise_modelled=False,
  GAP43_made_uM=float(d[586]*scale),GAP43_removed_uM=float(d[596]),AA_charged_uM=float(d[589]),
  GTP_used_uM=float(d[590]+d[591]),local_ATP_equiv_cost_uM=float(cost_local),somatic_RNA_ATP_equiv_cost_uM=float(cost_soma),
  ATP_cost_soma_equivalent_uM=float(cost_soma+cost_local/ratio),material_residual_uM=balances,
  scope='Finite model pools with inherited nutrient/ATP supply; not a closed unfed neuron, human calibration or measured outgrowth.')
