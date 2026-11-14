"""[codex] Read-only matched-control charge diagnostic, not a pH model.

Explicit Ca storage includes cytosolic buffer, ER and matrix; mobile-water
and reference-volume conventions are not interchangeable. PMCA countercharge
is reported as a hypothetical missing H-equivalent ledger, never applied to
the RHS, K, voltage, ATP or any checkpoint.
"""
from pathlib import Path
import json, math
B=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
columns=read(B/'columns.json')
for i,label in {130:'PMCA_gross_Ca_uM',144:'process_PMCA_Ca_uM',147:'Ca_exchange_amount_uM_um3',
 199:'soma_Na_i_uM',200:'soma_K_i_uM',219:'process_Na_i_uM',220:'process_K_i_uM',
 4413:'soma_Cl_i_uM',4419:'process_Cl_i_uM'}.items():assert columns[i]==label
rows=[]
for p in sorted(B.glob('J13_load36_*_checkpoint.json')):
    if not p.with_name(p.name.replace('_checkpoint_','_metrics_')).exists():continue
    name=p.name.split('_')[2]
    ctl=B/f'J13_load36_{name}_control_0_checkpoint.json'
    if not ctl.exists() or p==ctl:continue
    q,c=read(p),read(ctl)
    assert q['time_s']==c['time_s'] and q['fingerprint']==c['fingerprint']
    d=[a-b for a,b in zip(q['state'],c['state'])]
    cfg=read(p.with_name(p.name.replace('_checkpoint_','_config_')))
    water=cfg['mobile_water_volumes'];v=water['mobile_water_volume_um3'];ref=water['reference_volume_um3']
    geom=cfg['geometry'];areas=[geom['soma_area_um2'],2*math.pi*geom['process_radius_um']*geom['process_length_um']]
    cap=[cfg['HH_context'][0]*a/sum(areas) for a in areas]
    calcium=[(1+cfg['editing']['kappa'])*d[0]+d[115]+d[239],(1+geom['process_kappa'])*d[134]+d[243]]
    out=[]
    for j,(na,ki,cl,vm,pmca) in enumerate([(199,200,4413,185,130),(219,220,4419,192,144)]):
        monovalent=d[na]+d[ki]-d[cl]
        ca=2*calcium[j]*ref[j]/v[j]
        counter=2*d[pmca]*ref[j]/v[j]
        capacitive=cap[j]*d[vm]*10.36426965661773/v[j]
        # Ca diffusion moves charge between compartments in the biochemical
        # model but is absent from the axial voltage current. Its contribution
        # cancels globally; expose it rather than mislabel local error as PMCA.
        axial_ca=2*d[147]/v[j]*(1 if j==0 else -1)
        out.append(dict(compartment=['soma','process'][j],K_delta_uM=d[ki],Na_delta_uM=d[na],Cl_delta_uM=d[cl],
          monovalent_charge_delta_uM=monovalent,stored_Ca_charge_delta_mobile_uM=ca,
          excess_PMCA_Ca_export_mobile_uM=counter/2,
          hypothetical_2H_per_Ca_countercharge_mobile_uM=counter,capacitive_charge_delta_mobile_uM=capacitive,
          residual_without_countercharge_uM=monovalent+ca-capacitive,
          residual_with_hypothetical_countercharge_uM=monovalent+ca+counter-capacitive,
          axial_Ca_charge_bookkeeping_correction_uM=axial_ca,
          residual_with_countercharge_and_axial_Ca_uM=monovalent+ca+counter-capacitive+axial_ca,
          explicit_warning='Untracked proton-equivalent flux, not free H concentration or inferred pH. Other missing charge species/fluxes are not ruled out.'))
    rows.append(dict(source=p.name,candidate=name,compartments=out,
      weighted_residual_without_uM=sum(x['residual_without_countercharge_uM']*vol for x,vol in zip(out,v))/sum(v),
      weighted_residual_with_countercharge_uM=sum(x['residual_with_hypothetical_countercharge_uM']*vol for x,vol in zip(out,v))/sum(v)))
    assert abs(rows[-1]['weighted_residual_with_countercharge_uM'])<1e-4,rows[-1]
result=dict(author='codex',date='2026-11-13',rows=rows,
 status='Diagnostic only; no numerical trajectory changed. PMCA H countertransport and acid extrusion are not dynamically represented in this ion block.',
 interpretation='A near match identifies missing bookkeeping/physiology, not proof that an electrogenic PMCA current should be added. Electroneutral PMCA needs explicit H-equivalent balance and a separately justified acid-recovery route.',
 sources=[dict(title='The plasma membrane calcium ATPase (PMCA) of neurones is electroneutral and exchanges 2 H+ for each Ca2+ or Ba2+ ion extruded',doi='10.1113/jphysiol.2008.162453',system='Voltage-clamped snail neurons, 18-22 C; not human iPSC-MNs',read='Publisher HTML methods/results/discussion',url='https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/jphysiol.2008.162453'),
 dict(title='Ca2+ transport by reconstituted synaptosomal ATPase is associated with H+ countertransport and net charge displacement',doi='10.1074/jbc.273.29.18230',read='PubMed indexed abstract; contrasting reconstituted preparation',url='https://pubmed.ncbi.nlm.nih.gov/9660785/')])
(B/'charge_diagnostic.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps([dict(source=r['source'],without=r['weighted_residual_without_uM'],with_counter=r['weighted_residual_with_countercharge_uM']) for r in rows]))
