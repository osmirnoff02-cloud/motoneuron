"""[codex] Exact cumulative K owners; positive means K added to cytosol."""
from pathlib import Path
import json,numpy as np
B=Path(__file__).resolve().parent;D=B.parent;S=D/'2026-11-11-mn-pump-loads'
def observe(z,config):
 z=np.asarray(z,dtype=float)
 if z.ndim!=2 or z.shape[0]<1 or z.shape[1]<239 or not np.isfinite(z).all():raise ValueError('Finite trajectory with ion ledgers required')
 if 'ion_background_reference' not in config:raise ValueError('Explicit ion-background volume reference required')
 rows=[]
 for j,comp in enumerate(('soma','process')):
  k=199+20*j
  # Reference is conversion (1e9/F) / accessible cytosolic volume.
  vol=(1e9/96485.33212)/config['ion_background_reference'][j][6]
  if not np.isfinite(vol) or vol<=0:raise ValueError('Positive accessible volume required')
  dz=z-z[0]
  terms=dict(channels=-dz[:,k+6],pump=2*dz[:,k+4],background=dz[:,k+9],axial=dz[:,k+13]/vol)
  if 'mobile_water_volumes' in config:
   actual_volume=config['mobile_water_volumes']['mobile_water_volume_um3'][j]
   terms={key:value*vol/actual_volume for key,value in terms.items()};vol=actual_volume
  actual=dz[:,k+1];residual=actual-sum(terms.values())
  rows.append(dict(compartment=comp,volume_um3=vol,terms=terms,actual=actual,residual=residual))
 return rows
def main():
 traces={};data=[]
 for label in ('control','continuous','repeated','continuous_strong','repeated_strong'):
  a=np.load(S/f'MN11_{label}_trace_codex.npz');z=a['state'];t=a['time_s']
  cfg=json.loads((S/f'MN11_{label}_config_codex.json').read_text());o=observe(z,cfg)
  traces[label]=(t,o)
  rows=[]
  for j,q in enumerate(o):
   ct,co=traces['control'];ctl=co[j]
   row=dict(compartment=q['compartment'],volume_um3=q['volume_um3'],
    own_K_change_mM=float(q['actual'][-1]/1000),
    matched_K_change_mM=float((q['actual'][-1]-ctl['actual'][-1])/1000),
    exact_balance_residual_max_uM=float(abs(q['residual']).max()),
    own_terms_mM={k:float(v[-1]/1000) for k,v in q['terms'].items()},
    matched_terms_mM={k:float((v[-1]-ctl['terms'][k][-1])/1000) for k,v in q['terms'].items()})
   rows.append(row)
  data.append(dict(label=label,compartments=rows));print(label,rows[1],flush=True)
 result=dict(date='2026-11-12',author='codex',positive_sign='K entering cytosol',rows=data,
  scope='Exact cumulative material owners on existing trajectories; channel total includes HH/SK/differential KCNQ/AMPA. Not causal interventions.')
 (B/'ledger_audit_codex.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 return result
if __name__=='__main__':main()
