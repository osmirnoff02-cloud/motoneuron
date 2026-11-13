"""[codex]2026-10-31. Source-specific assays, NOT human-MN kinetic parameters.

Nyame2025 values are preserved per replicate. Interpolation stays inside the
measured range; no extrapolation to cellular clearance, cholesterol or ALS.
"""
from pathlib import Path
import json,numpy as np
B=Path(__file__).resolve().parent

def evidence():return json.loads((B/'organelle_evidence_codex.json').read_text(encoding='utf8'))

def _source(number,sheet):
 data=json.loads((B/'PLA2G15_source_values_codex.json').read_text(encoding='utf8'))
 return data[f'41586_2025_8942_MOESM{number}_ESM.xlsx'][sheet]

def _interp(x,values,query):
 x=np.asarray(x,float);values=np.asarray(values,float);q=np.asarray(query,float)
 if not np.isfinite(q).all() or np.any(q<x.min()) or np.any(q>x.max()):raise ValueError('Outside measured source range; extrapolation prohibited')
 return np.array([np.interp(q,x,v) for v in values.T]).T

def bmp_assay_timecourse(time_min,form='2,2',system='purified_PLA2G15'):
 """Nyame Fig2g: resolved positional peaks, three replicates, normalized0min.

20uM mixture,100nM recombinant PLA2G15; figure states pH5 whereas Methods
state pH4.5. Condition discrepancy retained, no physiological fit inferred.
"""
 if system!='purified_PLA2G15':raise ValueError('This table is a purified-enzyme assay, not human motoneuron calibration')
 if form not in ('2,2','2,3','3,3'):raise ValueError('Expected positional form2,2/2,3/3,3')
 rows=np.array(_source(14,'2g')[4:],float);j=1+3*('2,2','2,3','3,3').index(form)
 a=_interp(rows[:,0],rows[:,j:j+3],time_min)
 return dict(replicates=a,mean=np.mean(a,axis=-1),sd=np.std(a,axis=-1,ddof=1),unit='normalized resolved BMP peak',source='Nyame2025 Fig2g',system=system)

def bmp_assay_pH(pH):
 """Relative assay rate, maximum measured mean atpH5, not lysosomal pH model."""
 rows=np.array(_source(21,'4a')[8:],float);a=_interp(rows[:,0],rows[:,1:],pH)
 return dict(replicates_nmoles_min=a,mean_nmoles_min=np.mean(a,axis=-1),relative_to_maximum=np.mean(a,axis=-1)/np.max(rows[:,1:].mean(1)),source='Nyame2025 ExtendedData4a; purified PLA2G15,3,3BMP',human_calibrated=False)

def organelle_signal(organelle_amount,other_amount,recovery,response_factor,contamination_fraction):
 """Linear assay observation, amounts in compatible units; no fitted yields.

Supports separate whole-cell and enriched-fraction validation. A recovery
factor is required: enrichment cannot silently become organelle concentration.
"""
 vals=np.array([organelle_amount,other_amount,recovery,response_factor,contamination_fraction],float)
 if not np.isfinite(vals).all() or min(vals)<0 or recovery>1 or contamination_fraction>1:raise ValueError('Finite nonnegative inputs, recovery/contamination0..1 required')
 return response_factor*(recovery*organelle_amount+contamination_fraction*other_amount)
