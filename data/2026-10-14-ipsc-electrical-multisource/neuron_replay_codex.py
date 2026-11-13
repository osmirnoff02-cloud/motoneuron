"""Independent NEURON replay; parameter fitting never imports this module."""
from pathlib import Path
import json,csv
import numpy as np
from neuron import h
from electrical_model_codex import B,derived,spike_features,T
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'neuron-model'))
from model_contracts_codex import mechanism_path
h.load_file('stdrun.hoc')
if not hasattr(h,'ipscmulti'):h.nrn_load_dll(str(mechanism_path('ipsc-electrical-multisource-2026-10-14')))
h.cvode_active(0)
def run(p,c,current=0.,duration=500.,dt=.01,pre=100.,post=100.,rest=False,init=None,vc=None,vc_hold=None,rs=.001):
 s=h.Section(name='ipsc_multi');s.L=s.diam=np.sqrt(c[0]*100/np.pi);s.cm=1;s.nseg=1;s.insert('ipscmulti');k=s(.5).ipscmulti
 for name,val in zip(['gna','vm','km','gk','vn','kn','vh','kh','tm','thmin','threc','tn','ga','ta','tmfast','tndeact'],p):setattr(k,name,val/1000 if name in ('gna','gk','ga') else val)
 gl,el,hold=derived(p,c);k.gl=gl/1000;k.el=el;k.ena0=c[3];k.ek0=c[4]
 stim=h.IClamp(s(.5));stim.delay=pre;stim.dur=duration;stim.amp=current/1000
 bias=h.IClamp(s(.5));bias.delay=0;bias.dur=1e9;bias.amp=0 if rest or vc is not None else hold/1000
 clamp=None
 if vc is not None:
  clamp=h.SEClamp(s(.5));clamp.dur1=pre;clamp.amp1=c[5] if vc_hold is None else vc_hold;clamp.dur2=duration;clamp.amp2=vc;clamp.dur3=post;clamp.amp3=clamp.amp1;clamp.rs=rs
 ts=h.Vector().record(h._ref_t);vs=h.Vector().record(s(.5)._ref_v);nas=h.Vector().record(k._ref_ina0);ks=h.Vector().record(k._ref_ik0);ls=h.Vector().record(k._ref_il0)
 h.dt=dt;h.steps_per_ms=1/dt;h.secondorder=0;h.finitialize(c[5] if init is None else init);h.continuerun(pre+duration+post)
 out={'t':np.array(ts),'v':np.array(vs),'ina':np.array(nas)*1000,'ik':np.array(ks)*1000,'il':np.array(ls)*1000,'holding_pA':float(hold),'area_um2':float(h.area(.5,sec=s))}
 out['features']=spike_features(out['v'],dt,pre,duration)
 h.delete_section(sec=s);return out
def passive(p,c):
 rest=run(p,c,duration=3000,pre=0,post=0,rest=True,init=-70)
 tail=rest['v'][rest['t']>2800];cr=c.copy();cr[5]=c[6] if len(c)>6 else c[5]
 zero=run(p,cr,0);minus=run(p,cr,-1)
 r=float(np.mean(zero['v'][(zero['t']>550)&(zero['t']<599)])-np.mean(minus['v'][(minus['t']>550)&(minus['t']<599)]))
 perturb=[run(p,c,duration=3000,pre=0,post=0,rest=True,init=c[2]+d)['v'][-1] for d in (-1,1)]
 return {'Cm_pF':float(rest['area_um2']*.01),'Rin_GOhm':r,'Rin_reference_mV':float(cr[5]),'rest_mV':float(np.mean(tail)),'tail_range_mV':float(np.ptp(tail)),'perturbed_final_mV':[float(x) for x in perturb],'holding_pA':float(derived(p,c)[2]),'Rin_measurement_bias_pA':zero['holding_pA'],'stable':bool(np.ptp(tail)<.2 and abs(np.mean(tail)-c[2])<5)},rest
def burley():
 frozen=json.loads((B/'burley-local_codex.json').read_text());p=np.array(frozen['parameters']);c=np.array(frozen['context'])
 passive_result,rest=passive(p,c);rows=[];traces={'rest_t':rest['t'],'rest_v':rest['v']};firesults=[]
 for t in T['targets']:
  metric=t['metric'];x=t['input']
  if metric.startswith('FI'):
   a=run(p,c,x);fine=run(p,c,x,dt=.005);pred=a['features']['zero_crossings'];other=a['features']['count'];tolerance=max(1,t['sem'])+.05
   traces[f'FI_{x}_t']=a['t'];traces[f'FI_{x}_v']=a['v'];firesults.append({'current_pA':x,'counts_dt01':pred,'counts_dt005':fine['features']['zero_crossings'],'prominence_count':other,'max_dvdt':a['features'].get('max_dvdt'),'halfwidth_ms':a['features'].get('halfwidth_ms')})
  else:
   a=run(p,c,vc=x,duration=100);mask=(a['t']>100.005)&(a['t']<200)
   pred=float(np.min(a['ina'][mask])) if metric.startswith('Na') else float(a['ik'][np.where(mask)[0][-1]])
   tolerance=10.
  rows.append({**t,'role':'reused_development','previous_role':t['role'],'prediction':pred,'error':pred-t['value'],'limit':tolerance,'within_limit':bool(abs(pred-t['value'])<=tolerance)})
 summary={'author':'codex','date':'2026-10-15','software':h.nrnversion(),'passive':passive_result,'FI_details':firesults,'numerical_counts_agree':all(r['counts_dt01']==r['counts_dt005'] for r in firesults),'metrics':{}}
 for metric in ('FI_AP_per_500ms','Na_peak_pA_per_pF','K_late_pA_per_pF'):
  for subset in ('all','previous_test'):
   rr=[r for r in rows if r['metric']==metric and (subset=='all' or r['previous_role']=='test')]
   summary['metrics'][metric+'_'+subset]={'n':len(rr),'MAE':float(np.mean([abs(r['error']) for r in rr])),'RMSE':float(np.sqrt(np.mean([r['error']**2 for r in rr]))),'passed':sum(r['within_limit'] for r in rr)}
 # Measurement sensitivity only, not an inferred electrode resistance.
 rsrows=[]
 for rs in (5.,10.,20.):
  for x in (-30.,-10.,30.,70.):
   a=run(p,c,vc=x,duration=100,rs=rs);mask=(a['t']>100.005)&(a['t']<200)
   rsrows.append({'series_MOhm_scenario':rs,'command_mV':x,'min_Na_pA_per_pF':float(np.min(a['ina'][mask])),'last_K_pA_per_pF':float(a['ik'][np.where(mask)[0][-1]]),'last_actual_mV':float(a['v'][np.where(mask)[0][-1]])})
 summary['series_resistance_sensitivity']=rsrows
 (B/'burley-neuron-verification_codex.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 with (B/'burley-comparison_codex.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 np.savez_compressed(B/'burley-neuron-traces_codex.npz',**traces)
 print(json.dumps({k:v for k,v in summary.items() if k!='series_resistance_sensitivity'},indent=2),flush=True)
if __name__=='__main__':burley()
