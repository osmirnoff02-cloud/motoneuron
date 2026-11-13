"""[codex] Audit completed runs only; missing checks cannot become passes."""
from pathlib import Path
import json
import numpy as np
from scipy.signal import find_peaks
B=Path(__file__).resolve().parent;D=B.parent
TARGETS={'Burley':[3,4,3],'Selvaraj-Con1':[5,6,4],'Selvaraj-Con2':[5,6,7]}
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(name,q):(B/(name+'_codex.json')).write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    reduced={}
    for profile in TARGETS:
        p=B/f'reduced_search_{profile}_codex.json'
        if p.exists():reduced[profile]=read(p)
    common=set.intersection(*[{q['name'] for q in rr if q.get('count_square_error')==0 and q['name']!='C00'} for rr in reduced.values()]) if len(reduced)==3 else set()
    save('screen_summary',dict(completed={k:len(v) for k,v in reduced.items()},common_count_passes=sorted(common),
      scope='Short reduced screen only, frozen molecular pools. No full or human acceptance.'))
    rows=[];events=[]
    for p in sorted(B.glob('J13_electric_*_features_codex.json')):
        q=read(p);f=q['features'];on=q['start_s']+.2;off=on+(.5 if q['profile']=='Burley' else 1)
        z=np.load(p.with_name(p.name.replace('_features_codex.json','_trace_codex.npz')))
        t=z['time_s'];v=z['state'][:,185];keep=np.r_[True,np.diff(t)>0];t=t[keep];v=v[keep]
        ix,_=find_peaks(v,height=f['rest_mV']+15,prominence=10)
        ev=[]
        for i in ix:
            left=i-1
            while left>0 and v[left]>v[left-1] and t[i]-t[left]<.03:left-=1
            slope=float(np.max(np.diff(v[left:i+1])/np.diff(t[left:i+1]))/1000)
            ev.append(dict(time_s=float(t[i]),after_off_ms=float((t[i]-off)*1000),peak_mV=float(v[i]),rise_mV_ms=slope,
              phase='pre' if t[i]<on else 'pulse' if t[i]<off else 'post'))
        target=TARGETS[q['profile']][[15,25,50].index(q['current_pA'])]
        rows.append(dict(source=p.name,candidate=p.name.split('_')[2],profile=q['profile'],pA=q['current_pA'],start_s=q['start_s'],features=f,
          historical_count_target=target,count_pass=f['spikes']==target,
          pre_excursions=sum(x['phase']=='pre' for x in ev),
          delayed_post_excursions=sum(x['phase']=='post' and x['after_off_ms']>10 for x in ev)))
        events.append(dict(source=p.name,events=ev))
    initial=[r for r in rows if r['start_s']==0]
    groups={name:[r for r in initial if r['candidate']==name] for name in sorted({r['candidate'] for r in initial})}
    panels={k:dict(conditions=len(v),count_passes=sum(x['count_pass'] for x in v),complete_9of9=len(v)==9 and all(x['count_pass'] for x in v)) for k,v in groups.items()}
    save('electrical_summary',dict(rows=rows,panels=panels,
      warning='Counts are historical regression targets; 50pA Selvaraj not a measured human point. No waveform target is inferred from count pass.'))
    save('events',dict(rows=events,rule='rest+15mV, prominence10mV. Record all excursions, including slow and boundary ones; unchanged historical detector supplies count.'))
    pairs=[]
    for p in sorted(B.glob('J13_screen_*_load_metrics_codex.json')):
        q=read(p);rp=p.with_name(p.name.replace('_load_','_rest_'))
        if not rp.exists():continue
        ctl=read(rp);a=q['selected'];b=ctl['selected']
        pairs.append(dict(source=p.name,matched_final={k:a[k]['final']-b[k]['final'] for k in a},
          K_residual_uM=q['K_ledger_max_residual']))
    save('paired600',dict(rows=pairs,scope='Load minus own matched600s control; no instantaneous ion reset.'))
    longs=[]
    for p in sorted(B.glob('J13_load36_*_metrics_codex.json')):
        name=p.name.split('_')[2];ctl=B/f'J13_load36_{name}_control_0_metrics_codex.json'
        if not ctl.exists():continue
        q=read(p);co=read(ctl);a=q['selected'];b=co['selected']
        row=dict(source=p.name,candidate=name,matched_final={k:a[k]['final']-b[k]['final'] for k in a},
          minimum_process_ATP_percent_of_start=100*a['process_ATP']['min']/a['process_ATP']['initial'],
          minimum_process_PCr_percent_of_start=100*a['process_PCr']['min']/a['process_PCr']['initial'],
          K_terms_relative_to_control=[{k:q['K_terms'][j][k]-co['K_terms'][j][k] for k in q['K_terms'][j]} for j in range(2)],
          K_residual_uM=q['K_ledger_max_residual'])
        cfg=read(p.with_name(p.name.replace('_metrics_','_config_')))
        row['matched_K_molecules']={comp:row['matched_final'][comp+'_K']*vol*602.214076 for comp,vol in zip(('soma','process'),cfg['mobile_water_volumes']['mobile_water_volume_um3'])}
        longs.append(row)
    save('long_summary',dict(rows=longs,scope='72h preparation +36h continuation, each load compared to own control. Extrema at stored300s points, not continuous-time guarantees.'))
    allchecks=[]
    for p in B.glob('J13_*_metrics_codex.json'):
        q=read(p);allchecks.append(dict(source=p.name,finite=q['all_finite'],states=q['states'],K_residual_uM=q['K_ledger_max_residual'],largest_check=max(q['checks'].values())))
    save('completed_checks',dict(rows=allchecks,complete_full_runs=len(allchecks)))
    print(json.dumps(dict(reduced={k:len(v) for k,v in reduced.items()},screen_count_passes=sorted(common),full_runs=len(allchecks),initial_electrical=len(initial),initial_passes=sum(r['count_pass'] for r in initial),longs=len(longs))))
if __name__=='__main__':main()
