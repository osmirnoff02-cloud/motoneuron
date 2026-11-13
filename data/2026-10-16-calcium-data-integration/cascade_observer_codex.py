"""Codex 2026-10-16. Small empirical AP->fluorescence observer, NOT the CASCADE neural net.

Effective biexponential kernels include cell Ca handling and reporter response.
Held-out metric is scale-free, linearly detrended Pearson correlation, not absolute fluorescence accuracy.
"""
from pathlib import Path
import json, zipfile, io, csv
import numpy as np
from scipy.io import loadmat
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B=Path(__file__).resolve().parent
SRC=B.parent/'2026-10-16-optopatch-groundtruth'

def kernel_predict(counts,dt,rise,decay):
    if rise<=0 or decay<=rise:raise ValueError('0 < rise < decay required')
    x=signal.lfilter([1],[1,-np.exp(-dt/decay)],counts)-signal.lfilter([1],[1,-np.exp(-dt/rise)],counts)
    return x

def correlation(a,b):
    a=signal.detrend(a);b=signal.detrend(b)
    return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-20))

def get_trials():
    rng=np.random.default_rng(20261016);splits={};out=[]
    # Splits persisted before any fit/fluorescence metric is computed.
    for p in sorted(SRC.glob('DS*_source_codex.zip')):
        ds=p.name.removesuffix('_source_codex.zip')
        with zipfile.ZipFile(p) as z:
            names=sorted(z.namelist());perm=rng.permutation(len(names));test=set(perm[:int(np.ceil(.3*len(names)))])
            splits[ds]={n:('test' if i in test else 'train') for i,n in enumerate(names)}
    path=B/'cascade-cell-split_codex.json'
    if path.exists():assert json.loads(path.read_text())==splits,'Refuse changed split'
    else:path.write_text(json.dumps(splits,indent=2),encoding='utf-8')
    for p in sorted(SRC.glob('DS*_source_codex.zip')):
        ds=p.name.removesuffix('_source_codex.zip');requested_dt=.05 if ds.startswith('DS10') else .1
        with zipfile.ZipFile(p) as z:
            for n in sorted(z.namelist()):
                trials=loadmat(io.BytesIO(z.read(n)),simplify_cells=True)['CAttached']
                if isinstance(trials,dict):trials=[trials]
                for j,trial in enumerate(trials):
                    t=np.ravel(trial['fluo_time']);f=np.ravel(trial['fluo_mean']);events=np.ravel(trial['events_AP'])/1e4
                    # Never upsample fluorescence. Bin width >= max native frame interval.
                    dt=max(requested_dt,float(np.max(np.diff(t)))*1.01)
                    edges=np.arange(t[0],t[-1]+dt*.001,dt)
                    nc,_=np.histogram(t,edges);sums,_=np.histogram(t,edges,weights=f)
                    # Bin times with fewer raw observations are missing; no interpolation across gaps.
                    if np.any(nc==0):raise ValueError('Missing fluorescence bin')
                    counts,_=np.histogram(events,edges);fluo=sums/nc;times=(edges[:-1]+edges[1:])/2
                    out.append(dict(dataset=ds,file=n,trial=j,split=splits[ds][n],t=times,f=fluo,counts=counts,dt=dt,
                                    usable=bool(counts.sum()>=5 and len(fluo)>=100 and np.std(fluo)>0),events=int(counts.sum())))
    return out

def cells_score(trials,rise,decay):
    groups={}
    for r in trials:
        if r['usable']:
            c=correlation(kernel_predict(r['counts'],r['dt'],rise,decay),r['f'])
            groups.setdefault(r['file'],[]).append(c)
    return float(np.mean([np.mean(v) for v in groups.values()]))

def main():
    trials=get_trials();results=[];profiles={};grid=[];examples={}
    for ds in sorted(set(r['dataset'] for r in trials)):
        group=[r for r in trials if r['dataset']==ds];train=[r for r in group if r['split']=='train']
        candidates=[]
        for rise in [.01,.03,.06,.1,.2,.4,.8]:
            for decay in [.1,.2,.4,.8,1.6,3.2,6.4]:
                if decay>rise:
                    score=cells_score(train,rise,decay);candidates.append((score,rise,decay))
                    grid.append(dict(dataset=ds,rise_s=rise,decay_s=decay,train_cell_mean_r=score))
        score,rise,decay=max(candidates)
        profiles[ds]=dict(rise_s=rise,decay_s=decay,bin_s=group[0]['dt'],train_cell_mean_r=score,
                           interpretation='effective fluorescence response; not isolated sensor kinetics or pump rate',
                           species='mouse',use='technical observation profile, not human MN physiology')
        for r in group:
            if not r['usable']:continue
            pred=kernel_predict(r['counts'],r['dt'],rise,decay)
            results.append(dict(dataset=ds,file=r['file'],trial=r['trial'],split=r['split'],events=r['events'],
                                instantaneous_r=correlation(r['counts'],r['f']),
                                generic_kernel_r=correlation(kernel_predict(r['counts'],r['dt'],.03,.4),r['f']),
                                learned_kernel_r=correlation(pred,r['f'])))
            # Predefined lexicographic first usable test trial, not best-looking example.
            if r['split']=='test' and ds not in examples:examples[ds]=(r,pred)
    summary=[]
    for ds,profile in profiles.items():
        cells={}
        for r in results:
            if r['dataset']==ds and r['split']=='test':cells.setdefault(r['file'],[]).append(r)
        m={metric:float(np.mean([np.mean([r[metric] for r in rows]) for rows in cells.values()])) for metric in ['instantaneous_r','generic_kernel_r','learned_kernel_r']}
        summary.append(dict(dataset=ds,heldout_cells=len(cells),heldout_segments=sum(map(len,cells.values())),**m,**profile))
    for name,values in [('cascade-scores',results),('cascade-training-grid',grid)]:
        with (B/(name+'_codex.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=values[0].keys());w.writeheader();w.writerows(values)
    output=dict(author='codex',date='2026-10-16',method='causal biexponential forward observer, no CASCADE network execution',
                metric='cell-averaged Pearson r after linear detrending; held-out scaling is not estimated for metrics',
                total_trials=len(trials),excluded_trials=[{k:r[k] for k in ['dataset','file','trial','events']} for r in trials if not r['usable']],
                profiles=profiles,test_summary=summary,
                binning='Each trial uses max(requested 0.05/0.1 s, 1.01*maximum native interval); no upsampling. Profile bin_s is representative first trial, not all trials.')
    (B/'cascade-observer-profile_codex.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(3,2,figsize=(14,9),gridspec_kw={'width_ratios':[3,1]})
    for i,ds in enumerate(sorted(examples)):
        r,pred=examples[ds];t=r['t']-r['t'][0];end=min(60,t[-1]);ix=t<=end
        def z(x):x=signal.detrend(x);return x/max(np.std(x),1e-12)
        axes[i,0].plot(t[ix],z(r['f'])[ix],label='Measured fluorescence',lw=1)
        axes[i,0].plot(t[ix],z(pred)[ix],label='From known spikes',lw=1,alpha=.8)
        axes[i,0].set(title=ds+' | first held-out cell/trial',xlabel='Time (s)',ylabel='Detrended / SD');axes[i,0].legend(fontsize=8)
        s=next(s for s in summary if s['dataset']==ds)
        axes[i,1].bar(['Instant','Generic','Fitted'],[s[k] for k in ['instantaneous_r','generic_kernel_r','learned_kernel_r']],color=['gray','#c99048','#2688ac'])
        axes[i,1].set(ylim=(0,1),ylabel='Held-out cell mean r',title=str(s['heldout_cells'])+' held-out cells')
    fig.tight_layout();fig.savefig(B/'cascade-heldout-validation_codex.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
