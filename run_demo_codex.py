"""Codex, 2026-11-13: portable, read-only source wrapper for the full model.

This runs the inherited equations without changing their scientific parameters.
Only this wrapper is the supported public execution entry point in this snapshot.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=('smoke', 'pulse', 'rest72', 'load36'), default='smoke')
    parser.add_argument('--profile', choices=('Burley','Selvaraj-Con1','Selvaraj-Con2'), default='Burley')
    parser.add_argument('--candidate', default='C08', help='Explicit candidate from the declared grid; not a healthy default')
    parser.add_argument('--current-pa', type=float, default=25.)
    parser.add_argument('--branch', choices=('control','continuous','repeated'), default='repeated')
    parser.add_argument('--opening', type=float, choices=(.0001,.001), default=.001)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='A NEW directory, never an existing result directory')
    parser.add_argument('--full-trace', action='store_true', help='Also retain all states; can create large files')
    parser.add_argument('--check-restart', action='store_true', help='Smoke only: continue 20 to 40 ms and compare with an uninterrupted run')
    parser.add_argument('--forbid-root', type=Path, action='append', default=[], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_restart and args.protocol!='smoke':parser.error('--check-restart requires smoke')
    if args.protocol=='load36' and args.checkpoint is None:parser.error('load36 requires a matching rest72 checkpoint')
    if args.protocol=='rest72' and args.checkpoint is not None:parser.error('rest72 starts a new preparation')
    if args.protocol in ('rest72','load36') and args.profile!='Selvaraj-Con2':parser.error('published long protocol uses Selvaraj-Con2')
    out=args.output.expanduser().resolve()
    if out==ROOT or (out.is_relative_to(ROOT) and not out.is_relative_to(ROOT/'outputs')):
        parser.error('Inside the repository, output must be a new directory below outputs/')
    out.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('NUMBA_CACHE_DIR',str(out.parent/'numba-cache'))
    os.environ.setdefault('MPLCONFIGDIR',str(out.parent/'matplotlib-cache'))
    os.environ.setdefault('OMP_NUM_THREADS','1')
    sys.dont_write_bytecode=True
    forbidden=[x.resolve() for x in args.forbid_root]
    busy=False
    reads=set()

    def audit(event, data):
        nonlocal busy
        if event in ('socket.connect','socket.getaddrinfo'):
            raise RuntimeError('This example does not require network access')
        if busy or event!='open' or not data or not isinstance(data[0],(str,bytes,os.PathLike)):return
        busy=True
        try:
            path=Path(os.fsdecode(data[0])).resolve()
            if any(path.is_relative_to(x) for x in forbidden):raise RuntimeError('Read outside the exported project: '+str(path))
            mode=data[1] if len(data)>1 else None
            flags=data[2] if len(data)>2 else 0
            writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'outputs'):
                if writing:raise RuntimeError('Model attempted to modify an exported source: '+str(path))
                if path.is_file() and path.suffix not in ('.pyc','.nbc','.nbi'):reads.add(path.relative_to(ROOT).as_posix())
        finally:busy=False
    sys.addaudithook(audit)
    import numpy as np
    from threadpoolctl import threadpool_limits
    limits=threadpool_limits(limits=1)
    start=time.monotonic()
    folder=ROOT/'data/2026-11-13-joint-kcnq-pump-v2'
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('publication_full_runner',folder/'full_codex.py')
    full=importlib.util.module_from_spec(spec);sys.modules[spec.name]=full;spec.loader.exec_module(full)
    cp=None if args.checkpoint is None else json.loads(args.checkpoint.read_text(encoding='utf-8'))
    t0=0. if cp is None else cp['time_s']
    if args.protocol=='load36' and abs(t0-cp.get('origin_time_s',0.)-72*3600)>1e-6:
        parser.error('load36 requires a completed 72-hour preparation checkpoint, not an arbitrary saved state')
    kw=dict(rtol=1e-8,atol=1e-10,solver_method='BDF',solver_sparse_jacobian=True,
            retain_dense_solver_output=False,center_solver_coordinates=True)
    if args.protocol=='smoke':
        sg=full.r.schedule([(t0,t0+.02,0,1,0)])
        kw.update(output_step_s=.002,max_step_s=.002)
    elif args.protocol=='pulse':
        on=t0+.2;off=on+(.5 if args.profile=='Burley' else 1.)
        sg=full.r.schedule([(t0,on,0,1,0),(on,off,0,1,0),(off,off+1,0,1,0)])
        kw.update(output_step_s=.0002,max_step_s=.0025,electrical_schedule=np.c_[sg[:,:2],[0,args.current_pa,0],[0,0,0]])
    elif args.protocol=='rest72':
        edges=np.unique(np.r_[0,.01,.1,1,10,60,600,3600,np.arange(21600,72*3600,21600),72*3600])
        sg=full.r.schedule([(a,b,0,1,0) for a,b in zip(edges[:-1],edges[1:])])
        kw.update(output_step_s=300,max_step_s=300,rtol=1e-7,atol=full.c.old.legacy.long_atol())
    else:
        sg=full.r.schedule([(t0+h*3600,t0+(h+1)*3600,args.opening if ((args.branch=='continuous' and h<6) or (args.branch=='repeated' and h in (0,4,8,12,16,20))) else 0.,1,0) for h in range(36)])
        kw.update(output_step_s=300,max_step_s=300,rtol=1e-7,atol=full.c.old.legacy.long_atol())
    options,provenance=full.options(sg,args.profile,full.params(args.candidate))
    def observer(checkpoint,nfev):
        if args.protocol in ('rest72','load36'):
            # Recovery aid for this NEW output directory; packaged inputs stay read-only.
            path=out/('progress_'+str(int(checkpoint['time_s']))+'_checkpoint_codex.json')
            if not path.exists():
                path.write_text(json.dumps(checkpoint,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')
        print('time_s=',checkpoint['time_s'],'nfev=',nfev,flush=True)
    print('Integrating full 4425-state',args.candidate,args.profile,args.protocol,flush=True)
    result=full.r.api.run_integrated_neuron(sg,**options,checkpoint=cp,solver_observer=observer,**kw)
    checks={k:float(v) for k,v in result['checks'].items() if k!='minimum_state'}
    summary=dict(protocol=args.protocol,candidate=args.candidate,profile=args.profile,current_pA=args.current_pa if args.protocol=='pulse' else None,start_time_s=t0,states=result['state'].shape[1],
        samples=len(result['time_s']),all_finite=bool(np.isfinite(result['state']).all()),
        max_balance=max(abs(v) for v in checks.values()),checks=checks,nfev=result['nfev'],
        human_calibrated=False,elapsed_s=time.monotonic()-start,
        final_voltage_mV=float(result['state'][-1,185]),no_reset=bool(cp is None or np.array_equal(result['state'][0],cp['state'])))
    if args.protocol=='pulse':summary['features']=full.r.features_mod.features(result,on,off,args.profile)
    if args.check_restart:
        if cp is not None:parser.error('--check-restart requires a fresh smoke run')
        follow=full.r.schedule([(.02,.04,0,1,0)])
        continued=full.r.api.run_integrated_neuron(follow,**options,checkpoint=result['checkpoint'],**kw)
        whole=full.r.api.run_integrated_neuron(full.r.schedule([(0,.04,0,1,0)]),**options,**kw)
        delta=float(np.max(abs(continued['state'][-1]-whole['state'][-1])))
        identical_start=bool(np.array_equal(continued['state'][0],result['state'][-1]))
        matched=bool(np.allclose(continued['state'][-1],whole['state'][-1],rtol=2e-6,atol=1e-7))
        summary['restart']=dict(exact_initial_state=identical_start,max_end_difference=delta,within_tolerance=matched)
        if not identical_start or not matched:raise AssertionError('Checkpoint restart disagrees with uninterrupted integration')
    if summary['states']!=4425 or not summary['all_finite'] or summary['max_balance']>=3e-5 or not summary['no_reset']:
        raise AssertionError('Numerical validation failed: '+json.dumps(summary))
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        return str(x)
    # Configuration fingerprints are checked by the inherited model on restart.
    for name,value in [('summary',summary),('config',result['config']),('checkpoint',result['checkpoint']),('provenance',provenance),('columns',result['integrated_columns']),('runtime_reads',sorted(reads))]:
        (out/(name+'_codex.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=encode)+'\n',encoding='utf-8')
    selected={'soma_V_mV':185,'process_V_mV':192,'soma_ATP_uM':120,'process_ATP_uM':136,'process_PCr_uM':316,'soma_K_uM':200,'process_K_uM':220}
    with (out/'timeseries_codex.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['time_s']+list(selected))
        for t,z in zip(result['time_s'],result['state']):w.writerow([t]+[z[i] for i in selected.values()])
    if args.full_trace or args.protocol=='smoke':
        np.savez_compressed(out/'trace_codex.npz',time_s=result['time_s'],state=result['state'],baseline=result['baseline_state'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True,layout='constrained')
    axes[0].plot(result['time_s'],result['state'][:,185]);axes[0].set_ylabel('Soma voltage (mV)')
    axes[1].plot(result['time_s'],result['state'][:,220]);axes[1].set_ylabel('Process K (uM)');axes[1].set_xlabel('Time (s)')
    fig.suptitle(args.candidate+' conditional model: '+args.protocol+' — not human calibrated')
    fig.savefig(out/'figure_codex.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='checks'},default=encode),flush=True)
    print('Saved',out,flush=True)

if __name__=='__main__':main()
