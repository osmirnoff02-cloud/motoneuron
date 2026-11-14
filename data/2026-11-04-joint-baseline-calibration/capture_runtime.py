"""[codex] Actual dependency trace for a short joint-candidate run.

This verifies source availability and input reconstruction, not biological fit.
"""
from pathlib import Path
import sys,os,json,hashlib,time,importlib.util
import numpy as np
from threadpoolctl import threadpool_limits
_limit=threadpool_limits(limits=1)
B=Path(__file__).resolve().parent;R=B.parent.parent
reads=set();busy=False;start=time.monotonic()
def audit(event,args):
    global busy
    if busy or event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    mode=args[1] if len(args)>1 else None
    if mode and any(x in mode for x in 'wax+'):return
    busy=True
    try:
        p=Path(args[0]).resolve()
        if p.is_relative_to(R) and p.is_file() and p.suffix.lower() not in ('.pyc','.nbc','.nbi'):
            reads.add(p)
    finally:busy=False
sys.addaudithook(audit)
p=B/'run.py';spec=importlib.util.spec_from_file_location('joint_dependency_runner',p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
opts,provenance=m.options()
x=m.api.run_integrated_neuron(m.schedule([(0,.02,0,1,0)]),**opts,
    output_step_s=.01,max_step_s=.01,rtol=1e-8,atol=1e-10,
    solver_method='BDF',solver_sparse_jacobian=True)
checks={k:float(v) for k,v in x['checks'].items() if k!='minimum_state'}
assert x['state'].shape[1]==4413 and max(checks.values())<3e-5 and np.isfinite(x['state']).all()
for mod in tuple(sys.modules.values()):
    f=getattr(mod,'__file__',None)
    if f:
        p=Path(f).resolve()
        if p.is_relative_to(R) and p.suffix=='.py':reads.add(p)
files=[]
for p in sorted(reads):
    files.append(dict(path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
out=dict(author='codex',date='2026-11-04',states=4413,endpoint_s=.02,files=files,
    checks=checks,elapsed_s=time.monotonic()-start,
    scope='Actual import/read closure of the current joint candidate BDF 20ms; optional Ye-transfer source panel not invoked; not a retroactive trace of the long runs')
(B/'runtime_closure.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
np.savez_compressed(B/'runtime_smoke.npz',time_s=x['time_s'],state=x['state'])
print('Captured',len(files),'files;',max(checks.values()),'worst constraint;',out['elapsed_s'],'s',flush=True)
