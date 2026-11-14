"""[codex] Q1 metadata contract. Never changes physical parameters or trajectories."""
from pathlib import Path
from copy import deepcopy
from dataclasses import is_dataclass,asdict
from importlib.metadata import version,PackageNotFoundError
import hashlib,json,sys,platform,math
import numpy as np
B=Path(__file__).resolve().parent;R=B.parent.parent
CONTRACT='Q1-control-passport-v1'
def read(name):return json.loads((B/name).read_text(encoding='utf8'))
def cell_passport(profile):
    p=read('cell_passports.json')['profiles']
    if profile not in p:raise ValueError('No reviewed Q1 cell passport for '+repr(profile))
    return deepcopy(p[profile])
def native(x):
    if is_dataclass(x):return native(asdict(x))
    if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [native(v) for v in x]
    if isinstance(x,np.ndarray):return native(x.tolist())
    if isinstance(x,np.generic):return native(x.item())
    if isinstance(x,float) and not math.isfinite(x):return {'nonfinite':str(x)}
    if x is None or isinstance(x,(str,int,float,bool)):return x
    raise TypeError('Unsupported passport value '+type(x).__name__)
def digest(x):return hashlib.sha256(json.dumps(native(x),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def validate_context(context,kwargs):
    if context is None:return
    if not isinstance(context,dict):raise TypeError('control_context must be a dictionary or None')
    if context.get('healthy_human_calibrated') is not False:
        raise ValueError('Control context must explicitly state healthy_human_calibrated=False')
    profile=kwargs.get('profile','Burley')
    if context.get('profile')!=profile:raise ValueError('Control passport/profile mismatch')
    if not cell_passport(profile)['integrated_candidate_supported']:
        raise ValueError('This source passport has no reviewed full-neuron candidate')
    thermo=context.get('thermodynamics')
    if thermo is not None:
        ions=kwargs.get('ion_parameters')
        actual=getattr(ions,'temperature_C',23.)
        if thermo.get('temperature_C')!=actual:
            raise ValueError('Declared candidate temperature differs from active ion parameters')
def leaves(x,prefix=''):
    if isinstance(x,dict):
        for k,v in x.items():yield from leaves(v,prefix+'.'+str(k) if prefix else str(k))
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from leaves(v,f'{prefix}[{i}]')
    else:yield prefix,x
def run_passport(result,context=None):
    config=native(result['config']);profile=config.get('profile','unclassified')
    cells=read('cell_passports.json')['profiles'];conditions=read('conditions.json')
    cell=deepcopy(cells.get(profile))
    key='Selvaraj' if profile.startswith('Selvaraj-') else profile
    reference=conditions['recording_conditions'].get(key)
    ions=config.get('ions',{})
    comparisons=[]
    if reference:
        comparisons.append(dict(quantity='outside_Na_mM',source_value=reference['source_bath_Na_mM'],
          model_value=ions.get('sodium_out0_uM',float('nan'))/1000,
          source_kind='reported recipe; listed sodium salts',
          matches=abs(ions.get('sodium_out0_uM',float('nan'))/1000-reference['source_bath_Na_mM'])<1e-9))
        t=reference['temperature_C']['value']
        comparisons.append(dict(quantity='recording_temperature_C',source_value=t,model_value=ions.get('temperature_C'),
          matches=None if t is None else t==ions.get('temperature_C'),source_kind=reference['temperature_C']['status']))
    flat=list(leaves(config))
    flags={p:v for p,v in flat if isinstance(v,bool)}
    zeros={p:v for p,v in flat if not isinstance(v,bool) and isinstance(v,(int,float)) and v==0}
    volumes={p:v for p,v in flat if any(s in p.lower() for s in ['volume','area','length_um','diameter_um','radius_um','accessible_fraction'])}
    loaded={}
    for m in tuple(sys.modules.values()):
        f=getattr(m,'__file__',None)
        if not f:continue
        p=Path(f).resolve()
        if p.is_file() and p.is_relative_to(R) and p.suffix=='.py':loaded[p.relative_to(R).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    versions={'python':platform.python_version(),'executable':sys.executable}
    for package in ('numpy','scipy','NEURON'):
        try:versions[package]=version(package)
        except PackageNotFoundError:versions[package]=None
    state=np.asarray(result['state']);sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [B/'cell_passports.json',B/'conditions.json',B/'source_manifest.json']}
    warnings=['A source-cell label does not validate the biochemical baseline',
      'Boolean flags and zero leaves are a configuration inventory, not proof that every other pathway carries a nonzero flux',
      'Historical config prose such as fixed-voltage/no-AP or LSODA can describe an ancestor; use actual solver_settings, integrated_model and state outputs',
      'Whole-cell recording recipes do not determine intact-cell energy pools; pipette dialysis is not explicitly simulated']
    if any(x['matches'] is False for x in comparisons):warnings.append('Source recording conditions and active whole-model conditions differ; this run is not a matched experimental validation')
    if context is None:warnings.append('No explicit baseline provenance supplied; do not infer joint candidate or healthy control from numeric similarity')
    return dict(contract=CONTRACT,profile=profile,source_cell=cell,source_recording_conditions=reference,
      baseline_context=native(context) if context else None,baseline_class='explicit_conditional_candidate' if context else 'unclassified_research_configuration',
      healthy_human_calibrated=False,source_conditions_fully_matched=False,
      comparisons=comparisons,model_condition_policy=conditions['model_modules'],pool_meanings=conditions['pool_meanings'],
      active_configuration=config,configuration_sha256=digest(config),boolean_flags=flags,numeric_zero_leaves=zeros,geometry_and_volume_fields=volumes,
      external_CK_activity_multiplier=native(result.get('creatine_activity_multiplier',[1.,1.,1.])),
      diagnostic_interventions=native(result.get('diagnostic_interventions',{})),
      time_start_s=float(result['time_s'][0]),time_end_s=float(result['time_s'][-1]),
      state_count=state.shape[1],state_schema_sha256=digest(result['state_schema']),
      initial_state_sha256=hashlib.sha256(np.ascontiguousarray(state[0]).tobytes()).hexdigest(),
      final_state_sha256=hashlib.sha256(np.ascontiguousarray(state[-1]).tobytes()).hexdigest(),
      solver_settings=native(result.get('solver_settings')),versions=versions,
      loaded_project_python_sha256=loaded,passport_assets_sha256=sources,
      source_hash_scope='Loaded project Python and passport assets, not a trace of every runtime data read; use separate runtime closure audit for that',warnings=warnings)
