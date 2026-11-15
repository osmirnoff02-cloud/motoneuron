"""[codex] 2026-11-14. Read-only budgets and explicit, scoped baseline checks.

Counter 2080 is a subset of counter 2000: never charge its removal twice.
GAP43 completed chains are in somatic selected-ribosome equivalents, while
GAP43 and its degradation counter are in local reference-volume micromolar.
"""
import numpy as np

SCHEMA_SHA256='fcca4e0269700e6bf7c6f761cb7da16d8d52da772cb834b4f15f4d0d05ee5948'


def observe(result, start_s=None, end_s=None):
    """Audit a full physical-state output; display aliases are not accepted."""
    schema=result['state_schema']
    if schema.get('sha256')!=SCHEMA_SHA256 or schema.get('count')!=4431:
        raise ValueError('Unsupported physical state schema; update the observer mapping')
    t=np.asarray(result['time_s'],float); z=np.asarray(result['state'],float)
    if t.ndim!=1 or len(t)<2 or z.shape!=(len(t),4431) or not np.all(np.diff(t)>0):
        raise ValueError('Expected ordered physical trajectory with 4431 states')
    if not np.all(np.isfinite(z)):
        raise ValueError('Nonfinite trajectory')
    a=t[0] if start_s is None else float(start_s);b=t[-1] if end_s is None else float(end_s)
    if a<t[0] or b>t[-1] or a>=b:raise ValueError('Window outside trajectory')
    i=int(np.searchsorted(t,a));j=int(np.searchsorted(t,b,side='right')-1)
    if j<=i:raise ValueError('Window contains fewer than two samples')
    cfg=result['config'];genes=cfg['RP_production_genes']
    if len(genes)!=80 or len(set(genes))!=80 or genes[38]!='RPL5':
        raise ValueError('Unexpected ribosomal protein mapping')
    c=float(cfg['biogenesis']['selected_mature_pool_uM'])
    volumes=np.asarray(cfg['mobile_water_volumes']['reference_volume_um3'],float)
    if volumes.shape!=(2,) or not np.all(volumes>0) or not np.isfinite(c) or c<=0:
        raise ValueError('Invalid molecular reference volumes/pool')
    scale=c*volumes[0]/volumes[1];d=z[j]-z[i];dt=t[j]-t[i]
    rows=[]
    for g,gene in enumerate(genes):
        held=d[1600+g]+d[1680+g]+d[1760+g]
        made=d[1840+g];built=d[1920+g];removed=d[2000+g]
        rows.append(dict(gene=gene,free_start_uM=float(z[i,1600+g]),free_end_uM=float(z[j,1600+g]),
            nuclear_free_start_uM=float(z[i,4076+4*g]),nuclear_free_end_uM=float(z[j,4076+4*g]),
            made_uM=float(made),assembled_uM=float(built),removed_uM=float(removed),
            extra_free_removed_subset_uM=float(d[2080+g]),held_change_uM=float(held),
            residual_uM=float(held+built+removed-made),
            assembly_fraction=float(built/made) if made>0 else None))
    completed=d[586]*scale;degraded=d[596]
    # Numerical tolerances are QA choices, not biological normal ranges.
    rp_res=max(abs(x['residual_uM']) for x in rows)
    traffic=z[i:j+1,4073:4393].reshape(j-i+1,80,4)
    traffic_error=float(np.max(abs(traffic.sum(axis=2)-z[i:j+1,1600:1680])))
    gap_res=float(d[583]-completed+degraded)
    throughput=max(max(abs(x['made_uM']),abs(x['removed_uM']),abs(x['assembled_uM'])) for x in rows)
    rp_tol=1e-8+1e-6*throughput;gap_tol=1e-8+1e-6*max(abs(completed),abs(degraded))
    return dict(start_s=float(t[i]),end_s=float(t[j]),duration_s=float(dt),rp=rows,
        GAP43=dict(start_uM=float(z[i,583]),end_uM=float(z[j,583]),completed_uM=float(completed),
            degraded_uM=float(degraded),residual_uM=gap_res,volume_scale_uM_per_Ceq=float(scale),
            synthesis_uM_per_h=float(completed/dt*3600),removal_uM_per_h=float(degraded/dt*3600)),
        checks=dict(rp_budget_ok=bool(rp_res<=rp_tol),GAP43_budget_ok=bool(abs(gap_res)<=gap_tol),
            RP_transport_accounting_ok=traffic_error<=rp_tol,RP_transport_max_error_uM=traffic_error,
            max_rp_residual_uM=rp_res,rp_tolerance_uM=rp_tol,GAP43_tolerance_uM=gap_tol),
        scope='Production/assembly/removal accounting; not steady state or human calibration')


def baseline_check(result, window_h=72., drift_limit_percent=1., pool_floor=1e-8):
    """Certify ONLY listed molecular inventories, never all 4431 variables.

    Uses peak-to-peak, not just endpoint drift. Returns a negative result for
    incomplete duration. All defaults are explicit numerical acceptance choices.
    It does not edit a state, rate, reference concentration or cumulative ledger.
    """
    if not np.all(np.isfinite([window_h,drift_limit_percent,pool_floor])) or window_h<=0 or drift_limit_percent<=0 or pool_floor<=0:
        raise ValueError('Acceptance settings must be positive')
    t=np.asarray(result['time_s'],float);end=float(t[-1]);start=end-window_h*3600
    if start<t[0]-1e-7:
        return dict(qualified=False,reason='insufficient independent observation window',required_h=window_h)
    budget=observe(result,max(start,float(t[0])),end)
    genes=result['config']['RP_production_genes']
    ids=list(range(1600,1680))+list(range(4076,4393,4))+[330,331,385,386,387,390,391,564,565,583,607]
    labels=list(genes)+[g+'_nuclear_free' for g in genes]+['free40S','free60S','pre18S','pre28S','5S',
        'pre40S','pre60S','local40S','local60S','GAP43','local_GAP43_RNA']
    z=np.asarray(result['state']);sel=z[t>=start][:,ids]
    span=100*np.ptp(sel,axis=0)/np.maximum(abs(sel[0]),pool_floor)
    drift=100*(sel[-1]-sel[0])/np.maximum(abs(sel[0]),pool_floor)
    rows=[dict(pool=n,index=k,peak_to_peak_percent=float(s),end_change_percent=float(d))
          for n,k,s,d in zip(labels,ids,span,drift)]
    ok=max(span)<=drift_limit_percent and all(budget['checks'][k] for k in ('rp_budget_ok','GAP43_budget_ok','RP_transport_accounting_ok'))
    return dict(qualified=bool(ok),scope='80 total free RP + 80 nuclear free RP + 11 named assembly/local pools ONLY',
        whole_neuron_qualified=False,human_calibrated=False,window_h=window_h,
        drift_limit_percent=drift_limit_percent,pool_floor=pool_floor,
        acceptance_status='Numerical protocol choices, not measured physiological intervals',
        max_peak_to_peak_percent=float(max(span)),pools=rows,budgets=budget['checks'])


def gap43_turnover_scenarios():
    return dict(parameter='local_translation.protein_half_h',unit='h',
        current_default=24.,legacy_scenario_grid=[6.,24.,72.],
        human_whole_culture_anchor=100.19839036799999,
        source='Frankenfield et al. preprint 10.64898/2026.03.14.711828; local frozen site dataset, 2026-11-04',
        cell_type='Human iPSC-derived spinal motor neuron cultures; dSILAC total protein',
        limitations=['Six passing peptides are not six biological replicates',
          'No axon-specific turnover or uncertainty interval',
          'Existing ATP gate changes effective model half-life',
          'The whole-culture anchor is opt-in transfer, not a fitted local rate'],
        data_needed='Compartment-resolved GAP43 pulse-chase plus nascent synthesis, transport and ATP',
        candidate_owner='Authors of Frankenfield turnover platform; local transport: Fallini/Rossoll/Twiss/Schuman',
        correspondence_status='No new request sent in this task')


def gap43_fixed_flux_projection(budget, initial_uM, horizon_h, half_h, energy_gate):
    """Conditional exact one-pool solution; NOT a coupled-neuron prediction."""
    if not np.all(np.isfinite([initial_uM,horizon_h,half_h,energy_gate])) or initial_uM<0 or horizon_h<0 or half_h<=0 or not 0<energy_gate<=1:
        raise ValueError('Invalid projection parameters')
    production=budget['GAP43']['synthesis_uM_per_h']
    if production<0:raise ValueError('Invalid production counter')
    k=np.log(2)/half_h*energy_gate;eq=production/k
    return dict(end_uM=float(eq+(initial_uM-eq)*np.exp(-k*horizon_h)),steady_uM=float(eq),
        nominal_half_h=half_h,effective_half_h=float(half_h/energy_gate),energy_gate=energy_gate,
        scope='Frozen synthesis and ATP gate; no feedback, transport or biological calibration')


def gap43_apparent_half_to_parameter(apparent_half_h, ATP_uM, ATP_half_uM):
    """Convert an apparent turnover anchor under explicitly chosen ATP conditions.

    The ODE uses k=ln(2)/parameter * ATP/(ATP_half+ATP). Simply placing an
    apparent half-life in the parameter applies the ATP slowing a second time.
    This conversion does not make whole-culture dSILAC an axonal measurement.
    """
    if not np.all(np.isfinite([apparent_half_h,ATP_uM,ATP_half_uM])) or apparent_half_h<=0 or ATP_uM<=0 or ATP_half_uM<0:
        raise ValueError('Invalid turnover or ATP reference')
    gate=ATP_uM/(ATP_half_uM+ATP_uM)
    return dict(protein_half_h=float(apparent_half_h*gate),reference_energy_gate=float(gate),
        reference_apparent_half_h=float(apparent_half_h),
        status='Optional whole-pool-to-local transfer; uncalibrated compartment assignment')
