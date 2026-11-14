"""[codex] Measured assay references; no override of neuron RHS or defaults.

These are conditional comparison functions, not a calibrated motor neuron.
The caller supplies temperature, intracellular concentrations and Hill slope.
"""
from pathlib import Path
import json
import math

_B = Path(__file__).resolve().parent


def human_electrical_source_constraints():
    """Return fresh provenance and measured values (SD, SEM, CI kept distinct)."""
    return json.loads((_B / 'numeric_constraints.json').read_text(encoding='utf8'))


def measured_medium_reference(name):
    """Measured medium/CSF, not a product recipe or a simulation default."""
    contexts = human_electrical_source_constraints()['medium_contexts']
    if name not in contexts:
        raise ValueError(f'Unknown medium {name!r}; choose {tuple(contexts)}')
    return contexts[name]


def _finite(x, name, positive=False):
    value = float(x)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return value


def medium_nernst_reference(name, *, intracellular_mM, temperature_C):
    """Ideal concentration-based E_Na/E_K/E_Cl in mV at explicit conditions.

    Ca/Mg are excluded: colorimetric amounts do not establish free activities.
    Fixed bath and equal activity coefficients are assumptions. SD endpoints
    are transformed separately; they are not a confidence or prediction band.
    No ions are reset, no currents run and no ATP is spent by this diagnostic.
    """
    t = _finite(temperature_C, 'temperature_C')
    if t <= -273.15:
        raise ValueError('Temperature must exceed absolute zero')
    if set(intracellular_mM) != {'Na', 'K', 'Cl'}:
        raise ValueError('Supply exactly Na, K and Cl; free Ca/Mg are not established')
    ref = measured_medium_reference(name)
    rt_f = 8.31446261815324 * (273.15+t) / 96485.33212 * 1000
    values = {}
    for ion, z in [('Na',1),('K',1),('Cl',-1)]:
        ci = _finite(intracellular_mM[ion], ion+'_inside', True)
        co = ref['mean_mM'][ion]
        e = rt_f / z * math.log(co / ci)
        bounds = None
        if ref['SD_mM'] is not None:
            sd = ref['SD_mM'][ion]
            bounds = sorted([rt_f/z*math.log((co-sd)/ci), rt_f/z*math.log((co+sd)/ci)])
        values[ion] = dict(E_mV=e, SD_endpoint_envelope_mV=bounds)
    return dict(medium=name, temperature_C=t, intracellular_mM=dict(intracellular_mM),
                values=values, source=ref,
                assumptions='Ideal concentrations, equal activity coefficients, fixed bath. Not membrane voltage or motor-neuron calibration.')


def halliwell_ligand_reference(ligand):
    ref = human_electrical_source_constraints()['halliwell']
    if ligand not in ref['ligands']:
        raise ValueError('Choose AMPA, NMDA or kainate. Glutamate text/figure is unresolved.')
    return dict(ligand=ligand, **ref['ligands'][ligand], conditions={
        k:ref[k] for k in ('doi','cell','patch_temperature_C','holding_mV','bath_recipe_mM','nmda_required','missing','use')})


def halliwell_normalized_response(concentrations_uM, *, ligand, hill_coefficient,
                                  temperature_C, holding_mV, magnesium_mM, glycine_uM):
    """Conditional normalized Hill response using a published EC50.

    Slope is explicitly supplied, not measured/fitted here. The envelope only
    propagates EC50 CI endpoints at that slope, NOT total response uncertainty.
    No Imax, GluA2 editing, calcium fraction or synaptic kinetics is inferred.
    AMPA means the applied chemical agonist, NOT a glutamate concentration:
    this EC50 must not silently replace a synaptic glutamate response constant.
    """
    ref = halliwell_ligand_reference(ligand)
    slope = _finite(hill_coefficient, 'hill_coefficient', True)
    t = _finite(temperature_C, 'temperature_C')
    v = _finite(holding_mV, 'holding_mV')
    mg = _finite(magnesium_mM, 'magnesium_mM')
    gly = _finite(glycine_uM, 'glycine_uM')
    if not 22 <= t <= 25 or not math.isclose(v, -60, abs_tol=1e-9):
        raise ValueError('Source patch assay requires 22-25C and -60mV; no Q10/voltage extrapolation')
    if mg < 0 or gly < 0:
        raise ValueError('Concentrations must be nonnegative')
    if ligand == 'NMDA':
        if not math.isclose(mg, 0, abs_tol=1e-12) or not math.isclose(gly,1,abs_tol=1e-12):
            raise ValueError('NMDA assay requires 0mM Mg and 1uM glycine')
    elif not math.isclose(mg,2,abs_tol=1e-12) or not math.isclose(gly,0,abs_tol=1e-12):
        raise ValueError('AMPA/kainate reference used 2mM Mg and no added glycine; other conditions unvalidated')

    def response(c, ec50):
        if c == 0:
            return 0.0
        x = slope * (math.log(c)-math.log(ec50))
        return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))
    rows=[]
    for c in concentrations_uM:
        c=_finite(c,'concentration_uM')
        if c<0:
            raise ValueError('Concentrations must be nonnegative')
        low,high=ref['CI95_uM']
        rows.append(dict(concentration_uM=c, normalized_response=response(c,ref['EC50_uM']),
            conditional_EC50_envelope=[response(c,high),response(c,low)],
            within_tested_concentrations=ref['tested_uM'][0]<=c<=ref['tested_uM'][1]))
    return dict(ligand=ligand, hill_coefficient=slope,hill_status='Caller assumption, not a source measurement',
                source=ref,rows=rows,physiological_default_changed=False)


def lyckenvik_network_reference(*, panel, unit):
    """Published technical-replicate summaries, not independent cell F-I data.

    Unit is mandatory because the published supplement repeats label 4C.
    """
    src=human_electrical_source_constraints()
    rows=[r for r in src['network_summary'] if r['panel_as_printed']==panel and r['unit']==unit]
    if not rows:
        raise ValueError('No matching panel and unit in published Table2')
    return dict(rows=rows,protocol=src['network_protocol'],version='published2026')


def kleckner_pulse_reference():
    """Return count/rate-defined train with an explicit source timing conflict."""
    p=human_electrical_source_constraints()['kleckner']
    onsets=[i/p['train_Hz'] for i in range(p['pulses'])]
    return dict(onset_s=onsets,last_pulse_end_s=onsets[-1]+p['pulse_ms']/1000,
                source=p,warning=p['timing_conflict'])
