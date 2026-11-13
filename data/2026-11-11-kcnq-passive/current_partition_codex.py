"""[codex] Read-only ionic current decomposition. No fitting or current injection."""
import numpy as np

def observe(state,baseline,config,differential_KCNQ_multiplier=1.):
    if 'ion_background_reference' not in config:
        raise ValueError('Current decomposition requires an explicitly configured passive ion-background branch')
    if not np.isfinite(differential_KCNQ_multiplier) or differential_KCNQ_multiplier<0:
        raise ValueError('Differential KCNQ multiplier must be finite and nonnegative')
    z=np.atleast_2d(state);base=np.asarray(baseline)
    p=np.asarray(config['HH_parameters']);c=np.asarray(config['HH_context'])
    ref=np.asarray(config['ion_background_reference']);ip=config['ions']
    thermal=8.314462618*(273.15+ip['temperature_C'])/96485.33212*1000
    coupling=config['kcnq_coupling'];kp=config['kcnq_transport'];rows=[]
    frac=config['ion_background']['passive_fraction']
    for j in range(2):
        k=199+20*j;vi=185+7*j;v=z[:,vi];cap=ref[j,5]
        ena=thermal*np.log(z[:,k+2]/z[:,k]);ek=thermal*np.log(z[:,k+3]/z[:,k+1])
        surface=346+2*j
        raw=(z[:,surface]+kp['mutant_residual_function']*z[:,surface+6])/(base[surface]+kp['mutant_residual_function']*base[surface+6])
        ratio=np.maximum(0,1+kp['electrical_transfer_gain']*(raw-1))
        gm=coupling['g_reference_nS']*cap/c[0]
        currents=dict(HH_Na_pA=cap*p[0]*z[:,vi+1]**3*z[:,vi+2]*(v-ena),
            HH_K_fast_pA=cap*p[3]*z[:,vi+3]**4*(v-ek),
            HH_slow_K_unidentified_pA=cap*p[12]*z[:,vi+4]*(v-ek),
            differential_KCNQ_pA=differential_KCNQ_multiplier*(ratio-1)*gm*z[:,362+j]*(v-ek),
            passive_Na_pA=frac*ref[j,3]*(v-ena),passive_K_pA=frac*ref[j,4]*(v-ek))
        if 'passive_chloride' in config:
            off=config['passive_chloride_offset']+6*j
            ecl=thermal*np.log(z[:,off]/z[:,off+1]);gcl=config['passive_chloride_reference'][j][1]
            currents['passive_Cl_pA']=gcl*(v-ecl)
        rows.append(dict(currents=currents,KCNQ_surface_ratio=ratio))
    return dict(compartments=rows,scope='Selected membrane currents only; not total dV/dt. HH slow K is not identified absolute KCNQ. Differential-KCNQ zero does not simulate full XE991 blockade. Pump/Ca/axial/holding currents remain separate owners.')
