"""[codex] Q2 explicit recipe scenarios and read-only coupled-pool audit.

No additional flux, energy debit, fitted conductance, or hidden equilibration.
An as-printed bath is not proof of intact-cell conditions or human calibration.
"""
from dataclasses import replace
import numpy as np

SCENARIOS={
 'legacy_reference':dict(Na_out_mM=145.,kind='inherited generic bath'),
 'Burley_as_printed':dict(Na_out_mM=167.,kind='reported bath recipe; unresolved osmolarity inconsistency',doi='10.1038/s41598-022-09751-3',
   osmolality_reported_mOsm=300.,ideal_solute_estimate_mOsm=367.8,
   caveat='Ideal particle estimate is not measured osmolality. Do not guess a corrected NaCl concentration; author clarification needed.'),
 'Selvaraj_listed_salts':dict(Na_out_mM=154.25,kind='sum 127 NaCl + 26 NaHCO3 + 1.25 NaH2PO4; pH titrant not quantified',doi='10.1038/s41467-017-02729-0')
}

def inputs(api,profile='Burley',*,ionic_scenario='legacy_reference',**conditions):
    if ionic_scenario not in SCENARIOS:raise ValueError('Unknown ionic scenario')
    if ionic_scenario.startswith('Burley') and profile!='Burley':raise ValueError('Burley recipe requires Burley profile')
    if ionic_scenario.startswith('Selvaraj') and not profile.startswith('Selvaraj-'):raise ValueError('Selvaraj recipe requires Selvaraj profile')
    o,p=api.joint_baseline_inputs(profile,**conditions)
    source=dict(SCENARIOS[ionic_scenario])
    o['ion_parameters']=replace(o['ion_parameters'],sodium_out0_uM=source['Na_out_mM']*1000,
       provenance=o['ion_parameters'].provenance+'; Q2 '+ionic_scenario+'; Na_i and K_o reconstructed from existing ENa/EK, not independent measurements.')
    p['Q2_ionic_scenario']=dict(name=ionic_scenario,**source,
       reversal_policy='Retain profile ENa/EK and Ki; infer Nai/Ko via Nernst at implementation23C. Report those inferred values explicitly.',
       physiological_bath_validated=False,whole_cell_dialysis_simulated=False,
       CK_solution_vs_cell_ionic_strength_matched=False)
    o['control_context']=p
    return o,p

def audit(result):
    """Accept a full result or saved trace+config. No output feeds the RHS."""
    z=np.asarray(result['state'],float);base=np.asarray(result['baseline_state'],float)
    t=np.asarray(result['time_s'],float);c=result['config'];v=c['ions']
    if z.ndim!=2 or z.shape[1]<513 or base.shape!=(z.shape[1],) or len(t)!=len(z):raise ValueError('Full physical trace required')
    if not np.isfinite(z).all():raise ValueError('Nonfinite physical trace')
    en=c['energy'];ck=c['creatine'];g=c['geometry']
    vols=[g['soma_volume_um3'],np.pi*g['process_radius_um']**2*g['process_length_um']*g['process_accessible_fraction']]
    thermal=.08617333262*(273.15+v['temperature_C']);Kak=en['ATP0_uM']*en['AMP0_uM']/en['ADP0_uM']**2
    def extent(x):
        x=np.asarray(x,float)
        good=x[np.isfinite(x)]
        val=lambda a:float(a) if np.isfinite(a) else None
        return dict(initial=val(x[0]),final=val(x[-1]),minimum=float(good.min()) if len(good) else None,maximum=float(good.max()) if len(good) else None,undefined_samples=int((~np.isfinite(x)).sum()))
    def ratio(a,b):
        return np.divide(a,b,out=np.full_like(np.asarray(a,float),np.nan),where=np.asarray(b)>0)
    pools=[]
    for j,a in enumerate((120,136,282)):
        atp,adp,amp=z[:,a:a+3].T;pc=z[:,315+j];cr=ck['total_creatine_uM'][j]-pc
        pools.append(dict(compartment=('soma','process','terminal')[j],
          ATP_uM=extent(atp),ADP_uM=extent(adp),AMP_uM=extent(amp),PCr_uM=extent(pc),Cr_uM=extent(cr),
          CK_quotient_over_K=extent(ratio(atp*cr,adp*pc)*ck['reverse_equilibrium_factor']),
          AK_quotient_over_K=extent(ratio(atp*amp,adp**2)/Kak),
          chemical_potential_scope='quotient of represented species-sums; not free unbound or MgATP measurement'))
    ions=[]
    for j,b in enumerate((199,219)):
        d=z[:,b:b+20]-base[b:b+20];ni,ki,no,ko=z[:,b:b+4].T
        # Independent identities built from physical pool changes and existing
        # cumulative counters; axial counters are amounts (uM*um3), not uM.
        # Cumulative reaction counters retain Vref normalization; mobile ion
        # concentration changes use Vwater in the explicit-water branch.
        water=c.get('mobile_water_volumes',{}).get('mobile_water_volume_um3',vols)[j]
        factor=vols[j]/water
        rn=d[:,0]+(d[:,5]+3*d[:,4]-d[:,7]-d[:,8]-d[:,12]/vols[j])*factor
        rk=d[:,1]+(d[:,6]-2*d[:,4]-d[:,9]-d[:,13]/vols[j])*factor
        ions.append(dict(compartment=('soma','process')[j],Na_i_uM=extent(ni),K_i_uM=extent(ki),Na_o_uM=extent(no),K_o_uM=extent(ko),
          ENa_mV=extent(thermal*np.log(no/ni)),EK_mV=extent(thermal*np.log(ko/ki)),
          Na_ledger_max_abs_uM=float(abs(rn).max()),K_ledger_max_abs_uM=float(abs(rk).max()),
          NaK_ATP_used_uM=float(d[-1,4]),NaK_ATP_counter_normalization='reference volume',
          mobile_water_volume_um3=float(water),
          warning='Mass closure does not imply an appropriate baseline; residual leak, holding current, chloride and water remain unresolved'))
    G=z[:,499];D=z[:,500]
    return dict(schema='Q2_ion_energy_audit_v1',time_s=[float(t[0]),float(t[-1])],states=z.shape[1],
       pools=pools,ions=ions,AK_Kobs_used=float(Kak),CK_Kobs_used=1/ck['reverse_equilibrium_factor'],
       GTP_uM=extent(G),GDP_uM=extent(D),
       NDPK_quotient_over_K=extent(ratio(z[:,121]*G,z[:,120]*D*c['translation_substrates']['NDPK_equilibrium'])),
       scope='Bookkeeping and thermodynamic diagnostics; not a stationarity or human-calibration certificate',
       pump_stoichiometry='3Na out/2K in/1ATP; NCX is sodium-coupled, not a second direct ATP debit',
       CK_AK_NDPK='Internal phosphoryl transfers, not net ATP-energy production',
       missing=['matched intracellular Mg/pH/ionic strength and ATP/ADP/PCr','human pump capacity and temperature dependence','chloride/osmotic and electrode charge balance','whole-cell dialysis','basal nuclear/RNA/KCNQ stationarity'],
       healthy_baseline_accepted=False)
