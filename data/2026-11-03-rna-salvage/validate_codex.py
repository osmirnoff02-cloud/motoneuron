"""[codex]2026-11-03. Material/energy boundaries and integrated regression."""
from pathlib import Path
import sys,importlib.util,json
import numpy as np
from dataclasses import replace
from scipy.integrate import solve_ivp
B=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('salvage_prior_validation',B.parent/'2026-11-03-precursor-quality/validate_codex.py');q=importlib.util.module_from_spec(s);sys.modules[s.name]=q;s.loader.exec_module(q)
api=q.api;m=q.m;rs=api._rs42
def save(n,x):q.save.__globals__['B']=B;q.save(n,x)
def run(sg,**kw):return api.run_rna_salvage_neuron(sg,**kw)
def assess(r):
 z=r['state'];b=r['baseline_state'];ch=r['checks']
 assert z.shape[1]==len(r['columns'])==745
 assert max(ch.values())<3e-5,sorted(ch.items(),key=lambda kv:-kv[1])[:8]
 return dict(completed=True,states=745,checks=ch,nfev=r['nfev'],RNA_pools_end=z[-1,726:732],
  RNA_pools_min=z[:,726:732].min(0),RNA_counters=z[-1,732:745],rRNA_produced_Ceq=z[-1,392:394]-b[392:394],
  precursor_end_relative=z[-1,384:392]/b[384:392],local_ribosomes_end_relative=z[-1,564:566]/b[564:566],
  GAP43_end_relative=z[-1,583]/b[583],ATP_min_uM=z[:,[120,136]].min(0),K_end_relative=z[-1,[200,220]]/b[[200,220]])
def boundaries():
 v=rs.RNASalvageParameters().vector();b=np.zeros(16);b[0]=.1;b[1]=7;b[11:13]=[27000,240]
 z=np.zeros(745);z[120]=1000;z[121]=100;z[726:732]=10;old=np.zeros(726);ctl=np.ones(6)
 def f(a=z,c=ctl,o=old):return rs.finish(a,o,v,c,b,100.,1000.,1.)
 a=f();assert a[499]==a[497]==0 and a[120]<0 and a[120]+a[121]==0
 empty=z.copy();empty[726:732]=0;off=ctl.copy();off[0]=0;assert np.all(f(empty,off)==0)
 noATP=z.copy();noATP[120]=0;t=f(noATP,off);assert t[737]==t[738]==t[742]==0 and t[730]>=0
 noexp=ctl.copy();noexp[3]=0;assert f(z,noexp)[736]==0
 nohyd=ctl.copy();nohyd[1:3]=0;assert np.all(f(z,nohyd)[734:736]==0)
 assert rs.gate(empty,v)==0 and rs.gate(z,v)==.1
 o=old.copy();o[392]=.01;o[384]=0;o[398:400]=.01;o[669:671]=.01;o[678]=.01;o[708]=.003
 a=f(z,ctl,o);assert a[732]>0 and a[733]>0 and a[741]>0
 assert abs(a[742]-v[8]*a[737]-2*a[738])<1e-12
 assert abs(a[120]+a[742]-a[743])<1e-12
 assert abs(a[726:732].sum()+a[741]+a[740]+a[744]-a[739]-a[732]-a[733])<1e-12
 try:replace(rs.RNASalvageParameters(),material_recovery_fraction=2).vector()
 except ValueError:pass
 else:raise AssertionError('Invalid yield accepted')
 save('boundaries_codex.json',dict(passed=True,checks=['empty_no_source','zero_ATP_no_activation','RNA_does_not_fill_AA_GTP','adenylate_conserved','export_block','hydrolysis_block','finite_RNA_gating','source_ownership','energy_replacement_once','material_balance','invalid_fraction_rejected']))
 print('boundaries passed',flush=True)
def basic():
 r=run(m.schedule([(0,60,0,1,0)]),**m.m.hist.options(),output_step_s=10,max_step_s=2,rtol=1e-8,atol=1e-10)
 save('quick_codex.json',assess(r));np.savez_compressed(B/'baseline_codex.npz',state=r['baseline_state']);print('quick passed',r['state'].shape,flush=True)
def isolated_grid():
 rows=[]
 for fraction in (0.,.5,1.):
  for tau in (300.,3600.,21600.):
   for price in (1.,2.,4.):
    v=replace(rs.RNASalvageParameters(),material_recovery_fraction=fraction,export_s=tau,nucleoside_to_NMP_ATP_equiv=price,NMP0_uM_nt=10.,NTP_reserve0_uM_nt=10.).vector()
    b=np.zeros(16);b[0]=.1;b[1]=7;b[11:13]=[27000,240]
    z=np.zeros(745);z[120]=1000.;z[121]=100.;z[726:745]=rs.initial(v);z[727]=100.
    ctl=np.ones(6);ctl[0]=0
    def rhs(t,y):
     old=np.zeros(726);en=max(y[120],0.)/(100+max(y[120],0.))*1.1
     tr=np.log(2)/(7*86400)*rs.gate(y,v)*en
     old[384]=old[387]=old[392]=old[393]=tr
     old[120]=-b[0]*tr*(b[11]+b[12]);old[121]=-old[120];old[125]=-old[120]
     return rs.finish(y,old,v,ctl,b,100.,1000.,1.)
    r=solve_ivp(rhs,(0,86400),z,method='LSODA',rtol=1e-9,atol=1e-11,max_step=120)
    assert r.success;yy=r.y.T;ch=rs.checks(yy,z,v,b)
    assert max(ch.values())<1e-7,ch
    assert np.max(abs(yy[:,120]+yy[:,121]-1100))<1e-7
    assert np.max(abs(z[120]-yy[:,120]-yy[:,742]))<1e-7
    assert yy[-1,741]<=20+100*fraction+1e-8
    rows.append(dict(recovery=fraction,export_s=tau,activation_cost=price,new_RNA_nt_uM=yy[-1,741],RNA_energy_uM=yy[-1,742],max_balance=max(ch.values())))
 save('isolated_grid_codex.json',dict(passed=True,rows=rows,scope='27 closed-material isolated rRNA systems, not whole-neuron robustness or measured concentrations; initial100nt-uM lysosomal debris,10NMP+10RNA-NTP,ATP1000,ADP100uM;no external NMP/no ATP production'))
 print('isolated27 passed',flush=True)
def electrical():
 prev=q.q
 oldrun,oldassess,oldB=prev.run,prev.assess,prev.B
 prev.run,prev.assess,prev.B=run,assess,B
 try:prev.electrical()
 finally:prev.run,prev.assess,prev.B=oldrun,oldassess,oldB
def restart():
 sg=m.schedule([(0,120,0,1,0),(120,240,0,1,0)]);ctl=rs.controls(sg);ctl[:,2]=0
 opts=dict(**m.m.hist.options(),output_step_s=20,max_step_s=5,rtol=1e-8,atol=1e-10)
 r=run(sg,rna_salvage_schedule=ctl,**opts);a=run(sg[:1],rna_salvage_schedule=ctl[:1],**opts)
 b=run(sg[1:],rna_salvage_schedule=ctl[1:],checkpoint=a['checkpoint'],**opts)
 tighter=run(sg,rna_salvage_schedule=ctl,**dict(opts,rtol=1e-9,atol=1e-11))
 er=lambda a,b:float(np.max(abs(a-b)/(1+abs(b))))
 re=er(b['state'][-1],r['state'][-1]);te=er(tighter['state'][-1],r['state'][-1]);assert max(re,te)<3e-5
 for x in (r,a,b,tighter):assess(x)
 save('restart_codex.json',dict(passed=True,restart_error=re,tolerance_error=te));print('restart',re,te,flush=True)
def scenarios(names=None):
 sg=m.schedule([(0,7200,0,1,0),(7200,14400,0,1,0),(14400,28800,0,1,0),(28800,36000,0,1,0),(36000,86400,0,1,0)])
 names=names or ['control','limited_reuse','limited_no_reuse','limited_slow_export']
 for name in names:
  bc=np.c_[sg[:,:2],np.ones((5,7))];pc=q.pq.controls(sg);ctl=rs.controls(sg);par=rs.RNASalvageParameters()
  if name!='control':
   bc[[1,3],4]=0;pc[np.ix_([1,3],[2,3,4])]=1;ctl[1:,2]=.25
  if name=='limited_no_reuse':par=replace(par,material_recovery_fraction=0.)
  if name=='limited_slow_export':par=replace(par,export_s=21600.)
  r=run(sg,**m.m.hist.options(),biogenesis_schedule=bc,precursor_quality_schedule=pc,rna_salvage_schedule=ctl,rna_salvage_parameters=par,output_step_s=600,max_step_s=120,rtol=1e-8,atol=1e-10)
  row=assess(r);save(name+'_codex.json',row);save(name+'_checkpoint_codex.json',r['checkpoint']);np.savez_compressed(B/(name+'_codex.npz'),time_s=r['time_s'],state=r['state'])
  print(name,'RNA produced',row['rRNA_produced_Ceq'],'maxcheck',max(row['checks'].values()),flush=True)
if __name__=='__main__':
 action=sys.argv[1] if len(sys.argv)>1 else 'basic'
 if action=='basic':boundaries();basic()
 elif action=='integrated':restart();electrical();scenarios()
 elif action=='scenarios':scenarios(sys.argv[2:] or None)
 elif action=='electrical':electrical()
 elif action=='isolated':isolated_grid()
