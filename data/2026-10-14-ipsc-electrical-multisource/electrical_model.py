"""Codex: independent electrical model, explicit solutions and stable-tail check.

Units mV, ms, pA, pF, mS/cm2; Cspec=1uF/cm2 is an assumption.
No calcium equations. Effective gates are not identified gene products.
"""
from pathlib import Path
import json,time
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
B=Path(__file__).resolve().parent
OLD=B.parent/'2026-10-14-healthy-ipsc-electrical'
T=json.loads((OLD/'targets-frozen.json').read_text(encoding='utf-8'))
NAMES=['gna','vm','km','gk','vn','kn','vh','kh','tm','thmin','threc','tn','ga','ta','tmfast','tndeact']
BOUNDS=[(.5,20),(-50,0),(3,20),(.05,3),(-30,30),(5,30),(-80,-35),(2,12),(.03,2),(.1,4),(1,100),(.2,30),(0,.4),(5,400),(.01,.5),(.5,50)]
def nernst(outside,inside,temp=23):return 8.314462618*(273.15+temp)/96485.33212*1000*np.log(outside/inside)
# Salt concentrations, not ionic activities. pH titration additions unreported.
BURLEY=np.array([24.395086091,1.413754718,-37.130904184,nernst(167,7.2),nernst(2.4,140),-70.])
@njit
def sig(x):return 1/(1+np.exp(-max(-700.,min(700.,x))))
@njit
def gates(v,p):return sig((v-p[1])/p[2]),sig((p[6]-v)/p[7]),sig((v-p[4])/p[5]),sig((v+20)/5)
@njit
def taus(v,p):return p[14]+(p[8]-p[14])*sig(-(v+30)/10),p[9]+p[10]*sig(-(v+40)/5),p[15]+(p[11]-p[15])*sig((v+20)/10),2+(p[13]-2)*sig(-(v+20)/5)
@njit
def active(v,p,c):
 m,h,n,w=gates(v,p);return p[0]*m**3*h*(v-c[3])+(p[3]*n**4+p[12]*w)*(v-c[4])
@njit
def derived(p,c):
 rv=c[6] if len(c)>6 else c[5]
 gl=1/(c[0]*c[1])-(active(rv+.01,p,c)-active(rv-.01,p,c))/.02
 el=c[2]+active(c[2],p,c)/gl
 hold=c[0]*(active(c[5],p,c)+gl*(c[5]-el))
 return gl,el,hold
@njit
def simulate(p,c,current=0.,duration=500.,dt=.025,holding=True,init=-70.):
 gl,el,hold=derived(p,c);v=init;m,h,n,w=gates(v,p)
 nt=int(round(duration/dt));out=np.empty(nt+1);out[0]=v;count=0;last=-1000.
 for i in range(nt):
  mi,hi,ni,wi=gates(v,p);tm,th,tn,tw=taus(v,p)
  m=mi+(m-mi)*np.exp(-dt/tm);h=hi+(h-hi)*np.exp(-dt/th);n=ni+(n-ni)*np.exp(-dt/tn);w=wi+(w-wi)*np.exp(-dt/tw)
  gn=p[0]*m**3*h;gk=p[3]*n**4+p[12]*w;gs=gn+gk+gl
  veq=((hold if holding else 0.)/c[0]+current/c[0]+gn*c[3]+gk*c[4]+gl*el)/gs
  vnew=veq+(v-veq)*np.exp(-dt*gs)
  if v<0 and vnew>=0 and i*dt-last>2:count+=1;last=i*dt
  v=vnew;out[i+1]=v
 return count,out
@njit
def iv(p,c,v,hold=-70.,duration=100.):
 m,h,n,w=gates(hold,p);mi,hi,ni,wi=gates(v,p);tm,th,tn,tw=taus(v,p)
 low=0.
 # Log-spaced time sampling; exact gates under ideal voltage step.
 for j in range(160):
  t=.001*np.exp(j*np.log(duration/.001)/159)
  mm=mi+(m-mi)*np.exp(-t/tm);hh=hi+(h-hi)*np.exp(-t/th)
  low=min(low,p[0]*mm**3*hh*(v-c[3]))
 nn=ni+(n-ni)*np.exp(-duration/tn);ww=wi+(w-wi)*np.exp(-duration/tw)
 return low,(p[3]*nn**4+p[12]*ww)*(v-c[4])
@njit
def protocol(p,c,current=30.,duration=500.,dt=.02,pre=100.,post=100.,ramp=False,holding=True,init=-70.):
 gl,el,hold=derived(p,c);v=init;m,h,n,w=gates(v,p)
 nt=int(round((pre+duration+post)/dt));out=np.empty(nt+1);out[0]=v
 for i in range(nt):
  t=i*dt;injected=0.
  if pre<=t<pre+duration:injected=current*((t-pre)/duration if ramp else 1.)
  mi,hi,ni,wi=gates(v,p);tm,th,tn,tw=taus(v,p)
  m=mi+(m-mi)*np.exp(-dt/tm);h=hi+(h-hi)*np.exp(-dt/th);n=ni+(n-ni)*np.exp(-dt/tn);w=wi+(w-wi)*np.exp(-dt/tw)
  gn=p[0]*m**3*h;gk=p[3]*n**4+p[12]*w;gs=gn+gk+gl
  veq=((hold if holding else 0.)/c[0]+injected/c[0]+gn*c[3]+gk*c[4]+gl*el)/gs
  v=veq+(v-veq)*np.exp(-dt*gs);out[i+1]=v
 return out
def spike_features(v,dt=.02,pre=100.,duration=500.,peak_min=-10.,prominence=20.):
 from scipy.signal import find_peaks
 peaks,props=find_peaks(v,height=peak_min,prominence=prominence,distance=max(1,round(2/dt)))
 peaks=peaks[(peaks*dt>=pre)&(peaks*dt<pre+duration)]
 out={'count':len(peaks),'zero_crossings':int(np.sum((v[:-1]<0)&(v[1:]>=0)&(np.arange(len(v)-1)*dt>=pre)&(np.arange(len(v)-1)*dt<pre+duration)))}
 if len(peaks)==0:return out
 k=peaks[0];dv=np.gradient(v,dt);left=max(round(pre/dt),k-round(30/dt))
 # Explicit operational threshold: final rising crossing of10mV/ms before first peak.
 crossings=np.where((dv[left:k-1]<10)&(dv[left+1:k]>=10))[0]+left+1
 th=int(crossings[-1]) if len(crossings) else left
 threshold=v[th];peak=v[k];half=(threshold+peak)/2
 before=np.where(v[th:k+1]>=half)[0];after=np.where(v[k:]<=half)[0]
 width=(k+after[0]-th-before[0])*dt if len(before) and len(after) else np.nan
 end=min(len(v),k+round(100/dt),peaks[1] if len(peaks)>1 else len(v));trough=np.min(v[k:end])
 out.update({'peak_mV':float(peak),'threshold_dv10_mV':float(threshold),'amplitude_from_threshold_mV':float(peak-threshold),'amplitude_from_hold_mV':float(peak-v[max(0,round(pre/dt)-1)]),'halfwidth_ms':float(width),'max_dvdt':float(np.max(dv[left:k+1])),'AHP_from_threshold_mV':float(threshold-trough),'first_peak_ms':float(k*dt)})
 return out
FI=[t for t in T['targets'] if t['metric'].startswith('FI') and t['input']>=5]
IV=[t for t in T['targets'] if not t['metric'].startswith('FI')]
def objective(p,detail=False):
 c=BURLEY;gl,el,hold=derived(p,c)
 if gl<=.002 or not -100<el<30 or hold>=0:return 1e5+abs(el)
 # All previously exposed Burley points are development data, not a new test set.
 counts=[simulate(p,c,t['input'])[0] for t in FI]
 fi=sum(((v-t['value'])/max(1,t['sem']))**2 for t,v in zip(FI,counts))
 vv=[((iv(p,c,t['input'])[0 if t['metric'].startswith('Na') else 1]-t['value'])/10)**2 for t in IV]
 _,trace=simulate(p,c,0,1600,.05,False)
 tail=trace[-4000:];rest=float(np.mean(tail));ptp=float(np.ptp(tail))
 stability=((rest-c[2])/2)**2+(ptp/2)**2
 base=simulate(p,c,0,400,.05)[1][-1];neg=simulate(p,c,-1,400,.05)[1][-1]
 rin=base-neg;passive=((rin-c[1])/.15)**2
 score=fi+sum(vv)+stability+passive
 if detail:return {'score':score,'FI_loss':fi,'IV_loss':sum(vv),'stability_loss':stability,'rest':rest,'tail_range':ptp,'Rin':rin,'gl':gl,'el':el,'hold':hold,'FI_counts':counts}
 return score
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=31);ap.add_argument('--iters',type=int,default=100);args=ap.parse_args()
 old=json.loads((OLD/'parameters-frozen.json').read_text())['parameters'];x=np.array(old+[.03,5.])
 st=time.time();print('start',args.seed,objective(x,True),flush=True);it=[0]
 def cb(x,convergence):
  it[0]+=1
  if it[0]%10==0:print('iteration',it[0],'loss',objective(x),'seconds',round(time.time()-st),flush=True)
 r=differential_evolution(objective,BOUNDS,maxiter=args.iters,popsize=6,seed=args.seed,x0=x,polish=False,callback=cb)
 out={'author':'codex','seed':args.seed,'nfev':r.nfev,'elapsed_s':time.time()-st,'names':NAMES,'parameters':r.x.tolist(),'context':BURLEY.tolist(),'detail':objective(r.x,True),'bounds':BOUNDS,'validation_status':'Burley reused development data; no held-out claim'}
 (B/f'burley-fit{args.seed}.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
