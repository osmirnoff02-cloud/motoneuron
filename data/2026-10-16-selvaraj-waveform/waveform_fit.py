"""Same effective mechanism, joint FI and representative-shape refinement."""
from pathlib import Path
import json,sys,time,argparse
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
B=Path(__file__).resolve().parent;ROOT=B.parents[1];OLD=ROOT/'data/2026-10-16-ipsc-expanded-calibration'
sys.path.insert(0,str(ROOT/'data/2026-10-14-ipsc-electrical-multisource'))
import electrical_model as core
FROZEN=json.loads((OLD/'profiles-frozen.json').read_text())
TARGETS={'Con1':np.array([1.50,4.65,5.90,5.14]),'Con2':np.array([1.03,4.72,6.69,6.28])}
LIMITS={'Con1':.5296461082647546,'Con2':.5181216073471556}
CURRENTS=np.array([5.,15.,25.,35.])
INDICES=np.array([0,1,3,6,9,10,11,12,13,14,15])
BOUNDS=[(.15,30),(-56,-30),(.01,3),(-90,-45),(.15,4),(10,600),(.15,5),(0,.08),(10,500),(.015,.6),(.2,12)]
def save(name,x):(B/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n',encoding='utf-8')
def initial(line):
 j=next(j for j in FROZEN['profiles'] if line in j['id']);return np.array(j['parameters']),np.array(j['context'])
@njit
def band(x,lo,hi):return max(lo-x,0.)+max(x-hi,0.)
@njit
def features(v,dt,pre,duration,rest):
 begin=int(round(pre/dt));end=int(round((pre+duration)/dt));peaks=np.empty(1000,np.int64);n=0;last=-100000
 for k in range(begin+1,min(end,len(v)-1)):
  if v[k]<=rest+15 or v[k]<v[k-1] or v[k]<=v[k+1] or (k-last)*dt<3:continue
  left=k-1
  while left>begin and v[left]>v[left-1]:left-=1
  left=max(left,k-int(30/dt));right=min(len(v),k+int(25/dt))
  if v[k]-max(v[left],np.min(v[k:right]))<10:continue
  dvmax=np.max((v[left+1:k+1]-v[left:k])/dt)
  if dvmax<5:continue
  peaks[n]=k;n+=1;last=k
 if n==0:return np.array([0.,0.,0.,0.,100.,float(np.min(v[begin:])-rest),float(np.min(v[end:])-rest),float(v[min(len(v)-1,end+int(200/dt))]-rest),float(v[-1]-rest)])
 amp=0.;troughs=0.;trn=0;hw=0.
 for j in range(n):
  k=peaks[j];amp+=v[k]-rest
  half=rest+(v[k]-rest)/2;lo=k;hi=k
  while lo>begin and v[lo]>half:lo-=1
  while hi<len(v)-1 and v[hi]>half:hi+=1
  hw+=(hi-lo)*dt
  if j<n-1:
   troughs+=np.min(v[k:peaks[j+1]])-rest;trn+=1
 if trn:troughs/=trn
 return np.array([float(n),(peaks[n-1]*dt-pre),amp/n,troughs,hw/n,float(np.min(v[begin:])-rest),float(np.min(v[end:])-rest),float(v[min(len(v)-1,end+int(200/dt))]-rest),float(v[-1]-rest)])
FEATURE_NAMES=['count','last_peak_ms','mean_peak_relative_rest_mV','mean_interspike_trough_relative_rest_mV','mean_width_rest_peak_half_ms','minimum_relative_rest_mV','poststimulus_min_relative_rest_mV','delta_200ms_after_mV','final_delta_mV']
@njit
def evaluate(p,c,target,limit):
 gl,el,_=core.derived(p,c)
 if gl<.001 or gl>1 or not -120<el<50:return 1e6+abs(el),np.zeros((4,9)),999.,999.,999.
 _,rest=core.simulate(p,c,0,1000,.1,False,c[2]+1)
 delta=abs(rest[-1]-c[2]);rng=np.max(rest[-1000:])-np.min(rest[-1000:]);stability=100*(delta**2+rng**2)
 rows=np.empty((4,9));fi=0.;beyond=0.;safety=0.;shape=1e6
 for q in range(4):
  v=core.protocol(p,c,CURRENTS[q],1000,.05,100,400,False,False,c[2]);f=features(v,.05,100,1000,c[2]);rows[q]=f
  diff=abs(f[0]-target[q]);fi+=diff**2;beyond+=max(diff-1,0)**2
  safety+=(max(-5-f[5],0)/3)**2+(abs(f[8])/2)**2
  if f[0]>2 and f[1]<650:safety+=((650-f[1])/50)**2
  if q>0:
   z=(band(f[2],25,40)/3)**2+(band(f[3],0,10)/2)**2+(band(f[4],1,15)/2)**2+(band(f[6],-3,5)/2)**2+(max(abs(f[7])-5,0)/2)**2
   z+=10*(f[0]-5)**2
   if z<shape:shape=z
 rmse=np.sqrt(fi/4)
 loss=25*fi+200*beyond+1000*max(rmse-limit,0)**2+10*shape+5*safety+stability
 return loss,rows,delta,rng,shape
def detailed(p,c,dt=.01):
 out=[]
 for current in CURRENTS:
  v=core.protocol(p,c,current,1000,dt,200,1800,False,False,c[2]);f=features(v,dt,200,1000,c[2]);out.append({'current_pA':float(current),**dict(zip(FEATURE_NAMES,f.tolist())),'old_detector':core.spike_features(v,dt,200,1000)})
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--line',choices=list(TARGETS),default='Con1');ap.add_argument('--iterations',type=int,default=160);ap.add_argument('--seed',type=int,default=120);ap.add_argument('--start-file');args=ap.parse_args();p0,c=initial(args.line)
 start=p0.copy()
 if args.start_file:start=np.array(json.loads(Path(args.start_file).read_text())['parameters'])
 else:start[12]=.01;start[10]=180;start[15]=2;start[0]*=.3
 x0=np.clip(start[INDICES],np.array(BOUNDS)[:,0],np.array(BOUNDS)[:,1]);target=TARGETS[args.line];limit=LIMITS[args.line];t=time.time();steps=[0];history=[]
 def unpack(x):p=p0.copy();p[INDICES]=x;return p
 def obj(x):return evaluate(unpack(x),c,target,limit)[0]
 print('baseline',args.line,evaluate(p0,c,target,limit),flush=True)
 def cb(x,convergence):
  steps[0]+=1;a=evaluate(unpack(x),c,target,limit)
  if steps[0]%10==0:
   rec={'iteration':steps[0],'loss':a[0],'counts':a[1][:,0],'shape':a[4],'elapsed_s':time.time()-t};history.append(rec);print(args.line,steps[0],round(a[0],2),a[1][:,0],round(a[4],2),round(time.time()-t),flush=True)
   save(f'{args.line}-progress.json',{'history':history,'parameters':unpack(x),'context':c})
  return a[0]<30 and a[4]<.01
 opt=differential_evolution(obj,BOUNDS,x0=x0,maxiter=args.iterations,popsize=6,seed=args.seed,polish=False,callback=cb,workers=1,tol=.00001)
 p=unpack(opt.x);out={'author':'codex','date':'2026-10-16','line':args.line,'parameter_names':core.NAMES,'parameters':p,'context':c,'varied_indices':INDICES,'bounds':BOUNDS,'seed':args.seed,'nfev':opt.nfev,'elapsed_s':time.time()-t,'objective':opt.fun,'coarse':evaluate(p,c,target,limit),'fine':detailed(p,c),'history':history,'status':'candidate, pending NEURON and exact acceptance checks'}
 save(f'{args.line}-fit{args.seed}.json',out);print('RESULT',args.line,opt.fun,[(f['current_pA'],f['count'],round(f['mean_peak_relative_rest_mV'],1),round(f['minimum_relative_rest_mV'],1)) for f in out['fine']],flush=True)
