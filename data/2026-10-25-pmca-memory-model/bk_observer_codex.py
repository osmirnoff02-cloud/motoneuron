"""Codex: Constantin2025 SI Table3 BK observer, separate from human channels.

States C0..C4,O0..O4; Ca in uM, seconds, mV. No Ca clearance inferred
from a fitted current tau. Temperature is an explicit input (default 23C).
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit

BK_SOURCE_PARAMETERS=dict(gamma_open_per_s=(1/18.03,1/80.02,1/.98,1/.008,1/.006),
 gamma_close_per_s=(1/.0009,1/.016,1/.0012,1/.00016,1/.002),q_open=2.25,q_close=-.11,
 kc_on_per_uM_s=(34.1,3972.8,208.92,711.2),kc_off_per_s=(1741.68,1015.79,20617.6,4449.017),
 ko_on_per_uM_s=(537.19,727.16,32.63,314.8),ko_off_per_s=(7536.7,29.81,20.9,121.1))

def bk_generator(ca_uM,voltage_mV=50.,temperature_C=23.):
 if not np.isfinite([ca_uM,voltage_mV,temperature_C]).all() or ca_uM<0 or temperature_C<=-273.15:raise ValueError('Invalid BK protocol')
 p=BK_SOURCE_PARAMETERS;Q=np.zeros((10,10));v=voltage_mV*.001*96485.33212/(8.314462618*(273.15+temperature_C))
 def add(i,j,k):Q[j,i]+=k;Q[i,i]-=k
 for i in range(5):
  add(i,i+5,p['gamma_open_per_s'][i]*np.exp(p['q_open']*v));add(i+5,i,p['gamma_close_per_s'][i]*np.exp(p['q_close']*v))
 for offset,on,off in [(0,'kc_on_per_uM_s','kc_off_per_s'),(5,'ko_on_per_uM_s','ko_off_per_s')]:
  for i in range(4):add(offset+i,offset+i+1,p[on][i]*ca_uM);add(offset+i+1,offset+i,p[off][i])
 return Q

def stationary(Q):
 a=Q.copy();a[-1,:]=1.;b=np.zeros(10);b[-1]=1.;return np.linalg.solve(a,b)

def observe_bk_ca_steps(segments,*,voltage_mV=50.,temperature_C=23.,initial_ca_uM=0.,dt_s=.0001,initial_state=None):
 sg=np.asarray(segments,float)
 if sg.ndim!=2 or sg.shape[1]!=3 or not len(sg) or not np.isfinite(sg).all() or np.any(sg<0) or np.any(sg[:,1]<=sg[:,0]) or np.any(sg[1:,0]!=sg[:-1,1]) or not np.isfinite(dt_s) or dt_s<=0:raise ValueError('Invalid step schedule')
 z=stationary(bk_generator(initial_ca_uM,voltage_mV,temperature_C)) if initial_state is None else np.asarray(initial_state,float)
 if z.shape!=(10,) or not np.isfinite(z).all() or min(z)<-1e-10 or abs(z.sum()-1)>1e-9:raise ValueError('Invalid BK state')
 ts=[sg[0,0]];zs=[z.copy()]
 for start,end,c in sg:
  Q=bk_generator(c,voltage_mV,temperature_C);last=start;step=expm(Q*dt_s)
  for t in np.r_[np.arange(start+dt_s,end,dt_s),end]:
   z=(step if abs(t-last-dt_s)<1e-10 else expm(Q*(t-last)))@z;ts.append(t);zs.append(z.copy());last=t
 z=np.asarray(zs)
 if min(z.ravel())<-1e-9 or max(abs(z.sum(axis=1)-1))>1e-8:raise ArithmeticError('BK probability balance')
 return dict(time_s=np.asarray(ts),state=z,open_probability=z[:,5:].sum(axis=1),voltage_mV=voltage_mV,temperature_C=temperature_C,
             source='Constantin2025 SI Table3, calibrated there against Berkefeld2013; human_MN_fit=False')
