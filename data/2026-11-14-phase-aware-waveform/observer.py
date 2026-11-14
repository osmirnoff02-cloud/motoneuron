"""Codex 2026-11-14. Offset and last-actual-spike recovery; fixed .2s onset.
Scope: .2ms uniform observer grid, 1mV final-return band, no fitting.
"""
import numpy as np
from scipy.signal import find_peaks

def observe(t,v,off,rest=None):
    keep=np.r_[True,np.diff(t)>0];t=t[keep];v=v[keep]
    if rest is None:rest=float(np.mean(v[(t>=.1)&(t<.2)]))
    # Uniform grid prevents adaptive sampling density from changing detection.
    g=np.arange(t[0],t[-1]+1e-9,.0002);u=np.interp(g,t,v)
    pp,_=find_peaks(u,height=rest+15,prominence=10,distance=15)
    ev=[]
    for k in pp:
        lo=max(0,k-150)
        if max(np.diff(u[lo:k+1])/.2,default=0)<5:continue
        ev.append(float(g[k]))
    ev=[x for x in ev if x>=.2]
    tail=g>=off-1e-10;outside=np.flatnonzero(tail&(abs(u-rest)>1))
    ret=None if len(outside) and outside[-1]==len(g)-1 else (float(g[outside[-1]+1]-off)*1000 if len(outside) else 0.)
    last=max(ev) if ev else None
    return dict(rest_mV=rest,event_times_s=ev,pulse_count=sum(x<off-1e-8 for x in ev),post_count=sum(x>=off-1e-8 for x in ev),
        return_from_off_ms=ret,return_from_last_actual_ms=None if ret is None or last is None else ret+(off-last)*1000,
        post_min_mV=float(u[tail].min()-rest),post_abs_area_mVs=float(np.trapezoid(abs(u[tail]-rest),g[tail])),
        recovery_censored=ret is None,tail_s=float(g[-1]-off),last_event_s=last)
