"""Codex. Late-spike, censoring and sampling-invariance regression tests."""
import numpy as np
from observer import observe
def main():
    t=np.arange(0,1.70001,.0002);v=np.full(t.shape,-65.)
    for peak in (.4,.7032):v+=80*np.exp(-((t-peak)/.0007)**2)
    q=observe(t,v,.7)
    assert q['pulse_count']==1 and q['post_count']==1,q
    assert abs(q['last_event_s']-.7032)<.00021,q
    assert abs(q['return_from_off_ms']-q['return_from_last_actual_ms']-3.2)<.21,q
    dup_t=np.repeat(t,2);dup_v=np.repeat(v,2)
    assert observe(dup_t,dup_v,.7)==q
    w=v.copy();w[t>=.71]-=5
    z=observe(t,w,.7);assert z['recovery_censored'] and z['return_from_last_actual_ms'] is None,z
    r=observe(t,np.full(t.shape,-65.),.7)
    assert r['pulse_count']==r['post_count']==0 and r['return_from_off_ms']==0 and r['return_from_last_actual_ms'] is None,r
    print('4 observer regressions passed')
if __name__=='__main__':main()
