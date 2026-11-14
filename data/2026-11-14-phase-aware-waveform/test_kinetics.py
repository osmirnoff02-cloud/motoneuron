"""Codex. Original-input protection and double-application guard."""
from copy import deepcopy
import numpy as np
import kinetics as k
def main():
    refs=k.phase_kinetic_reference()['profiles'];checked=[]
    for profile,q in refs.items():
        p=np.array(q['original_parameters']);ctx={'keep':{'value':1}}
        o=dict(profile=profile,profile_p=p,control_context=ctx);original=deepcopy(o)
        out=k.apply_phase_kinetics(o)
        assert np.array_equal(p,original['profile_p']) and ctx==original['control_context']
        assert np.array_equal(out['profile_p'],np.array(q['parameters']))
        out['control_context']['keep']['value']=2;assert ctx['keep']['value']==1
        try:k.apply_phase_kinetics(out)
        except ValueError:pass
        else:raise AssertionError('Double application accepted')
        bad=deepcopy(o);bad['profile_p'][0]+=1e-5
        try:k.apply_phase_kinetics(bad)
        except ValueError:pass
        else:raise AssertionError('Foreign baseline accepted')
        bare=k.apply_phase_kinetics(dict(profile=profile,profile_p=p))
        assert 'phase14' in bare['control_context']
        try:k.apply_phase_kinetics(bare)
        except ValueError:pass
        else:raise AssertionError('Double application without initial context accepted')
        checked.append(profile)
    print('kinetic input guards passed:',checked)
if __name__=='__main__':main()
