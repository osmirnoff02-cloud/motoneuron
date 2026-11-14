"""[codex] Optional API0.13 profile compensation; NOT new measured M currents.

Selected from 27 candidates against existing frequency/shape references.
Both Selvaraj gM values +5%; Burley untouched. This is in-sample numerical
reconciliation, not independent validation or joint human Na/ATP calibration.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'2026-10-26-connected-spatial-neuron'))
from connected_parameters import inputs
M_COMPENSATION={'Burley':1.,'Selvaraj-Con1':1.05,'Selvaraj-Con2':1.05}
M_COMPENSATION_SEARCH_BOUNDS=[.85,1.2]
def recovery_neuron_inputs(profile='Burley',**kwargs):
    result=inputs(profile,**kwargs)
    result['profile_p']=result['profile_p'].copy()
    result['profile_p'][12]*=M_COMPENSATION[profile]
    return result
