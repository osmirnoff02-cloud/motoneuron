"""Codex 2026-11-13: defer optional NEURON imports, never emulate its results.

The published full integrated ODE does not call these legacy replay routines.
Explicit replay calls still require the real NEURON and compiled mechanisms.
"""
import importlib

class _Lazy:
    def __init__(self, loader):
        object.__setattr__(self,'_loader',loader)
        object.__setattr__(self,'_value',None)
    def _get(self):
        if self._value is None:object.__setattr__(self,'_value',self._loader())
        return self._value
    def __getattr__(self,name):return getattr(self._get(),name)
    def __setattr__(self,name,value):setattr(self._get(),name,value)

def _replay():
    try:return importlib.import_module('neuron_replay_codex')
    except (ImportError,FileNotFoundError) as exc:
        raise ImportError('Optional legacy NEURON replay is outside the tested publication interface. Use run_demo_codex.py for the complete SciPy/Numba model; see docs/SCOPE_codex.md.') from exc

nr=_Lazy(_replay)
h=_Lazy(lambda:_replay().h)
