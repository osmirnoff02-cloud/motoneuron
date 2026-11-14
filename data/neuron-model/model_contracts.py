"""[codex] 2026-11-03. Runtime contracts, not biological calibration."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import threading

_LOAD_LOCK = threading.RLock()


def load_module(path, name):
    """Reuse a loaded file under explicit aliases; reject name/path collisions.

    Historical imports outside this loader can still create duplicate identities.
    No bare filename alias is installed: e.g. transport.py is not unique.
    """
    path = Path(path).resolve()
    with _LOAD_LOCK:
        named = sys.modules.get(name)
        if named is not None:
            if Path(getattr(named, '__file__', '')).resolve() != path:
                raise ImportError(f'Module alias {name!r} already belongs to another file')
            return named
        for module in tuple(sys.modules.values()):
            file = getattr(module, '__file__', None)
            if file and Path(file).resolve() == path:
                sys.modules[name] = module
                return module
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load {path}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            if sys.modules.get(name) is module:
                del sys.modules[name]  # only Python cache, never filesystem data
            raise
        return module


def tools_root():
    """Portable sibling default, with an explicit override for another PC."""
    default = Path(__file__).resolve().parents[3] / 'als-tools'
    return Path(os.environ.get('ALS_TOOLS_ROOT', default)).expanduser().resolve()


def mechanism_path(run):
    path = tools_root() / 'runs' / run / 'nrnmech.dll'
    if not path.is_file():
        raise FileNotFoundError(f'NEURON mechanism missing: {path}; build it for this platform/NEURON version or set ALS_TOOLS_ROOT')
    return path


def state_schema(columns, config):
    """Check the frozen positional ABI before integrating or reading checkpoints.

    The JSON is reviewed data, never regenerated during a simulation. Changes to
    state order require an explicit schema migration and review of JIT indices.
    """
    manifest = json.loads(Path(__file__).with_name('state_layout_v1.json').read_text(encoding='utf8'))
    actual = list(columns)
    expected = manifest['base_1600'][:min(1600, len(actual))]
    for key, block in (('individual_RP', 'individual_RP'), ('RP_production', 'RP_production'),
                       ('nuclear_transport', 'nuclear_transport')):
        if key in config:
            expected += manifest[block]
    if 'RNA_export' in config:
        expected += manifest['RNA_export_full' if 'individual_RP' in config else 'RNA_export_minimal']
    if 'passive_chloride' in config:
        expected += manifest['passive_chloride']
    # [claude] 2026-11-13: optional six-state acid-extrusion block after passive Cl (plan N05.1.4).
    if 'acid_extrusion' in config:
        expected += manifest['acid_extrusion_claude']
    # Alternate names describe the same legacy slot, not an allowed permutation.
    for index, aliases in manifest['legacy_aliases'].items():
        index = int(index)
        if index < min(len(actual), 1600) and actual[index] in aliases:
            actual[index] = manifest['base_1600'][index]
    if actual != expected:
        first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
        raise ValueError(f'State layout changed at index {first}; review positional kernels and migrate state_layout_v1.json')
    digest = hashlib.sha256(json.dumps(list(columns), ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()
    return dict(version=manifest['version'], sha256=digest, count=len(columns),
                labels='backend physical state order, before facade display aliases')


def check_checkpoint_schema(checkpoint, schema):
    if checkpoint is None:
        return 'new'
    stored = checkpoint.get('state_schema')
    if stored is None:
        return 'legacy_without_schema'  # legacy parameter/baseline checks still run
    if any(stored.get(key) != schema[key] for key in ('version', 'sha256', 'count')):
        raise ValueError('Checkpoint state schema differs; explicit migration is required')
    return 'verified'


def solver_atol(atol, count):
    import numpy as np
    value = np.asarray(atol, dtype=float)
    if value.ndim > 1 or (value.ndim == 1 and value.shape != (count,)):
        raise ValueError(f'atol must be scalar or have exactly {count} components')
    if not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError('atol must be finite and strictly positive')
    return float(value) if value.ndim == 0 else value.copy()


def retire_limitations(result, rule_id, prefixes):
    """Legacy adapter: retain provenance and fail if upstream prose changed."""
    prefixes = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
    removed = [text for text in result['limitations'] if text.startswith(prefixes)]
    if not removed:
        raise RuntimeError(f'Limitation contract {rule_id}: no matching ancestor; review metadata migration')
    result.setdefault('limitation_history', []).append(dict(id=rule_id, retired=removed))
    return [text for text in result['limitations'] if not text.startswith(prefixes)]


def revise_limitations(result, rule_id, replacements):
    """Checked compatibility adapter; structured observation keys are preferred."""
    current = list(result['limitations'])
    changes = []
    for old, new in replacements:
        found = [text for text in current if old in text]
        if not found:
            raise RuntimeError(f'Limitation contract {rule_id}: missing phrase {old!r}')
        changes.append(dict(before_fragment=old, after_fragment=new))
        current = [text.replace(old, new) for text in current]
    result.setdefault('limitation_history', []).append(dict(id=rule_id, replacements=changes))
    return current
