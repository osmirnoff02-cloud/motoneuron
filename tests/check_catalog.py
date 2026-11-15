"""Check that model manifests and experiment records agree with each other and the package."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED_KEYS = ('release', 'state_count', 'state_schema_sha256')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def resolve_pointer(document, pointer):
    for token in pointer.strip('/').split('/'):
        document = document[int(token)] if isinstance(document, list) else document[token]
    return document


def check_model(path, errors):
    name = path.relative_to(ROOT).as_posix()
    model = read_json(path)
    runtime = model['runtime']
    if model['id'] != path.stem:
        errors.append(f'{name}: id differs from the file name')
    if model.get('human_calibrated') is not False:
        errors.append(f'{name}: human_calibrated must be false')
    for key in ('evaluation', 'base_model'):
        if key in model and not (ROOT / model[key]).is_file():
            errors.append(f'{name}: missing {key} {model[key]}')
    for key in ('entry_point', 'state_layout'):
        if not (ROOT / runtime[key]).is_file():
            errors.append(f'{name}: missing {key} {runtime[key]}')
    for source in model['parameter_sources']:
        target = ROOT / source['path']
        if not target.is_file():
            errors.append(f'{name}: missing parameter source {source["path"]}')
        elif 'json_pointer' in source:
            try:
                resolve_pointer(read_json(target), source['json_pointer'])
            except (KeyError, IndexError, ValueError, TypeError):
                errors.append(f'{name}: {source["json_pointer"]} does not resolve in {source["path"]}')
    for listed in model['experiments']:
        if not (ROOT / listed).is_file():
            errors.append(f'{name}: missing experiment {listed}')
            continue
        experiment = read_json(ROOT / listed)
        declared = experiment['model']
        if declared.get('id') != model['id'] or declared.get('manifest') != name:
            errors.append(f'{listed}: does not point back to {name}')
        for key in SHARED_KEYS:
            expected = model['release'] if key == 'release' else runtime[key]
            if declared.get(key) != expected:
                errors.append(f'{listed}: {key} differs from {name}')
        scenario = experiment['scenario']
        for key in ('profile', 'candidate'):
            if scenario[key] != model[key]:
                errors.append(f'{listed}: {key} differs from {name}')
        if scenario['redox_v15'] != runtime['redox_v15']:
            errors.append(f'{listed}: redox_v15 differs from {name}')
    return model


def main():
    errors = []
    models = [check_model(path, errors) for path in sorted((ROOT / 'models').glob('*.json'))]
    listed = {experiment for model in models for experiment in model['experiments']}
    for path in sorted((ROOT / 'experiments').glob('*.json')):
        name = path.relative_to(ROOT).as_posix()
        record = read_json(path)
        if record['id'] != path.stem:
            errors.append(f'{name}: id differs from the file name')
        if name not in listed:
            errors.append(f'{name}: not listed by any model manifest')
        reference = record['reference'].get('path')
        if reference and not (ROOT / reference).is_file():
            errors.append(f'{name}: missing reference {reference}')
    if errors:
        raise SystemExit('Catalog check failed:\n' + '\n'.join(errors))
    print(f'{len(models)} model manifests and {len(listed)} experiment records are consistent.')


if __name__ == '__main__':
    main()
