"""Resolve public candidate IDs without importing the numerical model."""

import json
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent


def params(candidate):
    grid = json.loads((DIRECTORY / 'declared_candidates.json').read_text(encoding='utf-8'))
    candidates = {f'C{index:02d}': parameters for index, parameters in enumerate(grid)}
    selected = DIRECTORY / 'selected_parameters.json'
    candidates.update(json.loads(selected.read_text(encoding='utf-8')))
    try:
        return candidates[candidate]
    except KeyError:
        available = ', '.join(candidates)
        raise ValueError(f'Unknown candidate {candidate!r}. Choose from: {available}') from None
