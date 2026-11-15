"""Verify a public simulation output against its experiment record."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'experiments'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def find_experiment(summary, requested):
    if summary.get('checkpoint_used', False):
        require(requested is None, 'Reference experiments require a fresh run.')
        return None
    duration = summary.get('pulse_duration_s')
    if summary['protocol'] == 'pulse' and duration is None:
        duration = 0.5 if summary['profile'] == 'Burley' else 1.0
    settings = dict(summary, pulse_duration_s=duration)
    for path in sorted(EXPERIMENTS.glob('*.json')):
        record = read_json(path)
        if requested is None or record['id'] == requested:
            if all(settings.get(key) == value for key, value in record['scenario'].items()):
                return record
    require(requested is None, f'Output does not match the {requested!r} experiment record.')
    return None


def check_smoke(output, experiment, checkpoint, report):
    reference = experiment['reference']
    count = experiment['model']['state_count']
    end = experiment['stimulus']['interval_s'][1]
    with np.load(output / 'trace.npz', allow_pickle=False) as trace:
        time, state, baseline = (trace[key] for key in ('time_s', 'state', 'baseline'))
    with np.load(ROOT / reference['path'], allow_pickle=False) as trace:
        ref_time, ref_state, ref_baseline = (trace[key] for key in ('time_s', 'state', 'baseline'))

    require(time.ndim == 1 and len(time) >= 2 and state.shape == (len(time), count)
            and baseline.shape == (count,), 'Smoke trace shape differs from the experiment.')
    require(np.isfinite(time).all() and np.isfinite(state).all()
            and np.isfinite(baseline).all(), 'Smoke trace contains nonfinite values.')
    require((np.diff(time) >= 0).all(), 'Smoke trace time moves backwards.')
    require(np.array_equal(state[-1], checkpoint['state'])
            and np.array_equal(baseline, checkpoint['baseline_state']),
            'Saved trace and checkpoint disagree.')
    require(np.allclose(baseline, ref_baseline, rtol=reference['initial_rtol'],
                        atol=reference['initial_atol']), 'Initial baseline differs from the reference.')
    require(np.allclose(time[[0, -1]], [0, end], rtol=0, atol=reference['time_atol_s'])
            and np.allclose(time[[0, -1]], ref_time[[0, -1]], rtol=0, atol=reference['time_atol_s']),
            'Smoke trace does not cover the reference interval.')

    same_grid = np.array_equal(time, ref_time)
    actual = state if same_grid else state[[0, -1]]
    expected = ref_state if same_grid else ref_state[[0, -1]]
    require(np.allclose(actual, expected, rtol=reference['rtol'], atol=reference['atol']),
            'All-state smoke comparison failed.')
    report.update(
        reference=reference['path'],
        comparison='all_samples' if same_grid else 'endpoints',
        all_state_final_max_difference=float(np.max(np.abs(state[-1] - ref_state[-1]))),
    )
    if not same_grid:
        # Diagnostic only: solver sample times differ, so interior samples are not a pass criterion.
        interpolated = np.column_stack([np.interp(time, ref_time, ref_state[:, index])
                                        for index in range(count)])
        report['interpolated_path_max_difference'] = float(np.max(np.abs(state - interpolated)))


def check_pulse(summary, experiment, report):
    reference = experiment['reference']
    target = next(
        row for row in read_json(ROOT / reference['path'])['rows']
        if row['candidate'] == summary['candidate'] and row['profile'] == summary['profile']
        and row['pA'] == summary['current_pA'] and row['start_s'] == summary['start_time_s']
    )
    features = summary['features']
    require(features['spikes'] == target['features']['spikes'] == reference['spikes_during_step'],
            'Spike count during the current step differs from the reference.')
    peak_difference = abs(features['peak_mV'] - target['features']['peak_mV'])
    require(peak_difference < reference['peak_voltage_atol_mV'],
            'Peak voltage differs from the reference.')
    report.update(reference=reference['path'], spikes_during_step=features['spikes'],
                  peak_mV=features['peak_mV'], peak_voltage_abs_difference_mV=peak_difference)


def check_output(output, requested=None):
    summary = read_json(output / 'summary.json')
    experiment = find_experiment(summary, requested)
    expected_states = 4503 if summary.get('redox_v15') else 4431 if summary.get('acid_tau_s') is not None else 4425
    require(summary['states'] == expected_states, 'Unexpected state count for the model configuration.')
    require(summary['all_finite'] and summary['no_reset'] and summary['human_calibrated'] is False,
            'Simulation reported nonfinite states, a reset, or unexpected calibration metadata.')
    limit = experiment['reference']['balance_atol'] if experiment else 3e-5
    balances = list(summary['checks'].values())
    require(balances and np.isfinite(balances).all() and np.max(np.abs(balances)) < limit,
            'One or more numerical balances exceed the output-check tolerance.')
    checkpoint = read_json(output / 'checkpoint.json')
    schema = checkpoint['state_schema']
    require(schema['count'] == expected_states and len(checkpoint['state']) == expected_states
            and np.isfinite(checkpoint['state']).all(), 'Invalid output checkpoint or state layout.')
    require((output / 'figure.png').is_file(), 'Missing output figure.')
    report = {'protocol': summary['protocol'], 'state_count': expected_states, 'balances_passed': True}

    restart = summary.get('restart')
    if requested and experiment['reference']['kind'] == 'full_state_smoke':
        require(restart is not None, 'Run this reference experiment with --check-restart.')
    if restart is not None:
        require(restart['exact_initial_state'] and restart['within_tolerance'],
                'Checkpoint continuation failed its restart check.')
        if experiment is not None and 'restart_rtol' in experiment['reference']:
            reference = experiment['reference']
            require(restart.get('rtol') == reference['restart_rtol']
                    and restart.get('atol') == reference['restart_atol'],
                    'Restart tolerances used by the run differ from the experiment record.')
        report['restart_passed'] = True
    if experiment is None:
        report['scope'] = 'Numerical checks only; no matching reference experiment.'
    else:
        require(schema['sha256'] == experiment['model']['state_schema_sha256'],
                'Checkpoint state layout differs from the reference experiment.')
        report['experiment'] = experiment['id']
        if experiment['reference']['kind'] == 'full_state_smoke':
            check_smoke(output, experiment, checkpoint, report)
        else:
            check_pulse(summary, experiment, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--experiment', choices=sorted(path.stem for path in EXPERIMENTS.glob('*.json')),
                        help='Require the output to match this reference experiment.')
    args = parser.parse_args()
    try:
        report = check_output(args.output, args.experiment)
    except (ValueError, OSError, KeyError, StopIteration) as error:
        parser.exit(1, f'Output verification failed: {error}\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
