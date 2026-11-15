"""Exercise rejection paths through the public simulation command."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def run_demo(*args):
    return subprocess.run(
        [sys.executable, '-I', '-B', '-X', 'utf8', str(ROOT / 'run_demo.py'), *args],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
    )


def expect_error(result, message, expected_exit=None):
    if (result.returncode == 0 or message not in result.stderr
            or (expected_exit is not None and result.returncode != expected_exit)):
        raise SystemExit(f'Expected {message!r}, got exit {result.returncode}:\n{result.stderr}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('smoke_output', type=Path, help='Successful default-smoke experiment output')
    args = parser.parse_args()
    checkpoint = json.loads((args.smoke_output / 'checkpoint.json').read_text(encoding='utf-8'))

    # Keep caches with the simulation output and clean the test files on exit.
    with tempfile.TemporaryDirectory(prefix='input-check-', dir=args.smoke_output) as folder:
        directory = Path(folder).resolve()
        output = directory / 'invalid-candidate'
        for candidate in ('C-1', 'C8', 'C99', 'unknown'):
            result = run_demo('--candidate', candidate, '--output', str(output))
            expect_error(result, f'Unknown candidate {candidate!r}', expected_exit=2)
            if output.exists():
                raise SystemExit('Invalid candidate created an output directory.')

        checkpoint['state_schema']['sha256'] = '0' * 64
        invalid = directory / 'incompatible.json'
        invalid.write_text(json.dumps(checkpoint), encoding='utf-8')
        result = run_demo('--protocol', 'smoke', '--checkpoint', str(invalid),
                          '--output', str(directory / 'incompatible-output'))
        expect_error(result, 'Checkpoint state schema differs')
    print('Invalid candidate IDs and incompatible checkpoints were rejected.')


if __name__ == '__main__':
    main()
