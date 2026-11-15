# Motoneuron

Motoneuron is a Python research model for studying how electrical activity and cellular maintenance interact in motor neurons in the context of amyotrophic lateral sclerosis (ALS). It couples membrane currents, calcium and potassium handling, and ATP supply with selected mechanisms for protein turnover, RNA processing, and translation. Researchers can compare modeled responses to current injection and AMPA receptor activation and identify measurements that would distinguish competing assumptions. [Model equations](docs/MODEL.md) · [Research scope](docs/SCOPE.md)

Release **0.7.0** includes an opt-in scenario for amino-acid delivery, glutathione (GSH) recycling, and NADPH supply. The default model has 4,425 state variables and the redox scenario has 4,503, including bookkeeping counters. C08, the configuration used below, is a candidate from a parameter search. Human motor-neuron calibration is still open, and the redox scenario's initial GSH pool drifts even in unstimulated controls. The simulations cannot establish ALS causation or predict treatment efficacy. [Redox scenario and limitations](docs/REDOX_STAGE_0.7.0.md) · [Model assumptions](docs/ASSUMPTIONS.md)

## Quick start

Clone the repository and create a Python **3.12** virtual environment. Installation uses the pinned dependencies. The simulations run locally.

```bash
git clone https://github.com/osmirnoff02-cloud/motoneuron.git
cd motoneuron
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, use `py -3.12 -m venv .venv` followed by `.\.venv\Scripts\Activate.ps1` for the environment setup. The original export validation used Windows. [Validation record](docs/VALIDATION.md)

From the repository directory, run:

```bash
python -m pip install -r requirements-lock.txt
python -I -B -X utf8 verify_package.py
python -I -B -X utf8 run_demo.py --protocol smoke --profile Burley --candidate C08 --check-restart --output outputs/default-smoke
python -I -B -X utf8 tests/check_output.py outputs/default-smoke --experiment default-smoke
```

The default smoke test integrates all 4,425 states for 20 ms, checks numerical balances, and verifies checkpoint continuation. Choose a new output directory for each run.

## Run an experiment

The experiment index lists runnable reference scenarios, their expected outputs, and the status of historical studies. Start there for the Burley 25 pA pulse or the 4,503-state redox smoke test. [Experiment index](experiments/README.md) · [Extended protocols](docs/RUNNING.md)

Each run writes a summary, configuration, provenance, checkpoint, selected time series, and a figure. Smoke runs also save a full-state trace. Use `--full-trace` to save one for another protocol. [Output formats](docs/RUNNING.md#files-written) · [Bundled results](results/README.md)

## Explore the repository

| Path | Contents |
| --- | --- |
| [models](models/README.md) | Supported configurations, parameter sources, and evaluations |
| [experiments](experiments/README.md) | Reference protocols, expected results, and reproduction status |
| [run_demo.py](run_demo.py) | Supported entry point for simulations |
| [data/neuron-model](data/neuron-model/README.md) | Integrated model, interface, and state layout |
| [data](data/README.md) | Dated model components and source-linked inputs |
| [parameters](parameters/README.md) | Parameter inventory, scenario settings, and experimental data catalog |
| [docs](docs/README.md) | Model documentation, assumptions, and validation |
| [provenance](provenance/README.md) | Package checksums, source records, and release verification |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## Contribute or cite

The project seeks experimental data for 43 groups of unresolved model questions. Matched measurements from human iPSC-derived motor neurons, including recording conditions, cell maturity, and morphology, can help constrain competing mechanisms. Model reviews and reproducibility reports are also welcome. [Experimental data requests](DATA_REQUESTS.md) · [Contributing](CONTRIBUTING.md) · [Issue templates](.github/ISSUE_TEMPLATE/README.md)

Use the project citation when referencing this software. [Citation metadata](CITATION.cff) · [Project mission](MISSION.md)

## License

The Motoneuron Noncommercial Research License 1.0 permits noncommercial scientific research. Commercial use of material offered only under these terms requires a separate written agreement. Version 0.1.2 remains available under AGPL-3.0-only, and separately licensed third-party material retains its terms. [License](LICENSE.txt) · [Commercial enquiries](COMMERCIAL_LICENSE.md) · [License scope](docs/LICENSE_SCOPE.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)
