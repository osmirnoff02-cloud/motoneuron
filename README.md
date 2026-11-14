## Current version 0.6.0

[Stimulus offsets, recovery and expanded kinetic search](docs/PHASE_STAGE_0.6.0.md). Explicit opt-in: `--acid-tau-s 120 --phase-v14`, with 4431 states. Legacy defaults use 4425 states; prior scenarios remain available. Conditional prototype; human calibration remains open.

## Previous stage 0.5.0

[Waveform and voltage recovery stage](docs/WAVEFORM_STAGE_0.5.0.md). Includes the 0.4.0 data requests and filename migration. Conditional research prototype; human calibration remains open.

# Motoneuron / Мотонейрон

**[Help improve the model: experimental data requests (U01–U43)](DATA_REQUESTS.md)** — share measurements, dataset links or corrections for unresolved parameters.

**Research snapshot 0.6.0, 2026-11-14. Research prototype; not a calibrated human motor neuron or a predictor of treatment efficacy.**

This repository makes a selected computational branch of the ALS research project inspectable and reproducible. It contains the full coupled neuron model used by the included examples, its actual runtime inputs, a parameter inventory, selected historical results and an explicit account of limitations. It is not the complete private research archive.

[Русская инструкция](README_RU.md) · [Scope](docs/SCOPE.md) · [Scientific model](docs/MODEL.md) · [Assumptions](docs/ASSUMPTIONS.md) · [Validation](docs/VALIDATION.md)

## Mission

The mission of Motoneuron is to keep the model open and accessible for research into ALS and related neurological diseases, and to support collaborative development of methods for their prevention and treatment. Achieving this goal should not end the availability of published versions.

[Approved mission in Russian and English](MISSION.md). This statement does not add conditions to the license.

## Research question

How do membrane currents, ion transport, energy supply, protein/RNA maintenance and nuclear transport interact in conditional motor-neuron scenarios? Which missing measurements most constrain these mechanisms? Current examples examine joint KCNQ/passive-current/Na-K-pump configurations. They do not establish ALS causation or therapeutic benefit.

The integrated numerical model has **4425 state variables, including bookkeeping counters and projections**. This is not 4425 experimentally measured independent biological quantities. The documented example uses candidate **C08**, which is an explicit experimental configuration, not an accepted healthy default. The wider project also investigates TDP-43, TMEM106B/lysosomes, iron/GPX4, microglia, membranes and resistant/vulnerable motor-neuron populations; these directions are not all implemented in this executable model. See [scope](docs/SCOPE.md).

## License and commercial use

**Free for noncommercial scientific research; commercial use requires a separate written agreement.** Licensors: **Максим Смирнов (Maxim Smirnov) and Олег Смирнов (Oleg Smirnov)**. The custom [research license](LICENSE.txt) permits research use, modification and sharing. See [commercial enquiries](COMMERCIAL_LICENSE.md). This is source-available software, not OSI open source.

The new terms start with **0.2.0** and later releases expressly carrying them. **[0.1.2 remains AGPL-3.0-only](https://github.com/osmirnoff02-cloud/motoneuron/tree/bfd151cb24cc057af04efe7ea8c6aed129729385)**; existing rights cannot be withdrawn. Release 0.2.0 changed licensing/documentation only; code already released in 0.1.2 remains available there under AGPL, including for compliant commercial use. Later original additions are governed by their stated terms. [Version boundary and third-party scope](docs/LICENSE_SCOPE.md).

## Quick start

Get the code from [GitHub](https://github.com/osmirnoff02-cloud/motoneuron) using **Code → Download ZIP**, then extract it, or run:

```bash
git clone https://github.com/osmirnoff02-cloud/motoneuron.git
cd motoneuron
```

Use Python **3.12**. The tested system and measured runtime are recorded in [validation](docs/VALIDATION.md). Run from the repository directory.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -I -B -X utf8 verify_package.py
.\.venv\Scripts\python.exe -I -B -X utf8 run_demo.py --protocol smoke --check-restart --output outputs/smoke
.\.venv\Scripts\python.exe -I -B -X utf8 tests/check_output.py outputs/smoke
```

Python-venv equivalent for Linux/macOS (not platform-validated by this release):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -I -B verify_package.py
.venv/bin/python -I -B run_demo.py --protocol smoke --check-restart --output outputs/smoke
.venv/bin/python -I -B tests/check_output.py outputs/smoke
```

The smoke example integrates **all states for 20 ms**, checks numerical balances and optionally compares checkpoint continuation with an uninterrupted 40-ms calculation. It is an installation/regression check, not a long-term biological validation. Initial imports and Numba compilation can take minutes. A reused output directory is rejected; choose a new name for each run. Nothing is automatically deleted.

## An electrical example

```powershell
.\.venv\Scripts\python.exe -I -B -X utf8 run_demo.py --protocol pulse --profile Burley --candidate C08 --current-pa 25 --output outputs/pulse25
.\.venv\Scripts\python.exe -I -B -X utf8 tests/check_output.py outputs/pulse25
```

This is a 25-pA step from 0.2 to 0.7 s, followed by 1 s recovery. The historical model regression target is **4 spikes during the step**. It is not a new independent human measurement. The example writes `figure.png`, `timeseries.csv`, numerical checks, configuration, provenance and the complete final checkpoint. Full trajectories are saved for the smoke test; add `--full-trace` for other protocols if needed. Large generated files stay under ignored `outputs/`.

[Extended protocols and outputs](docs/RUNNING.md) · [Selected historical results](results/README.md) · [Parameter inventory](parameters/README.md) · [Data availability](docs/DATA.md)

## What would help

- Review the equations, units, state ownership and conservation accounting.
- Supply or identify matched human iPSC-MN KCNQ activation/deactivation, passive Na/K/Cl and pump/recovery measurements, with recording solutions, temperature, cell maturity and morphology.
- Propose independent protocols that distinguish competing mechanisms and falsify model assumptions.
- Check reproducibility and report problems using the issue templates; preserve the exact candidate, configuration and version.

Core limitations include unresolved long-duration potassium recovery, incomplete PMCA/proton/pH coupling, uncertain absolute molecular pools and rates, and incomplete human calibration. A numerical pass does not remove these limitations. Negative historical outcomes are included.

## Provenance and reuse

The computational source layout retains dated directories to preserve path-based imports. [The source manifest](provenance/source_manifest.json) identifies each original relative path and both source/export SHA-256 values. Two legacy import paths were made lazy to avoid loading NEURON for a SciPy/Numba calculation; equations and numerical parameter values were not changed. The original private Git history, correspondence, patient records, downloaded papers and software installations are not included.

Original project material in this release is offered under **Motoneuron Noncommercial Research License 1.0**, subject to the prior-license and third-party exceptions described above. The [mission](MISSION.md) is a statement of purpose, not an extra license condition. Third-party software and data retain their own terms; [AGPL for 0.1.2 is preserved](provenance/AGPL-3.0-0.1.2.txt).

Repository: [https://github.com/osmirnoff02-cloud/motoneuron](https://github.com/osmirnoff02-cloud/motoneuron). Maintained through the `osmirnoff02-cloud` account. Citation uses project-level attribution; a complete list of human contributors has not been supplied. This snapshot does not include later development in the private workspace. See [third-party notes](THIRD_PARTY_NOTICES.md), [citation](CITATION.cff) and [release checklist](PUBLICATION_CHECKLIST.md). Computational code was developed with AI assistance.


## Acid-removal stage 0.3.0

[New optional 4431-state scenario, results and commands](docs/ACID_STAGE_0.3.0.md). Prior 4425-state demo remains available with acid disabled. Human calibration and waveform limitations remain.
