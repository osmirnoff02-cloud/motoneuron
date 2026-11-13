# Running and interpreting experiments

Codex, 2026-11-13. [Home](../README.md).

Use the root wrapper with Python 3.12 and `requirements-lock.txt`. Dependency installation uses the package index; the simulation itself requires no network. `-I` avoids inheriting workstation PYTHONPATH entries. Generated output directories must be new. The repository sources are protected against model writes; no cleanup is automatic.

## Commands

```powershell
# Brief full-state calculation and restart verification
.\.venv\Scripts\python.exe -I -B -X utf8 run_demo_codex.py --protocol smoke --check-restart --output outputs/smoke-1

# Current step; profiles: Burley, Selvaraj-Con1, Selvaraj-Con2
.\.venv\Scripts\python.exe -I -B -X utf8 run_demo_codex.py --protocol pulse --profile Burley --candidate C08 --current-pa 25 --output outputs/pulse-1

# Long preparation (much more expensive; not part of the quick check)
.\.venv\Scripts\python.exe -I -B -X utf8 run_demo_codex.py --protocol rest72 --profile Selvaraj-Con2 --candidate C08 --output outputs/rest-C08

# Each branch starts from the SAME matching 72-hour checkpoint
.\.venv\Scripts\python.exe -I -B -X utf8 run_demo_codex.py --protocol load36 --profile Selvaraj-Con2 --candidate C08 --branch repeated --opening 0.001 --checkpoint outputs/rest-C08/checkpoint_codex.json --output outputs/repeated-C08
.\.venv\Scripts\python.exe -I -B -X utf8 run_demo_codex.py --protocol load36 --profile Selvaraj-Con2 --candidate C08 --branch control --checkpoint outputs/rest-C08/checkpoint_codex.json --output outputs/control-C08
```

`continuous` applies six hours at the beginning; `repeated` applies six one-hour intervals starting at hours 0/4/8/12/16/20 over 36 hours. `control` has no AMPA activation. Equal integrated AMPA opening does not guarantee equal Ca entry. A load checkpoint must be at the end of its 72-hour preparation and match candidate, profile, state schema and configuration. The model rejects incompatible configurations; never edit checkpoint fingerprints to bypass this.

Long schedules follow the inherited protocol construction, but complete 72+36-hour revalidation is outside the export test. They may require hours or longer depending on hardware and solver behavior. Evaluate short runs first, monitor resources and retain failures. They are not background jobs launched by installation.

## Files written

| File | Purpose |
|---|---|
| `summary_codex.json` | All-state numerical checks, protocol, runtime and electrical features when applicable |
| `config_codex.json` / `provenance_codex.json` | Numerical configuration and its scientific qualifications |
| `checkpoint_codex.json` | Complete final state, original baseline, time and configuration contract |
| `columns_codex.json` | All analysis-state labels |
| `timeseries_codex.csv` | Selected voltage/ATP/PCr/K time series |
| `figure_codex.png` | Soma voltage and process potassium for inspection |
| `runtime_reads_codex.json` | Packaged files read by the tested run |
| `trace_codex.npz` | All states for smoke; for other protocols only with `--full-trace` |

Long runs also save per-segment progress checkpoints in their new output directory. A partial progress checkpoint is not a completed 72-hour preparation and cannot be supplied to load36.

The compact CSV does not preserve every possible future analysis. Use `--full-trace` for experiments requiring full trajectory retention, then keep the large output outside ordinary Git. A checkpoint is sufficient for a compatible continuation, not for reconstructing the path that preceded it.

Solver: SciPy BDF, complete numerical Jacobian represented sparsely, with the inherited Numba kernels. Short tests use rtol 1e−8 / atol 1e−10; long protocols use the inherited per-state tolerance vector. Sparse storage is not removal of biological states. Source diagnostics can include additional solver points, so output sample counts need not equal duration/output_step.
