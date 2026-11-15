# Motoneuron experiments

Install the [pinned Python 3.12 environment](../README.md#quick-start) and run commands from the repository root. Each run needs a new output directory. Experiment records follow the [MIASE reporting checklist](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1001122) and specify the model, initialization, stimulus, solver, outputs, reference checks, and interpretation limits.

| Experiment | What the check establishes | Record |
| --- | --- | --- |
| Default smoke | 4,425 states advance for 20 ms. A checkpoint continuation agrees with integration from the original baseline through 40 ms. | [Default smoke](default-smoke.json) |
| Burley 25 pA pulse | A fresh C08 run produces the historical target of four spikes during the step and a peak voltage within 0.25 mV of the reference. | [Burley 25 pA](burley-pulse-25pa.json) |
| Redox smoke | 4,503 states, including the amino-acid, glutathione, and NADPH extension, pass the 20 ms integration and 40 ms restart checks. | [Redox smoke](redox-smoke.json) |

The [model catalog](../models/README.md) links each configuration to its parameters and evaluation. The default smoke and Burley pulse use [C08 / Burley](../models/c08-burley.json). The redox smoke uses [C08 / Burley with redox supply](../models/c08-burley-redox.json).

## Commands

```bash
python -I -B -X utf8 run_demo.py --protocol smoke --profile Burley --candidate C08 --check-restart --output outputs/default-smoke
python -I -B -X utf8 tests/check_output.py outputs/default-smoke --experiment default-smoke

python -I -B -X utf8 run_demo.py --protocol pulse --profile Burley --candidate C08 --current-pa 25 --output outputs/burley-pulse-25pa
python -I -B -X utf8 tests/check_output.py outputs/burley-pulse-25pa --experiment burley-pulse-25pa

python -I -B -X utf8 run_demo.py --protocol smoke --profile Burley --candidate C08 --redox-v15 --check-restart --output outputs/redox-smoke
python -I -B -X utf8 tests/check_output.py outputs/redox-smoke --experiment redox-smoke
```

Smoke checks compare every state at every sample when the solver time grids match, or every state at the start and end when the grids differ. Initial reference comparisons allow for platform roundoff. The saved checkpoint must agree exactly with the matching arrays from the same run. The pulse verifies selected summary metrics. These checks establish computational reproducibility for the listed scenarios. Human calibration and long-term stability require separate evidence. [Output formats](../docs/RUNNING.md#files-written) · [Model and evidence](../docs/MODEL.md)

## Other protocols and historical results

| Study | Reproduction status |
| --- | --- |
| Legacy 72-hour preparation and 36-hour AMPA loads | Commands are public. Generate a matching 72-hour checkpoint and run each load from that same state. These long runs are outside routine checks. [Instructions](../docs/RUNNING.md#long-protocols) |
| Waveform and stimulus-offset studies | Short runs are available via the documented flags. The published acceptance panels include historical results and full trajectories held in the research archive. [Waveform refinement](../docs/WAVEFORM_STAGE_0.5.0.md) · [Stimulus-offset checks](../docs/PHASE_STAGE_0.6.0.md) |
| Redox 72-hour studies | Published summaries are available. Reproduction starts from the inherited Burley preparation at 600 h, `data/2026-11-14-ribosome-gap43-balance/RGB14_rest_Burley_312to600h_checkpoint.json`, before state extension by the shared-redox migration. The checkpoint is outside the public checkout. [Checkpoint declaration](../data/2026-11-14-gap43-compensation/protocol.json) · [Redox study](../docs/REDOX_STAGE_0.7.0.md) |
