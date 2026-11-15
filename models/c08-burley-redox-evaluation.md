# C08 / Burley with redox supply evaluation

Use this 4,503-state configuration to study conditional changes in amino-acid delivery, glutathione (GSH) recycling, and explicit NADPH supply under the documented assumptions. The public runner selects the fixed configuration with `--redox-v15`, including the phase-tested Burley gates and 120 s acid removal. [Model manifest](c08-burley-redox.json) · [Phase parameter record](../data/2026-11-14-phase-aware-waveform/accepted.json) · [Redox parameters and evidence](../parameters/redox_assumptions.json)

## Numerical evidence

| Study | Evaluation evidence |
| --- | --- |
| [20 ms redox smoke and restart](../experiments/redox-smoke.json) | The checker compares all 4,503 states with the bundled reference, checks numerical balances, and requires a successful checkpoint restart check through 40 ms. [Reference trace](../results/redox_reference_smoke.npz) |
| Fresh Burley 25 pA redox pulse | The release verification recorded four spikes during the current step. The command is public and the release record contains the numerical checks. [Run instructions and limitations](../docs/REDOX_STAGE_0.7.0.md#running) · [Verification record](../provenance/redox_validation.json) |
| Historical 72-hour redox scenarios | The research runs started from the prepared Burley state at 600 h. That checkpoint is outside this checkout. The published outputs are available as summaries. [Checkpoint declaration](../data/2026-11-14-gap43-compensation/protocol.json) · [Scenario summaries](../results/redox_results.json) |

## Interpretation

In the historical middle-supply control, process GSH fell by about half without stimulation. Seven of nine reported branches missed the pre-declared resource screen. The model therefore has an unresolved baseline drift that matters when attributing changes to stimulation. The G6PD inhibition anchor comes from a purified human-enzyme assay. Calibration of redox supply in human motor neurons remains open, and glucose and other reducing substrates enter through open model boundaries. [Redox study and limitations](../docs/REDOX_STAGE_0.7.0.md) · [Scenario summary figure](../results/redox_summary.png)

[Model catalog](README.md) · [Experiment index](../experiments/README.md)
