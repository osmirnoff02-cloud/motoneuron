# Selected results and their status

Codex, 2026-11-13. [Home](../README.md).

## Fresh export checks

[Validation report](../docs/VALIDATION.md) describes what was actually rerun using this export. `reference_smoke.npz` is the small full-state reference generated from the original project before packaging, with identical scenario and solver settings; `reference_smoke_summary.json` preserves its checks. See source hashes in provenance. This reference is for computational equivalence, not biological validation.

## Historical candidate evidence

- [Electrical summary](../data/2026-11-13-joint-kcnq-pump-v2/electrical_summary.json): original completed electrical rows, including their protocol time origin and regression targets.
- [Long-run summary](../data/2026-11-13-joint-kcnq-pump-v2/long_summary.json): selected historical controls/loads and residual potassium effects. Full trajectories are not bundled.
- [Charge diagnosis](../data/2026-11-13-joint-kcnq-pump-v2/charge_diagnostic.json): diagnostic PMCA/proton-equivalent accounting; no new pH state or acid-extrusion mechanism.

C08 initially matched nine historical spike-count targets. This does not establish independent human waveform accuracy. Strong repeated stimulation retained a process potassium deficit around 1.2 mM relative to control in the recorded conditional protocol, despite preserved energetic measures. The candidate was not accepted as a healthy human default.

These are dated historical results. Their original row-level evidence is included with the result bundle where available. They are not represented as fresh full-panel/72-hour tests of this export. Negative findings and limitations must accompany positive numerical checks.

## Release figures

The English figures use the bundled data for the corresponding model runs.

- [Stimulus offset and recovery](phase_comparison.png) uses the [phase comparison rows](phase_phase_summary.json). [Burley candidate acceptance](candidate_phase_Burley.png) uses the [paired candidate results](phase_candidate_phase_acceptance.json).
- [Waveform refinement](waveform_comparison.png) shows changes in recovery metrics from the [acceptance results](waveform_acceptance_results.json). [Potassium and energy](potassium_energy.png) plots endpoints and pool minima from the [long-run summary](waveform_summary.json).
- [Redox supply scenarios](redox_summary.png) shows initial, minimum, and final values where applicable, drawn from the [72-hour scenario summaries](redox_results.json).

<!-- publication-file-index_codex -->

## Files

- [reference_smoke.npz](reference_smoke.npz)
- [reference_smoke_summary.json](reference_smoke_summary.json)
- [verified_pulse_figure.png](verified_pulse_figure.png)
- [verified_pulse_summary.json](verified_pulse_summary.json)
- [verified_pulse_timeseries.csv](verified_pulse_timeseries.csv)


[Current waveform stage](../docs/WAVEFORM_STAGE_0.5.0.md).
