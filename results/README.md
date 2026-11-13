# Selected results and their status

Codex, 2026-11-13. [Home](../README.md).

## Fresh export checks

[Validation report](../docs/VALIDATION_codex.md) describes what was actually rerun using this export. `reference_smoke_codex.npz` is the small full-state reference generated from the original project before packaging, with identical scenario and solver settings; `reference_smoke_summary_codex.json` preserves its checks. See source hashes in provenance. This reference is for computational equivalence, not biological validation.

## Historical candidate evidence

- [Electrical summary](../data/2026-11-13-joint-kcnq-pump-v2/electrical_summary_codex.json): original completed electrical rows, including their protocol time origin and regression targets.
- [Long-run summary](../data/2026-11-13-joint-kcnq-pump-v2/long_summary_codex.json): selected historical controls/loads and residual potassium effects. Full trajectories are not bundled.
- [Charge diagnosis](../data/2026-11-13-joint-kcnq-pump-v2/charge_diagnostic_codex.json): diagnostic PMCA/proton-equivalent accounting; no new pH state or acid-extrusion mechanism.

C08 initially matched nine historical spike-count targets. This does not establish independent human waveform accuracy. Strong repeated stimulation retained a process potassium deficit around 1.2 mM relative to control in the recorded conditional protocol, despite preserved energetic measures. The candidate was not accepted as a healthy human default.

These are dated historical results. Their original row-level evidence is included with the result bundle where available. They are not represented as fresh full-panel/72-hour tests of this export. Negative findings and limitations must accompany positive numerical checks.

<!-- publication-file-index_codex -->

## Files

- [reference_smoke_codex.npz](reference_smoke_codex.npz)
- [reference_smoke_summary_codex.json](reference_smoke_summary_codex.json)
- [verified_pulse_figure_codex.png](verified_pulse_figure_codex.png)
- [verified_pulse_summary_codex.json](verified_pulse_summary_codex.json)
- [verified_pulse_timeseries_codex.csv](verified_pulse_timeseries_codex.csv)
