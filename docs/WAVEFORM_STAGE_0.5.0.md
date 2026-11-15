# 0.5.0: partial waveform and post-discharge recovery refinement

Codex, 2026-11-14. Completed computational substage N04.1.3 with N04.2.3/N17.1.3 regression checks. Parent physiological tasks remain partial. Conditional research prototype, not a calibrated healthy human neuron.

Run the existing installation instructions and choose the optional scenario explicitly:

```powershell
python -B run_demo.py --protocol smoke --profile Selvaraj-Con2 --acid-tau-s 120 --waveform-v14 --check-restart --output outputs/wave-smoke
python -B run_demo.py --protocol pulse --profile Burley --current-pa 25 --acid-tau-s 120 --waveform-v14 --output outputs/wave-pulse
python -B run_demo.py --protocol rest72 --profile Selvaraj-Con2 --acid-tau-s 120 --waveform-v14 --output outputs/wave-rest
python -B run_demo.py --protocol load36 --profile Selvaraj-Con2 --acid-tau-s 120 --waveform-v14 --branch repeated --checkpoint outputs/wave-rest/checkpoint.json --output outputs/wave-load
```

Also run branch control from the same complete rest checkpoint. Source inputs remain read-only. Legacy profiles stay available without --waveform-v14. Do not combine a checkpoint with different kinetic parameters.

Five effective HH gate time parameters were jointly varied within 0.6–1.4 of the old values. Conductances, steady-state gates, pump, KCNQ, acid balance and geometry were fixed. Con1 was retained because no meaningful constrained improvement was found. These factors are not measured isolated human channel rates or confidence intervals.

The full 4431-state model retains the initial counts Burley 3/4/3, Con1 5/6/4, Con2 5/6/7 at 15/25/50 pA. These are historical regressions, not nine independent human observations; Selvaraj 50 pA is not a published experimental target. Waveform/AHP improvement is partial: Con2 25-pA AHP is slightly deeper. See all before/after metrics in [results](../results/waveform_summary.json).

Fresh 72-hour preparation plus paired 36-hour continuous/repeated/control protocols retain all states. Process K residual is -1.219738 uM after repeated load and -1.220252 uM after continuous load versus matched control. Post108-hour Con2 counts remain 5/6/7 in both control and repeated-load branches. Long runs cover Con2 only and are saved every300 seconds. ATP minimum 99.97251%; PCr 98.71961% of the repeated-load starting level.

Historical 23°C / external Na145mM conditions remain; publication baths, absolute waveform/Na currents, molecular stationarity and human calibration are still open. Burley50 rest-referenced half-height width decreased, Con2 15/25 widths increased slightly, and Burley spike height increased. The main gain is post-discharge recovery, not a demonstrated global waveform fit. Long potassium recovery does not establish these. The approximate search contains a frozen-molecular correction, but that correction is absent from the full accepted simulations and public model. No ion resets or extra restoring fluxes were added.

[Acceptance](../results/waveform_acceptance_results.json) · [Waveform metrics figure](../results/waveform_comparison.png) · [Potassium and energy figure](../results/potassium_energy.png) · [Numerical refinement](../results/waveform_refinement.json) · [Parameters](../parameters/waveform_selected.json) · [Ranges](../parameters/waveform_ranges.json).

Existing noncommercial research license and prior AGPL rights retained. No primary downloaded data, private Git history, correspondence or environments included.


Independent Windows validation of the renamed public package reproduced all 4431 states of Con2 smoke/restart and Burley25 within 1e-8 absolute difference, with research folders, E:/als-tools and network access blocked. The exported long runs were not rerun. [Proof](../provenance/waveform_validation.json).
