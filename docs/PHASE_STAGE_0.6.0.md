# 0.6.0: stimulus-offset checks and expanded kinetic search

Codex, 2026-11-14. Computational substage N04.1.3 with N04.2.3/N17.1.3 checks; the physiological model remains partial and conditional.

Twenty-eight complete 4431-state A14/W14 branches tested different stimulus endings, preserving every pool and history. The recovery observer distinguishes the last actual spike from the last spike inside the command. At Con2/50pA, W14 median recovery from the actual last spike is approximately497ms versus504ms previously: the earlier larger improvement measured from command end depended on spike timing. Burley shows stronger recovery improvement, with phase-dependent AHP tradeoffs. Duration changes also change cumulative load, so this is not a pure causal isolation of phase or one channel.

The joint search expanded from five to seven effective gate time parameters, adding tm and tn. There were 246 standard-duration evaluations across three profiles, followed by phase ranking and full-model checks. Conductances, steady-state gates, pump, ion budgets and geometry remained fixed. Multipliers relative to original A14 time parameters (in ms): tm 0.5–1.5; tn 0.25–2; thmin/threc/ta/tmfast/tndeact 0.6–1.4. These are computational search bounds, not human confidence intervals. The phase grid was used during selection, not held out as independent experimental validation. New profiles adopted after full checks: **Burley**. See the full acceptance results; rejected proposals are not silently adopted.

```powershell
python -B run_demo.py --protocol smoke --profile Selvaraj-Con2 --acid-tau-s 120 --phase-v14 --check-restart --output outputs/phase-smoke
python -B run_demo.py --protocol pulse --profile Burley --current-pa 25 --acid-tau-s 120 --phase-v14 --output outputs/phase-burley
python -B run_demo.py --protocol pulse --profile Selvaraj-Con2 --current-pa 50 --pulse-duration-s 0.85 --acid-tau-s 120 --phase-v14 --output outputs/phase-offset
```

Do not combine --phase-v14 and --waveform-v14 or reuse a checkpoint across parameter sets. All historical profiles remain available. Historical15/25/50pA spike counts are regression targets, not nine independent human measurements. Calibration to matched human bath/temperature, isolated currents, waveform and a healthy molecular baseline remains open. Long AMPA protocols are not continuous electrical firing. Full state arrays remain in the research archive; this package contains selected numerical results and runnable model code.

[Acceptance](../results/phase_acceptance_results.json) · [Offset checks](../results/phase_phase_summary.json) · [Selected parameters](../parameters/phase_accepted.json) · [Criteria](../results/phase_criteria.json) · [Plot](../results/phase_comparison.png).

Existing noncommercial research license and prior AGPL rights retained. No correspondence, primary article downloads, personal data or private Git history are included.

Burley, full phase comparison: median absolute voltage-deviation integral decreased by 10.40%; recovery from the last actual spike shortened by 4.94%. Phase acceptance: True. Individual AHP tradeoffs remain in the tables.

Burley, 72 h preparation followed by 36 h without reset: long acceptance True; failures []. These are average AMPA-opening protocols, not continuous action-potential firing. Rest 72 h process-K drift: 0.158723uM. repeated final process-K load-minus-control: -0.237903uM. continuous final process-K load-minus-control: -0.237086uM. 

[Full candidate phase comparison](../results/candidate_phase_Burley.png).


Independent package verification: isolated Python, blocked private project/tool roots and network. Con2 smoke/restart and Burley25 pulse reproduce every one of4431states bitwise. A direct0.35s Burley pulse agrees with the preserved-state offset branch within tolerance. Observer and parameter-input regressions pass. Long protocols were checked in the working model, not rerun in this independent export. [Evidence](../provenance/phase_validation.json).
