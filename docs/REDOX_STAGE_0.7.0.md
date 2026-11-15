# 0.7.0: finite amino-acid/GSH recycling and NADPH supply (4503 states)

Claude, 2026-11-15; model stages by Codex, with standalone lipid, peroxide and NADPH blocks by Claude. This is an out-of-turn snapshot of the current working model published at the project owner's request. **No development-plan item is declared closed by this release**: N09.3.3, N11.2.3 and N04.2.3 remain open, and there is no human calibration.

## What changed since 0.6.0

The working model advanced through several connected stages; each keeps all earlier states and never resets a trajectory:

1. Bounded Selvaraj potassium split and a molecular-baseline audit.
2. Ribosomal-protein and GAP43 budgets with a prepared 600-hour molecular baseline.
3. Prepared 72-hour loads with GAP43 turnover scenarios and preserved electrical regressions.
4. GAP43 mRNA supply compensation; lipid peroxidation and peroxide blocks connected to the full model.
5. A shared antioxidant resource budget (GAP43 and redox draw on the same finite ATP and amino-acid pools).
6. Finite amino-acid delivery, GSH recycling into soma and process, finite GSH delivery, and an explicit NADPH supply that **replaces** the earlier NADPH source. The integrated model now has **4503 state variables** (4431 inherited + 72 declared at connection), including bookkeeping counters.

The extension is opt-in. Legacy defaults (4425 states) and the 0.3.0–0.6.0 scenarios (4431 states) are unchanged and remain available.

## Running

```powershell
python -B run_demo.py --protocol smoke --profile Burley --redox-v15 --check-restart --output outputs/redox-smoke
python -B run_demo.py --protocol pulse --profile Burley --current-pa 25 --redox-v15 --output outputs/redox-burley
```

`--redox-v15` is a fixed scenario: candidate C08, acid removal tau 120 s, the phase-tested kinetics accepted in 0.6.0, and the "middle" supply parameters. Do not combine it with `--phase-v14`, `--waveform-v14`, `--acid-tau-s` or a checkpoint. A 1.7-second pulse takes on the order of 15–20 minutes on one core. The 72-hour branches below started from a prepared 600-hour full-state checkpoint that is not distributed (size); they were run in the working project and are reported here as selected results, not re-run from this package.

## Selected results (working project)

Eight 72-hour scenarios plus one tighter-tolerance repeat, all continued from the 600-hour state without reset. Percentages are relative to the start of each trajectory, not human norms.

| Scenario | Process ATP minimum, % | Cys minimum, % | GSH at end, uM | GAP43 at end, % | Process K vs paired control, uM |
|---|---:|---:|---:|---:|---:|
| continuous_middle_72h | 99.93 | 60.72 | 1018 | 112.6 | -0.2411 |
| control_high_72h | 100 | 100 | 1001 | 112.5 | 0 |
| control_middle_72h | 100 | 60.72 | 1018 | 112.6 | 0 |
| control_soma_high_72h | 100 | 60.65 | 1780 | 112.8 | 0 |
| repeated_high_72h | 99.93 | 100 | 1001 | 112.5 | -0.2426 |
| repeated_middle_72h | 99.93 | 60.7 | 1018 | 112.6 | -0.2426 |
| repeated_middle_72h_tight | 99.93 | 60.7 | 1018 | 112.6 | -0.2426 |
| repeated_noPPP_72h | 99.93 | 60.7 | 1018 | 112.6 | no paired control |
| repeated_soma_high_72h | 99.93 | 60.62 | 1780 | 112.8 | -0.2421 |

Middle: external amino-acid input 0.1 uM/s, GSH recovery fraction 0.5, GSH exchange 3600 s. High: 0.3 uM/s, 0.9, 600 s. `soma_high` changes only somatic GSH synthesis capacity from 0.2 to 0.6 uM/s.

Checks in the working project: all 9 historical initial Burley/Selvaraj protocols (15/25/50 pA) kept their spike counts and voltage-shape tolerances; Burley 25 pA after 672 hours gives 4 spikes from the exact end state of the repeated middle load; baseline transfer, GSH/amino-acid material balances, recovery allocation, single ATP charging and replacement of the NADPH source were verified independently; tighter tolerances agreed within the stored limits.

## Limitations — read before using the numbers

- **The initial GSH pool is not stationary.** In `middle`, GSH falls by about half even without stimulation, and Cys drops to about 61% of its start. The decline is a property of the chosen baseline supply, not an effect of load. The pre-declared screen criterion (ATP at least 95%, selected amino acids/GTP at least 80%) is **not** met by seven of the nine branches.
- GAP43 rises almost identically with and without load; this shows compensation under the chosen supply, not a benefit of activity or protection against ALS.
- Only the G6PD Ki of 9 uM is a measured human-enzyme anchor (purified recombinant enzyme, 25 C, pH 8), not a motor-neuron measurement. All other new rates are scenario values; ranges are in [the assumptions file](../parameters/redox_assumptions.json).
- Glucose phosphorylation and the split of carbon between PPP, glycolysis and OXPHOS are not connected to this branch: G6P and other NADPH donors are open boundaries. Enzyme amounts are fixed and their production cost is not modelled.
- Amino-acid transport is an aggregate Cys:Glu:Gly = 1:1:1 packet, not identified transporters; the extracellular precursor pool size is an assumption.
- The package also contains later, **dormant** opt-in code that the current model interface imports: a standalone donor GLUT/HK–carbon module, a carbon-coupling adapter and a GSH-turnover observer. They are off by default, are not part of the validated 4503-state runs and are work in progress.
- Long AMPA protocols are average-opening schedules, not continuous action-potential firing. Historical spike counts are regression targets, not independent human measurements.

[Results](../results/redox_results.json) · [Verification](../results/redox_verification.json) · [Unit checks](../results/redox_unit_checks.json) · [Trajectories](../results/redox_trajectories.png) · [Assumptions and ranges](../parameters/redox_assumptions.json).

GLUT/HK code in `data/2026-11-15-donor-carbon` is adapted from Blue Brain Project/EPFL material under Apache-2.0; its [license](../data/2026-11-15-donor-carbon/LICENSE_BlueBrain.txt) and [notice](../data/2026-11-15-donor-carbon/NOTICE.txt) are included. The existing noncommercial research license and prior AGPL rights are retained. No correspondence, primary article downloads, personal data or private Git history are included.


## Independent package verification

Isolated Python environment with the private project/tool folders and the network blocked. The Burley 25 pA pulse run from this package reproduces every one of the 4503 states of the working-project reference bitwise (4 spikes). A 20 ms smoke run passes all balance checks and a checkpoint restart agrees with uninterrupted integration within tolerance. The 72-hour branches and the full 9-protocol electrical panel were checked in the working project and were not rerun from this export. [Evidence](../provenance/redox_validation.json).
