# C08 / Burley evaluation

Use this 4,425-state configuration to inspect the integrated model's response to a short somatic current step and check numerical integration and checkpoint continuation. C08 is a search candidate. The Burley profile uses inherited electrical fits and reference recording assumptions. [Model manifest](c08-burley.json) · [C08 declaration](../data/2026-11-13-joint-kcnq-pump-v2/declared_candidates.json) · [Source-cell evidence](../data/2026-11-04-Q1-control-passports/cell_passports.json)

## Runnable references

| Experiment | Evaluation evidence |
| --- | --- |
| [20 ms smoke and restart](../experiments/default-smoke.json) | All states are compared with the bundled reference. The continuation from 20 ms is compared with uninterrupted integration through 40 ms. [Reference trace](../results/reference_smoke.npz) · [Export verification](../docs/VALIDATION.md) |
| [Burley 25 pA pulse](../experiments/burley-pulse-25pa.json) | The historical computational reference reports four spikes during the step. The checker compares peak soma voltage within 0.25 mV of the reference. The recorded reference run took about 11 minutes on the original validation workstation. [Reference metrics](../data/2026-11-13-joint-kcnq-pump-v2/electrical_summary.json) · [Reference trace plot](../results/verified_pulse_figure.png) |

## Interpretation

The pulse metrics track computational reproducibility. Independent comparison against matched human motor-neuron recordings remains open. The configuration leaves explicit acid extrusion disabled. Historical long-load studies report residual process potassium loss despite largely preserved ATP. Use the short references to assess numerical behavior over their documented intervals. [Model assumptions](../docs/ASSUMPTIONS.md) · [Historical long-load results](../data/2026-11-13-joint-kcnq-pump-v2/long_summary.json)

[Model catalog](README.md) · [Experiment index](../experiments/README.md)
