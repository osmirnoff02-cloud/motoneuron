# Motoneuron model catalog

The catalog describes two configurations in release 0.7.0. Choose a model, then open an experiment for the exact stimulus, solver, command, and expected result.

| Model | Scope | Manifest | Evaluation |
| --- | --- | --- | --- |
| C08 / Burley | 4,425-state configuration for reference integration, checkpoint continuation, and somatic current-clamp experiments | [Model manifest](c08-burley.json) | [Evaluation and evidence](c08-burley-evaluation.md) |
| C08 / Burley with redox supply | 4,503-state configuration with acid removal, phase-tested Burley kinetics, and amino-acid, glutathione, and NADPH supply | [Model manifest](c08-burley-redox.json) | [Evaluation and evidence](c08-burley-redox-evaluation.md) |

Manifest file paths are relative to the repository root. An optional `json_pointer` selects a particular record within a JSON source. The model arguments identify the configuration in the public runner. The [experiment records](../experiments/README.md) specify the rest of each command. Use the generated `config.json` to inspect the resolved parameters for a specific run.

[Project README](../README.md) · [Model equations and compartments](../docs/MODEL.md) · [Parameter inventory](../parameters/README.md)
