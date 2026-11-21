# Motoneuron

**Investigating motor-neuron dysfunction and testing possible mechanisms of correction.**

Motoneuron is a mechanistic computational model in Python developed in the context of amyotrophic lateral sclerosis (ALS) research. It connects electrical activity with ion handling, energy supply, and selected processes of RNA and protein maintenance.

Our goal is to investigate how a defined molecular disturbance—such as reduced protein function, altered RNA processing, or limited resource supply—can disrupt cellular function, and which interventions could restore function or compensate for the disturbance. A further goal is to connect these model-based interventions to experimental compound databases and prioritize molecules for laboratory testing.

**Status:** research prototype. The public 0.7.0 snapshot supports the documented simulation scenarios. Whole-model calibration against matched human motor-neuron data remains open. Automated compound selection is a planned capability, not a feature of this release.

[Model and equations](docs/MODEL.md) · [Runnable experiments](experiments/README.md) · [Experimental data requests](DATA_REQUESTS.md)

## Research workflow

The intended workflow links mechanistic experiments to candidate interventions:

1. **Define a disturbance** in a represented protein function or cellular process.
2. **Follow its consequences** over time against a matched control: what changes first, which other processes respond, and whether recovery occurs.
3. **Test corrective mechanisms** by restoring the affected function or modifying a represented compensatory pathway.
4. **Find candidate molecules** with relevant experimental evidence for their targets and direction of action.
5. **Evaluate their proposed effects** in the model, assess uncertainty, and prioritize experiments.

Steps 1–3 are possible for selected mechanisms and configurations. Steps 4–5 require further development: connecting compound data to the relevant model parameters and evaluating concentration-dependent effects. The project does not yet provide an automated pipeline from a compound database to a validated cellular response.

## Model architecture

**The electrical core uses Hodgkin–Huxley-type equations.** The soma and process are represented as coupled compartments, with additional subsystems for nuclear processes and selected terminal functions. Electrical parameters are partially constrained by selected published recordings from induced pluripotent stem cell-derived motor neurons (iPSC-MNs). Geometry and exchange laws are simplified and depend on the configuration.

| Layer | Represented processes |
| --- | --- |
| Electrical activity and ions | Membrane voltage, selected channel currents, Na/K/Cl pools, calcium handling, and ion transport |
| Energy | ATP/ADP/AMP, creatine/phosphocreatine, and production and consumption associated with implemented processes |
| TDP-43 and RNA | Selected relationships between nuclear TDP-43 function and transcript processing, including STMN2 and KCNQ2 branches |
| Cellular maintenance | Nuclear transport, RNA and protein synthesis and turnover, selected ribosome processes, and local translation |
| Optional antioxidant scenario | Peroxide and lipid processes, amino-acid supply, glutathione (GSH) recycling, and NADPH supply |

These layers share resources at the implemented level of detail. Pumps consume ATP, protein synthesis uses energy and amino acids, and changes in functional channel availability can alter electrical responses. Some relationships are mechanistic, others are effective or scenario-based; their evidence and assumptions must be interpreted separately.

## Perturbations and possible correction

Supported modules allow changes to selected activities, production rates, transport, or turnover. Examples include conditional TDP-43-dependent RNA effects and changes in KCNQ-associated current. STMN2-related outputs represent selected aspects of neurite support rather than a complete physical axon-growth model. Each experiment needs a compatible configuration and an explicit protocol; there is no universal command to knock out any protein.

We distinguish **loss of function**, **loss of production**, and **increased removal**. A protein that remains present but inactive is different from one whose synthesis stops while its existing pool gradually declines. Perturbations must respect the model's coupled states and material accounting.

Correction is assessed through multiple represented functions, such as electrical responses, ion recovery, energy availability, and molecular maintenance. A change in one favorable readout does not establish overall recovery. Missing functions or pathways limit the interpretation of both positive and negative results.

For future compound screening, databases such as [ChEMBL](https://www.ebi.ac.uk/chembl/) can supply bioactivity evidence. Each proposed compound-to-model link will need a target, direction of action, concentration–effect relationship, assay context, and uncertainty assessment. Binding affinity alone does not define a cellular effect. The intended output is a justified set of hypotheses and candidate molecules for experimental validation.

## Current configurations

| Configuration in 0.7.0 | Scope |
| --- | --- |
| C08 / Burley | 4,425-state reference configuration for integration, checkpoint continuation, and current-clamp experiments |
| C08 / Burley with redox supply | 4,503-state opt-in configuration with acid removal, selected Burley kinetics, and amino-acid, GSH, and NADPH supply |

C08 is a candidate from a parameter search. State counts include bookkeeping counters and do not measure biological completeness. The [model catalog](models/README.md) identifies the parameters and evaluation of each configuration. Later working branches are not automatically part of the public snapshot.

## Quick start

Use Python **3.12** and the pinned dependencies. Simulations run locally; the export validation was performed on Windows.

Clone the repository:

```bash
git clone https://github.com/osmirnoff02-cloud/motoneuron.git
cd motoneuron
```

Create and activate an environment on Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

From the repository root, install the dependencies and run the default check:

```bash
python -m pip install -r requirements-lock.txt
python -I -B -X utf8 verify_package.py
python -I -B -X utf8 run_demo.py --protocol smoke --profile Burley --candidate C08 --check-restart --output outputs/default-smoke
python -I -B -X utf8 tests/check_output.py outputs/default-smoke --experiment default-smoke
```

The smoke test integrates all 4,425 states for 20 ms, checks numerical balances, and compares checkpoint continuation with integration from the original baseline through 40 ms. Use a new output directory for each run. This is a reproducibility check, not a demonstration of long-term biological stability.

## Run an experiment

The [experiment index](experiments/README.md) supplies complete commands, initial conditions, expected results, and reproduction status for:

- the default smoke and checkpoint-restart check;
- a Burley 25 pA current step;
- the 4,503-state redox smoke and restart check.

It also distinguishes runnable long protocols from historical studies that require checkpoints outside the public checkout. Dated scripts under `data` are not all standalone applications; use the supported `run_demo.py` entry point and documented scenarios.

Each run saves its configuration, provenance, numerical checks, checkpoint, selected time series, and a figure. Smoke runs also save every state; add `--full-trace` for other protocols when needed. Report the version or commit, configuration, initial state, intervention, and solver settings when comparing results.

Long AMPA protocols use average receptor-opening schedules. Separate electrical probes after a load do not mean that every action potential was simulated continuously throughout the preceding hours or days.

[Extended protocols and outputs](docs/RUNNING.md) · [Validation record](docs/VALIDATION.md) · [Bundled results](results/README.md)

## Evidence and limitations

- **Implementation, numerical verification, and experimental validation are distinct.** Historical spike-count targets are regression references, not new independent human measurements.
- **Human calibration is incomplete.** Parameters combine measurements, fits, transfers from other systems, and explicit assumptions. Not all protein functions, metabolic pathways, cell interactions, or neuromuscular functions are represented.
- **A common healthy steady state has not been established.** In the published redox scenario, GSH drifts even without stimulation. Such changes must be compared with matched controls rather than attributed automatically to the intervention.
- **Conclusions depend on the represented mechanisms and parameter uncertainty.** An absent effect in the model does not establish that a protein is biologically unimportant.
- **The model supports hypothesis testing.** It does not yet establish the cause of ALS in an individual or predict the clinical efficacy or safety of an intervention.

[Model assumptions](docs/ASSUMPTIONS.md) · [Redox scenario and limitations](docs/REDOX_STAGE_0.7.0.md)

## Contribute or cite

We welcome independent reproduction, checks of equations and units, alternative mechanisms, and experimental data that can constrain unresolved assumptions. Null results and contradictory evidence are useful.

[DATA_REQUESTS.md](DATA_REQUESTS.md) describes 43 groups of open questions. Particularly useful contributions include matched human iPSC-MN measurements of electrical responses, ions, energy pools, protein abundance/localization, RNA processing, and changes after perturbation and recovery. Include units, cell maturity, recording conditions, calibration, replicates, uncertainty, and reuse terms. A link to an existing dataset is enough to start.

When citing Motoneuron, identify the version and, where relevant, the commit and configuration. See [citation metadata](CITATION.cff), [contribution guidance](CONTRIBUTING.md), and [issue templates](.github/ISSUE_TEMPLATE/README.md).

## Explore the repository

| Path | Contents |
| --- | --- |
| [models](models) | Supported configurations, parameter sources, and evaluations |
| [experiments](experiments) | Protocols, commands, reference checks, and reproduction status |
| [run_demo.py](run_demo.py) | Supported simulation entry point |
| [data/neuron-model](data/neuron-model) | Integrated interface and state layout |
| [data](data) | Dated components and source-linked inputs |
| [parameters](parameters) | Parameter inventories, ranges, and scenario settings |
| [docs](docs) | Model documentation, assumptions, and validation |
| [provenance](provenance) | Source records, checksums, and release verification |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## Mission

The project aims to keep published model versions accessible for research into ALS and related neurological diseases, and to support collaboration on prevention and treatment research. [Original mission statement](provenance/MISSION.md).

## License

The **Motoneuron Noncommercial Research License 1.0** permits noncommercial scientific research. Commercial use of material offered only under these terms requires a separate written agreement. Version 0.1.2 remains available under AGPL-3.0-only, and separately licensed third-party material retains its terms.

[License](LICENSE.txt) · [Commercial enquiries](COMMERCIAL_LICENSE.md) · [License scope](docs/LICENSE_SCOPE.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)
