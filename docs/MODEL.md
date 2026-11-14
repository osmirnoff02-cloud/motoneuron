# Scientific and computational model

Codex, 2026-11-13. [Home](../README.md).

The model couples a soma, a process and selected terminal pools, with molecular and nuclear subsystems. A common state vector, original reference baseline and checkpoint carry the physical history. It is a compartmental conditional model, not an anatomical reconstruction of a patient's neuron.

## Conservation and coupling

The generic pool equation is d(amount)/dt = sum(inputs) − sum(outputs). Concentration changes require the appropriate accessible volume; equal numeric concentrations in different compartments need not be equal amounts. Electrical changes follow membrane current balance with the stored conductances, capacitances and driving forces. A positive outward current is not interchangeable with a positive inward concentration flux; source modules define these conventions.

The Na/K pump transports 3 Na outward and 2 K inward per ATP. The integrated host is the common owner of these ion and ATP fluxes. Adding a source assay or observation does not create a new membrane flux. ATP/ADP/AMP, creatine/phosphocreatine, guanylate pools, translation resources and RNA/protein inventories have explicit accounting at the implemented level of detail. Some counters and observed totals are projections of shared states, not additional physical molecules.

| Component | Primary source entry |
|---|---|
| Public facade | `data/neuron-model/model_api.py` |
| Composition and state ownership | `data/neuron-model/integrated_neuron.py` |
| State/checkpoint contract | `data/neuron-model/model_contracts.py` and `state_layout_v1.json` |
| Full integration and coupled right-hand side | `data/2026-10-28-takeuchi-balaji/adaptive_recycling_neuron.py` |
| Current candidate construction | `data/2026-11-13-joint-kcnq-pump-v2/options.py` |
| Candidate grid | `data/2026-11-13-joint-kcnq-pump-v2/declared_candidates.json` |

The implementation is the exact equation-level reference; this document summarizes its structure, not every thousands-state equation. State labels are returned in `columns.json` by the example. Analysis labels disambiguate RNA and protein; backend ordering and checkpoint schema are preserved.

## Units and protocols

Public example schedules use seconds; membrane voltage uses mV and applied electrical current pA. Conductance inputs include nS and ionic/molecular concentrations commonly use micromolar. Some historical isolated electrical routines use milliseconds and some molecular states have reference normalization. Never infer a unit merely from a numeric array. Each result includes the configuration and labels, and the registry explicitly marks unresolved units.

The current C08 scenario uses inherited reference-bath and temperature assumptions. The same name as a published cell line does not establish a fully matched recording protocol. In particular, nine historical spike-count targets are regression references; the Selvaraj 50-pA target is not a published experimental measurement.

## Scientific limits

No common healthy human steady state has been established. Some long protocols maintain ATP while retaining a potassium deficit. PMCA-associated proton equivalents were diagnosed, but explicit pH buffering/acid extrusion is incomplete in this snapshot. A finite numerical solution and small ledger residual cannot prove that this missing biological mechanism is represented. Human calibration, independent discrimination of hypotheses and prediction of interventions remain open.
