# Assumptions, parameter evidence and missing measurements

Codex, 2026-11-13. [Home](../README.md) · [Machine-readable registry](../parameters/registry.json).

The registry preserves original values, units, classifications, source-code locations and limitations for entries associated with the exported runtime. Entries can describe historical optimization bounds, scenario grids, fixed unknowns or source-specific observations. They are not all active in C08 and are not confidence intervals for human motor neurons. Current exact inputs are the candidate declaration plus the generated configuration/provenance.

| Uncertainty | Evidence distinction | Needed measurement |
|---|---|---|
| Absolute KCNQ conductance and replaced fraction of legacy slow-K | Scenario choices; a historical fitted K current is not a channel-specific measurement | Matched KCNQ subtraction/tail currents, capacitance and recording geometry |
| KCNQ activation/deactivation and G–V | Source-cell transfer and scenario voltage anchors; voltage-dependent tau is a hypothesis | Raw voltage-clamp traces at multiple voltages and temperatures |
| Passive Na/K/Cl fractions and initial chloride | Assumed initial partition; input conductance ratios in another species are not membrane-specific norms | Ion substitutions, near-rest I–V, intracellular Cl and cell volume |
| Na/K pump capacity, isoforms and ion dependence | Coupled initialization and scenario capacity, not an independently measured human value | Pump current, Na recovery, ATP dependence and membrane area |
| Accessible water volume | Explicit volume scenario; geometry is not automatically accessible cytosol | Compartment volumes, excluded volumes and concentration calibration |
| ATP production, creatine and energetic reserve | Biochemical source constraints plus unknown cell-specific capacities | Paired ATP/ADP/PCr or flux measurements under matched demand |
| PMCA/H/pH and acid extrusion | Diagnosed missing mechanism; charge equivalents do not equal free H concentration | Ca/pH/Na time courses, buffer capacity and pathway perturbations |
| RNA/protein/transport rates and pools | Many transferred or assumed quantities; RNA is not protein function | Absolute amounts, turnover, export/import rates and matched cell identities |

Record a new value with its units, source DOI/URL, species/cells, measurement or assumption status, correlated parameters, protocol and result affected, and a falsifiable test. Do not convert an unmeasured quantity to zero. Do not remove a mechanism solely because human data are missing. Fix numerical defects independently of biological uncertainty.

No personal contact list, correspondence status or promise of data access is included. Potential collaborators should be approached separately by the project owner. Implementation, coupling, numerical verification and human calibration are separate milestones.


0.6.0 supplement: [seven effective-gate search bounds and active factors](../parameters/phase_ranges.json). These supplement the historical inventory; they are not human calibration.


0.7.0 supplement: [amino-acid/GSH recycling and NADPH supply scenario ranges, active values and needed measurements](../parameters/redox_assumptions.json). Scenario grids, not measured human intervals; needed: same-culture labelled Cys/Glu/Gly and GSH pulse-chase, soma/axon GSH/GSSG, NADPH/NADP and glucose/PPP flux in healthy human iPSC motor neurons.
