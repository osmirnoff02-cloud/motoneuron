# Help constrain Motoneuron with experimental data

**An invitation to researchers, laboratories and data custodians**  
Current release: **0.7.0** · The original 43 request groups come from the **2026-11-14** registry snapshot.

[Project and quick start](README.md) · [Assumptions](docs/ASSUMPTIONS.md) · [Machine-readable request catalogue](parameters/data_requests.json)

Motoneuron is a mechanistic research prototype for studying interactions among motor-neuron electrical activity, ion handling, energy supply and molecular maintenance. We invite you to help replace uncertain assumptions with traceable experimental constraints. A link to an existing dataset, a relevant supplementary table, a corrected interpretation or a negative result can be as valuable as a new experiment.

Some quantities are currently explored as ranges, alternative fixed values or effective parameters because suitable measurements are unavailable to this project or cannot yet be transferred reliably to the modeled cell and protocol. **These choices are not measured healthy-human reference intervals.** The model is not yet calibrated as a complete human motor neuron and does not establish treatment efficacy.

This document summarizes **43 request groups (U01–U43)** from our unknown-parameter registry. The accompanying catalogue preserves **1,044 numerical records**, including historical optimization bounds, protocol grids and variant values. Those records are not 1,044 independent missing biological constants. They span the wider research project, including branches and experiments outside this released runtime. A source-path match does not establish that a parameter is active in the selected example. The separately maintained [released-runtime inventory](parameters/README.md) has a narrower scope.

## Measurements relevant to release 0.7.0

Matched time series from human iPSC-derived motor neurons would help evaluate the opt-in redox scenario. If available, include unstimulated controls and measurements before, during, and after a defined stimulation protocol. The modeled GSH pool declines even without stimulation, so the controls can help assess baseline drift separately from the response to stimulation. [Redox scenario and limitations](docs/REDOX_STAGE_0.7.0.md)

- **Amino-acid supply and GSH turnover.** Track labelled cysteine, glutamate, and glycine uptake and their incorporation into glutathione in pulse-chase experiments to constrain precursor delivery and recycling.
- **Glutathione pools.** Measure reduced and oxidized glutathione (GSH and GSSG) in the soma and axon or neurites to constrain pool sizes and compartment-specific changes.
- **NADPH supply.** Pair NADPH and NADP⁺ measurements with glucose uptake and pentose phosphate pathway flux estimates from the same culture to constrain NADPH supply. [Redox assumptions and scenario ranges](parameters/redox_assumptions.json)

If you have one of these measurements, the [contribution guide below](#what-to-include-with-a-contribution) lists the context that will help interpret it.

## Where matched measurements would be especially useful

The examples below make several current assumptions concrete. They are entry points for collaboration, not a global ranking of biological importance. Parameters interact: a small effect in one sensitivity screen does not eliminate the need for measurements under other protocols.

| Quantity / related groups | Current exploratory choices | Measurements that could constrain them |
|---|---|---|
| KCNQ/M-current magnitude (U01, U12, U14) | 0, 0.5, 2 and 8 nS per modeled membrane in an initial screen; zero is a negative control | Matched control and XE991-sensitive current traces, membrane capacitance, series resistance, voltage protocol, temperature and cell maturity. Report both absolute current and normalization. |
| KCNQ voltage dependence (U01, U12) | Half-activation values −40, −27.3966 and −20 mV; slope 11.7788 mV in that screen | Conductance–voltage and tail-current curves with solutions, subunits and temperature. The center/slope reference comes from CHO measurements; the outer values are scenarios, not human bounds. |
| KCNQ kinetics (U01, U12) | A simplified constant time scale of 20, 80 or 300 ms | Voltage-dependent activation and deactivation measured separately, with uncertainty and the complete pulse sequence. |
| Passive Na/K/Cl allocation (U01, U14) | Na+K fractions 0.25, 0.5 or 0.75, remainder chloride; initial ionic balance and chloride reversal involve assumptions | Near-rest component I–V curves, ion-substitution experiments, intracellular chloride/reversal potential and morphology. Total leak alone does not separate the components. |
| Na/K-pump capacity (U14) | 0.5×, 1× or 2× a model baseline; 1× is not a measured healthy norm | Pump-specific current and sodium/potassium recovery paired with ATP, cell size, surface isoforms and controlled external ions. |
| Accessible neurite water volume (U01, U04) | Fractions 0.2, 0.5 or 0.8 for selected soluble pools; calcium has separate volume/buffer accounting | Three-dimensional morphology and organelle/accessible-volume estimates. Geometry changes must also constrain area, capacitance and exchange. |
| ATP production and creatine reserve (U05, U15) | ATP-production multipliers 0.5×, 1× or 2×; total creatine 2.5, 10 or 20 mM | Calibrated ATP/ADP/AMP and phosphocreatine/creatine, compartment volumes and load/recovery traces. Phosphocreatine is derived with the CK balance, not varied as an independent initial pool. |
| Acid removal and pH observation (U43) | Recovery time 20, 120 or 600 s; pH offsets 7.02/7.18/7.24 and buffers 10/20/46 mM per pH unit | Paired calcium export, sodium and calibrated pH, buffering capacity and inhibition/washout. The pH observer does not provide pH-dependent feedback to membrane channels. |

Later fitting bounds can differ from these initial-screen choices. The catalogue retains record IDs, exact values and source locations so that different variants are not silently merged. Measurements in other species or cell types are welcome as contextual evidence, with transfer limits retained.

## Complete request list

Each stable U identifier can be used in an issue, dataset description or correspondence. The numerical entries linked to a group are in [parameters/data_requests.json](parameters/data_requests.json), under `records` filtered by `question_id`. The catalogue presents units, source notes, and classifications in English. The descriptions below explain the requested measurements. Units inferred only from field names remain provisional. An unresolved unit is a request for clarification, not permission to assume one.

### U01 — Electrical properties and soma–neurite geometry

**Requested data:** Healthy human iPSC-derived motor-neuron reconstructions with lengths, diameters and accessible volumes; matched soma/axon voltage recordings, capacitance, conductance, axon initial segment properties, local sodium and response delays.

**Interpretation limit:** Historical fit bounds are not measured human physiological intervals. Nine firing conditions do not uniquely determine geometry.

### U02 — ADAR2, GluA2/AMPA and calcium buffering

**Requested data:** GluA2 Q/R editing counts, surface AMPA subunits and currents paired with calibrated calcium time courses, resting calcium, indicator loading and added buffer, stimulation and washout.

**Interpretation limit:** Calcium decay alone cannot separate buffer capacity from extrusion. Measurements from the same cell line are not necessarily paired measurements from the same cell.

### U03 — PMCA2 abundance, turnover and regulation

**Requested data:** Isoform-resolved active surface-pump counts or density, accessible volume, calcium/calmodulin and PIP2 conditions, PMCA-specific calcium flux and ATP cost during load, pause and repeat stimulation.

**Interpretation limit:** Density, active fraction and turnover can compensate for one another. Measurements in Jurkat, Sf9 or CHO cells are not human motor-neuron calibration.

### U04 — Spatial calcium/ATP exchange and pump delivery

**Requested data:** Paired soma/neurite calcium and ATP traces with compartment volumes; effective exchange; PMCA2-specific cargo numbers, transport speeds, pauses, insertion, retrieval and energy cost.

**Interpretation limit:** Lysosomal transport speed does not identify PMCA2 transport. Short and long neurites require separate geometry and exchange accounting.

### U05 — ATP supply and competing consumption

**Requested data:** ATP, ADP, AMP and free MgATP during rest, load and recovery; respiration, glycolysis, oxygen/glucose conditions and calcium-pump fluxes; protein maintenance and transport costs.

**Interpretation limit:** Stoichiometry does not determine total flux. An unknown cost is not zero; normalized respiration is not an absolute ATP-production flux.

### U06 — Calpain, calpastatin and substrate cleavage

**Requested data:** Paired calcium and CAPN1/2 activity time courses, calpastatin and autolysis, washout/rechallenge, intact TDP-43 and spectrin products by compartment, with synthesis/clearance controls.

**Interpretation limit:** Accumulated FRET cleavage is not instantaneous enzyme activity. Local substrate cleavage does not directly measure nuclear TDP-43 function.

### U07 — TDP-43 pools, synthesis, transport and aggregation

**Requested data:** Absolute nuclear/cytoplasmic TDP-43 pools, TARDBP RNA/NMD, independent synthesis and pulse-chase turnover, import/export, and soluble/insoluble forms with epitope and survival controls.

**Interpretation limit:** Exogenous aggregates, endogenous intact protein and cleavage fragments require distinct interpretation; cell division or selective survival can bias turnover estimates.

### U08 — Nuclear TDP-43, STMN2 and regrowth

**Requested data:** Matched nuclear TDP-43, canonical/cryptic STMN2 RNA, soma/axon protein, perturbation/rescue and regrowth measurements; independent neuromuscular-junction function.

**Interpretation limit:** Regrowth is not NMJ function. Mouse and primate STMN2 regulation cannot be treated as interchangeable.

### U09 — ER calcium, SOCE and calcium-activated potassium currents

**Requested data:** Matched ER/cytosolic calcium, voltage and ATP; ER volume/loading, SERCA, release/leak, store-operated currents and SK/BK recordings with calibrated indicators and buffers.

**Interpretation limit:** Relative fluorescence or caffeine responses alone do not determine absolute buffering or a universal SK activation threshold.

### U10 — Iron, ferritinophagy, lysosomes and GPX4

**Requested data:** Compartment-resolved total iron and labile Fe2+, ferritin iron/protein, NCOA4/HERC2 turnover, lysosomal pH/TRPML1, GPX4 activity, GSH/GSSG and lipid-oxidation time courses.

**Interpretation limit:** Total elemental iron is not the labile pool; protein abundance is not enzymatic activity. Preserve soma, axon and lysosome distinctions.

### U11 — PKN1 and secretory/Golgi proteostasis

**Requested data:** PKN1 activity and isoform-specific function, Rab1/ERGIC/Golgi transit times, cargo transport, autophagic flux, clearance and ATP cost under controlled perturbations.

**Interpretation limit:** Several couplings are hypotheses, including the possibility of no effect. Normalized gains are not measured kinetic rates.

### U12 — RNA-to-functional KCNQ2 and ATP2B2 protein

**Requested data:** Correctly spliced RNA, total and surface protein, M-current or PMCA flux, turnover, delivery pauses and local translation in matched samples, including rescue.

**Interpretation limit:** An RNA fold change cannot be substituted for a conductance or pump-flux fold change.

### U13 — Observers, protocols and unclassified numerical entries

**Requested data:** Help identify the biological quantity, units and calibration behind unresolved entries, and distinguish biological parameters from stimulation, solver, fitting and observation settings.

**Interpretation limit:** Not every number is a missing biological measurement. Protocol grids and numerical tolerances should not be calibrated as physiological ranges.

### U14 — Na/K pump, NCX and ionic cost of firing

**Requested data:** Matched intracellular sodium, extracellular potassium, voltage and ATP; pump current, surface alpha1/alpha3, beta and FXYD composition; sodium/calcium exchange and recovery after defined loads.

**Interpretation limit:** Isoform fractions and affinity/Hill parameters from other systems do not establish healthy human motor-neuron values.

### U15 — Mitochondrial calcium and energy coupling

**Requested data:** Calibrated cytosolic/matrix calcium, matrix volume and buffering, MCU influx/efflux, matrix sodium, membrane potential, ATP/ADP and respiration during load/recovery.

**Interpretation limit:** A dimensionless energy variable is not membrane potential. Transport capacity, kinetics and effective costs remain conditional.

### U16 — Absolute TDP-43/protein turnover and energy cost

**Requested data:** Compartment- and form-resolved protein amounts, proteasomal/lysosomal clearance fractions at defined MgATP/ADP/Pi, pulse-chase data, ubiquitin/chaperone overhead and substrate-specific ATP cost.

**Interpretation limit:** ATP costs for purified systems or other substrates cannot be assigned directly to human TDP-43 degradation; scenario costs are not measurements.

### U17 — Motor-terminal energetics and NMJ transmission

**Requested data:** Terminal action potentials/calcium, quantal release, endplate potentials, muscle action potentials, readily releasable pool, ATP, terminal volume, rundown/recovery and contraction.

**Interpretation limit:** Release or a passive endplate response is not muscle force; downstream functional coupling remains incomplete.

### U18 — Receptors, FXYD7 and CaV1.2 turnover

**Requested data:** Matched RNA and functional protein, GABA/NMDA currents and reversal potentials, calcium, FXYD occupancy and pump affinity; surface CaV1.2 insertion/removal pulse-chase and currents.

**Interpretation limit:** RNA abundance does not determine functional surface protein. Insertion, degradation and channel activity must be distinguished.

### U19 — Nuclear TDP-43, RNA processing/export and translation

**Requested data:** Volume-aware live nuclear/cytoplasmic import/export, RNA metabolic labeling/decay, calibrated RNA-seq and ribosome profiling, absolute protein synthesis/turnover in healthy motor neurons.

**Interpretation limit:** Elongation estimates from HEK cells and cycloheximide-based motor-neuron measurements have different applicability and biases.

### U20 — Stages of the ribosome cycle

**Requested data:** Absolute 40S/60S/80S occupancy, joining/recycling, eligible stalls, rescue times, active-ribosome abundance, protein flux and ATP/GTP use.

**Interpretation limit:** A mechanism established in a reconstituted system does not supply its rate in a human motor neuron.

### U21 — Ribosome biogenesis and nascent-chain energetics

**Requested data:** Active ribosome abundance and protein flux, rRNA pulse-chase/processing/export/turnover, stalled 80S/60S and nascent-chain pools, orphan clearance and energy costs.

**Interpretation limit:** New stage rates, capacities and effective energy costs are scenario parameters until independently constrained.

### U22 — Shared translation resources for ribosomal proteins, KCNQ2 and PKN1

**Requested data:** Absolute subunit, KCNQ2 and PKN1 abundance/synthesis/turnover, ribosomal-protein stoichiometry, assembly, amino-acid, tRNA and ATP/GTP use.

**Interpretation limit:** Known sequence lengths constrain material accounting but do not identify synthesis rates or competition between transcripts.

### U23 — Amino acids, tRNA charging and nucleotide energy

**Requested data:** Free amino acids, charged/uncharged tRNAs, ATP/ADP/GTP/GDP and synthesis during deprivation/recovery, plus amino-acid return from mature-protein turnover.

**Interpretation limit:** Balance-derived NDPK flux is not an independent measurement. Pooled substrates and charging parameters are approximations.

### U24 — Lysosomal amino-acid export and stress feedback

**Requested data:** Labeled protein turnover, cytosolic/lysosomal amino-acid pulse-chase, charged tRNAs, ATP/GTP, phospho-eIF2alpha, ATF4, GADD34, rRNA and protein synthesis through deprivation/recovery.

**Interpretation limit:** Normalized stress readouts are not absolute molecular pools; selected residues do not represent every amino acid.

### U25 — PERK versus GCN2, ATF4 and GADD34

**Requested data:** Matched ER stress versus amino-acid withdrawal, washout/rechallenge, phospho-PERK/GCN2/eIF2alpha, absolute ATF4/GADD34 RNA and translation, ATP, amino acids and charged tRNA, including GADD34 loss.

**Interpretation limit:** Human motor-neuron kinetic rates remain unresolved. Sequence length is not a kinetic measurement.

### U26 — Ribosomal-subunit maturation and delivery

**Requested data:** Soma/neurite precursor and mature-subunit pulse-chase, delivery/return, local synthesis, ribosomal-protein exchange and ATP/GTP costs.

**Interpretation limit:** Assembly of a new subunit and protein exchange within an existing mature particle are different processes.

### U27 — Local translation with finite resources

**Requested data:** Absolute local mRNA, translating ribosomes, completed proteins and turnover; amino acids, tRNA and ATP/GTP; local volume/exchange during inhibition, washout and repeat load.

**Interpretation limit:** Protein production, transport, local translation and degradation must be separated rather than inferred from one endpoint.

### U28 — GAP43 RNA delivery

**Requested data:** Free/bound GAP43 RNA in soma/axon, transit-time distributions, decay, local protein pulse-chase and ATP/GTP use.

**Interpretation limit:** RNA localization alone does not quantify local synthesis; transported protein and newly translated protein need independent labels.

### U29 — Local GAP43 function and axon regrowth

**Requested data:** Local RNA, nascent/total/phosphorylated GAP43, ATP, axon length and growth speed after injury, with knockdown and localized versus soma-retained rescue across donors.

**Interpretation limit:** A normalized growth readout is not physical growth, its ATP cost or NMJ repair.

### U30 — GAP43-dependent initiation versus extension

**Requested data:** Time-lapse initiation latency/frequency, elongation conditional on initiation, local RNA/protein/ATP and independent NMJ outcomes.

**Interpretation limit:** An endpoint population mean cannot identify initiation probability separately from subsequent elongation speed.

### U31 — Local ribosome-associated quality control

**Requested data:** Stalled 80S and 60S–nascent-chain complexes, free subunits, RNA fate, chain release/proteolysis, amino-acid return and ATP/GTP under stress and washout.

**Interpretation limit:** The selected eligible-stall pathway does not cover every no-go-decay, CAT-tail or tRNA-repair process.

### U32 — Ribosomal-protein exchange and ribosome retirement

**Requested data:** Compartment-resolved pulse-SILAC of free and ribosome-bound RPS26/other proteins, intact subunits, rRNA degradation/export and energy use during stress/washout.

**Interpretation limit:** Whole-cell protein turnover cannot identify exchange and whole-particle retirement separately.

### U33 — Ribosome maintenance, production and transport

**Requested data:** Matched pre-rRNA, mature/free subunits, free/bound new ribosomal proteins, transport, ATP/GTP and amino acids with baseline/stress/recovery labeling.

**Interpretation limit:** Demand-derived supply feedback is a modeling assumption, not a measured steady state.

### U34 — Precursor surveillance and sequence-specific protein costs

**Requested data:** Free/nuclear/assembled ribosomal-protein and pre-rRNA pulse-chase under assembly arrest/recovery; absolute ATP/GTP/amino acids and ubiquitin/FUBI coproduct fate.

**Interpretation limit:** Sequence lengths constrain accounting, but do not establish human assembly stoichiometry or effective surveillance costs.

### U35 — Precursor quality and selective removal

**Requested data:** Normal/aberrant precursor RNA, free/bound proteins, marking, clearance and mature output with ATP/amino-acid measurements; distinguish assembly delay from injury.

**Interpretation limit:** Recognition and damage rates are effective hypotheses, not measured human quality-classification rates.

### U36 — rRNA degradation and nucleotide salvage

**Requested data:** Isotope-labeled RNA fragments, nuclear/lysosomal nucleosides, cytosolic NMP/NTP, new RNA, ATP/ADP, volumes and medium loss; RNASET2, transporter and kinase perturbations.

**Interpretation limit:** Pooling nucleotide bases and routing nuclear NMP or lysosomal nucleosides are model assumptions requiring discrimination.

### U37 — GAP43 codons, collisions and tRNA repair

**Requested data:** Local amino acids/nucleotides and volumes, charged/modified tRNA, initiation/runoff, disomes, CCA/cyclic-phosphate repair, ELAC1/TRNT1/ANKZF1 perturbation and nascent protein.

**Interpretation limit:** The detailed branch is not a codon-level model of every protein; its structural approximations remain relevant.

### U38 — Shared codon-level resources

**Requested data:** Local amino acids and charged tRNAs, GAP43 polysomes/synthesis, ELAC1/TRNT1/RQC, ATP/GTP/CTP and exchange with the soma.

**Interpretation limit:** Mean-field dynamics are not a finite-copy stochastic simulation; other proteins remain pooled.

### U39 — Individual ribosomal-protein turnover and pool allocation

**Requested data:** Human motor-neuron peptide assignments, biological replicates, precursor enrichment, free/nucleolar/assembled protein pulse-chase, intact-subunit stoichiometry, RNA scaffolds and ATP/amino acids.

**Interpretation limit:** Rounded whole-cell turnover anchors for 75 proteins do not define free-pool rates; exceptions and missing measurements must remain explicit.

### U40 — TDP-43 nuclear transport and Ran energetics

**Requested data:** Matched nuclear/cytoplasmic volumes, free/bound TDP-43, KPNA/KPNB1/CAS/NTF2 pools, import/export, RanGTP/GDP, GTP/GDP/ATP/ADP and pore permeability under perturbation/recovery.

**Interpretation limit:** A selected-cargo transport model does not represent all nuclear traffic; absolute pools and rates need matched constraints.

### U41 — RNA, protein and preribosome nuclear transport

**Requested data:** Matched absolute pools, volumes, fraction yields/spike-ins, RNA metabolic labeling, export/import, ribosomal-protein receptors, XPO1 stoichiometry, ATP/GTP and Ran perturbations.

**Interpretation limit:** A 111-gene TPM inventory does not provide transport rates, and an initial pool does not establish steady state.

### U42 — Individual ribosomal-protein mRNA synthesis and decay

**Requested data:** Matched 4sU/SLAM/RATE data with biological replicates, molecules per cell, volumes/spike-ins, nuclear/cytoplasmic flow, decay, isoforms/readthrough and gene-specific production costs.

**Interpretation limit:** Neuron and K562 loss priors and relative production shapes do not set absolute human motor-neuron production or compartment-specific loss.

### U43 — PMCA proton equivalents, Na/H exchange and pH recovery

**Requested data:** Calcium export paired with calibrated pH and sodium during load/recovery, buffering capacity, Na/H inhibition, bicarbonate conditions, medium composition and temperature in the same healthy human motor neurons.

**Interpretation limit:** The optional acid-removal module is conditional. The 2 H/Ca reference is from snail neurons; recovery times are scenarios. pH/buffer settings describe an observer, not channel or RHS feedback.

## What to include with a contribution

Please share whatever is available; incomplete but well-described data can still be useful. For quantitative use, the following context is particularly helpful:

- **Material:** species, cell type/subtype, donor or line identifiers suitable for public release, genotype/control relationship, differentiation method, culture age/maturity and soma/axon/terminal compartment.
- **Conditions:** temperature, recording and culture solutions, pH, relevant ions/free magnesium, oxygen/glucose, buffers, drugs and concentrations, stimulation, washout and recovery timing.
- **Measurement:** quantity and units, absolute versus normalized signal, volume/area/capacitance used for normalization, instrument/indicator calibration, sampling interval and processing steps.
- **Replicates and uncertainty:** cells, donors and independent differentiations distinguished from technical repeats; individual observations where possible; what error bars represent (SD, SEM or confidence interval), exclusions and detection limits.
- **Provenance and permissions:** DOI/accession or dataset URL, version, figure/table or file/column identifiers, license or permitted uses, and preferred citation. Please identify whether observations were obtained in the same cells, matched cultures or unrelated experiments.

Time series and perturbation/recovery experiments can help distinguish mechanisms that fit the same endpoint. Null effects, inconsistent results and evidence against an assumed connection are welcome. There is no expectation that a contribution supports the model.

## How to share or suggest data

1. Open a [GitHub issue](https://github.com/osmirnoff02-cloud/motoneuron/issues/new) with a title such as **Data contribution — U14 — sodium/pump recovery**, or submit a pull request with a dataset description.
2. Include the U identifier(s), what was measured, species/cells, units, protocol, a DOI/accession or repository link, and the applicable reuse terms. A reference to an already published source is enough to start.
3. For unpublished or access-controlled material, first describe its availability and constraints. Use [email](mailto:osmirnoff02@gmail.com) to discuss an appropriate transfer route. Do not post identifiable patient data, private correspondence or files you are not authorized to share in a public issue.

Suggested issue text:

```text
Request group(s): U__
Dataset / publication / accession:
Measured quantity and units:
Species, cell type, maturity and compartment:
Protocol and calibration:
Replicates, uncertainty and limitations:
Available files or access procedure:
Reuse permissions and preferred citation:
Which model assumption this may constrain:
```

Contributors retain their rights. Sharing a link or starting a discussion does not transfer ownership or automatically relicense a dataset. We will record the supplied terms and agreed attribution; any integration or redistribution must respect them. Authorship of a future paper is a separate discussion based on actual contributions. The project's [software license](LICENSE.txt) does not replace a dataset's own license or consent/access conditions.

## How evidence will be assessed

We will distinguish receipt, review, implementation, numerical verification and human calibration. A newly received dataset does not automatically validate the model. We will check units, protocol compatibility, source provenance and parameter identifiability; preserve the original assumptions and reasons for revisions; and test affected outputs and interacting parameters. Results from other species or systems will remain labeled as such. Please report contradictory evidence or an incorrect interpretation as readily as missing values.

## Scope and provenance of this request

This is a curated English rendering of the project's parameter-request registry and assumption summary, not a new literature review or a claim that no such measurements exist anywhere. “Missing” means missing, unresolved or not yet applicable in this project's registry snapshot. The catalogue records source hashes and whether each referenced source was included in this public package. Unbundled paths are provenance identifiers, not download links; source code alone does not establish experimental evidence.

The released execution interface remains described in [the README](README.md). Source files now use names without author suffixes; original-to-public paths and hashes are retained in [the publication path map](provenance/publication_path_map.json). Development attribution and prior release history are preserved.

Prepared with Codex assistance on 2026-11-14 from existing project records. Scientific calibration remains open.


0.6.0 supplement: [seven effective-gate search bounds and active factors](parameters/phase_ranges.json). These supplement the historical inventory; they are not human calibration.


0.7.0 supplement: [amino-acid/GSH recycling and NADPH supply scenario ranges, active values and needed measurements](parameters/redox_assumptions.json). Scenario grids, not measured human intervals; needed: same-culture labelled Cys/Glu/Gly and GSH pulse-chase, soma/axon GSH/GSSG, NADPH/NADP and glucose/PPP flux in healthy human iPSC motor neurons.
