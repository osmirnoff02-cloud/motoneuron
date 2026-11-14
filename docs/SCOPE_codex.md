# Current 0.3.0 stage

[Acid stage and current limits](../docs/ACID_STAGE_0.3.0_codex.md). Earlier validation below is historical; it does not substitute for the new isolated package checks.

# Scope of the exported project

Codex, 2026-11-13. [Home](../README.md).

The supported executable interface is `run_demo_codex.py`: the integrated 4425-state ODE with explicit C08 (or another declared candidate) and documented smoke, pulse, rest72 and load36 protocols. The data directories retain actual import dependencies. Their historical scripts are not all standalone public applications: some expect omitted raw files or optional NEURON mechanisms. Use the supported wrapper, which protects packaged inputs from writes.

| Research direction | Scope in this package |
|---|---|
| TDP-43, RNA, translation, nuclear transport | Selected mechanistic/phenomenological modules in the integrated model; human rates and absolute pools remain uncertain |
| Calcium, KCNQ, Na/K/Cl, pump and energy | Main executable demonstration and current candidate comparisons |
| Terminal recycling and proteostasis | Conditional coupled modules and bookkeeping; not a validated full neuromuscular system |
| TMEM106B/lysosomes, iron/GPX4, microglia and membranes | Broader research directions; this snapshot does not claim complete validated implementations of each direction |
| Resistant/vulnerable, oculomotor and Onuf neurons | Requires comparable functional measurements; transcript differences alone are insufficient; no validated three-pool predictive model is provided |
| RNA-seq and proteomics reanalysis | Original raw files and full processing pipelines are not part of this first computational-model release |

The export is a fixed research snapshot, not the whole evolving project. Later results must be versioned separately. Long historical runs are supplied as selected summaries with source hashes; they were not all rerun during export. A fresh brief reproduction cannot retroactively validate all previous trajectories.

The optional legacy NEURON replay uses real NEURON and platform-compiled mechanisms. Those binaries and their full environment are not distributed here. Imports were deferred in two exported copies so the full SciPy/Numba model can run without them. No fake NEURON simulator or replacement numerical output is supplied.
