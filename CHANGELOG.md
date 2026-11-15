# Unreleased — changes on main after 0.7.0

These changes are on `main` and are not part of the `v0.7.0` tag. Model equations, parameter values and the existing numerical references are unchanged.

- Licence notice: the copyright line naming individuals was removed from `LICENSE.txt` and the Licensors are now defined without names. The terms are otherwise unchanged and earlier releases keep their published text. [Record](provenance/license_notice_2026-11-15.json)
- The repository is English-only. Parameter registry labels were translated; no numeric value changed.
- Added [experiment records](experiments/README.md), a [model catalog](models/README.md), a redox smoke reference and a verification workflow that runs both smoke experiments on Linux, macOS and Windows.
- `run_demo.py` accepts only declared candidate names such as `C08`; forms such as `C8` or `C-1` are rejected before an output directory is created. `summary.json` gains `checkpoint_used` and the restart tolerances that were applied.
- `tests/check_output.py` takes `--experiment ID`, compares the initial baseline within a tolerance instead of bitwise, and requires the restart tolerances used by a run to equal the experiment record.
- Experiment records and model manifests name the runtime state-schema hash `state_schema_sha256` (previously `state_layout_sha256`); it is not the hash of `state_layout_v1.json`. `tests/check_catalog.py` checks that the catalog and the records agree.
- The mission statement and publication checklist moved to `provenance/`.

# 0.7.0 — 2026-11-15

Opt-in 4503-state branch: GAP43/ribosomal budgets, lipid and peroxide blocks, shared antioxidant budget, finite amino-acid/GSH recycling and explicit NADPH supply; new `--redox-v15` run option. Out-of-turn snapshot; open items and limits are listed in the [details](docs/REDOX_STAGE_0.7.0.md). Earlier scenarios, license terms and prior AGPL rights unchanged.

# 0.6.0 — phase-aware recovery

Added full stimulus-offset comparisons, corrected actual-last-spike recovery, and expanded kinetic search with full acceptance. [Details](docs/PHASE_STAGE_0.6.0.md).

# 0.5.0 - 2026-11-14

Optional effective-gate waveform refinement, preserved historical counts and paired long potassium recovery. [Validation and limits](docs/WAVEFORM_STAGE_0.5.0.md). The 0.4.0 data-request work below is retained.

# 0.4.0 — 2026-11-14

- Add DATA_REQUESTS.md beside README, with 43 English experimental-data requests and a sanitized 1,044-record numeric catalogue. Link it from both GitHub's displayed README and the root README.
- Remove _codex and _claude suffixes from public filenames; update imports, commands and references. Entry point is now run_demo.py; generated output filenames also omit suffixes. Original paths and hashes remain in provenance. This is a filename/interface migration, not a change to model equations or parameter values.
- Retain the optional acid-removal scenario from 0.3.0. Human calibration remains open.
- Correct inherited homepage wording to retain the 0.2.0 license boundary; license terms and prior AGPL rights are unchanged.
- Validation details are recorded in docs/DATA_REQUESTS_RELEASE.md.

# 0.3.0 — 2026-11-14

Optional PMCA proton/Na-H exchange closure, full-state restart, historical electrical panel and long conditional recovery results. [Details](docs/ACID_STAGE_0.3.0.md). Existing license and earlier AGPL rights preserved; no human calibration claim.

# 0.2.0 — 2026-11-14

- Adopted Motoneuron Noncommercial Research License 1.0 for releases carrying these terms: noncommercial scientific research is free; commercial use requires a separate agreement.
- Preserved 0.1.2 and previously granted AGPL rights. Unchanged code remains available from that release under AGPL, including compliant commercial use.
- Updated bilingual guidance, mission interpretation, contribution policy and citation license URL; no model code or numerical input changes.
- Restored the Apache-2.0 license and attribution notice referenced by the Blue Brain/EPFL CK adaptation but missing from the earlier package.
- Numerical tests were not rerun for this documentation/notice-only change. Package hashes, links, citation fields and retained executable bytes were checked.


# 0.1.2 — 2026-11-14

- Prepared the owner-authorized public snapshot for `osmirnoff02-cloud/motoneuron`.
- Added repository and download instructions, project-level citation and current publication guidance.
- Retained the tested executable snapshot, inputs, numerical references and AGPL-3.0-only license unchanged.
- Regenerated integrity hashes and a separate ZIP; earlier archives retained.


# 0.1.1-prepublication — 2026-11-14

- Applied owner-selected GNU AGPL version 3 only, using the unchanged official license text.
- Added the owner-approved mission; it is not an additional license condition.
- Updated citation metadata, contribution/release instructions and licensing scope; personal attribution remains provisional.
- Model source, inputs, parameters and numerical references are byte-identical to 0.1.0-prepublication. Numerical tests were not rerun for this documentation/license-only update.
- Regenerated package hashes and verified the new ZIP. The previous ZIP is retained; no publication or deletion.


# 0.1.0-prepublication — 2026-11-13

- Owner-selected project title: Motoneuron.
- Selected full integrated-model dependency closure and small runtime data exported with SHA-256 provenance.
- Added a portable read-only wrapper, short/full electrical examples, long-protocol commands and output checks.
- Deferred optional legacy NEURON initialization in two exported files; scientific equations and parameter values retained.
- Added project documentation, parameter inventory without correspondence fields, selected historical evidence and explicit limitations.
- Prepared an isolated-environment reproduction check. See docs/VALIDATION.md for the observed result and its limits.
- No publication, deletion of originals, transfer of private Git history or automatic license grant.
