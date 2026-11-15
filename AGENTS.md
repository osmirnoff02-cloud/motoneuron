# Agent instructions

These rules apply to every coding agent working in this repository, including Claude and Codex. Read them before changing anything.

Motoneuron is a published research model of a motor neuron in the context of ALS. Its value depends on three things staying true: results are reproducible byte for byte, historical records are never silently altered, and no text claims more than the evidence supports. Most pitfalls below are ways of breaking one of those.

## Before you start

- Another agent may be editing this working tree at the same time. Run `git status` first and treat uncommitted changes you did not make as someone else's work in progress.
- Never revert, stash, reset, or check out over changes you did not make. Stage exact paths; never use `git add -A` or `git add .`.
- If a file you need is being changed by someone else, say so and ask rather than overwriting it.
- Do not commit, push, tag, or publish a release unless the owner asks. The repository is public.
- Never rewrite Git history. The licence boundary refers to the 0.1.2 commit by hash.

Before editing, read the guides for the area you’re changing. Start with the relevant row below and read the local README if there is one.

| Area | Read first |
| --- | --- |
| Model equations, parameters, or state layout | [Model](docs/MODEL.md), [Assumptions](docs/ASSUMPTIONS.md), and the relevant dated stage note |
| Runner, protocols, or experiment definitions | [Running guide](docs/RUNNING.md) and [Experiment index](experiments/README.md) |
| Package integrity, release records, or archived results | [Provenance](provenance/README.md) and [Validation](docs/VALIDATION.md) |

If a guide, the implementation, and a reference result disagree, identify the discrepancy before relying on any of them for the change.

## Running the model

- Use Python 3.12 and `requirements-lock.txt` only. Do not upgrade, add, or unpin dependencies; numerical references depend on the exact versions.
- `run_demo.py` is the only supported entry point. Always invoke it as `python -I -B -X utf8 run_demo.py ...` from the repository root.
- Do not run scripts inside `data/` directly. Many are historical: they expect files that are not in the package, and some write result files next to themselves. The two phase unit tests named in the CI workflow are the exception.
- `--output` must be a directory that does not exist yet, below `outputs/` or outside the repository. Reusing a name fails. Never delete an earlier output to free the name; pick a new one.
- Know the cost before launching a run:

| Run | Approximate cost |
| --- | --- |
| Smoke (20 ms) with restart check | A few minutes, longer on a cold Numba cache |
| Burley 25 pA pulse | About 11 minutes and several GB of RAM |
| Redox pulse | 15 to 20 minutes on one core |
| `rest72`, `load36` | Hours or longer; never start these unless asked |

- Runs are deliberately single-threaded. Do not raise thread limits or parallelise the solver to save time.
- The 72-hour redox studies cannot be reproduced from this checkout because their prepared checkpoint is not distributed. Do not try to substitute another checkpoint.
- NEURON is optional and not installed. Never emulate it or fabricate its output.

Exact commands and expected results are in the [experiment index](experiments/README.md) and [running guide](docs/RUNNING.md).

## Verification

Run the checks that cover the change. If several rows apply, run the checks from each. See the [CI workflow](.github/workflows/verify.yml) for the automated checks and the [experiment index](experiments/README.md) for exact commands and expected results.

| What changed | What to run |
| --- | --- |
| Any packaged file, including documentation | Update the affected manifest entries, then run `python -I -B -X utf8 verify_package.py`. |
| Anything under `models/` or `experiments/` | Also run `python -I -B -X utf8 tests/check_catalog.py` to check that the catalogs agree. |
| Runtime code, model inputs, or numerical behavior | Run the applicable smoke scenario with `--check-restart`, then check its output with `tests/check_output.py OUTPUT --experiment ID`. If the change concerns behavior the smoke run does not exercise, use the relevant documented regression and follow the run limits above. |
| Argument validation, checkpoint acceptance, or output-path guards | Also run `tests/check_rejected_inputs.py OUTPUT` using the default smoke output. |
| Phase kinetics or the recovery observer | Also run the applicable focused test listed in the CI workflow. |

Rules that follow from this:

- A passing smoke test shows the integration is numerically reproducible over 20 ms. It says nothing about biology or long-term stability, so never report it as validation of the model.
- Never loosen a tolerance in `experiments/*.json`, `run_demo.py`, or `tests/` to make a check pass. Find the cause.
- Never regenerate a reference file in `results/` to match new output. A reference changes only when the owner decides the model changed, and the record must state the commit, platform, and reason.
- Small floating-point differences between platforms are expected. Compare with the recorded tolerances, never with bitwise equality across platforms.
- Report what you actually ran. If you skipped the pulse because it is slow, say so.

## The package manifest

`provenance/package_manifest.json` lists every file in the repository except itself, with its byte size and SHA-256. `verify_package.py` fails on any mismatch.

- Any edit to any listed file, including documentation, breaks verification until its manifest entry is updated. Update the entry in the same change.
- Add an entry for every new file and remove the entry for every deleted one. Entries are sorted by path, ignoring case.
- There is no generator script in the repository. Recompute `bytes` and `sha256` from the file on disk, and touch only the entries for files you changed.
- If another agent has uncommitted manifest changes, commit only your own entries and leave theirs in the working tree.
- Never edit a hash to hide a change you cannot explain. An unexpected mismatch means a file changed that should not have.
- The manifest uses CRLF line endings and two-space indentation. Keep both.

`verify_package.py` also rejects broken relative Markdown links, absolute Windows user-profile paths, access tokens, private keys, and files over 100 MB. Keep local paths and credentials out of every file.

## Bytes are part of the record

- `.gitattributes` sets `* -text`, so Git stores files exactly as written. About half the files use CRLF and the rest LF. Keep each file's existing line endings and encoding.
- Do not run formatters, linters with autofix, or import sorters over inherited code. The dense style in `data/` and `run_demo.py` is inherited, and exported function bodies were compared against the original source.
- Do not make cosmetic edits to files you are not otherwise changing. Every changed byte needs a manifest update and a reason.

## Model code

- The dated directories under `data/` are import dependencies, not an archive. Modules load each other through `sys.path` insertion and path-based imports with short names such as `full`, `capture`, `options`, and `run`. Do not rename, move, merge, or delete these directories or files, and do not convert them into a package.
- Add new work in a new dated directory rather than editing an earlier stage.
- State order and the checkpoint schema are fixed. Code reads states by number; index 185 is soma voltage. Append new states at the end and never reorder or remove existing ones.
- State counts identify the configuration: 4,425 by default, 4,431 with acid extrusion, 4,503 with the redox scenario. A different count means you changed the model.
- Never change a historical default or an existing parameter value. A new mechanism or value must be an explicit opt-in flag that leaves earlier scenarios bit-for-bit unchanged.
- `run_demo.py` rejects untested flag combinations on purpose. Do not relax those guards to make a command work.
- Never edit a checkpoint or its fingerprints so that it will be accepted. `load36` needs a completed `rest72` checkpoint with matching candidate, profile, schema, and configuration.
- Do not enable the dormant modules for donor carbon, carbon coupling, and GSH turnover. They are imported but off by default and are not part of any validated run.
- Do not turn an unmeasured quantity into zero, and do not remove a mechanism because human data are missing.
- Fix numerical defects separately from scientific changes, in separate commits.

## Scientific claims

- The model is not calibrated to human motor neurons. `human_calibrated` is `False` in every summary and the checker requires it. Never set it to true.
- C08 is a candidate from a parameter search. Do not call it a default, healthy, validated, or accepted parameter set.
- The simulations cannot establish what causes ALS or predict whether a treatment works. Do not write text that implies either.
- Keep four things distinct in code comments, documentation, and reports: published measurements, values transferred from other cell types, numerical results, and hypotheses.
- Scenario grids and search bounds are not measured intervals or confidence intervals. Do not relabel them.
- Historical spike counts are regression targets. They are not independent human observations.
- Report negative results and limitations next to positive ones. For example, the redox scenario's GSH pool declines without any stimulation, and most of its 72-hour branches fail the pre-declared screen.
- Record every new value with units, source DOI or URL, species and cell type, whether it is measured or assumed, and the test that could falsify it.
- Never infer a unit from a number. Schedules use seconds, voltage mV, current pA, conductance nS, and concentrations mostly micromolar, but some historical routines use milliseconds or normalised states.
- Never invent a citation, a measurement, or a result you did not run.

## Documentation and provenance

- All content is English.
- Older documents carry stacked headers from past releases. For example, several open with "Current 0.3.0 stage", which is out of date. Check `CHANGELOG.md` and the newest stage note in `docs/` for the current release rather than trusting a heading.
- Files in `provenance/` and the dated stage notes are historical records. Add a new record for new work; do not rewrite an old one to match the present.
- Do not hand-edit `parameters/registry.csv` or `parameters/registry.json`. They are generated exports and must stay consistent with each other.
- When behaviour changes, update the documents that describe it: the root README, `docs/RUNNING.md`, the experiment index, the model catalog in `models/`, and the changelog. Changes made after a tagged release go under the "Unreleased" heading in `CHANGELOG.md`.
- State counts and schema hashes appear in both `models/*.json` and `experiments/*.json`. Change them together.
- Start commit subjects with `[claude]` or `[codex]` so the history records which agent made the change.

## Privacy and licensing

- Never add patient records, correspondence, contact details, email addresses, or downloaded copies of journal articles.
- Do not change `LICENSE.txt`, the preserved AGPL text, or third-party notices unless the owner asks. Do not invent prices or commercial terms.
- The owner removed the licensors' personal names from the current files on purpose. Do not add them back to any file, and do not restore them in older records.
- Do not import code whose licence conflicts with the Motoneuron Noncommercial Research License. Code adapted from Blue Brain Project/EPFL must keep its Apache-2.0 licence and notice files.
- Keep generated trajectories, caches, and virtual environments out of Git. `outputs/` is ignored for this reason.

Further background: [contributing](CONTRIBUTING.md), [scope](docs/SCOPE.md), [assumptions](docs/ASSUMPTIONS.md), [validation](docs/VALIDATION.md).
