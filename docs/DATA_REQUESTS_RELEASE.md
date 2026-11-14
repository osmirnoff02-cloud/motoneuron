# Data requests and filename migration — 0.4.0

Codex, 2026-11-14. [Home](../README.md) · [Experimental data requests](../DATA_REQUESTS.md).

This release adds a public English request document covering all 43 groups in the registry snapshot and a sanitized catalogue of 1,044 numerical records. The catalogue includes historical fit bounds, protocol grids and variants beyond the shipped runtime. It is not a list of 1,044 independently unknown biological parameters. Contacts and correspondence metadata are excluded.

Public filenames no longer carry `_codex` or `_claude` suffixes. Use `run_demo.py`, `verify_package.py` and `tests/check_output.py`. Generated filenames such as `checkpoint.json`, `summary.json` and `trace.npz` also omit the suffix. Update scripts that refer to old names. The old releases remain available unchanged. Development attribution and historical provenance are preserved; original and current paths/hashes are recorded in [the path map](../provenance/publication_path_map.json).

## Verification performed for this release

- Compared all 1,044 catalogue records with the captured source registry: identifiers, values, units, classification and notes are unchanged. The source hash and generation timestamp are in the catalogue.
- Checked all 208 Python files: AST structure and numerical literals are unchanged except identifier/string spelling used for renamed files and imports.
- Executed the 4,425-state Burley C08 smoke example and the optional 4,431-state Selvaraj-Con2 C08 acid-removal smoke example (tau = 120 s), each with full-state restart checking. Both passed finite-state, numerical-balance and restart criteria, and all saved states matched their packaged pre-migration references within rtol 2e-7 / atol 1e-8.
- Ran in isolated Python mode with access to the private research tree and tool tree forbidden, and network access blocked by the runner. The existing publication virtual environment was reused; this was not a fresh dependency-installation test.

Exact differences and environment are in [the validation record](../provenance/data_requests_validation.json). These are short numerical regressions, not human calibration. Pulse and long-duration protocols were not repeated for this release; the 0.3.0 results remain historical evidence under their original conditions. Filename changes do not resolve biological limitations.

The package hash manifest and local Markdown links are checked by `python -I -B verify_package.py`. Licensing is unchanged: the custom terms began with 0.2.0; prior 0.1.2 AGPL rights and third-party terms are preserved.
