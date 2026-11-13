# Data availability and provenance

Codex, 2026-11-13. [Home](../README.md).

All non-software runtime inputs needed for the documented smoke/pulse examples are included under `data/`. The source manifest lists paths, sizes and SHA-256; numerical input JSONs are small derived tables, model profiles, schemas and conditional constraints. These inputs are not raw patient records and are not newly measured human constants. Original relative paths and author suffixes are retained.

The selected checkpoint files are derived model states. Prepared Golding/Q3 states retain their original hashes and conditional provenance; they are not silently substituted into the fresh C08 example.

Source URLs occurring in runtime/input files are indexed in [references](../references/source_links_codex.json). This extraction is a discovery aid, not confirmation that every referenced article was fully read. Species/cell-system limitations remain in source annotations and parameter records. Extracted data and third-party components require their own redistribution/attribution review; see [notices](../THIRD_PARTY_NOTICES_codex.md).

The full project's primary FASTQ/RAW files, PDFs, transcriptome indices and historical trajectories remain in the private workspace and are not required by the demonstrated model runs. The full RNA-seq/proteomics analyses are not represented as reproducible here. If those branches are released later, each must add accession identifiers, sample design, exact raw-file manifest, actual access routes, reference versions and complete processing commands. A checksum without a usable data source is insufficient.

Selected long-run and electrical summaries are historical evidence, clearly distinguished from freshly reproduced checks. The included commands allow a new conditional experiment; exact bitwise reproduction of every historical run is not claimed. In particular, historical runtime/environment snapshots and a later export snapshot can differ.
