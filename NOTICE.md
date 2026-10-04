# Notice, attribution and reuse

## Authorship
**Concept, research design and interpretation: Rahul Shukla.**
**Engineering assistance: OpenAI Codex and Anthropic Claude Code.**
Codex and Claude Code provided software/engineering assistance; they are not authors
of the research design or its interpretation.

## Underlying data — UNIDO INDSTAT Rev.4 (licensed)
The economic accounts behind this tool come from **UNIDO INDSTAT Rev.4**, which is a
**licensed** database. UNIDO remains the source of, and rights-holder in, that data.

- **Raw UNIDO files are not redistributed** in this repository.
- Only compact **derived indicators** (growth rates, ratios, a rebased employment
  index) and **model outputs** (calibrated risk probabilities and tiers) are shipped,
  as a minimal deployment artefact with documented provenance
  (`data/processed/meta.json`). These are transformations and model results, not the
  underlying licensed values.
- To regenerate the derived data you need your own licensed UNIDO INDSTAT Rev.4 copy;
  see `README.md`. Review UNIDO's data terms before any onward redistribution of
  derived rows.

## Reuse of the application and method
The ownership and licensing of Rahul Shukla's original interface, method and model
are **reserved** — this repository does **not** grant an open-source (MIT/Apache) or
Creative Commons licence. Please contact the author for reuse beyond viewing and
running the research preview. The attribution above must be retained. This notice
does not grant any rights in the UNIDO data, which remain governed by UNIDO's terms.

## Nature of the output
Outputs are **descriptive, calibrated risk associations for screening**, not causal
estimates, forecasts or policy prescriptions.
