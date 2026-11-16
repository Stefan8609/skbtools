# Claude Code instructions for skbtools

@AGENTS.md

## Branches

- Do all work on the `development` branch. `master` is the stable version behind the
  results in the submitted paper; never commit to or modify it.
- Before and after code changes, run the relevant tests and Ruff checks (per AGENTS.md),
  and run the full test suite before committing.
- Changes requested by reviewers count as an explicit request to change scientific
  behavior, but note any change to results that appear in the paper.

## Paper revision (local only)

Revision materials live in `revision/`, which is gitignored and never pushed:
the submitted manuscript, the reviewer reports, the revision instructions and the
revision plan. If that folder exists, follow `revision/INSTRUCTIONS.md`.

@revision/INSTRUCTIONS.md
