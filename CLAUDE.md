# Claude Code instructions for skbtools

@AGENTS.md

## Branches

- Do all work on the `development` branch. `master` is the stable version behind the
  results in the submitted paper; never commit to or modify it.
- Changes requested by reviewers count as an explicit request to change scientific
  behavior, but note any change to results that appear in the paper.

## Testing

The tests run with the Numba JIT off (`tests/conftest.py`). The few that run full
inversions are marked `@pytest.mark.slow` and skipped by default, so `pytest` takes
about 10 s.

- While working, run Ruff and the test files that cover the modules you changed.
  Run the default suite (`pytest`) before committing.
- Run the slow tests (`pytest -m slow`, about 40 s) only when a change touches the
  Gauss-Newton inversion, the synthetic generator or the MCMC sampler.
- Keep tests small: use the fewest epochs and MCMC iterations that still exercise
  the code. Mark any new test that takes more than a few seconds as slow.

## Paper revision (local only)

Revision materials live in `revision/`, which is gitignored and never pushed:
the submitted manuscript, the reviewer reports, the revision instructions and the
revision plan. If that folder exists, follow `revision/INSTRUCTIONS.md`.

@revision/INSTRUCTIONS.md
