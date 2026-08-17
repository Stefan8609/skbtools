# Repository Guidelines

These instructions apply across the repository. Preserve established public APIs and
scientific behavior unless the task explicitly calls for a change.

## Priorities

- Favor correctness, readability, and traceability over cleverness.
- Keep the codebase uniform. Before changing a workflow, inspect the closest sibling
  workflow and follow its structure, naming, inputs, returns, saved keys, and level of
  abstraction unless the scientific requirements demand a difference.
- Keep implementations barebones. Make the smallest change that completely solves the
  problem, and avoid unrelated cleanup.
- Preserve raw observations and source indices. Derive filtered or fitted data rather
  than silently replacing the inputs.
- Make assumptions about units, reference frames, time systems, and data quality
  explicit at the point where they matter.

## Code Style

- Write straightforward Python with descriptive names and easy-to-follow control flow.
- Follow the naming used by the surrounding workflow, including established scientific
  names such as `GPS_Coordinates` and `CDOG_data`. Use `snake_case` for new internal
  names when there is no local convention to follow.
- Keep functions focused, but do not split linear code into helpers that merely move a
  few obvious lines elsewhere.
- Introduce a helper when it names a meaningful operation, removes real repetition, or
  isolates complexity. Do not pursue DRY at the expense of readability.
- Prefer direct data structures and function parameters over configuration classes,
  registries, factories, or other framework-like abstractions.
- Do not turn one campaign script into a standalone framework. Campaign initializers
  should look and behave alike; move code into a shared module only after a real shared
  need is established.
- Keep equivalent campaign functions aligned in argument order, return structure, save
  behavior, and output key names. Make required campaign-specific differences obvious
  and local.
- Do not add command-line argument parsing. Executable scripts should expose a short
  module-level configuration block and call the processing function directly under
  `if __name__ == "__main__":`.
- Avoid new dependencies unless they materially simplify the scientific work and the
  existing stack cannot do the job clearly.
- Follow the Ruff configuration in `pyproject.toml`: 88-character lines, double quotes,
  and the enabled E, F, and B rules.

## Documentation and Comments

- Use short, informative docstrings. A single sentence is enough for simple functions;
  document shapes, units, coordinate conventions, and return values when they are not
  evident from the signature.
- Add comments only when they explain intent, a scientific assumption, a non-obvious
  constraint, or why an apparently simpler approach is unsafe.
- Do not narrate code with comments or mechanically add docstrings, type hints, or
  section headings where they provide no new information.
- Put units in names or nearby documentation. Avoid unexplained magic numbers.

## Scientific and Array Code

- Validate array shapes, required fields, units, and bounds at input boundaries. Error
  messages should say what was expected and what was received.
- Preserve time ordering and provenance. Do not silently sort, interpolate, clamp, or
  discard observations.
- Prefer NumPy operations when they are clearer than loops. Use Numba only for measured
  hot paths and keep compiled kernels small and simple.
- Centralize campaign-specific geometry, thresholds, and paths as editable constants or
  explicit function arguments.
- Record filtering criteria and enough metadata to reproduce derived products.

## Testing and Changes

- Add small, deterministic tests for new behavior and regressions. Cover normal input,
  important boundary conditions, and invalid shapes or values when relevant.
- Keep tests readable; avoid helpers or fixtures that obscure the case being tested.
- Run the narrowest relevant tests plus Ruff checks for touched Python files.
- Preserve user changes in a dirty worktree. Do not reformat or refactor unrelated files.
