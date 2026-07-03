# test/_tmp/ — dev benchmarks, profile dumps, and one-time measurements

This directory is the long-term home for **non-test** Python files in the
project — anything that's useful for development or debugging but doesn't
qualify as a regression-protection test.

## Convention (per CLAUDE.md + project norm)

Files in `test/_tmp/` are **not** pytest tests. They're standalone scripts
run by hand (typically with `python test/_tmp/<name>.py [args]` or with
`pytest -s`). The two patterns:

1. **Dev benchmarks / profile dumps** — measure perf, dump top-N kernels,
   per-mb timing tables, etc. Long-lived reference tools kept here because
   they're useful to re-run after a refactor.
2. **Test-first repros for a model/loop bug** — when fixing a model or
   loop bug, write a small repro here first (per the user's CLAUDE.md
   "test first" rule), then promote the *correctness assertion* to a
   proper `def test_*` in `test/` (with the historical bug documented in
   the docstring + memory), and **delete the repro here** once the
   regression test is in place.

## What does NOT belong here

- Files that define `def test_*(...)` — those go in `test/`.
- Files that document a known regression and assert it stays fixed —
  those go in `test/` as pytest tests (this is the user's normalization
  rule: "everything in `test/` should protect against a historical bug").

## Naming

The `test_` prefix is kept so `git grep` for test files still finds them,
but the files are explicitly dev tools, not pytest tests. Run them via
`python test/_tmp/<name>.py` or `pytest -s test/_tmp/<name>.py`, NOT via
plain `pytest test/_tmp/` (they have no `def test_*` to collect).

## When in doubt

If a dev script encodes a known historical bug's correctness check, promote
it to a real pytest test in `test/` with a docstring citing the fix
commit, and delete the `_tmp/` copy.
