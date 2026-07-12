# Contributing to HippoLM

This file covers development workflow conventions. Project overview,
hardware target, architecture, and roadmap live in
[`AGENTS.md`](AGENTS.md). Historical incidents and one-off
debugging notes live in the project's auto-memory.

## Reading order for new work

1. Skim [`AGENTS.md`](AGENTS.md) (project overview +
   skill/rule index).
2. Read this file end-to-end.
3. For the area you're touching, find the relevant skill in
   `.claude/skills/` (indexed in `AGENTS.md`).
4. For path-scoped rules that auto-inject when relevant, see
   `.claude/rules/` (indexed in `AGENTS.md`).

These rules are also encoded in the auto-memory; this file is the
canonical written form, the memory entries are the carry-over
across conversations.

## Hard rules

### Test-first on model changes

When changing model code, the training loop, the optimizer layout,
or any data-path code that interacts with autograd, sketch a pytest
file in `test/_tmp/<feature>.py` **before** the change. Cover:
numerical correctness of the new logic, edge cases (empty, single,
batched), and a "no regression" check on the unchanged code path.

Once the bug is fixed, promote the *correctness assertion* into a
proper `def test_*` in `test/` (with a memory entry capturing the
historical bug), and delete `test/_tmp/<feature>.py`. See
[`test/_tmp/README.md`](test/_tmp/README.md) for what does and
doesn't belong in `_tmp/`.

### Smoke test before commit (training-loop / optimizer / data path / model)

Any change that touches the training loop, optimizer layout, data
path, or model code must go through the smoke test before commit.
See
[`.claude/skills/run-smoke-test/SKILL.md`](.claude/skills/run-smoke-test/SKILL.md)
for the exact command and PASS/FAIL criterion (non-empty
checkpoint `state_dict` after the run, measured against the
5060 Ti 16 GB ceiling).

### Verify correctness before claiming speedup

A perf change is not done until the sweep includes:

1. `torch.isfinite().all()` on all outputs (no silent NaN — see
   the CHUNK=32 incident in the auto-memory
   `feedback_verify_correctness.md`).
2. `max-diff vs reference` within operator tolerance.
3. FLOPs counted properly (every `tl.dot` is `2 * M * K * N`,
   not `M * K * N`), AI = FLOPs / bytes, and ridge comparison
   before claiming "compute-bound" (see
   `feedback_roofline_discipline.md` for the worked example).

See
[`.claude/skills/kda-correctness-sweep/SKILL.md`](.claude/skills/kda-correctness-sweep/SKILL.md)
for the sweep procedure.

### Autotune for testing only

Production kernels ship with hardcoded `BLOCK_*`, `num_warps`,
`num_stages` configs. Autotune (`triton.autotune`) is allowed
only in `test/_tmp/` or test files; never in
`src/models/ops/cuda/**`.

Use autotune in `test/_tmp/` to find good configs, but the
production code must hardcode them. (See auto-memory
`feedback_autotune_production.md`.)

### Shim deletion protocol

When removing a `_legacy` / shim / backwards-compat / re-export
file:

1. Grep absolute: `from src.pkg.mod import X` and
   `import src.pkg.mod`.
2. Grep relative: `from .mod import X` and
   `from ..pkg.mod import X`.
3. Smoke-test the import chain:
   `python -c "import pkg_a; import pkg_b"` for every affected
   package (catches the relative case the grep missed;
   sub-second if no CUDA init).
4. If a relative importer is found, fix it to point at the
   canonical path (don't restore the shim — that defeats the
   cleanup).

The 2026-07-11 `src/models/kda.py` deletion nearly broke
`src/models/model.py` via a relative import — this protocol is
the reason step 2 exists. (Encoded as
[`.claude/rules/shim-deletion-protocol.md`](.claude/rules/shim-deletion-protocol.md).)

## Coding conventions

- **Comments and identifiers in English.** Reply / conversation
  in 中文. (See auto-memory `user_language.md`.)
- Match existing repo style — don't reformat untouched code in a
  drive-by edit.
- No backward-compat shims, unused `_var` renames, or
  "// removed" comments for deleted code. If it's dead,
  delete it.
- Lazy imports via PEP 562 module-level `__getattr__` when a
  package re-exports a class whose module transitively depends on
  this package's siblings. Test the cycle before claiming the
  refactor works:
  `python -c "import pkg; from pkg.dependent_module import X"`.
  (See `feedback_pep562_lazy_getattr.md`.)
- Don't propose `torch.compile(model, mode="reduce-overhead")` —
  the post-accumulate-grad hooks + custom TP autograd Functions
  don't compose with it. Just set
  `torch.set_float32_matmul_precision("high")`.

## Commits

Conventional Commit prefixes: `feat:`, `fix:`, `refactor:`,
`chore:`, `test:`, `docs:`. One logical change per commit.
Don't amend published commits. Don't push to `main` without
explicit per-action approval.
