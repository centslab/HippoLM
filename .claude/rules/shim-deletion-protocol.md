---
paths:
  - "**/*_legacy.*"
  - "**/_legacy/**"
  - "**/_shim*"
---

# Shim / `_legacy` deletion protocol

Before deleting any shim / backwards-compat / `_legacy` file in this
repo, three checks. The full rule lives in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md) → *Shim deletion protocol*;
this rule is just the path-triggered reminder.

1. **Grep absolute**: `rg "from src\.pkg\.mod " .` and
   `rg "import src\.pkg\.mod" .`.
2. **Grep relative** (same package and subpackages):
   `rg "from \.mod_name" src/pkg/` and
   `rg "from \.\.pkg\.mod" src/pkg/`.
3. **Smoke-import** the affected packages:
   `python -c "import pkg_a; import pkg_b"` for every package that
   could be affected (sub-second if no CUDA init; catches the
   relative-import case the grep missed).

If a relative importer is found, fix it to point at the canonical
path. **Do not restore the shim** — that defeats the cleanup.

The 2026-07-11 `src/models/kda.py` deletion nearly broke
`src/models/model.py` via `from .kda import KDA` (the relative form
didn't match the absolute-only grep). The post-delete smoke import
caught the failure before commit; the protocol is the reason this
rule exists.
