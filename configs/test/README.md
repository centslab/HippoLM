# End-to-end test configs

One yml file per e2e test scenario. Each config is runnable on
its own with no extra CLI flags:

```
python scripts/train.py --config configs/test/<scenario>.yml
```

## Why multiple yml files (no `--profile` flag)

The yml overlay mechanism in `scripts/cli.py:parse_args` already
supports this pattern: `--config` points at a yml, the yml's top-
level keys become parser defaults via `set_defaults(**yml_dict)`,
and any explicit CLI flag still wins on top. Adding a `--profile`
flag would be redundant — and it would force the test matrix into
the argparse schema, where each new scenario needs a CLI change.

## `extends:` — DRY inheritance from `base.yml`

Each yml in this directory starts with `extends: ../base.yml` so
it inherits the production defaults (precision block, optimizer
hparams, data source, output settings, etc.) and only has to list
the test-specific overrides. Implemented in
`scripts/cli.py:_load_yaml_with_extends` — recursive, with a
visited-set guard against circular chains. Merge is **shallow**:
the child wins for any top-level key it specifies, but cannot
reach inside a nested dict. If you need to override one sub-key
of `precision` (say just `precision.adamw_m.dtype`), repeat the
whole `precision` block in the test yml.

## Conventions

1. **Each yml starts with `extends: ../base.yml`** so production
   defaults (precision, optimizer hparams, data source) flow
   through automatically. Only list the test-specific overrides
   below that line.

2. **Each yml starts with a `# configs/test/<name>.yml` header
   comment and a 2-3 line "what does this test" blurb.** Future
   readers need the why, not just the diff. The "Diff vs base.yml"
   block at the bottom is the authoritative spec.

3. **Always set `use_dummy_data: true` in test configs.** The
   streaming / tokenize path is orthogonal to what these tests
   cover, and skipping it removes the network + tokenizer as a
   source of flakes.

4. **Always set `output_dir: output/<scenario>`.** Don't let
   multiple test runs clobber each other's logs.

5. **Set `checkpoint_interval: 0` unless the test is specifically
   exercising checkpoint save/load.** Checkpoint roundtrips have
   their own test in `test/test_tp_model_state_dict.py`; the e2e
   configs focus on forward / backward / optimizer.

6. **Use small dims.** The goal is to exercise the code path, not
   to time the production model. `num_layers=2`, `num_heads=2`,
   `head_dim=32-64` is plenty.

7. **Pin the fields that name the scenario explicitly.** E.g.
   `kda_chunk.yml` sets `kda_mode: chunk` even though it's the
   default — so the intent is visible in the file. Otherwise a
   future default change silently mutates the test.

## Scenarios

| Config | What it tests | Notable flags |
| --- | --- | --- |
| `quick.yml` | Cheapest end-to-end sanity. ~30s. | `use_dummy_data`, 2 layers, 2 steps |
| `kda_chunk.yml` | KDA chunkwise Triton fwd + bwd. | `kda_mode: chunk`, K=128 head_dim |
| `kda_fused_recurrent.yml` | KDA single-token recurrence. | `kda_mode: fused_recurrent` |
| `no_short_conv.yml` | Chunk path with short conv skipped. | `use_short_conv: false` |
| `with_short_conv.yml` | Chunk path with short conv enabled (default). | `use_short_conv: true` |
| `tp1.yml` | TP world size 1. | `tp_size: 1` |
| `tp2.yml` | TP world size 2, gloo sim on 1 GPU. | `tp_sim: true`, `tp_size: 2` |
| `checkpoint.yml` | Model-weight checkpoint save + prune path. | `checkpoint_interval: 1`, `checkpoint_keep_last_n: 1` |
| `data_loading.yml` | Real streaming data path (MS → HF fallback). | `use_dummy_data: false`, **network-bound** |

## When to add a new config

Add a new yml when:

- You have a new code path that's worth pinning (e.g. a new
  optimizer option, a new attention mode, a new precision mode).
- The scenario doesn't fit any existing config's intent.
- You find yourself repeatedly setting the same 3-4 CLI flags
  on top of an existing yml — that flag combo is a missing config.

DO NOT add a new yml that just re-orders existing fields or
changes default values; that's a refactor of `base.yml`, not a
new test.

## Don't forget to add a test for new argparse flags

The yml overlay adds new keys to the `Namespace` even without a
matching `add_argument` — so you can add a yml-only field and
it will flow through. But the field won't show up in `--help` or
have a type check. If a new field is worth a test config, it's
worth an `add_argument` in `scripts/cli.py:build_parser` too.

## How to extend the schema

If a test needs a flag that doesn't exist in `scripts/cli.py`,
add the `add_argument` to `build_parser()`. The yml overlay
flows new keys through as `Namespace` attributes even without a
matching `add_argument`, but adding the explicit flag keeps
`--help` complete and gives the field a type.

## Smoke-test command (production parity)

The canonical smoke test (run from the project root after any
change) is:

```
python scripts/train.py --config configs/test/quick.yml
```

If `quick.yml` passes, run the TP variant:

```
python scripts/train.py --config configs/test/tp2.yml
```

If both pass, the dev box is in a known-good state.
