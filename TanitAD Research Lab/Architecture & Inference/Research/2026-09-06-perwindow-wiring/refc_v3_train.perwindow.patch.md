# `refc_v3_train.py` — the trainer-side half of D-PWVOCAB-1

**SPECIFIED, DELIBERATELY NOT APPLIED.** ⛔ It lands **in one commit** with the
`v6.py` / `refc_v3.py` change in `PERWINDOW_WIRING.md` §4.2, or not at all — a
flag whose consumer cannot be built is the *"module built, wired to nothing"*
defect the refcv5 plan carries as **P6**, and this stream will not add a second
instance of it.

⚠️ **Anchors are function names and unique code strings, never line numbers** —
both live trainers were edited by siblings during this session.

---

## 1. `V3Dataset` — emit the per-window targets

**Anchor:** the class attribute block in `class V3Dataset(RouteV21Dataset)`,
beside `nav_from_v7` / `nav_args_enabled` / `agent_join`.

```python
#: --perwindow-vocab (D-PWVOCAB-1): emit the v4 per-window KINEMATIC
#: tactical targets (5 lat x 6 lon) from the episode's OWN poses, as a
#: THIRD source for the z_tac heads beside kin3 and v7.2.
#: ⛔ Default False keeps every banked arm bit-identical: while it is
#: False the batch carries no `lat_pw`/`lon_pw` and `compute_losses_v3`
#: takes exactly the path it takes today.
#: ⭐ NO ARTIFACT IS READ. `v4_labels.lat_target/lon_target` are pure
#: functions of `poses [T,4]`, MEASURED at 0.383 ms/window, and are
#: bitwise equal to the banked `mint_episode` rows (negative control:
#: anchor shifted by 1 -> not equal). The lost `labels_train_v4.pt` was
#: a cache of this function, not a source of truth.
perwindow_vocab: bool = False
perwindow_stats: dict | None = None
```

In `__getitem__`, **after** the existing item is built and **only** when
`self.perwindow_vocab`:

```python
if self.perwindow_vocab:
    poses = <the episode poses this item already resolved>
    L = <the window's last pose index — the SAME index the existing
         `pose_last` is taken from; do not recompute it independently>
    item["lat_pw"] = _pw_lat(poses, L)     # 5-token, IGNORE elsewhere
    item["lon_pw"] = V4.lon_target(poses, L)
```

⛔ **`_pw_lat` masks the two dead tokens** rather than sizing a head that can
never train them:

```python
# MEASURED 68,377 windows / 400 episodes: abort_lc 4, pull_over 0.
_PW_LAT_KEEP = ("lane_keep", "lc_left", "lc_right",
                "nudge_left", "nudge_right")
def _pw_lat(poses, L):
    i = V4.lat_target(poses, L)
    t = V4.LAT_TOKENS[i] if 0 <= i < len(V4.LAT_TOKENS) else None
    return (_PW_LAT_KEEP.index(t) if t in _PW_LAT_KEEP
            else v7l.IGNORE_ID)
```

⚠️ **Call `lat_target`/`lon_target`, never `mint_window`,** in `__getitem__`:
`mint_window` re-runs `savgol` + `vtarget_v2` over the whole track per call for
fields the tactical heads do not use.

⛔ **Import `v4_labels` at module scope beside `import refb_labels`** — a
lazy import inside `__getitem__` pays it per worker per item and hides an
`ImportError` until after the rollout, which is the analysis-time-import trap.

## 2. The flag

**Anchor:** `build_parser`, beside `--v7-labels`.

```
--perwindow-vocab      store_true, default False.
  Help: "Supervise the z_tac tactical heads with the v4 PER-WINDOW kinematic
  vocabulary (5 lat x 6 lon) derived from poses, instead of the runtime kin3
  3x3. No label artifact is read. Mutually exclusive with --v7-labels."
```

⛔ **REFUSE `--perwindow-vocab` together with `--v7-labels`** — two label
sources on one head is the *"train silently wrong classes"* defect the existing
width refusals were written for. Refuse loudly at parse time.

## 3. `_pin_trainer_cfg` — the vocabulary follows the labels

**Anchor:** the existing

```python
cfg.tac_vocab_version = ("v7.0" if getattr(args, "v7_labels", None) else "kin3")
```

becomes a three-way selection with `"kin76"` for `--perwindow-vocab`. ⛔ Keep
the existing comment's contract intact: **the vocabulary follows THE LABELS,
never a hardcode** — that pin is what stopped an 8-wide build training on 3
classes.

## 4. `compute_losses_v3` — a third source through the SAME seam

**Anchor:** the block beginning `use_v7 = "lat_v7" in batch`.

```python
if "lat_pw" in batch:                      # D-PWVOCAB-1
    lat_t, lon_t = batch["lat_pw"].to(device), batch["lon_pw"].to(device)
    n_lat_expect, n_lon_expect = 5, 6
    src = "kin76 (5x6)"
elif use_v7:
    ...unchanged...
else:
    ...unchanged...
```

⛔ **Change nothing above this block.** `lat_k`/`lon_k`, `loss_lat`, `loss_lon`,
`update_tactical_prior` and the CORE width refusal stay exactly as they are —
the core surface is structurally kin3 in `refc.py` and this stream does not
touch it. ⭐ The existing z_tac width refusal then guards the new source for
free, because it compares against `n_lat_expect`/`n_lon_expect`.

⚠️ The existing all-IGNORE guard (`lat_ok`/`lon_ok`, *"an ALL-ignored batch is
NaN, not zero, so it is SKIPPED"*) is **required** by this source, because
`_pw_lat` masks `abort_lc`/`pull_over` to IGNORE. It is already there; do not
remove it.

## 5. The stamp

**Anchor:** `effective_weights_stamp_v3` / `_seam_stamp`.

Stamp into `config.json`: `perwindow_vocab` (the flag as the operator passed
it), `tac_vocab_version` (the effective value), the **built head widths**, the
per-token label histogram over the train split, and ⛔ **`man5_active`** — see
`PERWINDOW_WIRING.md` §4.1: any non-kin3 vocabulary silently sets `man5 = None`
and drops the H19 lateral prior. **The effective-weight guard exists to refuse
a weight an operator typed that a later layer zeroes; this must not become the
same failure wearing a vocabulary costume.**

## 6. ⛔ The two tests, and what each must prove

**`test_perwindow_off_is_bit_identical`** — a fixed-seed forward/backward with
the flag OFF, compared **tensor by tensor** against the same step on the
pre-change trainer. Assert **0 bitwise differences over the full tensor set**
and print the count compared (precedent: 341 tensors, 0 differences). ⛔ An
assertion that OFF is unchanged is not proof; the comparison is.

**`test_perwindow_path_is_reached`** — ⛔ **not an inspection.** Run a real
`train()` step with the flag ON and assert the `kin76` branch **executed** (an
instrumented counter, or the stamped `tac_vocab_version` + head widths read
back off the built model), **and** assert the same test **FAILS with the flag
OFF**. `main` runs `preflight` only under `--preflight` and otherwise calls
`train()` directly, so a check that lives in `preflight` is **not on the
training path** — this test must exercise `train()`.

⚠️ **Mutation, not census.** An AST census once read 0 suspects on **both** the
fixed and the broken trainer. Reintroduce the defect (point the branch at kin3
labels while claiming `kin76`) and prove the width refusal fires.
