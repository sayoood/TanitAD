# RESULT — the session's load-bearing eval dumps are out of the temp directory

**Date** 2026-09-06 · **Stream** Architecture & Inference · **Evidence MEASURED (ours)** · **GPU-days 0** (I/O only; the A40 was not touched)
**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-dump-banking/`
**Sidecar** `raw/BANK_MANIFEST.json` — every md5, byte size, episode count, window count and identity below is machine-readable there.

## 0. Answer

**Is the session's load-bearing evidence still in a temp directory? — NO.** All four distinct dumps are
banked in-repo as md5-verified tarballs, each one re-opened after banking and asserted to carry real
values. The fifth path was a byte-identical duplicate and is recorded as such rather than banked twice.

**Nothing was too large to bank.** The whole corpus is **19,951,616 B (~19.0 MiB) of source**, compressing
to **13,590,121 B (~13.0 MiB)** of tarball — comparable to the single `navflip_dump.tgz` already in the
repo. No escalation was needed on size.

## 1. Sizes BEFORE anything was moved

| dump (temp path) | source bytes | files | episodes | windows |
|---|---|---|---|---|
| `scratchpad/dump/refcv4b_t1_dump` | **11,763,914** | 284 | 141 | 4,823 |
| `scratchpad/dump40284/refcv3_40284_dump` | **2,293,188** | 283 | 141 | 4,823 |
| `scratchpad/rc3dump/refcv3_40284_dump` | **2,293,188** | 283 | 141 | 4,823 |
| `scratchpad/dump30k/full_dump` | **2,295,778** | 283 | 141 | 4,823 |
| `scratchpad/dk/dump/full` | **1,305,548** | 283 | 141 | **282** |
| **total** | **19,951,616** | 1,416 | | |

## 2. What was banked

| tarball | bytes | md5 | content digest (gzip-mtime-independent) |
|---|---|---|---|
| `raw/refcv4b_t1_dump.tgz` | 10,531,256 | `f7d3a8ca0eaec4a6c624a8d947ae6dfe` | `427529e56ebd644876aa512e4b1f2835` |
| `raw/refcv3_40284_dump.tgz` | 1,385,352 | `0079766f9a53acaf3f0b1adc61af786e` | `56c5e5d308c4c3141d2a737a0b1155f7` |
| `raw/refcv3_30k_full_dump.tgz` | 1,387,427 | `a4515312577860c4ea3397ba9995ee0e` | `44bbb5e9138ff158f1bff38e0e43b520` |
| `raw/refav1_t1_dump.tgz` | 286,086 | `ded1c675982cc669b84c21bb32defa2f` | `5b5bb19947e6f505a345d3c37846199f` |

⚠️ **Verify a banked tar with the CONTENT DIGEST, not the `.tgz` md5.** gzip embeds an mtime, so
re-tarring the same bytes yields a different `.tgz` md5 — MEASURED here across two runs of the banker.
The content digest is md5 over sorted `<member> <md5(member bytes)>` lines and is stable.

## 3. The verification, and why the not-all-NUL assertion actually discriminates

This mount has produced **correct-size ALL-NUL files**, so size proves nothing. Every tarball got:

1. **md5 both ends** — computed on the local tar, recomputed on the repo copy, recorded only on match.
   All four matched on the first attempt; a mismatch would have retried up to 5× and then reported
   **INCONCLUSIVE**, never a pass.
2. ⭐ **A DISCRIMINATING CONTROL, in the same run.** A genuine all-NUL file (262,144 B) was created and
   pushed through the **same** non-zero counter. **MEASURED: the control read 0 non-zero bytes in its
   first 64 KiB** while the banked tars read **65,337 / 65,234 / 65,240 / 65,238**. Without the control
   reading 0, ">0 non-zero bytes" would have been an assertion about nothing.
3. ⭐ **The banked (repo) tar was re-opened** — not the local one — its `manifest.json` parsed and an
   `ep000.npz` loaded with numpy. `g` finite, `nonzero_frac = 1.0000`, `|sum| > 1` for all four.

## 4. Identity — verified from each manifest, NOT from its directory name

⚠️ **A name is not provenance**: a sibling found a local "refcv4b dump" that was really refcv3. Two
independent verifications were run.

**(a) From the manifest** (`model.ckpt` / `.step` / `.n_anchors` / `.cfg.core.anchors` / arm key set):

| banked tarball | ckpt (from manifest) | step | n_anchors | anchors cfg | arms |
|---|---|---|---|---|---|
| `refcv4b_t1_dump.tgz` | `/workspace/experiments/refcv4b-b1-v72-40k/ckpt_40284_FINAL.pt` | 40284 | **117** | `{n_anchors:117, pool_size:4096, seed:0, `**`v0_conditioned:true`**`, ref_speed_ms:10.0, control_units:"alat", alat_v_floor_ms:4.0, kappa_cap:0.12}` | os, ha, ha0, **ha0_ext**, os_navshuf, os_navzero, **oracle_sel** |
| `refcv3_40284_dump.tgz` | `/workspace/experiments/refcv3-b1-v72-30k/ckpt_40284_FINAL.pt` | 40284 | **128** | `{n_anchors:128, pool_size:4096, seed:0}` — ⛔ **no `v0_conditioned` key at all** | os, ha, ha0, os_navshuf, os_navzero, oracle_sel |
| `refcv3_30k_full_dump.tgz` | `C:\Users\Admin\run_refcv3_ol\ckpt\ckpt_30000.pt` | 30000 | 128 | `{n_anchors:128, pool_size:4096, seed:0}` | os, ha, ha0, os_navshuf, os_navzero, oracle_sel |
| `refav1_t1_dump.tgz` | `/home/nvidia/experiments/refav1-b1-v72-ep3-speed/ckpt.pt` | 21109 | n/a | n/a | cl, ha, ha0, ol |

This reproduces the expected bank configs exactly, and `refcv3-b1-v72-**30k**` holding a
`ckpt_40284_FINAL.pt` is a **stale run-directory name**, not a step mismatch — the step field says 40284.

**(b) A numeric discriminator over all 141 episodes** — the decisive one. Ground truth must be
bit-identical (same eval grid) while the model's prediction must differ (different model):

| pair | `g` identical | `os` identical | max\|Δos\| | mean\|Δos\| | verdict |
|---|---|---|---|---|---|
| refcv4b vs refcv3@40284 | **True** (max\|Δ\| = 0) | **False** | 6.90476 | 0.198314 | **DIFFERENT MODEL / same windows** |
| refcv3@40284 vs refcv3@30k | **True** | False | 3.77343 | 0.119577 | different checkpoint / same windows |
| refcv3@40284 vs `rc3dump` | **True** | **True** (max\|Δ\| = **0**) | 0 | 0 | **SAME dump — duplicate** |

`g` sums to **272810.9308184178** in all of them. ⇒ The dump named `refcv4b_t1_dump` **is** refcv4b.

**(c) Checkpoint md5.** MEASURED for exactly one arm — refcv3@30000, whose ckpt is on local disk:
**428,519,790 B, md5 `00da81c6efcd91e7b618a1fbddb3b78f`**. The other three are pod-side
(`/workspace/…` on the A40, `/home/nvidia/…` on Thor); this box does not reach them and **no manifest
records a ckpt hash**. For those, identity rests on (a) + (b) above — stated rather than guessed.

## 5. Not banked, and why

`scratchpad/rc3dump/refcv3_40284_dump` (2,293,188 B) is a **byte-identical duplicate** of
`dump40284/refcv3_40284_dump`, on three independent probes: `diff -r -q` finds no difference,
`manifest.json` and `ep000.npz` md5s are equal, and the `os` array concatenated over all 141 episodes
is bit-identical. `refcv3_40284_dump.tgz` **is** that dump; it was not banked twice.

## 6. ⭐ Two corrections to earlier statements

### 6.1 `C3_os_reproduction` PASSES — the tool is sound, and the A40 family IS quotable

MEASURED in `…/2026-09-06-refcv4b-navpred/raw/paired_navpred_SELFTEST_on_banked_landing_dump.json`,
run against **`dump/refcv4b_t1_dump`** — the very dump banked here as `refcv4b_t1_dump.tgz`:

* `C3_os_reproduction`: measured **0.2975**, banked **0.2975**, **abs_diff 2.6e-05**, tolerance 0.001 → **PASS**
* `C1_model_free_known_value`: `ha` 0.2996 vs 0.2996 and `ha0` 0.6723 vs 0.6723, **abs_diff 0.0** → **PASS**
* `C2_grid`: 4,823 windows / 141 episodes → **PASS**

⇒ **The 0.001031 FAIL is the Thor cross-hardware roll only** — A40/x86_64/torch 2.8.0+cu128 versus
Thor/aarch64/torch 2.13.0+cu130, argmax tie-breaks, `n_distinct` 50 → 51.
⇒ ⛔ **The A40 family (`os` 0.2975, `os_navzero` 0.3928) IS QUOTABLE; the Thor family (0.2965 / 0.3926)
is not.** Earlier statements that *both* families were unquotable are **too strong** and are corrected here.

⭐ Independently corroborated while banking: the two dumps really are different hardware. The dump
banked here carries ckpt `/workspace/experiments/refcv4b-b1-v72-40k/…` (A40 pod); the in-repo
`navpred_dump.tgz` carries `/home/nvidia/refcv4b/ckpt_40284_FINAL.pt` (Thor) — same step 40284, same
117 anchors, same 141/4,823 grid, different box.

### 6.2 The one in-repo "dump" that predates this banking is a SYNTHETIC FIXTURE

`TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-refcv3-arm/raw/fixture_dump` is **not a
real arm** and must never be mistaken for one. MEASURED from its manifest: **step 11, 3 episodes,
42 windows, 20 anchors**, ckpt under a temp scratchpad belonging to a *different* session id.

⛔⛔ **AND THE SELF-LABEL DOES NOT DISCRIMINATE IT — this corrects the way that fixture has been
identified.** The string *"UNVERIFIED on a real checkpoint … random-init RefCV3Model at
refc_v3_smoke_config … synthetic 3-episode slice only"* is **boilerplate emitted by
`taniteval/tools/refcv3_arm.py`**, and MEASURED it is present on **all four** refcv3-family
manifests — **including the refcv4b landing arm** at step 40284 over 141 episodes / 4,823 windows,
whose `state_dict_load` reports `missing_keys: []` and `unexpected_keys: []` (a random-init model
loads no state dict). It is absent only from the refav1 dump, which a **different tool** wrote.

⇒ **Never use `_unverified` to tell a fixture from a real dump — it would misclassify every real
refcv3-family dump we hold.** The discriminators are `model.ckpt`, `model.step`,
`grid.n_episodes`/`n_windows`, and `model.n_anchors`:

| | fixture | refcv4b landing arm |
|---|---|---|
| `model.step` | **11** | **40284** |
| `grid.n_episodes` / `n_windows` | **3 / 42** | **141 / 4,823** |
| `model.n_anchors` | **20** | **117** |
| `_unverified` present | **True** | ⛔ **True — identical string** |

*Same family as the `df` / Thor `free` / cgroup `usage_in_bytes` traps: a true statement (the tool
**was** validated only on a fixture) quoted outside its scope, where it reads as a fact about the
artifact rather than about the tool.*

## 7. ⚠️ `oracle_sel` — where it is and where it is not

The **navpred/navflip dumps DROP `oracle_sel` entirely.** VERIFIED by opening `navpred_dump.tgz`:
arms are `[os, ha, ha0, ha0_ext, os_navshuf, os_navzero, os_navpred]` — no `oracle_sel`. **That is why
the `os_navpred` roll has no ceiling arm.** Recorded so nobody looks for it later.

The dumps banked here **do** carry it: `refcv4b_t1_dump.tgz` and both refcv3 tarballs have `oracle_sel`
as a real `[W, 4, 2]` float32 array.

## 8. Escalated, not changed (this agent owns no code file)

* **Stale docstring in the refcv4b manifest**: `sidecar_schema.anchor_acc` says *"chance = 1/128 =
  0.0078"*, but that dump has **117** anchors, so chance is **1/117 = 0.00855**. The string is
  hardcoded in `taniteval/tools/refcv3_arm.py` and was not re-derived for the 117-anchor bank. Anyone
  quoting `anchor_acc` against chance from this manifest uses the wrong denominator. **Not edited.**
* **The `_unverified` boilerplate itself** (§6.2) should either be re-derived per run or dropped when a
  real checkpoint is loaded. **Not edited** — it is a code change in a live tool.

## 9. Scope honesty

Absence claims here are from **scoped** probes with an **adjacent interleaved control** that had to read
non-zero in the same breath (e.g. listing `…/Implementation/incoming/` returned 194 entries, stable over
three repeats, with a sibling package listing real files). The two paths cited in the brief resolved to
**different locations** than stated — `2026-09-06-refcv4b-navpred` and `2026-09-03-refcv3-arm` live under
`Research/`, not `Implementation/incoming/`, and the latter is in the **Benchmarks & Evals** FlyWheel —
found via `git ls-files` (a positive index assertion, immune to the `ls-tree` truncation trap). ⛔ Nothing
here should be read as a settled repo-wide absence.
