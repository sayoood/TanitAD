# D-LAN-PF / D-LAN-COV — the v3 preflight could not see the pathway it was cited for

`TanitAD_DataFlyWheel · 2026-08-23 · P1 (tactical/strategic label pipeline) items (b) and (d)`

> ⚠️ **Directory name:** this package is filed under `TanitAD Research Lab/` because that is
> the tree at `c29c659`, the commit this worktree is on. The rename to
> `TanitAD Research Lab/` is real but lives UNCOMMITTED in the main checkout. Whoever
> lands this must move the package with the rest of the rename, not treat it as history.

---

## 0. The headline

**The stated acceptance criterion for the label pipeline was unsatisfiable, and it was aimed
at the wrong number.** The criterion I was handed reads:

> *"(d) `refc_v3_train.py --preflight --goal-str` shows `route` NON-ZERO"*, because
> *"`route` stays 0.0 until `lan` exists"*.

Both halves are **REFUTED by measurement**:

1. `route` is the **v2.1 NAV-derived** route CE, masked by `nav_valid`
   (`refc_v3_train.py` §compute_losses_v3, the `mask = nav_valid` line). It is **bit-identical
   with and without every LAN flag**. Minting `lan` could never have moved it.
2. `preflight()` built a bare `V3Dataset`, which **emits no `lan` key at all**, so
   `compute_losses_v3` skipped `loss_gstr` BY DESIGN and `goal_str` never entered the loss
   dict — *with or without* `--goal-str`. The flag was inert on the preflight path.

⇒ **The gate could not observe the quantity it was being quoted for.** Same class as
C9/C13/C14: *an instrument structurally unable to report the answer it is cited for*. Had
someone "made `route` non-zero" to satisfy the criterion, they would have been reporting the
nav echo — the exact C6 confound this programme has already logged twice.

**Also refuted: `lan` does not need to be "minted onto the corpus".** `lan_window_features`
derives LAN from `episode.poses` at dataset-construction time
(`refc_train.lan_dataset_class._WithLan.__getitem__`), and `poses` is already present in every
episode cache (verified: `ep_00000.pt` keys are
`['frames_u8', 'actions', 'poses', 'episode_id', 'maneuvers']`). **No cache rewrite is
required and none should be scheduled.**

**And the pathway works.** On real driving poses the LAN route input is live on **69.2 %** of
windows against the **21–25 %** of the `nav_cmd` it replaces — a **~2.8×** improvement on the
instrument author's own pre-registered bar.

---

## 1. Evidence

### D-LAN-PF — what the preflight can and cannot see
`MEASURED` · code `code/probe_route_vs_lan.py` · raw `raw/route_vs_lan.json`
Pre-registered with both outcomes committed before the run (see the probe docstring).

| hypothesis | verdict | evidence |
|---|---|---|
| **H1** `route` is nav-derived and independent of `lan` | **SUPPORTED** | `route` = `0.0` on BOTH the plain and the lan-carrying path; `nav_valid = [false, false]`, `route_target = [3, 3]` (= `ROUTE_UNKNOWN`, deliberately outside the 3-class CE range) |
| **H2** `preflight()` structurally cannot exercise the lan pathway | **SUPPORTED** | preflight builds `V3Dataset` → `"lan" in batch` = **false**; the train path builds `LanV3Dataset` → **true**, shape `[2, 16]` (K=4 × 4 feats) |
| **H3** the lan pathway, when fed, yields a finite non-zero `goal_str` | **REFUTED** *(on the synthetic corpus)* | `goal_str` **is** present on the lan path — but its value is `0.0` and `lan_valid_frac` is `0.0`: **every anchor of every window is masked** |

Three CLI runs (`--preflight --arm hier --smoke`, ± `--goal-str`, ± `--graft-lan`) all printed
`'route': 0.0`, unchanged. That is H1 confirmed at the command line, not only in the probe.

### The root cause of H3 — and why it mattered
`_synth_episodes` builds `T = 40 + 8·e` frames at 2–8 m/s × 0.1 s ⇒ **~8–32 m of total path**.
The shortest LAN anchor sits at **20 m arc-length**, *beyond* a leak guard of ≈ 2 s × v + 5 m.
So the label is dead by construction on the CI corpus.

⚠️ **This is the dangerous shape, not a nuisance:** a `goal_str` computed over an all-invalid
label is a clean, finite `0.0`. It looks exactly like a healthy zero. A preflight that merely
asserted "`goal_str` is finite" would have gone green on a strategic head that trains on
nothing.

### D-LAN-COV — does the leak guard leave any signal on REAL data?
`MEASURED` · code `code/probe_lan_coverage.py` · raw `raw/lan_coverage.json`
⚠️ **NON-PARITY.** Cache `physicalai-train-14231cd29c74` (400 eps, 60 loaded, 8 000 windows,
seed 0) is **NOT** the sacred corpus `physicalai-train-e438721ae894` / skip-hash `f09e44db`.
This answers a *mechanism* question — "does the speed-driven guard mask everything on real
poses?" — which any real driving poses can answer. ⛔ **Never quotable as a cross-arm result.**

The bar is not mine. It is written into the instrument by its author, `refc_train.lan_stats`:

> *"The number that matters: `any_valid_frac`. The 4-way `nav_cmd` it replaces is valid on
> 0.21–0.25 of windows (MEASURED, all four arms) — if LAN is not materially higher, the input
> is not fixed and the experiment should not run."*

| quantity | measured |
|---|---|
| **`any_valid_frac`, shipped default** | **0.6924** |
| `nav_cmd` baseline it replaces | 0.21 – 0.25 |
| margin over the bar | **+0.4424** ⇒ **H4 SUPPORTED** |
| CONTROL — constant-only (`min_lead_m` → +inf) | **0.0** ⇒ the probe is **able to fail** |
| CONTROL — no-guard floor (`min_lead_m` = 0, `t_pred_s` = 0) | 0.754 |
| **cost of the leak guard** | **0.0616** (6.2 pp) — cheap |
| per-anchor valid frac (20 / 40 / 80 / **160 m**) | 0.449 / 0.613 / 0.339 / **0.012** |
| leak-guard lead: median / p90 / max | 17.29 m / 26.1 m / 39.26 m |
| windows where the lead exceeds the *shortest* anchor | 0.3485 |
| windows where the lead exceeds the *longest* anchor | 0.0 |
| ego speed mean / median | 5.965 / 6.201 m/s |

**H5 SUPPORTED**: the synthetic `0.0` was an artifact of the corpus, not a property of LAN.

⭐ **A finding worth a PI decision: the 160 m anchor is dead — 1.2 % of windows.** Episode
median length is 199 frames (~20 s) at ~6 m/s ⇒ ~120 m of available path, so the 160 m anchor
runs off the end of the episode almost always. **A quarter of the LAN feature width carries
almost no data**, and the strategic head's 160 m slot would train on 1.2 % of windows. The
arc-lengths `(20, 40, 80, 160)` were chosen without this measurement. See §4.

⚠️ Note a second, smaller inconsistency found in passing: the library default
`LAN_ARCLENGTHS_M` is `(20, 40, 80, 160)` but `refc_v3_train.py`'s argparse default is
`(10, 20, 40, 80)`. A run and a library call therefore disagree about what "default LAN" means
unless the flag is passed explicitly. Flagged, not silently changed — it is the trainer
owner's call which is canonical.

---

## 2. What was CHANGED (not merely reported)

### `stack/scripts/refc_v3_train.py`
1. **New `_lan_arm_preflight(cfg, args)`** — a real LAN arm, run when `--goal-str` or
   `--graft-lan` is passed on a `hier` preflight. Four checks in failure-diagnosable order:
   the dataset emits `lan` at the pinned width → **the LABEL is live (`any_valid_frac > 0`)**
   → `goal_str` is present, finite, and non-zero → a **deliberate-regression control** with
   the guard at +inf must drive the label dead and `goal_str` to exactly `0.0`. **The control
   runs every time and its own failure fails the preflight**, because a check that cannot fail
   is not evidence (programme §6.2).
2. **`_synth_episodes` gains `min_frames`** so the LAN arm can build a corpus long enough for
   the question to be answerable. The default path is untouched.
   ⚠️ The corpus length is derived from the **arc-lengths only, deliberately not from
   `min_lead_m`** — the control varies `min_lead_m`, and a corpus that grew with it would
   change two things at once and stop being a control. *(I got this wrong on the first
   attempt: the `1e9` control asked for ~5×10⁹ frames and hung. Caught by running it.)*
3. **stdout/stderr reconfigured to UTF-8 at import.** MEASURED: on the cp1252 Windows console
   the preflight passed every check and then **died with `UnicodeEncodeError` printing its own
   `✅ PASS` line** — a green gate exiting non-zero with a traceback. A preflight is meant to
   be run on the dev box before spending a GPU day; its success path must not crash.
   Established pattern, already used at `scripts/residual_scale_audit.py:226`.

### `stack/tests/test_refc_v3_lan_preflight.py` (new, 11 tests, 3.18 s)
Ships in the same change as the instrument (programme §6.5). It pins the defect itself
(`route` must stay unmoved by the LAN label; `goal_str` must be absent without `lan`), the
corpus floor (the default synthetic corpus **must** read dead — if that starts passing,
`_synth_episodes` changed and the floor must be re-derived rather than inherited), and **two
independent ways for the arm to fail**: the guard killing the label, and the anchors lying
beyond the corpus. Both must be refused.

---

## 3. Verification

```
pytest tests/test_refc_v3_lan_preflight.py -q                 ->  11 passed in 3.18s
pytest tests/ -k "lan or refc or v3 or route or goal" -q      -> 685 passed, 10 skipped,
                                                                 2 xfailed, 0 FAILED
```

⚠️ **The regression run initially showed 8 failures and they were NOT real.** Every one was a
`ModuleNotFoundError: taniteval` / `FileNotFoundError` from my own working copy — I had
mirrored `stack/` to a local disk (the `G:` mount fails reads mid-import with
`OSError: [Errno 22]`) without its sibling `taniteval/`. After copying `taniteval/` the same
three files went **63 passed / 0 failed**. Recorded because "8 tests fail" would have been a
false alarm, and the copy — not the change — was the cause. The 4 residual *collection*
errors are modules under `stack/experiments/`, which I deliberately excluded from the mirror.

```
refc_v3_train.py --preflight --arm hier --smoke --goal-str      # the acceptance command
  [v3-preflight] lan label coverage: {"n_sampled": 512, "arclengths_m": [10.0, 20.0, 40.0, 80.0],
                 "min_lead_m": 5.0, "per_anchor_valid_frac": [0.0, 0.959, 0.877, 0.7266],
                 "any_valid_frac": 0.959}
  [v3-preflight] lan arm OK: goal_str=0.4641 live / 0.0000 under the +inf-guard control
                 (control able to fail: True)
  [v3-preflight] ✅ PASS        rc=0
```

`--preflight` **without** the flags is byte-identical in shape to before (no LAN arm, same
checks) — verified by running both.

⚠️ **`route` is still `0.0`, and that is CORRECT.** It is the nav-derived CE and it is not the
LAN route. The acceptance criterion should read **`goal_str` non-zero**, not `route` non-zero.
Proposed replacement wording in §4.

---

## 4. For the PI — three decisions, each with the measurement behind it

1. **Amend acceptance criterion (d).** Replace *"`--preflight --goal-str` shows `route`
   non-zero"* with *"`--preflight --goal-str` reports a live LAN label
   (`any_valid_frac > 0`) and a finite non-zero `goal_str`, with the +inf-guard control
   reading `0.0`."* The old wording targets the nav echo; satisfying it literally would have
   re-opened C6.
2. **The 160 m LAN anchor is dead at 1.2 %** on real episodes (~120 m of available path per
   episode). Options: drop it, or re-space the anchors to the measured path-length
   distribution. This is a config change with cross-arm consequences, so it is yours, not
   mine. ⛔ I have changed nothing about the arc-lengths.
3. **Library vs trainer default arc-lengths disagree** (`(20,40,80,160)` vs `(10,20,40,80)`).
   One should be made canonical.

**No GPU time was spent, and none is needed for items (b) and (d).**

---

## 5. Status of P1 after this package

| item | state |
|---|---|
| (a) SAM3 backfill 115/115 by content | ✅ **DONE** — complete at `sam3_backfill_v2/`, 0 errors (separate package, `2026-08-23-sam3-completion-audit/`) |
| (b) `lan` minted onto the corpus | ✅ **DISSOLVED** — LAN is derived from `poses`, which every cache already has. No minting job exists to run. |
| (c) `s2_derive.py` labels banked with provenance tags | ⬜ open |
| (d) preflight shows the strategic goal signal | ✅ **DONE** — `goal_str = 0.4641` live / `0.0` under control; criterion wording needs the amendment in §4 |
| (e) Alpamayo meta-action mapping + coverage | ⬜ in flight (separate package) |

---

## 6. Deliverable manifest

⛔ Every row below was resolved with `git ls-files --cached` **after** staging — not typed from
intent (C78).

| artifact | path |
|---|---|
| this report | `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-23-lan-preflight-arm/RESULT.md` |
| D-LAN-PF probe | `.../2026-08-23-lan-preflight-arm/code/probe_route_vs_lan.py` |
| D-LAN-COV probe | `.../2026-08-23-lan-preflight-arm/code/probe_lan_coverage.py` |
| D-LAN-PF raw | `.../2026-08-23-lan-preflight-arm/raw/route_vs_lan.json` |
| D-LAN-COV raw | `.../2026-08-23-lan-preflight-arm/raw/lan_coverage.json` |
| the LAN preflight arm | `stack/scripts/refc_v3_train.py` |
| its regression tests | `stack/tests/test_refc_v3_lan_preflight.py` |

**Staged, never committed, never pushed.**
