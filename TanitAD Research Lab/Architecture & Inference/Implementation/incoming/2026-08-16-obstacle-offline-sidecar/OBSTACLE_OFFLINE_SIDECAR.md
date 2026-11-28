# `obstacle.offline` reachability — the last mile, and a stale premise corrected

**Date:** 2026-08-16 · **Agent:** arch-inf · **Branch:** `agent/arch-inf-20260803` · **CPU-only**

---

## 0. Headline

**The briefed task was already built.** `obstacle.offline` is reachable, the sidecar exists, the
LONGITUDINAL distance-keeping scorer is wired, and it was measured on the canonical val40 sixteen
days ago (commit `670f614`). Building a second sidecar would have been duplicate work and a
**second implementation that can drift** — the exact failure `build_lead_block.py`'s own docstring
warns against.

**What was NOT reachable, and now is:** the only lead block banked in this repo is an `.npz`, and
the canonical eval tool loaded `--lead` with `torch.load`, which **raises** on it. The data was
present, correct, and banked; the reader could not open it. That is fixed, tested, and verified
end-to-end.

| | evidence class |
|---|---|
| Pipeline already complete (schema, sidecar, scorer, tests, measured numbers) | **MEASURED** (this doc §2) |
| `.npz` lead block unreadable by `eval_four_families.py --lead` | **MEASURED** (§3) |
| Fix reproduces `670f614`'s published headline exactly | **MEASURED** (§4) |
| Our ingest reads **5** of 36 features, not 4 | **MEASURED** (§5) — corrects `CLAUDE.md` |

---

## 1. What I was asked to build vs. what exists

| brief item | status | where it already lives |
|---|---|---|
| 1. Schema from a real sample | **DONE 2026-08-03** | `…/Data Engineering/…/2026-08-03-obstacle-offline-join/raw/obstacle_schema_probe.json` |
| 2. Sidecar extractor | **DONE** | `stack/scripts/build_obstacle_join.py` (759 lines) |
| 3. Lead-agent rule + per-frame gap/time-gap/TTC | **DONE** | `taniteval/taniteval/lead_source.py`, `lead_metrics.py` |
| 4. LONGITUDINAL scorer wiring | **DONE, with a hole** → **FIXED HERE** | `taniteval/taniteval/four_families.py::_distance_keeping` |
| 5. Per-state `n` + reason, never silently dropped | **DONE** | three states `LEAD`/`NO_LEAD`/`NO_LABEL` |
| 6. Tests | **DONE (61 pre-existing)** + **5 added here** | see §6 |

⚠️ **The brief's paths were stale.** `taniteval` is at **repo root** (`taniteval/taniteval/`), not
`stack/taniteval/`. `physicalai_r0.py` is at `stack/scripts/`, not `stack/tanitad/data/`. The
feature constants are at `physicalai.py:233-235`, not `:153-154` (those lines are a comment).

---

## 2. The schema, MEASURED from real gated bytes

Source: `raw/obstacle_schema_probe.json`, `_root = C:\Users\Admin\tanitad-data\physicalai`,
6 chunks / 12 clips. The gated corpus **is local**: 57 `obstacle.offline` chunk zips (2,385.5 MB,
**5,317 distinct clip_ids**) + 197 `egomotion` zips.

**Columns (16):** `timestamp_us · source · track_id · center_{x,y,z} · size_{x,y,z} ·
orientation_{x,y,z,w} · label_class · reference_frame · reference_frame_timestamp_us`

* `reference_frame` = **`rig`** on every probed clip; `reference_frame_timestamp_us == timestamp_us`
  on every probed clip (no second clock to reconcile).
* **Span ~20 s** (19.932–20.000 s) while `egomotion` runs 20–140 s ⇒ **most of a long clip is
  `NO_LABEL`, not empty road.** This is why the three-state split is load-bearing.
* Track cadence 0.1 s; median track life 1.35 s; median 14 samples/track.
* Rig frame is **x-forward / y-left** — *MEASURED, not assumed*: of 2,778 tracks living ≥2 s,
  **1,756 (63.2 %) are world-static** under x-fwd/y-left vs 236 under the mirrored lateral
  (`raw/frame_convention.json`). A parked car is only parked under the right handedness.

### Coverage — COUNTED FROM RECORDS

From the dataset's **own** presence table (`metadata/feature_presence.parquet`, 306,152 rows,
37 columns ⇒ **36 features** + `clip_id`):

| scope | n | with `obstacle.offline` | frac |
|---|---|---|---|
| corpus-wide | 306,152 | 298,326 | **0.9744** |
| phase-0 selection | 3,000 | 2,907 | 0.9690 |
| r0 selection | 500 | 495 | 0.9900 |
| **canonical val40** | 40 | **39** | **0.9750** |

⇒ The brief's **97.44 %** is **CONFIRMED** as the corpus-wide figure. A second probe over
locally-held chunks gives 614/636 = 0.9654 — *a different denominator, not a contradiction.*

### Class enum — one nuance worth flagging

My probe counted **9 classes / 66,883 cuboids**; the brief (and `CLAUDE.md`) say **10 classes /
87,481 cuboids**. These reconcile: the 10-class figure is
`…/2026-07-26-physicalai-feature-probe/` over a different 12 clips, and the 10th class
`other_vehicle` is genuinely **rare (0.24 %,** `DATA_STRATEGY_FOR_HIERARCHY.md:133`**)** — absent
from my sample, present at two other probes. **`VEHICLE_CLASSES` including `other_vehicle` is
correct; I nearly logged a false retraction and the second probe stopped it.**

⚠️ **Low-confidence observation, NOT a retraction:** the enum is described as *"10 classes, **all
dynamic agents**"*, but `protruding_object` (187 cuboids in my sample) is not obviously a dynamic
agent. It is **excluded** from `VEHICLE_CLASSES` either way, so no measured number moves. Flagged
for whoever owns that sentence; I did not re-probe it.

---

## 3. THE DEFECT — the artifact existed and the reader could not open it

Two builders emit lead blocks and **they do not agree on the container**:

| builder | container | call |
|---|---|---|
| `taniteval/tools/build_lead_block.py:275` | `.pt` | `torch.save` |
| `…/2026-08-04-distance-keeping-arms/code/build_val40_lead_block.py` | **`.npz`** | `np.savez` |

The consumer, `taniteval/tools/eval_four_families.py`, used `torch.load` unconditionally. The
**only lead block banked in this repo** is the `.npz`. MEASURED 2026-08-16:

```
torch.load('…/raw/val40_lead_block.npz')
  -> RuntimeError: [enforce fail at inline_container.cc:180]
     file in archive is not in a subdirectory: leads.npy
```

**It is a hard stop, not a degraded read** — so the distance-keeping half of LONGITUDINAL stayed
`UNAVAILABLE` for anyone using the canonical CLI, *despite the data being present and correct*.
Same family as the Vulkan-ICD and `ps -C python3` traps: **absence reported by a probe that was
looking in the wrong shape.**

**Second, latent, defect:** the tool assembled `win["lead"]` without `path_steps`/`dt_s`. The
scorer's own ⛔ comment explains that a coarse lead track must be **time-joined** onto a dense
path, never truncated. Dropping those keys reverts the join to a bare shape match. It does not
bite today only because the banked block (`ts_rel_s = 0.5/1.0/1.5/2.0`) and the banked dumps
(`wp_steps = [5,10,15,20]`) happen to be **both 4 wide** — MEASURED, `pred (881,4,2)` vs
`leads (881,4,2)`. It would bite the first time a dense dump is scored.

---

## 4. The fix, and end-to-end verification

`taniteval/tools/eval_four_families.py`:
* **`load_lead_block(path)`** — dispatches on extension, returns a **plain `dict`** (never a lazy
  `NpzFile`, whose `.get` semantics differ from `dict`'s — exactly how an optional join key would
  go missing without an error), and **exits loudly** on a block missing any of
  `leads/lead_lens/speeds/state/eid`.
* **Time-join pass-through** of `path_steps`/`dt_s` when the block declares them.
* The attach line now prints the state histogram **counted from the records** instead of a
  `counts` field the `.npz` does not carry (it printed `counts=None`).

**End-to-end, CPU-only, through the fixed loader** — real banked `.npz` + real banked window dump:

```
LOADED real banked block: dict, 881 rows
states: {'LEAD': 270, 'NO_LABEL': 60, 'NO_LEAD': 551}

distance-keeping (flagship-30k, val40, T0 windows):
  status OK   n 228 / 881
  mean_headway_min_m   30.5717
  mean_time_gap_min_s   3.1299   n_time_gap 223
  mean_min_ttc_s       24.7384   n_closing  115
```

✅ **Reproduces commit `670f614`'s published headline exactly** — 30.57 m mean headway, lead kept
in **228/270** windows. Window states match the pre-registration (270/550/61 registered → 270/551/60
measured).

⚠️ **Quote `n_closing` beside `mean_min_ttc_s`, never the mean alone**: 113 of 228 windows never
close on the lead and are **censored at 30 s**. The instrument says so itself in `censoring_note`.

### Where the metric genuinely cannot be computed — reported, never dropped

| state | n | why |
|---|---|---|
| `LEAD` | **270** | causal in-corridor vehicle ahead — scorable |
| `NO_LEAD` | **551** | labels present, road genuinely clear |
| `NO_LABEL` | **60** | no `obstacle.offline` for the clip (1 of 40 val clips), **or** `t0` outside the ~20 s labelled span |
| scored | **228** | of the 270 `LEAD` windows, the arm retains the lead in 228 |

⛔ Collapsing `NO_LABEL` into `NO_LEAD` would manufacture free-flow and flatter every arm. The
instrument keeps all three.

---

## 5. `CLAUDE.md` correction — the ingest reads **5** of 36, not 4

**MEASURED by content** (`stack/tanitad/data/physicalai.py`):

| feature | where |
|---|---|
| `egomotion` | `scripts/physicalai_r0.py:36` |
| `camera_front_wide_120fov` | `scripts/physicalai_r0.py:37-38` |
| `camera_intrinsics` | `physicalai.py:233` |
| `sensor_extrinsics` | `physicalai.py:234` |
| **`vehicle_dimensions`** | **`physicalai.py:235`**, read at `:359-386` (per-clip wheelbase) |

`CLAUDE.md` currently says *"our ingest reads **4** of 36 features"*. `vehicle_dimensions` was
added for the per-clip wheelbase and makes it **5**. This is the **third** revision of that same
sentence (2 → 4 → 5) — and `CLAUDE.md` already flags the previous one as *"a stale absence-claim
living inside the rule that warns about stale absence-claims."* **Root-cause class: a COUNT
embedded in prose, with no test pinning it to the source.**

---

## 6. Tests

Appended to `taniteval/tests/test_eval_four_families_tool.py` (13 → **18**), all passing:

| test | pins |
|---|---|
| `test_lead_block_loads_from_npz_the_banked_container` | the banked container loads; result is a real `dict` |
| `test_lead_block_loads_from_pt_the_builder_container` | the `torch.save` path keeps working |
| `test_torch_load_really_cannot_read_the_npz_container` | **the root cause**, so nobody "simplifies" back to `torch.load` |
| `test_lead_block_missing_a_required_field_exits_loudly` | a partial block is refused, never scored into a plausible number |
| `test_time_join_keys_are_carried_when_the_block_declares_them` | `path_steps`/`dt_s` survive the attach |

**Suite results (this box, CPU):**
* `taniteval/`: **1023 passed** (2:01)
* `stack/`: **2897 passed, 3 failed, 17 skipped, 2 xfailed** (5:37)

⚠️ **The 3 `stack` failures are NOT mine and NOT pre-existing-on-`HEAD`.** They are in
`stack/tests/test_e_wc2_sigma_star.py`, a file that is **staged (`A`) but never committed** — a
concurrent agent's in-flight work. I touched nothing under `stack/`. My entire diff is **138
insertions across 2 files**, both under `taniteval/`. The brief's stated baseline (2,810 passed) is
also stale; the suite has grown to 2,900.

---

## 7. Parity

**Untouched.** No episode re-selection, no change to the parity-bound build, no new cache. The
sidecar design was already the correct one: a **separate artifact keyed by episode id**, joined
positionally against an already-scored window dump, with a **refuse-never-truncate** guard on row
count. Train caches are byte-identical. `physicalai-train-e438721ae894` / skip-hash `f09e44db`
unaffected.

---

## 8. Deliverable manifest

| artifact | repo path | state |
|---|---|---|
| Lead-block loader + time-join pass-through | `taniteval/tools/eval_four_families.py` | **staged** |
| 5 new tests | `taniteval/tests/test_eval_four_families_tool.py` | **staged** |
| This writeup | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-obstacle-offline-sidecar/OBSTACLE_OFFLINE_SIDECAR.md` | **staged** |

**Not committed, not pushed** (per the operating standard).

## 9. Escalations

1. ⚠️ **`CLAUDE.md` "4 of 36" → 5 of 36** (§5). I did not edit `CLAUDE.md` — it is program-steering
   prose owned by the orchestrator. **Recommend a test that pins the count to the source** so the
   sentence cannot go stale a fourth time.
2. ⚠️ **3 failing tests in another agent's staged `stack/tests/test_e_wc2_sigma_star.py`** — that
   agent should be told before anything commits the index, since a pathspec-free commit would sweep
   a red test into history.
3. ⚠️ **Two lead-block containers is the real root cause.** The loader now tolerates both, but the
   durable fix is for `build_val40_lead_block.py` to emit the same container as
   `build_lead_block.py`. Filed as an observation, not done here — it lives in Benchmarks & Eval.
4. ⚠️ **This task was commissioned as duplicate work** because the completed state was recorded in
   commit messages and hub packages under **Data Engineering**, while the brief searched
   Architecture & Inference. Cost: one agent-session. **The `2026-08-03-obstacle-offline-join`
   package is under Data Engineering; the A&I-side `INTAKE.md` still names the wiring as open** even
   though it landed — that stale INTAKE is what the brief was written from.

   **Verified, `…/2026-08-03-longitudinal-distance-keeping/INTAKE.md:73-75`:** *"Wiring it into the
   *eval* path (val40 windows → `win["lead"]`) needs the `obstacle.offline` chunks for the 40 val
   episodes on the eval host. **Until that lands, arm evals will still report the family
   UNAVAILABLE**"*. That landed on **2026-08-03** (`lead_source.py`) and was **measured on
   2026-08-14** (`670f614`). ⇒ **`INTAKE.md` should be marked CLOSED.** Root-cause class: *an
   INTAKE that states a blocker is never revisited when the blocker clears* — the same class as
   the orthogonality instrument that sat unmerged 10 days because the request lived in a README
   nobody re-read.
