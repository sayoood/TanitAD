# refcv5 is CODE-READY — and the last blocker was never the data, it was a weight that trains nothing

**Stream:** Architecture & Inference · **2026-09-05** · branch `agent/arch-inf-20260803`
**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refcv5-training-readiness/`
**Evidence class:** MEASURED (ours) unless stated; every number names its artifact.
**Pre-registration:** `SPEC.md`, banked **before** any arm ran (sha256 `9d7c17cfdeccc218…`).

---

## The headline, in five lines

| | |
|---|---|
| **P1a** | `agent_losses(cam=…)` now takes a **per-ROW camera**, and a `RigCameraBank` resolves it from the batch's own `agent_ep`. ⭐ **The per-clip table is BUILT: 2,400/2,400 parity clips, clip set verified against `parity_manifest.json`'s committed digest.** |
| **P1b** | ⛔⛔ **`--agent-w-ground` COMPUTES A TAUTOLOGY** — loss **2.61e-08**, gradient **8.73e-11**. A camera is built (M18 satisfied), the term runs, and it trains **nothing**. It now REFUSES, and the refusal is itself a measurement. |
| **P1c** | The camera-height band is MEASURED on the parity corpus for the first time: **1.2131–1.6672 m, 554 distinct values in 2,400 clips**. It refuted BOTH prior bands — and refuted a sentence *I wrote an hour earlier*. |
| **P2** | `stack/scripts/refcv5_preflight.py` runs: **14 PASS · 1 FAIL · 2 INCONCLUSIVE**. The old blocker *"no parity-corpus episode source"* is **REFUTED**: the complete 85.00 GB v2 cache is on HF, the join is on this box, md5 exact. |
| **P3** | `V-RC5-READY` **PASSES all 5 gates**: four deliberate regressions refused before a checkpoint, the converse control not refused, and the replicate arm **byte-identical** — so the refusals are evidence, not noise. |

⛔ **GO/NO-GO: NO-GO for a parity arm today**, on **two blockers that are neither code nor data**: no GPU, and v7.2 supervision covers **190/2,400 = 7.92 %** of parity against the trainer's own 50 % floor. Both are PI/DataFlyWheel items. §4 states them exactly.

---

## P1 — the per-clip camera, and the defect it uncovered

### P1a. The signature fix

`agent_losses(cam=…)` took **one** `RigCamera`. The corpus has a **per-CLIP** mount pose, so a run declared one mount pose for 2,400 clips. That is **retraction class C28** — *"a constant where the quantity is per-clip"* (`RETRACTION_LOG.md:332`) — recurring inside the very module that warns about the unit/scope family.

**What changed** (`stack/tanitad/refs/refc_agents.py`):

* `cam` accepts `None` · ONE `RigCamera` · **a per-row sequence of length B**. `row_cameras()` normalises the three and **REFUSES a partial list** rather than padding it — padding would attach clip *k*'s mount pose to row *k+1*, a corruption no downstream metric could attribute.
* `RigCameraBank` maps `stable_episode_id(clip_id) → RigCamera`, with `for_episodes()` and `coverage()`. ⛔ It is **refused inside the loss**: the loss has no episode ids, and inventing a lookup there is how a camera silently attaches to the wrong clip.
* The batch now carries **`agent_ep`**, selected with the **same mask** as the targets — a camera list built from the unselected ids would be the right length only by accident.
* Both monocular terms **COUNT** rows without a camera (`n_rows_no_cam`), and `agent_losses` reports `cam_scope ∈ {none, single, per-row}`.
* `config.json` stamps **`mount_pose_scope`** on every branch: `PER-CLIP` / `SINGLE-CAMERA-WHOLE-CORPUS` / `NONE`. A one-camera arm stays legal; what was missing was the **word** that distinguishes it.
* A per-clip table that does not cover the run's episodes **REFUSES** unless `--agent-rig-extrinsics-allow-partial` is passed by name, and the coverage is stamped either way.

**Controls, all green:** a list of B *identical* cameras reproduces the single-camera path to **< 1e-12** on both terms (so no banked arm shifts); two different mount heights give different losses and a **mixed** batch lands strictly between them (so `cams[b]` is really read); a bank reaching the loss unresolved raises.

### ⭐ P1a is not a stub — the table exists, for the whole parity corpus

`stack/scripts/build_rig_extrinsics_table.py` reads PhysicalAI-AV's own `calibration/sensor_extrinsics` and emits the per-clip table the trainer consumes.

```
[extr] clip_id_sha256_sorted read e61a04553df5b9d5 / manifest e61a04553df5b9d5 -> MATCH
[extr] wrote rig_extrinsics_train2400.json  2400/2400 clips
[extr] height  min 1.2131  median 1.2993  max 1.6672  (554 distinct)
[extr] forward 1.6969 - 2.1635 m   pitch -2.440 .. +3.945 deg
```

⛔ **The clip set is ASSERTED, not assumed.** The builder refuses to write unless the sorted clip-id sha256 equals `parity_manifest.json`'s committed `clip_id_sha256_sorted`. **It fired on its first run**: `r0_selection.parquet` holds 500 clips, not 2,400, and a table over those would have silently supplied cameras for a different corpus. The parity list came from the v2 cache's own 2,400 filenames and matched the committed digest exactly.

### P1c. The camera height, measured on the corpus that will be trained

| sample | span [m] | distinct | source |
|---|---|---|---|
| 12 clips, one `obstacle.offline` chunk | 1.43 – 1.56 | — | `D-V5A-GROUND1`, the shipped `CAM_HEIGHT_RANGE_M` |
| 3 clips, the banked render table | 1.2922 – 1.5758 | 3 | `…/refcv3_five_panel_step40284/extrinsics_used.json` |
| 40 clips (INHERITED) | 1.245 – 1.607 | 37 | `pai_extrinsics_table.py` / C28 |
| **2,400 clips — THE PARITY CORPUS** | **1.2131 – 1.6672** | **554** | **MEASURED here** |

Each earlier band strictly **contained** the previous one, and the audit predicted the next would widen again. It did, at **both** ends. `CAM_HEIGHT_RANGE_M` is now the union and `CAM_HEIGHT_SAMPLES` names every sample, so the next widening is a table row rather than a silent edit.

⚠️ **AND IT REFUTED A SENTENCE I WROTE AN HOUR EARLIER.** I had written *"1.22 m is below every observed minimum"* — true on 40 clips, **false on parity**, whose minimum is **1.2131 m**. 1.22 is inside the range. It remains wrong *as a constant* — 554 distinct values — but the stronger claim is **RETRACTED** in all three places it had reached. Same class as the band it corrected: a small-sample extremum quoted as a bound.

### P1b. ⛔⛔ The defect the per-clip work uncovered: a weight that trains nothing

While writing the known-value control for `ground_range_prior` I could not construct a case where it read non-zero. It cannot:

```
ROAD_PLANE_Z_M = 0.0
loss = 2.6144570952624235e-08    n = 119    frac_hit = 0.9297
grad absmax = 8.731149137020111e-11
```

`ground_range_prior` projects a slot's foot at rig **z = 0** and back-projects that pixel onto the plane **z = ROAD_PLANE_Z_M**, which **is 0.0**. `RigCamera.project` and `RigCamera.ground_intersection` are exact inverses, so `r_back == r_pred` **by construction**. There is no independent image-plane quantity for it to constrain, because `AgentSlotDecoder` emits no pixel — only BEV metres.

⇒ An arm passing `--agent-w-ground 0.5` **trains, converges, writes a checkpoint, stamps the weight into `config.json`, and adds exactly zero to the total.** It would afterwards read as *"the ground prior does not help."*

⭐ **This is the M18 dead-flag defect ONE LEVEL IN.** M18 fixed *"the camera is `None` so the term never runs"*. Here the flag is set, the camera **is** built, the term **does** run — and it is identically zero. **No flag guard and no camera guard can see that. Only a gradient probe can.**

**The fix, and why it is a measurement rather than a ban:** `assert_ground_prior_is_supervised(model, args)` runs at model setup, drives the term on synthetic boxes with **the run's own camera**, and refuses if the gradient is at or below `1e-7`. It **lifts itself** the day the term becomes real. ⚠️ It carries a **same-breath positive control** — `monocular_projection_loss` through the same tensors, which must read non-zero — because a flat gradient is otherwise indistinguishable from a probe that never ran; if the control is also flat the run stops as **INCONCLUSIVE**, never as a pass.

⇒ **ESCALATION:** to make the prior real, the decoder needs a **per-slot image-plane output** (a foot row) so the back-projection has something independent to constrain. That is `agent_slots.py` surgery plus a param-band change, i.e. a prereg and a Master Mind ruling — not a silent edit.

### P1d. The other constants that carried a val40-only justification

Audited across the whole refcv5 surface. The ones that still bind:

| constant | value | basis | status |
|---|---|---|---|
| `agent_slots.N_QUERIES_DEFAULT` | **16** | none — self-labelled placeholder | **REFUTED and still shipped.** The trainer passes 100 explicitly, but any `AgentSlotDecoder` built without `n_queries` inherits 16, and `config.json` records it as `n_queries_default_upstream`. |
| `SlotDecodeRanges` decode box | 60 m × ±16 m | "the P8 spec", no corpus | the **hard drop predicate** of `visible_target_filter`; the train fraction outside it is UNMEASURED |
| `FOV_HALF_ANGLE_RAD` | 60° | the rig's name | the encoder's own field measured **117°** on one sub-frame ⇒ this is an **upper bound**, and an over-wide cut admits unobservable targets |
| the 61.8 % / 80.1 % out-of-field fractions | — | **val40, at n_queries = 16** | the source itself notes the fraction RISES with N; at N = 100 on train it is unknown |
| `ROAD_PLANE_Z_M`, `ALL_CLASSES` | 0.0, 10 classes | **12 clips, one chunk** | the narrowest basis in the whole surface — 0.5 % of the corpus, and not a random sample |

⚠️ The narrowest basis is **not val40 — it is 12 clips**. `ROAD_PLANE_Z_M` is the one I would still expect to hold: it is a fact about where the rig origin is defined, with a genuine opposite-direction control (`protruding_object` at +1.68 m). The risk was never that the road plane moves; it was that a term multiplies plane-relative geometry by a mount height that moves **per clip by ±14 %** — which is exactly what P1a fixed and P1b found to be moot for the ground term.

---

## P2 — the preflight, and what it says

`stack/scripts/refcv5_preflight.py` — zero GPU, every check a POSITIVE assertion, **INCONCLUSIVE counted as a FAILURE**. Run against the real parity inputs (`raw/preflight_parity_inputs.json`):

| | check | reading |
|---|---|---|
| ✅ | refcv5 modules import | 12/12 |
| ✅ | class enum is the corpus enum | n=10 identical; control: `automobile` present |
| ✅ | query budget covers the train corpus | `--agent-queries` **100** ≥ train max 94 |
| ⛔ | **every weighted term has a gradient** | `agent_w_project` **1.462e-03 LIVE** · `agent_w_ground` **1.164e-10 DEAD** |
| ✅ | the seam guards refuse their defects | **4/4** fired; control (legal config) passed |
| ✅ | every knob reaches `config.json` | **18** knobs derived from argparse, 0 missing |
| ✅ | the camera scope is stated | `PER-CLIP`, **2,400** clips |
| ✅ | v2 cache present with stable ids | 96 local `*.v2ep.pt` (control: 98 dir entries), manifest present, parity key in path |
| ✅ | episode ids are collision-free | 96/96 distinct stable ids |
| ✅ | **the corpus clip list is the PARITY set** | **2,400 clips, sha256 `e61a04553df5…` MATCHES the committed manifest** |
| ✅ | agent join joins to the cache | **2,308/2,400 = 0.9617 on the STABLE id, 0 legacy-only**; 433,040 records |
| ✅ | join digest declares its scope | `md5 24cbdca8… scope=compressed`, `declared_by: backfill-MEASURED` |
| ⚠️ | v7.2 tactical labels cover the corpus | **INCONCLUSIVE at run time; MEASURED separately: 190/2,400 = 0.0792** vs the trainer's 0.50 floor |
| ✅ | per-clip cameras cover the corpus | **2,400/2,400 = 1.0000** |
| ⚠️ | anchor vocabulary declares its units | INCONCLUSIVE — **no `anchors.pt` on this box** |
| ✅ | a 2-step run states every weight | 18 knobs stamped, config read back from disk |

**14 PASS · 1 FAIL · 2 INCONCLUSIVE ⇒ NO-GO.**

### ⭐ The old blocker is REFUTED, and it was an absence-at-one-location error

*"No parity-corpus episode source on this box"* was true about the **box** and false about the **programme**:

| artifact | where | size | evidence |
|---|---|---|---|
| parity v2 cache, **complete** | HF `Sayood/tanitad-physicalai-w120-256x640cyl` → `physicalai-train-e438721ae894-w120-256x640cyl/` | **84,984,604,480 B = 84.98 GB**, 2,400 `*.v2ep.pt` | `HfApi().repo_info(files_metadata=True)`; `MODEL_REGISTRY.md:1067` records it PARITY VERIFIED |
| the agent join | dev box `C:/Users/Admin/tanitad-data/joins/joins/train2400_agents.jsonl.xz` | 136,689,648 B | md5 `24cbdca8c3b23aafc2fb17e6bf99cf76` **exact** |
| the per-clip camera table | built here, **banked in this package** | 606,399 B | 2,400/2,400, digest MATCH |
| parity v2 cache, local | dev box HF snapshot | 3.50 GB | **96 of 2,400 = 4.0 %** |

⚠️ **Priced on the CONSUMER'S loader**, `stack/tanitad/data/v2_dataset.py::V2CompressedCache` (globs `*.v2ep.pt`, reads `_v2manifest.pt`) — **85.00 GB**, *not* the 349.52 GB raw `epcache-256px-phase0/`. HF → pod ≈ **12 min** at the documented ~118 MB/s (INHERITED); HF → dev box **MEASURED at 7.7 MB/s ⇒ ~3.1 h**.

---

## P3 — `V-RC5-READY`: the validation, and it PASSES

`SPEC.md` was banked **before** any arm ran, with both outcomes committed. `stack/scripts/refcv5_validate.py` executes it (`raw/validate_V-RC5-READY.json`).

| arm | configuration | expected | got |
|---|---|---|---|
| A0 | baseline, seams off | reach | ✅ `summary.json` |
| **A0r** | **identical flags, identical seed, run again** | reach | ✅ `summary.json` |
| A1 | `--agents oracle` | reach | ✅ `summary.json` |
| **R1** | `--agent-rig-camera nominal --agent-w-ground 0.5` | ⛔ refuse | ✅ refused, no checkpoint |
| **R2** | `--agents head --w-agent 1.0`, no join | ⛔ refuse | ✅ refused |
| **R3** | `--agents off --w-agent 1.0` | ⛔ refuse | ✅ refused |
| **R4** | `--sampler ddim --w-u0 0` | ⛔ refuse | ✅ refused |
| **C1** | `--agent-rig-camera nominal --agent-w-project 0.2` | reach | ✅ reached — the guard is not over-broad |

| gate | result |
|---|---|
| **G1** every deliberate regression REFUSED before a checkpoint | **PASS** (4/4) |
| **G2** the converse control C1 is NOT refused | **PASS** |
| **G3** A0/A0r/A1 reach `summary.json` | **PASS** |
| **G4** the replicate control is **byte-identical** | **PASS** |
| **G5** the gradient probe separates DEAD from LIVE | **PASS** — `agent_w_ground` DEAD, `agent_w_project` LIVE |

⭐ **G4 is why G1 is admissible.** `H-ESTIM-SEED-1` says a separated CI from a one-seed arm is necessary and not sufficient, because the episode bootstrap is blind to training variance. The corresponding discipline here is that a refusal which fired once on a non-deterministic rig is not evidence it fires. A0 and A0r — same flags, same seed, run again — produced a **byte-identical `ckpt.pt`**, so this rig's run-to-run noise floor is exactly zero and every refusal above is a property of the configuration.

⭐ **R1 is not a synthetic regression. It is the live defect**: a configuration that satisfies every existing guard — flag set, camera built, term running — and supervises nothing.

---

## P4 — GO / NO-GO

| # | precondition | MEASURED | artifact |
|---|---|---|---|
| 1 | refcv5 modules import; trainer builds | ✅ 12/12 | `raw/preflight_parity_inputs.json` |
| 2 | seam guards refuse their defects | ✅ 4/4 + control | ibid. |
| 3 | every knob reaches `config.json` | ✅ 18/18 | ibid. + `raw/validate_V-RC5-READY.json` |
| 4 | deliberate regressions caught | ✅ 4/4, replicate byte-identical | `raw/validate_V-RC5-READY.json` |
| 5 | **every weighted term has a gradient** | ⛔ **`agent_w_ground` DEAD (1.16e-10)** — now REFUSED at startup | ibid. |
| 6 | query budget covers train (max 94) | ✅ 100 | `raw/preflight_parity_inputs.json` |
| 7 | class enum is the corpus enum | ✅ 10/10 | ibid. |
| 8 | parity clip set is the committed one | ✅ sha `e61a04553df5…` MATCH | `raw/parity_clip_ids.json` |
| 9 | per-clip cameras cover the corpus | ✅ **2,400/2,400** | `raw/rig_extrinsics_train2400.json` |
| 10 | agent join joins on the STABLE id | ✅ **2,308/2,400**, 0 legacy-only | `raw/preflight_parity_inputs.json` |
| 11 | join digest declares its scope | ✅ `scope=compressed`, MEASURED | ibid. |
| 12 | v2 episode source reachable | ✅ **85.00 GB on HF**, complete | ibid. + §P2 |
| 13 | v2 cache present **on a training box** | ⛔ **96/2,400 local (4.0 %)**; 85 GB must move | ibid. |
| 14 | **v7.2 tactical labels cover parity** | ⛔ **190/2,400 = 7.92 %** vs the trainer's **0.50** floor | measured this session, §P2 |
| 15 | anchor vocabulary with declared units | ⛔ **no `anchors.pt` on this box** | `raw/preflight_parity_inputs.json` |
| 16 | a GPU that is not already committed | ⛔ **none** — `tanitad-refcv3` is at 86 % util on refcv4b step 26,650/40,284; 5 legacy pods refuse connections; the 4060 is saturated; Thor is off-limits | §P2 |

### ⛔ **CAN refcv5 START A REAL ARM ON THE PARITY CORPUS TODAY? NO.**

**The code is ready and is now protected against the class of defect that would have made the arm unfalsifiable. What is missing is four things, none of them code, and each has an owner:**

1. **A GPU** — a newly provisioned pod. *(PI)*
2. **The 85.00 GB parity v2 cache moved onto it** — HF → pod, ≈ 12 min. *(Master Mind / any operator, once the pod exists)*
3. **The refcv4b anchor bank** (`anchors.pt` with declared `control_units`) — not on this box; a run without one silently falls back to the synthetic default whose oracle-in-vocabulary ADE is **1.0882 m** against **0.3796 m**. *(Arch+Inference / Master Mind — say which bank)*
4. **A parity-scoped v7.2 label build**, or a PI ruling to run the arm on the kinematic 3×3 vocabulary. The released v7.2 labels are **B1-scoped** (4,572 records, 190 of them in parity). ⚠️ The PI made v7.2 supervision MANDATORY for the refcv4b launch, so running without it is a **decision, not a workaround**. *(DataFlyWheel to build; PI to rule)*

**And one code item that is mine and is now closed rather than open:** `--agent-w-ground` is refused, with the measurement in the refusal, so no refcv5 arm can spend a GPU-day producing *"the ground prior does not help."*

---

## Deliverable manifest

All paths in the repo, branch `agent/arch-inf-20260803`, staged. Nothing lives only on a pod or a worktree.

| path | what |
|---|---|
| `stack/tanitad/refs/refc_agents.py` | `RigCameraBank`, `row_cameras`, per-row cameras in both monocular terms, `cam_scope` + row counts |
| `stack/tanitad/data/rig_projection.py` | `CAM_HEIGHT_SAMPLES` (every sample with its provenance), `CAM_HEIGHT_RANGE_M` = the parity union |
| `stack/scripts/refc_v3_train.py` | per-clip extrinsics table, `mount_pose_scope`, `agent_ep`, `_resolve_rig_cameras`, `assert_rig_camera_covers`, `assert_ground_prior_is_supervised`, `--agent-rig-extrinsics-allow-partial` |
| `stack/scripts/refcv5_preflight.py` | **NEW** — the runnable readiness checklist |
| `stack/scripts/refcv5_validate.py` | **NEW** — the `V-RC5-READY` arm runner and gates |
| `stack/scripts/build_rig_extrinsics_table.py` | **NEW** — the per-clip camera table builder, with the parity-digest refusal |
| `stack/tests/test_refc_agents_per_clip_camera.py` | **NEW** — 13 tests; the known-value control and the tautology pin |
| `stack/tests/test_refc_v3_per_clip_camera.py` | **NEW** — 12 tests; the three trainer-side refusals and their converse controls |
| `…/2026-09-05-refcv5-training-readiness/SPEC.md` | the pre-registration, banked before any arm ran |
| `…/raw/preflight_parity_inputs.json` | the preflight run against the real parity inputs |
| `…/raw/validate_V-RC5-READY.json` | the eight arms and the five gates |
| `…/raw/rig_extrinsics_train2400.json` | **the per-clip camera table, 2,400/2,400 clips** |
| `…/raw/rig_extrinsics_train2400.stats.json` | its distributions and the parity-digest match |
| `…/raw/parity_clip_ids.json` | the 2,400 parity clip ids, sha-verified against the manifest |

**Suite:** `264 passed, 4 skipped` across the twelve refcv5 / refc / rig-projection / agent-slot test modules.

---

## ⚠️ ESCALATIONS — for the Master Mind

1. ⛔ **`ground_range_prior` needs an image-plane output to become real.** `AgentSlotDecoder` emits no pixel, so the prior has nothing independent to constrain. A per-slot foot-row head is `agent_slots.py` surgery plus a param-band change ⇒ prereg + ruling.
2. ⛔ **`N_QUERIES_DEFAULT = 16` is refuted and still shipped — and the reason it cannot be fixed with one edit is now MEASURED.** The blast radius, by grep with a same-breath control:
   * `agent_slots.py:199` the constant; `:274` the `AgentSlotDecoder.__init__` default;
   * `v6.py:3966` `V6Config.n_slot_queries = N_QUERIES_DEFAULT`; `v6.py:4998` — the only production construction — passes `n_queries=cfg.n_slot_queries` **explicitly**;
   * ⛔ **`train_v6_staged.py:5313` carries a SECOND SPELLING: `getattr(a, "n_slot_queries", 16)` — a hardcoded 16 that does not read `N_QUERIES_DEFAULT` at all.** Correcting the constant would leave the live v6 trainer at 16 while every audit reported the new value. That is the `advect` precedent — two implementations of one number — and only grepping the literal catches it.
   * ✅ **refcv5 is NOT exposed**: `refc_agents.build_agent_head` always passes `n_queries=cfg.queries` (100).
   ⇒ The fix is a **two-site** change touching the live v6 trainer, which is why it is escalated rather than done here. `refcv5_preflight.py` now FAILS on the duplicate spelling, so it cannot be forgotten.
3. ⛔ **v7.2 supervision covers 7.92 % of parity.** Either a parity-scoped label build, or a PI ruling that the refcv5 parity arm runs on the kinematic 3×3 vocabulary. It cannot be decided inside this stream.
4. ✅ **CLOSED BY MEASUREMENT, not escalated.** The out-of-field and decode-box fractions were val40-scale; they are now read on the whole **2,308-clip train join (12,122,129 boxes)** from the banked `…/2026-09-05-agent-join-into-batch/raw/train_agent_density.json`: in-field **4,977,314 = 41.06 % ⇒ 58.94 % OUT OF FIELD**, in-field ∩ decode box **1,899,481 = 15.67 % ⇒ 84.33 % OUTSIDE THE DECODE BOX** (val40, among kept targets at N = 16, read 61.8 % and 80.1 %). ⚠️ The denominators differ — val40's is *targets kept by* `match_slots`, train's is *every box in the join* — but the conclusion holds in both and the decode-box cut is **more** severe on train, so the filter's default-ON is now a train-corpus fact. Written into `refc_agents.filter_targets_to_visible`'s docstring beside the val40 numbers rather than replacing them.
   ⚠️ **What remains open is narrower than the original escalation**: the decode box itself (60 m × ±16 m) still rests on "the P8 spec" with no corpus behind it, and the **nearest sacrificed range** at that cut is unmeasured — the same number that refuted `--agent-queries 32` at 13.1 m.
5. ⚠️ **The live flagship `refcv4b-b1-v72-40k` trains off-parity** — B1 overlaps the parity corpus by **7.92 %**, by design of the B1 selection. Consistent with `parity.py`, but worth stating in one place, because a refcv5 parity arm and refcv4b are then **not** on the same corpus.
6. ⚠️ **`stack/scripts/dinov3_fp8_encode_ship.py` writes a corpus artifact and
   calls neither `parity.guard_corpus_build` nor carries a stated reason.**
   `test_build_parity_guard.py::test_every_derived_corpus_writer_is_gated_or_classified`
   names it, and it is **pre-existing and unrelated to this change** (it is not
   in this commit and none of the tests reference any file touched here). A
   corpus that becomes supervision must be checked against the deployed val
   BEFORE it is built (`parity.py` §10c, RETRACTION_LOG C112/C113). Not this
   stream's file → escalated rather than edited.

---

## Suite verdict, stated with its scope

`1319 passed, 31 skipped` over the whole affected slice
(`-k "refc or agent or rig or anchor or calib or bev or v4 or kinematic or parity"`),
plus `207 passed` on a clean re-run of the nine directly touched modules.

**Five non-passes, and NONE is caused by this change** — each classified rather
than waved away:

| non-pass | class | evidence |
|---|---|---|
| `test_bev_consumer_fov.py::…launch_chains`, `test_eval_contamination.py` (12 items) | **mirror-absence artifact** — both need files under `TanitAD Research Lab/` which the off-Drive mirror (`stack/` + `taniteval/` only) does not carry | `git cat-file -e HEAD:…/parity_ls.txt` reads **PRESENT**, with a passing control on `CLAUDE.md` |
| `test_build_parity_guard.py::…gated_or_classified` | **pre-existing, unrelated** — `dinov3_fp8_encode_ship.py` is an unclassified corpus writer | not in this commit (`git show --stat HEAD \| grep -c dinov3` = **0**); the test references none of the files touched here |
| `test_guard_mutation_audit.py` (3 items) | **transient contention** — a concurrent `guard_mutation_audit.py` run from another stream left `.guard_mutation_backup` in the tree mid-run; the directory does not exist now and all three pass cleanly | re-run in isolation: **31 passed** |
| 3 refcv5 modules, in one interleaved run | **my own race** — I rewrote `refc_agents.py` while a background pytest was importing it (the "never edit a script while it runs" family) | each passes in isolation; the clean re-run reads 207 passed |

⚠️ Reported this way on purpose: *"the suite is green apart from some
unrelated failures"* is exactly the sentence that hides a real regression. Each
of the five is named, classified, and given the positive assertion that places
it outside this change.
