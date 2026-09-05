# Four record corrections — refcv4b registration, queued cost-scale rows, the `a = 0` grid retraction, the unit-less anchor artifact

**STATUS: COMPLETE — 2026-09-05 (Arch+Inference FlyWheel, zero GPU). Four items, four commits, every path blob-verified and content-asserted in `HEAD` at the end of the turn.**

| # | item | status | commit | content check (`git show HEAD:<path>`) |
|---|---|---|---|---|
| 1 | register `refcv4b-b1-v72-40k` in `MODEL_REGISTRY.md` (TRAINING row, MODEL-FREE gate figure) | ✅ landed | `8cb69ac` | `### 4.6 REF-C **v4b**` ×1; blob index = worktree |
| 2 | apply the queued register rows (`D-REFAV1-COST-SCALE`, `D-REFAV1-COST-FORM`) + the `D-REFAV1-EPOCH-PLAN-VOID` mechanism amendment + RETRACTION_LOG #21 | ✅ landed | `7fce999` | rows ×2/×1, `# 2026-09-04 (#21)` ×1, registry §2.4 amendment ×1, PREREG note ×1; 4/4 blobs verified; TURNCOV1 / FLYLOW1 / DDAUDIT-1 / NAV-WIRING survived (1/1/2/1) |
| 3 | retract *"every grid contains `a = 0` exactly"* (`2026-09-04-refcv4-gate-validation/RESULT.md` §3.5) + downstream fixes, RETRACTION_LOG #22 | ✅ landed | `8ee4549` | `# 2026-09-05 (#22)` ×1, RESULT.md correction ×1, KINVOCAB1 qualifier ×1, probe annotation ×1; 4/4 blobs verified; `KINVOCAB_PROBE.json` untouched (md5 `bae65bb401a07607cc0a5406b3e8339e`) |
| 4 | self-describing anchor artifact: builders write units/provenance, loader REQUIRES `control_units`, trainer stamps the seam booleans, 15 tests, CLAUDE.md trap | ✅ landed | the commit that carries this file (id in the report) | markers asserted in HEAD for all 7 paths (see the manifest) |

Predecessor on this task died twice at the API limit before landing anything; hence the per-item banking.

⚠️ **Mount log (MEASURED):** the G: mount went into its hard-failure mode (every content read
`Invalid request code` / `Errno 22`, `.git` unreadable, listings fine, a local control file readable in
the same breath) at ~04:05 local; the Drive client was restarted at 04:17:47 under the PI's standing
authorisation (memory `drive-client-restart-authorized`) and the mount returned within ~2 min; it wedged
again at ~04:45 and a second restart brought it back in 5 s. Everything was AUTHORED ON LOCAL DISK
(`…/scratchpad/`, the off-Drive clone `C:\Users\Admin\refcv4b_suite\stack`) and applied with an
exact-anchor script that reads the live file immediately before editing and READS BACK every write
(`apply_docs.py`, `patch_item4.py`).

## Inputs read (all MEASURED from the worktree unless marked)

* `C:\Users\Admin\refcv4b_viz\config.json` — md5 `58ad809aaf241729c1695bec2ab30d8e` (the brief states it is md5-identical to the pod's; INHERITED, not re-verified against the pod from here).
* `C:\Users\Admin\navcomp\ckpt\anchors.pt` — md5 `297f6f1db52f6a56094846b0d7f71ed9`, 10,597 B; sha256 per its sidecar `e86cf507…e8fb`, equal to `config.json['anchors']['file_sha256']`. ⛔ Not modified.
* `C:\Users\Admin\navcomp\ckpt\anchors.units.json` + `anchors.build.json` — the sidecars written 2026-09-04 (units record + full builder record).
* `2026-09-04-refcv4b-vocabulary/raw/refc_anchors_6s_v0cond_alat_117.pt.json` — the held-out gate: `oracle_in_vocabulary_ade_0_2s_m 0.1987`, `along_mae_m 0.1432`, `lat_mae_m 0.1037`, `paired_delta_vs_ha [-0.1009, -0.1213, -0.0813]`, `bar_ha_m 0.2996`, 4,823 windows / 141 episodes.
* `2026-09-04-refcv4-gate-validation/raw/scripts/kinvocab_probe.py:102` — `a_g = np.linspace(-4.0, 3.0, na)`; `:92-95` the comment asserting every grid holds `a = 0`; script md5 before the annotation `7119d4cffbad61ec8d17ce1bb0a85c25`.
* MEASURED 2026-09-05 (numpy): `np.linspace(-4, 3, n)` contains `0.0` for **none** of n ∈ {7, 9, 11, 13, 17} (nearest values −0.5, +0.375, +0.2, +0.0833, −0.0625); `np.linspace(-0.06, 0.06, n)` contains `0.0` for every odd n. Only the κ axis was symmetric-and-odd.
* `Project Steering/PREREG_REFC_V4.md` §A10.2 already carried a *"CORRECTION TO THE PRIOR PASS"* naming the accel-axis defect on 2026-09-04 — never logged in `RETRACTION_LOG.md`, never fixed at the origin, and the trainer's refusal text still prescribed "odd counts".
* `2026-09-04-refav1-cost-scale/PROPOSED_REGISTER_ROWS.md` (tracked; its worktree copy never hydrates on this machine — read via `git show HEAD:<path>`): §1 `D-REFAV1-COST-SCALE`, §2 `D-REFAV1-COST-FORM`, §3 the `D-REFAV1-EPOCH-PLAN-VOID` amendment, §4 the retraction entry.
* `2026-09-04-refcv4b-seam-state/SEAM_STATE.md` — the seam booleans absent from `config.json` at every nesting level, and the exact `anchors.pt` key set.

## ⚠️ Escalation — item 2's source file was not the one the brief named

The brief pointed item 2 at `2026-09-03-cost-repair/PROPOSED_REGISTER_ROWS.md` (commit `7555b70`) and
described it as *"two new rows, the amendment to `D-REFAV1-EPOCH-PLAN-VOID`'s mechanism sentence, its
RETRACTION_LOG entry"*. That file carries FOUR rows (`D-COST-CHORD`, `H-COST-WEIGHTS-1`,
`D-COST-ARGMIN-MOVES`, `D-COST-SURFACE-REPRODUCED`), a §5 prediction-retraction and §6 qualifiers, and
**no** EPOCH-PLAN-VOID amendment; `D-COST-CHORD` is already in the register in the Master Mind's own
framing (*"FAILS its own pre-registered criterion"*, with BACKLOG R41 struck-with-FAIL, R55, R56), and
its three siblings carry verdicts (`H-COST-WEIGHTS-1` *"SUPPORTED at T0"*) that framing overruled.
The file whose contents match the brief's description EXACTLY is the cost-**scale** one
(`2026-09-04-refav1-cost-scale/PROPOSED_REGISTER_ROWS.md`: two rows + §3 amendment + §4 retraction).
**That is the one applied.** The cost-repair siblings were NOT applied — if the Master Mind does want
`D-COST-ARGMIN-MOVES` / `D-COST-SURFACE-REPRODUCED` / the §5 prediction-retraction in the register,
they are one `apply_docs.py`-style insertion away; `H-COST-WEIGHTS-1` should not go in as written.

## Item 1 — `refcv4b-b1-v72-40k` registered (commit `8cb69ac`)

`MODEL_REGISTRY.md` §4.6 (inserted before §5 P2): lineage (refcv4 ABORTED at 6,400; the flat-κ refcv4b
STOPPED at 200), the five levers vs refcv3, params **107,058,488** (config.json), horizons, the B1
non-parity corpus, full argv, ORACLE nav from step 0 with the train/eval distributions, goal provenance,
the anchor vocabulary BY CONTENT (file sha256 `e86cf507…e8fb`, anchors `51f930dc…`, controls
`b072f4c0…`, `control_units alat`, a_lon re-centred with 0.0 at index 7), the **MODEL-FREE** gate
(0.1987 / 0.1432 / 0.1037, −0.1009 [−0.1213, −0.0813] vs `ha`, n = 4,823 / 141, paired
episode-cluster bootstrap) stamped per `GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md`, the flyability finding
(`D-REFCV4B-FLYLOW1`: ~16.27 % of the fan undrivable at the window's own v0; oracle-best anchor over
μ = 0.7 on 15/4,823 = 0.31 %), turn coverage, pace (3.844 s/step, the regression refuted), location,
status (no checkpoint evaluated; first model-inclusive gate = `oracle_sel` from a pulled checkpoint),
and four caveats (attribution forfeited; seam booleans absent from ITS config.json; the unit-less
`anchors.pt`; the fixed-10 m/s withheld-row roll). Nothing in the row is a driving result.

## Item 2 — cost-scale rows, amendment, retraction #21 (commit `7fce999`)

* `GOALS_AND_CLAIMS.md`: `D-REFAV1-COST-SCALE` and `D-REFAV1-COST-FORM` inserted directly after
  `D-REFAV1-EPOCH-HEADS` (4-column form of the proposed text, each stamped "applied 2026-09-05 from the
  package's PROPOSED_REGISTER_ROWS.md"); `D-REFAV1-EPOCH-PLAN-VOID`'s mechanism clause AMENDED IN PLACE
  with the original wording quoted inside the amendment (1.63e-10 was the incumbent-checkpoint figure —
  step 21,109 reads median 1.99e-06; float32 costs resolution, not the signal; the −2.00 ulp mislabel
  mechanism; jerk is the binding penalty; the 149/149 scope). The verdict stands.
* `MODEL_REGISTRY.md` §2.4: the same clause amended so the ONLY quotable source agrees with the register.
* `PREREG_TACTICAL_DECODER.md`: the registered prediction left as written; a dated note under it corrects
  its premise (a pre-registration's text is history; only its premise is annotated).
* `RETRACTION_LOG.md` #21 (dated 2026-09-04 as proposed, applied 2026-09-05): class *"a true measurement
  quoted outside its scope — the object being a CHECKPOINT"*, plus the population half.

## Item 3 — retraction #22 and its downstream fixes (commit `8ee4549`)

* `RETRACTION_LOG.md` #22 — class: **a property asserted of a family of grids from inspecting one
  member, about the ZERO element every no-action control depends on**; the extracted rule ("odd counts")
  named a proxy (parity) instead of the property (0.0 is a node).
* `…/2026-09-04-refcv4-gate-validation/RESULT.md` §3.5 — dated correction block (original sentence kept
  visible above it).
* `GOALS_AND_CLAIMS.md` `D-REFCV4-KINVOCAB1` — qualifier: 0.2572 / 0.2610 are numbers for grids WITHOUT
  the constant-velocity control; the direction stands (re-centred 13 × 9 → 0.2610, live alat 117 → 0.1987).
* `raw/scripts/kinvocab_probe.py:92` — comment-only annotation above the false comment (md5 before
  `7119d4cffbad61ec8d17ce1bb0a85c25`); `KINVOCAB_PROBE.json` untouched.
* `refc_v3_train.py`'s *"Rebuild with odd counts"* refusal → corrected in the item-4 commit (one file, one
  commit); pinned by `test_grid_refusal_no_longer_prescribes_odd_counts`.
* Checked and left alone: `build_kinvocab6s.py:100-101` and `emit_anchors_alat.py` assert presence by
  content (`np.any(grid == 0.0)`); `patch_trainer.py:124` is the historical patch that carried the wording
  into the trainer — a banked raw script, cited in the entry rather than rewritten.

## Item 4 — the anchor artifact carries its own units (the commit that carries this file)

| file | change |
|---|---|
| `stack/tanitad/refs/anchor_meta.py` (NEW) | `SCHEMA = "tanitad.anchor_artifact/1"`; `build_anchor_artifact(anchors, controls, *, control_units, horizons, dt, ref_speed_ms, kappa_cap, alat_v_floor, builder, extra)` writes `control_units`, `horizon_s` (DERIVED = max(horizons)·dt, slot count asserted), `dt`, `ref_speed_ms`, `kappa_cap`, `alat_v_floor`, sha256s and a provenance stamp (builder basename/path/sha256, argv, UTC time, torch version) INTO the `.pt`; `read_anchor_artifact(path, cli_control_units=None)` RESOLVES units (file only → `file`; file+cli equal → `file+cli`; legacy + cli → `cli-override-legacy-file`; legacy alone → `AnchorUnitsMissing`, whose message quotes the 396 g / 0.31 g incident and the override; disagreement → `AnchorUnitsConflict`); `mismatches(art, …)` compares DECLARED constants to the consumer's; `describe()` for the launch log |
| `stack/scripts/build_refc_anchors.py` | saves `anchor_meta.build_anchor_artifact(anchors, None, control_units="paths", horizons, dt=0.1, builder=__file__, extra=meta)` — legacy keys kept |
| `stack/scripts/refc_v3_train.py` | `--anchor-control-units` default **None** (explicit override only, help text names the rule); `_read_anchor_artifact(args)` runs FIRST in `train()` and `preflight()` (a units-less controls file is refused before data/GPU work; the resolved units are written back into `args`; declared `kappa_cap` / `alat_v_floor` adopted by `_pin_trainer_cfg` on BOTH arms so the hier/flat delta is unchanged); `_check_anchor_artifact_against_cfg` refuses a declared `horizon_s` / `dt` / `ref_speed_ms` / `kappa_cap` / `alat_v_floor` that differs from what the decoder would use; `_anchor_stamp(..., art=)` adds `control_units_source`, `artifact_schema`, `artifact_declared`, `artifact_provenance` to `config.json['anchors']`; NEW `_seam_stamp(cfg, args)` → `config.json['seams']` = `hier, hierarchy, graft_maneuver, factored_maneuver, graft_prior_center, graft_target_latent, grounded_selector, graft_imagination, tactical_speed_input, lan_enable, graft_lan, goal_str`; the `{a=0,κ=0}` refusal asks for 0.0 to be a NODE of both axes and names `linspace(-4, 3, 13)`. ⛔ The LIVE file keeps loading through `--anchor-control-units alat`; nothing was shipped to the pod |
| `…/2026-09-04-refcv4b-vocabulary/scripts/emit_anchors_alat.py` | emits through `anchor_meta` (units, horizon, provenance in the `.pt`; the sidecar JSON = the non-tensor part + file sha256); an importable `tanitad` wins over the hard-coded rig path; reads its own output back through the resolver (`control_units_source == "file"`). Re-run locally: the emitted `anchors` / `controls` sha256s reproduce the LIVE file's (`51f930dc…` / `b072f4c0…`) — recorded in the report. ⛔ The live `anchors.pt` is untouched |
| `stack/tests/test_anchor_meta.py` (NEW) | 15 tests: build fields + `weights_only=True` round-trip; refusals; legacy refused / override recorded; the F×C matrix; fixed-path needs no units; mismatches only on declared fields; the FPS builder writes the fields; `_seam_stamp`; flag default None + pin adoption; four 1-step smoke `train()` runs (legacy refused with "396 g" and no `config.json` written; legacy + override → `control_units_source: cli-override-legacy-file` + `seams`; self-describing file with no flag → `file`; conflicting flag and wrong `horizon_s` refused); the refusal text no longer says "odd counts" |
| `CLAUDE.md` | traps-preflight bullet: *a correct formula applied under the wrong units reads exactly like an answer* — 396 g vs 0.31 g, the rule (find the units line in the FILE) and the durable fix |

**Tests (off-Drive, `C:\Users\Admin\refcv4b_suite\stack`, `PYTHONPATH=<clone>/stack`, venv `tanitad`,
torch 2.11.0+cu128, `PYTHONIOENCODING=utf-8`):** `test_anchor_meta.py` **15 passed**; with
`test_build_refc_anchors_v4.py` + `test_refc_v3_nav_from_v7.py` **34 passed, 0 failed** (0 × `Errno 22`).
Wider REF-C subset (`test_refc_v3`, `test_refc_v4`, `test_refc`, `test_refc_v3_lan_preflight`,
`test_refc_v3_nav_and_uplink`, `test_refc_v3_save_before_eval`, `test_refc_v3_scale_matrix`,
`test_refc_v3_u8_batches`, `test_refc_select`, `test_anchor_flyability`): **145 passed, 5 failed** — the
five are `test_refc_v4::test_T8b/T9/T9b/T9d/T9e` and **fail identically on a pre-patch copy of the clone**
(`ModuleNotFoundError: No module named 'taniteval'` from `echo_gate.py:450`: the `stack/`-only mirror does
not carry `taniteval/`, memory `devbox-run-recipe-off-drive` #2) — pre-existing, not a regression.
`test_refcv3_arm.py` cannot be collected in this clone for the same reason.

⚠️ One defect the tests caught before commit: `torch.__version__` is a `TorchVersion` object, not
weights-only-safe — `provenance_stamp` records `str(torch.__version__)`.

## Deliverable manifest (where everything lives)

| artifact | where |
|---|---|
| registry row §4.6 | `repo:Project Steering/MODEL_REGISTRY.md` (commit `8cb69ac`) |
| `D-REFAV1-COST-SCALE`, `D-REFAV1-COST-FORM`, EPOCH-PLAN-VOID amendment, KINVOCAB1 qualifier | `repo:Project Steering/GOALS_AND_CLAIMS.md` (`7fce999`, `8ee4549`) |
| registry §2.4 amendment | `repo:Project Steering/MODEL_REGISTRY.md` (`7fce999`) |
| PREREG note | `repo:Project Steering/PREREG_TACTICAL_DECODER.md` (`7fce999`) |
| retractions #21, #22 | `repo:Project Steering/RETRACTION_LOG.md` (`7fce999`, `8ee4549`) |
| §3.5 correction + probe annotation | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-refcv4-gate-validation/{RESULT.md, raw/scripts/kinvocab_probe.py}` (`8ee4549`) |
| `anchor_meta.py`, trainer, FPS builder, tests, alat builder, CLAUDE.md bullet, this file | `repo:` paths in the item-4 table (item-4 commit) — also mirrored in `C:\Users\Admin\refcv4b_suite\stack` (tests ran there) |
| the edit tools (re-runnable, exact anchors) | `devbox:…\scratchpad\{apply_docs.py, patch_item4.py}` — session scratch, not banked (the edits they made are in the commits) |
| pre-patch comparison tree | `devbox:C:\Users\Admin\refcv4b_suite_orig\stack` (disposable) |

Nothing is stranded on a pod. No file was shipped to `tanitad-refcv3`; the live run and its `anchors.pt`
are untouched.
