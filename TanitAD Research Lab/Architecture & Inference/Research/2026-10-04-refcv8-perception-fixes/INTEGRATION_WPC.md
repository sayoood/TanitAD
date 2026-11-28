# INTEGRATION_WPC -- what WP-B must wire into `stack/scripts/refc_v3_train.py` (and the registry) for the WP-C fixes

*WP-C (perception fixes), 2026-10-04. Base tree: `50efa52`. Every module below is ALREADY written, tested and OFF by default; the only thing
missing is the trainer's call site, which WP-B owns. Line numbers are the tip's `refc_v3_train.py` (CRLF worktree = LF blob; `git archive` output
is CRLF). Nothing here changes a default run: each item is an OPT-IN flag whose absence is the pre-change code path (proved by the golden-digest
tests named per item).*

Order of work for WP-B: **I1 (NMS) -> I2 (z/h by range) -> I3 (join masks) -> I4 (refcv8's own F1/F4/F4b files)**; I1/I2 touch only the eval row,
I3 touches the TRAIN label stream, I4 is a run step, not code.

## I1  F4b centre-distance NMS in the in-run eval row  (module: `stack/tanitad/eval/detection_nms.py`, config: `stack/tanitad/configs/refcv7_det_nms_train.json`)

| | |
|---|---|
| flag | `--det-nms <json>` (default `None`) -- `tanitad.det_nms/1`: per head `{radius_m, gate, p_floor}`; `refcv7_det_nms_train.json` = box3d 2.5 m / gate 0.2145, agent 3.0 m / gate 0.1809 (refcv7-r101-s0 @ 50,400; **refcv8 needs its own, I4**) |
| needs | `--slot-vis1` (same reason as `--det-presence-gates`: the packs exist only with VIS-1) |
| validation | next to the `det_presence_gates` block, **`refc_v3_train.py:1482-1494`** (`_pin_slot_refine`): `if getattr(args, "det_nms", None):` require `--slot-vis1`, call `_det_nms.load_head_nms(args.det_nms)` inside `try/except (ValueError, OSError)` -> `SystemExit("[v3] ... --det-nms %r: %s")` |
| load | beside **`:8770-8773`**: `det_nms_cfg, _dn_stamp = _det_nms.load_head_nms(args.det_nms); calib_stamp["det_nms"] = _dn_stamp` (the sha256 then lands in `config.json[vis1][calib]`, like the gates) |
| import | next to **`:133`** `from tanitad.eval import detection_metrics as _det_metrics`: `from tanitad.eval import detection_nms as _det_nms` |
| emit | in the eval-row loop, right after the `gated_census_keys` loop at **`:9790-9793`** -- `for _dk, _dv in _det_nms.nms_census_keys(_pk, _hd, det_nms_cfg).items(): erow[_dk] = (None if (isinstance(_dv, float) and _dv != _dv) else round(float(_dv), 5))` -- `det_nms_cfg=None` returns `{}`, so a default run adds no key and computes nothing |
| new keys | `eval_{box3d,agent}_nms_{radius_m, gate, n_conf, tp, n_pos, prec, rec, f1, conf_ratio, conf_ratio_alarm, ap2m, boxes_per_object, frac_objects_ge2}` (13 per head) |
| G-DVB | `declared_vs_built.py`, after the `det_presence_gates` entry (**`:997`**): `register("det_nms", "data", reason="the per-head NMS radius + gate JSON (refcv8 WP-C F4b, OPT-IN); read ONLY by the in-run eval census (detection_nms.nms_census_keys, new eval_<head>_nms_* keys) -- never by the model or the planner; the validator refuses a missing / invalid file or one without --slot-vis1 before config.json, sha256 stamped in config.json[vis1][calib][det_nms]")` -- the registry refuses a flag without an entry; **bump the pin in `tests/test_declared_vs_built.py:268` (`len(dvb.REGISTRY) == 223` at the tip; +1 per flag: 224 with I1, 225 with I2, 226 with I3)** |
| tests to add | a wiring test mirroring `tests/test_refcv7_infer_fixes_wiring.py` for F4: (a) default argv -> `erow` has no `eval_*_nms_*` key and the existing eval row is key-for-key the tip's; (b) `--det-nms` -> the 26 keys appear, `eval_box3d_nms_radius_m == 2.5`; (c) `--det-nms` without `--slot-vis1` -> `SystemExit`. The module-side behaviour is pinned (`test_refcv8_det_nms.py`, 25 tests; `..._acceptance_full.py`, 6) |

**Do NOT** pass the NMS'd packs, the NMS gate or any `*_nms_*` value to `AgentTokenEmbed` / the planner / any loss. `tests/test_refcv8_det_nms.py::
test_E_no_planner_side_module_imports_the_nms` scans `tanitad/{models,refs,train}` for `detection_nms` and goes red on a hit -- if the trainer needs the
import, it lives in `scripts/` only (it does: `scripts/` is not scanned).

**Radius and gate are ONE operating point.** At the old gate (0.2589) the NMS halves `conf_ratio` (0.43, out of the [0.5, 1.5] band; MEASURED on
EVAL, route package sec. 4); the keys above are therefore always read at the config's own `gate`. The declared `eval_<head>_conf_ratio` / its A10
alarm keep their meaning (a Watch contract is unchanged).

## I2  z / h by range, beyond 30 m flagged LOW-TRUST  (modules: `detection_metrics.window_packs(.., with_zh_range=True)`, `stack/tanitad/eval/detection_zh.py`; config: `refcv8_box_zh_range_trust.json`)

| | |
|---|---|
| flag | `--det-zh-trust <json>` (default `None`) -- `tanitad.det_zh_trust/1` (D3 `base_face_by_range`; the near field, 30 m, is DERIVED by the loader from the table, not hand-set) |
| needs | `--slot-vis1`, `--w-box3d > 0` and `--join3d` (z/h labels exist only with the 3-D join; without it `pair_z_err` is empty and the report would be all-NaN) |
| carrier | `compute_losses_v3` has no `args`; mirror what F1 did for the thresholds: set `model._det_zh_range = True` in `train()` when the flag is given (and in `eval/refcv7_loader.py` where it sets the F1/F4 carriers), and at **`refc_v3_train.py:5575`** add `with_zh_range=bool(getattr(model, "_det_zh_range", False))` to the `_det_metrics.window_packs(...)` call of the **box3d** head (the agent head has no `cz`/`h`; the flag is a no-op there) |
| emit | in the eval-row loop, `if _hd == "box3d" and det_zh_cfg is not None:` `for k, v in _det_zh.zh_range_keys(_pk, "box3d", det_zh_cfg).items(): erow[k] = ...` |
| new keys | `eval_box3d_zh_{0_30,30_60,60_inf}_{n,z_mae,z_med,h_mae,h_med,low_trust}`, `eval_box3d_zh_nearfield_{n,z_mae,z_med,h_mae,h_med,range_m}`, `eval_box3d_zh_lowtrust_pair_frac` (**the near-field keys are the only z/h numbers that may be quoted**) |
| G-DVB | `register("det_zh_trust", "data", reason=...)` (same wording pattern); pin +1 (225) |
| default proof | `tests/test_refcv8_det_zh_range.py::test_default_window_packs_is_byte_identical_to_the_tip_and_carries_no_new_key` (golden digest of the TIP's `window_packs`) |

Optional second lever, NOT done (a training change, not a metric): restrict `zh_mask` to range < 30 m in the loss. D3 says only 0.9 % of boxes < 30 m have a
floating base face against 11.7 % at 30-100 m, so the z/h L1 terms are mostly training on label noise beyond 30 m. It would be a one-line mask in
`agent_cuboid_gt.zh_for_frame`'s consumer (`refc_v3_train.py:3402`) behind its own flag and needs its own v7-tiny arm (`TanitAD_ValidateAIDesign`).

## I3  join label masks: the ego as an agent box (D-2) and track-id-switch rate targets  (modules: `stack/tanitad/data/join_label_hygiene.py`, `JoinFileReader(.., defect_masks=)` in `stack/scripts/train_p8_occupancy.py`; list: `stack/tanitad/configs/refcv8_join_label_defects.json`)

| | |
|---|---|
| flag | `--join-defect-masks <json>` (default `None`) -- `tanitad.join_label_defects/1`, sha12-keyed; the shipped list is MINED from the SAME join D3 audited and reproduces D3's aggregates EXACTLY (693 boxes / 18 clips; 6,410 event frames / 887 clips / 853 pose-glitch / 5,822 single-track) |
| needs | `--agent-join` |
| construct | at the **TRAIN** reader, **`refc_v3_train.py:8654-8664`**: `masks = JoinDefectMasks.load(args.join_defect_masks) if args.join_defect_masks else None`; pass `defect_masks=masks` to `JoinFileReader(...)`; then `agent_stats["defect_masks"] = _rd.defect_stats()` (counts + the file's sha256 land in `config.json[agent_join_stats][train]`) |
| EVAL reader | **leave `:8908` and `refcv7_loader.py:824` WITHOUT the mask** (0 eval clips carry the ego box; masking eval rate targets would change eval comparability with refcv7). If you want the rate hygiene on eval too, pass it there and say so in the stamp |
| validation | refuse the flag without `--agent-join`; `JoinDefectMasks.load` refuses a wrong schema, a missing section and any non-sha12 key |
| G-DVB | `register("join_defect_masks", "data", reason=...)`; pin +1 (226) |
| default proof | `tests/test_refcv8_join_label_hygiene.py::test_default_reader_is_bit_identical_to_the_unmodified_tip_reader` (golden digest of every array the reader serves, computed from the TIP's reader) |

What the mask does (MEASURED on the real join, `raw/join_masks_acceptance.json`, 57 clips = the 18 ego clips + the 40 most-jumping clips, 11,345 records):
ego-footprint boxes **693 -> 0** (693 rows removed from exactly the listed frames, every aligned array -- classes, track ids, rates, raster -- loses the same
row: 0 alignment violations); on the 40 jump clips the largest |v_rel| among observed rate rows **764.1 -> 77.9 m/s**, rows above 100 m/s **5 -> 0**, **3,470 rate rows
masked** (the two records a jump contaminates, per event). **Scope of what it leaves:** 160 boxes of two clips carry a footprint track id but sit at x in
[-4.5, -1.1] m (BEHIND the ego, outside the forward FOV, already dropped by `visible_target_filter`); the D3 footprint rule deliberately does not touch them.

Regenerate the list for any new join: `python stack/scripts/mine_join_label_defects.py --join <join.jsonl.xz> --manifest <train _v2manifest.pt> --manifest <eval _v2manifest.pt> --out <json> [--check-d3 <c3b_agents_detail.json>]` (2.8 min, single process, CPU only; peak working set 551 MB MEASURED with `code/peak_rss.py`; a second run reproduced the shipped list byte for byte apart from the wall-clock field).

## I4  refcv8's OWN F1 / F4 / F4b files  (a run step; tool: `stack/scripts/refit_perception_thresholds.py`)

The three shipped files are bound to refcv7-r101-s0 @ 50,400 (the F1 loader REFUSES other class weights; F4/F4b gates are checkpoint-specific). After the refcv8
checkpoint exists:

1. run the diagnostics harness' TRAIN-DIAG pass on it (`2026-10-04-refcv7-map-box-diagnostics/code/run_diag.py --split train_diag`, ~10 min on Thor) -> `train.packs.pkl`, `train.acc.pt`;
2. `python stack/scripts/refit_perception_thresholds.py --run <name> --checkpoint-step <n> --checkpoint-md5 <md5> --fit-set "<one line>" --box-train-packs train.packs.pkl --out-gates g.json --nms --out-nms n.json --map-train-acc train.acc.pt --out-map m.json` (each file is re-READ by the stack's loader before the tool reports success);
3. score the refcv8 EVAL with `--map-hires-decision-rule class_threshold --map-hires-class-thresholds m.json --det-presence-gates g.json --det-nms n.json`.

`scripts/launch_gate.py` pins `--map-hires-decision-rule prior_corrected` in the refcv7 launch profile; a launch argv using the F1 flag is refused until the refcv8 profile is amended (a Master Mind call, noted in the F1/F4 landing).

## Status of the other WP-C items (nothing for WP-B to wire)

* **F1 / F4 (landed `b60cba6`)**: load and reproduce -- see the report; no change.
* **Fix 6 (eval clips without map GT)**: already masked at three layers; pinned by `tests/test_refcv8_eval_map_mask.py` (10 tests). No code change.
* **The `render_refcv7_replay.bev_nms` copy** (the PI's replay tool) is an unpinned, `p_floor`-less duplicate of `detection_nms.centre_nms_keep` with the same boundary rule (suppress iff distance <= r); single-sourcing it is an EvalFlyWheel cleanup, not a defect.
