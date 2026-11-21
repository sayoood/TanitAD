# PROPOSED `PROFILES["refcv9"]` for `stack/scripts/launch_gate.py` (NOT applied)

*Architecture/Training engineering agent for the Master Mind, 2026-10-27. Status: PROPOSAL. Under the PI authorisation of
2026-10-27 the patch may change `launch_gate.py` ONLY to add G-LIVE rules for `w_r9_*`, so this profile is NOT in the
tree. Code base: scratch tree `F:/refcv9_code/tree` = tip `bacd704924590f18f5772fd9f6c620de2c3dcfeb` + REFCV9_INMODEL.patch
+ `--r9-alloc-force-schedule`.*

## 1. What the PI ruled (2026-10-27), and how the profile carries it

| ruling | profile mechanism | status |
|---|---|---|
| WARM START from refcv8 EMA-48,852 | `--init-from` required (inherited from `_R8`); content bound to md5 `b4364029b68f29a33eeb4aebbbbd002b` = `_MAPFT_INIT_MD5` (INHERITED from `launch_gate.py:1177`, whose comment says it was MEASURED 2026-10-26 off `ckpt_ema_48852.pt`; I did not re-hash it) | ⚠️ **R9-INIT-BIND**: today the gate binds `--init-from` by content ONLY through the `map_only` block (`launch_gate.py:2412-2424`). A refcv9 binding needs a gate change outside the current authorisation |
| perception frozen, planner + tactical heads trained | none possible yet | ⛔ **R9-FREEZE (BLOCKER)**: no trainer flag implements this (see §3) |
| inputs unchanged from refcv8, incl. the future-derived speed band | every refcv8 input literal kept: `--r8-speed-input label`, `--max-speed-input-v6`, both `--speed-max-sidecar-v6*`, `--r8-nav-from-v9`, `--r8-no-rc`, `--r8-v9-*`, `--pose-sync-sidecar`, `--v2-cache` ... (inherited from `_R8["required_values"]` unchanged) | ready |
| (H-R9-ALLOC-ORACLE-1, refcv9 L2 candidate 1) train on predicted-only pools | `--r9-alloc-force-schedule` REQUIRED; value = **R9-SCHEDULE** | flag built + tested; value is a MM/PI choice |
| R9-PF progress head | FORBIDDEN in this profile: it is a separate stage-2 calibration fit on the fresh split (REFCV9_INMODEL_DESIGN §1, S2) and needs its own profile | ready |

## 2. The dict text (to be appended after `PROFILES["refcv8-mapft"]`)

```python
#: refcv9 WARM START (PI 2026-10-27): refcv8 EMA-48,852 -> perception frozen, planner + tactical heads trained, every
#: refcv8 input unchanged (incl. the future-derived speed band), the GT-in-pool schedule (H-R9-ALLOC-ORACLE-1) ON.
_R9_SCHED_FLAG = "--r9-alloc-force-schedule"
_R9_INIT_MD5 = _MAPFT_INIT_MD5          # ckpt_ema_48852.pt -- binding needs R9-INIT-BIND (gate change)
_R9_BUDGET = ("--steps", "--warmup", "--ema-decay", "--eval-every")
_R9_PF_FLAGS = ("--r9-progress-head", "--r9-calib-stage", "--w-r9-prog", "--w-r9-regime", "--r9-decision",
                "--r9-hidden", "--r9-decel-tau", "--r9-seed")
_R9_OPEN_ITEMS = (
    {"id": "R9-FREEZE",
     "what": "PI 2026-10-27 'perception frozen, planner + tactical heads trained' has NO trainer mechanism: "
             "--r8-train-only-map freezes the planner; --encoder-lr-mult scales only core.encoder (BEV / map / box / agent "
             "heads keep training, BN running stats move unless --trunk-frozen-bn; lr-mult 0.0 acceptance UNVERIFIED)",
     "owner": "the Master Mind (design: a map_only_ft-style literal module list + FrozenStateGuard + G-DVB lever + the "
              "gate's frozen-tensor bit-identity rule and live_dead_admitted entries -- a gate change the PI must authorise)",
     "closes_when": "a --r9-freeze-perception lever is built, gated and its trainable set written here as literals"},
    {"id": "R9-INIT-BIND",
     "what": "--init-from must be bound by CONTENT to ckpt_ema_48852.pt (md5 b4364029...); the gate binds init md5 only "
             "inside the map_only block",
     "owner": "the Master Mind (gate change, PI authorisation)",
     "closes_when": "a profile-level init_from_md5 rule exists and this profile pins _R9_INIT_MD5"},
    {"id": "R9-SCHEDULE",
     "what": "the --r9-alloc-force-schedule literal. Candidates: 'linear:1.0:0.0:<warmup>:<steps//2>' (start in "
             "refcv8's trained regime, finish the second half on inference-like pools) or 'const:0.0' (the pure removal)",
     "owner": "the Master Mind / PI",
     "closes_when": "the literal is written into required_values below"},
    {"id": "R9-BUDGET",
     "what": "--steps / --warmup / --ema-decay / --eval-every: refcv8's full-run literals (48,852 / 1,000 / 0.9998 / 500) "
             "are NOT inherited for a warm start; note EMA 0.9998 averages over ~5,000 steps",
     "owner": "the Master Mind (sizing from a measured G-LIVE cost row)",
     "closes_when": "the four literals are in required_values"},
    {"id": "R9-TF-RESTART",
     "what": "refcv8's constraint teacher-forcing schedule RESTARTS on a warm start: tf_ratio(step, steps, 1.0, 0.25) "
             "begins at 1.0 again although refcv8 ended at 0.25 (refc_v3_train.py, the per-step r8 block). Decide "
             "--r8-tf-start / --r8-tf-end explicitly (e.g. 0.25 / 0.25 = continue where refcv8 ended)",
     "owner": "the Master Mind", "closes_when": "both literals in required_values"},
    {"id": "R9-EMIT-START",
     "what": "the inherited --r8-alloc-emit-start 1000 re-runs refcv8's emission-OFF phase for 1,000 steps on a model "
             "that already emits its allocated extras; decide 0 or keep 1000 deliberately",
     "owner": "the Master Mind", "closes_when": "the literal is decided here"},
    {"id": "R9-B1",
     "what": "the B1 guard (--b1-guard-ref-ap2m / --b1-guard-step) cuts the head LR on a box3d AP drop; under a perception "
             "freeze the box head may not move at all (depends on R9-FREEZE's module list)",
     "owner": "the Master Mind", "closes_when": "kept with a reason, or dropped like refcv8-mapft's _MAPFT_B1"},
)
PROFILES["refcv9"] = dict(
    _R8, name="refcv9",
    open_items=_R9_OPEN_ITEMS,
    required_flags=tuple(_R8["required_flags"]) + (_R9_SCHED_FLAG, "--r9-alloc-force-seed"),
    required_flags_why=dict(_R8.get("required_flags_why") or {}, **{
        _R9_SCHED_FLAG: "H-R9-ALLOC-ORACLE-1: refcv8 trains only on pools that hold the GT hypothesis; the schedule lets "
                        "training see the predicted-only pool inference sees (value: R9-SCHEDULE)",
        "--r9-alloc-force-seed": "the schedule's PRIVATE per-(seed, step) coin stream, written as a literal",
        "--init-from": "PI 2026-10-27: the warm start IS refcv8 EMA-48,852 (content binding: R9-INIT-BIND)"}),
    required_values=dict({k: v for k, v in _R8["required_values"].items() if k not in _R9_BUDGET}, **{
        "--r9-alloc-force-seed": "20261027"}),
    required_values_why=dict({k: v for k, v in (_R8.get("required_values_why") or {}).items() if k not in _R9_BUDGET},
                             **{"--r9-alloc-force-seed": "refcv9_alloc_force.DEFAULT_SEED (a literal, not re-derived)"}),
    forbidden_levers=tuple(_R8["forbidden_levers"]) + (
        (_R9_PF_FLAGS, "R9-PF is the SEPARATE stage-2 calibration fit on the fresh split (REFCV9_INMODEL_DESIGN s1, S2); "
                       "the trainer refuses the head outside --r9-calib-stage and the schedule inside it"),
        (("--r8-train-only-map",), "the map-only fine-tune FREEZES the planner that refcv9 trains")),
)
```

## 3. Why R9-FREEZE is a blocker, not a profile detail

* MEASURED by source read of the scratch tree: the only freeze levers are `--r8-train-only-map`
  (`tanitad/train/map_only_ft.py`, trains three map modules only) and `--r9-calib-stage` (trains the R9 head only).
  `--encoder-lr-mult` (`refc_v3_train.py` `build_optimizer`, `--opt dd`) scales the `core.encoder` group only; every
  BEV / map / box / agent tensor would still train at the head rate.
* A real perception freeze touches G-LIVE (frozen tensors read zero gradient -> `live_dead_admitted`), G-DVB (a
  `built` lever for the trainable set) and the frozen-tensor bit-identity rule the map-only route uses -- each a gate
  edit outside the 2026-10-27 authorisation ("liveness rules for `w_r9_*` only"). ⇒ Named blocker under RULE ZERO (3):
  **unblocked by** a PI authorisation of the gate route + the MM's literal module list.

## 4. What the schedule flag guarantees (MEASURED on the scratch tree, CPU)

* Flag ABSENT: forward (234/272/280 outputs, 391/408/408 state tensors) and one real `compute_losses_v3` + AdamW step
  (350/373 loss-and-grad digests, 391/408 post-step tensors) bit-identical to the pristine tip
  (`F:/refcv9_code/bitid/bitid_REPORT_pristine_vs_final_flags_off.json`, `stepid_REPORT_pristine_vs_absent.json`).
* `const:1.0` = refcv8 as trained, bit for bit through the same step (two new telemetry keys `r9_pool_p` /
  `r9_pool_forced_frac` excluded) -- `stepid_REPORT_pristine_vs_const1.json`.
* `const:0.0` moves the step (87 digests, 73 tensors differ: `stepid_RED_pristine_vs_const0.json`); on the rig the
  posterior allocated the GT in 0 of 3 windows (`r8_alloc_gt_rows` 3 -> 0).
