#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THE LAUNCH GATE -- no training step runs until this writes a PASS token for the EXACT
(commit, code tree, argv, data) being launched.  SPEC_REFCV7.md section 2 (BINDING).

PI, 2026-09-26: *"Assure that this is not happening again because we lost a lot of time and
energy. If we are starting a training session, we are sure about the correctness of the config."*

WHY IT EXISTS -- three silent defects on ONE run (refcv6-r101-s0), each stamped into config.json:
  * F3's per-stage cascade loss never reached the loss for 34,500 steps (D-REFCV6-F3-WHITELIST);
  * the declared `--equalize-bottom-rows 43` never reached the trunk for 38,211 steps
    (D-REFCV6-EQUALIZE-DROPPED);
  * 2 of 3 declared selection mechanisms were never built (D-REFCV6-CONFIG-BUILD);
  and tactical labels were read ~0.37 s early (D-REFCV6-LABEL-CLOCK).
Every one was a DECLARED lever the built model or the live loss did not have. Every standing guard
was green. The gate therefore checks the THING, never the declaration:

  check    what must hold                                            regression arm (must FAIL)
  G-HYG    config dataclasses refuse undeclared attributes, AND the  set an undeclared attribute
           trainer's pinned config tree carries none                 (`undeclared_equalize`)
  G-DVB    `tanitad.train.declared_vs_built.check(model, args)`      drop the FIX-3 field
           returns [], every explicitly passed trainer flag is       (`drop_fix3_field`); the
           covered, and the gate's own behavioural probe of the      module missing
           trunk equalisation agrees with argv                       (`missing_dvb_module`)
  G-LIVE   a smoke on the REAL argv (steps -> 30) with the REAL      remove `cascade` from the
           trainer: every declared loss term present + finite, every pass-through
           declared-trainable leaf module gets a non-zero gradient,  (`no_cascade_passthrough`,
           `cascade` when F3, residual prior non-zero where v0 > 0   `no_cascade_silent`)
  G-CLOCK  |t_trainer - t_true| <= 0.05 s on EVERY window of every   the historical
           labelled clip of the train AND eval caches, t_true from   `(t + w - 1) * 0.1` clock
           an independent measurement; the legacy clock must read    (`legacy_label_clock`)
           RED in the same breath (positive control)
  G-EVAL   the eval loader builds the IDENTICAL model: strict 0/0,   a loader that skips
           bit-identical tensors, bit-identical forward outputs on   `_pin_trainer_cfg`
           a fixed real batch; a perturbed copy must DIFFER          (`loader_skips_pin`)
  G-CKPT   the trainer's own save -> load and resume are
           bit-identical: model, optimizer, data position
  G-SUITE  pytest on a CLEAN tree of the launch commit: 0 new
           failures vs the baseline (by test id), no SKIP whose
           reason says it could not run here (a skip is not a
           pass), and every guard of `guard_mutation_audit.py` CAUGHT

  refcv7 profile, SPEC section 6 (Amendment A1):
  * 6.1  DrivoR-T's `--refcv7`, `--w-r7-wta`, `--w-r7-scorer` (and every `--r7-*`, plus the
         post-rename `--drivort*` names) must be ABSENT, and the built model must carry no
         `refcv7_wta` / `refcv7_scorer` (G-DVB; arm `drivort_flag`).
  * 6.2  NEW-2, the 10 cm map head -- every check keyed on `--map-hires on` and INERT when off
         (flag / key / module names PROVISIONAL until the NEW-2 agent lands): G-DVB the branch is
         built, supervised and declares its class weights; G-EVAL/G-DVB (probe forward) 600x320
         logits from a 240x128 (0.25 m) lift, `fmap_s8` at stride 8, the branch in BOTH builds;
         G-LIVE `fmap_s8` reaches the output on every forward (arm `no_fmap_s8_passthrough`),
         the 10 cm loss is finite, the hi-res leaf modules get gradient, config.json records the
         class-weight sha256, and the Thor cost (s/step, `max_memory_allocated`): more than
         +25 % over refcv6's 6.4 s/step is PI-DECISION, which is never a PASS.
  * 7    (A2, PI ruling E1) `--graft-tac8-prior`, `--graft-nav-compliance`, `--speed-ceiling-filter`
         REQUIRED ON and BUILT (`declared_vs_built.check_refcv7_required(model, args, tau_file=)`),
         tau equal to the BANKED record whose sha256 the token binds (`gate:nav-tau-record`), the
         ceiling mask inactive in training and active at eval (G-LIVE). Arms: `required_on_missing`,
         `unwire_selection_term` (SPEC 2's "unwire one selection term"), `tau_differs`,
         `tau_file_missing`, `ceiling_active_in_training`.
  * 8    (A3) G-MAP: per-class 10 cm signal + per-class eval IoU (G-LIVE / G-DVB), and
         G-MAP-OVERFIT, the overfit protocol's PASS record, as a launch prerequisite.
  * 10   (A5) NEW-1: `--residual-prior ha0_ext_pose` (past poses only) with `--ego-history` is
         REQUIRED; `off`, `ha0_ext` and `cv_yawrate` are REFUSED by name (arms
         `residual_prior_ha0_ext`, `residual_prior_cv_yawrate`). A value decision that is still
         OPEN goes in `pi_pending_values` and makes the token PI-DECISION (empty for refcv7 today).
  Every argv-only profile rule is re-derived INSIDE `finalize`: a stand-alone `finalize` cannot
  mint a token that `run` would refuse.

THE TOKEN. `PASS_<commit12>.json` is written ONLY if every required check PASSed on evidence bound
to the same (profile, commit, tree_sha256, argv_sha256), and every input each check READ has the
same fingerprint as the launch host's data manifest. It is HMAC-SHA256-signed with a local key
(default `~/.config/tanitad/launch_gate.key`, generated on first use, NEVER in the repo). Any failure
writes `FAIL_<commit12>.json` with the reasons (and demotes a stale PASS for that commit); missing
evidence writes `INCOMPLETE_<commit12>.json`. A token is never hand-edited: the HMAC refuses it.

    # dev box (CPU; G-HYG, G-DVB, G-EVAL, G-SUITE):
    python stack/scripts/launch_gate.py run --profile refcv7 --stage devbox \
        --tree <clean tree of C> --commit <C> --argv-file <launch argv.json> --out-dir <gate dir> \
        --git-dir C:/Users/Admin/tanitad-push/.git --baseline <tip> --cpu-only --min-free-gb 8 \
        --path-map /home/nvidia/data=D:/refcv6_eval_kit/data --eval-kit D:/refcv6_eval_kit
    # Thor (G-LIVE + G-CKPT on the real data, G-CLOCK on the real caches), then the token:
    python stack/scripts/launch_gate.py run --profile refcv7 --stage thor --tree <shipped C> \
        --commit <C> --argv-file <launch argv.json> --out-dir <gate dir> --import-evidence <devbox evidence>
    # the supervisor, before every launch:
    python stack/scripts/launch_gate.py verify --token <PASS_...json> --argv-file ... --tree <C> --json <v.json>
    python stack/scripts/launch_gate.py exec   --token <PASS_...json> --argv-file ... --tree <C> --json <e.json>

ON-DISK ARTIFACTS ARE THE VERDICT. A missing evidence file is an ERROR, never a PASS; an exit code
is a claim by a process about itself (CLAUDE.md: "assert on the artifact, not the status").
A check that could not RUN is ERROR. An evidence file produced under a regression arm can never
contribute to a PASS.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import hmac
import importlib
import importlib.util
import io
import json
import math
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

GATE_SCHEMA = "tanitad.launch_gate/1"
CHECKS = ("G-HYG", "G-DVB", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE",
          "G-MAP-OVERFIT", "G-SUITE-PINNED", "G-BOX-OVERFIT")
#: the checks every launch needs; a profile adds its own (refcv7: G-SUITE-PINNED and both overfit
#: records), and G-MAP-OVERFIT joins any profile's list when the 10 cm map head is on
BASE_CHECKS = CHECKS[:7]
#: the checks that share one child process (one model build / one smoke / one clock pass).
#: ⛔ ORDER MATTERS: the smoke runs BEFORE the model job, so G-EVAL can rebuild the eval model
#: from the config.json the trainer wrote for THIS argv (never from another run's record --
#: MEASURED 2026-09-26: refcv6's record rebuilds refcv6 AS TRAINED, an unequalised trunk, and
#: every trunk-downstream output then differs from the fixed trainer's)
JOBS = (("smoke", ("G-LIVE", "G-CKPT")), ("model", ("G-HYG", "G-DVB", "G-EVAL")),
        ("clock", ("G-CLOCK",)), ("suite", ("G-SUITE",)), ("pinned", ("G-SUITE-PINNED",)),
        ("record", ("G-MAP-OVERFIT", "G-BOX-OVERFIT")))
#: the verdict a PI-reserved decision produces (the Thor cost rule, an open value decision):
#: the gate STOPS and the token never verifies -- but it is not a FAIL
PI_DECISION = "PI-DECISION"
ARM_ENV = "TANITAD_GATE_ARM"
KEY_ENV = "TANITAD_GATE_KEY_FILE"
TREE_ROOTS = ("stack", "taniteval")
_TREE_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".guard_mutation_backup",
                             ".mypy_cache", ".ruff_cache"})
_TREE_SKIP_SUFFIX = (".pyc", ".pyo")
#: files a cache directory may (re)write on first use -- listed, never content-bound
_DIR_SKIP_NAMES = frozenset({"_v2manifest.pt", "_v2manifest.pt.tmp", "__pycache__"})
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# --------------------------------------------------------------------------------------------- #
# profiles                                                                                        #
# --------------------------------------------------------------------------------------------- #
_EVAL_LOADER_REL = ("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/"
                    "battery/code/refcv6_loader.py")
_AUDIT_RAW_REL = ("TanitAD Research Lab/Architecture & Inference/Research/"
                  "2026-09-26-refcv6-frozen-trunk-audit/raw")

_REFC = {
    "trainer": "stack/scripts/refc_v3_train.py",
    "required_checks": BASE_CHECKS,
    #: dev box: CPU, no training data; Thor: the real caches. G-HYG / G-DVB are cheap and run
    #: on both (the launch host's own evidence wins over imported evidence). G-MAP-OVERFIT is a
    #: record check, dropped from a run whose argv has the 10 cm head off.
    "stages": {"devbox": ("G-HYG", "G-DVB", "G-EVAL", "G-SUITE", "G-MAP-OVERFIT"),
               "thor": ("G-HYG", "G-DVB", "G-LIVE", "G-CKPT", "G-CLOCK", "G-MAP-OVERFIT"),
               "all": CHECKS[:8]},
    "smoke_steps": 30,
    "resume_extra_steps": 2,
    "output_flags": ("--out", "--eval-window-dump"),
    #: inputs the MODEL BUILD reads (anchors -> buffers + decoder constants, extrinsics -> lift
    #: bank + rig camera, the banked tau file, the 10 cm class weights). Every other data flag is
    #: replaced by a NON-EXISTENT sentinel in the model job, so a build that silently reads more
    #: than this list FAILS instead of passing.
    "model_input_flags": ("--anchors", "--agent-rig-extrinsics", "--nav-compliance-tau-file",
                          "--map-hires-class-weights"),
    "required_flags": (),
    "suites": ("stack", "taniteval", "tools"),
    #: G-SUITE: "run" = pytest over the suites on clean trees HERE (the generic family);
    #: "consume" = the Master Mind's Thor full-suite verdict, BOUND to the launch commit and tree
    #: (`suite-bind`), plus G-SUITE-PINNED on the dev box (refcv7)
    "suite_mode": "run",
    #: (repo path, sha256) of the PINNED Thor environment-failure list; None = no pin
    "suite_env_failures": None,
    #: G-SUITE-PINNED: SKIP reasons REGISTERED as admissible in the pinned files (regex); every
    #: other skip there is NOT a pass
    "suite_pinned_skip_ok": (),
    "clock_tolerance_s": 0.05,
    "clock_reference": (f"{_AUDIT_RAW_REL}/q4c_grid_vs_egolog_ALLTRAIN.json",),
    "eval_loader": _EVAL_LOADER_REL,
    "check_timeout_s": {"model": 3 * 3600, "smoke": 6 * 3600, "clock": 2 * 3600,
                        "suite": 6 * 3600, "pinned": 3 * 3600, "record": 600},
    #: residual-prior output keys the G-LIVE probe looks for when `--residual-prior` is passed
    #: (NEW-1's forward emits `residual_prior_path` [B,S,2] and the prior's own speed
    #: `residual_prior_v` -- withheld rows carry a different v than pose_last)
    "residual_prior_keys": ("residual_prior_path", "residual_prior", "prior_traj", "kin_prior",
                            "plan_prior", "prior"),
    #: levers that must be OFF for this profile (none for the generic refc family)
    "forbidden_levers": (),
    "forbidden_prefixes": (),
    "forbidden_built": (),
    #: `declared_vs_built.check(model, args, parser, forbid_kinds=...)` (the fixes agent's frozen
    #: API): train() calls it WITHOUT forbidding DrivoR-T, so a refcv7 GATE adds the forbid.
    "dvb_forbid_kinds": (),
    #: FIX-5: `read_speed_max_sidecar_v6` REFUSES without `<sidecar>.meta.json`. The gate checks
    #: the file EXISTS on the launch host (and binds its content, like every other meta it finds).
    "meta_required_flags": ("--speed-max-sidecar-v6", "--speed-max-sidecar-v6-eval"),
    #: FIX-2 / G3: the share of a split's clips allowed without a measured clock. ⛔ The gate uses
    #: the trainer's constant and NEVER a larger value (coordinator 2026-09-26: "do not raise
    #: the cap"; the eval split's handling is the E2 decision).
    "clock_max_unverified_frac": 0.01,
    #: E2(b) (landed ab436ee): on the EVAL split a clip without a measured clock is not capped
    #: but EXCLUDED from the TACTICAL family, exactly as train() runs G3 there
    "clock_eval_mode": "exclude_tactical",
    #: FIX-4: `--graft-nav-compliance` needs a RECORDED tau derived on the TRAIN split, whose `tau`
    #: equals `--nav-compliance-tau-rad`. The banked record (ab436ee) was written by the fixes
    #: agent's `code/derive_navc_tau.py` ON THOR; `stack/scripts/refcv7_derive_nav_tau.py --json`
    #: is the in-repo writer. Both schemas are read (see `nav_tau_reasons`). Batch 2 (b4a59b9)
    #: stamps the file as `config.json["seams"]["nav_compliance_tau_file"]` = {path, sha256, tau}.
    "nav_compliance": {"switch": "--graft-nav-compliance", "tau_flag": "--nav-compliance-tau-rad",
                       "tau_file_flag": "--nav-compliance-tau-file",
                       "tau_stamp_path": ("seams", "nav_compliance_tau_file"),
                       "cache_flag": "--v2-cache", "labels_flag": "--v7-labels"},
    #: flags that must be ON (and BUILT) for this profile; empty for the generic family
    "required_on": (),
    #: flags that must carry one exact value
    "required_values": {},
    #: flags that must be passed with a value OTHER than these
    "forbidden_values": {},
    #: flags whose VALUE is a decision the PI has not taken yet: {flag: why}. While a flag sits
    #: here and `required_values` does not fix it, the token is PI-DECISION (never PASS). The
    #: ruling is recorded by moving the flag into `required_values` -- a reviewed code change.
    "pi_pending_values": {},
    #: G-SUITE: a SKIP whose reason matches one of these is NOT a pass (coordinator 2026-09-26:
    #: the FIX-5 resnet34 tests skip where the checkpoint is not cached -- they must RUN where it
    #: is). Every other skip reason is recorded in the evidence, counted by reason.
    "suite_skip_not_pass": (r"not in the local HF cache",),
    #: the banked tau derivation a `--graft-nav-compliance` launch is bound to, when the operator
    #: names none (repo-relative; resolved against the tree's enclosing repo)
    "nav_tau_record_default": None,
    #: SPEC_REFCV7 section 7: the max-speed ceiling filter acts at INFERENCE ONLY -- the smoke
    #: must show it inactive in a training step and active in an eval step. Keyed on the flag.
    "ceiling_filter_flag": "--speed-ceiling-filter",
    #: G-HYG: every third-party module the trainer can reach (a STATIC import closure from the
    #: trainer over the tree's own modules) must be importable in THIS venv -- MEASURED
    #: 2026-09-27: scipy is ABSENT from Thor's training venv (Master Mind)
    "import_closure": True,
    #: SPEC_REFCV7 6.2 / A6 / A7 / A8 (NEW-2): the 10 cm map head. The flag names are the NEW-2
    #: builder's final table (`…/2026-09-26-refcv7-map-hires/BUILD.md` sec. 10). Every check is
    #: keyed on `switch` and INERT when it is off.
    "map_hires": {
        "switch": ("--map-hires", "on"),
        "weight_flag": "--w-map-hires",
        "class_weights_flag": "--map-hires-class-weights",
        "fmap_key": "fmap_s8",
        "loss_key": "map_hires",
        "name_token": "hires",
        #: A7 (SPEC_REFCV7 12): 100 m ahead x +-30 m -- the 10 cm grid 1000 x 600, the lift at
        #: 0.25 m 400 x 240; the planner BEV cropped to 60 m x +-16 m
        "logits_hw": (1000, 600),
        "lift_hw": (400, 240),
        "trunk_stride": 8,
        "cost_ref_s_per_step": 6.4,          # refcv6-r101-s0, MEASURED on Thor at batch 16
        "cost_max_ratio": 1.25,              # > 8.0 s/step goes to the PI (PI-DECISION)
        "cost_warmup_steps": 10,             # compile + cuDNN autotune + loader warm-up
        # ---- G-MAP. The SPELLING is the registered one (LOGGING_SPEC_MAP10 sec. 1-2, the
        # NEW-2 builder's `map_head_hires.per_class_key`), written here as LITERALS; when the
        # module is importable its CLASS_KEYS / BAND_KEYS must agree (a drift FAILS LOUD) ------
        "classes": ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched",
                    "sidewalk"),                             # CLASS_KEYS, code order 0..7
        "thin_classes": ("lane", "crosswalk", "arrow", "edge", "hatched"),
        "unseen_code": 255,
        "min_cells": 50,                                     # M: labelled 10 cm cells per class
        #: A7: every 20 m band of the 100 m grid (the bars apply in every band, SPEC 13 item 2)
        "bands": ("0_20", "20_40", "40_60", "60_80", "80_100"),
        "gated_band": "0_20",
        #: eval rows (each carries the `eval_` prefix): POOLED IoU under the declared rule and
        #: under the raw rule, the loss share, and the argmax COUNTS the IoU is pooled from.
        #: A8 item 3 -- the Watch contract: the 40 `eval_map_hires_iou_{cls}_{band}` keys
        "eval_key_fmt": "map_hires_iou_{cls}_{band}",
        "eval_iouraw_fmt": "map_hires_iouraw_{cls}_{band}",
        "eval_lshare_fmt": "map_hires_lshare_{cls}_{band}",
        "eval_count_fmts": ("map_hires_inter_{cls}_{band}", "map_hires_union_{cls}_{band}"),
        "raw_count_fmts": ("map_hires_interraw_{cls}_{band}", "map_hires_unionraw_{cls}_{band}"),
        "n_fmt": "map_hires_n_{cls}_{band}",
        #: train rows: the per-class x band loss CONTRIBUTIONS, which sum to `map_hires`
        "lc_fmt": "map_hires_lc_{cls}_{band}",
        "lc_sum_rel_tol": 1e-5,
        "eval_key_list_attrs": ("MAP_HIRES_EVAL_KEYS", "MAP_EVAL_KEYS"),
        #: the declared decision rule (`raw` | `prior_corrected`) in config.json
        "decision_rule_path": ("map_hires", "decision_rule"),
        "decision_rules": ("raw", "prior_corrected"),
        #: A8 item 1: the class weights are sqrt(median-frequency); the loader stamps
        #: `config.json["map_hires"]["class_weights"]` = {definition_id, pre_registered, sha256}
        "class_weights_stamp_path": ("map_hires", "class_weights"),
        "class_weights_definition": "sqrt_mf",
        #: the NEW-2 per-part gradient reach (BUILD.md sec. 1): the four `ga_mh_*` parts, and
        #: `ga_bev_pool` (the planner's pooled BEV) under `--bev-source map_hires_pool`
        "ga_keys": ("ga_mh_lift", "ga_mh_encoder", "ga_mh_refine", "ga_mh_trunk_s8_stage"),
        "ga_keys_pool": ("ga_bev_pool",),
        "bev_source_flag": "--bev-source",
        #: LOGGING_SPEC_MAP10 sec. 6 item 4: the refcv7 Training Watch, built BY THE GATE from the
        #: smoke's own metrics.jsonl. A command template ({python}, {metrics}, {out}, {tree}).
        #: DECLARED by the Master Mind 2026-09-27: the Watch agent ran exactly this template as the
        #: gate formats it -- rc 0, `map_watch_reasons` [] (INHERITED; package
        #: `…/2026-09-27-refcv7-training-watch/`). A non-zero exit is a named FAIL.
        "watch_builder": ("{python}", "{tree}/taniteval/tools/training_watch/build_watch_refcv7.py",
                          "--metrics", "{metrics}", "--out", "{out}"),
    },
    #: OPEN ITEMS: undecided parts of a profile's launch flag set (refcv7 declares its own; see
    #: `_REFCV7_OPEN_ITEMS`). Empty = the flag set is decided.
    "open_items": (),
    #: ⛔ SPEC_REFCV7 24 (A19): MAIN-only BINDING overfit records. None = every must-fail arm must
    #: RUN and fail as registered (the pre-A19 rule). A profile adopting A19 names the policy, its
    #: SPEC source and the must-fail evidence it INHERITS (see PROFILES['refcv7']).
    "overfit_main_only": None,
    #: G-MAP-OVERFIT: the pre-registered protocol's LITERALS
    #: (`…/2026-09-26-map-signal-audit/raw/PREREG_G_MAP_OVERFIT.md` + `gmo_spec.json`; SPEC 9
    #: item 2). The gate checks the record against these, never against the bars it carries.
    "map_overfit": {
        "schema": "tanitad.g_map_overfit_record/1",
        #: the harness the closure record must wrap (`closure_run.py -- <harness> ...`)
        "harness": "stack/scripts/map_hires_overfit.py",
        "band": "0_20",
        "min_cells": 1000,
        #: prereg sec. 6.4: each class's CE at the end <= this x its step-0 CE (re-judged from the
        #: record's own numbers under A19)
        "ce_ratio_max": 0.5,
        "iou_bars": {"nocls": 0.85, "drivable": 0.85, "sidewalk": 0.85, "lane": 0.5,
                     "crosswalk": 0.5, "arrow": 0.5, "edge": 0.5, "hatched": 0.5},
        #: ``near_block_zeros`` (edge): SPEC_REFCV7 20 (A15), kept by 23 (A18) -- zeros into the
        #: near refine block leave the A12 function, so an edge pass must come from the block
        "must_fail_any": {"lane_w0": ("lane",), "near_block_zeros": ("edge",)},
        "must_fail_all": {"s8_zeros": ("lane", "crosswalk", "arrow", "edge", "hatched")},
        "controls": ("C1", "C2", "C3"),
        "extent": {"x_max_m": 100.0, "y_half_m": 30.0},
        #: ⛔ SPEC_REFCV7 23 (A18, the PI 2026-09-27 "Budget to 3,000 steps"): the PROTOCOL the
        #: binding record must have RUN -- the registered A18 spec by sha256
        #: (`…/2026-09-26-refcv7-map-hires/raw/gmo_spec_A18.json`), 3,000 steps, lr 1e-3 held to
        #: step 2,700 then cosine to 0 (A17.1, moved by A18), batch 4, seed 0 -- on A15's map
        #: path (the near lift 20 m + ONE near refine block; the argv's required values below).
        #: The prereg's 1,000-step protocol is SUPERSEDED (its FAILs stay on the record): a
        #: record of it is REFUSED here, never re-judged against these bars.
        "spec_sha256": "5cb4f6fc4a8c15660237f541f077bfb334337dd287f674f8de02b7d414077c58",
        "protocol": {"steps": 3000, "lr": 1e-3, "batch": 4, "seed": 0,
                     "lr_decay": {"kind": "cosine_to_zero", "start_step": 2700}},
        "near_lift_m": 20.0, "near_refine_blocks": 1,
        "frameset_md5": "4eafa03c2b6a6e6d6336be1d78acb91d",
        "frames": (("10497f0d664b", 10), ("10497f0d664b", 110), ("10497f0d664b", 130),
                   ("10497f0d664b", 150), ("05c575ed45be", 50), ("05c575ed45be", 85),
                   ("05c575ed45be", 105), ("05c575ed45be", 130), ("104d79ae052d", 10),
                   ("104d79ae052d", 90), ("104d79ae052d", 115), ("104d79ae052d", 135),
                   ("0dbd6c9cd776", 15), ("0dbd6c9cd776", 35), ("0dbd6c9cd776", 60),
                   ("0dbd6c9cd776", 90)),
    },
    #: G-BOX-OVERFIT: the pre-registered protocol's LITERALS (SPEC_REFCV7 15.1, A10:
    #: `…/2026-09-26-box-head-audit/raw/PREREG_G_BOX_OVERFIT.md` md5 594c7119..., reconciled to
    #: A9's IGNORE rule). ⚠️ The record SCHEMA is PROVISIONAL until the box-head builder's harness
    #: lands; every field the gate needs is named, and an absent one is a named FAIL.
    "box_overfit": {
        #: the record's identity: the harness writes `tool`, no schema field
        "tool": "g_box_overfit.py",
        #: the harness the closure record must wrap -- the box-head builder's
        #: `…/2026-09-27-refcv7-box-head/code/new/stack/scripts/g_box_overfit.py` (PROVISIONAL
        #: until that package lands)
        "harness": "stack/scripts/g_box_overfit.py",
        "prereg_md5": "594c71196cc5bbd527fee40b2cb0e3f1",
        "frameset_md5": "b291404c36f83c3e397e3b90367e8e7b",
        "n_frames": 16, "n_pos": 113, "n_ignore": 77,
        "steps": 2000, "batch": 4, "seed": 0, "gate": 0.5,
        #: SPEC_REFCV7 A13 (sec. 18): the LAUNCH optimiser AS BUILT (`build_optimizer`, `--opt dd --lr 1e-4`:
        #: AdamW, the trunk group at --encoder-lr-mult 0.5) + clip 10 -- replacing the prereg's `lr 2e-4 on
        #: every tensor`; the record's per-arm `optimizer` spec (built, i.e. PEAK lrs) is read
        "optimizer": {"class": "AdamW", "group_lrs": (5e-05, 1e-04), "weight_decay": 1e-4, "clip": 10.0},
        #: SPEC_REFCV7 A17 (sec. 22, the PI): the peaks held through step 1,799, then a cosine decay to 0 over
        #: steps 1,800-2,000 (both groups, the ratio kept) -- the record's `literals.lr_schedule`, verbatim
        "lr_schedule": {"rule": "A17", "hold_peak_through_step": 1799, "decay": "cosine",
                        "decay_from_step": 1800, "decay_to_step": 2000, "final_factor": 0.0},
        #: PREREG sec. 5, numbered as the prereg numbers them; each term is (row key, op, literal)
        #: on the arm's FINAL row. `count_rel` = |n_conf - 113| / 113 and `presence_ratio` =
        #: presence(2000) / presence(0) are DERIVED by the gate; 6 also needs the loss finite at
        #: every step (the harness's criterion-6 flag, the only place it is recorded)
        "criteria": {"1": (("ap2m", ">=", 0.90),),
                     "2": (("prec", ">=", 0.90), ("rec", ">=", 0.90)),
                     "3": (("count_rel", "<=", 0.10),),
                     "4": (("centre_p50_m", "<=", 0.30), ("size_p50_m", "<=", 0.30),
                           ("z_p50_m", "<=", 0.15)),
                     "5": (("cls_acc", ">=", 0.90),),
                     "6": (("presence_ratio", "<=", 0.25),)},
        #: PREREG sec. 6: each must-fail arm FAILS every criterion named (one passing = VOID)
        "must_fail": {"memory_zeros": ("1",), "presence_w0": ("2", "3")},
        #: C1-C3 must read their known values; C4 (step 0) is reported, not judged
        "controls": ("C1", "C2", "C3"),
    },
    #: A9 / A10 15.2 -- G-LIVE-PRES: at the END of the smoke, on BOTH slot heads, the fraction of
    #: slots whose sigma(presence logit) >= `gate` is < `max_frac` (refcv6 red arm: 71-99 of 100).
    #: The heads are found in the forward output by the key token; None = not checked (generic).
    "live_pres": None,
    #: G-LIVE's gradient-reach rows (coordinator 2026-09-26, the audit's MEASURED finding: refcv6
    #: logged 0 `ga_*` keys in 4,621 rows -- an off-by-one against `step += 1`). The groups are
    #: the trainer's `grad_reach_report` parts, written here as LITERAL attribute paths on the
    #: built model; a group is REQUIRED in every logged row when its module exists and has a
    #: parameter the optimizer trains. Batch 2's config.json declaration is held to the rows
    #: too (`declared_vs_built.check_logged_rows`).
    "ga_groups": {"trunk": "core.encoder", "planner": "core.decoder",
                  "tac_decoder": "tac_decoder_v6", "lift": "_perception.lift",
                  "bev_encoder": "_perception.map_branch.encoder",
                  "map_head": "_perception.map_branch.head", "box_memory": "_perception.box_mem",
                  "box_decoder": "_perception.box_dec"},
}
#: ⛔ OPEN ITEMS (Master Mind 2026-09-27): parts of the launch flag set that are NOT DECIDED
#: yet. While ANY is open this profile cannot mint a PASS -- `finalize` names each one and the
#: verdict is at best INCOMPLETE. An item is closed by landing its flags as `required_values`
#: (and in the canonical argv `stack/ops/runs.d/refcv7-r101-s0.argv.json`) and deleting it here.
#: BOX-HEAD was CLOSED 2026-09-27 (the box head landed as 28d8365): its flags are `required_values` /
#: `required_flags` below, built-checked by `box_required` in G-DVB; +R6 was not adopted (A14.1: LRP).
_REFCV7_OPEN_ITEMS = (
    # MAP-LIFT: CLOSED (SPEC_REFCV7 17 A12 + 20 A15 + 23 A18) -- its two flags are required
    # values below and in the canonical argv; the binding G-MAP-OVERFIT runs A18's protocol
)
#: ⛔ SPEC_REFCV7 section 6.1: DrivoR-T's code still answers to `refcv7_*` names. Until its rename
#: a refcv7 launch passes NONE of these flags, and G-DVB lists them as DrivoR-T levers that must be
#: OFF. Keyed on the flag NAMES, today's and the post-rename ones, so the rule survives the rename.
_DRIVORT_OFF = (
    (("--refcv7", "--drivort"), "DrivoR-T's switch (SPEC_DRIVORT.md; SPEC_REFCV7 6.1)"),
    (("--w-r7-wta", "--w-drivort-wta"), "DrivoR-T's WTA proposal-decoder weight"),
    (("--w-r7-scorer", "--w-drivort-scorer"), "DrivoR-T's disentangled-scorer weight"),
)
#: the pinned Thor failure list -- 165 ENV ids (fixes-batch-2 gate) + 1 FLAKY id with its cited
#: record (fixes-batch-3 gate), in 33 files -- and its sha256: a changed list is a reviewed edit of
#: BOTH lines
_THOR_ENV_FAILURES = ("stack/ops/launch_gate_thor_env_failures.json",
                      "fc1214e5978383c694af7dbfc6c26f44758fcf8a93c4386700efb792072b4ed4")
PROFILES: dict[str, dict] = {
    "refc": dict(_REFC, name="refc"),
    #: the arm the PI stopped on 2026-09-26: kept so the gate can be run on its argv as evidence
    "refcv6": dict(_REFC, name="refcv6"),
    #: SPEC_REFCV7 = refcv6 + FIX-1..5 + NEW-1 (the residual on a kinematic prior) + NEW-2 (the map
    #: at 10 cm, one lift, 100 m x +-30 m) + A9/A10 (the refined box head)
    "refcv7": dict(
        _REFC, name="refcv7",
        #: G-EVAL's loader: refcv7's OWN (`stack/tanitad/eval/refcv7_loader.py`). The refcv6 battery
        #: loader cannot build NEW-2's 10 cm branch -- MEASURED 2026-09-27: its strict load fails on
        #: `_map_hires.lift.unobserved`. On Thor pass `--eval-kit /home/nvidia` (the kit layout).
        eval_loader="stack/tanitad/eval/refcv7_loader.py",
        #: ⛔ SPEC_REFCV7 24 (A19, the PI 2026-09-27 17:46 / 17:51 Berlin: "Binding = MAIN only,
        #: both" / "Keep MAIN-only for both"): a BINDING overfit record whose must-fail arms were NOT
        #: RUN is accepted -- MAIN is re-judged from its OWN registered literals, never from the
        #: harness's overall verdict (which reads FAIL without the arms). A must-fail arm that RAN
        #: and PASSED is still a VOID (A19 excuses absence, never a VOID). The must-fail evidence is
        #: INHERITED -- stated exactly below and carried in the gate's PASS text, so the token says
        #: what it does not cover.
        overfit_main_only={
            "policy": "A19_MAIN_ONLY_BINDING",
            "source": "SPEC_REFCV7 24 (A19, landed 36cc332)",
            "map": {
                #: the A18 spec with its must-fail rows REMOVED -- the registered harness refuses
                #: `--arms healthy` on a spec that names must-fail arms (main(): "the gated arm ...
                #: is not in --arms"); every other literal is A18's. `…/raw/gmo_spec_A19_MAP_MAIN.json`
                "spec_sha256": "56dea067fdef45e9b905f5741b562e59ad037396067cdbe43278c37f62be4f2a",
                "inherited": ("the map must-fail arms are INHERITED from A17.1 (the launch map config "
                              "+ the decay, 1,000 steps): s8_zeros read 0 on every thin class, "
                              "near_block_zeros read edge 0.199, lane_w0 read lane 0 (A12 and A15 "
                              "held them too)"),
                "evidence": ("TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires/raw/gmo_early/"
                             "g_map_overfit_A171.EARLY_NONBINDING.json",),
            },
            "box": {
                "inherited": ("presence_w0 is robust by construction (zero presence-loss weight gives "
                              "the presence head no gradient); memory_zeros was NOT measured at full "
                              "scale on the launch head -- it held in the TOY test on the A17 code "
                              "(18/18 cells, ap2m 0.0022-0.0171) and on the pre-A14 head, and "
                              "learned_ref also reads BEV tokens pooled from the map branch, so a leak "
                              "around the blinded memory would not be seen before launch; the PI "
                              "accepted this knowingly (A19 Q2)"),
                "evidence": ("TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/raw/toy/",
                             "TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/raw/thor_gbo/gbo_early_nonbinding.json",
                             "stack/tests/test_g_box_overfit.py::"
                             "test_TOY_the_loop_memorises_and_the_memory_zeros_arm_cannot"),
            },
        },
        required_checks=BASE_CHECKS + ("G-SUITE-PINNED", "G-MAP-OVERFIT", "G-BOX-OVERFIT"),
        #: EVERYTHING that needs the data or the launch venv runs on Thor; the dev box runs only
        #: G-SUITE-PINNED (a full-commit archive with the HF cache) -- its evidence is imported
        stages={"thor": ("G-HYG", "G-DVB", "G-LIVE", "G-CKPT", "G-EVAL", "G-CLOCK", "G-SUITE",
                         "G-MAP-OVERFIT", "G-BOX-OVERFIT"),
                "devbox": ("G-SUITE-PINNED",),
                "all": CHECKS},
        suite_mode="consume",
        suite_env_failures=_THOR_ENV_FAILURES,
        open_items=_REFCV7_OPEN_ITEMS,
        #: G-SUITE-PINNED (b): the REGISTERED skips of the dev-box half, as (test id, reason)
        #: PAIRS -- a reason admits ITS test only. PROPOSED by the gate agent 2026-09-27 from the
        #: MEASURED 3cca805 rehearsals (`…/2026-09-26-refcv7-launch-gate/RESULT.md` sec. 3); landing
        #: this file is the Master Mind's approval. Any other skip is still a FAIL.
        suite_pinned_skip_ok=(
            # per-backend: Windows-CPU BLAS is not bit-exact across gemm tilings. ⚠️ On Thor the
            # same test is a PINNED ENV failure (the pin file), so it passes on NEITHER host.
            (r"tests\.test_anchor_prefilter::test_prefilter_is_BIT_EXACT_on_every_candidate_it_decodes",
             r"^Windows-CPU BLAS picks shape-dependent gemm tilings"),
            # KNOWN GAP: refc_anchors_small64.pt is NOT in the repo (absent at 3cca805; asserted
            # with a same-breath control), so this test runs NOWHERE from a git archive
            (r"tests\.test_anchor_prefilter::test_replicates_the_measured_survivor_counts_on_the_canonical_val",
             r"^anchor fixture absent: refc_anchors_small64\.pt$"),
            # KNOWN GAP: the T1 kinematic-gate hook is not written (Decisions/2026-09-05-mm-
            # decisions.md M23/M24); these contract tests ACTIVATE the moment it lands. Owner:
            # UNASSIGNED (the Master Mind assigns)
            (r"tests\.test_kingate_contract::test_gate_k_1_must_be_reachable_so_the_identity_control_can_actually_run",
             r"^the T1 gate hook is not written yet$"),
            (r"tests\.test_kingate_contract::test_the_zero_flag_path_must_be_declared_bit_identical",
             r"^the T1 gate hook is not written yet$"),
            # the dev-box half runs CPU-only by the Master Mind's rule (CUDA_VISIBLE_DEVICES=-1)
            (r"tests\.test_refa_v1_precision::test_cuda_bf16_autocast_tiny_step_has_finite_loss_and_grad_norm",
             r"^no CUDA device$"),
        ),
        required_flags=("--residual-prior", "--ego-history", "--slot-deep-supervision", "--slot-vis1"),
        required_flags_why={
            "--slot-deep-supervision": "SPEC_REFCV7 14 (A9 R2): per-layer supervision of both slot heads",
            "--slot-vis1": "SPEC_REFCV7 14 + 15.1 (A9 R3, A10): VIS-1 targets and IGNORE semantics",
        },
        #: the BOX-HEAD requirement BUILT (G-DVB): the box-head package's guard -- the A9 levers ON in argv
        #: AND built on BOTH slot heads, 300 queries, the learned reference points (A14.1)
        #: ⭐ G-LIVE's ADMITTED dead groups (PI 2026-09-27 ~20:55: "We dont need these modules for
        #: refcv7"; MEASURED by the launch smoke on 6fa5e8b: exactly these 10 of 474 leaf groups took
        #: ZERO gradient). Each is BYPASSED BY CONSTRUCTION under a flag of the launch argv and kept
        #: BUILT for strict checkpoint loads; it is admitted ONLY while its flag is in the argv, with its
        #: reason in the evidence -- any OTHER dead group still FAILS G-LIVE. The model-side freeze +
        #: declaration (tanitad/models/_gradreach, batch 3's design) is owed at the next restart: it
        #: touches code the binding runs recorded.
        live_dead_admitted=(
            ("core.strategic.gru", "--no-strategic",
             "the strategic context encoder; --no-strategic (PI 2026-09-06) bypasses the layer, never deletes it"),
            ("core.strategic.proj", "--no-strategic",
             "the strategic context projection; bypassed with the layer"),
            ("nav_to_str", "--no-strategic",
             "nav -> strategic ctx; the nav command still reaches tactical (nav_to_tac) and operative (meas_in)"),
            ("str_goal_head", "--no-strategic",
             "the strategic goal head; its hindsight loss is off with the layer"),
            ("gstr_embed", "--no-strategic",
             "S-BYPASS-2: the strategic goal's FiLM on the tactical latent is skipped"),
            ("gstr_film", "--no-strategic",
             "S-BYPASS-2: the strategic goal's FiLM on the tactical latent is skipped"),
            ("core.decoder.ctx_to_cond", "--no-strategic",
             "S-BYPASS-1: the strategic ctx is not handed to the operative decoder"),
            ("core.route_head", "--no-strategic",
             "the strategic route read-out; route_loss_applied is False under --no-strategic"),
            ("core.decoder.lat_to_anchor", "--graft-tac8-prior",
             "refcv6 4: the tactical 8x8 posterior REPLACES the image-only lat3 prior (refc.py refuses both)"),
            ("core.decoder.lon_to_anchor", "--graft-tac8-prior",
             "refcv6 4: the tactical 8x8 posterior REPLACES the image-only lon3 prior (refc.py refuses both)"),
        ),
        box_required={"module": "tanitad.train.box_head_guard", "fn": "check_refcv7_box_required"},
        dvb_forbid_kinds=("drivort",),
        forbidden_levers=_DRIVORT_OFF,
        forbidden_prefixes=("--r7-", "--w-r7-", "--drivort-", "--w-drivort-"),
        forbidden_built=("refcv7_wta", "refcv7_scorer", "drivort_wta", "drivort_scorer"),
        # SPEC_REFCV7 section 7 (A2, PI ruling E1): all three selection mechanisms ON
        required_on=("--graft-tac8-prior", "--graft-nav-compliance", "--speed-ceiling-filter"),
        # every lever with ONE admissible value, each stated with its source (a flag the argv
        # leaves at the trainer's default is REFUSED: the value must be written in the argv)
        required_values={
            "--residual-prior": "ha0_ext_pose",
            "--map-hires": "on",
            "--bev-source": "map_hires_pool",
            "--map-hires-decision-rule": "prior_corrected",
            "--map-hires-x-max-m": "100",
            "--map-hires-y-half-m": "30",
            "--map-hires-grad-ckpt": "on",
            "--bev-planner-crop-m": ["60", "16"],
            "--w-map": "0",
            "--slot-presence-loss": "focal",
            "--slot-presence-prior": "0.01",
            "--slot-query-select": "learned_ref",
            "--vis1-sidecar": "/home/nvidia/data/refcv7/vis1_sidecar_refcv6b1_train4369_eval139.npz",
            "--map-hires-near-lift-m": "20",
            "--map-hires-near-refine-blocks": "1",
        },
        required_values_why={
            "--residual-prior": "SPEC_REFCV7 10 (A5): the prior is ha0_ext_pose, past poses only",
            "--map-hires": "SPEC_REFCV7 8.1: the 10 cm map head",
            "--bev-source": "SPEC_REFCV7 11.1 (A6 option c): EVERY BEV consumer reads the "
                            "pooled high-resolution BEV",
            "--map-hires-decision-rule": "SPEC_REFCV7 9 item 1 (A4): the prior-corrected argmax",
            "--map-hires-x-max-m": "SPEC_REFCV7 12 (A7): 100 m ahead",
            "--map-hires-y-half-m": "SPEC_REFCV7 12 (A7): +-30 m",
            "--map-hires-grad-ckpt": "SPEC_REFCV7 12 item 4 (A7): checkpointing ON for the "
                                     "10 cm decoder",
            "--bev-planner-crop-m": "SPEC_REFCV7 12 item 2 (A7): the planner BEV cropped to "
                                    "60 m x +-16 m",
            "--w-map": "SPEC_REFCV7 11.1 (A6): the 0.5 m map head and loss are REMOVED",
            "--slot-presence-loss": "SPEC_REFCV7 14 (A9 R1): sigmoid focal presence + the focal matching cost",
            "--slot-presence-prior": "SPEC_REFCV7 14 (A9 R1): the presence prior 0.01",
            "--slot-query-select": "SPEC_REFCV7 19.1 (A14.1): the box head's learned reference points (the "
                                   "full-model one-frame test PASS; the launch ruling 2026-09-27)",
            "--vis1-sidecar": "SPEC_REFCV7 14 (A9 R3): the VIS-1 sidecar placed by the Master Mind 2026-09-27 "
                              "(sha256 278443b3..., the box-head package raw/thor/vis1_full_record.json)",
            "--map-hires-near-lift-m": "SPEC_REFCV7 17 (A12): the 0.1 m near-range lift over "
                                       "x 0-20 m; 23 (A18): the map path G-MAP-OVERFIT ran",
            "--map-hires-near-refine-blocks": "SPEC_REFCV7 20 (A15): ONE near refine block; 23 "
                                              "(A18): A15's configuration",
        },
        #: levers that must be passed with a value strictly above 0 (a built head with no live
        #: weight is declared-but-inert)
        required_positive=("--w-map-hires",),
        # the other prior modes are REFUSED by name (each has its regression arm)
        forbidden_values={"--residual-prior": ("off", "ha0_ext", "cv_yawrate")},
        forbidden_values_why={"--residual-prior": (
            "SPEC_REFCV7 10 (A5): refcv7's prior is ha0_ext_pose; `off` is the refcv6 decoder, "
            "`ha0_ext` reads the steer channel, `cv_yawrate` drops the acceleration")},
        # every ruling is recorded above, so nothing is pending for refcv7 (the mechanism
        # stays: a future open value decision goes here, and is PI-DECISION)
        pi_pending_values={},
        nav_tau_record_default=(
            "TanitAD Research Lab/Architecture & Inference/Research/"
            "2026-09-26-declared-vs-built/raw/nav_compliance_tau_train.json"),
        # A9 / A10 15.2: BOTH slot heads (the planner's agent head and box3d) carry a presence
        # logit named `presence_logit` in the forward output
        live_pres={"key_token": "presence_logit", "min_heads": 2, "gate": 0.5,
                   "max_frac": 0.5},
    ),
}


class GateError(RuntimeError):
    """A condition under which the gate cannot certify anything."""


# --------------------------------------------------------------------------------------------- #
# small utilities                                                                                 #
# --------------------------------------------------------------------------------------------- #
def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str | os.PathLike, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Key-sorted, compact, UTF-8. The ONE byte form every hash and HMAC here is taken over."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=str).encode("utf-8")


def argv_sha256(argv: list[str]) -> str:
    """sha256 of the argv as an ORDERED JSON array (order is meaning; never sort it)."""
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise GateError("argv must be a list of strings")
    return sha256_bytes(json.dumps(argv, ensure_ascii=False, separators=(",", ":"))
                        .encode("utf-8"))


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def scrub(obj: Any) -> Any:
    """Every raw clip UUID -> `sha12:<...>` (the programme's clip-id rule for banked JSON)."""
    if isinstance(obj, str):
        return _UUID.sub(lambda m: "sha12:" + sha12(m.group(0)), obj)
    if isinstance(obj, dict):
        return {scrub(k): scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


def write_json(path: str | os.PathLike, obj: Any) -> Path:
    """Atomic, ASCII-safe (a cp1252 console can never kill the success path)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=True, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return p


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _say(msg: str) -> None:
    print(f"[gate] {msg}", flush=True)


def _harden_streams() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):   # pragma: no cover
            pass


def ram_available_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:                          # noqa: BLE001 -- psutil absent: unknown, not zero
        return None


# --------------------------------------------------------------------------------------------- #
# argv                                                                                            #
# --------------------------------------------------------------------------------------------- #
def load_argv_file(path: str | os.PathLike) -> list[str]:
    """A JSON array of strings, or a run's `config.json` (its `argv`). Nothing else."""
    obj = read_json(path)
    argv = obj.get("argv") if isinstance(obj, dict) else obj
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise GateError(f"{path}: not a non-empty JSON list of strings (nor a config.json argv)")
    return list(argv)


def flag_pairs(argv: list[str]) -> list[tuple[str, list[str]]]:
    """`[--flag v v, --flag2, ...]` -> [(flag, [values])]. `--flag=value` is split."""
    out: list[tuple[str, list[str]]] = []
    for tok in argv:
        if tok.startswith("--"):
            if "=" in tok:
                f, v = tok.split("=", 1)
                out.append((f, [v]))
            else:
                out.append((tok, []))
        elif out:
            out[-1][1].append(tok)
        else:
            raise GateError(f"argv starts with a bare value {tok!r}; the trainer argv is flags only")
    return out


def has_flag(argv: list[str], flag: str) -> bool:
    return any(f == flag for f, _ in flag_pairs(argv))


def flag_values(argv: list[str], flag: str) -> list[str] | None:
    vals = None
    for f, v in flag_pairs(argv):
        if f == flag:
            vals = list(v)
    return vals


def set_flag(argv: list[str], flag: str, values: list[str] | None) -> list[str]:
    """Replace every occurrence of `flag` by ONE `flag values...` (append when absent);
    `values=None` removes the flag."""
    out: list[str] = []
    done = False
    for f, v in flag_pairs(argv):
        if f == flag:
            if values is not None and not done:
                out += [flag, *values]
                done = True
            continue
        out += [f, *v]
    if values is not None and not done:
        out += [flag, *values]
    return out


def _looks_like_path(v: str) -> bool:
    return ("/" in v or "\\" in v) and not v.startswith("--")


def data_inputs(argv: list[str], profile: dict) -> list[tuple[str, str]]:
    """Every (flag, path) the trainer READS: each flag value that is a path, minus the outputs.
    Derived from the argv, not a hand list -- a new path flag is bound the day it is passed."""
    out: list[tuple[str, str]] = []
    outs = set(profile.get("output_flags", ()))
    for f, vals in flag_pairs(argv):
        if f in outs:
            continue
        paths = [v for v in vals if _looks_like_path(v)]
        for i, v in enumerate(paths):
            out.append((f if len(paths) == 1 else f"{f}[{i}]", v))
    return out


def parse_path_map(items: Iterable[str] | None) -> list[tuple[str, str]]:
    pm = []
    for it in items or ():
        if "=" not in it:
            raise GateError(f"--path-map {it!r}: expected FROM=TO")
        a, b = it.split("=", 1)
        pm.append((a.rstrip("/\\"), b.rstrip("/\\")))
    return sorted(pm, key=lambda ab: -len(ab[0]))


def map_path(p: str, pmap: list[tuple[str, str]]) -> str:
    for a, b in pmap:
        if p == a or p.startswith(a + "/") or p.startswith(a + "\\"):
            return b + p[len(a):]
    return p


# --------------------------------------------------------------------------------------------- #
# the code tree                                                                                   #
# --------------------------------------------------------------------------------------------- #
def _skip_tree_path(rel: str) -> bool:
    parts = rel.split("/")
    return (any(p in _TREE_SKIP_DIRS for p in parts[:-1])
            or parts[-1].endswith(_TREE_SKIP_SUFFIX) or parts[-1] in _TREE_SKIP_DIRS)


def tree_manifest(tree: str | os.PathLike) -> dict[str, str]:
    """{posix relpath: sha256} over `stack/` + `taniteval/` -- exactly what ships to a launch host.
    Caches (`__pycache__`, `.pytest_cache`, `*.pyc`) are excluded; everything else counts."""
    tree = Path(tree)
    out: dict[str, str] = {}
    for root in TREE_ROOTS:
        base = tree / root
        if not base.is_dir():
            raise GateError(f"tree {tree} has no {root}/ -- not a code tree")
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _TREE_SKIP_DIRS)
            for fn in filenames:
                full = Path(dirpath) / fn
                rel = full.relative_to(tree).as_posix()
                if _skip_tree_path(rel):
                    continue
                out[rel] = sha256_file(full)
    if not out:
        raise GateError(f"tree {tree}: zero files")
    return out


def tree_digest(manifest: dict[str, str]) -> str:
    d = sha256_bytes("".join(f"{p}\t{h}\n" for p, h in sorted(manifest.items())).encode("utf-8"))
    if not _HEX64.match(d):                    # shape before trust (CLAUDE.md, the blob hole)
        raise GateError("tree digest has the wrong shape")
    return d


def git(git_dir: str, *args: str, binary: bool = False) -> Any:
    env = dict(os.environ, GIT_DIR=str(git_dir))
    p = subprocess.run(["git", *args], env=env, capture_output=True,
                       **({} if binary else {"text": True, "encoding": "utf-8",
                                            "errors": "replace"}))
    if p.returncode != 0:
        raise GateError(f"git {' '.join(args)} failed rc={p.returncode}: "
                        f"{(p.stderr if not binary else p.stderr.decode(errors='replace'))[-400:]}")
    return p.stdout


def resolve_commit(git_dir: str, ref: str) -> str:
    full = git(git_dir, "rev-parse", ref + "^{commit}").strip()
    if not _HEX40.match(full):
        raise GateError(f"{ref!r} does not resolve to a 40-char commit (got {full!r})")
    return full


def git_archive_bytes(git_dir: str, commit: str, roots: Iterable[str]) -> bytes:
    """`git archive` with `core.autocrlf=false`: the REPO's bytes, never a CRLF rewrite."""
    raw = git(git_dir, "-c", "core.autocrlf=false", "archive", commit, *roots, binary=True)
    if not raw:
        raise GateError(f"git archive {commit} produced nothing")
    return raw


def tree_manifest_from_archive(raw: bytes) -> dict[str, str]:
    out: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            rel = m.name
            if rel.split("/", 1)[0] not in TREE_ROOTS or _skip_tree_path(rel):
                continue
            out[rel] = sha256_bytes(tf.extractfile(m).read())
    return out


def extract_archive(raw: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        for m in tf.getmembers():
            if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts:
                raise GateError(f"refusing archive member {m.name!r}")
        tf.extractall(dest)


def tree_diff(want: dict[str, str], got: dict[str, str], limit: int = 20) -> dict:
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    differ = sorted(p for p in set(want) & set(got) if want[p] != got[p])
    return {"n_missing": len(missing), "n_extra": len(extra), "n_differ": len(differ),
            "missing": missing[:limit], "extra": extra[:limit], "differ": differ[:limit]}


# --------------------------------------------------------------------------------------------- #
# data fingerprints                                                                               #
# --------------------------------------------------------------------------------------------- #
def fingerprint(path: str | os.PathLike) -> dict:
    """A file: its sha256. A directory: its LISTING (relpath + size of every file; content NOT
    hashed -- a 161 GB cache is not re-read per launch; stated in `method`). Missing: missing."""
    p = Path(path)
    if p.is_file():
        return {"kind": "file", "bytes": p.stat().st_size, "sha256": sha256_file(p),
                "method": "sha256-v1"}
    if p.is_dir():
        lines, n, total = [], 0, 0
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = sorted(d for d in dirnames if d not in _DIR_SKIP_NAMES)
            for fn in filenames:
                if fn in _DIR_SKIP_NAMES:
                    continue
                f = Path(dirpath) / fn
                st = f.stat()
                lines.append(f"{f.relative_to(p).as_posix()}\t{st.st_size}\n")
                n += 1
                total += st.st_size
        lines.sort()
        return {"kind": "dir", "n_files": n, "bytes": total,
                "listing_sha256": sha256_bytes("".join(lines).encode("utf-8")),
                "method": "dir-listing-v1 (relpath + size of every file; contents NOT hashed; "
                          "_v2manifest.pt excluded -- a cache rewrites it on first use)"}
    return {"kind": "missing"}


def fp_key(fp: dict) -> str:
    """The comparable identity of a fingerprint (method + content digest)."""
    if fp.get("kind") == "file":
        return f"file:{fp.get('sha256')}"
    if fp.get("kind") == "dir":
        return f"dir:{fp.get('n_files')}:{fp.get('bytes')}:{fp.get('listing_sha256')}"
    return f"{fp.get('kind')}:{fp.get('sha256', fp.get('why', ''))}"


def implicit_inputs(argv: list[str]) -> list[tuple[str, dict]]:
    """Inputs the argv implies but does not name. Today: the ImageNet init a `--trunk timm` run
    loads from the Hugging Face cache (refcv7 trains from it -- SPEC section 4)."""
    out: list[tuple[str, dict]] = []
    trunk = flag_values(argv, "--trunk")
    name = flag_values(argv, "--trunk-name")
    if trunk and trunk[0] == "timm" and name and not has_flag(argv, "--trunk-no-pretrained"):
        key = f"implicit:timm-pretrained:{name[0]}"
        try:
            import timm                                   # noqa: F401
            from huggingface_hub import try_to_load_from_cache
            pc = timm.models.get_pretrained_cfg(name[0])
            hub = getattr(pc, "hf_hub_id", None) if pc is not None else None
            if not hub:
                out.append((key, {"kind": "unresolved", "why": f"no hf_hub_id for {name[0]}"}))
                return out
            for fn in ("model.safetensors", "pytorch_model.bin"):
                r = try_to_load_from_cache(hub, fn)
                if isinstance(r, str) and os.path.isfile(r):
                    fp = fingerprint(r)
                    fp["hub"] = f"{hub}/{fn}"
                    out.append((key, fp))
                    return out
            out.append((key, {"kind": "unresolved", "why": f"{hub} not in the local HF cache"}))
        except Exception as e:                            # noqa: BLE001 -- stated, never guessed
            out.append((key, {"kind": "unresolved", "why": f"{type(e).__name__}: {e}"}))
    return out


def data_manifest(argv: list[str], profile: dict, pmap: list[tuple[str, str]],
                  gate_inputs: dict[str, str] | None = None) -> dict[str, dict]:
    """Every input the launch reads, fingerprinted on THIS host: the argv's data paths, the
    `.meta.json` beside any of them (a sidecar's provenance is content too; the profile's
    `meta_required_flags` make an ABSENT one a finding), the implied inputs (ImageNet init), and
    the gate's own inputs (a clock reference, a recorded tau) keyed `gate:*`."""
    man: dict[str, dict] = {}
    req = set(profile.get("meta_required_flags", ()))
    for flag, p in data_inputs(argv, profile):
        local = map_path(p, pmap)
        man[flag] = {"path": p, **({"local": local} if local != p else {}), **fingerprint(local)}
        meta = local + ".meta.json"
        if flag in req or os.path.isfile(meta):
            man[f"{flag}:meta"] = {"path": p + ".meta.json", **fingerprint(meta),
                                   **({"required": True} if flag in req else {})}
    for key, fp in implicit_inputs(argv):
        man[key] = fp
    for key, path in sorted((gate_inputs or {}).items()):
        man[key] = {"path": str(path), **fingerprint(path)}
    return man


def data_manifest_problems(data: dict[str, dict]) -> list[str]:
    out = []
    for k, v in sorted(data.items()):
        if v.get("kind") not in ("missing", "unresolved"):
            continue
        if k.endswith(":meta") and v.get("required"):
            out.append(f"FIX-5: {v.get('path')} is ABSENT -- `read_speed_max_sidecar_v6` refuses "
                       f"a sidecar without its meta (it is the only record of which label blob "
                       f"the sidecar was built over)")
        else:
            out.append(f"data input {k} ({v.get('path', '')}) is {v.get('kind')} on this host"
                       + (f": {v.get('why')}" if v.get("why") else ""))
    return out


# --------------------------------------------------------------------------------------------- #
# the key and the signature                                                                       #
# --------------------------------------------------------------------------------------------- #
def default_key_file() -> Path:
    env = os.environ.get(KEY_ENV)
    return Path(env) if env else Path.home() / ".config" / "tanitad" / "launch_gate.key"


def load_key(path: str | os.PathLike | None = None, *, create: bool = True,
             forbid_under: Iterable[str | os.PathLike] = ()) -> bytes:
    """32 random bytes, hex on disk, created on first use with mode 0600. ⛔ Refused inside a
    code tree: a key that can be committed is not a key (it is also git-ignored by name)."""
    p = Path(path) if path else default_key_file()
    rp = os.path.normcase(str(p.resolve()))
    for t in forbid_under:
        rt = os.path.normcase(str(Path(t).resolve()))
        if rp == rt or rp.startswith(rt + os.sep):
            raise GateError(f"refusing a key file inside the code tree {t}: {p}")
    if not p.exists():
        if not create:
            raise GateError(f"no gate key at {p} (create one by running the gate on this host)")
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            fh.write(secrets.token_hex(32) + "\n")
        try:
            os.chmod(p, 0o600)
        except OSError:                                   # pragma: no cover (Windows ACLs)
            pass
    txt = p.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", txt):
        raise GateError(f"gate key {p} is not 64 hex chars")
    return bytes.fromhex(txt)


def key_id(key: bytes) -> str:
    return sha256_bytes(b"tanitad-launch-gate-key-id:" + key)[:16]


def sign(token_wo_sig: dict, key: bytes) -> str:
    return hmac.new(key, canonical_json(token_wo_sig), hashlib.sha256).hexdigest()


def signature_ok(token: dict, key: bytes) -> bool:
    body = {k: v for k, v in token.items() if k != "hmac_sha256"}
    got = str(token.get("hmac_sha256", ""))
    return len(got) == 64 and hmac.compare_digest(sign(body, key), got)


# --------------------------------------------------------------------------------------------- #
# context                                                                                         #
# --------------------------------------------------------------------------------------------- #
@dataclasses.dataclass
class Ctx:
    profile: str
    tree: str
    commit: str
    argv: list
    out_dir: str
    path_map: list
    tree_sha256: str
    argv_sha256: str
    options: dict
    arm: str | None = None

    @property
    def prof(self) -> dict:
        return PROFILES[self.profile]

    def binding(self) -> dict:
        return {"profile": self.profile, "commit": self.commit, "tree_sha256": self.tree_sha256,
                "argv_sha256": self.argv_sha256}

    def dump(self, p: Path) -> Path:
        return write_json(p, dataclasses.asdict(self))

    @classmethod
    def load(cls, p: str | os.PathLike) -> "Ctx":
        return cls(**read_json(p))

    def local_argv(self) -> list[str]:
        """The argv with every data path mapped to this host (the binding keeps the original)."""
        out, pmap = [], [tuple(x) for x in self.path_map]
        outs = set(self.prof.get("output_flags", ()))
        for f, vals in flag_pairs(self.argv):
            out.append(f)
            out += [map_path(v, pmap) if (f not in outs and _looks_like_path(v)) else v
                    for v in vals]
        return out


def _gate_self_sha() -> str:
    return sha256_file(Path(__file__).resolve())


def _host() -> dict:
    h = {"node": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch
        h["torch"] = torch.__version__
        h["cuda"] = bool(torch.cuda.is_available())
        if h["cuda"]:
            h["device"] = torch.cuda.get_device_name(0)
    except Exception:                                     # noqa: BLE001
        pass
    return h


def new_evidence(ctx: Ctx, check: str) -> dict:
    return {"schema": GATE_SCHEMA, "kind": "evidence", "check": check, "status": "ERROR",
            "reasons": [], "binding": ctx.binding(), "inputs_read": {}, "host": _host(),
            "arm": ctx.arm, "gate_sha256": _gate_self_sha(), "started_utc": utc_now(),
            "details": {}}


def finish_evidence(ev: dict, status: str, reasons: list[str] | None = None) -> dict:
    ev["status"] = status
    ev["reasons"] = list(reasons or [])
    ev["finished_utc"] = utc_now()
    return ev


def write_evidence(out_dir: str | os.PathLike, ev: dict) -> Path:
    d = Path(out_dir) / "evidence"
    ev = scrub(ev)
    p = write_json(d / f"{ev['check']}.json", ev)
    hist = d / "history" / f"{ev['check']}__{ev.get('finished_utc', utc_now()).replace(':', '')}.json"
    write_json(hist, ev)
    return p


# --------------------------------------------------------------------------------------------- #
# child side: the tree under test, the trainer, the capture                                        #
# --------------------------------------------------------------------------------------------- #
class _Stop(Exception):
    """Raised by a capture hook to end `train()` at a chosen point (never escapes the gate)."""


class _ForwardCaptured(Exception):
    def __init__(self, out):
        super().__init__("forward captured")
        self.out = out


class _Patcher:
    def __init__(self):
        self._undo: list[tuple[Any, str, Any, bool]] = []

    def set(self, obj: Any, name: str, value: Any) -> None:
        had = name in getattr(obj, "__dict__", {})
        self._undo.append((obj, name, getattr(obj, name, None), had))
        setattr(obj, name, value)

    def restore(self) -> None:
        for obj, name, old, had in reversed(self._undo):
            if had:
                setattr(obj, name, old)
            else:
                try:
                    delattr(obj, name)
                except AttributeError:
                    setattr(obj, name, old)
        self._undo.clear()


def child_bootstrap(ctx: Ctx) -> dict:
    """sys.path on the tree under test, and ASSERT `tanitad` really came from it (the MSYS /
    editable-install false pass: a green run on the wrong disk)."""
    tree = Path(ctx.tree).resolve()
    for p in (tree / "taniteval" / "tools", tree / "stack" / "scripts", tree / "taniteval",
              tree / "stack"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    sys.dont_write_bytecode = True
    import tanitad
    here = os.path.normcase(os.path.realpath(tanitad.__file__))
    want = os.path.normcase(os.path.realpath(tree / "stack"))
    if not here.startswith(want + os.sep):
        raise GateError(f"tanitad imported from {here}, not from the tree under test {want}")
    import torch
    torch.set_num_threads(int(ctx.options.get("omp", 4)))
    if ctx.options.get("cpu_only") and torch.cuda.is_available():
        raise GateError("--cpu-only but CUDA is visible (on Windows set CUDA_VISIBLE_DEVICES=-1; "
                        "an EMPTY value deletes the variable)")
    return {"tanitad": here, "torch": torch.__version__, "cuda": bool(torch.cuda.is_available())}


_TRAINER_MOD = "refc_v3_train_gate"


def load_trainer(ctx: Ctx):
    if _TRAINER_MOD in sys.modules:
        return sys.modules[_TRAINER_MOD]
    path = Path(ctx.tree) / ctx.prof["trainer"]
    spec = importlib.util.spec_from_file_location(_TRAINER_MOD, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_TRAINER_MOD] = mod
    spec.loader.exec_module(mod)
    return mod


def arm_is(ctx: Ctx, name: str) -> bool:
    return ctx.arm == name


class _ImportBlocker:
    """A meta-path finder that makes ONE module un-importable (the `missing_*_module` arms)."""

    def __init__(self, fullname: str):
        self.fullname = fullname

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.fullname:
            raise ModuleNotFoundError(f"No module named {fullname!r} (gate arm)")
        return None


def import_optional(ctx: Ctx, fullname: str, arm: str):
    """-> (module | None, why). The arm makes the import fail exactly as an unlanded module would."""
    if arm_is(ctx, arm):
        blocker = _ImportBlocker(fullname)
        sys.meta_path.insert(0, blocker)
        sys.modules.pop(fullname, None)
        try:
            importlib.import_module(fullname)
        except ModuleNotFoundError as e:
            return None, f"{type(e).__name__}: {e}"
        finally:
            sys.meta_path.remove(blocker)
    try:
        return importlib.import_module(fullname), ""
    except ModuleNotFoundError as e:
        return None, f"{type(e).__name__}: {e}"
    except Exception as e:                                # noqa: BLE001 -- a broken module is absent
        return None, f"{type(e).__name__} while importing: {e}"


@contextlib.contextmanager
def trainer_capture(T, stop_at: str, on_built: Callable[[Any], Any] | None = None):
    """Run the REAL `train()` and capture what it built. `stop_at`:
      * "config" -- raise at `RefCV3Model(cfg)` before the model is built (cfg captured);
      * "model"  -- let the model build COMPLETE (anchors, rig camera, perception branch...) and
                    raise at the first data-source call, before any training data is read.
    `on_built(model)` runs the moment the model is constructed, before `train()` sees it -- the
    hook a regression arm uses to be BORN with a defect (its return value lands in
    `cap["on_built"]`); never set outside an arm."""
    from tanitad.refs import refc_v3 as v3
    cap: dict[str, Any] = {}
    pat = _Patcher()
    orig_init = v3.RefCV3Model.__init__
    orig_train = T.train

    def init_wrap(self, cfg, *a, **k):
        cap["cfg"] = cfg
        if stop_at == "config":
            raise _Stop("config captured")
        orig_init(self, cfg, *a, **k)
        if on_built is not None:
            cap["on_built"] = on_built(self)
        cap["model"] = self

    def train_wrap(args):
        cap["args"] = args
        return orig_train(args)

    def stop(*_a, **_k):
        raise _Stop("model captured at the data source")

    pat.set(v3.RefCV3Model, "__init__", init_wrap)
    pat.set(T, "train", train_wrap)
    # FIX-5 OBSERVED, not assumed: the per-stage ImageNet fingerprint check the trunk build runs
    # (`timm_trunk._assert_stages_pretrained`, fixes agent) is recorded with its fingerprints --
    # read BEFORE any BN fold rewrites the conv weights -- on the host that builds.
    try:
        from tanitad.models import timm_trunk as _tt
    except Exception:                                     # noqa: BLE001
        _tt = None
    if _tt is not None and callable(getattr(_tt, "_assert_stages_pretrained", None)):
        orig_asp = _tt._assert_stages_pretrained

        def asp_wrap(net, model_name, *a, **k):
            r: dict[str, Any] = {"model_name": str(model_name)}
            try:
                if callable(getattr(_tt, "stage_fingerprints", None)):
                    r["fingerprints"] = dict(_tt.stage_fingerprints(net))
            except Exception as e:                        # noqa: BLE001
                r["fingerprints_error"] = f"{type(e).__name__}: {e}"
            try:
                out = orig_asp(net, model_name, *a, **k)
                r["outcome"] = "PASS"
                return out
            except BaseException as e:
                r["outcome"] = f"{type(e).__name__}: {str(e)[:400]}"
                raise
            finally:
                cap.setdefault("fix5", []).append(r)
        pat.set(_tt, "_assert_stages_pretrained", asp_wrap)
        cap["fix5_present"] = True
    else:
        cap["fix5_present"] = False
    if stop_at == "model":
        for name in ("refuse_eval_clips_in_train", "_synth_episodes", "load_cached_episodes"):
            if hasattr(T, name):
                pat.set(T, name, stop)
    try:
        yield cap
    finally:
        pat.restore()


def run_trainer_until(T, argv: list[str], stop_at: str,
                      on_built: Callable[[Any], Any] | None = None) -> dict:
    with trainer_capture(T, stop_at, on_built) as cap:
        try:
            T.main(list(argv))
        except _Stop:
            pass
    if stop_at == "model" and "model" not in cap:
        raise GateError("the trainer returned without building a model (capture missed)")
    if "cfg" not in cap:
        raise GateError("the trainer never reached RefCV3Model(cfg)")
    return cap


def tensor_digest(t) -> str:
    import torch
    x = t.detach()
    if x.device.type != "cpu":
        x = x.cpu()
    # ⛔ `.contiguous()` is NOT enough: a size-1 dim keeps any stride and still counts as
    # contiguous, so a flatten can come back strided (MEASURED 2026-09-26: stride 12 on a forward
    # output). A fresh dense buffer is the only safe source of bytes.
    flat = x.reshape(-1)
    dense = torch.empty(flat.shape, dtype=flat.dtype)
    dense.copy_(flat)
    b = dense.view(torch.uint8).numpy().tobytes() if dense.numel() else b""
    return sha256_bytes(f"{x.dtype}|{tuple(t.shape)}|".encode() + b)


def state_digests(sd: dict) -> dict[str, str]:
    import torch
    return {k: (tensor_digest(v) if torch.is_tensor(v) else f"py:{v!r}") for k, v in sd.items()}


def optimizer_digests(osd: dict) -> dict:
    import torch
    st = {}
    for idx, s in osd.get("state", {}).items():
        for k, v in s.items():
            st[f"{idx}.{k}"] = tensor_digest(v) if torch.is_tensor(v) else f"py:{v!r}"
    groups = []
    for g in osd.get("param_groups", []):
        groups.append({k: (list(v) if isinstance(v, (list, tuple)) else v)
                       for k, v in g.items()})
    return {"state": st, "param_groups": groups}


def output_digests(out: Any, prefix: str = "out") -> dict[str, str]:
    """Flatten a forward's output (dicts / lists / tensors / scalars) to {path: digest}."""
    import torch
    res: dict[str, str] = {}
    if torch.is_tensor(out):
        res[prefix] = tensor_digest(out)
    elif isinstance(out, dict):
        for k in sorted(out, key=str):
            res.update(output_digests(out[k], f"{prefix}.{k}"))
    elif isinstance(out, (list, tuple)):
        for i, v in enumerate(out):
            res.update(output_digests(v, f"{prefix}[{i}]"))
    elif isinstance(out, (int, float, bool, str)) or out is None:
        res[prefix] = f"py:{out!r}"
    else:
        res[prefix] = f"type:{type(out).__name__}"
    return res


# --------------------------------------------------------------------------------------------- #
# G-HYG                                                                                           #
# --------------------------------------------------------------------------------------------- #
_HYG_PROBE_ATTR = "_g_hyg_probe_undeclared_attribute"


def _dataclass_instances(root: Any, path: str = "cfg") -> list[tuple[str, Any]]:
    out, seen, stack = [], set(), [(path, root)]
    while stack:
        p, obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            out.append((p, obj))
            items = [(f".{k}", v) for k, v in vars(obj).items()]
        elif isinstance(obj, (list, tuple)):
            items = [(f"[{i}]", v) for i, v in enumerate(obj)]
        elif isinstance(obj, dict):
            items = [(f"[{k!r}]", v) for k, v in obj.items()]
        else:
            items = []
        for s, v in items:
            if dataclasses.is_dataclass(v) or isinstance(v, (list, tuple, dict)):
                stack.append((p + s, v))
    return sorted(out, key=lambda pv: pv[0])


#: the tree's OWN import roots (never checked against the venv: they ARE the tree)
_OWN_ROOTS = ("tanitad", "taniteval")
_GUARD_EXC = ("ImportError", "ModuleNotFoundError", "Exception", "BaseException")


def import_closure(tree: Path, entry_rel: str) -> dict:
    """STATIC import closure of the launch: the trainer script and every module of THIS tree it
    imports, transitively -- module-level AND function-level imports (a function-level `import
    scipy` crashes the run the first time that path executes, possibly hours in). Returns every
    third-party top-level name with its import sites, each marked `guarded` when it sits inside a
    `try:` whose handler catches ImportError (an optional dependency). Nothing is imported."""
    import ast
    roots = [tree / "stack", tree / "taniteval", tree / "stack" / "scripts"]
    std = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}

    def resolve(mod: str) -> Path | None:
        parts = mod.split(".")
        for base in roots:
            p = base.joinpath(*parts)
            if p.with_suffix(".py").is_file():
                return p.with_suffix(".py")
            if (p / "__init__.py").is_file():
                return p / "__init__.py"
        return None

    def pkg_of(f: Path) -> list[str]:
        for base in roots:
            try:
                rel = f.relative_to(base)
            except ValueError:
                continue
            parts = list(rel.with_suffix("").parts)
            return parts[:-1] if parts and parts[-1] != "__init__" else parts[:-1]
        return []

    entry = tree / entry_rel
    queue, seen = [entry], set()
    third: dict[str, list[dict]] = {}
    while queue:
        f = queue.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        # utf-8-sig, as Python decodes a source file: `tanitad/eval/__init__.py` starts with a
        # UTF-8 BOM, and `ast.parse` of a plain utf-8 decode dies on U+FEFF (the launch config's
        # G-HYG crashed on it, Master Mind 2026-09-27)
        mod = ast.parse(f.read_text(encoding="utf-8-sig", errors="replace"), filename=str(f))
        guarded: set[int] = set()
        fn_level: set[int] = set()
        for node in ast.walk(mod):
            if isinstance(node, ast.Try) and any(
                    h.type is None or any(isinstance(n, ast.Name) and n.id in _GUARD_EXC
                                          for n in ast.walk(h.type)) for h in node.handlers):
                for stmt in node.body:
                    for sub in ast.walk(stmt):
                        if isinstance(sub, (ast.Import, ast.ImportFrom)):
                            guarded.add(id(sub))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                # a function-level import runs only when that function runs (a lazy import)
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        fn_level.add(id(sub))
        for node in ast.walk(mod):
            full: list[str] = []
            if isinstance(node, ast.Import):
                full = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg_of(f)[: len(pkg_of(f)) - (node.level - 1) or None]
                    stem = ".".join([*base, *([node.module] if node.module else [])])
                else:
                    stem = node.module or ""
                full = [stem] + [f"{stem}.{a.name}" for a in node.names if stem]
            for name in full:
                if not name:
                    continue
                top = name.split(".")[0]
                own = top in _OWN_ROOTS or resolve(name) is not None
                if own:
                    target = resolve(name)
                    if target is not None:
                        queue.append(target)
                    continue
                if top in std or isinstance(node, ast.ImportFrom) and name != full[0]:
                    continue
                third.setdefault(top, []).append({
                    "file": f.relative_to(tree).as_posix(), "line": int(node.lineno),
                    "guarded": id(node) in guarded, "function_level": id(node) in fn_level})
    return {"entry": entry_rel, "n_files": len(seen), "third_party": third}


def _arm_import_ban(ctx: "Ctx") -> tuple[str, ...]:
    """Arm `venv_lacks_module`: the gate treats a module the launch path imports at MODULE level
    (pandas: `tanitad/data/physicalai.py`) as absent from the venv -- the check must go RED."""
    return ("pandas",) if arm_is(ctx, "venv_lacks_module") else ()


#: modules MEASURED absent from a launch venv: ANY import site in the closure is a FAIL, even a
#: lazy one inside a function (Master Mind 2026-09-27: "scipy is ABSENT from Thor's training venv.
#: If ANY refcv7 production path imports scipy, G-LIVE will crash")
IMPORT_STRICT = ("scipy",)


def judge_import_closure(clo: dict, *, ban: Iterable[str] = (),
                         strict: Iterable[str] = IMPORT_STRICT) -> tuple[list[str], dict]:
    """Each third-party name the closure found must resolve in THIS interpreter's venv
    (`importlib.util.find_spec`, which imports nothing).

    * a missing module imported UNGUARDED at MODULE level (it runs when the launch imports that
      module) is a FAIL;
    * a missing module in the STRICT list is a FAIL at ANY unguarded site, lazy ones included;
    * a missing module imported only inside a function (a deliberate lazy import, e.g.
      `metadrive_env._make_env`) or only under `try/except ImportError` is REPORTED, not gated --
      a static closure cannot show that function runs on the launch path; the smoke does."""
    banned, strict = set(ban), set(strict)
    hard: dict[str, list] = {}
    lazy: dict[str, list] = {}
    soft: dict[str, list] = {}
    for top, sites in sorted((clo.get("third_party") or {}).items()):
        try:
            present = top not in banned and importlib.util.find_spec(top) is not None
        except (ImportError, ValueError):
            present = False
        if present:
            continue
        unguarded = [s for s in sites if not s.get("guarded")]
        eager = [s for s in unguarded if not s.get("function_level")]
        if eager or (top in strict and unguarded):
            hard[top] = (eager or unguarded)[:6]
        elif unguarded:
            lazy[top] = unguarded[:6]
        else:
            soft[top] = sites[:6]
    det = {"venv": sys.executable, "n_files": clo.get("n_files"),
           "third_party": sorted(clo.get("third_party") or {}), "strict": sorted(strict),
           "missing_fail": hard, "missing_lazy_only": lazy, "missing_guarded_only": soft}
    reasons = [f"G-HYG import closure: `{top}` is NOT importable in this venv ({sys.executable}) "
               f"and the launch path imports it UNGUARDED at "
               f"{', '.join(s['file'] + ':' + str(s['line']) for s in sites[:3])}"
               f"{' (+more)' if len(sites) > 3 else ''} -- a crash when that code runs "
               f"(MEASURED 2026-09-27: scipy is absent from Thor's training venv)"
               for top, sites in hard.items()]
    if not clo.get("n_files"):
        reasons.append("G-HYG import closure: the closure read ZERO files -- the check read nothing")
    return reasons, det


def judge_hygiene(H, cfg) -> tuple[list[str], dict]:
    """The pure G-HYG verdict over a pinned config tree, given the hygiene module `H`."""
    reasons: list[str] = []
    det: dict[str, Any] = {}
    for name in ("undeclared_attributes", "is_strict"):
        if not callable(getattr(H, name, None)):
            reasons.append(f"tanitad.train.config_hygiene has no callable `{name}` (API drift)")
    if reasons:
        return reasons, det
    bad = [list(map(str, x)) for x in H.undeclared_attributes(cfg)]
    det["undeclared_attributes"] = bad
    if bad:
        reasons.append(f"{len(bad)} UNDECLARED attribute(s) on the trainer's pinned config: "
                       + "; ".join(".".join(x[::2]) + f" (on {x[1]})" for x in bad[:8]))
    inst = _dataclass_instances(cfg)
    classes = sorted({f"{type(o).__module__}.{type(o).__qualname__}" for _, o in inst})
    det["n_config_dataclass_instances"] = len(inst)
    det["config_classes"] = classes
    not_strict = sorted({f"{type(o).__module__}.{type(o).__qualname__}" for _, o in inst
                         if not H.is_strict(o)})
    det["not_strict"] = not_strict
    if not_strict:
        reasons.append(f"{len(not_strict)} config dataclass(es) do NOT refuse undeclared "
                       f"attributes at assignment: {', '.join(not_strict[:8])}")
    # ⛔ the ACTIVE probe -- the regression arm executed as a positive control: setting an
    # undeclared attribute on EVERY config object (on a deep copy) must RAISE.
    accepted = []
    try:
        probe = copy.deepcopy(cfg)
    except Exception as e:                                # noqa: BLE001
        reasons.append(f"the pinned config cannot be deep-copied for the probe: {e}")
        return reasons, det
    for p, o in _dataclass_instances(probe):
        try:
            setattr(o, _HYG_PROBE_ATTR, 1)
        except (AttributeError, TypeError, dataclasses.FrozenInstanceError):
            continue
        accepted.append(f"{p} ({type(o).__qualname__})")
    det["probe_assignments_accepted"] = accepted
    det["probe_assignments_tried"] = len(inst)
    if accepted:
        reasons.append(f"setting an undeclared attribute SUCCEEDED on {len(accepted)} config "
                       f"object(s) (the D-REFCV6-EQUALIZE-DROPPED mechanism): {accepted[:6]}")
    if not inst:
        reasons.append("the probe walked ZERO config dataclasses -- an instrument that read "
                       "nothing certifies nothing")
    return reasons, det


# --------------------------------------------------------------------------------------------- #
# G-DVB                                                                                           #
# --------------------------------------------------------------------------------------------- #
def _find_trunk(model):
    for m in model.modules():
        if type(m).__name__ == "TimmResNetTrunk":
            return m
    return None


def equalize_probe(model) -> dict:
    """MEASURE how many bottom rows the BUILT trunk equalises -- by behaviour, not by reading a
    config field: a flat 0.5 frame goes through `normalise`; the equalised rows come out at the
    value a 0.0 pixel normalises to, the others do not. Counters restored afterwards."""
    import torch
    trunk = _find_trunk(model)
    if trunk is None:
        return {"trunk": None}
    enc_cfg = getattr(getattr(getattr(model, "cfg", None), "core", None), "encoder", None)
    h, w = (enc_cfg.image_hw() if enc_cfg is not None else (416, 1024))
    c = int(getattr(enc_cfg, "in_channels", 9))
    saved = {k: getattr(trunk, k) for k in ("norm_calls", "equalize_calls") if hasattr(trunk, k)}
    try:
        with torch.no_grad():
            dev = trunk._mean.device
            x = torch.full((1, c, h, w), 0.5, device=dev)
            y = trunk.normalise(x).float().cpu()
            z = trunk.normalise(torch.zeros_like(x)).float().cpu()
    finally:
        for k, v in saved.items():
            setattr(trunk, k, v)
        if "equalize_calls" not in saved and hasattr(trunk, "equalize_calls"):
            delattr(trunk, "equalize_calls")
    rows = 0
    for r in range(h - 1, -1, -1):
        if torch.equal(y[..., r, :], z[..., r, :]):
            rows += 1
        else:
            break
    return {"trunk": type(trunk).__name__, "image_hw": [h, w], "rows_equalised_measured": rows,
            "trunk_cfg_equalize_bottom_rows": int(getattr(trunk.cfg, "equalize_bottom_rows", 0)
                                                  or 0)}


def forbidden_lever_reasons(prof: dict, argv: list[str], model=None) -> tuple[list[str], dict]:
    """Levers the PROFILE forbids (refcv7: DrivoR-T's). Read off the argv by flag NAME -- a
    passed flag is refused whatever its value ("a refcv7 launch passes none of those flags") --
    and off the BUILT object, so a default that turns one on is caught too."""
    reasons: list[str] = []
    det: dict[str, Any] = {"argv_hits": [], "built_hits": []}
    flags = [f for f, _ in flag_pairs(argv)]
    for names, why in prof.get("forbidden_levers", ()):
        hit = [f for f in flags if f in names]
        if hit:
            det["argv_hits"] += hit
            reasons.append(f"{hit[0]} is passed, but it is {why}: it must be OFF for a "
                           f"{prof['name']} launch (pass none of {list(names)})")
    for f in flags:
        if any(f.startswith(p) for p in prof.get("forbidden_prefixes", ())):
            det["argv_hits"].append(f)
            reasons.append(f"{f} is a DrivoR-T parameter flag; a {prof['name']} launch passes "
                           f"none of them")
    if model is not None:
        for attr in prof.get("forbidden_built", ()):
            if getattr(model, attr, None) is not None:
                det["built_hits"].append(attr)
                reasons.append(f"the BUILT model carries `{attr}` (a DrivoR-T module) -- it must "
                               f"be None for a {prof['name']} launch")
    return reasons, det


def forbidden_value_reasons(prof: dict, argv: list[str]) -> list[str]:
    out = []
    for f, bad in (prof.get("forbidden_values") or {}).items():
        v = flag_values(argv, f)
        if v is not None and (not v or v[0] in bad):
            why = (prof.get("forbidden_values_why") or {}).get(f)
            out.append(f"{f} {v} is REFUSED for a {prof['name']} launch (refused values: "
                       f"{list(bad)}" + (f"; {why})" if why else ")"))
    return out


def required_flag_reasons(prof: dict, argv: list[str]) -> list[str]:
    why = dict(prof.get("required_flags_why") or {})
    return [f"profile {prof['name']} requires {f} in the launch argv "
            f"({why.get(f, 'SPEC: NEW-1 is part of this arm')})"
            for f in prof.get("required_flags", ()) if not has_flag(argv, f)]


def box_required_reasons(prof: dict, model=None, args=None) -> tuple[list[str], dict]:
    """The BOX-HEAD requirement BUILT (the box-head package's guard, closed open item BOX-HEAD, 2026-09-27):
    `prof["box_required"]` names a module + function `fn(model, args) -> list[Mismatch]` ([] = a refcv7 box
    build: the A9 levers ON in argv AND built on both slot heads, 300 queries, the learned reference points).
    Inert for a profile that names none; a missing module or function is a named FAIL, never a pass."""
    spec = prof.get("box_required")
    if not spec:
        return [], {"box_required": None}
    det: dict[str, Any] = {"box_required": f"{spec.get('module')}.{spec.get('fn')}"}
    try:
        mod = importlib.import_module(str(spec.get("module")))
    except Exception as e:                                # noqa: BLE001 -- recorded, never a pass
        return [f"BOX-HEAD: {spec.get('module')} is not importable ({type(e).__name__}: {e}) -- the box "
                f"head's requirement cannot be read"], det
    fn = getattr(mod, str(spec.get("fn")), None)
    if not callable(fn):
        return [f"BOX-HEAD: {spec.get('module')} has no `{spec.get('fn')}` -- the box head's requirement "
                f"cannot be read"], det
    if model is None or args is None:
        return ["BOX-HEAD: no built model to check the box head against"], det
    try:
        res = fn(model, args)
    except Exception as e:                                # noqa: BLE001
        return [f"BOX-HEAD: {spec.get('fn')} crashed: {type(e).__name__}: {str(e)[:300]}"], det
    det["result"] = [str(x)[:300] for x in (res or [])] if isinstance(res, (list, tuple)) else repr(res)[:300]
    if not isinstance(res, (list, tuple)):
        return [f"BOX-HEAD: {spec.get('fn')} returned {type(res).__name__}, not a list"], det
    return ([f"BOX-HEAD: {len(res)} requirement(s) not met (SPEC_REFCV7 A9/A14.1): "
             + " | ".join(str(x)[:240] for x in list(res)[:6])] if res else []), det


def pi_pending_reasons(prof: dict, argv: list[str]) -> list[str]:
    """A flag whose VALUE awaits a PI ruling makes the token PI-DECISION -- unless the ruling is
    already recorded as a `required_values` entry (which is then checked for equality)."""
    fixed = dict(prof.get("required_values") or {})
    out = []
    for f, why in (prof.get("pi_pending_values") or {}).items():
        if f in fixed:
            continue
        v = flag_values(argv, f)
        if v:
            out.append(f"{f} {v[0]}: {why}")
    return out


def profile_argv_rules(prof: dict, argv: list[str], options: dict | None = None,
                       pmap: list[tuple[str, str]] | None = None) -> tuple[list[str], list[str]]:
    """The profile's rules that read the ARGV alone -> (refusals, pi_pending). ONE definition,
    used by `run` (to say it early) AND by `finalize` itself, so no path to a token -- including
    a stand-alone `finalize` -- can skip them."""
    options = options or {}
    refusals = required_flag_reasons(prof, argv)
    refusals += forbidden_value_reasons(prof, argv)
    refusals += forbidden_lever_reasons(prof, argv)[0]
    refusals += nav_tau_reasons(prof, argv, options.get("nav_tau_record"), pmap)[0]
    refusals += required_on_reasons(prof, argv)[0]
    return refusals, pi_pending_reasons(prof, argv)


def open_item_reasons(prof: dict) -> list[str]:
    """The profile's OPEN ITEMS, one named reason each (empty when the flag set is decided)."""
    return [f"OPEN ITEM {it.get('id')}: {it.get('what')} (owner: {it.get('owner')})"
            for it in (prof.get("open_items") or ())]


def residual_prior_on(argv: list[str]) -> bool:
    """NEW-1 is ON only when `--residual-prior` carries a mode other than `off`."""
    v = flag_values(argv, "--residual-prior")
    return bool(v) and v[0] != "off"


def hires_on(prof: dict, argv: list[str]) -> bool:
    h = prof.get("map_hires") or {}
    sw = h.get("switch")
    if not sw:
        return False
    v = flag_values(argv, sw[0])
    return v is not None and (v == [sw[1]] or (not v and sw[1] in ("", "on")))


def required_checks_of(prof: dict, argv: list[str]) -> list[str]:
    """The checks a token for THIS argv needs: the profile's list, plus G-MAP-OVERFIT whenever the
    10 cm head is on (a launch prerequisite on any profile). Ordered, without repeats."""
    req = list(prof["required_checks"])
    if hires_on(prof, argv):
        req.append("G-MAP-OVERFIT")
    return list(dict.fromkeys(req))


def hires_static_reasons(prof: dict, argv: list[str], model,
                         pmap: list[tuple[str, str]] | None = None) -> tuple[list[str], dict]:
    """NEW-2 on the BUILT object (G-DVB): the branch exists, is supervised, and names its
    weights -- a readable, non-dry-run, 8-entry class-weight file."""
    h = prof["map_hires"]
    reasons: list[str] = []
    tok = h["name_token"]
    mods = sorted({n.rsplit(".", 1)[0] for n, _ in model.named_parameters() if tok in n})
    n_par = int(sum(p.numel() for n, p in model.named_parameters() if tok in n))
    det = {"modules": mods[:30], "n_params": n_par}
    if not mods:
        reasons.append(f"{h['switch'][0]} {h['switch'][1]} but the built model has NO parameter "
                       f"whose name carries '{tok}' -- the 10 cm branch was not built")
    w = flag_values(argv, h["weight_flag"])
    try:
        wv = float(w[0]) if w else 0.0
    except ValueError:
        wv = float("nan")
    det["weight"] = wv
    if not (wv > 0.0):
        reasons.append(f"{h['switch'][0]} {h['switch'][1]} with {h['weight_flag']} {w} -- a built "
                       f"head with no live weight is declared-but-inert")
    cw = flag_values(argv, h["class_weights_flag"])
    det["class_weights"] = cw
    if not cw:
        reasons.append(f"{h['switch'][0]} {h['switch'][1]} without {h['class_weights_flag']}: the "
                       f"SPEC's median-frequency weights must be DECLARED (and are content-bound)")
    else:
        local = map_path(cw[0], list(pmap or []))
        try:
            d = read_json(local)
        except Exception as e:                            # noqa: BLE001
            d = None
            reasons.append(f"{h['class_weights_flag']} {cw[0]} is not a readable JSON here "
                           f"({type(e).__name__})")
        if isinstance(d, dict):
            wl = d.get("weights")
            det["class_weights_file"] = {"n": len(wl) if isinstance(wl, list) else None,
                                         "dry_run": d.get("dry_run")}
            if d.get("dry_run"):
                reasons.append(f"{h['class_weights_flag']} is a DRY-RUN weights file -- the "
                               f"launch weights come from the TRAIN split (PREREG "
                               f"G-MAP-OVERFIT sec. 3: a dry-run file is REFUSED)")
            if not (isinstance(wl, list) and len(wl) == len(h["classes"])
                    and all(_finite_num(x) and float(x) >= 0 for x in wl)):
                reasons.append(f"{h['class_weights_flag']}: `weights` must be {len(h['classes'])} "
                               f"finite non-negative numbers, one per class in code order")
    return reasons, det


def _tensor_paths(out: Any, prefix: str = "out") -> list[tuple[str, tuple]]:
    import torch
    res = []
    if torch.is_tensor(out):
        res.append((prefix, tuple(out.shape)))
    elif isinstance(out, dict):
        for k in out:
            res += _tensor_paths(out[k], f"{prefix}.{k}")
    elif isinstance(out, (list, tuple)):
        for i, v in enumerate(out):
            res += _tensor_paths(v, f"{prefix}[{i}]")
    return res


def hires_output_reasons(prof: dict, out: Any, image_hw: tuple[int, int]) -> tuple[list[str], dict]:
    """NEW-2 on a FORWARD's output: `fmap_s8` reaches the output (the F3 pass-through class), the
    logits are 600 x 320, the map-only lift is 240 x 128 (0.25 m)."""
    h = prof["map_hires"]
    tok = h["name_token"]
    reasons: list[str] = []
    paths = _tensor_paths(out)
    det: dict[str, Any] = {"hires_paths": [(p, s) for p, s in paths if tok in p.lower()][:20]}
    fk = h["fmap_key"]
    fm = [s for p, s in paths if p == f"out.{fk}"]
    det[fk] = fm[0] if fm else None
    if not fm:
        reasons.append(f"`{fk}` is ABSENT from the forward output -- the stride-8 map does not "
                       f"reach the 10 cm branch's consumers (the F3 pass-through class)")
    else:
        want = (image_hw[0] // h["trunk_stride"], image_hw[1] // h["trunk_stride"])
        if tuple(fm[0][-2:]) != want:
            reasons.append(f"`{fk}` spatial {tuple(fm[0][-2:])} != stride-{h['trunk_stride']} "
                           f"{want}")
    lg = [(p, s) for p, s in paths if tok in p.lower() and "logit" in p.lower()]
    det["logits"] = lg[:4]
    if not lg:
        reasons.append(f"no forward output named with '{tok}' and 'logit' -- the 10 cm logits "
                       f"were not produced")
    elif any(tuple(s[-2:]) != tuple(h["logits_hw"]) for _, s in lg):
        reasons.append(f"10 cm logits {[s for _, s in lg]} are not {tuple(h['logits_hw'])}")
    lf = [(p, s) for p, s in paths if tok in p.lower() and "logit" not in p.lower()
          and ("lift" in p.lower() or "bev" in p.lower()) and len(s) >= 3]
    det["lift"] = lf[:4]
    if not lf:
        reasons.append(f"no forward output named with '{tok}' and 'lift'/'bev' -- the 0.25 m "
                       f"map-only lift cannot be verified")
    elif not any(tuple(s[-2:]) == tuple(h["lift_hw"]) for _, s in lf):
        reasons.append(f"the '{tok}' lift/bev outputs {[s for _, s in lf]} carry no "
                       f"{tuple(h['lift_hw'])} (0.25 m) grid")
    return reasons, det


def _pnorm(p: Any) -> str:
    return str(p).replace("\\", "/").rstrip("/")


def nav_tau_reasons(prof: dict, argv: list[str], record: str | None,
                    pmap: list[tuple[str, str]] | None = None) -> tuple[list[str], dict]:
    """FIX-4: `--graft-nav-compliance` is REFUSED without a RECORDED tau derived on THIS arm's
    train split, status OK, and equal to `--nav-compliance-tau-rad`. Two record schemas:

    * BANKED (ab436ee, `…/2026-09-26-declared-vs-built/raw/nav_compliance_tau_train.json`,
      written on Thor by the fixes agent's `derive_navc_tau.py`): `inputs.manifest.path` is the
      `_v2manifest.pt` INSIDE the arm's `--v2-cache`, `inputs.labels.path` IS its `--v7-labels`,
      and `inputs.labels.sha256` must equal the labels file THIS host reads (content, not name);
    * in-repo (`stack/scripts/refcv7_derive_nav_tau.py --json`): `split_v2_cache` / `labels`.
    The record's own sha256 is bound in the token (`gate:nav-tau-record`)."""
    nc = prof.get("nav_compliance") or {}
    sw = nc.get("switch")
    if not sw or not has_flag(argv, sw):
        return [], {"on": False}
    det: dict[str, Any] = {"on": True, "record": record}
    if not record:
        return [f"{sw} is passed without a RECORDED tau: give the gate --nav-tau-record <the "
                f"banked nav_compliance_tau_train.json> (FIX-4: an invented tolerance decides "
                f"what counts as compliant with no evidence class)"], det
    if not Path(record).is_file():
        return [f"--nav-tau-record {record} does not exist"], det
    rec = read_json(record)
    det["sha256"] = sha256_file(record)
    reasons = []
    if rec.get("status") != "OK":
        reasons.append(f"the tau record's status is {rec.get('status')!r}, not OK")
    tv = flag_values(argv, nc["tau_flag"])
    try:
        t_argv = float(tv[0]) if tv else None
    except ValueError:
        t_argv = None
    t_rec = rec.get("tau")
    det.update(tau_record=t_rec, tau_argv=t_argv)
    if t_rec is None or t_argv is None or abs(float(t_rec) - t_argv) > 1e-9 * max(1.0, abs(t_argv)):
        reasons.append(f"{nc['tau_flag']} {tv} is not the recorded tau {t_rec}")
    cache_flag = nc.get("cache_flag", "--v2-cache")
    labels_flag = nc.get("labels_flag", "--v7-labels")
    want_cache, want_labels = flag_values(argv, cache_flag), flag_values(argv, labels_flag)
    inputs = rec.get("inputs") if isinstance(rec.get("inputs"), dict) else None
    if inputs is not None:
        det["schema"] = "banked (inputs.manifest / inputs.labels)"
        man = (inputs.get("manifest") or {}).get("path")
        lab = inputs.get("labels") or {}
        det.update(manifest=man, labels=lab.get("path"))
        man_dir = _pnorm(man).rsplit("/", 1)[0] if man else None
        if not want_cache or man_dir != _pnorm(want_cache[0]):
            reasons.append(f"the tau record was derived on the manifest {man!r}, not inside this "
                           f"arm's {cache_flag} {want_cache} -- a tau from another split is "
                           f"tuning elsewhere")
        if not want_labels or _pnorm(lab.get("path")) != _pnorm(want_labels[0]):
            reasons.append(f"the tau record was derived on the labels {lab.get('path')!r}, not "
                           f"on this arm's {labels_flag} {want_labels}")
        elif lab.get("sha256"):
            local = map_path(want_labels[0], pmap or [])
            if not Path(local).is_file():
                reasons.append(f"the tau record's labels sha256 cannot be checked: {local} is "
                               f"not readable on this host")
            else:
                got = sha256_file(local)
                det["labels_sha256_here"] = got
                if got != lab["sha256"]:
                    reasons.append(f"the labels this host reads ({got[:16]}) are not the labels "
                                   f"the tau was derived on ({str(lab['sha256'])[:16]})")
        else:
            reasons.append("the tau record carries no inputs.labels.sha256 -- its split cannot be "
                           "bound by content")
    else:
        det["schema"] = "in-repo (split_v2_cache / labels)"
        for key, flag in (("split_v2_cache", cache_flag), ("labels", labels_flag)):
            want = flag_values(argv, flag)
            got = rec.get(key)
            det[key] = got
            if not want or _pnorm(got) != _pnorm(want[0]):
                reasons.append(f"the tau record was derived on {key}={got!r}, not on this arm's "
                               f"{flag} {want} -- a tau from another split is tuning elsewhere")
    return reasons, det


#: the fixes agent's frozen REQUIRED-ON helper (landed ab436ee, `declared_vs_built.py`):
#: `check_refcv7_required(model, args, *, tau_file=None) -> list[Mismatch]`
DVB_REQUIRED_HELPER = "check_refcv7_required"


def required_on_reasons(prof: dict, argv: list[str], D=None, model=None,
                        args=None, tau_file: str | None = None) -> tuple[list[str], dict]:
    """SPEC_REFCV7 section 7 (A2, PI ruling E1, "assure that the three selection mechanism are
    on"): every `required_on` flag is PASSED, and -- when the model exists -- the fixes agent's
    frozen helper `declared_vs_built.check_refcv7_required(model, args, tau_file=)` confirms each
    one BUILT and ON (and, with the banked tau file, argv's tau equal to it). Inert for a profile
    that requires nothing."""
    req = tuple(prof.get("required_on", ()))
    vals = dict(prof.get("required_values") or {})
    if not req and not vals and not prof.get("required_positive"):
        return [], {"required_on": []}
    det: dict[str, Any] = {"required_on": list(req), "required_values": vals}
    reasons = [f"{f} is REQUIRED ON for a {prof['name']} launch (SPEC_REFCV7 7, PI ruling E1) "
               f"and the argv does not pass it" for f in req if not has_flag(argv, f)]
    for f, v in vals.items():
        got = flag_values(argv, f)
        want = [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]
        if got != want:
            why = (prof.get("required_values_why") or {}).get(f, "a recorded ruling")
            reasons.append(f"{f} {' '.join(want)} is REQUIRED for a {prof['name']} launch "
                           f"({why}) and the argv has {got}")
    for f in prof.get("required_positive", ()):
        got = flag_values(argv, f)
        try:
            ok = bool(got) and math.isfinite(float(got[0])) and float(got[0]) > 0.0
        except ValueError:
            ok = False
        if not ok:
            reasons.append(f"{f} must be passed with a value > 0 for a {prof['name']} launch (a "
                           f"built head with no live weight is declared-but-inert); the argv "
                           f"has {got}")
    if not req:
        return reasons, det
    if D is None or model is None:
        return reasons, det
    lst = getattr(D, "REFCV7_REQUIRED_ON", None)
    norm = {(s if str(s).startswith("--") else "--" + str(s).replace("_", "-"))
            for s in (lst or ())}
    det["dvb_required_on"] = sorted(norm) if lst is not None else None
    if lst is None:
        reasons.append("declared_vs_built publishes no REFCV7_REQUIRED_ON list (the fixes "
                       "agent's frozen API is not landed) -- BUILT-and-ON cannot be read")
    elif not set(req) <= norm:
        reasons.append(f"declared_vs_built.REFCV7_REQUIRED_ON {sorted(norm)} omits "
                       f"{sorted(set(req) - norm)}")
    fn = getattr(D, DVB_REQUIRED_HELPER, None)
    det["helper"] = DVB_REQUIRED_HELPER if callable(fn) else None
    if not callable(fn):
        reasons.append(f"declared_vs_built has no `{DVB_REQUIRED_HELPER}` (the fixes agent's "
                       f"frozen REQUIRED-ON API) -- BUILT-and-ON cannot be read")
        return reasons, det
    tf = tau_file if (tau_file and Path(tau_file).is_file()) else None
    det["helper_call"] = (f"{DVB_REQUIRED_HELPER}(model, args, tau_file="
                          f"{'<the banked tau record>' if tf else None})")
    try:
        res = fn(model, args, tau_file=tf)
    except SystemExit as e:
        reasons.append(f"REQUIRED-ON ({DVB_REQUIRED_HELPER}) refused: {str(e)[:400]}")
        return reasons, det
    except TypeError as e:
        reasons.append(f"REQUIRED-ON helper {det['helper_call']} -- API drift: {e}")
        return reasons, det
    det["helper_result"] = [str(x)[:300] for x in (res or [])] if isinstance(
        res, (list, tuple)) else repr(res)[:300]
    if not isinstance(res, (list, tuple)):
        reasons.append(f"{DVB_REQUIRED_HELPER} returned {type(res).__name__}, not a list")
    elif res:
        reasons.append(f"REQUIRED-ON: {len(res)} mechanism(s) not BUILT and ON: "
                       + " | ".join(str(x)[:240] for x in list(res)[:6]))
    return reasons, det


def judge_ceiling(prof: dict, argv: list[str], calls: list[dict], *,
                  eval_in_argv: bool) -> tuple[list[str], dict]:
    """SPEC_REFCV7 section 7: the max-speed ceiling filter is INFERENCE-ONLY. `calls` records every
    `SpeedCeilingFilter` application in the smoke, with the mode of the model forward it ran in.
    Keyed on the flag; inert without it."""
    flag = prof.get("ceiling_filter_flag")
    if not flag or not has_flag(argv, flag):
        return [], {"on": False}
    tr = [c for c in calls if c.get("training")]
    ev = [c for c in calls if not c.get("training")]
    det = {"on": True, "n_applied_in_training": len(tr), "n_applied_in_eval": len(ev),
           "eval_clipped_frac_max": max((c.get("clipped", 0.0) for c in ev), default=None)}
    reasons = []
    if tr:
        reasons.append(f"the max-speed ceiling mask was ACTIVE in {len(tr)} TRAINING forward(s): "
                       f"the oracle ceiling (ego-future) would shape training (SPEC_REFCV7 7 -- "
                       f"inference only)")
    if not eval_in_argv:
        reasons.append("the launch argv runs no in-training eval, so the smoke cannot show the "
                       "ceiling mask ACTIVE in an eval step")
    elif not ev:
        reasons.append("the ceiling mask was never ACTIVE in an eval-mode forward -- the filter "
                       "the argv declares did not act at inference")
    return reasons, det


def tau_config_reasons(record: str | None, config: dict | None) -> tuple[list[str], dict]:
    """SPEC_REFCV7 section 7: the run record carries the tau FILE. The fixes agent's batch 2
    (`--nav-compliance-tau-file <json>`) has the trainer read tau from the file and stamp
    `{path, sha256, tau}` into config.json; the gate keys on that stamp -- a dict node anywhere
    in config.json with `sha256` == the banked record's sha256 and `tau` == its tau."""
    if not record or not Path(record).is_file():
        return [], {}
    want = sha256_file(record)
    t_want = read_json(record).get("tau")
    found: list[dict] = []

    def walk(o):
        if isinstance(o, dict):
            if "sha256" in o and "tau" in o:
                found.append({k: o.get(k) for k in ("path", "sha256", "tau")})
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(config or {})
    det = {"tau_file_sha256": want, "tau": t_want, "stamps_found": found[:4]}
    ok = [s for s in found if s.get("sha256") == want and _finite_num(s.get("tau"))
          and t_want is not None and float(s["tau"]) == float(t_want)]
    if not ok:
        return [f"FIX-4: config.json carries no {{path, sha256, tau}} stamp of the tau file "
                f"(sha256 {want[:16]}, tau {t_want}) -- found {found[:2]}; the trainer stamps it "
                f"from --nav-compliance-tau-file (fixes batch 2)"], det
    return [], det


def tau_file_flag_reasons(prof: dict, argv: list[str], parser, record: str | None,
                          pmap: list[tuple[str, str]] | None = None) -> tuple[list[str], dict]:
    """SPEC_REFCV7 section 7 via the fixes agent's batch 2: a `--graft-nav-compliance` launch
    passes `--nav-compliance-tau-file`, and it names the SAME banked file the gate binds (by
    content). A trainer without the flag cannot put the tau file into config.json."""
    nc = prof.get("nav_compliance") or {}
    sw, ff = nc.get("switch"), nc.get("tau_file_flag")
    if not sw or not ff or not has_flag(argv, sw):
        return [], {"on": False}
    known = any(ff in a.option_strings for a in getattr(parser, "_actions", []))
    det: dict[str, Any] = {"on": True, "trainer_has_flag": known}
    if not known:
        return [f"FIX-4: the trainer has no {ff} (the fixes agent's batch 2 is not in this "
                f"tree) -- the tau FILE cannot reach config.json (SPEC_REFCV7 7)"], det
    v = flag_values(argv, ff)
    if not v:
        return [f"FIX-4: {sw} is passed without {ff} <the banked tau record>"], det
    local = map_path(v[0], list(pmap or []))
    det["argv_file"] = v[0]
    if not Path(local).is_file():
        return [f"FIX-4: {ff} {v[0]} is not readable on this host"], det
    got = sha256_file(local)
    det["argv_file_sha256"] = got
    if not record or not Path(record).is_file():
        return [f"FIX-4: the gate has no tau record to compare {ff} with"], det
    if got != sha256_file(record):
        return [f"FIX-4: {ff} names a file ({got[:16]}) that is not the banked tau record "
                f"({sha256_file(record)[:16]})"], det
    return [], det


def passthrough_reasons(prof: dict, argv: list[str], args, T) -> tuple[list[str], dict]:
    """The decoder pass-through, read off the ONE tuple the forward iterates
    (`RefCModel.DECODER_PASSTHROUGH`, fixes agent 2026-09-26): F3's per-stage keys when F3 is
    declared, NEW-2's `fmap_s8` when the 10 cm head is. The regression arms patch this tuple."""
    from tanitad.refs import refc
    pt = getattr(refc.RefCModel, "DECODER_PASSTHROUGH", None)
    det: dict[str, Any] = {"decoder_passthrough": list(pt) if pt is not None else None}
    reasons: list[str] = []
    need = []
    try:
        fl = T.refcv6_flags_from_args(args)
        if fl is not None and bool(getattr(fl, "f3_per_layer", False)):
            need += ["layer_u0_hat", "layer_logits"]
    except Exception as e:                                # noqa: BLE001
        det["f3_flags_error"] = str(e)
    if hires_on(prof, argv):
        need.append(prof["map_hires"]["fmap_key"])
    det["required"] = need
    if pt is None:
        if need:
            det["note"] = ("this tree has no RefCModel.DECODER_PASSTHROUGH (pre-fix): the "
                           "pass-through is an inline list the gate cannot read statically; "
                           "G-LIVE asserts the keys at runtime")
            if hires_on(prof, argv):
                reasons.append("NEW-2 needs RefCModel.DECODER_PASSTHROUGH to carry "
                               f"`{prof['map_hires']['fmap_key']}` and this tree has no such "
                               "tuple")
        return reasons, det
    miss = [k for k in need if k not in pt]
    if miss:
        reasons.append(f"RefCModel.DECODER_PASSTHROUGH {list(pt)} omits {miss}: the forward "
                       f"drops them (the D-REFCV6-F3-WHITELIST class)")
    return reasons, det


def judge_dvb(D, model, args, parser, argv, why_missing: str = "", *,
              forbid_kinds: tuple = ()) -> tuple[list[str], dict]:
    reasons: list[str] = []
    det: dict[str, Any] = {}
    # --- the gate's own behavioural probe: independent of the module under test --------------
    eq = equalize_probe(model)
    det["gate_probe_equalize"] = eq
    want_eq = int(getattr(args, "equalize_bottom_rows", 0) or 0)
    if eq.get("trunk") is None:
        if want_eq:
            reasons.append(f"argv declares --equalize-bottom-rows {want_eq} but the model has no "
                           f"TimmResNetTrunk to equalise")
    elif eq["rows_equalised_measured"] != want_eq:
        reasons.append(f"trunk equalisation: argv declares {want_eq} bottom rows, the BUILT trunk "
                       f"equalises {eq['rows_equalised_measured']} (measured by behaviour; "
                       f"D-REFCV6-EQUALIZE-DROPPED)")
    # --- the module (the fixes agent's frozen API) ------------------------------------------
    if D is None:
        reasons.append(f"tanitad.train.declared_vs_built is not importable -- module not yet "
                       f"landed => the gate FAILS ({why_missing})")
        return reasons, det
    fn = getattr(D, "check", None)
    if not callable(fn):
        reasons.append("tanitad.train.declared_vs_built has no callable `check`")
        return reasons, det
    call = f"check(model, args, build_parser(), forbid_kinds={tuple(forbid_kinds)!r})"
    det["dvb_call"] = call
    try:
        mism = fn(model, args, parser, forbid_kinds=tuple(forbid_kinds))
    except TypeError as e:
        reasons.append(f"declared_vs_built.{call} is not accepted -- API drift: {e}")
        return reasons, det
    if not isinstance(mism, (list, tuple)):
        reasons.append(f"declared_vs_built.check returned {type(mism).__name__}, not a list")
        return reasons, det
    det["mismatches"] = [(dataclasses.asdict(m) if dataclasses.is_dataclass(m) else
                          (m if isinstance(m, (dict, str)) else repr(m))) for m in mism]
    if mism:
        reasons.append(f"declared_vs_built: {len(mism)} mismatch(es): "
                       + " | ".join(str(m)[:240] for m in list(mism)[:12]))
    cov = getattr(D, "coverage", None)
    if callable(cov):
        unc = list(cov(parser))
        det["uncovered_dests"] = unc
        det["registry_size"] = len(getattr(D, "REGISTRY", {}) or {})
    else:
        reasons.append("declared_vs_built has no coverage(parser): the gate cannot show that "
                       "every trainer flag has an entry")
    # --- the module's own positive control: flip one built lever, it must SEE it -------------
    trunk = _find_trunk(model)
    if trunk is not None and hasattr(trunk.cfg, "equalize_bottom_rows"):
        old = trunk.cfg.equalize_bottom_rows
        try:
            object.__setattr__(trunk.cfg, "equalize_bottom_rows", 0 if int(old or 0) else 43)
            ctl = fn(model, args, parser, forbid_kinds=tuple(forbid_kinds))
        finally:
            object.__setattr__(trunk.cfg, "equalize_bottom_rows", old)
        n_ctl = len(ctl) if isinstance(ctl, (list, tuple)) else None
        det["control_flip_equalize_n_mismatches"] = n_ctl
        if not (n_ctl is not None and n_ctl > len(mism)):
            reasons.append("declared_vs_built is BLIND to the FIX-3 lever: flipping the built "
                           "trunk's equalize_bottom_rows did not add a mismatch (control)")
    else:
        det["control_flip_equalize_n_mismatches"] = "not applicable (no timm trunk)"
    return reasons, det


# --------------------------------------------------------------------------------------------- #
# G-LIVE: declared loss terms, derived from the argv and exhaustive over the trainer's registries  #
# --------------------------------------------------------------------------------------------- #
def _w(a, dest: str) -> float:
    try:
        return float(getattr(a, dest, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


#: every dest in the trainer's `REFC_WEIGHT_GATES` -> (predicate over args, loss keys, min
#: fraction of training steps, why). ⛔ EXHAUSTIVE: a weight dest the trainer registers and this
#: table does not name makes G-LIVE FAIL -- the next `--w-*` cannot ship unchecked.
LIVE_WEIGHT_RULES: dict[str, tuple[Callable, tuple[str, ...], float, str]] = {
    "w_u0": (lambda a: _w(a, "w_u0") > 0 and str(getattr(a, "sampler", "none")) == "ddim"
             and not bool(getattr(a, "f6_w_u0_zero", False)), ("u0",), 1.0,
             "the control-space x0 loss (WP-4)"),
    "w_agent": (lambda a: str(getattr(a, "agents", "off")) == "head" and _w(a, "w_agent") > 0,
                ("agent_presence", "agent_centre"), 0.5,
                "the detection set loss (label-dependent: a batch with no labelled window skips)"),
    "w_bev_aux": (lambda a: _w(a, "w_bev_aux") > 0 and str(getattr(a, "bev_aux", "off")) != "off",
                  ("bev",), 0.5, "WP-D BEV auxiliary"),
    "w_tac_goal": (lambda a: _w(a, "w_tac_goal") > 0, ("tac_goal",), 1.0,
                   "the 22-token tactical goal set"),
    "agent_w_project": (lambda a: _w(a, "agent_w_project") > 0
                        and str(getattr(a, "agents", "off")) != "off", ("agent_project",), 0.5,
                        "monocular projection term"),
    "agent_w_ground": (lambda a: _w(a, "agent_w_ground") > 0
                       and str(getattr(a, "agents", "off")) != "off", ("agent_ground",), 0.5,
                       "ground prior term"),
    "goal_point_w": (lambda a: bool(getattr(a, "goal_point_inject", False))
                     and _w(a, "goal_point_w") > 0, ("goal_point",), 1.0, "E15 goal point"),
    "w_map": (lambda a: _w(a, "w_map") > 0, ("map",), 1.0, "refcv6 SAM3 BEV map"),
    "w_box3d": (lambda a: _w(a, "w_box3d") > 0, ("box3d",), 1.0, "refcv6 3-D boxes"),
    "w_tac_v6": (lambda a: _w(a, "w_tac_v6") > 0, ("tac_v6",), 1.0,
                 "refcv6 tactical behaviour decoder"),
    "w_r7_wta": (lambda a: _w(a, "w_r7_wta") > 0, ("r7_wta",), 1.0,
                 "DrivoR-T WTA decoder (named refcv7_* until its rename)"),
    "w_r7_scorer": (lambda a: _w(a, "w_r7_scorer") > 0, ("r7_scorer",), 1.0,
                    "DrivoR-T scorer (named refcv7_* until its rename)"),
    # ⚠️ PROVISIONAL (SPEC_REFCV7 6.2, NEW-2): dest and loss key follow the NEW-2 agent's names
    # until its LANDING_READY. Registered BEFORE the trainer has the weight, so the day it lands
    # the exhaustiveness check above already knows it.
    "w_map_hires": (lambda a: _w(a, "w_map_hires") > 0
                    and str(getattr(a, "map_hires", "off")) == "on", ("map_hires",), 1.0,
                    "NEW-2: the 10 cm map head's hard-label CE"),
}
#: every field of `refcv6_diffusion.DiffusionFlags` -> the loss keys it declares ( () = none,
#: with the reason). ⛔ EXHAUSTIVE for the same reason.
LIVE_FLAG_RULES: dict[str, tuple[tuple[str, ...], str]] = {
    "f3_per_layer": (("cascade",), "F3 per-stage cascade loss (D-REFCV6-F3-WHITELIST)"),
    "f1_random_t": ((), "a sampling change; its effect is G-DVB's"),
    "f1_t_max": ((), "a constant of F1"),
    "f2_dd_step": ((), "step semantics; no loss term"),
    "f4_adaln": ((), "architecture; its modulations must get GRADIENT (leaf check)"),
    "f4_zero_init": ((), "an init choice"),
    "f5_emitting_conf": ((), "the emitting pass's confidence; trained through `cls`"),
    "f5_refuse_blind_rank": ((), "a refusal"),
    "f5_focal": ((), "changes HOW `cls` is computed, not whether"),
    "f5_focal_gamma": ((), "a constant of F5"),
    "f5_focal_alpha": ((), "a constant of F5"),
    "f6_w_u0_zero": ((), "REMOVES the u0 term (handled in the w_u0 rule)"),
    "f7_samples_per_anchor": ((), "samples per anchor; no new key"),
    "f7_ack_eval_join": ((), "an acknowledgement"),
    "f8_flat_waypoint_noise": ((), "noise shape; no loss key"),
    "f9_assert_vocab": ((), "an assertion"),
    "f9_n_anchors": ((), "a constant of F9"),
}
LIVE_ALWAYS = (("loss", "traj", "cls"), "the planner objective and the anchor classifier")


def declared_terms(T, args) -> tuple[list[dict], list[str]]:
    """-> (rules that apply to THIS argv, coverage problems). Nothing is hand-listed per ARM:
    the table is keyed by the TRAINER's own registries and must cover all of them."""
    problems: list[str] = []
    terms = [{"id": "planner", "keys": list(LIVE_ALWAYS[0]), "min_frac": 1.0,
              "why": LIVE_ALWAYS[1]}]
    registry = getattr(T, "REFC_WEIGHT_GATES", None)
    if not isinstance(registry, dict):
        problems.append("the trainer has no REFC_WEIGHT_GATES registry to derive terms from")
        registry = {}
    for dest in sorted(registry):
        if dest not in LIVE_WEIGHT_RULES:
            problems.append(f"the trainer registers weight `{dest}` and G-LIVE has no rule for it")
    for dest, (pred, keys, frac, why) in LIVE_WEIGHT_RULES.items():
        if pred(args):
            terms.append({"id": dest, "keys": list(keys), "min_frac": frac, "why": why})
    flags = None
    try:
        flags = T.refcv6_flags_from_args(args)
        fields = [f.name for f in dataclasses.fields(T._rv6.DiffusionFlags)]
    except Exception as e:                                # noqa: BLE001
        fields = []
        problems.append(f"cannot enumerate the refcv6 diffusion flags: {e}")
    for f in fields:
        if f not in LIVE_FLAG_RULES:
            problems.append(f"refcv6 DiffusionFlags.{f} has no G-LIVE rule")
    if flags is not None:
        for f, (keys, why) in LIVE_FLAG_RULES.items():
            if keys and bool(getattr(flags, f, False)):
                terms.append({"id": f, "keys": list(keys), "min_frac": 1.0, "why": why})
    return terms, problems


def judge_live(terms: list[dict], problems: list[str], steps: list[dict], *,
               expect_steps: int, summary: dict | None, train_rows: list[dict],
               grads: dict, prior: dict | None, exit_msg: str | None,
               admitted_dead: dict[str, str] | None = None) -> tuple[list[str], dict]:
    """The pure G-LIVE verdict. `admitted_dead`: {leaf group: reason} -- the profile's
    `live_dead_admitted` rows whose FLAG is in the launch argv (the caller filters by argv). `steps`: one record per TRAINING `compute_losses_v3` call:
    {"keys": [...], "nonfinite": [...], "vals": {k: float}}."""
    reasons = list(problems)
    det: dict[str, Any] = {"n_train_steps_seen": len(steps), "expect_steps": expect_steps}
    if exit_msg:
        reasons.append(f"the smoke did not complete: {exit_msg[:600]}")
    if not (summary and summary.get("done") is True and int(summary.get("step", -1)) == expect_steps):
        reasons.append(f"summary.json does not say done at step {expect_steps} "
                       f"(got {summary!r}) -- the artifact, not an exit code, is the verdict")
    if len(steps) != expect_steps:
        reasons.append(f"{len(steps)} training loss calls observed, expected {expect_steps}")
    n = max(len(steps), 1)
    per_term = []
    for t in terms:
        for k in t["keys"]:
            present = sum(1 for s in steps if k in s["keys"])
            bad = sum(1 for s in steps if k in s.get("nonfinite", ()))
            frac = present / n
            per_term.append({"rule": t["id"], "key": k, "present_steps": present,
                             "frac": round(frac, 4), "min_frac": t["min_frac"],
                             "nonfinite_steps": bad, "why": t["why"]})
            if present == 0:
                reasons.append(f"declared term `{k}` ({t['why']}) is ABSENT from all {len(steps)} "
                               f"training steps")
            elif frac < t["min_frac"]:
                reasons.append(f"declared term `{k}` present on only {present}/{len(steps)} steps "
                               f"(< {t['min_frac']:.0%})")
            if bad:
                reasons.append(f"declared term `{k}` was non-finite on {bad} step(s)")
    det["terms"] = per_term
    nonfin_any = sorted({k for s in steps for k in s.get("nonfinite", ())})
    det["nonfinite_keys_any"] = nonfin_any
    if nonfin_any:
        reasons.append(f"non-finite loss outputs: {nonfin_any[:10]}")
    losses = [s["vals"].get("loss") for s in steps if "loss" in s.get("vals", {})]
    det["loss_first_last"] = [losses[0], losses[-1]] if losses else None
    if len(losses) >= 2 and len({round(float(x), 8) for x in losses}) < 2:
        reasons.append("the total loss is CONSTANT across the smoke -- a disconnected graph")
    # the run's own record: the final training row must carry the declared keys it computed
    last = steps[-1] if steps else {}
    rows = [r for r in train_rows if int(r.get("step", -1)) == expect_steps]
    det["final_train_row_present"] = bool(rows)
    if not rows:
        reasons.append(f"metrics.jsonl has no training row at step {expect_steps}")
    else:
        row = rows[-1]
        det["final_row_keys"] = sorted(row)
        for t in terms:
            for k in t["keys"]:
                if k in last.get("keys", ()) and k not in row:
                    reasons.append(f"`{k}` was computed at the last step but is ABSENT from the "
                                   f"metrics.jsonl row -- the run record would not show it")
                # ⭐ control: the wrapper and the log must agree (the gate read the same calls)
                if k in row and k in last.get("vals", {}):
                    a, b = float(row[k]), float(last["vals"][k])
                    if not (math.isfinite(a) and abs(a - round(b, 5)) <= 1.5e-5):
                        reasons.append(f"control: metrics.jsonl `{k}`={a} vs captured {b:.6g} -- "
                                       f"the gate observed different calls than the log")
    # gradients: every declared-trainable LEAF module
    det["grads"] = {k: v for k, v in grads.items() if k != "per_param"}
    if grads.get("n_trainable_params", 0) == 0:
        reasons.append("no trainable parameter was observed at opt.step -- the capture missed")
    if grads.get("dead_groups"):
        adm = admitted_dead or {}
        dg = [d for d in grads["dead_groups"] if d["group"] not in adm]
        det["dead_groups_admitted"] = {d["group"]: adm[d["group"]]
                                       for d in grads["dead_groups"] if d["group"] in adm}
        if dg:
            reasons.append(f"{len(dg)} declared-trainable leaf module(s) received ZERO gradient over "
                           f"the whole smoke: {[d['group'] for d in dg[:10]]}")
    if grads.get("nonfinite_params"):
        reasons.append(f"non-finite gradients on {len(grads['nonfinite_params'])} parameter(s): "
                       f"{grads['nonfinite_params'][:5]}")
    if prior is not None:
        det["residual_prior"] = prior
        if not prior.get("found"):
            reasons.append(f"--residual-prior is declared but no prior tensor was found in the "
                           f"forward output (looked for {prior.get('looked_for')}; saw "
                           f"{prior.get('out_keys_sample')})")
        elif prior.get("n_rows_v0_pos", 0) == 0:
            reasons.append("residual prior: no training row had v0 > 0 -- the check read nothing")
        elif prior.get("n_rows_v0_pos_prior_zero", 0) > 0:
            reasons.append(f"residual prior is ZERO on {prior['n_rows_v0_pos_prior_zero']} of "
                           f"{prior['n_rows_v0_pos']} rows with v0 > 0")
        if prior.get("n_nonfinite", 0):
            reasons.append(f"residual prior non-finite on {prior['n_nonfinite']} row(s)")
        # the OWNER's consumer check (the refcv7 model agent's `kinematic_prior.plan_check`):
        # the plan is P + delta, P non-zero wherever v > 0, everything finite -- read on the same
        # training forwards; the gate's own measurement above stays independent of it
        pc = prior.get("owner_plan_check")
        if pc is None:
            pass
        elif pc.get("unavailable"):
            reasons.append(f"NEW-1: the owner's plan_check is unavailable ({pc['unavailable']}) -- "
                           f"the plan-is-P+delta clause cannot be read")
        elif pc.get("calls", 0) == 0:
            reasons.append("NEW-1: the owner's plan_check never completed on a training forward"
                           + (f": {pc['errors'][:2]}" if pc.get("errors") else ""))
        else:
            if pc.get("errors"):
                reasons.append(f"NEW-1: plan_check raised on {len(pc['errors'])} forward(s): "
                               f"{pc['errors'][:2]}")
            if pc.get("not_ok", 0):
                reasons.append(f"NEW-1: plan_check is NOT ok on {pc['not_ok']} of {pc['calls']} "
                               f"training forwards (the plan is not P + delta, or non-finite)")
            if pc.get("rows_v_gt_0_prior_nonzero") != pc.get("rows_v_gt_0"):
                reasons.append(f"NEW-1: plan_check: the prior is non-zero on "
                               f"{pc.get('rows_v_gt_0_prior_nonzero')} of {pc.get('rows_v_gt_0')} "
                               f"rows with v > 0")
    return reasons, det


def grad_table(records: dict, model_names: dict) -> dict:
    """records: {param_name: {"numel", "steps", "nz_steps", "nonfinite", "requires_grad"}}."""
    groups: dict[str, dict] = {}
    for name, r in records.items():
        g = name.rsplit(".", 1)[0] if "." in name else name
        e = groups.setdefault(g, {"group": g, "numel": 0, "params": 0, "nz_params": 0})
        e["numel"] += int(r["numel"])
        e["params"] += 1
        e["nz_params"] += int(r["nz_steps"] > 0)
    dead = sorted((e for e in groups.values() if e["nz_params"] == 0), key=lambda e: e["group"])
    dead_params = sorted(n for n, r in records.items() if r["nz_steps"] == 0)
    return {"groups": sorted(groups.values(), key=lambda e: e["group"]),
            "n_trainable_params": len(records),
            "n_trainable_numel": int(sum(r["numel"] for r in records.values())),
            "n_leaf_groups": len(groups), "dead_groups": dead,
            "n_dead_params": len(dead_params), "dead_params_sample": dead_params[:40],
            "nonfinite_params": sorted(n for n, r in records.items() if r["nonfinite"]),
            "frozen_in_optimizer": sorted(model_names.get("frozen_in_opt", []))[:40],
            "not_in_optimizer_but_requires_grad": sorted(
                model_names.get("requires_grad_not_in_opt", []))[:40]}


# --------------------------------------------------------------------------------------------- #
# the smoke (G-LIVE run 1, G-CKPT run 2)                                                          #
# --------------------------------------------------------------------------------------------- #
def _named_tensors(out: Any, prefix: str = "out"):
    import torch
    if torch.is_tensor(out):
        yield prefix, out
    elif isinstance(out, dict):
        for k, v in out.items():
            yield from _named_tensors(v, f"{prefix}.{k}")
    elif isinstance(out, (list, tuple)):
        for i, v in enumerate(out):
            yield from _named_tensors(v, f"{prefix}[{i}]")


def _probe_grad_unreachable(ctx: "Ctx", T, model, batch) -> dict:
    """The fixes agent's batch-3 probe (landed 4797ffb),
    `declared_vs_built.probe_grad_unreachable(model, backward) -> list[Mismatch]`, on the REAL
    smoke batch: every module DECLARED grad-unreachable (`_gradreach`) receives NO gradient when
    re-opened for one backward, and something outside them does (its positive control). The
    callable runs the trainer's own `compute_losses_v3` and `.backward()`. Absent (a pre-batch-3
    tree) -> recorded as absent; a TypeError -> API drift, judged."""
    try:
        D = importlib.import_module("tanitad.train.declared_vs_built")
    except Exception as e:                                # noqa: BLE001
        return {"present": False, "why": f"{type(e).__name__}: {e}"[:200]}
    fn = getattr(D, "probe_grad_unreachable", None)
    if not callable(fn):
        return {"present": False, "why": "declared_vs_built has no probe_grad_unreachable "
                                         "(fixes batch 3 not in this tree)"}
    if model is None or batch is None:
        return {"present": True, "error": "no model / no training batch was captured"}
    dev = next((p.device for p in model.parameters()), "cpu")

    def backward():
        # batch 3's contract (declared_vs_built.probe_grad_unreachable): "the caller's forward +
        # loss + .backward()" -- the probe re-opens the declared tensors around this ONE call
        model.train()
        loss = T.compute_losses_v3(model, batch, dev)["loss"]
        loss.backward()
        return loss
    try:
        res = fn(model, backward)
    except TypeError as e:
        return {"present": True, "error": f"API drift: probe_grad_unreachable(model, closure): {e}"}
    except SystemExit as e:
        return {"present": True, "result": [str(e)[:600]], "refused": True}
    except Exception as e:                                # noqa: BLE001
        return {"present": True, "error": f"{type(e).__name__}: {e}"[:400]}
    items = list(res) if isinstance(res, (list, tuple)) else ([] if not res else [res])
    return {"present": True, "result": [str(x)[:300] for x in items]}


def live_pres_fracs(named: Iterable, lp: dict) -> tuple[dict, dict]:
    """(heads, aux): per presence tensor, the fraction of slots at sigma >= the gate. A9 R2's
    per-layer decodes (a path segment `aux`) are INFORMATIVE: the declared gate is the LAST
    layer's (the detection rule), and an aux layer never counts as a slot head."""
    import torch
    heads: dict[str, dict] = {}
    aux: dict[str, dict] = {}
    for pth, t in named:
        if lp["key_token"] in pth and torch.is_tensor(t) and t.is_floating_point() and t.numel():
            d = {"frac": float((torch.sigmoid(t.detach().float()) >= float(lp["gate"]))
                               .float().mean()), "n": int(t.numel())}
            (aux if re.search(r"\.aux(\[|\.|$)", pth) else heads)[pth] = d
    return heads, aux


def judge_live_pres(prof: dict, pres_last: dict | None,
                    pres_aux: dict | None = None) -> tuple[list[str], dict]:
    """G-LIVE-PRES (SPEC_REFCV7 A9, A10 15.2): at the END of the smoke, on BOTH slot heads, the
    fraction of slots with sigma(presence) >= 0.5 is < 0.5 (refcv6's red arm: 71-99 of 100).
    G-LIVE-COUNT is NOT APPLICABLE to a focal head without a calibration map (15.2) -- recorded
    as such, never as passed."""
    lp = prof.get("live_pres")
    if not lp:
        return [], {}
    pres = pres_last or {}
    det = {"heads": pres, "gate": lp["gate"], "max_frac": lp["max_frac"],
           "aux_layers_informative": pres_aux or {},
           "g_live_count": "NOT APPLICABLE (focal presence without calibration, A10 15.2)"}
    if len(pres) < int(lp["min_heads"]):
        return [f"G-LIVE-PRES: {len(pres)} presence head(s) found in the forward output (key token "
                f"'{lp['key_token']}'), {lp['min_heads']} required -- BOTH slot heads must be "
                f"checked: {sorted(pres)}"], det
    bad = {k: v for k, v in pres.items() if not (v["frac"] < float(lp["max_frac"]))}
    reasons = [f"G-LIVE-PRES: {k}: {v['frac']:.3f} of {v['n']} slots have sigma >= {lp['gate']} "
               f"at the end of the smoke (must be < {lp['max_frac']}; refcv6: 71-99 of 100)"
               for k, v in sorted(bad.items())]
    return reasons, det


def judge_grad_unreachable_probe(probe: dict | None) -> tuple[list[str], dict]:
    if not probe or not probe.get("present"):
        return [], {"probe": probe}                       # recorded; batch 3 makes it present
    if probe.get("error"):
        return [f"G-LIVE: probe_grad_unreachable: {probe['error']}"], {"probe": probe}
    if probe.get("result"):
        return [f"G-LIVE: probe_grad_unreachable reports {len(probe['result'])} problem(s): "
                f"{probe['result'][:3]}"], {"probe": probe}
    return [], {"probe": probe}


def class_weights_stamp_reasons(prof: dict, config: dict | None,
                                file_sha256: str | None) -> tuple[list[str], dict]:
    """A8 item 1: the loader stamps `config.json["map_hires"]["class_weights"]` =
    {definition_id: sqrt_mf, pre_registered: true, sha256: <the launch weights file>}."""
    h = prof.get("map_hires") or {}
    node: Any = config or {}
    for k in h.get("class_weights_stamp_path", ()):
        node = node.get(k) if isinstance(node, dict) else None
    det = {"stamp": node, "file_sha256": file_sha256}
    if not isinstance(node, dict):
        return [f"NEW-2 (A8): config.json carries no "
                f"`{'.'.join(h.get('class_weights_stamp_path', ()))}` stamp"], det
    out = []
    if node.get("definition_id") != h.get("class_weights_definition"):
        out.append(f"NEW-2 (A8): the class weights are {node.get('definition_id')!r}, the SPEC "
                   f"registers {h.get('class_weights_definition')!r}")
    if node.get("pre_registered") is not True:
        out.append("NEW-2 (A8): the class-weight definition is not stamped pre_registered")
    if file_sha256 and node.get("sha256") != file_sha256:
        out.append(f"NEW-2 (A8): config.json stamps weights {str(node.get('sha256'))[:16]}, the "
                   f"argv's file is {file_sha256[:16]}")
    return out, det


def _find_prior(out: Any, keys: tuple[str, ...]):
    import torch
    if isinstance(out, dict):
        for k in keys:
            v = out.get(k)
            if torch.is_tensor(v):
                return k, v
    return None, None


def run_smoke(ctx: Ctx, T, argv: list[str], *, snapshot: str) -> dict:
    """Run the REAL trainer (`T.main(argv)`) with observing hooks only. `snapshot`:
    "final" -- digest model+optimizer after the run; "post_load" -- digest at the first training
    loss call (i.e. right after a resume has loaded ckpt.pt)."""
    import torch
    cur_mode: dict[str, Any] = {"training": None}
    cur_batch: dict[str, Any] = {"b": None}
    rec: dict[str, Any] = {"ceiling_calls": [], "map_class": {},
                           "steps": [], "eval_calls": 0, "grads": {}, "sampler_iters": [],
                           "out_keys": set(), "prior": None, "exit": None, "step_times": [],
                           "fwd_train_calls": 0, "fmap_hits": 0, "hires_first": None,
                           "ga_groups": {}, "ga_on": None}
    pat = _Patcher()
    state: dict[str, Any] = {"model": None, "opt": None, "names": {}, "dpos": None}
    prior_keys = tuple(ctx.prof.get("residual_prior_keys", ()))
    want_prior = residual_prior_on(ctx.argv)
    hires = hires_on(ctx.prof, ctx.argv)
    fmap_key = (ctx.prof.get("map_hires") or {}).get("fmap_key", "fmap_s8")
    pstats = {"found": False, "key": None, "looked_for": list(prior_keys), "n_rows": 0,
              "n_rows_v0_pos": 0, "n_rows_v0_pos_prior_zero": 0, "n_nonfinite": 0,
              "out_keys_sample": []}
    cur_v0: dict[str, Any] = {"v0": None}
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:                                     # noqa: BLE001
        pass

    def _hook_map_classes(out):
        """G-MAP: a tensor hook on the 10 cm logits accumulates, per class, the gradient at that
        class's own labelled cells (all channels, and its own channel) -- on the real batch."""
        h = ctx.prof["map_hires"]
        lh, lw = h["logits_hw"]
        lg = next((t for pth, t in _named_tensors(out)
                   if h["name_token"] in pth.lower() and "logit" in pth.lower()
                   and t.ndim == 4 and tuple(t.shape[-2:]) == (lh, lw) and t.requires_grad), None)
        b = cur_batch.get("b")
        tgt = None
        if isinstance(b, dict):
            tgt = next((v for v in b.values() if torch.is_tensor(v) and v.ndim == 3
                        and tuple(v.shape[-2:]) == (lh, lw) and not v.is_floating_point()), None)
        if lg is None or tgt is None:
            return
        tgt = tgt.to(lg.device)
        st = rec["map_class"]
        st["_hooked"] = st.get("_hooked", 0) + 1

        def ghook(g):
            with torch.no_grad():
                ga = g.detach().float().abs()
                for c in range(int(g.shape[1])):
                    m = tgt == c
                    n = int(m.sum())
                    e = st.setdefault(str(c), {"cells": 0, "grad_all": 0.0, "grad_own": 0.0,
                                               "nonfinite": False})
                    e["cells"] += n
                    if n:
                        e["grad_all"] += float(ga.sum(1)[m].sum())
                        e["grad_own"] += float(ga[:, c][m].sum())
                    if not bool(torch.isfinite(g[:, c]).all()):
                        e["nonfinite"] = True
            return None
        lg.register_hook(ghook)

    def fwd_hook(module, inputs, out):
        if not module.training:
            return
        rec["fwd_train_calls"] += 1
        if isinstance(out, dict):
            rec["out_keys"].update(map(str, out.keys()))
            if hires:
                rec["fmap_hits"] += int(torch.is_tensor(out.get(fmap_key)))
                if rec["hires_first"] is None:
                    try:
                        hw = tuple(module.cfg.core.encoder.image_hw())
                        rec["hires_first"] = hires_output_reasons(ctx.prof, out, hw)
                    except Exception as e:                # noqa: BLE001
                        rec["hires_first"] = ([f"the NEW-2 output probe crashed: {e}"], {})
            if hires:
                _hook_map_classes(out)
            lp = ctx.prof.get("live_pres")
            if lp:
                # A9/A10 15.2: the fraction of slots at sigma >= gate, per presence tensor, at
                # THIS forward -- overwritten each step, so the verdict reads the END of the smoke
                with torch.no_grad():
                    rec["pres_last"], rec["pres_aux_last"] = live_pres_fracs(
                        _named_tensors(out), lp)
            if want_prior:
                k, p = _find_prior(out, prior_keys)
                pstats["out_keys_sample"] = sorted(map(str, out.keys()))[:40]
                # the prior's OWN speed when the forward emits it (a withheld row carries a
                # different v than pose_last, and its prior is zero by design)
                pv = out.get("residual_prior_v")
                v0 = pv if torch.is_tensor(pv) else cur_v0["v0"]
                pstats["v_source"] = "residual_prior_v" if torch.is_tensor(pv) else "pose_last"
                if p is not None and v0 is not None:
                    pstats["found"], pstats["key"] = True, k
                    pp = p.detach().float().reshape(p.shape[0], -1)
                    v = v0.detach().float().reshape(-1).to(pp.device)
                    fin = torch.isfinite(pp).all(dim=1)
                    mag = pp.abs().amax(dim=1)
                    pos = v > 0                  # SPEC_REFCV7 2, literally: "where v0 > 0"
                    pstats["n_rows"] += int(pp.shape[0])
                    pstats["n_rows_v0_pos"] += int(pos.sum())
                    pstats["n_rows_v0_pos_prior_zero"] += int((pos & (mag == 0)).sum())
                    pstats["n_nonfinite"] += int((~fin).sum())
                _owner_plan_check(module, out)

    def _owner_plan_check(module, out):
        """The refcv7 model agent's consumer check, read on the SAME training forward. Observes
        only: no grad, and a forked RNG so the run's draws cannot move."""
        pc = pstats.setdefault("owner_plan_check", {"calls": 0, "not_ok": 0, "rows_v_gt_0": 0,
                                                    "rows_v_gt_0_prior_nonzero": 0, "errors": []})
        if pc.get("unavailable"):
            return
        try:
            kp = importlib.import_module("tanitad.models.kinematic_prior")
            fn = getattr(kp, "plan_check")
        except Exception as e:                            # noqa: BLE001
            pc["unavailable"] = f"{type(e).__name__}: {e}"[:300]
            return
        dec = getattr(getattr(module, "core", None), "decoder", None)
        try:
            with torch.no_grad(), torch.random.fork_rng():
                rep = fn(out, getattr(dec, "anchor_controls", None), decoder=dec)
            pc["calls"] += 1
            pc["not_ok"] += int(not bool(rep.get("ok")))
            pc["rows_v_gt_0"] += int(rep.get("rows_v_gt_0", 0))
            pc["rows_v_gt_0_prior_nonzero"] += int(rep.get("rows_v_gt_0_prior_nonzero", 0))
        except Exception as e:                            # noqa: BLE001 -- recorded, judged
            if len(pc["errors"]) < 5:
                pc["errors"].append(f"{type(e).__name__}: {str(e)[:200]}")

    def build_opt_wrap(model, args, _orig=T.build_optimizer):
        opt = _orig(model, args)
        state["model"], state["opt"] = model, opt
        names = {id(p): n for n, p in model.named_parameters()}
        in_opt = [p for g in opt.param_groups for p in g["params"]]
        ids_in = {id(p) for p in in_opt}
        state["names"] = {
            "frozen_in_opt": [names.get(id(p), "?") for p in in_opt if not p.requires_grad],
            "requires_grad_not_in_opt": [n for n, p in model.named_parameters()
                                         if p.requires_grad and id(p) not in ids_in]}
        track = [(names.get(id(p), f"?{i}"), p) for i, p in enumerate(in_opt) if p.requires_grad]
        for n_, p_ in track:
            rec["grads"][n_] = {"numel": int(p_.numel()), "steps": 0, "nz_steps": 0,
                                "nonfinite": False}
        # G-LIVE ga-reach: which reach groups EXIST on the built model and are TRAINED (the
        # literal attribute paths of the profile; trainability from the optimizer, never from
        # whether a gradient arrived -- a dead group must not read as "not trainable")
        ids_tr = {id(p_) for _, p_ in track}
        grp: dict[str, dict] = {}
        for g_, path in (ctx.prof.get("ga_groups") or {}).items():
            m_ = _resolve_attr(model, path)
            if m_ is None or not hasattr(m_, "parameters"):
                grp[g_] = {"exists": False, "path": path}
                continue
            ps = list(m_.parameters())
            grp[g_] = {"exists": True, "path": path, "params": int(sum(p.numel() for p in ps)),
                       "trainable_params": int(sum(p.numel() for p in ps if id(p) in ids_tr))}
        rec["ga_groups"] = grp
        rec["ga_on"] = (getattr(model, "_perception", None) is not None
                        or getattr(model, "tac_decoder_v6", None) is not None)
        orig_step = opt.step

        def step_wrap(*a, **k):
            rec["step_times"].append(time.perf_counter())
            with torch.no_grad():
                for n_, p_ in track:
                    r = rec["grads"][n_]
                    r["steps"] += 1
                    g = p_.grad
                    if g is not None:
                        if bool(torch.count_nonzero(g) > 0):
                            r["nz_steps"] += 1
                        if not bool(torch.isfinite(g).all()):
                            r["nonfinite"] = True
            return orig_step(*a, **k)
        opt.step = step_wrap
        if arm_is(ctx, "presence_saturated"):
            # the refcv6 red pattern re-installed: every presence logit +20 (all slots confident);
            # registered BEFORE the observer, so the observer and the loss see what the arm made
            def _saturate(o):
                if isinstance(o, dict):
                    return {k: (v + 20.0 if torch.is_tensor(v) and "presence_logit" in str(k)
                                else _saturate(v)) for k, v in o.items()}
                return o
            model.register_forward_hook(lambda m, i, o: _saturate(o))
        model.register_forward_hook(fwd_hook)
        return opt

    def losses_wrap(model, batch, device, *a, _orig=T.compute_losses_v3, **k):
        cur_mode["training"] = bool(model.training)       # read by the ceiling-filter observer
        cur_batch["b"] = batch if model.training else None   # read by the G-MAP class probe
        if model.training:
            cur_batch["last_train"] = batch
        if model.training:
            if snapshot == "post_load" and "post_load" not in rec:
                rec["post_load"] = {"model": state_digests(model.state_dict()),
                                    "opt": optimizer_digests(state["opt"].state_dict())
                                    if state["opt"] is not None else None}
            pl = batch.get("pose_last") if isinstance(batch, dict) else None
            cur_v0["v0"] = pl[:, 3] if pl is not None else None
        out = _orig(model, batch, device, *a, **k)
        h = ctx.prof.get("map_hires") or {}
        if arm_is(ctx, "map_drivable_only_logging") and isinstance(out, dict):
            # G-MAP's arm RL1: every per-class 10 cm key is logged for DRIVABLE only, in train
            # and eval rows alike (refcv6's pattern for 38,000 steps)
            others = [f"_{c}_" for c in h.get("classes", ()) if c != "drivable"]
            out = {kk: vv for kk, vv in out.items()
                   if not (str(kk).startswith("map_hires_") and any(o in str(kk) for o in others))}
        if arm_is(ctx, "map_iou_not_counts") and isinstance(out, dict) and not model.training:
            # G-MAP's arm RL2: the eval row carries per-batch IoUs, not the argmax COUNTS
            cnt = tuple(f.split("{", 1)[0] for f in (*h.get("eval_count_fmts", ()),
                                                     *h.get("raw_count_fmts", ())))
            out = {kk: vv for kk, vv in out.items() if not str(kk).startswith(cnt)}
        if model.training:
            keys, nonfin, vals = [], [], {}
            for kk, vv in out.items():
                if torch.is_tensor(vv) and vv.ndim == 0:
                    f = float(vv.detach())
                elif isinstance(vv, (int, float, bool)):
                    f = float(vv)
                else:
                    continue
                keys.append(kk)
                vals[kk] = f
                if not math.isfinite(f):
                    nonfin.append(kk)
            rec["steps"].append({"keys": sorted(keys), "nonfinite": sorted(nonfin), "vals": vals})
        else:
            rec["eval_calls"] += 1
        return out

    def ntb_wrap(it, dl, sampler, dpos, _orig=T.next_train_batch):
        state["dpos"] = dpos
        return _orig(it, dl, sampler, dpos)

    orig_iter = T.ResumableEpochSampler.__iter__

    def iter_wrap(self):
        rec["sampler_iters"].append({"epoch": int(self.epoch), "skip": int(self.skip_batches),
                                     "n": int(self.n), "batch": int(self.batch)})
        return orig_iter(self)

    pat.set(T, "build_optimizer", build_opt_wrap)
    pat.set(T, "compute_losses_v3", losses_wrap)
    pat.set(T, "next_train_batch", ntb_wrap)
    pat.set(T.ResumableEpochSampler, "__iter__", iter_wrap)
    # SPEC_REFCV7 7: every application of the max-speed ceiling mask, with the mode of the
    # model forward it ran in (the filter module is built per call, so ITS own .training is
    # meaningless -- the model's mode is read at the loss call that owns the forward)
    try:
        from tanitad.refs import refcv6_selection as _v6sel
    except Exception:                                     # noqa: BLE001
        _v6sel = None
    if _v6sel is not None and hasattr(_v6sel, "SpeedCeilingFilter"):
        orig_scf = _v6sel.SpeedCeilingFilter.forward

        def scf_wrap(self, *a, **k):
            keep, tele = orig_scf(self, *a, **k)
            rec["ceiling_calls"].append({
                "training": cur_mode.get("training"),
                "clipped": float((tele or {}).get("speed_frac_candidates_clipped", 0.0)),
                "rows_empty": int((tele or {}).get("speed_rows_empty", 0))})
            return keep, tele
        pat.set(_v6sel.SpeedCeilingFilter, "forward", scf_wrap)
    _apply_smoke_arm(ctx, T, pat, argv)
    # ⛔ MEASURED 2026-09-27 on Thor (6fa5e8b): the trainer stamps `config.json["argv"] =
    # sys.argv[1:]`, and in-process that was the GATE's own command line ("check --ctx ..."), so
    # G-EVAL crashed rebuilding from the smoke's record. The smoke runs as a launch would: the
    # process argv IS the trainer argv while T.main runs (restored by `pat.restore()`).
    pat.set(sys, "argv", [str(getattr(T, "__file__", None) or "refc_v3_train.py")] + list(argv))
    t0 = time.time()
    try:
        T.main(list(argv))
    except SystemExit as e:
        rec["exit"] = f"SystemExit: {e}"
    except BaseException as e:                            # noqa: BLE001 -- the verdict records it
        rec["exit"] = f"{type(e).__name__}: {e}\n" + "".join(traceback.format_exc()[-2500:])
    finally:
        pat.restore()
    rec["elapsed_s"] = round(time.time() - t0, 1)
    try:
        if torch.cuda.is_available():
            rec["cuda_max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    except Exception:                                     # noqa: BLE001
        pass
    if snapshot == "final" and state["model"] is not None:
        rec["final"] = {"model": state_digests(state["model"].state_dict()),
                        "opt": optimizer_digests(state["opt"].state_dict())}
        rec["dpos_final"] = dict(state["dpos"]) if state["dpos"] is not None else None
        # ⛔ AFTER the G-CKPT snapshot: the probe runs one more TRAINING forward + backward,
        # which moves BN running stats and the EMA priors (MEASURED 2026-09-27 on the tiny rig:
        # probing first made 38 tensors differ from the checkpoint the trainer had just saved)
        rec["grad_unreachable_probe"] = _probe_grad_unreachable(
            ctx, T, state.get("model"), cur_batch.get("last_train"))
    rec["out_keys"] = sorted(rec["out_keys"])
    rec["prior"] = pstats if want_prior else None
    rec["grad_names"] = state["names"]
    return rec


def _drop_from_passthrough(pat: _Patcher, keys: tuple[str, ...], *, fallback_pop: bool) -> str:
    """The pass-through defect, applied where it LIVES: remove `keys` from
    `RefCModel.DECODER_PASSTHROUGH` (the tuple the forward iterates, fixes agent 2026-09-26).
    A pre-fix tree has an inline list instead; there, when `fallback_pop`, the keys are popped
    from the forward's output -- the same observable defect. -> how it was applied."""
    from tanitad.refs import refc
    pt = getattr(refc.RefCModel, "DECODER_PASSTHROUGH", None)
    if pt is not None:
        pat.set(refc.RefCModel, "DECODER_PASSTHROUGH", tuple(k for k in pt if k not in keys))
        return f"DECODER_PASSTHROUGH without {list(keys)}"
    if not fallback_pop:
        return "no DECODER_PASSTHROUGH on this tree: the arm cannot be applied"
    orig = refc.RefCModel.forward

    def as_shipped(self, *a, **k):
        o = orig(self, *a, **k)
        if isinstance(o, dict):
            for key in keys:
                o.pop(key, None)
        return o
    pat.set(refc.RefCModel, "forward", as_shipped)
    return f"pre-fix tree: {list(keys)} popped from RefCModel.forward's output"


def _resolve_attr(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


class _GaFinalRowOnlyJson:
    """The `json` module as the trainer sees it, except that a TRAINING row's `ga_*` keys are
    dropped unless it is the final step's row (arm `ga_off_by_one`)."""

    def __init__(self, real, final_step: int):
        self._real, self._final = real, int(final_step)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def dumps(self, obj, *a, **k):
        if (isinstance(obj, dict) and "loss" in obj and isinstance(obj.get("step"), int)
                and obj["step"] != self._final):
            obj = {kk: vv for kk, vv in obj.items() if not str(kk).startswith("ga_")}
        return self._real.dumps(obj, *a, **k)


def _apply_smoke_arm(ctx: Ctx, T, pat: _Patcher, argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else ctx.argv)
    if arm_is(ctx, "no_cascade_passthrough") or arm_is(ctx, "no_cascade_silent"):
        # D-REFCV6-F3-WHITELIST exactly: the production forward's output carries neither key
        _say("arm: " + _drop_from_passthrough(pat, ("layer_u0_hat", "layer_logits"),
                                              fallback_pop=True))
    if arm_is(ctx, "no_cascade_silent"):
        # ...and the loss SKIPS silently, as it did for 34,500 steps: the loss path reads the
        # F3 flag off the model, so hand it a copy with F3 off (the model keeps F3 BUILT).
        orig_flags = T._refcv6_flags_of

        def f3_blind(model):
            f = orig_flags(model)
            return dataclasses.replace(f, f3_per_layer=False)
        pat.set(T, "_refcv6_flags_of", f3_blind)
    if arm_is(ctx, "ceiling_active_in_training"):
        # SPEC_REFCV7 7's regression arm: the ceiling mask acts in a TRAINING step. On a tree
        # with the fixes agent's class switch (`speed_ceiling_in_training`, ab436ee) the arm
        # flips exactly that -- the pre-ruling behaviour; otherwise the decoder runs its eval
        # branch during training (and the gate must still name the ceiling)
        from tanitad.refs import refc
        if hasattr(refc.AnchoredDiffusionDecoder, "speed_ceiling_in_training"):
            pat.set(refc.AnchoredDiffusionDecoder, "speed_ceiling_in_training", True)
        else:
            orig_dec = refc.AnchoredDiffusionDecoder.forward

            def dec_eval(self, *a, **k):
                was = self.training
                self.training = False
                try:
                    return orig_dec(self, *a, **k)
                finally:
                    self.training = was
            pat.set(refc.AnchoredDiffusionDecoder, "forward", dec_eval)
    if arm_is(ctx, "ga_off_by_one"):
        # the refcv6 reach trigger re-installed where it is OBSERVABLE: `_pr_row` is filled
        # before `step += 1` and written after it, so `ga_*` keys reach ONLY the final row
        final = int((flag_values(argv, "--steps") or ["0"])[0])
        pat.set(T, "json", _GaFinalRowOnlyJson(T.json, final))
    if arm_is(ctx, "no_fmap_s8_passthrough"):
        # NEW-2's F3-class defect: the stride-8 map is computed and dropped by the pass-through.
        # ⛔ No fallback: the coordinator keys the check on DECODER_PASSTHROUGH, and an arm that
        # did not touch the real mechanism must be able to ESCAPE (and be reported as such).
        key = (ctx.prof.get("map_hires") or {}).get("fmap_key", "fmap_s8")
        _say("arm: " + _drop_from_passthrough(pat, (key,), fallback_pop=False))


def step_cost(step_times: list[float], warmup: int) -> dict:
    """s/step from the loop's own opt.step timestamps, after `warmup` intervals (compile, cuDNN
    autotune and loader warm-up are not the run's cost)."""
    iv = [b - a for a, b in zip(step_times, step_times[1:])]
    use = iv[warmup:] if len(iv) > warmup else []
    if not use:
        return {"n_intervals": len(iv), "n_used": 0, "s_per_step_mean": None}
    s = sorted(use)
    return {"n_intervals": len(iv), "n_used": len(use), "warmup_skipped": warmup,
            "s_per_step_mean": round(sum(use) / len(use), 4),
            "s_per_step_median": round(s[len(s) // 2], 4), "s_per_step_max": round(s[-1], 4)}


def judge_new2_live(prof: dict, run1: dict, config: dict | None, argv_local: list[str],
                    grads: dict, *, cuda: bool, approval: dict | None) -> tuple[list[str], bool, dict]:
    """-> (reasons, needs_pi, details) for NEW-2 in the smoke. Inert (no reasons) when off."""
    h = prof["map_hires"]
    reasons: list[str] = []
    det: dict[str, Any] = {}
    n_f = int(run1.get("fwd_train_calls", 0))
    det["fmap_hits"] = f"{run1.get('fmap_hits', 0)}/{n_f}"
    if n_f == 0:
        reasons.append("NEW-2: no training forward was observed")
    elif int(run1.get("fmap_hits", 0)) != n_f:
        reasons.append(f"NEW-2: `{h['fmap_key']}` reached the forward output on only "
                       f"{run1.get('fmap_hits', 0)} of {n_f} training forwards (the F3 "
                       f"pass-through class)")
    hf = run1.get("hires_first") or (["NEW-2: the output probe never ran"], {})
    det["outputs"] = hf[1]
    reasons += [r if r.startswith("NEW-2") else f"NEW-2: {r}" for r in hf[0]]
    tok = h["name_token"]
    live = [g for g in grads.get("groups", []) if tok in g["group"]]
    det["hires_leaf_groups"] = len(live)
    if not live:
        reasons.append(f"NEW-2: no declared-trainable leaf module carries '{tok}' -- the new "
                       f"decoder's gradient cannot be verified")
    cw = flag_values(argv_local, h["class_weights_flag"])
    if cw and Path(cw[0]).is_file():
        want = sha256_file(cw[0])
        found = []

        def walk(o, p=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    kp = f"{p}.{k}"
                    if isinstance(v, str) and "class_weight" in kp.lower() and _HEX64.match(v):
                        found.append((kp, v))
                    walk(v, kp)
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f"{p}[{i}]")
        walk(config or {})
        det["class_weights_sha256"] = {"file": want, "config_json": found[:4]}
        if not any(v == want for _, v in found):
            reasons.append(f"NEW-2: config.json does not record the class-weight JSON's sha256 "
                           f"{want[:16]} (found {[v[:16] for _, v in found]})")
    else:
        reasons.append(f"NEW-2: {h['class_weights_flag']} {cw} is not a readable file")
    cost = step_cost(run1.get("step_times") or [], int(h["cost_warmup_steps"]))
    cost["cuda_max_mem_gb"] = run1.get("cuda_max_mem_gb")
    cost["ref_s_per_step"] = h["cost_ref_s_per_step"]
    cost["max_ratio"] = h["cost_max_ratio"]
    needs_pi = False
    if not cuda:
        cost["verdict"] = "not measured: a CPU host's s/step says nothing about Thor"
        reasons.append("NEW-2: the cost gate needs the Thor smoke (CUDA); this host cannot answer it")
    elif cost.get("s_per_step_mean") is None:
        reasons.append("NEW-2: too few steps after warm-up to measure s/step")
    else:
        ratio = cost["s_per_step_mean"] / float(h["cost_ref_s_per_step"])
        cost["ratio"] = round(ratio, 4)
        if ratio > float(h["cost_max_ratio"]):
            ok_pi = (approval is not None
                     and cost["s_per_step_mean"] <= float(approval.get("max_s_per_step", -1)))
            cost["pi_approval"] = approval
            if ok_pi:
                cost["verdict"] = "above +25 % but within the PI's recorded approval"
            else:
                needs_pi = True
                cost["verdict"] = (f"PI-DECISION: {cost['s_per_step_mean']} s/step is "
                                   f"{ratio:.2f}x refcv6's {h['cost_ref_s_per_step']} "
                                   f"(> {h['cost_max_ratio']}x) -- SPEC 6.2 item 5")
        else:
            cost["verdict"] = "within +25 %"
    det["cost"] = cost
    return reasons, needs_pi, det


def _fmt_keys(fmt: str, h: dict, bands: Iterable[str] | None = None) -> list[str]:
    return [fmt.format(cls=c, band=b) for c in h.get("classes", ())
            for b in (tuple(bands) if bands is not None else h.get("bands", ()))]


def map_expected_eval_keys(prof: dict) -> list[str]:
    """The per-class x band keys an eval row must carry (without its `eval_` prefix): the POOLED
    IoU and the loss share -- 24 + 24 (LOGGING_SPEC_MAP10 sec. 2)."""
    h = prof.get("map_hires") or {}
    return _fmt_keys(h["eval_key_fmt"], h) + _fmt_keys(h["eval_lshare_fmt"], h)


def _finite_num(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def map_spelling_reasons(prof: dict) -> tuple[list[str], dict]:
    """The registered spelling (LITERALS in the profile) against the builder's module, when it
    exists: a drift between the two FAILS, it is never resolved silently."""
    h = prof["map_hires"]
    try:
        M = importlib.import_module("tanitad.models.map_head_hires")
    except Exception as e:                                # noqa: BLE001
        return [], {"module": f"not importable ({type(e).__name__}) -- literals only"}
    det: dict[str, Any] = {"module": "tanitad.models.map_head_hires"}
    reasons = []
    got = getattr(M, "CLASS_KEYS", None)
    det["CLASS_KEYS"] = list(got) if got is not None else None
    if got is None or tuple(got) != tuple(h["classes"]):
        reasons.append(f"G-MAP: map_head_hires.CLASS_KEYS {got!r} is not the registered spelling "
                       f"{tuple(h['classes'])!r} (LOGGING_SPEC_MAP10 sec. 1) -- one spelling, not "
                       f"two")
    # the bands of the DECLARED extent (A7: 1000 rows at 10 cm). ⚠️ `map_head_hires.BAND_KEYS` is
    # the /2 constant (3 bands); the extent-aware form is `band_keys_for_rows` (landed cef9709)
    fn = getattr(M, "band_keys_for_rows", None)
    bands = tuple(fn(int(h["logits_hw"][0]))) if fn is not None else getattr(M, "BAND_KEYS", None)
    det["bands_at_extent"] = list(bands) if bands is not None else None
    if bands is None or tuple(bands) != tuple(h["bands"]):
        reasons.append(f"G-MAP: map_head_hires' bands at the {h['logits_hw'][0]}-row extent "
                       f"{bands!r} are not the registered {tuple(h['bands'])!r}")
    pk = getattr(M, "per_class_key", None)
    if pk is not None and not reasons:
        mine = sorted(f"eval_{k}" for k in _fmt_keys(h["eval_key_fmt"], h))
        theirs = sorted(pk("iou", c, b, "eval_") for c in range(len(h["classes"]))
                        for b in h["bands"])
        det["per_class_key_agrees"] = mine == theirs
        if mine != theirs:
            reasons.append(f"G-MAP: map_head_hires.per_class_key spells the IoU keys "
                           f"{theirs[:2]}..., the registered spelling is {mine[:2]}...")
    return reasons, det


def judge_map_classes(prof: dict, stats: dict | None) -> tuple[list[str], dict]:
    """G-MAP (G-LIVE part): every one of the 8 map classes PRESENT in the smoke batches with at
    least `min_cells` labelled 10 cm cells has a finite, NON-ZERO gradient at its own cells --
    on all logit channels (its loss contribution: a weighted CE sends exactly zero gradient from
    a class whose weight is 0) and on its OWN channel. `stats[c]` accumulates, over the smoke's
    training steps, what a tensor hook on the 10 cm logits measured."""
    h = prof["map_hires"]
    if not stats or stats.get("_hooked", 0) == 0:
        return ["G-MAP: the per-class probe never ran -- no 10 cm logits tensor with a gradient "
                "and a 10 cm target were found in a training step"], {"stats": stats}
    reasons, rows = [], {}
    for i, c in enumerate(h["classes"]):
        s = stats.get(str(i)) or {"cells": 0, "grad_all": 0.0, "grad_own": 0.0,
                                  "nonfinite": False}
        present = int(s["cells"]) >= int(h["min_cells"])
        rows[c] = {**s, "present": present}
        if s.get("nonfinite"):
            reasons.append(f"G-MAP: class {c!r}: a non-finite gradient on the 10 cm logits")
        if not present:
            continue
        if not (float(s["grad_all"]) > 0.0):
            reasons.append(f"G-MAP: class {c!r} has {s['cells']} labelled cells in the smoke and "
                           f"ZERO loss contribution (no gradient at its cells)")
        elif not (float(s["grad_own"]) > 0.0):
            reasons.append(f"G-MAP: class {c!r}: zero gradient on its OWN logit channel")
    det = {"per_class": rows, "min_cells": h["min_cells"],
           "n_present": sum(1 for r in rows.values() if r["present"]),
           "n_hooked_steps": stats.get("_hooked", 0)}
    return reasons, det


def map_eval_key_reasons(prof: dict, eval_rows: list[dict], *,
                         weighted: bool = True) -> tuple[list[str], dict]:
    """G-MAP, read off the smoke's ARTIFACT (LOGGING_SPEC_MAP10 sec. 6, items 1, 3, 6). The FIRST
    in-run eval row carries all 24 IoU + 24 loss-share keys (an undefined IoU is `null`, never 0);
    all 48 argmax COUNTS the IoU is pooled from are finite -- plus the 48 RAW-rule counts when the
    loss is class-weighted; and every class has labelled cells in the gated band (refcv6 logged
    the drivable class only, for 38,000 steps)."""
    h = prof["map_hires"]
    if not eval_rows:
        return ["G-MAP: the smoke wrote no in-run eval row, so the per-class map IoU logging "
                "cannot be observed"], {}
    row = eval_rows[0]
    reasons: list[str] = []
    want = [f"eval_{k}" for k in map_expected_eval_keys(prof)]
    miss = [k for k in want if k not in row]
    det: dict[str, Any] = {"eval_step": row.get("step"), "expected": len(want),
                           "missing": miss[:30], "n_missing": len(miss), "weighted": weighted}
    if miss:
        reasons.append(f"G-MAP: the in-run eval logs {len(want) - len(miss)} of the {len(want)} "
                       f"per-class 10 cm IoU + loss-share keys (first missing: {miss[:4]})")
    fmts = list(h["eval_count_fmts"]) + (list(h["raw_count_fmts"]) if weighted else [])
    counts = [f"eval_{k}" for f in fmts for k in _fmt_keys(f, h)]
    bad = [k for k in counts if not _finite_num(row.get(k))]
    det.update(counts_expected=len(counts), counts_bad=bad[:20], n_counts_bad=len(bad))
    if bad:
        reasons.append(f"G-MAP: {len(bad)} of the {len(counts)} per-class argmax COUNT keys are "
                       f"absent or non-finite in the eval row (first: {bad[:4]}) -- the pooled "
                       f"IoU needs the counts, not per-batch IoUs")
    gb = h["gated_band"]
    empty = []
    for c in h["classes"]:
        v = row.get(f"eval_{h['n_fmt'].format(cls=c, band=gb)}")
        if not (_finite_num(v) and float(v) > 0):
            empty.append(c)
    det["classes_without_cells_in_gated_band"] = empty
    if empty:
        reasons.append(f"G-MAP: the smoke's eval windows carry no labelled {gb} m cells for "
                       f"{empty} -- extend the eval set; the check is not waived")
    return reasons, det


def map_train_row_reasons(prof: dict, train_rows: list[dict]) -> tuple[list[str], dict]:
    """G-MAP (LOGGING_SPEC_MAP10 sec. 6 item 2): on EVERY training row that carries the 10 cm
    loss, the 24 per-class x band loss CONTRIBUTIONS are present and sum to it within
    1e-5 x |loss| -- plus the rows' own 5-dp rounding (0.5e-5 per rounded term), stated."""
    h = prof["map_hires"]
    lk = h["loss_key"]
    lcs = _fmt_keys(h["lc_fmt"], h)
    rows = [r for r in train_rows if lk in r]
    if not rows:
        return [f"G-MAP: no training row carries `{lk}` -- the 10 cm loss never reached the "
                f"log"], {}
    slack = 0.5e-5 * (len(lcs) + 1)
    missing, bad = [], []
    for r in rows:
        miss = [k for k in lcs if k not in r]
        if miss:
            missing.append((r.get("step"), len(miss)))
            continue
        tot = sum(float(r[k]) for k in lcs)
        loss = float(r[lk])
        tol = float(h["lc_sum_rel_tol"]) * abs(loss) + slack
        if not (math.isfinite(tot) and abs(tot - loss) <= tol):
            bad.append((r.get("step"), tot, loss))
    det = {"n_rows": len(rows), "rounding_slack": slack, "rows_missing_lc": missing[:10],
           "rows_sum_mismatch": bad[:10]}
    reasons = []
    if missing:
        reasons.append(f"G-MAP: {len(missing)} of {len(rows)} training rows lack the per-class "
                       f"loss contributions `{h['lc_fmt']}` (first: step {missing[0][0]}, "
                       f"{missing[0][1]} keys missing)")
    if bad:
        reasons.append(f"G-MAP: on {len(bad)} training row(s) the per-class contributions do not "
                       f"sum to `{lk}` (first: step {bad[0][0]}: {bad[0][1]:.7g} vs "
                       f"{bad[0][2]:.7g})")
    return reasons, det


def map_watch_reasons(prof: dict, artifact: str | None) -> tuple[list[str], dict]:
    """G-MAP (LOGGING_SPEC_MAP10 sec. 6 item 4): the refcv7 Training Watch, built from THIS
    smoke's metrics.jsonl, carries all 24 per-class 10 cm IoU series and the thin-class alarm
    tile. The gate reads the ARTIFACT; with none given the item is a named FAIL, not a skip."""
    h = prof["map_hires"]
    if not artifact:
        return ["G-MAP: no refcv7 Training Watch artifact (--map-watch-artifact) built from the "
                "smoke's metrics.jsonl -- LOGGING_SPEC_MAP10 sec. 6 item 4 (24 per-class IoU "
                "series + the thin-class alarm tile) cannot be seen"], {"artifact": None}
    p = Path(artifact)
    if not p.is_file():
        return [f"G-MAP: the Watch artifact {artifact} does not exist"], {"artifact": artifact}
    txt = p.read_text(encoding="utf-8", errors="replace")
    want = [f"eval_{k}" for k in _fmt_keys(h["eval_key_fmt"], h)]
    miss = [k for k in want if k not in txt]
    alarm = re.search(r"thin[-_ ]?class[-_ ]?alarm", txt, re.I) is not None
    det = {"artifact": artifact, "sha256": sha256_file(p), "series_missing": miss[:10],
           "alarm_tile": alarm}
    reasons = []
    if miss:
        reasons.append(f"G-MAP: the Watch lacks {len(miss)} of the {len(want)} per-class IoU "
                       f"series (first: {miss[:3]})")
    if not alarm:
        reasons.append("G-MAP: the Watch carries no thin-class alarm tile")
    return reasons, det


def map_decision_rule_reasons(prof: dict, config: dict | None) -> tuple[list[str], dict]:
    """G-MAP (LOGGING_SPEC_MAP10 sec. 3 / sec. 6 item 6): config.json declares the argmax rule."""
    h = prof["map_hires"]
    node: Any = config or {}
    for k in h["decision_rule_path"]:
        node = node.get(k) if isinstance(node, dict) else None
    det = {"decision_rule": node, "path": ".".join(h["decision_rule_path"])}
    if node not in h["decision_rules"]:
        return [f"G-MAP: config.json `{det['path']}` is {node!r}, not one of "
                f"{list(h['decision_rules'])} -- the argmax rule must be DECLARED"], det
    return [], det


def map_eval_list_reasons(prof: dict, T, model=None) -> tuple[list[str], dict]:
    """G-MAP (G-DVB part): the per-class x band IoU and loss-share keys the TRAINER emits, read
    without a smoke. Two forms: a DECLARED module-level list (`eval_key_list_attrs`), or -- the
    landed NEW-2 (cef9709) -- the trainer's OWN eval-row builder `_eval_row_from_acc(acc, nb_e,
    model)` run on the BUILT model (it derives every class x band of the built branch's extent
    through `map_head_hires.derived_per_class`); the gate reads the keys that row carries."""
    h = prof["map_hires"]
    want = set(map_expected_eval_keys(prof))
    builder = getattr(T, "_eval_row_from_acc", None)
    if builder is not None and model is not None:
        try:
            row = builder({}, 1, model)
        except Exception as e:                            # noqa: BLE001 -- a named FAIL
            return [f"G-MAP: the trainer's eval-row builder raised on the built model: "
                    f"{type(e).__name__}: {e}"], {"source": "_eval_row_from_acc"}
        have = {str(k)[len("eval_"):] for k in row if str(k).startswith("eval_")}
        miss = sorted(want - have)
        det = {"source": "_eval_row_from_acc on the BUILT model", "n_emitted": len(have),
               "missing": miss[:30]}
        if miss:
            return [f"G-MAP: the trainer's eval-row builder omits {len(miss)} of the {len(want)} "
                    f"per-class x band 10 cm keys (first: {miss[:4]})"], det
        return [], det
    for attr in h.get("eval_key_list_attrs", ()):
        lst = getattr(T, attr, None)
        if lst is not None:
            have = {str(x).replace("eval_", "", 1) if str(x).startswith("eval_") else str(x)
                    for x in lst}
            miss = sorted(want - have)
            det = {"attr": attr, "n_declared": len(have), "missing": miss[:30]}
            if miss:
                return [f"G-MAP: the trainer's {attr} omits {len(miss)} of the {len(want)} "
                        f"per-class x band 10 cm IoU keys (first: {miss[:4]})"], det
            return [], det
    return [f"G-MAP: the trainer declares no per-class map eval key list "
            f"({list(h.get('eval_key_list_attrs', ()))}) and has no eval-row builder "
            f"(`_eval_row_from_acc`) the gate can run on the built model"], {"attr": None}


def judge_ga_reach(prof: dict, rows: list[dict], *, log_every: int, final_step: int,
                   groups: dict | None, ga_on: bool, hires: bool) -> tuple[list[str], dict]:
    """G-LIVE gradient reach (coordinator 2026-09-26, from the map-signal audit's MEASURED
    finding: refcv6 logged 0 `ga_*` keys in 4,621 rows, an off-by-one against `step += 1`).

    Every LOGGED training row -- each `log_every` multiple and the final step -- carries
    `ga_<group>` for every group that EXISTS on the built model and has a parameter the
    optimizer trains (`groups`, measured by the gate from the literal attribute paths), each
    finite and > 0; with the 10 cm head, also the four `ga_mh_*` parts. The smoke must run at
    least 2 x log_every steps, so a periodic row that is NOT the final row exists: the old
    off-by-one only ever fills the final row."""
    det: dict[str, Any] = {"ga_on": ga_on, "log_every": log_every, "final_step": final_step}
    if not ga_on and not hires:
        det["note"] = "inert: the arm builds neither the perception branch nor the tactical decoder"
        return [], det
    reasons: list[str] = []
    if final_step < 2 * log_every:
        reasons.append(f"G-LIVE ga-reach: the smoke ran {final_step} steps < 2 x log_every "
                       f"({log_every}) -- it cannot show a periodic row, where the off-by-one "
                       f"lives")
    want = sorted(g for g, v in (groups or {}).items()
                  if v.get("exists") and v.get("trainable_params", 0) > 0) if ga_on else []
    det["required_groups"] = want
    keys = [f"ga_{g}" for g in want]
    if hires:
        keys += list(prof["map_hires"]["ga_keys"])
    logged = [r for r in rows if isinstance(r.get("step"), int)
              and (r["step"] % max(1, log_every) == 0 or r["step"] == final_step)]
    det["logged_rows"] = [r["step"] for r in logged][:20]
    if not logged:
        reasons.append("G-LIVE ga-reach: the smoke wrote no logged training row")
        return reasons, det
    missing_rows, bad_vals = [], []
    for r in logged:
        miss = [k for k in keys if k not in r]
        if miss:
            missing_rows.append((r["step"], miss))
        for k in keys:
            if k in r and not (_finite_num(r[k]) and float(r[k]) > 0):
                bad_vals.append((r["step"], k, r[k]))
    det.update(required_keys=keys, rows_missing=[(s, m[:6]) for s, m in missing_rows[:10]],
               bad_values=bad_vals[:10])
    if missing_rows:
        s0, m0 = missing_rows[0]
        reasons.append(f"G-LIVE ga-reach: {len(missing_rows)} of {len(logged)} logged rows lack "
                       f"gradient-reach keys (first: step {s0} lacks {m0[:4]}) -- the reach log "
                       f"is DEAD (refcv6: 0 `ga_*` keys in 4,621 rows, an off-by-one against "
                       f"`step += 1`)")
    if bad_vals:
        s0, k0, v0 = bad_vals[0]
        reasons.append(f"G-LIVE ga-reach: {len(bad_vals)} reach value(s) are not finite and > 0 "
                       f"for a trainable group (first: step {s0} {k0} = {v0}) -- a group the "
                       f"optimizer trains receives no gradient")
    return reasons, det


def _first(d: dict, *paths: tuple[str, ...]) -> Any:
    for p in paths:
        node: Any = d
        for k in p:
            node = node.get(k) if isinstance(node, dict) else None
        if node is not None:
            return node
    return None


def _overfit_binding_reasons(tag: str, rec: dict, commit: str, argv_sha: str,
                             lit: dict) -> list[str]:
    """What the overfit PASS record itself binds (Master Mind 2026-09-27): its schema and the
    LAUNCH argv sha256. The CODE it ran on is bound by its closure record (`judge_closure`), not
    by a commit -- the binding runs can then start as soon as the MODEL code has landed; the
    record's own `launch_commit` is informative."""
    out = []
    if rec.get("schema") != lit.get("schema"):
        out.append(f"{tag}: the record's schema is {rec.get('schema')!r}, not "
                   f"{lit.get('schema')!r}")
    r_argv = _first(rec, ("launch_argv_sha256",), ("launch", "argv_sha256"))
    if argv_sha and str(r_argv) != argv_sha:
        out.append(f"{tag}: the record is bound to argv {str(r_argv)[:12] or '<none>'}, not the "
                   f"launch argv {argv_sha[:12]}")
    return out


#: what an overfit PASS certifies, stated in the gate's own PASS text (Master Mind 2026-09-27)
OVERFIT_SCOPE = ("covers the head IN ISOLATION (trunk + branch on the frozen frames); the "
                 "launch model's WIRING is G-LIVE's and G-DVB's job at the launch commit")
CLOSURE_SCHEMA = "tanitad.launch_gate.closure/1"
#: where the tree's own modules resolve from (mirrors `closure_run.IMPORT_ROOTS`)
_CLOSURE_IMPORT_ROOTS = ("stack", "taniteval", "stack/scripts", "tools")
#: the environment the binding runs must share with the launch host (Master Mind 2026-09-27:
#: a mismatch FAILS, it does not warn)
_CLOSURE_ENV_KEYS = ("torch", "timm", "cuda", "cudnn")


def _git_blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def _closure_digest(modules: list, probed: list, absent: list, data: list) -> str:
    """ONE definition with `closure_run.closure_digest` (a test pins the two together)."""
    return sha256_bytes(json.dumps({"modules": modules, "probed": probed, "absent": absent,
                                    "data": data}, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True).encode())


def _closure_resolve(tree: Path, name: str, extra_roots: Iterable[Path] = ()) -> Path | None:
    parts = [x for x in name.split(".") if x]
    if not parts:
        return None
    for base in [*(tree / r for r in _CLOSURE_IMPORT_ROOTS), *extra_roots]:
        q = base.joinpath(*parts)
        for cand in (q.parent / (q.name + ".py"), q / "__init__.py"):
            if cand.is_file():
                return cand
    return None


def static_eager_own_files(tree: Path, script_rel: str) -> list[str]:
    """The tree files a script imports at MODULE level (top-level statements, and the bodies of
    top-level `try` blocks), transitively over the tree's own modules -- what ANY run of it
    imports whenever those files exist. A closure record must hold every one: a record assembled
    from a hand list misses them, and so does a run made before one of them existed."""
    import ast
    extra = [(tree / script_rel).parent]
    out: set[Path] = set()
    queue = [tree / script_rel]

    def top_imports(body: list) -> Iterable[str]:
        for node in body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    yield a.name
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                yield node.module
                for a in node.names:
                    yield f"{node.module}.{a.name}"
            elif isinstance(node, ast.Try):
                yield from top_imports(node.body + node.orelse + node.finalbody)
    while queue:
        f = queue.pop()
        if f in out or not f.is_file():
            continue
        out.add(f)
        try:                                     # utf-8-sig: a BOM'd source is not skipped
            mod = ast.parse(f.read_text(encoding="utf-8-sig", errors="replace"))
        except SyntaxError:
            continue
        for n in top_imports(mod.body):
            parts = n.split(".")
            for k in range(1, len(parts) + 1):            # the packages on the way are imported
                t = _closure_resolve(tree, ".".join(parts[:k]), extra)
                if t is not None:
                    queue.append(t)
    return sorted(q.relative_to(tree).as_posix() for q in out)


def _launch_blob(tree: Path, rel: str, git_dir: str | None, commit: str | None) -> str | None:
    """The launch commit's blob for `rel`: from the launch tree, else (a file outside the shipped
    tree, e.g. a Research Lab helper the harness loaded by path) from git AT THE LAUNCH COMMIT."""
    f = tree / rel
    if f.is_file():
        return _git_blob(f.read_bytes())
    if git_dir and commit:
        env = {k: v for k, v in os.environ.items() if k not in _GIT_ENV}
        p = subprocess.run(["git", f"--git-dir={git_dir}", "rev-parse", f"{commit}:{rel}"],
                           env=env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        b = p.stdout.strip()
        if p.returncode == 0 and _HEX40.match(b):
            return b
    return None


def judge_closure(tag: str, closure: str | None, result: str | None, tree: str | os.PathLike,
                  harness: str | None, *, host_env: dict | None = None,
                  pmap: list[tuple[str, str]] | None = None,
                  must_have_data_sha: Iterable[str] = (), git_dir: str | None = None,
                  commit: str | None = None) -> tuple[list[str], dict]:
    """The CODE-CLOSURE binding of an overfit PASS record (SPEC_REFCV7 A11, section 16):

    * the closure is BINDING (`--binding`), the wrapped harness EXITED 0, and it wraps the
      profile's harness;
    * it names THIS PASS JSON by sha256;
    * every recorded module's (and probed module's) git blob is recomputed AT THE LAUNCH COMMIT
      and must be identical -- one changed blob refuses; the recorded lists re-hash to
      `closure_sha256`, the argv to `argv_sha256`;
    * no module name the run looked for and did NOT find now resolves in the launch tree, and the
      record holds every file the harness imports at module level on the launch tree (an imported
      but unrecorded module refuses);
    * no module came from OUTSIDE the tree and the interpreter (shadowing), no child process
      imported code the record cannot see;
    * every recorded data file still hashes the same on this host;
    * torch / timm / CUDA / cuDNN equal the launch host's (`host_env`; None skips ONLY in unit
      tests of the other rules -- every job passes the host's)."""
    if not closure or not Path(closure).is_file():
        return [f"{tag}: no closure record ({closure!r}) -- the PASS record is unbound "
                f"(SPEC_REFCV7 A11)"], {}
    c = read_json(closure)
    tree = Path(tree)
    det: dict[str, Any] = {"closure": str(closure), "sha256": sha256_file(closure),
                           "n_modules": len(c.get("modules") or []),
                           "n_probed": len(c.get("probed") or []),
                           "n_absent": len(c.get("absent") or []),
                           "n_data": len(c.get("data") or []), "scope": OVERFIT_SCOPE}
    reasons: list[str] = []
    if c.get("schema") != CLOSURE_SCHEMA:
        reasons.append(f"{tag}: the closure's schema is {c.get('schema')!r}")
    if c.get("binding") is not True:
        reasons.append(f"{tag}: the closure is NOT binding (an early run) -- only a run made with "
                       f"closure_run.py --binding can be consumed")
    if c.get("exit_status") != 0:
        reasons.append(f"{tag}: the wrapped harness exited {c.get('exit_status')!r} "
                       f"({c.get('error') or 'no error text'}) -- only a completed run binds")
    if harness and c.get("script") != harness:
        reasons.append(f"{tag}: the closure wraps {c.get('script')!r}, not the harness {harness!r}")
    res = c.get("result") or {}
    if not result or not Path(result).is_file():
        reasons.append(f"{tag}: the PASS JSON {result!r} is not readable here")
    elif res.get("sha256") != sha256_file(result):
        reasons.append(f"{tag}: the closure names a PASS JSON {str(res.get('sha256'))[:16]}, the "
                       f"record given is {sha256_file(result)[:16]} -- not the same run")
    mods = [list(x) for x in (c.get("modules") or [])]
    probed = [list(x) for x in (c.get("probed") or [])]
    absent = list(c.get("absent") or [])
    data = [list(x) for x in (c.get("data") or [])]
    if _closure_digest(mods, probed, absent, data) != c.get("closure_sha256"):
        reasons.append(f"{tag}: the recorded lists do not re-hash to closure_sha256 -- edited")
    if hashlib.sha256(json.dumps(c.get("argv"), sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=True).encode()).hexdigest() != c.get("argv_sha256"):
        reasons.append(f"{tag}: the closure's argv does not hash to its argv_sha256")
    changed, missing = [], []
    for rel, blob in [*mods, *probed]:
        lb = _launch_blob(tree, rel, git_dir, commit)
        if lb is None:
            missing.append(rel)
        elif lb != blob:
            changed.append(rel)
    det.update(changed=changed[:20], missing_at_launch=missing[:20])
    if changed or missing:
        reasons.append(f"{tag}: the launch commit differs from the harness run's code: "
                       f"{len(changed)} changed blob(s) {changed[:4]}, {len(missing)} absent at "
                       f"the launch {missing[:4]} -- the PASS does not describe this code")
    extra = [(tree / str(c.get("script") or "")).parent] if c.get("script") else []
    appeared = [n for n in absent if _closure_resolve(tree, n, extra) is not None]
    det["absent_now_present"] = appeared[:20]
    if appeared:
        reasons.append(f"{tag}: the launch tree provides {len(appeared)} module(s) the run looked "
                       f"for and did NOT find ({appeared[:4]}) -- the launch code takes an import "
                       f"path the PASS never ran")
    if not any(str(r).startswith("stack/tanitad/") for r, _ in mods):
        reasons.append(f"{tag}: the closure holds NO stack/tanitad module -- the harness did not "
                       f"import the tree it claims")
    if c.get("outside_tree"):
        reasons.append(f"{tag}: {len(c['outside_tree'])} module(s) were imported from OUTSIDE the "
                       f"run's tree and interpreter (shadowing): {c['outside_tree'][:3]}")
    kids = [k for k in (c.get("children") or []) if "python" in str(k).lower()]
    if kids:
        reasons.append(f"{tag}: the run spawned {len(kids)} Python child process(es) whose imports "
                       f"the closure cannot see: {kids[:2]}")
    if harness and (tree / harness).is_file():
        need = static_eager_own_files(tree, harness)
        have = {r for r, _ in mods}
        miss = [x for x in need if x not in have]
        det["static_eager_files"] = len(need)
        if miss:
            reasons.append(f"{tag}: the record lacks {len(miss)} file(s) the harness imports at "
                           f"module level on the launch tree ({miss[:4]}) -- an imported but "
                           f"unrecorded module")
    bad_data = []
    for path, sha in data:
        local = map_path(path, list(pmap or []))
        if not Path(local).is_file():
            bad_data.append((path, "absent here"))
        elif sha256_file(local) != sha:
            bad_data.append((path, "changed"))
    det["data_problems"] = bad_data[:10]
    if bad_data:
        reasons.append(f"{tag}: {len(bad_data)} data file(s) the run read are absent or changed on "
                       f"this host: {bad_data[:3]}")
    shas = {sha for _, sha in data}
    for want in must_have_data_sha:
        if want and want not in shas:
            reasons.append(f"{tag}: the launch class-weight file ({want[:16]}) is not among the "
                           f"data the run read")
    if host_env is not None:
        env = c.get("env") or {}
        diff = {k: (env.get(k), host_env.get(k)) for k in _CLOSURE_ENV_KEYS
                if env.get(k) != host_env.get(k)}
        det["env"] = {"record": env, "host": host_env}
        if diff:
            reasons.append(f"{tag}: the run's environment differs from this host's {diff} -- a "
                           f"changed venv means the binding runs are re-done")
    return reasons, det


def _host_env() -> dict:
    """The launch host's environment, by the SAME definition as `closure_run.dist_versions`:
    torch / timm by distribution metadata, CUDA / cuDNN from torch."""
    from importlib import metadata
    out: dict[str, Any] = {"python": platform.python_version()}
    for d in ("torch", "timm"):
        try:
            out[d] = metadata.version(d)
        except Exception:                                 # noqa: BLE001 -- absent => None
            out[d] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        try:
            out["cudnn"] = torch.backends.cudnn.version()
        except Exception:                                 # noqa: BLE001
            out["cudnn"] = None
    except Exception:                                     # noqa: BLE001
        out.update(cuda=None, cudnn=None)
    return out


def judge_map_overfit(prof: dict, record: str | None, commit: str, *, argv_sha: str = "",
                      class_weights_sha256: str | None = None) -> tuple[list[str], dict]:
    """G-MAP-OVERFIT: the PASS record the NEW-2 builder's harness (`map_hires_overfit.py`, schema
    `tanitad.g_map_overfit_record/1`) writes for the PRE-REGISTERED protocol
    (`…/2026-09-26-map-signal-audit/raw/PREREG_G_MAP_OVERFIT.md` + `gmo_spec.json`; SPEC 9 item
    2), re-judged against the prereg's LITERALS (`PROFILES[...]["map_overfit"]`) -- never against
    the bars the record carries:

    * the harness verdict `verdict.G_MAP_OVERFIT` is PASS;
    * the record binds THIS launch: `launch_commit`, `launch_argv_sha256`, the launch class-weight
      sha256 (not a dry run, not uniform), the FROZEN frame set, the A7 extent;
    * MAIN (`results.healthy.final`, the declared rule): all 8 classes >= 1,000 scored cells in
      the 0-20 m band (fewer is INCONCLUSIVE => FAIL, never "absent") and IoU >= the literal bar;
      the per-class CE ratio and the finite loss the prereg names (`verdict.MAIN`);
    * `lane_w0` FAILS lane; `s8_zeros` FAILS ALL FIVE thin classes; C1-C3 reproduced; the 1 ms
      time guard held on every clip;
    * ⛔ SPEC_REFCV7 23 (A18): the record RAN the registered protocol -- the A18 spec (sha256),
      the optimiser's steps / lr / batch / seed / lr_decay, and the map path (near lift, near
      refine blocks) -- each against `map_overfit`'s literals. A record of the superseded
      1,000-step protocol is REFUSED (the trap the box judge fell into with the prereg's lr);
    * ⛔ SPEC_REFCV7 24 (A19): under the profile's `overfit_main_only` policy a must-fail arm that
      did NOT RUN is excused and MAIN is re-judged from the record's own numbers (the harness's
      overall verdict reads FAIL without the arms and is NOT read); an arm that RAN is judged from
      its results and a pass is still a VOID. Without the policy every arm must run (pre-A19)."""
    mo = prof.get("map_overfit") or {}
    tag = "G-MAP-OVERFIT"
    if not record or not Path(record).is_file():
        return [f"{tag}: no overfit PASS record ({record!r}) -- a launch prerequisite"], {}
    rec = read_json(record)
    v = rec.get("verdict") if isinstance(rec.get("verdict"), dict) else {}
    det: dict[str, Any] = {"record": record, "sha256": sha256_file(record),
                           "verdict": v.get("G_MAP_OVERFIT"),
                           "launch_commit": rec.get("launch_commit"),
                           "launch_argv_sha256": rec.get("launch_argv_sha256")}
    reasons: list[str] = []
    # ⛔ SPEC_REFCV7 24 (A19): which of the GATE'S must-fail arms ran (a result, or a regression
    # row that says so); an ABSENT arm is excused only under the profile's A19 policy
    a19 = prof.get("overfit_main_only") or {}
    a19m = a19.get("map") if isinstance(a19.get("map"), dict) else {}
    results = rec.get("results") if isinstance(rec.get("results"), dict) else {}
    reg = v.get("regression_arms") if isinstance(v.get("regression_arms"), dict) else {}
    gate_mf = [(a_, tuple(c_), False) for a_, c_ in (mo.get("must_fail_any") or {}).items()]
    gate_mf += [(a_, tuple(c_), True) for a_, c_ in (mo.get("must_fail_all") or {}).items()]
    ran = {a_: isinstance(results.get(a_), dict) or bool((reg.get(a_) or {}).get("ran"))
           for a_, _c, _n in gate_mf}
    absent = [a_ for a_, r_ in ran.items() if not r_]
    main_only = bool(a19m) and bool(absent)
    det["a19"] = {"policy": a19.get("policy"), "source": a19.get("source"), "applied": main_only,
                  "absent_must_fail_arms": absent,
                  "harness_verdict_not_read": v.get("G_MAP_OVERFIT") if main_only else None}
    if main_only:
        det["a19"]["statement"] = (
            f"MAIN-ONLY binding under {a19.get('source')}: the must-fail arm(s) {absent} were NOT "
            f"RUN -- {a19m.get('inherited')} (evidence: {'; '.join(a19m.get('evidence') or ())})")
    elif v.get("G_MAP_OVERFIT") != "PASS":
        reasons.append(f"{tag}: the record's verdict is {v.get('G_MAP_OVERFIT')!r}, not PASS")
    reasons += _overfit_binding_reasons(tag, rec, commit, argv_sha, mo)
    cw = rec.get("class_weights") if isinstance(rec.get("class_weights"), dict) else {}
    det["class_weights"] = cw
    if cw.get("uniform") or cw.get("dry_run") or not cw.get("sha256"):
        reasons.append(f"{tag}: the record's class weights {cw} are not the launch weights "
                       f"(uniform, dry-run or unstamped files are refused)")
    elif class_weights_sha256 and cw["sha256"] != class_weights_sha256:
        reasons.append(f"{tag}: the record trained class weights {cw['sha256'][:16]}, the launch "
                       f"argv declares {class_weights_sha256[:16]}")
    frames = [tuple(x) for x in zip(rec.get("frames_sha12") or [], rec.get("raw_frames") or [])]
    det["frames_match"] = frames == [tuple(x) for x in mo.get("frames", ())]
    if not det["frames_match"]:
        reasons.append(f"{tag}: the record's frames ({len(frames)}) are not the FROZEN 16-frame set "
                       f"(frameset md5 {mo.get('frameset_md5', '')[:8]}...)")
    if rec.get("frameset_md5") != mo.get("frameset_md5"):
        reasons.append(f"{tag}: frameset md5 {rec.get('frameset_md5')} != the registered "
                       f"{mo.get('frameset_md5')}")
    ext = rec.get("extent") if isinstance(rec.get("extent"), dict) else {}
    for k, want in (mo.get("extent") or {}).items():
        if not (_finite_num(ext.get(k)) and float(ext[k]) == float(want)):
            reasons.append(f"{tag}: the record ran at extent {k}={ext.get(k)}, not the launch "
                           f"{want} (SPEC_REFCV7 12)")
    # ⛔ SPEC_REFCV7 23 (A18): the PROTOCOL the record ran
    ok_specs = [x for x in (mo.get("spec_sha256"), a19m.get("spec_sha256")) if x]
    if ok_specs and rec.get("spec_sha256") not in ok_specs:
        reasons.append(f"{tag}: the record ran spec {str(rec.get('spec_sha256'))[:16]}..., not the "
                       f"registered A18 spec {mo['spec_sha256'][:16]}... (SPEC_REFCV7 23)"
                       + (f" or the A19 MAIN-only spec {a19m['spec_sha256'][:16]}... (SPEC_REFCV7 24)"
                          if a19m.get("spec_sha256") else ""))
    opt = rec.get("optimiser") if isinstance(rec.get("optimiser"), dict) else {}
    det["optimiser"] = opt
    for k, want in (mo.get("protocol") or {}).items():
        got = opt.get(k)
        ok = (got == want if isinstance(want, dict)
              else (_finite_num(got) and not isinstance(got, bool) and float(got) == float(want)))
        if not ok:
            reasons.append(f"{tag}: the record's optimiser {k} is {got!r}, not the registered "
                           f"{want!r} (SPEC_REFCV7 23, A18)")
    for k in ("near_lift_m", "near_refine_blocks"):
        if k in mo and not (_finite_num(rec.get(k)) and not isinstance(rec.get(k), bool)
                            and float(rec[k]) == float(mo[k])):
            reasons.append(f"{tag}: the record ran {k}={rec.get(k)!r}, not the launch map "
                           f"path's {mo[k]} (SPEC_REFCV7 17 / 20)")
    band, rule = v.get("band"), rec.get("decision_rule") or v.get("decision_rule")
    det.update(band=band, decision_rule=rule)
    if band != mo.get("band"):
        reasons.append(f"{tag}: the verdict is gated on band {band!r}, not the registered "
                       f"{mo.get('band')!r}")
    if rule not in ("raw", "prior_corrected"):
        reasons.append(f"{tag}: the record states no decision rule ({rule!r}); the prereg gates on "
                       f"the launch config's DECLARED rule (sec. 5)")
    main = v.get("MAIN") if isinstance(v.get("MAIN"), dict) else {}
    final = _first(rec, ("results", "healthy", "final")) or {}
    n, iou = final.get("n") or {}, final.get("iou") or {}
    det["classes"] = {}
    for c, bar in (mo.get("iou_bars") or {}).items():
        nc, ic = n.get(c), iou.get(c)
        det["classes"][c] = {"n": nc, "iou": ic, "bar": bar}
        if not (_finite_num(nc) and nc >= mo["min_cells"]):
            reasons.append(f"{tag}: class {c!r} has {nc} scored cells < {mo['min_cells']} -- "
                           f"INCONCLUSIVE => FAIL (prereg sec. 6.1)")
        if not (_finite_num(ic) and float(ic) >= float(bar)):
            reasons.append(f"{tag}: class {c!r} IoU {ic} < the registered bar {bar}")
    if main_only:
        # A19: the CE ratio and the finite loss from the record's OWN numbers (MAIN's step-0 and
        # final per-class CE, its finite flag) -- never from the harness's verdict object
        h_ = results.get("healthy") if isinstance(results.get("healthy"), dict) else {}
        ce0 = (h_.get("step0") or {}).get("ce_mean") or {}
        ce1 = (h_.get("final") or {}).get("ce_mean") or {}
        lim = float(mo.get("ce_ratio_max", 0.5))
        bad_ce = sorted(c for c in (mo.get("iou_bars") or {})
                        if not (_finite_num(ce0.get(c)) and _finite_num(ce1.get(c))
                                and float(ce0[c]) > 0.0 and float(ce1[c]) <= lim * float(ce0[c])))
        finite_ok = h_.get("loss_finite_every_step") is True
    else:
        ce_ok = main.get("ce_ratio_ok") if isinstance(main.get("ce_ratio_ok"), dict) else {}
        bad_ce = sorted(c for c in (mo.get("iou_bars") or {}) if ce_ok.get(c) is not True)
        finite_ok = main.get("loss_finite_every_step") is True
    if bad_ce:
        reasons.append(f"{tag}: the per-class CE did not fall to <= 0.5x its step-0 value for "
                       f"{bad_ce} (prereg sec. 6.4)")
    if not finite_ok:
        reasons.append(f"{tag}: the MAIN loss is not recorded finite at every step (sec. 6.5)")
    det["regression_arms"] = reg
    det["must_fail"] = {}
    for arm, classes, need_all in gate_mf:
        a_ = reg.get(arm) or {}
        if not ran[arm] and main_only:           # ⛔ A19 excuses ABSENCE -- never a VOID
            det["must_fail"][arm] = f"NOT RUN -- excused under {a19.get('source')}; inherited"
            continue
        fin_ = (results.get(arm) or {}).get("final") if isinstance(results.get(arm), dict) else None
        if isinstance(fin_, dict) and isinstance(fin_.get("iou"), dict):
            # judged from the arm's OWN result against the gate's literal bars
            failed = [c for c, bar in (mo.get("iou_bars") or {}).items()
                      if not (_finite_num(fin_["iou"].get(c)) and float(fin_["iou"][c]) >= float(bar))]
            shown = a_ or {"ran": True, "failed": failed, "from": "results"}
        else:
            failed = list(a_.get("failed") or []) if ran[arm] else []
            shown = a_ or "not run"
        det["must_fail"][arm] = {"ran": ran[arm], "failed": failed}
        ok = ran[arm] and (set(classes) <= set(failed) if need_all else bool(set(classes) & set(failed)))
        void = " -- VOID: A19 excuses absence, never a VOID" if (a19m and ran[arm]) else ""
        if not ok and need_all:
            reasons.append(f"{tag}: must-fail arm {arm!r} must FAIL ALL of {list(classes)} "
                           f"({shown}) -- a thin class passing without image "
                           f"information means the harness scores something else" + void)
        elif not ok:
            reasons.append(f"{tag}: must-fail arm {arm!r} did not FAIL {list(classes)} "
                           f"({shown})" + void)
    ctl = v.get("controls") if isinstance(v.get("controls"), dict) else {}
    det["controls_reproduced"] = v.get("controls_reproduced")
    for c in mo.get("controls", ()):
        hit = [x for k, x in ctl.items() if str(k).split("_")[0] == c]
        if not (hit and isinstance(hit[0], dict) and hit[0].get("reproduced") is True):
            reasons.append(f"{tag}: control {c} is not recorded as reproduced "
                           f"({hit[0] if hit else 'absent'})")
    if v.get("controls_reproduced") is not True:
        reasons.append(f"{tag}: the harness does not record controls_reproduced = true")
    tg = v.get("time_guard") if isinstance(v.get("time_guard"), dict) else {}
    if tg.get("time_1ms_on_every_clip") is not True:
        reasons.append(f"{tag}: the 1 ms label-time guard did not hold on every clip "
                       f"({tg or 'absent'}) -- the prereg's gate condition (sec. 9a.6)")
    return reasons, det


def judge_box_overfit(prof: dict, record: str | None, commit: str, *, argv_sha: str = "",
                      argv: list[str] | None = None) -> tuple[list[str], dict]:
    """G-BOX-OVERFIT (SPEC_REFCV7 A9, A10 sec. 15.1): the record the box-head builder's harness
    (`stack/scripts/g_box_overfit.py`, record `tool: g_box_overfit.py`) writes, RE-JUDGED against
    the prereg's LITERALS (`PROFILES[...]["box_overfit"]`, PREREG_G_BOX_OVERFIT.md md5 594c7119...)
    -- never against the pass flags or bars the record carries:

    * `RESULT` is PASS; the prereg and the frame set are the registered bytes; the literals (steps
      2,000, batch 4, seed 0, 113 POSITIVE / 77 IGNORE after A10) are the prereg's; every arm trained
      with the LAUNCH optimiser (A13: AdamW, groups 5e-5 / 1e-4, weight decay 1e-4, clip 10) on A17's
      schedule (peaks through 1,799, cosine to 0 over 1,800-2,000); the record binds THIS launch argv;
      the harness's G-DVB self-check found no mismatch;
    * `main`'s FINAL row (step 2,000) meets EVERY sec. 5 criterion, recomputed here from the row
      values (the count and the presence ratio derived by the gate);
    * each must-fail arm (sec. 6) FAILS every criterion it names, recomputed the same way -- one
      passing makes the result VOID;
    * controls C1, C2, C3 read their known values; C4 (step 0) is REPORTED.
    ⚠️ PROVISIONAL until the box-head package lands: the schema is the harness as it stands in
    `…/2026-09-27-refcv7-box-head/code/new/`; each field the gate reads is named when absent."""
    bo = prof.get("box_overfit") or {}
    tag = "G-BOX-OVERFIT"
    if not record or not Path(record).is_file():
        return [f"{tag}: no box overfit PASS record ({record!r}) -- a launch prerequisite "
                f"(SPEC_REFCV7 A9/A10)"], {}
    rec = read_json(record)
    det: dict[str, Any] = {"record": record, "sha256": sha256_file(record),
                           "result": rec.get("RESULT"),
                           "harness_binding_informative": [rec.get("binding"), rec.get("commit")]}
    reasons: list[str] = []
    # ⛔ SPEC_REFCV7 24 (A19): a must-fail arm the record does not hold is excused only under the
    # profile's A19 policy; the harness then writes RESULT FAIL (its must-fail arms are absent), so
    # RESULT is NOT read -- main is re-judged below from its rows, as always
    a19 = prof.get("overfit_main_only") or {}
    a19b = a19.get("box") if isinstance(a19.get("box"), dict) else {}
    arms_a19 = rec.get("arms") if isinstance(rec.get("arms"), dict) else {}
    absent = [a_ for a_ in (bo.get("must_fail") or {}) if not isinstance(arms_a19.get(a_), dict)]
    main_only = bool(a19b) and bool(absent)
    det["a19"] = {"policy": a19.get("policy"), "source": a19.get("source"), "applied": main_only,
                  "absent_must_fail_arms": absent,
                  "harness_result_not_read": rec.get("RESULT") if main_only else None}
    if main_only:
        det["a19"]["statement"] = (
            f"MAIN-ONLY binding under {a19.get('source')}: the must-fail arm(s) {absent} were NOT "
            f"RUN -- {a19b.get('inherited')} (evidence: {'; '.join(a19b.get('evidence') or ())})")
    if rec.get("tool") != bo.get("tool"):
        reasons.append(f"{tag}: the record's tool is {rec.get('tool')!r}, not {bo.get('tool')!r}")
    if main_only:
        if rec.get("RESULT") == "VOID":
            reasons.append(f"{tag}: the record's RESULT is 'VOID' -- a must-fail arm passed "
                           f"(A19 excuses absence, never a VOID)")
    elif rec.get("RESULT") != "PASS":
        reasons.append(f"{tag}: the record's RESULT is {rec.get('RESULT')!r}, not PASS")
    # the launch argv: the gate's definition (ordered compact JSON), or the harness's own
    # `sha256(json.dumps(argv))` -- both are pure functions of the SAME list
    r_argv = rec.get("launch_argv_sha256")
    forms = {"gate": argv_sha}
    if argv is not None:
        forms["harness_json_dumps"] = hashlib.sha256(json.dumps(argv).encode()).hexdigest()
    hit = [k for k, v in forms.items() if v and v == r_argv]
    det["argv_form"] = hit[0] if hit else None
    if argv_sha and not hit:
        reasons.append(f"{tag}: the record is bound to argv {str(r_argv)[:12] or '<none>'}, not the "
                       f"launch argv {argv_sha[:12]}")
    for k in ("prereg_md5", "frameset_md5"):
        if rec.get(k) != bo.get(k):
            reasons.append(f"{tag}: {k} {rec.get(k)!r} != the registered {bo.get(k)!r}")
    lit = rec.get("literals") if isinstance(rec.get("literals"), dict) else {}
    for rk, pk in (("batch", "batch"), ("seed", "seed"), ("n_pos", "n_pos"), ("n_ign", "n_ignore")):
        if not (_finite_num(lit.get(rk)) and float(lit[rk]) == float(bo[pk])):
            reasons.append(f"{tag}: literal {rk} = {lit.get(rk)!r}, the prereg (with A10's "
                           f"reconciliation) fixes {bo[pk]}")
    # A17: the lr schedule, verbatim; A13: every arm's optimiser as built (the PEAK groups)
    if bo.get("lr_schedule") is not None and lit.get("lr_schedule") != dict(bo["lr_schedule"]):
        reasons.append(f"{tag}: literal lr_schedule = {lit.get('lr_schedule')!r}, SPEC_REFCV7 A17 fixes "
                       f"{dict(bo['lr_schedule'])!r}")
    want_opt = bo.get("optimizer")
    if want_opt is not None:
        arms0 = rec.get("arms") if isinstance(rec.get("arms"), dict) else {}
        for arm_name in ("main", *(bo.get("must_fail") or {})):
            spec = (arms0.get(arm_name) or {}).get("optimizer") if isinstance(arms0.get(arm_name), dict) else None
            if not isinstance(spec, dict):
                if arm_name in arms0:
                    reasons.append(f"{tag}: arm {arm_name!r} records no optimizer spec (SPEC_REFCV7 A13)")
                continue
            groups = spec.get("groups") if isinstance(spec.get("groups"), list) else []
            lrs = sorted(float(g.get("lr")) for g in groups if _finite_num(g.get("lr")))
            wds = {float(g.get("weight_decay")) for g in groups if _finite_num(g.get("weight_decay"))}
            ok = (spec.get("class") == want_opt["class"] and len(lrs) == len(groups)
                  and lrs == sorted(float(x) for x in want_opt["group_lrs"])
                  and wds == {float(want_opt["weight_decay"])} and len(wds) == 1
                  and _finite_num(spec.get("clip")) and float(spec["clip"]) == float(want_opt["clip"]))
            if not ok:
                reasons.append(f"{tag}: arm {arm_name!r} trained with {spec.get('class')!r} groups "
                               f"{lrs} weight decay {sorted(wds)} clip {spec.get('clip')!r}, not the "
                               f"LAUNCH optimiser (SPEC_REFCV7 A13: {want_opt['class']} "
                               f"{list(want_opt['group_lrs'])}, weight decay {want_opt['weight_decay']}, "
                               f"clip {want_opt['clip']})")
    if rec.get("steps") != bo.get("steps"):
        reasons.append(f"{tag}: steps = {rec.get('steps')!r}, the prereg fixes {bo.get('steps')}")
    if rec.get("dvb_mismatches"):
        reasons.append(f"{tag}: the harness's G-DVB self-check found {len(rec['dvb_mismatches'])} "
                       f"mismatch(es): {rec['dvb_mismatches'][:3]}")
    arms = rec.get("arms") if isinstance(rec.get("arms"), dict) else {}
    npos = float(bo["n_pos"])

    def judged(arm: str) -> tuple[dict, str | None]:
        a = arms.get(arm) if isinstance(arms.get(arm), dict) else None
        rows = (a or {}).get("rows") or []
        if not a or not rows:
            return {}, f"arm {arm!r} did not run (no rows)"
        final, r0 = rows[-1], rows[0]
        if final.get("step") != bo["steps"]:
            return {}, f"arm {arm!r} ends at step {final.get('step')!r}, not {bo['steps']}"
        d = dict(final)
        d["count_rel"] = (abs(float(final["n_conf"]) - npos) / npos
                          if _finite_num(final.get("n_conf")) else None)
        p0, pe = r0.get("presence"), final.get("presence")
        d["presence_ratio"] = (float(pe) / float(p0) if _finite_num(pe) and _finite_num(p0)
                               and float(p0) != 0.0 else None)
        c6 = ((a.get("criteria") or {}).get("6") or {}).get("value")
        finite = isinstance(c6, list) and bool(c6) and c6[0] is True

        def meets(k: str, op: str, lv: float) -> bool:
            x = d.get(k)
            if not _finite_num(x):
                return False                              # NaN / absent never passes
            return float(x) >= float(lv) if op == ">=" else float(x) <= float(lv)
        out = {c: all(meets(k, op, lv) for k, op, lv in terms)
               for c, terms in bo["criteria"].items()}
        out["6"] = out["6"] and finite
        return {"criteria": out, "final": {k: d.get(k) for k in (
            "ap2m", "prec", "rec", "n_conf", "count_rel", "centre_p50_m", "size_p50_m", "z_p50_m",
            "cls_acc", "presence_ratio")}, "finite": finite}, None
    main, why = judged("main")
    det["main"] = main
    if why:
        reasons.append(f"{tag}: {why}")
    else:
        for c, ok in main["criteria"].items():
            if not ok:
                reasons.append(f"{tag}: main fails prereg criterion {c} "
                               f"{[t[0] for t in bo['criteria'][c]]} at step {bo['steps']}: "
                               f"{ {t[0]: main['final'].get(t[0]) for t in bo['criteria'][c]} }"
                               + ("" if c != "6" or main["finite"] else " (loss not finite)"))
        if (arms.get("main") or {}).get("verdict") != "PASS":
            reasons.append(f"{tag}: the record's own main verdict is "
                           f"{(arms.get('main') or {}).get('verdict')!r}")
    det["must_fail"] = {}
    for arm, crits in (bo.get("must_fail") or {}).items():
        if main_only and arm in absent:          # ⛔ A19 excuses ABSENCE -- never a VOID
            det["must_fail"][arm] = f"NOT RUN -- excused under {a19.get('source')}; inherited"
            continue
        res, why = judged(arm)
        det["must_fail"][arm] = res.get("criteria") if res else why
        if why:
            reasons.append(f"{tag}: must-fail {why}")
        elif any(res["criteria"][c] for c in crits):
            reasons.append(f"{tag}: must-fail arm {arm!r} must FAIL prereg criteria {list(crits)} "
                           f"and passes {[c for c in crits if res['criteria'][c]]} -- the result is "
                           f"VOID (prereg sec. 7)")
    c12 = rec.get("C1_C2") if isinstance(rec.get("C1_C2"), dict) else {}
    ctl = {"C1": c12.get("C1"), "C2": c12.get("C2"), "C3": rec.get("C3")}
    for c in bo.get("controls", ()):
        v = ctl.get(c)
        if not (isinstance(v, dict) and v.get("pass") is True):
            reasons.append(f"{tag}: control {c} does not read its known value "
                           f"({v if v is not None else 'absent'})")
    if not isinstance(rec.get("C4_step0"), dict):
        reasons.append(f"{tag}: C4 (step 0, the launch init) is not reported")
    return reasons, det


def read_metrics(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:                                 # noqa: BLE001 -- a torn line is skipped
            continue
        if isinstance(r, dict):
            rows.append(r)
    return rows


def train_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if "loss" in r and "step" in r
            and not any(str(k).startswith("eval_") for k in r)]


def judge_ckpt(ckpt: dict | None, run1: dict, run2: dict | None, *, smoke_steps: int,
               extra: int, batch: int, seed: int, run2_rows: list[dict],
               run2_config: dict | None, run2_summary: dict | None) -> tuple[list[str], dict]:
    reasons: list[str] = []
    det: dict[str, Any] = {}
    if ckpt is None:
        return [f"the smoke wrote no ckpt.pt at step {smoke_steps}"], det
    miss = sorted({"model", "opt", "step", "data_pos"} - set(ckpt))
    det["ckpt_keys"] = sorted(ckpt)
    if miss:
        reasons.append(f"ckpt.pt lacks {miss} -- a resume could not restore them")
    if int(ckpt.get("step", -1)) != smoke_steps:
        reasons.append(f"ckpt.pt step {ckpt.get('step')} != {smoke_steps}")
    fin = run1.get("final")
    if not fin:
        return reasons + ["run 1 produced no in-memory final state to compare"], det
    ck_model = state_digests(ckpt.get("model", {}))
    ck_opt = optimizer_digests(ckpt.get("opt", {}))

    def cmp(name, a, b):
        ka, kb = set(a), set(b)
        diff = sorted(k for k in ka & kb if a[k] != b[k])
        out = {"n": len(ka), "only_first": sorted(ka - kb)[:10], "only_second": sorted(kb - ka)[:10],
               "n_differ": len(diff), "differ": diff[:10]}
        det[name] = out
        if ka != kb or diff:
            reasons.append(f"{name}: key sets or tensors differ ({out['n_differ']} differ, "
                           f"{len(ka - kb)}/{len(kb - ka)} one-sided)")
    cmp("save_model_vs_memory", fin["model"], ck_model)
    cmp("save_opt_state_vs_memory", fin["opt"]["state"], ck_opt["state"])
    if fin["opt"]["param_groups"] != ck_opt["param_groups"]:
        reasons.append("save: optimizer param_groups differ from the in-memory optimizer")
    dp = ckpt.get("data_pos") or {}
    det["data_pos"] = dp
    mem = run1.get("dpos_final") or {}
    if not dp or int(dp.get("epoch", -1)) != int(mem.get("epoch", -2)) or \
            int(dp.get("batch", -1)) != int(mem.get("batch", -2)):
        reasons.append(f"data_pos {dp} != the loop's final position {mem}")
    if dp and (int(dp.get("batch_size", -1)) != batch or int(dp.get("seed", -1)) != seed):
        reasons.append(f"data_pos batch_size/seed {dp.get('batch_size')}/{dp.get('seed')} != "
                       f"argv {batch}/{seed}")
    if run2 is None:
        return reasons + ["the resume run did not happen"], det
    det["resume_exit"] = run2.get("exit")
    if run2.get("exit"):
        reasons.append(f"the resume run failed: {str(run2['exit'])[:400]}")
    pl = run2.get("post_load")
    if not pl:
        reasons.append("the resume run never reached a training step (no post-load state)")
    else:
        cmp("resume_model_vs_saved", fin["model"], pl["model"])
        if pl.get("opt"):
            cmp("resume_opt_state_vs_saved", fin["opt"]["state"], pl["opt"]["state"])
            g1 = [{k: v for k, v in g.items() if k != "lr"} for g in fin["opt"]["param_groups"]]
            g2 = [{k: v for k, v in g.items() if k != "lr"} for g in pl["opt"]["param_groups"]]
            if g1 != g2:
                reasons.append("resume: optimizer param_groups (lr excepted) differ from the saved")
    its = run2.get("sampler_iters") or []
    det["resume_first_sampler_iter"] = its[0] if its else None
    if not its or its[0]["epoch"] != int(dp.get("epoch", -1)) or \
            its[0]["skip"] != int(dp.get("batch", -1)):
        reasons.append(f"resume: the sampler started at {its[0] if its else None}, the checkpoint "
                       f"says {dp} -- the data order would replay or skip")
    steps2 = sorted({int(r["step"]) for r in run2_rows})
    det["resume_row_steps"] = steps2
    if not steps2 or min(steps2) <= smoke_steps:
        reasons.append(f"resume: training rows {steps2} do not continue after step {smoke_steps}")
    do = (run2_config or {}).get("data_order") or {}
    det["resume_config_data_order"] = do
    if int(do.get("resumed_at_step", -1)) != smoke_steps:
        reasons.append(f"config.json data_order.resumed_at_step {do.get('resumed_at_step')} != "
                       f"{smoke_steps}")
    if not (run2_summary and run2_summary.get("done") is True
            and int(run2_summary.get("step", -1)) == smoke_steps + extra):
        reasons.append(f"resume run summary {run2_summary} is not done at {smoke_steps + extra}")
    return reasons, det


# --------------------------------------------------------------------------------------------- #
# G-CLOCK                                                                                         #
# --------------------------------------------------------------------------------------------- #
def load_clock_reference(paths: Iterable[str | os.PathLike]) -> tuple[dict[str, tuple], dict]:
    """q4c-style JSON(s) -> {sha12(clip_id): (grid_start_s, dt_s)} plus provenance. The per-clip
    clock MEASURED from the 100 Hz egomotion log by the A16 audit (an instrument independent of
    the trainer's sidecar reader). Clips whose fit residual is >= 5 ms are NOT references."""
    table: dict[str, tuple] = {}
    prov = []
    for p in paths:
        d = read_json(p)
        n = 0
        for split, v in (d.get("splits") or {}).items():
            for r in v.get("per_clip") or []:
                if float(r.get("fit_resid_max_ms", 1e9)) >= 5.0:
                    continue
                table[str(r["clip"])] = (float(r["grid_start_on_label_timeline_s"]),
                                         float(r["dt_s"]))
                n += 1
        prov.append({"path": str(p), "sha256": sha256_file(p), "n_clips": n,
                     "controls": d.get("controls")})
    return table, {"files": prov, "n_reference_clips": len(table)}


def judge_clock_windows(windows: Iterable[tuple], tol: float, *, n_clips: int,
                        cap: float) -> dict:
    """The pure G-CLOCK arithmetic. Each window: (clip_key, t_trainer, t_true | None, t_legacy,
    source) where `source` is the trainer's own clock source for the clip ("sidecar" = MEASURED,
    anything else = a fallback).

    * a MEASURED clip with a reference: |t_trainer - t_true| <= tol, or a VIOLATION;
    * a clip on a fallback clock, or MEASURED but without an independent reference, is
      UNVERIFIED -- admitted only while unverified clips <= `cap` x `n_clips` (the trainer's
      G3 rule, LABEL_CLOCK_MAX_UNVERIFIED_FRAC; the gate never raises it);
    * the legacy `(t + w - 1) * 0.1` clock is scored on every referenced window as a POSITIVE
      CONTROL: an instrument that reads 0 violations for it cannot see the defect it exists for.
    """
    n = n_viol = n_ctl = 0
    worst = 0.0
    fb_worst = 0.0
    viol_clips, fallback, unref, fb_ref = set(), set(), set(), set()
    for ck, t_tr, t_true, t_leg, source in windows:
        n += 1
        if t_true is not None and not (abs(float(t_leg) - float(t_true)) <= tol):
            n_ctl += 1
        if source != "sidecar":
            fallback.add(ck)
            if t_true is not None:
                fb_ref.add(ck)
                fb_worst = max(fb_worst, abs(float(t_tr) - float(t_true)))
            continue
        if t_true is None:
            unref.add(ck)
            continue
        d = abs(float(t_tr) - float(t_true))
        worst = max(worst, d)
        if not (d <= tol):
            n_viol += 1
            viol_clips.add(ck)
    unver = fallback | unref
    frac = len(unver) / max(int(n_clips), 1)
    return {"n_windows": n, "n_clips": int(n_clips), "n_violations": n_viol,
            "worst_abs_s": round(worst, 6), "n_violating_clips": len(viol_clips),
            "violating_clips": sorted(viol_clips)[:40],
            "n_unverified_clips": len(unver), "unverified_frac": round(frac, 6), "cap": cap,
            "cap_ok": frac <= cap,
            "n_fallback_clock_clips": len(fallback), "n_measured_but_unreferenced": len(unref),
            # None, not 0.0, when no fallback clip has a reference: "unmeasured" is not "exact"
            "fallback_worst_abs_s_vs_reference": round(fb_worst, 6) if fb_ref else None,
            "n_fallback_clips_with_reference": len(fb_ref),
            "unverified_clips": sorted(unver)[:40],
            "legacy_control_violations": n_ctl}


# --------------------------------------------------------------------------------------------- #
# the jobs (child side)                                                                           #
# --------------------------------------------------------------------------------------------- #
def _sentinel_argv(ctx: Ctx, keep_flags: Iterable[str], scratch: Path) -> tuple[list[str], dict]:
    """The launch argv on THIS host: `keep_flags` mapped to their local copies, every OTHER data
    path pointed at a NON-EXISTENT sentinel (a build that reads it must fail, not pass)."""
    keep = set(keep_flags)
    pmap = [tuple(x) for x in ctx.path_map]
    outs = set(ctx.prof.get("output_flags", ()))
    out, rec = [], {"kept": {}, "sentinel": {}}
    for f, vals in flag_pairs(ctx.argv):
        out.append(f)
        for v in vals:
            if f in outs or not _looks_like_path(v):
                out.append(v)
            elif f in keep:
                out.append(map_path(v, pmap))
                rec["kept"][f] = map_path(v, pmap)
            else:
                s = str(scratch / "__gate_unread__" / f.strip("-"))
                out.append(s)
                rec["sentinel"][f] = s
    if ctx.options.get("cpu_only") and has_flag(out, "--trunk-compile"):
        out = set_flag(out, "--trunk-compile", None)
        rec["dropped"] = {"--trunk-compile": "no Triton on the CPU host; torch.compile wraps the "
                                              "backbone CALL only (module tree and state_dict "
                                              "unchanged) -- the eval loader drops it too"}
    return out, rec


def _inputs_read(ctx: Ctx, flags: Iterable[str]) -> dict:
    pmap = [tuple(x) for x in ctx.path_map]
    out = {}
    for f, p in data_inputs(ctx.argv, ctx.prof):
        if f in flags:
            out[f] = fingerprint(map_path(p, pmap))
    return out


#: SPEC_REFCV7 2, G-DVB's regression arm "unwire one selection term": the model is BORN without a
#: built selection mechanism while the argv still declares it -- D-REFCV6-CONFIG-BUILD re-installed.
#: The attribute names are the decoder's own (the ones `declared_vs_built._sel_built` reads, landed
#: ab436ee); the arm tries the nav-compliance gate first, then the 8x8 tactical graft.
_UNWIRE_CANDIDATES = (("navc_gate", "--graft-nav-compliance"),
                      ("tac8_lat_to_anchor", "--graft-tac8-prior"))


def _arm_unwire_selection(model) -> dict:
    dec = getattr(getattr(model, "core", None), "decoder", None)
    for attr, flag in _UNWIRE_CANDIDATES:
        if dec is not None and getattr(dec, attr, None) is not None:
            setattr(dec, attr, None)
            return {"unwired": f"core.decoder.{attr}", "declared_by": flag}
    return {"unwired": None, "why": "the built decoder carries no selection term to unwire (the "
                                    "argv declares none, or the tree predates FIX-4)"}


def job_model(ctx: Ctx, checks: list[str]) -> dict[str, dict]:
    import torch
    T = load_trainer(ctx)
    out_dir = Path(ctx.out_dir)
    scratch = out_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    evs = {c: new_evidence(ctx, c) for c in checks}
    argv, arec = _sentinel_argv(ctx, ctx.prof["model_input_flags"], scratch)
    parser = T.build_parser()
    pat = _Patcher()
    if arm_is(ctx, "undeclared_equalize") or arm_is(ctx, "drop_fix3_field"):
        orig_pin = T._pin_trainer_cfg

        def pin_arm(cfg, args, _o=orig_pin):
            c = _o(cfg, args)
            enc = c.core.encoder
            if arm_is(ctx, "undeclared_equalize"):
                # the historical pattern: a lever set as an UNDECLARED attribute during the pin
                setattr(enc, "trunk_equalize_bottom_rows_undeclared",
                        int(getattr(args, "equalize_bottom_rows", 0) or 0))
            else:
                # the historical DROP: a rebuild through the dataclass's fields resets the lever
                fl = [f.name for f in dataclasses.fields(enc) if "equaliz" in f.name]
                if fl:
                    c.core.encoder = dataclasses.replace(enc, **{fl[0]: 0})
            return c
        pat.set(T, "_pin_trainer_cfg", pin_arm)
    # ⛔ profile-forbidden levers (refcv7: DrivoR-T's) are read off the ARGV first, so they are
    # reported even when the trainer refuses the argv before a model exists
    forb_reasons, forb_det = forbidden_lever_reasons(ctx.prof, ctx.argv)
    t0 = time.time()
    try:
        stop = "model" if ({"G-DVB", "G-EVAL"} & set(checks)) else "config"
        arm_rec: dict[str, Any] = {}
        on_built = ((lambda m: arm_rec.update(_arm_unwire_selection(m)) or dict(arm_rec))
                    if arm_is(ctx, "unwire_selection_term") else None)
        cap = run_trainer_until(T, argv, stop, on_built)
    except SystemExit as e:
        msg = f"the trainer REFUSED this argv while building: {e}"
        # the profile's argv-only rules are stated even when no model exists to check
        prof_rules = profile_argv_rules(ctx.prof, ctx.argv, ctx.options,
                                        [tuple(x) for x in ctx.path_map])[0]
        for c, ev in evs.items():
            ev["details"]["argv_local"] = arec
            if c == "G-DVB":
                ev["details"]["forbidden_levers"] = forb_det
                if arm_rec:
                    ev["details"]["arm_unwire"] = dict(arm_rec)
            finish_evidence(ev, "FAIL", (list(dict.fromkeys([*forb_reasons, *prof_rules]))
                                         if c == "G-DVB" else []) + [msg])
        return evs
    finally:
        pat.restore()
    build_s = round(time.time() - t0, 1)
    cfg, model, args = cap["cfg"], cap.get("model"), cap.get("args")
    common = {"argv_local": arec, "build_s": build_s,
              "model_params": (int(sum(p.numel() for p in model.parameters()))
                               if model is not None else None)}
    hires = hires_on(ctx.prof, ctx.argv)
    probe_out = {}
    # ---- G-HYG --------------------------------------------------------------------------- #
    if "G-HYG" in evs:
        ev = evs["G-HYG"]
        ev["details"].update(common)
        ev["inputs_read"] = _inputs_read(ctx, ("--anchors",))
        H, why = import_optional(ctx, "tanitad.train.config_hygiene", "missing_hygiene_module")
        if H is None:
            reasons, det = [f"tanitad.train.config_hygiene is not importable -- module not yet "
                            f"landed => the gate FAILS ({why})"], {}
        else:
            reasons, det = judge_hygiene(H, cfg)
        if ctx.prof.get("import_closure"):
            # every third-party module the trainer can reach must import in THIS (the launch) venv
            try:
                clo = import_closure(Path(ctx.tree), ctx.prof["trainer"])
                ir, idet = judge_import_closure(clo, ban=_arm_import_ban(ctx))
            except Exception as e:                        # noqa: BLE001 -- a crash is not a pass
                ir, idet = [f"G-HYG import closure crashed: {type(e).__name__}: {e}"], {}
            det["import_closure"] = idet
            reasons += ir
        ev["details"].update(det)
        finish_evidence(ev, "FAIL" if reasons else "PASS", reasons)
    # ---- G-EVAL (first: its probe forward also feeds G-DVB's NEW-2 shape checks) ---------- #
    if "G-EVAL" in evs:
        ev = evs["G-EVAL"]
        ev["details"].update(common)
        ev["inputs_read"] = _inputs_read(ctx, ctx.prof["model_input_flags"])
        try:
            reasons, det = eval_identity(ctx, T, model, args, scratch, probe_out)
        except SystemExit as e:
            reasons, det = [f"the eval loader REFUSED: {e}"], {}
        except Exception as e:                            # noqa: BLE001
            reasons, det = [f"G-EVAL crashed: {type(e).__name__}: {e}"], {
                "traceback": traceback.format_exc()[-2500:]}
        if hires and "trainer" in probe_out:
            r2, d2 = hires_output_reasons(ctx.prof, probe_out["trainer"],
                                          tuple(cfg.core.encoder.image_hw()))
            det["map_hires_outputs"] = d2
            reasons += [f"NEW-2: {r}" for r in r2]
        ev["details"].update(det)
        finish_evidence(ev, "FAIL" if reasons else "PASS", reasons)
    # ---- G-DVB --------------------------------------------------------------------------- #
    if "G-DVB" in evs:
        ev = evs["G-DVB"]
        ev["details"].update(common)
        ev["inputs_read"] = _inputs_read(ctx, ctx.prof["model_input_flags"])
        for k, fp in implicit_inputs(ctx.argv):          # the ImageNet init the build loaded
            ev["inputs_read"][k] = fp
        if ctx.options.get("nav_tau_record"):
            ev["inputs_read"]["gate:nav-tau-record"] = fingerprint(ctx.options["nav_tau_record"])
        D, why = import_optional(ctx, "tanitad.train.declared_vs_built", "missing_dvb_module")
        try:
            reasons, det = judge_dvb(D, model, args, parser, argv, why,
                                     forbid_kinds=tuple(ctx.prof.get("dvb_forbid_kinds", ())))
        except Exception as e:                            # noqa: BLE001 -- a crash is not a pass
            reasons, det = [f"G-DVB crashed: {type(e).__name__}: {e}"], {
                "traceback": traceback.format_exc()[-2000:]}
        fr, fd = forbidden_lever_reasons(ctx.prof, ctx.argv, model)
        det["forbidden_levers"] = fd
        reasons = fr + reasons
        pr, pd = passthrough_reasons(ctx.prof, ctx.argv, args, T)
        det["passthrough"] = pd
        reasons += pr
        pmap = [tuple(x) for x in ctx.path_map]
        nr, nd = nav_tau_reasons(ctx.prof, ctx.argv, ctx.options.get("nav_tau_record"), pmap)
        det["nav_compliance"] = nd
        reasons += [f"FIX-4: {r}" for r in nr]
        rr, rd = required_on_reasons(ctx.prof, ctx.argv, D, model, args,
                                     tau_file=ctx.options.get("nav_tau_record"))
        det["required_on"] = rd
        reasons += rr
        br_, bd_ = box_required_reasons(ctx.prof, model, args)
        det["box_required"] = bd_
        reasons += br_
        tfr, tfd = tau_file_flag_reasons(ctx.prof, ctx.argv, parser,
                                         ctx.options.get("nav_tau_record"), pmap)
        det["nav_tau_file_flag"] = tfd
        reasons += tfr
        # the profile's argv-only rules, here too, so this evidence says what the token will say
        reasons += required_flag_reasons(ctx.prof, ctx.argv)
        reasons += forbidden_value_reasons(ctx.prof, ctx.argv)
        det["pi_pending"] = pi_pending_reasons(ctx.prof, ctx.argv)
        if "on_built" in cap:
            det["arm_unwire"] = cap["on_built"]
        # FIX-5 on THIS host's cached weights (the coordinator's launch-prep item 2)
        det["fix5_stage_check"] = {"present_in_tree": cap.get("fix5_present"),
                                   "calls": cap.get("fix5", [])}
        if (cap.get("fix5_present") and _find_trunk(model) is not None
                and not has_flag(ctx.argv, "--trunk-no-pretrained")):
            calls = cap.get("fix5", [])
            if not calls:
                reasons.append("FIX-5: the per-stage pretrained check did NOT run during this "
                               "build (the trunk was built without it)")
            elif any(c.get("outcome") != "PASS" for c in calls):
                reasons.append(f"FIX-5: per-stage pretrained check: "
                               f"{[c.get('outcome') for c in calls]}")
        det["map_hires"] = {"on": hires}
        if hires:
            r2, d2 = hires_static_reasons(ctx.prof, ctx.argv, model, pmap)
            det["map_hires"].update(d2)
            reasons += [f"NEW-2: {r}" for r in r2]
            if arm_is(ctx, "map_drivable_only_logging") or arm_is(ctx, "map_drivable_only_static"):
                # the static half of G-MAP's arm: the declared list / the eval-row builder is
                # drivable-only (refcv6's pattern)
                for attr in ctx.prof["map_hires"].get("eval_key_list_attrs", ()):
                    if getattr(T, attr, None) is not None:
                        setattr(T, attr, tuple(k for k in getattr(T, attr) if "drivable" in k
                                               and "nondrivable" not in k))
                b0 = getattr(T, "_eval_row_from_acc", None)
                if b0 is not None:
                    others = tuple(f"_{c}_" for c in ctx.prof["map_hires"]["classes"]
                                   if c != "drivable")
                    T._eval_row_from_acc = lambda acc, nb, m, _b=b0, _o=others: {
                        k: v for k, v in _b(acc, nb, m).items()
                        if not (str(k).startswith("eval_map_hires_") and any(o in str(k)
                                                                              for o in _o))}
            lr, ld = map_eval_list_reasons(ctx.prof, T, model=model)
            det["map_hires"]["eval_key_list"] = ld
            reasons += lr
            if "trainer" in probe_out:
                r3, d3 = hires_output_reasons(ctx.prof, probe_out["trainer"],
                                              tuple(cfg.core.encoder.image_hw()))
                det["map_hires"]["outputs"] = d3
                reasons += [f"NEW-2: {r}" for r in r3]
            else:
                det["map_hires"]["outputs"] = ("no probe forward in this job: the 600x320 / "
                                               "240x128 / fmap_s8 shapes are asserted by G-EVAL "
                                               "(dev box) and G-LIVE (Thor)")
        ev["details"].update(det)
        finish_evidence(ev, "FAIL" if reasons else "PASS", reasons)
    del model
    return evs


def _load_loader(ctx: Ctx):
    path = Path(ctx.options.get("eval_loader") or "")
    if not path.is_file():
        raise GateError(f"eval loader {path} not found")
    os.environ["REFCV6_REPO"] = str(Path(ctx.tree).resolve())
    if ctx.options.get("eval_kit"):
        os.environ["REFCV6_KIT"] = str(ctx.options["eval_kit"])
    if ctx.options.get("eval_remap_overrides"):
        # the loader's own RECORDED override ({flag: local path}); every remap lands in its record
        os.environ["REFCV6_REMAP_OVERRIDES"] = str(ctx.options["eval_remap_overrides"])
    name = "gate_eval_loader"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, sha256_file(path)


def _forward_out(T, model, batch, args, seed: int):
    """The forward exactly as the trainer prepares it (`compute_losses_v3`), stopped at the
    model's output by a hook -- so no loss target is needed and nothing after it runs."""
    import torch
    box = {}

    def hook(_m, _i, out):
        raise _ForwardCaptured(out)
    h = model.register_forward_hook(hook)
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(seed)
            try:
                T.compute_losses_v3(model, batch, "cpu", mode=getattr(args, "mode", "diffusion"))
            except _ForwardCaptured as fc:
                box["out"] = fc.out
    finally:
        h.remove()
    if "out" not in box:
        raise GateError("the forward hook never fired -- the probe measured nothing")
    return box["out"]


#: flags that do not enter the MODEL (the run length, the output dir, the host-only compile)
_NON_MODEL_FLAGS = ("--steps", "--out", "--trunk-compile", "--eval-window-dump")


def _argv_model_view(argv: list[str]) -> list[tuple[str, list[str]]]:
    return sorted((f, v) for f, v in flag_pairs(list(argv)) if f not in _NON_MODEL_FLAGS)


def latest_smoke_config(ctx: "Ctx") -> Path | None:
    """The config.json the trainer wrote in THIS gate dir's newest smoke (same ctx, same argv)."""
    d = Path(ctx.out_dir) / "smoke"
    runs = sorted(d.glob("*/config.json"), key=lambda q: q.stat().st_mtime) if d.is_dir() else []
    return runs[-1] if runs else None


def eval_identity(ctx: Ctx, T, model_t, args_t, scratch: Path,
                  probe_out: dict | None = None) -> tuple[list[str], dict]:
    import torch
    reasons: list[str] = []
    det: dict[str, Any] = {}
    probe_out = {} if probe_out is None else probe_out
    model_t.eval()
    ck = scratch / "g_eval_trainer_state.pt"
    torch.save({"model": model_t.state_dict(), "step": 0}, ck)
    L, lsha = _load_loader(ctx)
    det["eval_loader"] = {"path": str(ctx.options.get("eval_loader")), "sha256": lsha}
    # ⛔ the RUN RECORD the loader rebuilds from must be THIS launch's: the config.json the trainer
    # wrote in this gate dir's smoke for this very argv. MEASURED 2026-09-26: rebuilt from
    # refcv6-r101-s0's own record, the loader faithfully reproduced refcv6 AS TRAINED (an
    # UNEQUALISED trunk, `trunk_equalize_rows_as_trained`) while the fixed trainer builds the 43
    # declared rows -- 59 of 137 forward outputs differed, every one downstream of the trunk.
    smoke_cfg = latest_smoke_config(ctx)
    stamps_path = str(smoke_cfg) if smoke_cfg else ctx.options.get("eval_stamps")
    stamps = read_json(stamps_path) if stamps_path else {}
    config = {k: v for k, v in stamps.items() if k != "argv"}
    config["argv"] = list(ctx.argv)
    same_argv = _argv_model_view(stamps.get("argv") or []) == _argv_model_view(ctx.argv)
    det["eval_stamps"] = {"path": stamps_path, "sha256": sha256_file(stamps_path)
                          if stamps_path else None,
                          "source": ("this gate's smoke (the trainer's own config.json for THIS "
                                     "argv)" if smoke_cfg else "--eval-stamps"),
                          "written_for_this_argv": same_argv,
                          "used_keys": sorted(k for k in config if k != "argv")}
    if not stamps_path:
        return ["G-EVAL needs the launch's own run record: run the smoke (G-LIVE) in this gate "
                "dir first, or pass --eval-stamps <a config.json written for THIS argv>"], det
    if not same_argv:
        reasons.append("G-EVAL: the run record the loader rebuilds from was written for ANOTHER "
                       "argv (model-affecting flags differ) -- a rebuild from another run's "
                       "record answers a different question; use this launch's smoke config.json")
    pat = _Patcher()
    if arm_is(ctx, "loader_skips_pin"):
        tr = L.trainer()
        pat.set(tr, "_pin_trainer_cfg", lambda cfg, args: cfg)
    try:
        model_l, cfg_l, args_l, rec = L.build_model(config, str(ck), device="cpu", strict=True)
    except RuntimeError as e:
        pat.restore()
        return [f"the eval loader's STRICT load failed: {str(e)[:600]}"], det
    finally:
        pat.restore()
    sd = rec.get("state_dict", {})
    det["strict_load"] = {"missing": sd.get("missing"), "unexpected": sd.get("unexpected"),
                          "n_keys": sd.get("n_keys")}
    if sd.get("missing") or sd.get("unexpected"):
        reasons.append(f"strict load: {len(sd.get('missing') or [])} missing / "
                       f"{len(sd.get('unexpected') or [])} unexpected keys (must be 0/0)")
    det["loader_departures"] = rec.get("departures")
    det["loader_argv_remap"] = rec.get("argv_remap")
    # every tensor the model holds, bit for bit
    dt, dl = state_digests(model_t.state_dict()), state_digests(model_l.state_dict())
    diff = sorted(k for k in set(dt) & set(dl) if dt[k] != dl[k])
    det["state_dict"] = {"n_trainer": len(dt), "n_loader": len(dl), "n_differ": len(diff),
                         "differ": diff[:12], "only_trainer": sorted(set(dt) - set(dl))[:12],
                         "only_loader": sorted(set(dl) - set(dt))[:12]}
    if set(dt) != set(dl) or diff:
        reasons.append(f"state_dict: {len(diff)} tensor(s) differ, key sets "
                       f"{'equal' if set(dt) == set(dl) else 'DIFFER'}")
    if hires_on(ctx.prof, ctx.argv):
        tok = ctx.prof["map_hires"]["name_token"]
        nt, nl = sum(tok in k for k in dt), sum(tok in k for k in dl)
        det["map_hires_state_keys"] = {"trainer": nt, "loader": nl}
        if nt == 0 or nl == 0:
            reasons.append(f"NEW-2: the 10 cm branch is not in both builds ('{tok}' keys: trainer "
                           f"{nt}, loader {nl}) -- the eval loader must build it identically")
    # the attributes train() sets ON the model that the forward / loss read
    attrs = {}
    for a in sorted(set(vars(model_t)) | set(vars(model_l))):
        if not (a.startswith("_w_") or a in ("_cls_class_weight", "_map_lift_valid_mask",
                                             "_box3d_visible_filter", "_bev_shuffle")):
            continue
        va, vb = getattr(model_t, a, "<absent>"), getattr(model_l, a, "<absent>")
        if torch.is_tensor(va) and torch.is_tensor(vb):
            same = tensor_digest(va) == tensor_digest(vb)
        else:
            same = (va == vb) if not (torch.is_tensor(va) or torch.is_tensor(vb)) else False
        attrs[a] = {"trainer": repr(va)[:80], "loader": repr(vb)[:80], "same": bool(same)}
        if not same:
            reasons.append(f"model attribute {a}: trainer {repr(va)[:60]} vs loader {repr(vb)[:60]}")
    det["model_attributes"] = attrs
    pb_t = {k: int(v) for k, v in T.v3.param_breakdown_v3(model_t).items()}
    pb_l = {k: int(v) for k, v in T.v3.param_breakdown_v3(model_l).items()}
    det["param_breakdown"] = {"trainer": pb_t, "loader": pb_l}
    if pb_t != pb_l:
        reasons.append(f"param_breakdown differs: trainer {pb_t} vs loader {pb_l}")
    # the forward on a FIXED real batch
    e_ds, e_eps, drec = L.build_eval_dataset(model_l, cfg_l, args_l, config,
                                             with_perception_targets=False)
    idx = int(L.inrun_eval_perm(e_ds, 1, 1)[0])
    item = e_ds[idx]
    batch = torch.utils.data.default_collate([item])
    e_i, t = e_ds.index[idx]
    ep_id = int(e_eps[e_i].episode_id)
    if getattr(model_t, "_perception", None) is not None:
        batch["map_ep"] = torch.tensor([ep_id], dtype=torch.long)
    det["probe"] = {"window_index": idx, "episode_sid": ep_id, "t": int(t),
                    "n_eval_windows": len(e_ds),
                    "inputs": {k: v for k, v in (drec or {}).items()
                               if k in ("labels", "label_clock", "n_episodes", "n_windows")},
                    "seed": 1234}
    model_l.eval()
    t0 = time.time()
    raw_t = _forward_out(T, model_t, batch, args_t, 1234)
    probe_out["trainer"] = raw_t
    out_t = output_digests(raw_t)
    # ⭐ control: the SAME model, the SAME batch, the SAME seed must give the same bytes -- or a
    # trainer-vs-loader difference below says nothing about the loader
    rep_t = output_digests(_forward_out(T, model_t, batch, args_t, 1234))
    n_nondet = sum(1 for k in out_t if rep_t.get(k) != out_t[k])
    det["determinism_control_outputs_differing"] = n_nondet
    if n_nondet:
        reasons.append(f"control: the trainer model's own forward is NOT reproducible on the fixed "
                       f"batch under a forked RNG ({n_nondet} outputs differ between two identical "
                       f"calls) -- the trainer-vs-loader comparison is blind")
    out_l = output_digests(_forward_out(T, model_l, batch, args_l, 1234))
    det["probe"]["forward_s_each"] = round((time.time() - t0) / 2, 1)
    odiff = sorted(k for k in set(out_t) | set(out_l) if out_t.get(k) != out_l.get(k))
    det["forward"] = {"n_outputs": len(out_t), "n_differ": len(odiff), "differ": odiff[:20]}
    if odiff:
        reasons.append(f"forward outputs differ on the fixed batch: {len(odiff)} of {len(out_t)} "
                       f"(first: {odiff[:5]})")
    n_tensor = sum(1 for v in out_t.values() if not v.startswith(("py:", "type:")))
    if n_tensor == 0:
        reasons.append("control: the forward produced no tensor outputs -- nothing was compared")
    # ⭐ negative control: a perturbed copy MUST differ (proves the comparison can see a change)
    p0 = next((p for n_, p in model_l.named_parameters() if n_.startswith("core.decoder")), None)
    if p0 is None:
        p0 = next(iter(model_l.parameters()))
    with torch.no_grad():
        saved = p0.detach().clone()
        p0.add_(1e-2)
    try:
        out_p = output_digests(_forward_out(T, model_l, batch, args_l, 1234))
    finally:
        with torch.no_grad():
            p0.copy_(saved)
    n_moved = sum(1 for k in out_t if out_p.get(k) != out_t[k])
    det["negative_control_outputs_moved"] = n_moved
    if n_moved == 0:
        reasons.append("negative control: perturbing a decoder parameter moved NO output -- the "
                       "comparison is blind")
    try:
        ck.unlink()
    except OSError:
        pass
    return reasons, det


def smoke_step_count(T, base_argv: list[str], configured: int) -> tuple[int, int]:
    """-> (steps, log_every). The smoke runs at least 2 x the LAUNCH's `--log-every` (the
    coordinator 2026-09-26): a periodic row that is not the final row must exist, because the
    refcv6 reach off-by-one fills ONLY the final row. Never fewer than `configured`."""
    le = int(getattr(T.build_parser().parse_args(list(base_argv)), "log_every", 1) or 1)
    return max(int(configured), 2 * le), le


def job_smoke(ctx: Ctx, checks: list[str]) -> dict[str, dict]:
    T = load_trainer(ctx)
    evs = {c: new_evidence(ctx, c) for c in checks}
    extra = int(ctx.prof["resume_extra_steps"])
    run_dir = Path(ctx.out_dir) / "smoke" / time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    run_dir.mkdir(parents=True, exist_ok=False)          # FRESH: an old ckpt.pt would resume
    base = ctx.local_argv()
    if ctx.options.get("cpu_only") and has_flag(base, "--trunk-compile"):
        base = set_flag(base, "--trunk-compile", None)
    steps, log_every = smoke_step_count(
        T, base, int(ctx.options.get("smoke_steps") or ctx.prof["smoke_steps"]))
    argv1 = set_flag(set_flag(base, "--steps", [str(steps)]), "--out", [str(run_dir)])
    all_inputs = _inputs_read(ctx, [f for f, _ in data_inputs(ctx.argv, ctx.prof)])
    for k, fp in implicit_inputs(ctx.argv):
        all_inputs[k] = fp
    parser = T.build_parser()
    args = parser.parse_args(argv1)
    terms, problems = declared_terms(T, args)
    _say(f"G-LIVE smoke: {steps} steps, out {run_dir}")
    run1 = run_smoke(ctx, T, argv1, snapshot="final")
    rows1 = read_metrics(run_dir / "metrics.jsonl")
    summ1 = read_json(run_dir / "summary.json") if (run_dir / "summary.json").is_file() else None
    gt = grad_table(run1["grads"], run1.get("grad_names") or {})
    if "G-LIVE" in evs:
        import torch
        ev = evs["G-LIVE"]
        ev["inputs_read"] = all_inputs
        admitted = {g: f"{why} [admitted while {flag} is in the argv]"
                    for g, flag, why in ctx.prof.get("live_dead_admitted", ())
                    if has_flag(ctx.argv, flag)}
        reasons, det = judge_live(terms, problems, run1["steps"], expect_steps=steps,
                                  summary=summ1, train_rows=train_rows(rows1), grads=gt,
                                  prior=run1["prior"], exit_msg=run1.get("exit"),
                                  admitted_dead=admitted)
        det.update({"smoke_argv": argv1, "run_dir": str(run_dir), "elapsed_s": run1["elapsed_s"],
                    "declared_terms": terms, "eval_calls": run1["eval_calls"],
                    "out_keys": run1["out_keys"][:120],
                    "departures_from_launch_argv": {"--steps": steps, "--out": "smoke dir",
                                                    **({"--trunk-compile": "dropped on a CPU host"}
                                                       if ctx.options.get("cpu_only") else {})}})
        # the measured cost, ALWAYS recorded (on Thor `max_memory_allocated` is the only
        # admissible memory reading)
        h = ctx.prof.get("map_hires") or {}
        det["cost"] = {**step_cost(run1.get("step_times") or [], int(h.get("cost_warmup_steps", 10))),
                       "cuda_max_mem_gb": run1.get("cuda_max_mem_gb"),
                       "batch": int(args.batch)}
        needs_pi = False
        cfg1 = read_json(run_dir / "config.json") if (run_dir / "config.json").is_file() \
            else None
        # SPEC_REFCV7 7: the ceiling mask at inference only, and the tau file in config.json
        cr, cd = judge_ceiling(ctx.prof, ctx.argv, run1.get("ceiling_calls", []),
                               eval_in_argv=has_flag(ctx.argv, "--eval-cache"))
        det["ceiling_filter"] = cd
        reasons += cr
        nc = (ctx.prof.get("nav_compliance") or {}).get("switch")
        if nc and has_flag(ctx.argv, nc):
            tr_, td = tau_config_reasons(ctx.options.get("nav_tau_record"), cfg1)
            det["nav_tau_in_config"] = td
            reasons += tr_
            if ctx.options.get("nav_tau_record"):
                ev["inputs_read"]["gate:nav-tau-record"] = fingerprint(
                    ctx.options["nav_tau_record"])
        det["map_hires"] = {"on": hires_on(ctx.prof, ctx.argv)}
        # the gradient-reach rows: every logged row, every trainable group (+ NEW-2's ga_mh_*)
        gr, gd = judge_ga_reach(ctx.prof, train_rows(rows1), log_every=log_every,
                                final_step=steps, groups=run1.get("ga_groups"),
                                ga_on=bool(run1.get("ga_on")), hires=det["map_hires"]["on"])
        det["ga_reach"] = {**gd, "groups": run1.get("ga_groups")}
        reasons += gr
        # D3 (fixes batch 2): the run's OWN declaration of what it logs, held to every row
        try:
            D = importlib.import_module("tanitad.train.declared_vs_built")
            clr = getattr(D, "check_logged_rows", None)
        except Exception:                                 # noqa: BLE001
            clr = None
        if clr is None:
            reasons.append("G-LIVE D3: declared_vs_built.check_logged_rows is absent (fixes batch 2 "
                           "not in this tree) -- the declared-vs-LOGGED rows cannot be checked")
        else:
            mm = clr(cfg1 or {}, rows1)
            det["d3_logged_rows"] = [str(m)[:240] for m in mm][:12]
            if mm:
                reasons.append(f"G-LIVE D3: {len(mm)} declared-vs-LOGGED mismatch(es): "
                               + " | ".join(str(m)[:200] for m in mm[:3]))
            dec = (cfg1 or {}).get("grad_reach_logging") or {}
            want_mh = (list(h.get("ga_keys", ())) if det["map_hires"]["on"] else [])
            if det["map_hires"]["on"] and flag_values(ctx.argv, h.get("bev_source_flag", "")) == [
                    "map_hires_pool"]:
                want_mh += list(h.get("ga_keys_pool", ()))
            miss_dec = [k for k in want_mh if k not in (dec.get("keys") or [])]
            if miss_dec:
                reasons.append(f"G-LIVE D3: config.json's grad_reach_logging does not DECLARE "
                               f"{miss_dec} -- an undeclared part's dead reach row would pass "
                               f"the owner's first-row check")
        # A9/A10 G-LIVE-PRES, and batch 3's grad-unreachable probe on the real batch
        pr_, pd_ = judge_live_pres(ctx.prof, run1.get("pres_last"), run1.get("pres_aux_last"))
        det["live_pres"] = pd_
        reasons += pr_
        gr_, gd_ = judge_grad_unreachable_probe(run1.get("grad_unreachable_probe"))
        det["grad_unreachable_probe"] = gd_
        reasons += gr_
        if det["map_hires"]["on"]:
            # G-MAP (LOGGING_SPEC_MAP10 sec. 6): per-class signal at the logits (the gate's own
            # hook), the eval row's 24 IoU + 24 loss shares + counts, the train rows' loss
            # contributions, the declared decision rule, and the one spelling
            cwv = flag_values(ctx.argv, h["class_weights_flag"])
            weighted = True
            try:
                cw_local = map_path(cwv[0], [tuple(x) for x in ctx.path_map]) if cwv else None
                wl = read_json(cw_local).get("weights") if cw_local else None
                weighted = not (isinstance(wl, list) and len(set(map(float, wl))) <= 1)
            except Exception:                             # noqa: BLE001 -- unknown => weighted
                weighted = True
            mr, md = judge_map_classes(ctx.prof, run1.get("map_class"))
            er, ed = map_eval_key_reasons(ctx.prof, [r for r in rows1 if any(
                str(k).startswith("eval_") for k in r)], weighted=weighted)
            sr, sd = map_train_row_reasons(ctx.prof, train_rows(rows1))
            dr, dd = map_decision_rule_reasons(ctx.prof, cfg1)
            spr, spd = map_spelling_reasons(ctx.prof)
            wa, wb = ctx.options.get("map_watch_artifact"), h.get("watch_builder")
            wrun = None
            if wa is None and wb:
                # the Watch is built from THIS smoke's metrics.jsonl, by the gate
                out_w = run_dir / "watch_refcv7.html"
                cmd = [str(x).format(python=sys.executable, metrics=str(run_dir / "metrics.jsonl"),
                                     out=str(out_w), tree=ctx.tree) for x in wb]
                try:
                    wp = subprocess.run(cmd, cwd=ctx.tree, timeout=1800, capture_output=True,
                                        text=True, encoding="utf-8", errors="replace")
                    wrun = {"cmd": cmd, "rc": wp.returncode,
                            "stderr_tail": (wp.stderr or "")[-400:]}
                except Exception as e:                    # noqa: BLE001 -- judged below
                    wrun = {"cmd": cmd, "rc": None, "error": f"{type(e).__name__}: {e}"}
                    _say(f"the Watch builder failed: {type(e).__name__}: {e}")
                wa = str(out_w) if out_w.is_file() else None
            wr, wd = map_watch_reasons(ctx.prof, wa)
            if wrun is not None:
                wd = {**wd, "builder_run": wrun}
                if wrun["rc"] != 0:
                    wr = [*wr, f"G-MAP: the Watch builder exited {wrun['rc']} "
                               f"({wrun.get('error') or wrun.get('stderr_tail', '')[-160:]})"]
            cw_sha = None
            if cwv:
                cwl = map_path(cwv[0], [tuple(x) for x in ctx.path_map])
                cw_sha = sha256_file(cwl) if Path(cwl).is_file() else None
            csr, csd = class_weights_stamp_reasons(ctx.prof, cfg1, cw_sha)
            det["g_map"] = {"classes": md, "eval_keys": ed, "train_rows": sd,
                            "decision_rule": dd, "spelling": spd, "watch": wd,
                            "class_weights_stamp": csd}
            reasons += mr + er + sr + dr + spr + wr + csr
            appr = None
            ap = ctx.options.get("pi_cost_approval")
            if ap:
                appr = read_json(ap)
                appr = {**appr, "_file_sha256": sha256_file(ap)}
            r2, needs_pi, d2 = judge_new2_live(ctx.prof, run1, cfg1, argv1, gt,
                                               cuda=bool(torch.cuda.is_available()),
                                               approval=appr)
            det["map_hires"].update(d2)
            reasons += r2
        ev["details"].update(det)
        finish_evidence(ev, "FAIL" if reasons else ("PI-DECISION" if needs_pi else "PASS"), reasons
                        or ([det["map_hires"]["cost"]["verdict"]] if needs_pi else []))
    if "G-CKPT" in evs:
        import torch
        ev = evs["G-CKPT"]
        ev["inputs_read"] = all_inputs
        ckp = run_dir / "ckpt.pt"
        ckpt = torch.load(ckp, map_location="cpu", weights_only=False) if ckp.is_file() else None
        run2 = rows2 = cfg2 = summ2 = None
        if ckpt is not None and not run1.get("exit"):
            argv2 = set_flag(argv1, "--steps", [str(steps + extra)])
            (run_dir / "summary.json").unlink(missing_ok=True)   # run 1's marker, read above
            n_before = len(rows1)
            run2 = run_smoke(ctx, T, argv2, snapshot="post_load")
            rows_all = read_metrics(run_dir / "metrics.jsonl")
            rows2 = train_rows(rows_all[n_before:])
            cfg2 = read_json(run_dir / "config.json") if (run_dir / "config.json").is_file() else None
            summ2 = read_json(run_dir / "summary.json") if (run_dir / "summary.json").is_file() \
                else None
        reasons, det = judge_ckpt(ckpt, run1, run2, smoke_steps=steps, extra=extra,
                                  batch=int(args.batch), seed=int(args.seed),
                                  run2_rows=rows2 or [], run2_config=cfg2, run2_summary=summ2)
        ev["details"].update(det)
        finish_evidence(ev, "FAIL" if reasons else "PASS", reasons)
    return evs


def clock_cap(prof: dict, T, argv: list[str]) -> tuple[float, list[str], dict]:
    """The unverified-clip cap: the trainer's LABEL_CLOCK_MAX_UNVERIFIED_FRAC (fixes agent, G3)
    and never more than the profile's. ⛔ An argv that RAISES it (`--label-clock-max-unverified`)
    is refused: "do not raise the cap" (coordinator 2026-09-26; the eval split is decision E2)."""
    pc = prof.get("clock_max_unverified_frac")
    base = 0.01 if pc is None else float(pc)
    t_cap = getattr(T, "LABEL_CLOCK_MAX_UNVERIFIED_FRAC", None)
    cap = min(base, float(t_cap)) if t_cap is not None else base
    det = {"cap": cap, "profile_cap": base, "trainer_constant": t_cap}
    reasons = []
    v = flag_values(argv, "--label-clock-max-unverified")
    if v:
        det["argv_cap"] = v[0]
        try:
            raised = float(v[0]) > cap
        except ValueError:
            raised = True
        if raised:
            reasons.append(f"the argv raises the G3 cap to {v[0]} (> {cap}): the gate does not "
                           f"raise the cap -- that is a recorded PI decision (E2), not a flag")
    return cap, reasons, det


def _clock_split(ctx: Ctx, T, cfg, args, split: str, ref: dict, tol: float, cap: float) -> dict:
    from tanitad.data import v2_dataset as v2d
    cache = getattr(args, "v2_cache" if split == "train" else "eval_cache", None)
    labels = getattr(args, "v7_labels" if split == "train" else "eval_labels", None)
    if not cache or not labels:
        return {"skipped": f"no {split} cache/labels in argv"}
    cache = cache[0] if isinstance(cache, (list, tuple)) else cache
    man_override = (ctx.options.get("clock_manifest") or {}).get(split)
    if man_override:
        import torch
        man = torch.load(man_override, map_location="cpu", weights_only=False)
        vcache = v2d.V2CompressedCache(str(cache), lru_size=1)
        key = "episode_uid" if "episode_uid" in man else "episode_id"
        eps = []
        for i in range(len(man["files"])):
            ns = int(man["n_stack"][i])
            s = int(man["image_size"][i])
            h = int(man.get("image_h", [s] * len(man["files"]))[i])
            w = int(man.get("image_w", [s] * len(man["files"]))[i])
            # the SAME construction as `build_v2_providers` (v2_dataset.py), frames absent
            eps.append(v2d.LazyV2Episode(vcache, i, man["poses"][i], man["actions"][i],
                                         int(man[key][i]), torch.Size((int(man["T_out"][i]),
                                                                       3 * ns, h, w))))
        source = {"mode": "manifest (a copied _v2manifest.pt; frames not present)",
                  "manifest": str(man_override), "manifest_sha256": sha256_file(man_override)}
    else:
        man = v2d.load_or_build_manifest(cache, rebuild=False, verbose=False)
        eps = v2d.build_v2_providers([cache], lru_size=1, verbose=False)
        source = {"mode": "cache dir", "cache": str(cache)}
    uids = man.get("episode_uid") or [v2d.stable_episode_id(c) for c in man["clip_id"]]
    sid_to_clip = {int(u): str(c) for u, c in zip(uids, man["clip_id"])}
    sid_ns = {int(u): int(ns) for u, ns in zip(uids, man["n_stack"])}
    # train():7092-7101 + 7205-7215 -- the trainer's dataset class, kwargs and clock resolution
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    lab, lman = T.v7l.load_v7_labels(labels, allow_oracle_nav=True)
    ds.v7_by_sid = {v2d.stable_episode_id(l.clip_id): l for l in lab}
    ds.v7_dt = 0.1
    if arm_is(ctx, "legacy_label_clock"):
        ds.legacy_label_clock = True                       # the historical (t + w - 1) * 0.1
    rep = ds.enable_clip_clock(getattr(args, "clip_clock_sidecar", None))
    res: dict[str, Any] = {}
    # the TRAINER'S OWN G3 (fixes agent), when the tree has it: run EXACTLY as train() runs it
    # (the eval split in E2(b) mode: unverified clips EXCLUDED from TACTICAL, the cap never
    # raised -- refc_v3_train.py, the in-training eval block), and its verdict recorded
    g3 = getattr(ds, "assert_label_clock_true", None)
    e2b = split == "eval" and ctx.prof.get("clock_eval_mode") == "exclude_tactical"
    res["mode"] = "E2(b): eval TACTICAL on verified clips only" if e2b else "cap"
    if callable(g3):
        try:
            if e2b:
                res["trainer_g3"] = g3(getattr(args, "clip_clock_sidecar", None), split=split,
                                       max_unverified_frac=cap, exclude_unverified_tactical=True)
            else:
                res["trainer_g3"] = g3(getattr(args, "clip_clock_sidecar", None), split=split,
                                       max_unverified_frac=cap)
        except SystemExit as e:
            res["trainer_g3"] = {"g3": "REFUSED", "message": str(e)[:800]}
        except TypeError as e:
            res["trainer_g3"] = {"g3": "REFUSED", "message": f"this tree's G3 has no E2(b) eval "
                                                             f"mode (API: {e})"}
    else:
        res["trainer_g3"] = {"g3": "absent from this tree (pre-fix)"}
    w = int(ds.window)
    lc = getattr(ds, "_label_clock", None) or {}
    labelled = {int(ds.episodes[e_i].episode_id) for e_i, _ in ds.index
                if int(ds.episodes[e_i].episode_id) in ds.v7_by_sid}
    # (an unlabelled clip has no tactical target to exclude: compared over LABELLED clips)
    res["tactical_excluded_sids"] = sorted(int(x) for x in
                                           (getattr(ds, "tactical_excluded_sids", ()) or ())
                                           if int(x) in labelled)
    # the gate's own reading of which labelled clips run on a FALLBACK clock (no measurement)
    res["fallback_sids"] = sorted(s for s in labelled
                                  if (lc.get(s) or (None, None, "unknown"))[2] != "sidecar")

    def gen():
        for e_i, t in ds.index:
            ep = ds.episodes[e_i]
            sid = int(ep.episode_id)
            if sid not in ds.v7_by_sid:
                continue                                   # no label record: no label time used
            clip = sid_to_clip.get(sid)
            key = sha12(clip) if clip else f"sid:{sid}"
            rr = ref.get(key)
            r = int(t) + w - 1
            t_true = None if rr is None else rr[0] + (r + sid_ns.get(sid, 3) - 1) * rr[1]
            src = (lc.get(sid) or (None, None, "unknown"))[2]
            yield (key, ds._now_s(ep, t), t_true, r * 0.1, src)
    n_clips = len({int(e_i) for e_i, _ in ds.index})
    res.update(judge_clock_windows(gen(), tol, n_clips=n_clips, cap=cap))
    res["source"] = source
    res["label_clock_report"] = rep
    res["labels_md5"] = getattr(lman, "md5", None)
    res["n_episodes"] = len(eps)
    res["n_labelled_clips"] = sum(1 for e in eps if int(e.episode_id) in ds.v7_by_sid)
    return res


def job_clock(ctx: Ctx, checks: list[str]) -> dict[str, dict]:
    T = load_trainer(ctx)
    ev = new_evidence(ctx, "G-CLOCK")
    argv = ctx.local_argv()
    if ctx.options.get("cpu_only") and has_flag(argv, "--trunk-compile"):
        argv = set_flag(argv, "--trunk-compile", None)
    args = T.build_parser().parse_args(argv)
    cap_ = run_trainer_until(T, argv, "config")           # the trainer's own pinned cfg
    cfg = cap_["cfg"]
    refs = ctx.options.get("clock_reference") or []
    ref, rprov = load_clock_reference(refs)
    tol = float(ctx.prof["clock_tolerance_s"])
    cap, reasons, cdet = clock_cap(ctx.prof, T, ctx.argv)
    det = {"reference": rprov, "tolerance_s": tol, "window": int(cfg.core.window),
           "unverified_cap": cdet,
           "rule_checked": "MEASURED-clock clips: |V3Dataset._now_s(ep, t) - (g0_ref + (t + w - 1 "
                           "+ n_stack - 1) * dt_ref)| <= tol on EVERY window of every labelled clip "
                           "(ref = an independent measurement, not the sidecar the trainer reads); "
                           "fallback-clock or unreferenced clips are UNVERIFIED and admitted up to "
                           "the cap"}
    if not ref:
        reasons.append("no clock reference was given (--clock-reference): nothing is verifiable")
    for split in ("train", "eval"):
        r = _clock_split(ctx, T, cfg, args, split, ref, tol, cap)
        det[split] = r
        if "skipped" in r:
            if split == "train":
                reasons.append(f"G-CLOCK {split}: {r['skipped']}")
            continue
        if r["n_windows"] == 0:
            reasons.append(f"{split}: ZERO labelled windows checked -- the check read nothing")
        if r["n_violations"]:
            reasons.append(f"{split}: {r['n_violations']} of {r['n_windows']} windows on MEASURED "
                           f"clocks read labels more than {tol} s from the independent reference "
                           f"({r['n_violating_clips']} clips, worst {r['worst_abs_s']} s)")
        if r.get("mode", "").startswith("E2(b)"):
            # E2(b): a clip with no MEASURED clock is admitted on the eval split only when it is
            # EXCLUDED from the TACTICAL family -- the gate's own fallback set must be exactly the
            # trainer's excluded set; clips measured but without an independent reference are
            # still capped (they ARE scored tactically)
            fb, ex = set(r.get("fallback_sids") or []), set(r.get("tactical_excluded_sids") or [])
            if fb != ex:
                reasons.append(f"{split} (E2(b)): the {len(fb)} labelled clips on a fallback "
                               f"clock are not exactly the {len(ex)} clips the trainer excludes "
                               f"from TACTICAL (only in the gate's set: {sorted(fb - ex)[:5]}; "
                               f"only in the trainer's: {sorted(ex - fb)[:5]})")
            unref_frac = r["n_measured_but_unreferenced"] / max(r["n_clips"], 1)
            if unref_frac > cap:
                reasons.append(f"{split} (E2(b)): {r['n_measured_but_unreferenced']} of "
                               f"{r['n_clips']} TACTICALLY scored clips have no independent "
                               f"reference -- above the {cap:.0%} cap")
            g3m = str((r.get("trainer_g3") or {}).get("mode", ""))
            if (r.get("trainer_g3") or {}).get("g3") == "PASS" and "E2(b)" not in g3m:
                reasons.append(f"{split}: the trainer's G3 did not run in its E2(b) mode ({g3m!r})")
        elif not r["cap_ok"]:
            reasons.append(f"{split}: {r['n_unverified_clips']} of {r['n_clips']} clips "
                           f"({r['unverified_frac']:.2%}) are UNVERIFIED ({r['n_fallback_clock_clips']}"
                           f" on a fallback clock, {r['n_measured_but_unreferenced']} without an "
                           f"independent reference) -- above the {cap:.0%} cap")
        if r["legacy_control_violations"] == 0:
            reasons.append(f"{split}: control -- the historical clock also reads 0 violations, so "
                           f"the instrument cannot see the defect it exists for")
        g3 = r.get("trainer_g3") or {}
        if g3.get("g3") == "REFUSED":
            reasons.append(f"{split}: the trainer's own G3 REFUSES: {g3.get('message', '')[:300]}")
    ins = ("--v2-cache", "--eval-cache", "--v7-labels", "--eval-labels", "--clip-clock-sidecar")
    ev["inputs_read"] = _inputs_read(ctx, ins)
    for i, r_ in enumerate(refs):
        ev["inputs_read"][f"gate:clock-reference[{i}]"] = fingerprint(r_)
    cm = ctx.options.get("clock_manifest") or {}
    if cm:
        # the dir itself was NOT read in manifest mode: bind a key the launch host cannot match,
        # so this evidence can never stand in for the real caches
        for split, flag in (("train", "--v2-cache"), ("eval", "--eval-cache")):
            if split in cm:
                ev["inputs_read"][flag] = {"kind": "manifest-mode",
                                           "sha256": sha256_file(cm[split])}
        det["note_manifest_mode"] = ("cache dirs not read (manifest mode): this evidence cannot "
                                     "stand in for the launch host's caches")
    ev["details"].update(det)
    finish_evidence(ev, "FAIL" if reasons else "PASS", reasons)
    return {"G-CLOCK": ev}


def _parse_junit(xml_path: Path) -> dict:
    import xml.etree.ElementTree as ET
    root = ET.parse(xml_path).getroot()
    failed, n = set(), 0
    skips: dict[str, str] = {}
    for tc in root.iter("testcase"):
        n += 1
        tid = f"{tc.get('classname', '')}::{tc.get('name', '')}"
        if any(ch.tag in ("failure", "error") for ch in tc):
            failed.add(tid)
        for ch in tc:
            if ch.tag == "skipped":
                # the REASON is part of the verdict: a skip is not a pass (G-SUITE, FIX-5 tests)
                skips[tid] = (ch.get("message") or ch.text or "").strip()[:400]
    return {"n_tests": n, "failed": sorted(failed), "n_skipped": len(skips),
            "skipped": dict(sorted(skips.items()))}


def skip_not_pass(res: dict, patterns: Iterable[str]) -> dict[str, str]:
    """{test id: reason} for every SKIP whose reason matches a not-a-pass pattern."""
    pats = [re.compile(p, re.I) for p in patterns]
    return {t: r for t, r in (res.get("skipped") or {}).items() if any(p.search(r) for p in pats)}


def run_suites(tree: Path, suites: Iterable[str], logs: Path, tag: str, env: dict,
               timeout_s: int) -> dict:
    out = {}
    for s in suites:
        d = tree / s
        if not d.is_dir():
            out[s] = {"absent": True}
            continue
        xml = logs / f"junit_{tag}_{s}.xml"
        log = logs / f"pytest_{tag}_{s}.log"
        xml.unlink(missing_ok=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                f"--junitxml={xml}", "-o", "junit_family=xunit2"],
                               cwd=str(d), env=env, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout_s)
        tail = [ln for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip()][-1:] or ["<none>"]
        if not xml.is_file():
            out[s] = {"rc": p.returncode, "error": "no junit xml was written (the artifact is the "
                                                   "verdict)", "tail": tail[0][-300:]}
            continue
        r = _parse_junit(xml)
        r.update(rc=p.returncode, tail=tail[0][-300:], elapsed_s=round(time.time() - t0, 1))
        out[s] = r
    return out


def job_suite(ctx: Ctx, checks: list[str]) -> dict[str, dict]:
    if ctx.prof.get("suite_mode") == "consume":
        return job_suite_consume(ctx, checks)
    ev = new_evidence(ctx, "G-SUITE")
    o = ctx.options
    git_dir, baseline = o.get("git_dir"), o.get("baseline")
    reasons: list[str] = []
    det: dict[str, Any] = {}
    if not git_dir or not baseline:
        finish_evidence(ev, "FAIL", ["G-SUITE needs --git-dir and --baseline (a clean tree of the "
                                     "launch commit is built from git, never from a worktree)"])
        return {"G-SUITE": ev}
    work = Path(o.get("suite_work") or (Path(ctx.out_dir) / "suite"))
    work.mkdir(parents=True, exist_ok=True)
    logs = Path(ctx.out_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cand_sha = resolve_commit(git_dir, ctx.commit)
    base_sha = resolve_commit(git_dir, baseline)
    suites = tuple(o.get("suites") or ctx.prof["suites"])
    trees = {}
    for tag, sha in (("cand", cand_sha), ("base", base_sha)):
        if tag == "base" and sha == cand_sha:
            trees["base"] = trees["cand"]                  # one clean tree serves both
            continue
        dest = work / f"{tag[0]}{sha[:10]}"
        raw = git_archive_bytes(git_dir, sha, ("stack", "taniteval", "tools"))
        want = tree_manifest_from_archive(raw)
        if not dest.exists():
            extract_archive(raw, dest)
        got = tree_manifest(dest)
        if tree_digest(got) != tree_digest(want):
            finish_evidence(ev, "ERROR", [f"the {tag} clean tree at {dest} does not match git "
                                          f"archive {sha[:12]}: {tree_diff(want, got)}"])
            return {"G-SUITE": ev}
        trees[tag] = (dest, tree_digest(want))
    det["candidate"] = {"commit": cand_sha, "tree_sha256": trees["cand"][1]}
    det["baseline"] = {"commit": base_sha, "tree_sha256": trees["base"][1]}
    if trees["cand"][1] != ctx.tree_sha256:
        reasons.append(f"the launch commit {cand_sha[:12]}'s tree ({trees['cand'][1][:16]}) is NOT "
                       f"the tree being launched ({ctx.tree_sha256[:16]}) -- the commit label lies")
    timeout = int(o.get("suite_timeout_s") or 3 * 3600)

    def env_for(tree: Path) -> dict:
        e = dict(os.environ)
        e["PYTHONPATH"] = os.pathsep.join([str(tree / "stack"), str(tree / "taniteval")])
        e.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                 OMP_NUM_THREADS=str(o.get("omp", 4)), MKL_NUM_THREADS=str(o.get("omp", 4)))
        if o.get("cpu_only"):
            e["CUDA_VISIBLE_DEVICES"] = "-1"
        e.pop(ARM_ENV, None)
        return e
    cache = work / f"b{base_sha[:10]}.suite.json"
    if cand_sha == base_sha:
        cres = run_suites(trees["cand"][0], suites, logs, "cand", env_for(trees["cand"][0]), timeout)
        bres = cres
        det["baseline_is_candidate"] = True
    else:
        cres = run_suites(trees["cand"][0], suites, logs, "cand", env_for(trees["cand"][0]), timeout)
        if cache.is_file() and read_json(cache).get("tree_sha256") == trees["base"][1]:
            bres = read_json(cache)["results"]
            det["baseline_cached"] = str(cache)
        else:
            bres = run_suites(trees["base"][0], suites, logs, "base", env_for(trees["base"][0]),
                              timeout)
            write_json(cache, {"commit": base_sha, "tree_sha256": trees["base"][1],
                               "results": bres, "written_utc": utc_now()})
    det["suites"] = {}
    for s in suites:
        c, b = cres.get(s, {}), bres.get(s, {})
        if c.get("absent"):
            det["suites"][s] = {"absent": True}
            continue
        if "failed" not in c or "failed" not in b:
            reasons.append(f"suite {s}: no parsable result (cand {c.get('error')}, "
                           f"base {b.get('error')})")
            det["suites"][s] = {"cand": c, "base": b}
            continue
        new = sorted(set(c["failed"]) - set(b["failed"]))
        fixed = sorted(set(b["failed"]) - set(c["failed"]))
        det["suites"][s] = {"cand_tests": c["n_tests"], "base_tests": b["n_tests"],
                            "cand_failed": len(c["failed"]), "base_failed": len(b["failed"]),
                            "new_failures": new[:50], "n_new_failures": len(new),
                            "fixed": fixed[:20], "cand_tail": c.get("tail"),
                            "base_tail": b.get("tail")}
        if c["n_tests"] == 0:
            reasons.append(f"suite {s}: ZERO tests collected on the candidate")
        if new:
            reasons.append(f"suite {s}: {len(new)} NEW failure(s) vs the baseline: {new[:6]}")
        # ⛔ a SKIP is not a pass: every skip is recorded BY REASON, and a skip whose reason says
        # the test could not run HERE (the FIX-5 resnet34 checkpoint absent from the local HF
        # cache) makes G-SUITE not-passed -- run it where the test runs
        by_reason: dict[str, int] = {}
        for r in (c.get("skipped") or {}).values():
            by_reason[r[:160]] = by_reason.get(r[:160], 0) + 1
        det["suites"][s]["cand_skipped"] = c.get("n_skipped")
        det["suites"][s]["cand_skip_reasons"] = dict(sorted(by_reason.items(),
                                                            key=lambda kv: -kv[1])[:40])
        snp = skip_not_pass(c, ctx.prof.get("suite_skip_not_pass", ()))
        if snp:
            det["suites"][s]["skip_not_pass"] = snp
            reasons.append(f"suite {s}: {len(snp)} test(s) SKIPPED for a reason that is NOT a pass "
                           f"({sorted(set(snp.values()))[0][:140]}): {sorted(snp)[:4]} -- run "
                           f"G-SUITE on a host where they RUN")
    # every guard's mutation arm must be RED
    audit = trees["cand"][0] / "stack" / "scripts" / "guard_mutation_audit.py"
    if o.get("skip_mutation_audit"):
        reasons.append("the guard mutation audit was skipped (--skip-mutation-audit): not a PASS")
    elif not audit.is_file():
        reasons.append("stack/scripts/guard_mutation_audit.py is absent from the launch commit")
    else:
        log = logs / "guard_mutation_audit.log"
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            p = subprocess.run([sys.executable, str(audit)], cwd=str(trees["cand"][0] / "stack"),
                               env=env_for(trees["cand"][0]), stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout)
        txt = log.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"=== (\d+)/(\d+) defects CAUGHT", txt)
        verdicts = re.findall(r"^(CAUGHT|ESCAPED|MISCREDITED)\s+(\S+)", txt, flags=re.M)
        det["mutation_audit"] = {"rc": p.returncode, "summary": m.group(0) if m else None,
                                 "verdicts": verdicts, "log": str(log)}
        if not m or p.returncode != 0 or int(m.group(1)) != int(m.group(2)) or int(m.group(2)) < 1:
            reasons.append(f"guard mutation audit: {m.group(0) if m else 'no summary line'} "
                           f"(rc {p.returncode}); every guard's mutation arm must be CAUGHT")
    ev["details"].update(det)
    finish_evidence(ev, "FAIL" if reasons else "PASS", reasons)
    return {"G-SUITE": ev}


# --------------------------------------------------------------------------------------------- #
# G-SUITE, consume mode (refcv7; the Master Mind 2026-09-27): (a) the Thor full-suite verdict of   #
# the LAUNCH commit, bound to it and to the tree it ran on; (b) G-SUITE-PINNED on the dev box      #
# --------------------------------------------------------------------------------------------- #
SUITE_BIND_SCHEMA = "tanitad.launch_gate.suite_bound/1"
_ARCHIVE_DONE = ".gate_archive_complete.json"


def env_failure_pin(prof: dict, tree: str | os.PathLike) -> tuple[dict | None, str | None]:
    """-> (the pinned env-failure record, problem). The file ships IN the tree (so the tree
    digest covers it) and its sha256 must equal the profile's literal: extending the list is a
    reviewed edit of both, never a quiet one."""
    pin = prof.get("suite_env_failures")
    if not pin:
        return None, f"profile {prof['name']} pins no environment-failure list"
    rel, want = pin
    p = Path(tree) / rel
    if not p.is_file():
        return None, f"the pinned env-failure list {rel} is absent from the tree"
    got = sha256_file(p)
    if got != want:
        return None, (f"the pinned env-failure list {rel} has sha256 {got[:16]}, the profile pins "
                      f"{want[:16]} -- a changed list is a reviewed edit of the profile")
    rec = read_json(p)
    if not isinstance(rec.get("ids"), list) or not isinstance(rec.get("files"), list):
        return None, f"{rel} carries no ids/files lists"
    return rec, None


def bind_suite_verdict(verdict: str | os.PathLike, cand_tree: str | os.PathLike,
                       commit: str) -> dict:
    """The Thor verdict, stamped with WHAT it measured: the commit it is claimed for and the
    digest of the CAND tree it actually ran on (the same `tree_digest` the token binds)."""
    if not _HEX40.match(str(commit)):
        raise GateError(f"suite-bind: {commit!r} is not a 40-char commit")
    tman = tree_manifest(Path(cand_tree))
    return {"schema": SUITE_BIND_SCHEMA, "commit": str(commit),
            "cand_tree": str(cand_tree), "cand_tree_sha256": tree_digest(tman),
            "cand_tree_files": len(tman), "verdict_file": str(verdict),
            "verdict_sha256": sha256_file(verdict), "verdict": read_json(verdict),
            "bound_utc": utc_now(), "host": _host()}


def cmd_suite_bind(a) -> int:
    _harden_streams()
    rec = bind_suite_verdict(a.verdict, a.cand_tree, a.commit)
    write_json(a.out, rec)
    _say(f"bound {a.verdict} to {a.commit[:12]} (CAND tree {rec['cand_tree_sha256'][:16]}, "
         f"{rec['cand_tree_files']} files) -> {a.out}")
    print(f"ZZSUITE-BOUND-{a.commit[:12]}ZZ", flush=True)
    return 0


def flaky_admitted(pin: dict, tree: str | os.PathLike | None = None) -> tuple[set[str], dict]:
    """The FLAKY ids a failure may carry WITHOUT failing G-SUITE: each must CITE its record (a
    repo path and a 64-hex sha256). When the record resolves from `tree` (the dev box's
    full-commit archive), its bytes must hash to the cited sha256; where it does not (Thor's
    stack/taniteval/tools tree), the citation is recorded as not re-read here."""
    ok, det = set(), {}
    for f in pin.get("flaky") or []:
        tid, rel, sha = f.get("id"), f.get("record"), str(f.get("record_sha256") or "")
        if not tid:
            continue
        if not rel or not _HEX64.match(sha):
            det[tid] = "NOT admitted: the pin cites no record (path + sha256)"
            continue
        got = None
        if tree is not None:
            rp = _resolve_repo_rel(Path(tree), rel)
            got = sha256_file(rp) if rp.is_file() else None
        if got is not None and got != sha:
            det[tid] = f"NOT admitted: the cited record hashes {got[:16]}, the pin says {sha[:16]}"
            continue
        ok.add(tid)
        det[tid] = ("admitted: record cited and re-read" if got else
                    "admitted: record cited (sha256 pinned; not resolvable on this host)")
    return ok, det


def suite_fail_ids(v: dict) -> set[str]:
    """Every test id the verdict saw FAIL on the candidate (gate_verdict.py's four lists)."""
    ids = set(map(str, v.get("pre_existing_fail") or []))
    for k in ("regressions", "new_fail"):
        ids |= {str(r[0]) if isinstance(r, (list, tuple)) else str(r) for r in (v.get(k) or [])}
    ids |= set(map(str, v.get("tip_skip_cand_fail") or []))
    return ids


def judge_suite_verdict(bound: dict | None, commit: str, tree_sha: str,
                        pin: dict, tree: str | os.PathLike | None = None) -> tuple[list[str], dict]:
    """G-SUITE (a): PASS = the bound verdict is for THIS commit and THIS tree, no test file crashed
    without a junit, and every candidate failure is in the PINNED environment list (a pinned test
    that now passes is fine; a NEW failure is a FAIL)."""
    if bound is None:
        return ["G-SUITE (a): no Thor full-suite verdict (--suite-verdict <the suite-bind output>)"], {}
    reasons: list[str] = []
    if bound.get("schema") != SUITE_BIND_SCHEMA:
        reasons.append(f"G-SUITE (a): the verdict is not bound (schema {bound.get('schema')!r}; "
                       f"run `launch_gate.py suite-bind` on the host that ran the suite)")
    if bound.get("commit") != commit:
        reasons.append(f"G-SUITE (a): the verdict is for commit {str(bound.get('commit'))[:12]}, "
                       f"not the launch commit {commit[:12]} -- a stale or other-commit verdict")
    if bound.get("cand_tree_sha256") != tree_sha:
        reasons.append(f"G-SUITE (a): the suite ran on tree {str(bound.get('cand_tree_sha256'))[:16]}"
                       f", not the launched tree {tree_sha[:16]}")
    v = bound.get("verdict") if isinstance(bound.get("verdict"), dict) else {}
    cand = v.get("cand") or {}
    fails = suite_fail_ids(v)
    pinned = set(pin.get("ids") or [])
    flaky_ok, flaky_det = flaky_admitted(pin, tree)
    outside = sorted(fails - pinned - flaky_ok)
    det = {"commit": bound.get("commit"), "cand_tree_sha256": bound.get("cand_tree_sha256"),
           "verdict_sha256": bound.get("verdict_sha256"), "cand": cand, "tip": v.get("tip"),
           "n_fail": len(fails), "n_pinned": len(pinned),
           "n_pinned_failing": len(fails & pinned), "n_pinned_now_passing": len(pinned - fails),
           "outside_pinned": outside[:40], "n_outside_pinned": len(outside),
           "flaky_failing": sorted(fails & flaky_ok), "flaky": flaky_det,
           "missing_on_cand": list(v.get("missing_on_cand") or [])[:20],
           "rc_without_junit": list(v.get("rc_without_junit") or [])[:10]}
    if not int(cand.get("n") or 0):
        reasons.append("G-SUITE (a): the verdict counts ZERO candidate tests")
    if outside:
        reasons.append(f"G-SUITE (a): {len(outside)} Thor failure(s) OUTSIDE the pinned "
                       f"environment list: {outside[:6]}")
    if v.get("rc_without_junit"):
        reasons.append(f"G-SUITE (a): {len(v['rc_without_junit'])} test file(s) crashed without a "
                       f"junit (collection error): {[r[1] if isinstance(r, list) else r for r in v['rc_without_junit'][:4]]}")
    return reasons, det


def job_suite_consume(ctx: Ctx, checks: list[str]) -> dict[str, dict]:
    ev = new_evidence(ctx, "G-SUITE")
    pin, prob = env_failure_pin(ctx.prof, ctx.tree)
    reasons = [f"G-SUITE: {prob}"] if prob else []
    sv = ctx.options.get("suite_verdict")
    bound = read_json(sv) if sv and Path(sv).is_file() else None
    if sv:
        ev["inputs_read"]["gate:suite-verdict"] = fingerprint(sv)
    r, det = judge_suite_verdict(bound, ctx.commit, ctx.tree_sha256, pin or {}, ctx.tree)
    reasons += r
    det["pin"] = {"file": (ctx.prof.get("suite_env_failures") or [None])[0],
                  "n_ids": len((pin or {}).get("ids") or []),
                  "n_files": len((pin or {}).get("files") or [])}
    ev["details"].update(det)
    finish_evidence(ev, "FAIL" if reasons else "PASS", reasons)
    return {"G-SUITE": ev}


#: git environment variables a test must NOT inherit (they would redirect a test's `git` into
#: another repository -- tanitad-push's shared index among them)
_GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE")


def pinned_env(dest: Path, o: dict, base: dict) -> dict:
    """The pinned files' test environment: the archive's tree first on PYTHONPATH, NO git
    redirection (`_GIT_ENV`: a test's `git` must never reach tanitad-push's shared index), NO
    `MKL_NUM_THREADS` -- test_amp_and_threads scrubs OMP_NUM_THREADS to measure the module's own
    setdefault, and an MKL value overrides torch's count (MEASURED 2026-09-27: "the fixed order
    must yield 6 threads, got 4") -- and no regression-arm switch."""
    env = {k: v for k, v in base.items() if k not in _GIT_ENV and k != "MKL_NUM_THREADS"}
    env["PYTHONPATH"] = os.pathsep.join([str(dest / "stack"), str(dest / "taniteval")])
    env.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS=str(o.get("omp", 4)),
               # the full tracked-tree credential scan RUNS here (it skips on ci_gate's 15 s budget)
               SECRET_SCAN_FULL="1")
    if o.get("cpu_only"):
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    env.pop(ARM_ENV, None)
    return env


def throwaway_repo(dest: Path, git_dir: str, commit: str) -> dict:
    """A repo INSIDE the archive whose objects come read-only from `git_dir` (alternates), HEAD at
    `commit`, index = HEAD. Idempotent; its HEAD is asserted by the caller."""
    env = {k: v for k, v in os.environ.items() if k not in _GIT_ENV}

    def g(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(dest), *args], env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
    made = False
    if not (dest / ".git").is_dir():
        g("init", "-q")
        made = True
    alt = dest / ".git" / "objects" / "info" / "alternates"
    alt.parent.mkdir(parents=True, exist_ok=True)
    want = Path(git_dir).resolve().joinpath("objects").as_posix()
    line = (want + "\n").encode("utf-8")
    if not alt.is_file() or alt.read_bytes() != line:
        # BYTES, LF: a text-mode write on Windows ends the line in CRLF, and git then reads the
        # path WITH the carriage return ("objects? does not exist" -- MEASURED 2026-09-27)
        alt.write_bytes(line)
    steps = [g("update-ref", "HEAD", commit), g("-c", "core.autocrlf=false", "read-tree", "HEAD")]
    hp = g("rev-parse", "--verify", "-q", "HEAD")
    head = hp.stdout.strip() if hp.returncode == 0 else None
    # armed like any clone: the repo's own credential gate (the pinned
    # test_secret_scan::test_hook_is_installed_and_current asserts it is installed and current)
    hook = None
    scan = dest / "tools" / "secret_scan.py"
    if head == commit and scan.is_file():
        hk = subprocess.run([sys.executable, str(scan), "--install-hook"], cwd=str(dest), env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
        hook = {"rc": hk.returncode, "tail": (hk.stdout + hk.stderr).strip()[-200:]}
    return {"made": made, "head": head, "alternates": want, "hook": hook,
            "errors": [p.stderr.strip()[-300:] for p in steps if p.returncode != 0]}


def stream_archive(git_dir: str, commit: str, dest: Path) -> dict:
    """`git -c core.autocrlf=false archive <commit>` of the WHOLE commit, STREAMED member by
    member into `dest` (never held in memory: the full tree is ~6.6 GB). A `.gate_archive_complete`
    marker is written LAST; a directory without it is a partial extraction and is refused."""
    dest.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, GIT_DIR=str(git_dir))
    p = subprocess.Popen(["git", "-c", "core.autocrlf=false", "archive", "--format=tar", commit],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    n = nbytes = 0
    skipped = []
    with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
        for m in tf:
            if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts:
                raise GateError(f"refusing archive member {m.name!r}")
            target = dest / m.name
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not m.isfile():
                skipped.append(m.name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            with open(target, "wb") as fh:
                shutil.copyfileobj(src, fh, 1 << 20)
            n += 1
            nbytes += int(m.size)
    err = p.stderr.read().decode(errors="replace")
    rc = p.wait()
    if rc != 0:
        raise GateError(f"git archive {commit[:12]} failed rc={rc}: {err[-400:]}")
    rec = {"commit": commit, "files": n, "bytes": nbytes, "skipped_non_files": skipped[:20],
           "written_utc": utc_now()}
    write_json(dest / _ARCHIVE_DONE, rec)
    return rec


def judge_pinned_run(files: list[dict], results: dict, skip_ok: Iterable[str],
                     flaky_ok: Iterable[str] = ()) -> tuple[list[str], dict]:
    """G-SUITE-PINNED: every test in the pinned files PASSES, or SKIPS for a REGISTERED reason;
    a file that wrote no junit (collection error, crash, timeout) is a FAIL. A registry entry is a
    (test-id regex, reason regex) PAIR -- the reason admits THAT test only -- or, legacy, a reason
    regex alone."""
    pats = [(re.compile(x[0]), re.compile(x[1], re.I)) if isinstance(x, (tuple, list))
            else (re.compile(r".*"), re.compile(x, re.I)) for x in skip_ok]
    reasons: list[str] = []
    det: dict[str, Any] = {"files": {}, "unregistered_skips": {}, "failures": []}
    for f in files:
        key = f"{f['sub']}/{f['file']}"
        r = results.get(key) or {}
        det["files"][key] = {k: r.get(k) for k in ("rc", "n_tests", "n_skipped", "elapsed_s")}
        if "failed" not in r:
            reasons.append(f"G-SUITE-PINNED: {key} produced no junit (rc {r.get('rc')}): "
                           f"{str(r.get('tail'))[-200:]}")
            continue
        if not r.get("n_tests"):
            reasons.append(f"G-SUITE-PINNED: {key} collected ZERO tests")
        det["failures"] += r["failed"]
        for tid, why in (r.get("skipped") or {}).items():
            if not any(ip.fullmatch(tid) and rp.search(why) for ip, rp in pats):
                det["unregistered_skips"][tid] = why[:200]
    fl = set(flaky_ok)
    det["flaky_failing"] = sorted(t for t in det["failures"] if t in fl)
    det["failures"] = [t for t in det["failures"] if t not in fl]
    if det["failures"]:
        reasons.append(f"G-SUITE-PINNED: {len(det['failures'])} test(s) FAIL in the pinned files "
                       f"on a full-commit archive: {det['failures'][:6]}")
    if det["unregistered_skips"]:
        first = sorted(det["unregistered_skips"].items())[0]
        reasons.append(f"G-SUITE-PINNED: {len(det['unregistered_skips'])} test(s) SKIPPED for a "
                       f"reason that is not registered (a skip is not a pass), e.g. {first[0]}: "
                       f"{first[1][:120]}")
    det["n_failures"] = len(det["failures"])
    det["n_unregistered_skips"] = len(det["unregistered_skips"])
    return reasons, det


def job_suite_pinned(ctx: Ctx, checks: list[str]) -> dict[str, dict]:
    """G-SUITE-PINNED (Master Mind 2026-09-27): on a `git -c core.autocrlf=false archive` of the
    FULL launch commit (Research Lab files included, the HF cache present), run exactly the test
    FILES of the pinned Thor environment list -- where test_refcv6_trunk actually RUNS -- and
    every guard's mutation arm (SPEC 2)."""
    ev = new_evidence(ctx, "G-SUITE-PINNED")
    o = ctx.options
    reasons: list[str] = []
    det: dict[str, Any] = {}
    git_dir = o.get("git_dir")
    pin, prob = env_failure_pin(ctx.prof, ctx.tree)
    if prob or not git_dir:
        finish_evidence(ev, "FAIL", [f"G-SUITE-PINNED: {prob or 'needs --git-dir (the FULL commit is archived from git, never a worktree)'}"])
        return {"G-SUITE-PINNED": ev}
    commit = resolve_commit(git_dir, ctx.commit)
    work = Path(o.get("suite_work") or (Path(ctx.out_dir) / "suite"))
    work.mkdir(parents=True, exist_ok=True)
    dest = work / f"f{commit[:10]}"
    if dest.exists() and not (dest / _ARCHIVE_DONE).is_file():
        finish_evidence(ev, "ERROR", [f"G-SUITE-PINNED: {dest} exists WITHOUT its completion "
                                      f"marker -- a partial extraction; remove it (by hand) and "
                                      f"re-run"])
        return {"G-SUITE-PINNED": ev}
    if not dest.exists():
        _say(f"G-SUITE-PINNED: archiving the FULL commit {commit[:12]} -> {dest}")
        det["archive"] = stream_archive(git_dir, commit, dest)
    else:
        det["archive"] = read_json(dest / _ARCHIVE_DONE)
    if det["archive"].get("commit") != commit:
        finish_evidence(ev, "ERROR", [f"G-SUITE-PINNED: {dest} holds {det['archive'].get('commit')}"])
        return {"G-SUITE-PINNED": ev}
    # ⛔ git reads (test_refcv6_diffusion, secret_scan's hook test) need a repository. NEVER
    # tanitad-push's: its SHARED INDEX carries 1,297 real staged deletions, and some tests run
    # `git init` / `git add` / `git diff --cached` (Master Mind 2026-09-27). A THROWAWAY repo
    # inside the archive borrows the objects read-only through `alternates`; nothing is copied
    # and nothing can write into tanitad-push.
    det["throwaway_repo"] = throwaway_repo(dest, git_dir, commit)
    if det["throwaway_repo"].get("head") != commit:
        finish_evidence(ev, "ERROR", [f"G-SUITE-PINNED: the throwaway repo's HEAD is "
                                      f"{det['throwaway_repo'].get('head')!r}, not {commit}"])
        return {"G-SUITE-PINNED": ev}
    fsha = tree_digest(tree_manifest(dest))
    det["full_tree_stack_taniteval_sha256"] = fsha
    if fsha != ctx.tree_sha256:
        reasons.append(f"G-SUITE-PINNED: the full archive's stack/+taniteval/ ({fsha[:16]}) is not "
                       f"the launched tree ({ctx.tree_sha256[:16]})")
    logs = Path(ctx.out_dir) / "logs" / "pinned"
    logs.mkdir(parents=True, exist_ok=True)
    env = pinned_env(dest, o, os.environ)
    probe = subprocess.run([sys.executable, "-c", "import tanitad; print(tanitad.__file__)"],
                           cwd=str(dest / "stack"), env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    det["tanitad_file"] = probe.stdout.strip()
    if not os.path.normcase(probe.stdout.strip()).startswith(os.path.normcase(str(dest))):
        finish_evidence(ev, "ERROR", [f"G-SUITE-PINNED: tanitad imports from "
                                      f"{probe.stdout.strip() or probe.stderr[-200:]}, not {dest}"])
        return {"G-SUITE-PINNED": ev}
    results: dict[str, dict] = {}
    for f in pin["files"]:
        key = f"{f['sub']}/{f['file']}"
        xml = logs / (key.replace("/", "__") + ".xml")
        log = logs / (key.replace("/", "__") + ".log")
        xml.unlink(missing_ok=True)
        t0 = time.time()
        try:
            with open(log, "w", encoding="utf-8", errors="replace") as fh:
                p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                    "-rfEs", f"--junitxml={xml}", "-o", "junit_family=xunit2",
                                    f["file"]], cwd=str(dest / f["sub"]), env=env, stdout=fh,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   timeout=1800)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        tail = ([ln for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()
                 if ln.strip()] or ["<none>"])[-1]
        r = {"rc": rc, "tail": tail[-300:], "elapsed_s": round(time.time() - t0, 1)}
        if xml.is_file():
            r.update(_parse_junit(xml))
        results[key] = r
        _say(f"G-SUITE-PINNED {key}: rc {rc}, {r.get('n_tests')} tests, "
             f"{len(r.get('failed') or [])} failed, {r.get('n_skipped')} skipped")
    fok, fdet = flaky_admitted(pin, dest)
    det["flaky"] = fdet
    r2, d2 = judge_pinned_run(pin["files"], results, ctx.prof.get("suite_pinned_skip_ok", ()),
                              fok)
    reasons += r2
    det.update(d2)
    # every guard's mutation arm must be RED (SPEC 2), on this clean full-commit tree
    audit = dest / "stack" / "scripts" / "guard_mutation_audit.py"
    if o.get("skip_mutation_audit"):
        reasons.append("the guard mutation audit was skipped (--skip-mutation-audit): not a PASS")
    elif not audit.is_file():
        reasons.append("stack/scripts/guard_mutation_audit.py is absent from the launch commit")
    else:
        log = logs / "guard_mutation_audit.log"
        try:
            with open(log, "w", encoding="utf-8", errors="replace") as fh:
                p = subprocess.run([sys.executable, str(audit)], cwd=str(dest / "stack"), env=env,
                                   stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   timeout=3600)
            arc = p.returncode
        except subprocess.TimeoutExpired:
            arc = "timeout"
        txt = log.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"=== (\d+)/(\d+) defects CAUGHT", txt)
        det["mutation_audit"] = {"rc": arc, "summary": m.group(0) if m else None,
                                 "verdicts": re.findall(r"^(CAUGHT|ESCAPED|MISCREDITED)\s+(\S+)",
                                                        txt, flags=re.M), "log": str(log)}
        if not m or arc != 0 or int(m.group(1)) != int(m.group(2)) or int(m.group(2)) < 1:
            reasons.append(f"guard mutation audit: {m.group(0) if m else 'no summary line'} "
                           f"(rc {arc}); every guard's mutation arm must be CAUGHT")
    ev["details"].update(det)
    finish_evidence(ev, "FAIL" if reasons else "PASS", reasons)
    return {"G-SUITE-PINNED": ev}


def job_record(ctx: Ctx, checks: list[str]) -> dict[str, dict]:
    """The launch prerequisites read from their PASS records (no model, no data)."""
    out: dict[str, dict] = {}
    if "G-MAP-OVERFIT" in checks:
        out.update(_job_map_overfit(ctx))
    if "G-BOX-OVERFIT" in checks:
        out.update(_job_box_overfit(ctx))
    return out


def _job_box_overfit(ctx: Ctx) -> dict[str, dict]:
    ev = new_evidence(ctx, "G-BOX-OVERFIT")
    rec = ctx.options.get("box_overfit_record")
    if arm_is(ctx, "box_overfit_missing"):
        rec = str(Path(ctx.out_dir) / "__gate_arm_missing_box_overfit_record__.json")
    reasons, det = judge_box_overfit(ctx.prof, rec, ctx.commit, argv_sha=ctx.argv_sha256,
                                     argv=list(ctx.argv))
    clo = ctx.options.get("box_overfit_closure")
    cr, cd = judge_closure("G-BOX-OVERFIT", clo, rec, ctx.tree,
                           (ctx.prof.get("box_overfit") or {}).get("harness"),
                           host_env=_host_env(), pmap=[tuple(x) for x in ctx.path_map],
                           git_dir=ctx.options.get("git_dir"), commit=ctx.commit)
    reasons += cr
    det["closure"] = cd
    for k, f in (("gate:box-overfit-record", rec), ("gate:box-overfit-closure", clo)):
        if f and Path(f).is_file():
            ev["inputs_read"][k] = fingerprint(f)
    ev["details"].update(det)
    ok_text = f"G-BOX-OVERFIT PASS {OVERFIT_SCOPE}"
    if (det.get("a19") or {}).get("applied"):
        ok_text += f"; {det['a19']['statement']}"         # SPEC_REFCV7 24: what it does NOT cover
    finish_evidence(ev, "FAIL" if reasons else "PASS", reasons or [ok_text])
    return {"G-BOX-OVERFIT": ev}


def _job_map_overfit(ctx: Ctx) -> dict[str, dict]:
    """G-MAP-OVERFIT: a launch prerequisite read from its PASS record (no model, no data)."""
    ev = new_evidence(ctx, "G-MAP-OVERFIT")
    rec = ctx.options.get("map_overfit_record")
    if arm_is(ctx, "map_overfit_missing"):
        rec = str(Path(ctx.out_dir) / "__gate_arm_missing_overfit_record__.json")
    cw_flag = (ctx.prof.get("map_hires") or {}).get("class_weights_flag")
    cwv = flag_values(ctx.argv, cw_flag) if cw_flag else None
    cw_local = map_path(cwv[0], [tuple(x) for x in ctx.path_map]) if cwv else None
    cw_sha = sha256_file(cw_local) if cw_local and Path(cw_local).is_file() else None
    reasons, det = judge_map_overfit(ctx.prof, rec, ctx.commit, argv_sha=ctx.argv_sha256,
                                     class_weights_sha256=cw_sha)
    if cwv and cw_sha is None:
        reasons.append(f"G-MAP-OVERFIT: the launch class-weight file {cwv[0]} is not readable "
                       f"here, so the record's weights cannot be matched to it")
    clo = ctx.options.get("map_overfit_closure")
    cr, cd = judge_closure("G-MAP-OVERFIT", clo, rec, ctx.tree,
                           (ctx.prof.get("map_overfit") or {}).get("harness"),
                           host_env=_host_env(), pmap=[tuple(x) for x in ctx.path_map],
                           must_have_data_sha=[cw_sha] if cw_sha else [],
                           git_dir=ctx.options.get("git_dir"), commit=ctx.commit)
    reasons += cr
    det["closure"] = cd
    for k, f in (("gate:map-overfit-record", rec), ("gate:map-overfit-closure", clo)):
        if f and Path(f).is_file():
            ev["inputs_read"][k] = fingerprint(f)
    ev["details"].update(det)
    ok_text = f"G-MAP-OVERFIT PASS {OVERFIT_SCOPE}"
    if (det.get("a19") or {}).get("applied"):
        ok_text += f"; {det['a19']['statement']}"         # SPEC_REFCV7 24: what it does NOT cover
    finish_evidence(ev, "FAIL" if reasons else "PASS", reasons or [ok_text])
    return {"G-MAP-OVERFIT": ev}


JOB_FUNCS = {"model": job_model, "smoke": job_smoke, "clock": job_clock, "suite": job_suite,
             "pinned": job_suite_pinned, "record": job_record}


def cmd_check(a) -> int:
    """(child) run ONE job's checks in THIS process and write their evidence."""
    _harden_streams()
    ctx = Ctx.load(a.ctx)
    checks = [c for c in a.checks.split(",") if c]
    job = next(j for j, cs in JOBS if set(checks) <= set(cs))
    evs: dict[str, dict] = {}
    try:
        info = child_bootstrap(ctx)
        _say(f"{job}: tanitad from {info['tanitad']} (torch {info['torch']}, cuda {info['cuda']})")
        evs = JOB_FUNCS[job](ctx, checks)
        for ev in evs.values():
            ev.setdefault("details", {})["child"] = info
    except BaseException as e:                            # noqa: BLE001 -- never a silent pass
        tb = traceback.format_exc()[-3000:]
        for c in checks:
            if c not in evs or evs[c].get("status") not in ("PASS", "FAIL"):
                ev = evs.get(c) or new_evidence(ctx, c)
                ev["details"]["traceback"] = tb
                finish_evidence(ev, "ERROR", [f"{type(e).__name__}: {e}"])
                evs[c] = ev
    for c in checks:
        ev = evs.get(c)
        if ev is None:
            ev = finish_evidence(new_evidence(ctx, c), "ERROR", ["the job returned no evidence"])
        if "finished_utc" not in ev:
            finish_evidence(ev, ev.get("status", "ERROR"), ev.get("reasons"))
        write_evidence(ctx.out_dir, ev)
        _say(f"{c}: {ev['status']} ({len(ev.get('reasons') or [])} reason(s))")
    return 0


# --------------------------------------------------------------------------------------------- #
# orchestration (parent side)                                                                     #
# --------------------------------------------------------------------------------------------- #
def _child_env(ctx: Ctx) -> dict:
    e = dict(os.environ)
    tree = Path(ctx.tree).resolve()
    # ⛔ WINDOWS-style paths from Python itself (an MSYS `/c/...` entry is silently dropped and
    # the editable install then serves tanitad from another disk -- a green run on the wrong tree)
    e["PYTHONPATH"] = os.pathsep.join([str(tree / "stack"), str(tree / "taniteval")])
    omp = str(ctx.options.get("omp", 4))
    e.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=omp,
             MKL_NUM_THREADS=omp, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if ctx.options.get("cpu_only"):
        e["CUDA_VISIBLE_DEVICES"] = "-1"      # "-1", never "": an empty value DELETES it on Windows
    if ctx.arm:
        e[ARM_ENV] = ctx.arm
    else:
        e.pop(ARM_ENV, None)
    return e


def _wait_ram(min_gb: float, max_wait_s: int, samples: int = 3,
              gap_s: float = 20.0) -> tuple[bool, float | None]:
    """Start only after `samples` CONSECUTIVE readings at or above `min_gb` (the Master Mind's
    dev-box rule, 2026-09-27: "start at >= 7.5 GB free on 3 samples") -- one lucky reading
    between two busy ones is not headroom."""
    t0, run, av = time.time(), 0, None
    while True:
        av = ram_available_gb()
        if av is None:
            return True, av
        run = run + 1 if av >= min_gb else 0
        if run >= max(1, int(samples)):
            return True, av
        if time.time() - t0 > max_wait_s:
            return False, av
        time.sleep(gap_s if run else 60)


def _kill_tree(pid: int) -> None:
    """By EXPLICIT pid, with its children (never a name match). Best effort: the caller VERIFIES
    (`_stop_child`)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                           stdin=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(pid), 9)
    except Exception:                                     # noqa: BLE001 -- verified by the caller
        pass


def _stop_child(p: "subprocess.Popen", wait_s: float = 30.0) -> dict:
    """Stop a child started by `run_job` and PROVE it: the tree kill, then TerminateProcess /
    SIGKILL on the direct child if it is still there. MEASURED 2026-09-27: a taskkill under memory
    pressure did nothing, the child ran on for 8 minutes and overwrote the abort's ERROR."""
    out = {"pid": p.pid, "steps": []}
    for step in ("tree", "direct"):
        if step == "tree":
            _kill_tree(p.pid)
        else:
            try:
                p.kill()
            except Exception as e:                        # noqa: BLE001
                out["kill_error"] = f"{type(e).__name__}: {e}"
        try:
            p.wait(timeout=wait_s)
            out["steps"].append(f"{step}: exited rc {p.returncode}")
            out["stopped"] = True
            return out
        except subprocess.TimeoutExpired:
            out["steps"].append(f"{step}: STILL RUNNING after {wait_s:.0f} s")
    out["stopped"] = False
    return out


def run_job(ctx: Ctx, ctx_path: Path, job: str, checks: list[str], python: str) -> None:
    out = Path(ctx.out_dir)
    log = out / "logs" / f"job_{job}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    min_gb = float(ctx.options.get("min_free_gb") or 0)
    started = utc_now()
    if min_gb > 0:
        ok, av = _wait_ram(min_gb, int(ctx.options.get("ram_wait_s") or 3600))
        if not ok:
            for c in checks:
                write_evidence(out, finish_evidence(new_evidence(ctx, c), "ERROR", [
                    f"RAM floor: available {av:.2f} GB < {min_gb} GB for the whole wait -- the "
                    f"check did NOT run (the dev box is the PI's desktop)"]))
            return
    timeout = int(ctx.prof["check_timeout_s"].get(job, 3600))
    cmd = [python, str(Path(__file__).resolve()), "check", "--ctx", str(ctx_path),
           "--checks", ",".join(checks)]
    _say(f"job {job} ({', '.join(checks)}) -> {log}")
    kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
          else {"start_new_session": True})
    abort_gb = float(ctx.options.get("abort_free_gb") or 0)
    aborted = None
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n==== {started} {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.Popen(cmd, env=_child_env(ctx), stdout=fh, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, **kw)
        t_end = time.time() + timeout
        rc = None
        while rc is None:
            try:
                rc = p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if time.time() > t_end:
                    stop = _stop_child(p)
                    rc = "timeout" if stop["stopped"] else "timeout-and-still-running"
                    break
                av = ram_available_gb() if abort_gb > 0 else None
                if av is not None and av < abort_gb:
                    # the Master Mind's rule: abort below the floor (the dev box is the PI's
                    # desktop) -- by the child's EXPLICIT pid, never a name, and VERIFIED
                    aborted, rc = (av, _stop_child(p)), "aborted-ram"
                    break
    if aborted is not None:
        av, stop = aborted
        what = ("ABORTED, the check did NOT complete" if stop["stopped"] else
                f"ABORT FAILED: the child (pid {stop['pid']}) is STILL RUNNING after "
                f"{'; '.join(stop['steps'])} -- any evidence it writes later is NOT this job's "
                f"verdict; stop it by that pid")
        for c in checks:
            ev = new_evidence(ctx, c)
            ev["details"]["abort"] = stop
            write_evidence(out, finish_evidence(ev, "ERROR", [
                f"RAM floor: available {av:.2f} GB < {abort_gb} GB while the {job} job ran -- "
                f"{what}"]))
        return
    # ⛔ the verdict is the evidence FILE: a missing or stale one is an ERROR, never a pass
    for c in checks:
        ep = out / "evidence" / f"{c}.json"
        ok = False
        if ep.is_file():
            try:
                ev = read_json(ep)
                ok = str(ev.get("finished_utc", "")) >= started and ev.get("binding") == ctx.binding()
            except Exception:                             # noqa: BLE001
                ok = False
        if not ok:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            write_evidence(out, finish_evidence(new_evidence(ctx, c), "ERROR", [
                f"the {job} child (rc {rc}) left no fresh evidence for {c}"],))
            _say(f"{c}: ERROR -- no fresh evidence (rc {rc}); log tail: {tail[-3:]}")


def _collect_evidence(out_dir: Path, import_dirs: Iterable[str]) -> dict[str, list[tuple[Path, dict]]]:
    found: dict[str, list[tuple[Path, dict]]] = {}
    dirs = [out_dir / "evidence"] + [Path(d) for d in import_dirs]
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("G-*.json")):
            try:
                ev = read_json(p)
            except Exception:                             # noqa: BLE001
                continue
            if ev.get("kind") == "evidence" and ev.get("check") in CHECKS:
                found.setdefault(ev["check"], []).append((p, ev))
    return found


def finalize(ctx: Ctx, key: bytes, import_dirs: Iterable[str] = (),
             profile_refusals: Iterable[str] = ()) -> tuple[str, Path, dict]:
    out = Path(ctx.out_dir)
    pmap = [tuple(x) for x in ctx.path_map]
    gin = dict(ctx.options.get("gate_inputs") or {})
    data = data_manifest(ctx.argv, ctx.prof, pmap, gin)
    binding = {**ctx.binding(), "argv": list(ctx.argv), "trainer": ctx.prof["trainer"],
               "gate_inputs": gin, "data": data,
               "data_sha256": sha256_bytes(canonical_json(data))}
    # ⛔ the profile's argv rules are RE-DERIVED here from the context -- never trusted from the
    # caller -- so a stand-alone `finalize` cannot mint a token `run` would have refused
    own_refusals, pi_pending = profile_argv_rules(ctx.prof, ctx.argv, ctx.options, pmap)
    reasons = list(dict.fromkeys([*profile_refusals, *own_refusals]))
    reasons += data_manifest_problems(data)
    if ctx.arm or os.environ.get(ARM_ENV):
        reasons.append(f"a regression arm is active ({ctx.arm or os.environ.get(ARM_ENV)}) -- "
                       f"an arm run can never PASS")
    found = _collect_evidence(out, import_dirs)
    checks: dict[str, dict] = {}
    verdict_missing = []
    required = required_checks_of(ctx.prof, ctx.argv)
    for c in required:
        cands = []
        for p, ev in found.get(c, []):
            why = []
            if ev.get("binding") != ctx.binding():
                why.append("binding differs")
            if ev.get("arm"):
                why.append(f"arm {ev['arm']}")
            for f, fp in (ev.get("inputs_read") or {}).items():
                mine = data.get(f)
                if mine is None or fp_key(mine) != fp_key(fp):
                    why.append(f"input {f} read with a different fingerprint")
            cands.append((str(ev.get("finished_utc", "")), p, ev, why))
        usable = sorted([x for x in cands if not x[3]], key=lambda x: x[0])
        if not usable:
            verdict_missing.append(c)
            checks[c] = {"status": "MISSING",
                         "rejected": [{"path": str(p), "why": w} for _, p, _, w in cands][:6]}
            continue
        _, p, ev, _ = usable[-1]
        checks[c] = {"status": ev["status"], "evidence_path": str(p),
                     "evidence_sha256": sha256_file(p), "host": ev.get("host", {}).get("node"),
                     "finished_utc": ev.get("finished_utc"),
                     "reasons": list(ev.get("reasons") or [])[:8]}
        if ev["status"] not in ("PASS", "PI-DECISION"):
            reasons.append(f"{c} {ev['status']}: " + " | ".join((ev.get("reasons") or [])[:3]))
    needs_pi = [c for c, v in checks.items() if v["status"] == "PI-DECISION"]
    if any(v["status"] in ("FAIL", "ERROR") for v in checks.values()) or reasons:
        verdict = "FAIL"
    elif verdict_missing:
        verdict = "INCOMPLETE"
        reasons.append(f"no usable evidence yet for: {verdict_missing}")
    elif needs_pi or pi_pending:
        # ⛔ a PI decision is not a pass: the SPEC routes it to the PI BEFORE launch
        verdict = "PI-DECISION"
        reasons += [f"{c}: " + " | ".join(checks[c]["reasons"][:2]) for c in needs_pi]
        reasons += [f"PI decision pending: {r}" for r in pi_pending]
    else:
        verdict = "PASS"
    # ⛔ a profile with OPEN ITEMS is not a finished launch definition: never a PASS
    open_items = open_item_reasons(ctx.prof)
    if open_items:
        reasons += open_items
        if verdict in ("PASS", "PI-DECISION"):
            verdict = "INCOMPLETE"
    # the binding is NEVER scrubbed (it must be the exact launch); the prose is (clip-id rule)
    token = {"schema": GATE_SCHEMA, "kind": "launch-gate-token", "verdict": verdict,
             "binding": binding, "checks": scrub(checks), "reasons": scrub(reasons),
             "gate_sha256": _gate_self_sha(), "host": _host(), "created_utc": utc_now(),
             "key_id": key_id(key)}
    token["hmac_sha256"] = sign(token, key)
    c12 = ctx.commit[:12]
    path = out / f"{verdict}_{c12}.json"
    stamp = utc_now().replace(":", "")
    sup = out / "superseded"
    for other in ("PASS", "FAIL", "INCOMPLETE", "PI-DECISION"):
        op = out / f"{other}_{c12}.json"
        if op.exists():
            # ⛔ exactly ONE current token per commit; a stale PASS is DEMOTED, never left behind
            sup.mkdir(parents=True, exist_ok=True)
            os.replace(op, sup / f"{other}_{c12}.{stamp}.json")
    write_json(path, token)
    return verdict, path, token


def cmd_run(a) -> int:
    _harden_streams()
    profile = PROFILES[a.profile]
    tree = Path(a.tree).resolve()
    argv = load_argv_file(a.argv_file)
    commit = a.commit
    if a.git_dir and not _HEX40.match(commit or ""):
        commit = resolve_commit(a.git_dir, commit)
    if not _HEX40.match(commit or ""):
        raise GateError(f"--commit must be a 40-char sha (or resolvable with --git-dir): {commit!r}")
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    arm = a.arm or os.environ.get(ARM_ENV) or None
    key = load_key(a.key_file, create=True, forbid_under=[tree])
    _say(f"profile {a.profile} | commit {commit[:12]} | tree {tree}")
    tman = tree_manifest(tree)
    tsha = tree_digest(tman)
    write_json(out / "tree_manifest.json", {"tree": str(tree), "tree_sha256": tsha, "files": tman})
    if a.git_dir:
        want = tree_manifest_from_archive(git_archive_bytes(a.git_dir, commit, TREE_ROOTS))
        if tree_digest(want) != tsha:
            raise GateError(f"the tree {tree} is NOT commit {commit[:12]}: "
                            f"{json.dumps(tree_diff(want, tman))[:800]}")
        _say(f"tree == git archive {commit[:12]} ({len(tman)} files, {tsha[:16]})")
    options = {
        "cpu_only": bool(a.cpu_only), "omp": int(a.omp), "min_free_gb": float(a.min_free_gb),
        "ram_wait_s": int(a.ram_wait_s), "git_dir": a.git_dir, "baseline": a.baseline,
        "suites": a.suites.split(",") if a.suites else None,
        "suite_work": a.suite_work, "skip_mutation_audit": bool(a.skip_mutation_audit),
        "eval_loader": a.eval_loader or str(_resolve_repo_rel(tree, profile["eval_loader"])),
        "eval_kit": a.eval_kit, "eval_stamps": a.eval_stamps,
        "eval_remap_overrides": a.eval_remap_overrides,
        "clock_reference": a.clock_reference or [str(_resolve_repo_rel(tree, r))
                                                 for r in profile["clock_reference"]],
        "clock_manifest": dict(x.split("=", 1) for x in (a.clock_manifest or [])),
        "smoke_steps": a.smoke_steps,
        "pi_cost_approval": a.pi_cost_approval,
        "nav_tau_record": str(Path(a.nav_tau_record).resolve()) if a.nav_tau_record else None,
        "map_overfit_record": (str(Path(a.map_overfit_record).resolve())
                               if a.map_overfit_record else None),
        "map_watch_artifact": (str(Path(a.map_watch_artifact).resolve())
                               if a.map_watch_artifact else None),
        "box_overfit_record": (str(Path(a.box_overfit_record).resolve())
                               if a.box_overfit_record else None),
        "map_overfit_closure": (str(Path(a.map_overfit_closure).resolve())
                                if a.map_overfit_closure else None),
        "box_overfit_closure": (str(Path(a.box_overfit_closure).resolve())
                                if a.box_overfit_closure else None),
        "suite_verdict": (str(Path(a.suite_verdict).resolve()) if a.suite_verdict else None),
        "abort_free_gb": float(a.abort_free_gb),
    }
    if options["nav_tau_record"] is None and profile.get("nav_tau_record_default"):
        dflt = _resolve_repo_rel(tree, profile["nav_tau_record_default"])
        if dflt.is_file():
            options["nav_tau_record"] = str(dflt)
    # the gate's own inputs are content-bound in the token like the trainer's (the tau record's
    # sha256 is therefore IN the PASS token: binding.data["gate:nav-tau-record"].sha256)
    gin = {f"gate:clock-reference[{i}]": str(r)
           for i, r in enumerate(options["clock_reference"])}
    if options["nav_tau_record"]:
        gin["gate:nav-tau-record"] = options["nav_tau_record"]
    req = required_checks_of(profile, argv)
    if options["map_overfit_record"] and "G-MAP-OVERFIT" in req:
        gin["gate:map-overfit-record"] = options["map_overfit_record"]
    if options["box_overfit_record"] and "G-BOX-OVERFIT" in req:
        gin["gate:box-overfit-record"] = options["box_overfit_record"]
    if options["map_overfit_closure"] and "G-MAP-OVERFIT" in req:
        gin["gate:map-overfit-closure"] = options["map_overfit_closure"]
    if options["box_overfit_closure"] and "G-BOX-OVERFIT" in req:
        gin["gate:box-overfit-closure"] = options["box_overfit_closure"]
    if options["map_watch_artifact"] and hires_on(profile, argv):
        gin["gate:map-watch-artifact"] = options["map_watch_artifact"]
    if options["suite_verdict"] and profile.get("suite_mode") == "consume":
        gin["gate:suite-verdict"] = options["suite_verdict"]
    options["gate_inputs"] = gin
    pmap = parse_path_map(a.path_map)
    # ⛔ a map entry that matches NO argv path is a mangled map, not a no-op (MEASURED 2026-09-26:
    # MSYS rewrote `/home/nvidia/data=D:/kit` into `C:\Program Files\Git\home\...` and every
    # data path silently stayed unmapped)
    dpaths = [p for _, p in data_inputs(argv, profile)]
    for frm, _to in pmap:
        if not any(p == frm or p.startswith(frm + "/") or p.startswith(frm + "\\")
                   for p in dpaths):
            raise GateError(f"--path-map FROM {frm!r} matches no data path of the argv -- a "
                            f"mangled map (MSYS path conversion? set MSYS_NO_PATHCONV=1)")
    ctx = Ctx(profile=a.profile, tree=str(tree), commit=commit, argv=argv, out_dir=str(out),
              path_map=[list(x) for x in pmap], tree_sha256=tsha,
              argv_sha256=argv_sha256(argv), options=options, arm=arm)
    ctx_path = ctx.dump(out / "ctx.json")
    refusals, pi_pending = profile_argv_rules(profile, argv, options, pmap)
    for r in refusals:
        _say(f"REFUSAL: {r}")
    for r in pi_pending:
        _say(f"PI DECISION PENDING: {r}")
    wanted = (a.checks.split(",") if a.checks else list(profile["stages"][a.stage]))
    req = required_checks_of(profile, argv)
    # a record check the launch does not require is not run (e.g. G-MAP-OVERFIT with the 10 cm
    # head off on a generic refc argv); a REQUIRED one always runs, and FAILS without its record
    wanted = [c for c in wanted if c in req or c in BASE_CHECKS]
    bad = sorted(set(wanted) - set(CHECKS))
    if bad:
        raise GateError(f"unknown checks {bad}")
    for job, cs in JOBS:
        sel = [c for c in cs if c in wanted]
        if sel:
            run_job(ctx, ctx_path, job, sel, a.python or sys.executable)
    verdict, path, tok = finalize(ctx, key, a.import_evidence or [], refusals)
    for c, v in tok["checks"].items():
        _say(f"  {c:8s} {v['status']}")
    for r in tok["reasons"][:12]:
        _say(f"  - {r[:300]}")
    _say(f"verdict {verdict} -> {path}")
    print(f"ZZGATE-{verdict}-{commit[:12]}ZZ", flush=True)
    return {"PASS": 0, "INCOMPLETE": 2, "PI-DECISION": 4}.get(verdict, 1)


def _resolve_repo_rel(tree: Path, rel: str) -> Path:
    """A repo-relative path: in the tree if it ships there, else in the enclosing repo."""
    for base in (tree, tree.parent, *tree.parents):
        p = base / rel
        if p.exists():
            return p
    return tree / rel


def cmd_finalize(a) -> int:
    _harden_streams()
    ctx = Ctx.load(Path(a.out_dir) / "ctx.json")
    key = load_key(a.key_file, create=True, forbid_under=[ctx.tree])
    verdict, path, _ = finalize(ctx, key, a.import_evidence or [])
    _say(f"verdict {verdict} -> {path}")
    print(f"ZZGATE-{verdict}-{ctx.commit[:12]}ZZ", flush=True)
    return 0 if verdict == "PASS" else 1


# --------------------------------------------------------------------------------------------- #
# verify / exec (the supervisor's side)                                                           #
# --------------------------------------------------------------------------------------------- #
def verify_token(token_path: str | os.PathLike, argv: list[str], tree: str | os.PathLike, *,
                 key: bytes, commit: str | None = None,
                 path_map: list[tuple[str, str]] | None = None) -> tuple[bool, list[str], dict]:
    """Every field the token binds is RE-MEASURED here; any difference refuses."""
    reasons: list[str] = []
    info: dict[str, Any] = {"token": str(token_path)}
    try:
        tok = read_json(token_path)
    except Exception as e:                                # noqa: BLE001
        return False, [f"token unreadable: {type(e).__name__}: {e}"], info
    if tok.get("kind") != "launch-gate-token" or tok.get("schema") != GATE_SCHEMA:
        reasons.append("not a launch-gate token")
    if not signature_ok(tok, key):
        reasons.append("HMAC does not verify -- the token was edited, or signed with another key "
                       f"(token key_id {tok.get('key_id')}, this host's {key_id(key)})")
    if tok.get("verdict") != "PASS":
        reasons.append(f"token verdict is {tok.get('verdict')!r}, not PASS")
    b = tok.get("binding") or {}
    info["binding"] = {k: b.get(k) for k in ("profile", "commit", "tree_sha256", "argv_sha256")}
    if commit and b.get("commit") != commit:
        reasons.append(f"token commit {str(b.get('commit'))[:12]} != launch commit {commit[:12]}")
    a_sha = argv_sha256(argv)
    info["argv_sha256"] = a_sha
    if b.get("argv_sha256") != a_sha or b.get("argv") != argv:
        changed = [f"{x} -> {y}" for x, y in zip(b.get("argv") or [], argv) if x != y][:5]
        reasons.append(f"argv differs from the gated argv (sha {a_sha[:12]} vs "
                       f"{str(b.get('argv_sha256'))[:12]}; {changed or 'length differs'})")
    try:
        tman = tree_manifest(tree)
        tsha = tree_digest(tman)
        info["tree_sha256"] = tsha
        if tsha != b.get("tree_sha256"):
            reasons.append(f"code tree differs from the gated tree ({tsha[:16]} vs "
                           f"{str(b.get('tree_sha256'))[:16]})")
    except Exception as e:                                # noqa: BLE001
        reasons.append(f"cannot measure the code tree: {e}")
    prof = PROFILES.get(str(b.get("profile")), _REFC)
    data = data_manifest(argv, prof, path_map or [], b.get("gate_inputs") or {})
    want = b.get("data") or {}
    d_bad = []
    for k in sorted(set(want) | set(data)):
        if k not in want or k not in data or fp_key(want[k]) != fp_key(data[k]):
            d_bad.append(k)
    info["data_differs"] = d_bad
    if d_bad:
        reasons.append(f"data differs from the gated data on {d_bad[:8]}")
    return not reasons, reasons, info


def _emit_verify(a, ok: bool, reasons: list[str], info: dict, action: str) -> None:
    rec = {"schema": GATE_SCHEMA, "kind": f"launch-gate-{action}", "verdict": "MATCH" if ok
           else "REFUSED", "reasons": reasons, "info": info, "utc": utc_now(),
           "host": platform.node()}
    if a.json:
        write_json(a.json, rec)
    for r in reasons:
        _say(f"REFUSED: {r}")
    print(f"ZZGATE-{'MATCH' if ok else 'REFUSED'}ZZ", flush=True)


def cmd_verify(a) -> int:
    _harden_streams()
    try:
        key = load_key(a.key_file, create=False, forbid_under=[a.tree])
        argv = load_argv_file(a.argv_file)
        ok, reasons, info = verify_token(a.token, argv, a.tree, key=key, commit=a.commit,
                                         path_map=parse_path_map(a.path_map))
    except Exception as e:                                # noqa: BLE001 -- refuse, never pass
        ok, reasons, info = False, [f"{type(e).__name__}: {e}"], {}
    _emit_verify(a, ok, reasons, info, "verify")
    return 0 if ok else 5


def cmd_exec(a) -> int:
    """Verify, then REPLACE this process with the trainer on exactly the bound argv. The PID the
    supervisor recorded becomes the trainer's (the `exec` idiom of run_refcv6.sh)."""
    _harden_streams()
    try:
        key = load_key(a.key_file, create=False, forbid_under=[a.tree])
        argv = load_argv_file(a.argv_file)
        ok, reasons, info = verify_token(a.token, argv, a.tree, key=key, commit=a.commit)
    except Exception as e:                                # noqa: BLE001
        ok, reasons, info, argv = False, [f"{type(e).__name__}: {e}"], {}, []
    _emit_verify(a, ok, reasons, info, "exec")
    if not ok:
        return 5
    tree = Path(a.tree).resolve()
    tok = read_json(a.token)
    trainer = tree / tok["binding"].get("trainer", _REFC["trainer"])
    os.environ["PYTHONPATH"] = os.pathsep.join([str(tree / "stack"), str(tree / "taniteval")])
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.pop(ARM_ENV, None)
    sys.stdout.flush()
    sys.stderr.flush()
    cmd = [sys.executable, str(trainer), *argv]
    if a.dry_run:
        print("ZZGATE-EXEC-DRYRUNZZ " + json.dumps(cmd), flush=True)
        return 0
    if os.name == "nt":                                   # no true exec on Windows: run and relay
        return subprocess.call(cmd)
    os.execv(sys.executable, cmd)
    return 99                                             # pragma: no cover (unreachable)


def cmd_bind(a) -> int:
    _harden_streams()
    argv = load_argv_file(a.argv_file)
    prof = PROFILES[a.profile]
    tman = tree_manifest(a.tree)
    out = {"profile": a.profile, "argv_sha256": argv_sha256(argv), "tree_sha256": tree_digest(tman),
           "n_tree_files": len(tman),
           "data": data_manifest(argv, prof, parse_path_map(a.path_map))}
    print(json.dumps(out, indent=1))
    return 0


# --------------------------------------------------------------------------------------------- #
# CLI                                                                                             #
# --------------------------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the checks of a stage and write the token")
    r.add_argument("--profile", required=True, choices=sorted(PROFILES))
    r.add_argument("--tree", required=True, help="the code tree being launched (stack/ taniteval/)")
    r.add_argument("--commit", required=True, help="40-char sha (or a ref with --git-dir)")
    r.add_argument("--argv-file", required=True, help="JSON list: the trainer argv to launch")
    r.add_argument("--out-dir", required=True)
    g = r.add_mutually_exclusive_group()
    g.add_argument("--stage", choices=("devbox", "thor", "all"), default="all")
    g.add_argument("--checks", default=None, help="comma list, overrides --stage")
    r.add_argument("--path-map", action="append", default=[], help="FROM=TO for data paths")
    r.add_argument("--import-evidence", action="append", default=[],
                   help="an evidence dir from another host (bound checks only)")
    r.add_argument("--git-dir", default=None)
    r.add_argument("--baseline", default=None, help="G-SUITE baseline ref (the tip)")
    r.add_argument("--suites", default=None, help="comma list of suite dirs (default profile)")
    r.add_argument("--suite-work", default=None, help="short path for the clean trees (MAX_PATH)")
    r.add_argument("--skip-mutation-audit", action="store_true", help="(never a PASS)")
    r.add_argument("--eval-loader", default=None)
    r.add_argument("--eval-kit", default=None)
    r.add_argument("--eval-stamps", default=None, help="a config.json with the train-split stamps")
    r.add_argument("--eval-remap-overrides", default=None,
                   help="JSON {flag: local path} handed to the eval loader (REFCV6_REMAP_OVERRIDES)")
    r.add_argument("--clock-reference", action="append", default=None)
    r.add_argument("--clock-manifest", action="append", default=None,
                   help="train=<_v2manifest.pt> (dev box: manifest mode, not launch evidence)")
    r.add_argument("--smoke-steps", type=int, default=None)
    r.add_argument("--map-overfit-record", default=None,
                   help="G-MAP-OVERFIT: the overfit protocol's PASS record for THIS commit")
    r.add_argument("--nav-tau-record", default=None,
                   help="FIX-4: the banked tau record (default: the profile's repo path, "
                        "`…/2026-09-26-declared-vs-built/raw/nav_compliance_tau_train.json`) "
                        "that justifies --nav-compliance-tau-rad; its sha256 is bound in the "
                        "token (REQUIRED when --graft-nav-compliance)")
    r.add_argument("--map-watch-artifact", default=None,
                   help="G-MAP item 4: the refcv7 Training Watch built from the smoke's "
                        "metrics.jsonl (24 per-class IoU series + the thin-class alarm tile)")
    r.add_argument("--box-overfit-record", default=None,
                   help="G-BOX-OVERFIT: the box overfit harness's PASS JSON")
    r.add_argument("--map-overfit-closure", default=None,
                   help="G-MAP-OVERFIT: the closure_run.py record of the SAME harness run")
    r.add_argument("--box-overfit-closure", default=None,
                   help="G-BOX-OVERFIT: the closure_run.py record of the SAME harness run")
    r.add_argument("--suite-verdict", default=None,
                   help="G-SUITE (consume mode): the Thor full-suite verdict BOUND to the launch "
                        "commit and tree by `launch_gate.py suite-bind`")
    r.add_argument("--pi-cost-approval", default=None,
                   help="JSON {max_s_per_step, pi_verbatim, date}: the PI's recorded decision "
                        "that lifts PI-DECISION for a measured cost at or under max_s_per_step")
    r.add_argument("--cpu-only", action="store_true")
    r.add_argument("--omp", type=int, default=4)
    r.add_argument("--min-free-gb", type=float, default=0.0,
                   help="start a job only after 3 consecutive readings at or above this")
    r.add_argument("--abort-free-gb", type=float, default=0.0,
                   help="kill a running job (explicit pid) when free RAM falls below this")
    r.add_argument("--ram-wait-s", type=int, default=3600)
    r.add_argument("--key-file", default=None)
    r.add_argument("--python", default=None)
    r.add_argument("--arm", default=None, help="a regression arm (never PASSes)")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("check", help="(internal) run one job's checks in this process")
    c.add_argument("--ctx", required=True)
    c.add_argument("--checks", required=True)
    c.set_defaults(fn=cmd_check)
    f = sub.add_parser("finalize", help="re-derive the token from the evidence on disk")
    f.add_argument("--out-dir", required=True)
    f.add_argument("--import-evidence", action="append", default=[])
    f.add_argument("--key-file", default=None)
    f.set_defaults(fn=cmd_finalize)
    for name, fn in (("verify", cmd_verify), ("exec", cmd_exec)):
        v = sub.add_parser(name)
        v.add_argument("--token", required=True)
        v.add_argument("--argv-file", required=True)
        v.add_argument("--tree", required=True)
        v.add_argument("--commit", default=None)
        v.add_argument("--key-file", default=None)
        v.add_argument("--json", default=None, help="write the verdict here (the artifact)")
        v.add_argument("--path-map", action="append", default=[])
        if name == "exec":
            v.add_argument("--dry-run", action="store_true")
        v.set_defaults(fn=fn)
    b = sub.add_parser("bind", help="print the binding (tree, argv, data) for inspection")
    b.add_argument("--profile", required=True, choices=sorted(PROFILES))
    b.add_argument("--tree", required=True)
    b.add_argument("--argv-file", required=True)
    b.add_argument("--path-map", action="append", default=[])
    b.set_defaults(fn=cmd_bind)
    sb = sub.add_parser("suite-bind", help="bind a Thor full-suite verdict.json to the commit "
                                           "and the CAND tree it was measured on")
    sb.add_argument("--verdict", required=True, help="gate_verdict.py's verdict.json")
    sb.add_argument("--cand-tree", required=True, help="the CAND tree the suite ran in")
    sb.add_argument("--commit", required=True, help="the 40-char launch commit")
    sb.add_argument("--out", required=True)
    sb.set_defaults(fn=cmd_suite_bind)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return int(a.fn(a) or 0)
    except GateError as e:
        _say(f"GATE ERROR: {e}")
        print("ZZGATE-ERRORZZ", flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
