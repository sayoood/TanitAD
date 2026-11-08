#!/usr/bin/env python3
"""r9_e1_make_argv.py -- the refcv9 E1 perception-rig launch argv (C0 seed 0, C1 seed 1), derived from refcv8's launch argv.

``build(base_argv, seed, out, ...)`` removes every PLANNER / TACTICAL / refcv8-allocation flag (and its values) from the
refcv8 launch argv, keeps every perception / data / trunk flag, and appends the rig flags. The result is checked against the
trainer's own pins: the trainer's argparse parses it and ``refcv9_perc_rig.pin_args`` accepts it. A flag in the base argv
that is NEITHER kept NOR dropped is an error -- a new refcv8 flag cannot slip through silently.

  python r9_e1_make_argv.py --base <refcv8_launch_argv_base.json> --seed 0 --out <run dir> --write <argv.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

#: flags removed with all their values. Every planner / tactical / refcv8-allocation lever the rig cannot honour, plus the
#: flags whose only consumer is one of those modules.
DROPPED = (
    "--arm",                         # re-added below: the rig builds core only; hier/flat changes nothing the rig builds
    "--ego-history", "--no-strategic", "--sampler",
    "--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal", "--f5-emitting-conf",
    "--f6-w-u0-zero", "--f9-assert-vocab",
    "--anchors", "--anchor-v0-conditioned", "--n-anchors",
    "--v7-labels", "--nav-from-v7", "--eval-labels",
    "--tac-decoder-v6", "--w-tac-v6", "--tac-decoder-d-bev", "--graft-behaviour-sel", "--max-speed-input-v6",
    "--speed-max-sidecar-v6", "--speed-max-sidecar-v6-eval", "--graft-tac8-prior", "--graft-nav-compliance",
    "--nav-compliance-tau-rad", "--nav-compliance-tau-file", "--speed-ceiling-filter",
    "--bev-coupling", "--residual-prior",
    "--refcv8", "--w-r8-cons", "--r8-n-alloc", "--r8-alloc-emit", "--r8-alloc-emit-start", "--w-r8-alloc-l1",
    "--w-r8-listwise", "--w-r8-sat", "--w-r8-subscore", "--r8-lat-prior-dropout", "--r8-v9-labels",
    "--r8-v9-labels-eval", "--r8-v9-md5", "--r8-v9-eval-md5", "--r8-v9-lat-variant", "--r8-nav-from-v9", "--r8-no-rc",
    "--r8-nav-args-dropout", "--join-defect-masks",   # the latter is a PERCEPTION flag: re-added below, re-rooted (JOIN_MASKS_REL)
    "--r8-speed-input", "--r8-speed-unknown-p", "--w-r8-v9-cons",
    "--grad-share-every", "--grad-share-terms", "--grad-share-at",
    "--b1-guard-ref-ap2m", "--b1-guard-step",
    "--conflict-every",              # the conflict detector is OFF (below): no g_traj exists in the rig
    "--init-from", "--ema-decay", "--lr", "--warmup", "--seed", "--steps", "--out", "--batch",   # re-added with the rig values
    "--trunk-dedup-frames",          # the rig reads ONE (the newest) window row
    "--eval-every", "--eval-batches", "--save-every", "--log-every",    # re-added
    "--opt",
)

#: flags that survive unchanged (a documentation of the decision; the check below is "kept or dropped, never neither").
KEPT = (
    "--size", "--trunk", "--trunk-name", "--trunk-mode", "--trunk-fuse", "--image-hw", "--v2-cache", "--eval-cache",
    "--agents", "--agent-join", "--agent-rig-camera", "--agent-rig-extrinsics", "--agent-cls-weight", "--w-agent",
    "--w-map", "--map-gt-root", "--w-box3d", "--join3d", "--slot-presence-loss", "--slot-presence-prior",
    "--slot-deep-supervision", "--slot-vis1", "--vis1-sidecar", "--slot-query-select", "--equalize-bottom-rows",
    "--map-hires", "--w-map-hires", "--map-hires-class-weights", "--map-hires-decision-rule", "--map-hires-x-max-m",
    "--map-hires-y-half-m", "--map-hires-grad-ckpt", "--map-hires-near-lift-m", "--map-hires-near-refine-blocks",
    "--bev-source", "--bev-planner-crop-m", "--u8-batches", "--trunk-chunk-ckpt", "--trunk-frozen-bn", "--trunk-bf16",
    "--trunk-channels-last", "--trunk-fold-bn", "--trunk-compile", "--clip-clock-sidecar", "--pose-sync-sidecar",
    "--workers",
)

INIT_FROM = "/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt"

#: ``--join-defect-masks`` (MM 2026-11-02, decision (a) on phase A's open issue): a PERCEPTION label fix, NOT a planner flag -- the
#: TRAIN join reader drops ego-as-agent boxes and masks id-switch rows (trainer help, refc_v3_train.py), and refcv8 trained with
#: it. The base argv's value (a path under refcv8's own launch tree) is dropped like ``--arm`` and ``build`` ALWAYS re-emits the
#: flag, with the value re-rooted under the rig's launch tree
#: (the path must resolve INSIDE the shipped launch tree: the gate's token binds the tree sha256, so the file's bytes are bound
#: by the same token; ``r9w_ship.sh tree`` ships ``stack/tanitad/configs/``). The file's md5 must equal refcv8's.
LAUNCH_TREE = "/home/nvidia/r8ns/r9e1/launch_tree"
JOIN_MASKS_REL = "stack/tanitad/configs/refcv8_join_label_defects_train4572.json"
JOIN_MASKS_MD5 = "b2591cb2ec4f4089831124e4738a961c"          # refcv8's file (launch_gate.py `--join-defect-masks` why-line)

_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\/]")


def _thor_path(p: str, what: str) -> str:
    """``--out`` / ``--init-from`` name paths ON THOR (POSIX). Run from git-bash, an argument that starts with ``/`` is
    rewritten to ``C:/Program Files/Git/...`` (MEASURED 2026-11-02: the smoke argv and BOTH landed C0/C1 argv.json carried
    ``--out C:/Program Files/Git/home/nvidia/r8ns/r9e1/runs/...``, which on Thor is a RELATIVE path -- the run dir was created
    under the trainer's cwd and its metrics.jsonl was where nobody looked). Refuse a Windows-form path; run the builder with
    ``MSYS_NO_PATHCONV=1`` or pass the path through an env var."""
    q = str(p).replace("\\", "/")
    if _WINDOWS_PATH.match(q) or "Program Files/Git" in q or not q.startswith("/"):
        raise SystemExit("[r9-e1-argv] %s %r is not an absolute POSIX path on Thor (git-bash path conversion? run with "
                         "MSYS_NO_PATHCONV=1)" % (what, p))
    return str(p)


def split_groups(argv):
    """[tok, ...] -> [(flag, [values]), ...]; every flag starts with '--' (no negative-number values in this argv)."""
    out = []
    for t in argv:
        if t.startswith("--"):
            out.append((t, []))
        elif out:
            out[-1][1].append(t)
        else:
            raise ValueError("argv starts with a non-flag token %r" % t)
    return out


def join_masks_value(launch_tree: str = LAUNCH_TREE) -> str:
    return _thor_path(launch_tree, "--launch-tree").rstrip("/") + "/" + JOIN_MASKS_REL


def check_join_masks(base_argv, repo_file: "Path | None" = None) -> str:
    """The rig's ``--join-defect-masks`` file IS refcv8's file: the base argv names it (same basename) and the repo copy that
    ``r9w_ship.sh tree`` ships has refcv8's md5. Returns the md5. SystemExit otherwise."""
    base_val = [v for f, v in split_groups(base_argv) if f == "--join-defect-masks"]
    if base_val and Path(str(base_val[0][0]).replace("\\", "/")).name != Path(JOIN_MASKS_REL).name:
        raise SystemExit("[r9-e1-argv] the base argv's --join-defect-masks %r is not %s" % (base_val[0], Path(JOIN_MASKS_REL).name))
    f = repo_file or (Path(__file__).resolve().parents[2] / JOIN_MASKS_REL)
    md5 = hashlib.md5(Path(f).read_bytes()).hexdigest()
    if md5 != JOIN_MASKS_MD5:
        raise SystemExit("[r9-e1-argv] %s has md5 %s, refcv8's join-defect-mask file is %s" % (f, md5, JOIN_MASKS_MD5))
    return md5


def build(base_argv, seed: int, out: str, steps: int = 4000, eval_every: int = 500, eval_batches: int = 8,
          save_every: int = 500, log_every: int = 50, init_from: str = INIT_FROM, launch_tree: str = LAUNCH_TREE) -> list:
    out, init_from = _thor_path(out, "--out"), _thor_path(init_from, "--init-from")
    masks = join_masks_value(launch_tree)
    groups = split_groups(base_argv)
    unknown = [f for f, _ in groups if f not in DROPPED and f not in KEPT]
    if unknown:
        raise SystemExit("[r9-e1-argv] base-argv flag(s) neither KEPT nor DROPPED: %s -- decide, do not pass through" % unknown)
    argv = ["--arm", "flat"]
    for f, vals in groups:
        if f in KEPT:
            argv += [f] + vals
    argv += ["--join-defect-masks", masks]
    argv += ["--opt", "dd", "--r9-perc-rig", "--init-from", init_from, "--batch", "16", "--warmup", "200",
             "--steps", str(int(steps)), "--ema-decay", "0.999", "--lr", "1e-4", "--seed", str(int(seed)),
             "--conflict-detector", "off", "--eval-every", str(int(eval_every)), "--eval-batches", str(int(eval_batches)),
             "--save-every", str(int(save_every)), "--log-every", str(int(log_every)), "--out", out]
    return argv


#: M-GATE: the E1 map-lever arms (REFCV9_DESIGN 2.1; BRIEF_C1_MAPHEAD): C0's argv + the banded head + its map weight, seed 0, every other
#: flag equal to C0's (only `--out` differs: the arm's own run directory). w = the E0j-measured w* (LOOP_STATE 02:25): M2a 0.511
#: (layer-2 share 0.05), M2b 1.018 (0.15); M1 keeps C0's 1.0.
MAP_ARMS = {"M1": 1.0, "M2a": 0.511, "M2b": 1.018}
BANDED_FLAG = "--r9-map-banded"
BANDED_NEAR_LIFT_M = "10"                 # the banded near band is 0-10 m @ 0.10 m: the DEFAULT (M1 / M2a / M2b); NEAR20_ARMS override it per arm
#: E1-NEAR20 (PREREG_M1_NEAR20): M1N20 = M1's argv (w = 1.0) with the near band 0-20 m @ 0.10 m: `--r9-map-banded-near-m 20` right after
#: `--r9-map-banded` and `--map-hires-near-lift-m` = 20 (C0's own value, carried from C0's argv). ONE lever vs M1 (the near-band end).
NEAR_M_FLAG = "--r9-map-banded-near-m"
#: E1-M3N20 (PREREG_M3): M3N20 = M3's own encoder on M1N20's near band. It takes M3's weight (operator-supplied, REQUIRED), so its row carries NO
#: `w_map_hires`; a row WITH one is a fixed-weight arm (M1N20).
NEAR20_ARMS = {"M1N20": {"w_map_hires": 1.0, "near_m": "20", "near_lift_m": "20"},
               "M3N20": {"near_m": "20", "near_lift_m": "20"}}


def banded_near_lift_m(arm: str) -> str:
    """The banded arm's `--map-hires-near-lift-m` (= its near-band end): 10 for M1 / M2a / M2b / M3, 20 for M1N20 / M3N20."""
    return NEAR20_ARMS[arm]["near_lift_m"] if arm in NEAR20_ARMS else BANDED_NEAR_LIFT_M
MAP_ARM_OUT = "/home/nvidia/r8ns/r9e1/runs/r9e1-%s"
#: E1-REC2: M3 = the M2 winner's banded head + the map's OWN 128-channel BEV encoder (C2, REFCV9_DESIGN 2.2). Its `--w-map-hires` is the M2
#: winner's, chosen AFTER M2a / M2b are read, so it is a REQUIRED argument and there is no committed `r9e1_M3.argv.json` until then.
OWN_ENC_FLAG = "--r9-map-own-enc"
#: E1-M3N20: M3N20 = M3 + M1N20's near band (`NEAR20_ARMS`); both take an operator-supplied `--w-map-hires` (PREREG_M3: M1's 1.0, no M2 winner exists).
OWN_ENC_ARMS = ("M3", "M3N20")


def fixed_map_weight(arm: str):
    """The arm's committed `--w-map-hires` (M1 / M2a / M2b / M1N20), or None for an own-encoder arm (M3 / M3N20: operator-supplied)."""
    if arm in OWN_ENC_ARMS:
        return None
    return MAP_ARMS[arm] if arm in MAP_ARMS else NEAR20_ARMS[arm]["w_map_hires"]


def map_arm_argv(c0_argv, arm: str, out: str | None = None, w: float | None = None) -> list:
    """C0's argv with EXACTLY: `--r9-map-banded` added (after `--map-hires-near-refine-blocks`), `--w-map-hires` = the arm's w,
    `--map-hires-near-lift-m` 20 -> 10, `--out` = the arm's run directory. Anything else in C0's argv is carried verbatim.
    M3 (`w` REQUIRED, > 0; refused for the fixed-w arms) additionally gets `--r9-map-own-enc` right after `--r9-map-banded`.
    M3N20 (E1-M3N20) = M3 with M1N20's near band: `--r9-map-own-enc` right after `--r9-map-banded`, then `--r9-map-banded-near-m 20`, near lift 20."""
    if arm not in MAP_ARMS and arm not in OWN_ENC_ARMS and arm not in NEAR20_ARMS:
        raise SystemExit("[r9-e1-argv] map arm %r not in %s" % (arm, sorted(MAP_ARMS) + list(OWN_ENC_ARMS) + sorted(NEAR20_ARMS)))
    if arm in OWN_ENC_ARMS:
        if w is None:
            raise SystemExit("[r9-e1-argv] map arm %s needs --w-map-hires: it is the M2 winner's weight, chosen after M2a / M2b are read "
                             "(%s)" % (arm, ", ".join("%s %s" % kv for kv in sorted(MAP_ARMS.items()))))
        if not float(w) > 0.0:
            raise SystemExit("[r9-e1-argv] --w-map-hires %r must be > 0 (a map term with no weight builds a dead head)" % (w,))
    elif w is not None:
        raise SystemExit("[r9-e1-argv] --w-map-hires is for M3 only (M3 / M3N20 take it): %s's weight is %s" % (arm, fixed_map_weight(arm)))
    out = _thor_path(out or MAP_ARM_OUT % arm, "--out")
    groups = split_groups(c0_argv)
    have = [f for f, _ in groups]
    if BANDED_FLAG in have or OWN_ENC_FLAG in have or NEAR_M_FLAG in have:
        raise SystemExit("[r9-e1-argv] the base argv already carries %s -- derive map arms from C0's argv" % BANDED_FLAG)
    for f in ("--w-map-hires", "--map-hires-near-lift-m", "--map-hires-near-refine-blocks", "--out"):
        if have.count(f) != 1:
            raise SystemExit("[r9-e1-argv] C0's argv must carry %s exactly once (has %d)" % (f, have.count(f)))
    argv = []
    for f, vals in groups:
        if f == "--w-map-hires":
            vals = [repr(float(w if arm in OWN_ENC_ARMS else fixed_map_weight(arm)))]
        elif f == "--map-hires-near-lift-m":
            vals = [banded_near_lift_m(arm)]
        elif f == "--out":
            vals = [out]
        argv += [f] + vals
        if f == "--map-hires-near-refine-blocks":
            argv.append(BANDED_FLAG)
            if arm in OWN_ENC_ARMS:
                argv.append(OWN_ENC_FLAG)                              # M3 / M3N20: right after the banded flag
            if arm in NEAR20_ARMS:
                argv += [NEAR_M_FLAG, NEAR20_ARMS[arm]["near_m"]]      # E1-NEAR20: the near lever; an M1 / M2 / M3 argv never carries the flag
    return argv


#: E1-REC2: the E1 box-lever arms (REFCV9_DESIGN 2.4; B3's three levers, B3' = B3 + the in-loss BEV-NMS), seed 0, every other flag equal to C0's
#: (only `--out` differs). The flags go right after `--slot-query-select` (the last slot flag).
BOX_ARMS = {"B3": ("--r9-box-quality", "--r9-box-dn", "--r9-box-class-sqrt"),
            "B3p": ("--r9-box-quality", "--r9-box-dn", "--r9-box-class-sqrt", "--r9-box-inloss-nms")}
BOX_ANCHOR = "--slot-query-select"
#: E1-REC3: the TRAINING-ONLY box aux-head arms (`launch_gate.E1_BOX_AUX_FLAGS`): B1 depth (C3), B2 perspective 3D and B2c its parameter-matched BEV
#: control (C4). Each = C0's argv + ITS flag + ITS weight pair, right after `--slot-query-select <v>` (the flag defaults to False and the weight to
#: 0.0, so BOTH are named in the argv -- a flag whose default disables it must be in the argv or the arm did not test it). The weight 0.1 is C3's
#: choice and C4's default (their HANDOFFs; HYPOTHESIS, unpinned by the design); the weight flag is the head's own, never `--w-box3d`.
AUX_W = "0.1"
AUX_ARMS = {"B1": ("--r9-box-depth-aux", "--w-box-depth", AUX_W),
            "B2": ("--r9-box-persp-aux", "--w-box-persp", AUX_W),
            "B2c": ("--r9-box-bev2-aux", "--w-box-bev2", AUX_W)}


def box_arm_tokens(arm: str) -> tuple:
    """The tokens an arm adds to C0's argv: B3 / B3p = valueless own-head flags; B1 / B2 / B2c = (aux flag, weight flag, weight)."""
    if arm in BOX_ARMS:
        return tuple(BOX_ARMS[arm])
    if arm in AUX_ARMS:
        return tuple(AUX_ARMS[arm])
    raise SystemExit("[r9-e1-argv] box arm %r not in %s" % (arm, sorted(BOX_ARMS) + sorted(AUX_ARMS)))


def box_arm_argv(c0_argv, arm: str, out: str | None = None) -> list:
    """C0's argv with EXACTLY the arm's lever tokens added (right after `--slot-query-select <v>`) and `--out` = the arm's run directory. Anything
    else in C0's argv is carried verbatim. B3 / B3p add valueless `--r9-box-*` flags; B1 / B2 / B2c (E1-REC3) add an aux flag + its weight pair."""
    tokens = box_arm_tokens(arm)
    out = _thor_path(out or MAP_ARM_OUT % arm, "--out")
    groups = split_groups(c0_argv)
    have = [f for f, _ in groups]
    if any(f.startswith("--r9-box-") for f in have):
        raise SystemExit("[r9-e1-argv] the base argv already carries a --r9-box-* flag -- derive box arms from C0's argv")
    for f in (BOX_ANCHOR, "--out"):
        if have.count(f) != 1:
            raise SystemExit("[r9-e1-argv] C0's argv must carry %s exactly once (has %d)" % (f, have.count(f)))
    argv = []
    for f, vals in groups:
        if f == "--out":
            vals = [out]
        argv += [f] + vals
        if f == BOX_ANCHOR:
            argv += list(tokens)
    return argv


def check_box_arm_pins(argv) -> None:
    """`check_against_pins` + the box lever's own refusals (`tanitad/train/box_b3_flag.pin_args`, which needs the flags to parse as on)."""
    check_against_pins(argv)
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    import refc_v3_train as T
    from tanitad.train import box_b3_flag as BF
    args = T.build_parser().parse_args(argv)
    if not BF.is_any(args):
        raise SystemExit("[r9-e1-argv] no --r9-box-* flag parsed as on")
    BF.pin_args(args)


def check_aux_arm_pins(argv, arm: str) -> None:
    """E1-REC3: `check_against_pins` + the aux head's own refusals (`box_depth_aux_flag.pin_args` for B1, `persp3d_aux_flag.pin_args` for B2 / B2c),
    and the arm's head is the ONLY aux head parsed as on with NO own-head (B3) lever (the arm differs from C0 in its lever alone)."""
    if arm not in AUX_ARMS:
        raise SystemExit("[r9-e1-argv] %r is not an aux-head arm (%s)" % (arm, sorted(AUX_ARMS)))
    check_against_pins(argv)
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    import refc_v3_train as T
    from tanitad.train import box_b3_flag as BF
    from tanitad.train import box_depth_aux_flag as BDA
    from tanitad.train import persp3d_aux_flag as P3
    args = T.build_parser().parse_args(argv)
    on = {"B1": BDA.is_on(args), "B2": P3.is_persp(args), "B2c": P3.is_bev2(args)}
    if on != {a: (a == arm) for a in AUX_ARMS}:
        raise SystemExit("[r9-e1-argv] arm %s: the aux heads parsed as on are %s, expected exactly %s" % (arm, sorted(k for k, v in on.items() if v), [arm]))
    if BF.is_any(args):
        raise SystemExit("[r9-e1-argv] arm %s carries an own-head --r9-box-* lever (B3's): an aux arm differs from C0 in its aux lever alone" % arm)
    (BDA if arm == "B1" else P3).pin_args(args)


#: E1-REC4: the E1 T1 arms (REFCV9_DESIGN 4.3, D3 -- the temporal BEV warp `--r9-bev-warp`, `launch_gate.E1_T1_FLAGS`): T1 = C0 + `--r9-bev-warp`; T1s = T1 +
#: `--r9-bev-warp-shuffle` (the control: the history frames come from a DONOR episode, so only the history CONTENT is wrong). The lags are NOT named: T1's
#: default `0.5,1.0` s is the arms' lags (the default does not disable the lever, unlike the aux weights). The flags go right after
#: `--map-hires-near-refine-blocks <v>` (the last 10 cm map flag: T1 fuses into the 10 cm branch's lift), seed 0, every other flag equal to C0's.
#: T1h / T1hs (refcv9 E1, MM brief 2026-11-05): the SAME two arms plus TWO levers against T1's measured failure (the gate never opened, |g| 0.0077): `--r9-bev-warp-supervise-hist` (the 10 cm
#: map loss supervises lift_valid_now | hist_valid: the loss never saw the cells history could fix) and `--r9-bev-warp-fuse-init identity` (the COPY init W[c, p*(C+1)+c] = 1/P: the trained fuse
#: stayed at its random default init -- a ReZero dead start); T1hs is the content control (donor-episode history, the SAME supervision set and init). Nothing else differs from T1 / T1s.
#: and `--r9-bev-warp-fill-only` (the gated history term applies ONLY where the current lift is invalid: the history-only cells' gate gradient is sign-consistent, the ~10x more camera-seen cells swamp it).
#: ⚠️ T1h carries ALL THREE levers, so a T1h win is not attributable to any one alone (a single-lever ablation would be one more tuple here).
#: A tuple holds argv TOKENS: `--r9-bev-warp-fuse-init` is followed by its value token.
T1_ARMS = {"T1": ("--r9-bev-warp",), "T1s": ("--r9-bev-warp", "--r9-bev-warp-shuffle"),
           "T1h": ("--r9-bev-warp", "--r9-bev-warp-supervise-hist", "--r9-bev-warp-fuse-init", "identity", "--r9-bev-warp-fill-only"),
           "T1hs": ("--r9-bev-warp", "--r9-bev-warp-shuffle", "--r9-bev-warp-supervise-hist", "--r9-bev-warp-fuse-init", "identity", "--r9-bev-warp-fill-only")}
#: T1v (refcv9 E1, `REFCV9_E1_T1V_PREREG.md`): C0 + `--r9-hist-supervise-geom` ONLY -- T1h's supervision set (`lift_valid_now | hist_valid`, the same lags 0.5 / 1.0 s) from GEOMETRY ALONE: NO past-frame
#: encode / lift / feature warp / fuse and NO `bev_warp` parameter, so NONE of the four T1h flags (and no `--r9-bev-warp`, which the lever refuses). One token, no value.
T1_ARMS["T1v"] = ("--r9-hist-supervise-geom",)
T1_GEOM_ARMS = ("T1v",)
T1_SHUFFLE_ARMS = ("T1s", "T1hs")
T1_SUPERVISE_ARMS = ("T1h", "T1hs")
T1_FUSE_INIT_ARMS = ("T1h", "T1hs")
T1_FILL_ONLY_ARMS = ("T1h", "T1hs")
T1_ANCHOR = "--map-hires-near-refine-blocks"
T1_LAGS_DEFAULT_S = (0.5, 1.0)


def t1_arm_argv(c0_argv, arm: str, out: str | None = None) -> list:
    """C0's argv with EXACTLY the arm's T1 flags added (right after `--map-hires-near-refine-blocks <v>`) and `--out` = the arm's run directory.
    Anything else in C0's argv is carried verbatim."""
    if arm not in T1_ARMS:
        raise SystemExit("[r9-e1-argv] T1 arm %r not in %s" % (arm, sorted(T1_ARMS)))
    out = _thor_path(out or MAP_ARM_OUT % arm, "--out")
    groups = split_groups(c0_argv)
    have = [f for f, _ in groups]
    if any(f.startswith("--r9-bev-warp") or f == "--r9-hist-supervise-geom" for f in have):
        raise SystemExit("[r9-e1-argv] the base argv already carries a --r9-bev-warp* / --r9-hist-supervise-geom flag -- derive T1 arms from C0's argv")
    for f in (T1_ANCHOR, "--out"):
        if have.count(f) != 1:
            raise SystemExit("[r9-e1-argv] C0's argv must carry %s exactly once (has %d)" % (f, have.count(f)))
    argv = []
    for f, vals in groups:
        if f == "--out":
            vals = [out]
        argv += [f] + vals
        if f == T1_ANCHOR:
            argv += list(T1_ARMS[arm])
    return argv


def check_t1_arm_pins(argv, arm: str) -> None:
    """`check_against_pins` + T1's own refusals (`tanitad/train/bev_warp_flag.pin_args`: needs the rig, `--map-hires on`, the timm shared trunk, no freeze), and
    the arm's flags are the ONLY lever parsed as on: T1 on, the shuffle control on exactly for T1s, the default lags, and no map / box lever (the arm differs
    from C0 in its T1 flags alone)."""
    if arm not in T1_ARMS:
        raise SystemExit("[r9-e1-argv] %r is not a T1 arm (%s)" % (arm, sorted(T1_ARMS)))
    check_against_pins(argv)
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    import refc_v3_train as T
    from tanitad.train import bev_warp_flag as BWF
    from tanitad.train import box_b3_flag as BF
    from tanitad.train import map_banded_flag as MBF
    args = T.build_parser().parse_args(argv)
    if arm in T1_GEOM_ARMS:                                                    # T1v: the geometry-only supervision mask is the ONE lever; the warp bundle is not on
        if not BWF.geom_on(args):
            raise SystemExit("[r9-e1-argv] arm %s: --r9-hist-supervise-geom did not parse as on" % arm)
        if BWF.is_on(args):
            raise SystemExit("[r9-e1-argv] arm %s: --r9-bev-warp parsed as on -- T1v is the ALTERNATIVE to the T1h bundle, not an add-on" % arm)
    else:
        if not BWF.is_on(args):
            raise SystemExit("[r9-e1-argv] arm %s: --r9-bev-warp did not parse as on" % arm)
        if BWF.geom_on(args):
            raise SystemExit("[r9-e1-argv] arm %s: --r9-hist-supervise-geom parsed as on -- only T1v carries it" % arm)
    if bool(getattr(args, BWF.SHUF_DEST)) != (arm in T1_SHUFFLE_ARMS):
        raise SystemExit("[r9-e1-argv] arm %s: the shuffle control parsed as %s, expected %s" % (arm, getattr(args, BWF.SHUF_DEST), arm in T1_SHUFFLE_ARMS))
    if BWF.fuse_init_of(args) != ("identity" if arm in T1_FUSE_INIT_ARMS else "rezero"):
        raise SystemExit("[r9-e1-argv] arm %s: --r9-bev-warp-fuse-init parsed as %r, expected %r" % (arm, BWF.fuse_init_of(args), "identity" if arm in T1_FUSE_INIT_ARMS else "rezero"))
    if bool(getattr(args, BWF.FILL_DEST)) != (arm in T1_FILL_ONLY_ARMS):
        raise SystemExit("[r9-e1-argv] arm %s: --r9-bev-warp-fill-only parsed as %s, expected %s" % (arm, getattr(args, BWF.FILL_DEST), arm in T1_FILL_ONLY_ARMS))
    if bool(getattr(args, BWF.SUP_DEST)) != (arm in T1_SUPERVISE_ARMS):
        raise SystemExit("[r9-e1-argv] arm %s: --r9-bev-warp-supervise-hist parsed as %s, expected %s" % (arm, getattr(args, BWF.SUP_DEST), arm in T1_SUPERVISE_ARMS))
    if tuple(BWF.parse_lags(args.r9_bev_warp_lags)) != T1_LAGS_DEFAULT_S:
        raise SystemExit("[r9-e1-argv] arm %s: lags %r are not T1's default %s" % (arm, args.r9_bev_warp_lags, T1_LAGS_DEFAULT_S))
    if BF.is_any(args) or MBF.is_on(args):
        raise SystemExit("[r9-e1-argv] arm %s carries a map / box lever: a T1 arm differs from C0 in its T1 flags alone" % arm)
    BWF.pin_args(args)


def check_map_arm_pins(argv, arm: str | None = None) -> None:
    """`check_against_pins` + the banded flag's own refusals (`tanitad/train/map_banded_flag.pin_args`). With `arm`, the PARSED near-band end
    must be the arm's (E1-NEAR20: 20 for M1N20 / M3N20, the default 10 with the flag ABSENT for every other map arm) and the own-encoder flag
    must be present exactly for the own-encoder arms (E1-M3N20: M3 / M3N20)."""
    check_against_pins(argv)
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    import refc_v3_train as T
    from tanitad.train import map_banded_flag as MBF
    args = T.build_parser().parse_args(argv)
    if not MBF.is_on(args):
        raise SystemExit("[r9-e1-argv] --r9-map-banded did not parse as on")
    MBF.pin_args(args)
    if arm is not None:
        want = float(NEAR20_ARMS[arm]["near_m"]) if arm in NEAR20_ARMS else MBF.NEAR_M_DEFAULT
        if abs(MBF.near_m(args) - want) > 1e-9 or (NEAR_M_FLAG in argv) != (arm in NEAR20_ARMS):
            raise SystemExit("[r9-e1-argv] arm %s: the parsed near-band end is %r (flag %s), expected %g%s" % (
                arm, MBF.near_m(args), "present" if NEAR_M_FLAG in argv else "absent", want, " via " + NEAR_M_FLAG if arm in NEAR20_ARMS else ""))
    from tanitad.train import map_own_enc_flag as MOE
    if arm is not None and (OWN_ENC_FLAG in argv) != (arm in OWN_ENC_ARMS):         # E1-M3N20: the flag is the arm's, both ways
        raise SystemExit("[r9-e1-argv] arm %s: %s is %s in the argv, expected %s" % (
            arm, OWN_ENC_FLAG, "present" if OWN_ENC_FLAG in argv else "absent", "present" if arm in OWN_ENC_ARMS else "absent"))
    if OWN_ENC_FLAG in argv:                                  # E1-REC2: M3's own encoder -- parsed on + its own refusals
        if not MOE.is_on(args):
            raise SystemExit("[r9-e1-argv] --r9-map-own-enc did not parse as on")
        MOE.pin_args(args)
    elif MOE.is_on(args):
        raise SystemExit("[r9-e1-argv] --r9-map-own-enc parsed as on without being in the argv")


def check_against_pins(argv) -> None:
    """The trainer's own argparse parses the argv and the rig's pin accepts it. Raises SystemExit otherwise."""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(here.parent))
    import refc_v3_train as T
    from tanitad.refs import refcv9_perc_rig as R
    args = T.build_parser().parse_args(argv)
    if not args.r9_perc_rig:
        raise SystemExit("[r9-e1-argv] --r9-perc-rig did not parse as on")
    R.pin_args(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="refcv8_launch_argv_base.json (a dict with key 'argv')")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="the run directory on Thor")
    ap.add_argument("--map-arm", choices=sorted(MAP_ARMS) + list(OWN_ENC_ARMS) + sorted(NEAR20_ARMS), default=None,
                    help="M-GATE: emit the map-lever arm (M1 / M2a / M2b; E1-REC2: M3, which needs --w-map-hires; E1-NEAR20: M1N20 = M1 with the "
                         "near band 0-20 m; E1-M3N20: M3N20 = M3 with the near band 0-20 m, which needs --w-map-hires) from C0's argv.json given as --base (seed 0)")
    ap.add_argument("--w-map-hires", type=float, default=None,
                    help="E1-REC2: M3's map weight = the M2 winner's (REQUIRED for --map-arm M3 / M3N20, refused for the other arms; PREREG_M3: M1's 1.0)")
    ap.add_argument("--box-arm", choices=sorted(BOX_ARMS) + sorted(AUX_ARMS), default=None,
                    help="E1-REC2: emit the box-lever arm (B3 / B3p) from C0's argv.json given as --base (seed 0); E1-REC3: or the training-only "
                         "aux-head arm B1 (depth) / B2 (perspective 3D) / B2c (its BEV control), each C0 + its flag + its weight pair")
    ap.add_argument("--t1-arm", choices=sorted(T1_ARMS), default=None,
                    help="E1-REC4: emit the temporal-BEV-warp arm T1 (C0 + --r9-bev-warp) or its shuffled control T1s (T1 + --r9-bev-warp-shuffle) from "
                         "C0's argv.json given as --base (seed 0); T1h / T1hs = T1 / T1s + --r9-bev-warp-supervise-hist + --r9-bev-warp-fuse-init identity + --r9-bev-warp-fill-only; "
                         "T1v = C0 + --r9-hist-supervise-geom (T1h's supervision mask from geometry alone, no past-frame encode / warp / fuse)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--launch-tree", default=LAUNCH_TREE, help="the shipped launch tree ON THOR (the join-defect-mask file lives in it)")
    ap.add_argument("--write", required=True)
    ap.add_argument("--no-pin-check", action="store_true")
    a = ap.parse_args(argv)
    if sum(bool(x) for x in (a.map_arm, a.box_arm, a.t1_arm)) > 1:
        ap.error("--map-arm, --box-arm and --t1-arm are separate arms")
    if a.w_map_hires is not None and a.map_arm is None:
        ap.error("--w-map-hires is M3's (--map-arm M3)")
    if a.map_arm or a.box_arm or a.t1_arm:
        if not a.base:
            ap.error("--map-arm / --box-arm / --t1-arm need --base <C0's r9e1_C0.argv.json>")
        c0 = json.loads(Path(a.base).read_text(encoding="utf-8"))
        if c0.get("arm") != "r9e1_C0" or int(c0.get("seed", -1)) != 0:
            raise SystemExit("[r9-e1-argv] --base must be r9e1_C0 (seed 0), got %r seed %r" % (c0.get("arm"), c0.get("seed")))
        name = a.map_arm or a.box_arm or a.t1_arm
        if a.t1_arm:
            out = t1_arm_argv(c0["argv"], a.t1_arm, a.out)
            if not a.no_pin_check:
                check_t1_arm_pins(out, a.t1_arm)
            lever = {"flags": [t for t in T1_ARMS[a.t1_arm] if t.startswith("--")], "lags_s": list(T1_LAGS_DEFAULT_S), "lags_source": "T1's default (--r9-bev-warp-lags is not named)",
                     "shuffle_control": a.t1_arm in T1_SHUFFLE_ARMS, "bev_warp": "NOT certified by the overfit records (they stamp bev_warp_not_certified)"}
            if a.t1_arm in T1_GEOM_ARMS:
                lever["bev_warp"] = "NOT BUILT: no past-frame encode / lift / feature warp / fuse and no bev_warp parameter; the inherited records stamp bev_warp_not_certified == [--r9-hist-supervise-geom]"
                lever["hist_supervise_geom"] = "the 10 cm map loss (and the banded loss) supervises lift_valid_now | hist_valid_geom; hist_valid_geom = bev_warp.hist_valid_of over the lags' MASK-ONLY warps of the camera's own valid under the same T and has as T1h"
            if a.t1_arm in T1_SUPERVISE_ARMS:
                lever["supervise_hist"] = "the 10 cm map loss supervises lift_valid_now | hist_valid (hist_valid = geometry only: the lags' warp-valid union; identical under shuffle)"
            if a.t1_arm in T1_FILL_ONLY_ARMS:
                lever["fill_only"] = "the gated history term is added only where the current lift is invalid: out = bev + gate * fuse(x) * (1 - valid_now)"
            if a.t1_arm in T1_FUSE_INIT_ARMS:
                lever["fuse_init"] = "identity: W[c, p*(C+1)+c] = 1/P at fresh build (fuse(x) = mean of the lags' warped past BEV), gate exactly 0; never over a loaded checkpoint"
        elif a.map_arm:
            out = map_arm_argv(c0["argv"], a.map_arm, a.out, w=a.w_map_hires)
            if not a.no_pin_check:
                check_map_arm_pins(out, a.map_arm)
            w = float(a.w_map_hires) if a.map_arm in OWN_ENC_ARMS else fixed_map_weight(a.map_arm)
            lever = {"flag": BANDED_FLAG, "w_map_hires": w, "near_lift_m": float(banded_near_lift_m(a.map_arm))}
            if a.map_arm in NEAR20_ARMS:
                lever.update(near_m_flag=NEAR_M_FLAG, near_m=float(NEAR20_ARMS[a.map_arm]["near_m"]),
                             vs=("M3: the near-band end 10 -> 20 m, nothing else" if a.map_arm in OWN_ENC_ARMS
                                 else "M1: the near-band end 10 -> 20 m, nothing else"))
            if a.map_arm in OWN_ENC_ARMS:
                lever.update(own_enc_flag=OWN_ENC_FLAG,
                             w_map_hires_source=("PREREG_M3: M1's weight (operator-supplied --w-map-hires 1.0); no M2 winner exists" if w == MAP_ARMS["M1"]
                                                 else "the M2 winner (operator-supplied --w-map-hires)"))
        else:
            out = box_arm_argv(c0["argv"], a.box_arm, a.out)
            if a.box_arm in AUX_ARMS:
                if not a.no_pin_check:
                    check_aux_arm_pins(out, a.box_arm)
                f, wf, w = AUX_ARMS[a.box_arm]
                lever = {"flags": [f], "weight_flag": wf, "weight": float(w), "aux_head": "training-only (not certified by the box overfit record)",
                         "weight_source": "HYPOTHESIS: C3's choice / C4's default, unpinned by the design"}
            else:
                if not a.no_pin_check:
                    check_box_arm_pins(out)
                lever = {"flags": list(BOX_ARMS[a.box_arm])}
        rec = {"schema": "r9e1-argv-v1", "arm": "r9e1_%s" % name, "seed": 0, "steps": c0["steps"], "argv": out,
               "base_sha256": hashlib.sha256(Path(a.base).read_bytes()).hexdigest(), "derived_from": "r9e1_C0", "lever": lever}
        Path(a.write).write_text(json.dumps(rec, indent=1) + chr(10), encoding="utf-8", newline=chr(10))
        print("wrote %s (%d tokens)" % (a.write, len(out)))
        return 0
    if a.base is None or a.seed is None or a.out is None:
        ap.error("--base, --seed and --out are required (C0 / C1 mode)")
    base = json.loads(Path(a.base).read_text(encoding="utf-8"))["argv"]
    check_join_masks(base)
    out = build(base, a.seed, a.out, a.steps, a.eval_every, a.eval_batches, a.save_every, a.log_every, launch_tree=a.launch_tree)
    if not a.no_pin_check:
        check_against_pins(out)
    rec = {"schema": "r9e1-argv-v1", "arm": "r9e1_C%d" % a.seed, "seed": a.seed, "steps": a.steps, "argv": out,
           "base_sha256": hashlib.sha256(Path(a.base).read_bytes()).hexdigest()}
    Path(a.write).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote %s (%d tokens)" % (a.write, len(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
