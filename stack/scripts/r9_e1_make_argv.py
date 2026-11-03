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
BANDED_NEAR_LIFT_M = "10"                 # the banded near band is 0-10 m @ 0.10 m (MapHiresConfig(banded) refuses any other)
MAP_ARM_OUT = "/home/nvidia/r8ns/r9e1/runs/r9e1-%s"


def map_arm_argv(c0_argv, arm: str, out: str | None = None) -> list:
    """C0's argv with EXACTLY: `--r9-map-banded` added (after `--map-hires-near-refine-blocks`), `--w-map-hires` = the arm's w,
    `--map-hires-near-lift-m` 20 -> 10, `--out` = the arm's run directory. Anything else in C0's argv is carried verbatim."""
    if arm not in MAP_ARMS:
        raise SystemExit("[r9-e1-argv] map arm %r not in %s" % (arm, sorted(MAP_ARMS)))
    out = _thor_path(out or MAP_ARM_OUT % arm, "--out")
    groups = split_groups(c0_argv)
    have = [f for f, _ in groups]
    if BANDED_FLAG in have:
        raise SystemExit("[r9-e1-argv] the base argv already carries %s -- derive map arms from C0's argv" % BANDED_FLAG)
    for f in ("--w-map-hires", "--map-hires-near-lift-m", "--map-hires-near-refine-blocks", "--out"):
        if have.count(f) != 1:
            raise SystemExit("[r9-e1-argv] C0's argv must carry %s exactly once (has %d)" % (f, have.count(f)))
    argv = []
    for f, vals in groups:
        if f == "--w-map-hires":
            vals = [repr(float(MAP_ARMS[arm]))]
        elif f == "--map-hires-near-lift-m":
            vals = [BANDED_NEAR_LIFT_M]
        elif f == "--out":
            vals = [out]
        argv += [f] + vals
        if f == "--map-hires-near-refine-blocks":
            argv.append(BANDED_FLAG)
    return argv


def check_map_arm_pins(argv) -> None:
    """`check_against_pins` + the banded flag's own refusals (`tanitad/train/map_banded_flag.pin_args`)."""
    check_against_pins(argv)
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    import refc_v3_train as T
    from tanitad.train import map_banded_flag as MBF
    args = T.build_parser().parse_args(argv)
    if not MBF.is_on(args):
        raise SystemExit("[r9-e1-argv] --r9-map-banded did not parse as on")
    MBF.pin_args(args)


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
    ap.add_argument("--map-arm", choices=sorted(MAP_ARMS), default=None,
                    help="M-GATE: emit the map-lever arm (M1 / M2a / M2b) from C0's argv.json given as --base (seed 0)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--launch-tree", default=LAUNCH_TREE, help="the shipped launch tree ON THOR (the join-defect-mask file lives in it)")
    ap.add_argument("--write", required=True)
    ap.add_argument("--no-pin-check", action="store_true")
    a = ap.parse_args(argv)
    if a.map_arm:
        if not a.base:
            ap.error("--map-arm needs --base <C0's r9e1_C0.argv.json>")
        c0 = json.loads(Path(a.base).read_text(encoding="utf-8"))
        if c0.get("arm") != "r9e1_C0" or int(c0.get("seed", -1)) != 0:
            raise SystemExit("[r9-e1-argv] --base must be r9e1_C0 (seed 0), got %r seed %r" % (c0.get("arm"), c0.get("seed")))
        out = map_arm_argv(c0["argv"], a.map_arm, a.out)
        if not a.no_pin_check:
            check_map_arm_pins(out)
        rec = {"schema": "r9e1-argv-v1", "arm": "r9e1_%s" % a.map_arm, "seed": 0, "steps": c0["steps"], "argv": out,
               "base_sha256": hashlib.sha256(Path(a.base).read_bytes()).hexdigest(), "derived_from": "r9e1_C0",
               "lever": {"flag": BANDED_FLAG, "w_map_hires": MAP_ARMS[a.map_arm], "near_lift_m": float(BANDED_NEAR_LIFT_M)}}
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
