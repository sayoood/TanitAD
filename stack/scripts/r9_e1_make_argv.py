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
    "--r8-nav-args-dropout", "--join-defect-masks", "--r8-speed-input", "--r8-speed-unknown-p", "--w-r8-v9-cons",
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


def build(base_argv, seed: int, out: str, steps: int = 4000, eval_every: int = 500, eval_batches: int = 8,
          save_every: int = 500, log_every: int = 50, init_from: str = INIT_FROM) -> list:
    groups = split_groups(base_argv)
    unknown = [f for f, _ in groups if f not in DROPPED and f not in KEPT]
    if unknown:
        raise SystemExit("[r9-e1-argv] base-argv flag(s) neither KEPT nor DROPPED: %s -- decide, do not pass through" % unknown)
    argv = ["--arm", "flat"]
    for f, vals in groups:
        if f in KEPT:
            argv += [f] + vals
    argv += ["--opt", "dd", "--r9-perc-rig", "--init-from", init_from, "--batch", "16", "--warmup", "200",
             "--steps", str(int(steps)), "--ema-decay", "0.999", "--lr", "1e-4", "--seed", str(int(seed)),
             "--conflict-detector", "off", "--eval-every", str(int(eval_every)), "--eval-batches", str(int(eval_batches)),
             "--save-every", str(int(save_every)), "--log-every", str(int(log_every)), "--out", out]
    return argv


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
    ap.add_argument("--base", required=True, help="refcv8_launch_argv_base.json (a dict with key 'argv')")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="the run directory on Thor")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--write", required=True)
    ap.add_argument("--no-pin-check", action="store_true")
    a = ap.parse_args(argv)
    base = json.loads(Path(a.base).read_text(encoding="utf-8"))["argv"]
    out = build(base, a.seed, a.out, a.steps, a.eval_every, a.eval_batches, a.save_every, a.log_every)
    if not a.no_pin_check:
        check_against_pins(out)
    rec = {"schema": "r9e1-argv-v1", "arm": "r9e1_C%d" % a.seed, "seed": a.seed, "steps": a.steps, "argv": out,
           "base_sha256": hashlib.sha256(Path(a.base).read_bytes()).hexdigest()}
    Path(a.write).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote %s (%d tokens)" % (a.write, len(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
