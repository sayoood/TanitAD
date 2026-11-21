#!/usr/bin/env python3
"""chain_argcheck.py -- the chain's DRY-RUN check of one rollout command line. NO rollout, no model, no GPU.

It takes exactly the argv the chain would hand to ``v8f_t1_speedsub.py`` and parses it with BOTH parsers that will see it at run time (the wrapper's, then the pinned tool's), applies the wrapper's own early
refusals that need no model, and cross-checks the pins against the files the command names:

    python chain_argcheck.py --wrapper <v8f_t1_speedsub.py> -- <the wrapper argv: --mode ... run --ckpt ...>

prints one JSON line {"ok": bool, "problems": [...], ...}; exit 0 iff ok. A problem here is a rollout that would have died (or run on the wrong thing) hours into the chain.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check_one(W, rest):
    problems, info = [], {}
    wa, tool_argv = W.build_parser().parse_known_args(rest)
    tool, tool_sha = W.load_tool(wa.tool_root, wa.tool_sha)
    ta = tool.build_parser().parse_args(tool_argv)
    info.update({"mode": wa.mode, "tag": ta.tag, "limit": ta.limit, "tool_sha256": tool_sha[:12], "out_dir": ta.out_dir})
    if ta.mode != "run":
        problems.append(f"the tool mode is {ta.mode!r}, the wrapper wraps `run` only")
    if W.MODE_LIMIT[wa.mode] != ta.limit:
        problems.append(f"mode {wa.mode} is the --limit {W.MODE_LIMIT[wa.mode]} lane, the argv says --limit {ta.limit}")
    if not ta.dump_ticks:
        problems.append("--dump-ticks is missing")
    if str(ta.device) != "cpu":
        problems.append(f"--device {ta.device!r}")
    if not os.path.isfile(ta.ckpt):
        problems.append(f"checkpoint {ta.ckpt} is missing")
    elif not wa.expect_ckpt_sha:
        problems.append("--expect-ckpt-sha is missing")
    else:
        got = sha256_file(ta.ckpt)
        info["ckpt_sha256"] = got[:12]
        if got != wa.expect_ckpt_sha:
            problems.append(f"the checkpoint hashes to {got[:12]}..., the pin is {wa.expect_ckpt_sha[:12]}...")
    if wa.mode in W.SUBSTITUTING:
        if not wa.readout or not os.path.isfile(wa.readout):
            problems.append(f"read-out {wa.readout!r} is missing")
        elif not wa.readout_sha or sha256_file(wa.readout) != wa.readout_sha:
            problems.append(f"the read-out does not hash to its pin {wa.readout_sha[:12]}...")
    if wa.mode == "egoshufc" and not wa.shuffle_source:
        problems.append("egoshufc needs --shuffle-source (the egoc run's speedsub record)")
    if wa.mode != "egoshufc" and wa.shuffle_source:
        problems.append("--shuffle-source is for egoshufc only")
    if not os.path.isfile(wa.instr):
        problems.append(f"instrument {wa.instr} is missing")
    elif wa.instr_sha and sha256_file(wa.instr) != wa.instr_sha:
        problems.append("the instrument does not hash to its pin")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        problems.append("CUDA_VISIBLE_DEVICES is not -1")
    return {"ok": not problems, "problems": problems, **info}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--plan", default="", help="a file of command lines, one token per line, command lines separated by a line `---`")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("sst1_wrapper_under_check", a.wrapper)
    W = importlib.util.module_from_spec(spec)
    sys.modules["sst1_wrapper_under_check"] = W
    spec.loader.exec_module(W)
    lines = []
    if a.plan:
        cur = []
        for ln in open(a.plan, encoding="utf-8").read().splitlines():
            if ln == "---":
                lines.append(cur)
                cur = []
            elif ln != "":
                cur.append(ln)
        if cur:
            lines.append(cur)
    else:
        lines = [a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest]
    bad = 0
    for rest in lines:
        try:
            out = check_one(W, rest)
        except SystemExit as e:
            out = {"ok": False, "problems": [f"refused: {e}"]}
        bad += 0 if out["ok"] else 1
        print(json.dumps(out), flush=True)
    print(f"ARGCHECK {len(lines) - bad}/{len(lines)} ok", flush=True)
    return 0 if bad == 0 and lines else 1


if __name__ == "__main__":
    sys.exit(main())
