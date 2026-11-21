#!/usr/bin/env python3
"""SEARCH-NAVTEST-PIPE launcher: Step B's search / gate / merge, UNCHANGED, on ONE CHUNK of the sealed navtest chunk plan.

Same construction as spc_launch.py (which pins the two C1 reads): Step B's `stepB.run_search_cli(..., carve_sha256=)` and `g_reach_stepb.run_gate(..., carve_sha256=)` take the DATA CLAUSE as a Python parameter and
no CLI flag reaches it.  Here the parameter is looked up from the chunk plan by chunk id, and the plan is accepted only if its canonical sha256 equals `--plan-sha256` (which the sealed C2 amendment pins):

    navpipe_launch.py gate   --plan-file P --plan-sha256 H --chunk N --head <final head s0> --head-sha256 <..> --bundle --ctx --ckpt --ckpt-sha256 --refe-dir --config --config-sha256 --out-dir [--token K]
    navpipe_launch.py search --plan-file P --plan-sha256 H --chunk N --arm P2HEAD --bundle --ctx --ckpt --ckpt-sha256 --refe-dir --config --config-sha256 --gate <PASS stamp> --out-dir [--start --end --shard --threads]
    navpipe_launch.py merge  --out-dir D --arm P2HEAD --out search.npz --n N          (Step B's merge, passed through)
A bundle whose sorted token list does not hash to that chunk's `tokens_sha256` is REFUSED by the same line that refused Step A's real bundle for Step B.  No head argument reaches the search (P2HEAD only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import navpipe_core as NC  # noqa: E402
NC.add_code_paths()                                    # Step B's and L1's code: repo-relative siblings, SPC_STEPB_CODE / L1_CODE override
STEPB_CODE, L1_CODE = NC.STEPB_CODE, NC.L1_CODE


def chunk_clause(plan_file: str, plan_sha256: str, cid: int) -> str:
    """the data clause of chunk `cid`: its tokens_sha256 from a plan whose canonical sha256 equals the typed one"""
    plan = json.load(open(plan_file, encoding="utf-8"))
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if NC.canon_sha256(body) != plan_sha256 or plan.get("plan_sha256") != plan_sha256:     # @GUARD:L-PLAN-SHA
        raise SystemExit(f"REFUSED: the chunk plan {plan_file} does not hash to the typed --plan-sha256 {plan_sha256[:12]}")
    ch = [c for c in plan["chunks"] if int(c["id"]) == int(cid)]
    if len(ch) != 1:
        raise SystemExit(f"REFUSED: chunk {cid} is not in the plan")
    return ch[0]["tokens_sha256"]


def _split(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--plan-file", required=True)
    ap.add_argument("--plan-sha256", required=True)
    ap.add_argument("--chunk", type=int, required=True)
    a, rest = ap.parse_known_args(argv)
    return chunk_clause(a.plan_file, a.plan_sha256, a.chunk), rest


def cmd_search(argv) -> int:
    clause, rest = _split(argv)
    if any(x == "--head" or x.startswith("--head=") or x == "--head-sha256" or x.startswith("--head-sha256=") for x in rest):     # @GUARD:L-NOHEAD
        raise SystemExit("REFUSED: SEARCH-NAVTEST-PIPE is the P2 scorer with NO head substitution; a head argument is not accepted")
    import stepB
    ns = stepB.build_search_parser().parse_args(rest)
    if ns.arm != "P2HEAD":                                                                  # @GUARD:L-ARM
        raise SystemExit(f"REFUSED: the arm is P2HEAD only (got {ns.arm!r})")
    return stepB.run_search_cli(ns, enforce_stamp=True, carve_sha256=clause)


def cmd_gate(argv) -> int:
    clause, rest = _split(argv)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU only (set CUDA_VISIBLE_DEVICES=-1)"
    ap = argparse.ArgumentParser(prog="navpipe_launch.py gate")
    for k in ("--head", "--head-sha256", "--bundle", "--ctx", "--ckpt", "--ckpt-sha256", "--refe-dir", "--config", "--config-sha256", "--out-dir"):
        ap.add_argument(k, required=True)
    ap.add_argument("--token", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(rest)
    a.arm = "L2"                                       # g_reach_stepb's arm choices are ("L2",); the head is the gate's probe head and nothing of it reaches the P2HEAD search
    import torch
    torch.set_num_threads(a.threads)
    import g_reach_stepb
    st = g_reach_stepb.run_gate(a, a.out_dir, token=a.token, carve_sha256=clause)
    return 0 if st["verdict"] == "PASS" else 1


def cmd_merge(argv) -> int:
    import stepB
    return stepB.cmd_merge(argv)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"search": cmd_search, "gate": cmd_gate, "merge": cmd_merge}
    if not argv or argv[0] not in cmds:
        print(__doc__)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
