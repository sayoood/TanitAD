#!/usr/bin/env python3
"""SEARCH-P2-CONFIRM launcher: Step B's search / gate / merge, UNCHANGED, on the bundles of THIS package.

Step B's `stepB.py search` and `g_reach_stepb.py` hard-wire carve B's token-list sha256 as their DATA CLAUSE and give no CLI flag to change it (that was the point).  `stepB.run_search_cli(..., carve_sha256=)` and
`g_reach_stepb.run_gate(..., carve_sha256=)` DO take it as a Python parameter.  This file is the one caller: it maps `--read` to a PINNED constant, never to a free-form value, so a bundle that is not
exactly the sealed B-REST list (read C1) or the sealed C-ID control list (read CID) is REFUSED by the same line that refused Step A's real bundle for Step B.

    spc_launch.py gate   --read {C1,CID} --head <final head s0> --head-sha256 <..> --bundle --ctx --ckpt --ckpt-sha256 --refe-dir --config --config-sha256 --out-dir [--token N]
    spc_launch.py search --read {C1,CID} --arm P2HEAD --bundle --ctx --ckpt --ckpt-sha256 --refe-dir --config --config-sha256 --gate <PASS stamp> --out-dir [--start --end --shard --threads]
    spc_launch.py merge  --out-dir D --arm P2HEAD --out search.npz --n N          (Step B's merge, passed through)
The search arm is P2HEAD ONLY and carries NO head argument (a head flag is refused here, before Step B's own parser sees it).  The gate needs an L2 arm in Step B's code; the final head s0 file is its PROBE head
and nothing of it reaches the P2HEAD search (a P2HEAD stamp binds checkpoint, config, bundle and code).
"""
from __future__ import annotations

import argparse
import os
import sys

STEPB_CODE = os.environ.get("SPC_STEPB_CODE", "C:/Users/Admin/par18/l2stepb/code")
L1_CODE = os.environ.get("L1_CODE", "C:/Users/Admin/par18/l1run/code")
for p in (STEPB_CODE, L1_CODE):
    if p not in sys.path:
        sys.path.insert(0, p)

# PINNED data clauses (sha256 of the sorted token list joined by newlines = ctx.json tokens_sha256); the seal pins this file, so a changed constant voids the seal
B_REST_SHA256 = "3a85cc8124b3dd02297fffa176787b629369a75d82ed52c4440b32aa4cc70c4b"      # READ C1: the 905 held-out plan tokens in neither Step A's 400 nor carve B's 400
CID_SHA256 = "83bc0363c82afd9b2e64846a34d44700f5988de3854fd1721670455c0ed0b708"           # the C-ID control: 16 carve-B tokens (8 moved, 8 never moved by the banked search)
READS = {"C1": B_REST_SHA256, "CID": CID_SHA256}


def _split_read(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--read", required=True, choices=sorted(READS))
    a, rest = ap.parse_known_args(argv)
    return a.read, rest


def cmd_search(argv) -> int:
    read, rest = _split_read(argv)
    if any(x == "--head" or x.startswith("--head=") or x == "--head-sha256" or x.startswith("--head-sha256=") for x in rest):
        raise SystemExit("REFUSED: SEARCH-P2-CONFIRM is the P2 scorer with NO head substitution; a head argument is not accepted")
    import stepB
    ns = stepB.build_search_parser().parse_args(rest)
    if ns.arm != "P2HEAD":
        raise SystemExit(f"REFUSED: the arm is P2HEAD only (got {ns.arm!r})")
    return stepB.run_search_cli(ns, enforce_stamp=True, carve_sha256=READS[read])


def cmd_gate(argv) -> int:
    read, rest = _split_read(argv)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU only (set CUDA_VISIBLE_DEVICES=-1)"
    ap = argparse.ArgumentParser(prog="spc_launch.py gate")
    for k in ("--head", "--head-sha256", "--bundle", "--ctx", "--ckpt", "--ckpt-sha256", "--refe-dir", "--config", "--config-sha256", "--out-dir"):
        ap.add_argument(k, required=True)
    ap.add_argument("--token", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(rest)
    a.arm = "L2"                                       # g_reach_stepb's arm choices are ("L2",); the head is the gate's probe head
    import torch
    torch.set_num_threads(a.threads)
    import g_reach_stepb
    st = g_reach_stepb.run_gate(a, a.out_dir, token=a.token, carve_sha256=READS[read])
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
